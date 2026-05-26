#include "mfem.hpp"
#include <iostream>
#include <fstream>


#define SQUARE_SIZE 10.0



using namespace std;
using namespace mfem;

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   MappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       fun([](real_t x) {return x;}) {}
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<real_t(const real_t)> fun_,
                                 int comp=1)
      :GridFunctionCoefficient(gf, comp),
       fun(fun_) {}


   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
};


/// @brief Returns f(u(x)) - f(v(x)) where u, v are scalar GridFunctions and f:R → R
class DiffMappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   const GridFunction *OtherGridF;
   GridFunctionCoefficient OtherGridF_cf;
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   DiffMappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       OtherGridF(nullptr),
       OtherGridF_cf(),
       fun([](real_t x) {return x;}) {}
   DiffMappedGridFunctionCoefficient(const GridFunction *gf,
                                     const GridFunction *other_gf,
                                     std::function<real_t(const real_t)> fun_,
                                     int comp=1)
      :GridFunctionCoefficient(gf, comp),
       OtherGridF(other_gf),
       OtherGridF_cf(OtherGridF),
       fun(fun_) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const real_t value1 = fun(GridFunctionCoefficient::Eval(T, ip));
      const real_t value2 = fun(OtherGridF_cf.Eval(T, ip));
      return value1 - value2;
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
};


real_t inv_sigmoid(real_t x)
{
   real_t tol = 1e-12;
   x = std::min(std::max(tol,x), real_t(1.0)-tol);
   return std::log(x/(1.0-x));
}

/// @brief Sigmoid function
real_t sigmoid(real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      return std::exp(x)/(1.0+std::exp(x));
   }
}


real_t der_sigmoid(real_t x)
{
   real_t tmp = sigmoid(-x);
   return tmp - std::pow(tmp,2);
}


real_t proj(ParGridFunction &psi, real_t target_volume, real_t tol=1e-12,
            int max_its=10)
{
   MappedGridFunctionCoefficient sigmoid_psi(&psi, sigmoid);
   MappedGridFunctionCoefficient der_sigmoid_psi(&psi, der_sigmoid);

   ParLinearForm int_sigmoid_psi(psi.ParFESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   ParLinearForm int_der_sigmoid_psi(psi.ParFESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      real_t f = int_sigmoid_psi.Sum();
	  // cout << f << endl;
      MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      f -= target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      real_t df = int_der_sigmoid_psi.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &df, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);

      const real_t dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. "
                   "Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   real_t material_volume = int_sigmoid_psi.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &material_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   return material_volume;
}


// real_t indicator(const Vector &x) {
// 
// 	// Infty norm
// 
// 	real_t x0 = x(0) - SQUARE_SIZE/2;
// 	real_t y0 = x(1) - SQUARE_SIZE/2;
// 
// 	if (abs(x0) <= 1 && abs(y0) <= 1)
// 		return 1;
// 
// 	return 0;
// }

real_t indicator(const Vector &x) {
	real_t x0 = x(0) - SQUARE_SIZE/2;
	real_t y0 = x(1) - SQUARE_SIZE/2;
	real_t norm_sq = x0*x0/4 + y0*y0;

	if (norm_sq <= 1) return 0.9;

	return 0.1;
}

real_t inv_sig_indicator(const Vector &x) {
	return inv_sigmoid(indicator(x));
}


int main(int argc, char *argv[])
{
	// 0. Initialize MPI and HYPRE.
	Mpi::Init();
	int num_procs = Mpi::WorldSize();
	int myid = Mpi::WorldRank();
	Hypre::Init();

	// 1. Parse command-line options.
	int ref_levels = 5;
	int order = 2;
	real_t alpha = 1.0;
	real_t epsilon = 1e-3;
	real_t vol_fraction = 0.5;
	int max_it = 1000;
	real_t itol = 1e-1;
	real_t ntol = 1e-4;
	real_t rho_min = 1e-6;
	real_t lambda = 1.0;
	real_t mu = 1.0;
	bool glvis_visualization = true;
	bool paraview_output = false;

	int n_eig = 5;

	OptionsParser args(argc, argv);
	args.AddOption(&ref_levels, "-r", "--refine",
			"Number of times to refine the mesh uniformly.");
	args.AddOption(&order, "-o", "--order",
			"Order (degree) of the finite elements.");
	args.AddOption(&alpha, "-alpha", "--alpha-step-length",
			"Step length for gradient descent.");
	args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
			"Length scale for ρ.");
	args.AddOption(&max_it, "-mi", "--max-it",
			"Maximum number of gradient descent iterations.");
	args.AddOption(&ntol, "-ntol", "--rel-tol",
			"Normalized exit tolerance.");
	args.AddOption(&itol, "-itol", "--abs-tol",
			"Increment exit tolerance.");
	args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
			"Volume fraction for the material density.");
	args.AddOption(&lambda, "-lambda", "--lambda",
			"Lamé constant λ.");
	args.AddOption(&mu, "-mu", "--mu",
			"Lamé constant μ.");
	args.AddOption(&rho_min, "-rmin", "--psi-min",
			"Minimum of density coefficient.");
	args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
			"--no-visualization",
			"Enable or disable GLVis visualization.");
	args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
			"--no-paraview",
			"Enable or disable ParaView output.");
	args.Parse();
	if (!args.Good())
	{
		if (myid == 0)
		{
			args.PrintUsage(cout);
		}
		MPI_Finalize();
		return 1;
	}
	if (myid == 0)
	{
		mfem::out << num_procs << " number of process created.\n";
		args.PrintOptions(cout);
	}

	// Mesh

	int nx = 128; int ny = 128;

	Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::TRIANGLE, false, SQUARE_SIZE, SQUARE_SIZE);
	ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

	int dim = mesh.Dimension();


	// 4. Define the necessary finite element spaces on the mesh.
	H1_FECollection u_fec(order, dim); // space for u
	H1_FECollection rho_fec(order, dim); // space for ρ̃
	L2_FECollection psi_fec(order-1, dim,
			BasisType::GaussLobatto); // space for ψ

	ParFiniteElementSpace u_fes(pmesh, &u_fec);
	ParFiniteElementSpace rho_fes(pmesh, &rho_fec);
	ParFiniteElementSpace psi_fes(pmesh, &psi_fec);

	HYPRE_BigInt u_size = u_fes.GlobalTrueVSize();
	HYPRE_BigInt psi_size = psi_fes.GlobalTrueVSize();
	HYPRE_BigInt rho_size = rho_fes.GlobalTrueVSize();
	if (myid==0)
	{
		cout << "Number of u unknowns: " << u_size << endl;
		cout << "Number of rho unknowns: " << rho_size << endl;
		cout << "Number of psi unknowns: " << psi_size << endl;
	}

	// 5. Set the initial guess for ρ.
	ParGridFunction u(&u_fes);
	ParGridFunction psi(&psi_fes);
	ParGridFunction psi_old(&psi_fes);

	ParGridFunction grad(&psi_fes);

	FunctionCoefficient indicator_coef(indicator);
	FunctionCoefficient inv_sig_indicator_coef(inv_sig_indicator);
	ConstantCoefficient eps_sq(epsilon*epsilon);
	ConstantCoefficient eps(epsilon);
	ConstantCoefficient zero(0.0);

	ParGridFunction zerogf(&psi_fes);


	psi.ProjectCoefficient(inv_sig_indicator_coef);
	psi_old.ProjectCoefficient(inv_sig_indicator_coef);

	// ρ = sigmoid(ψ)
	MappedGridFunctionCoefficient rho(&psi, sigmoid);
	DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);



	ParGridFunction x(&u_fes);


	HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
	lobpcg->SetNumModes(n_eig);
	lobpcg->SetRandomSeed(75);
	lobpcg->SetMaxIter(1000);
	lobpcg->SetTol(1e-12);
	lobpcg->SetPrecondUsageMode(1);
	lobpcg->SetPrintLevel(0);

	// 10. Connect to GLVis. Prepare for VisIt output.
	char vishost[] = "localhost";
	int  visport   = 19916;

	socketstream rho_vis;
	socketstream psi_vis;
	socketstream grad_vis;
	socketstream u_vis;
	if (glvis_visualization)
	{
		rho_vis.open(vishost, visport);
		rho_vis.precision(8);

		psi_vis.open(vishost, visport);
		psi_vis.precision(8);

		grad_vis.open(vishost, visport);
		grad_vis.precision(8);

		u_vis.open(vishost, visport);
		u_vis.precision(8);
	}

	// 11. Iterate:
	for (int k = 1; k <= max_it; k++)
	{

		if (myid == 0)
		{
			cout << "\nStep = " << k << endl;
		}

		

		// Solve EVP

		ParBilinearForm a(&u_fes);
		a.AddDomainIntegrator(new DiffusionIntegrator(rho));
		a.AddDomainIntegrator(new DiffusionIntegrator(eps));
		a.Assemble();
		a.Finalize();
		
		ParBilinearForm m(&u_fes);
		m.AddDomainIntegrator(new MassIntegrator(rho));
		m.AddDomainIntegrator(new MassIntegrator(eps_sq));
		m.Assemble();
		m.Finalize();

		HypreParMatrix *A = a.ParallelAssemble();
		HypreParMatrix *M = m.ParallelAssemble();
		lobpcg->SetMassMatrix(*M);
		lobpcg->SetOperator(*A);

		HypreEuclid* euc = new HypreEuclid(*A);
		lobpcg->SetPreconditioner(*euc);



		Array<real_t> eigenvalues;
		lobpcg->Solve();

		lobpcg->GetEigenvalues(eigenvalues);

		cout << "Eig: " << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2] << endl;
		cout << "Target: " << eigenvalues[n_eig-1] << endl;

		u = lobpcg->GetEigenvector(n_eig-1);
		real_t m_norm_u = m.InnerProduct(u, u);
		u /= m_norm_u;

		lambda = eigenvalues[n_eig-1];


		// GRADIENT

		GridFunctionCoefficient u_coef(&u);
		GradientGridFunctionCoefficient du_coef(&u);

		ProductCoefficient u_sq_coef(u_coef, u_coef);
		InnerProductCoefficient du_sq_coef(du_coef, du_coef);

		SumCoefficient grad_coef(du_sq_coef, u_sq_coef, 1.0, -lambda);

		grad.ProjectCoefficient(grad_coef);

		psi.Add(alpha, grad);
		real_t vol = 15;
		const real_t material_volume = proj(psi, vol);

		real_t err = zerogf.ComputeL2Error(succ_diff_rho);
		cout << "rho increase: " << err << endl;
		psi_old = psi;
		


		x.ProjectCoefficient(rho);

		rho_vis << "parallel " << num_procs << " " << myid << "\n"
			<< "solution\n" << *pmesh << x << flush
			<< "window_title 'rho'" << endl;
		psi_vis << "parallel " << num_procs << " " << myid << "\n"
			<< "solution\n" << *pmesh << psi << flush
			<< "window_title 'psi'" << endl;
		grad_vis << "parallel " << num_procs << " " << myid << "\n"
			<< "solution\n" << *pmesh << grad << flush
			<< "window_title 'grad'" << endl;
		u_vis << "parallel " << num_procs << " " << myid << "\n"
			<< "solution\n" << *pmesh << u << flush
			<< "window_title 'u'" << endl;

		char c;
		cout << "Next" << endl;
		//cin >> c;

		delete A;
		delete M;
		delete euc;
	}

	return 0;
}

