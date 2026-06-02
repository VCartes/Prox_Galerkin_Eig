# Requirements

This project requires a working installation of [MFEM](https://mfem.org/building/). It is designed with the MPI parallel 4.9 version in mind, but it might work for other versions.

To visualize the solutions, a working installation of [GLVis](https://glvis.org/building/) is required. This project was built using GLVis version 4.5.

# Compiling the project

The compiler will search for the MFEM directory at `/home/<you>/MFEM/mfem-4.9` by default. To manually specify the location of your MFEM directory run
```
make config MFEM_DIR=<your_mfem_location>
```

When using the MPI version of MFEM, the MPI module will have to be loaded to compile the project. This can be achieved in **Fedora Linux** by executing
```
module load mpi
```

To compile the project, run
```
make
```
or, to run it inmediately after compilation, type
```
make run
```

# Running the project

Before running the program and to be able to visualize the solutions as they get approximated, GLVis must be running. With this done, go to the project directory and run the executable directly from the `bin/` directory or use
```
make run
```

By default, the second eigenvalue will be optimized (counting the zero eigenvalues). To manually select which eigenvalue will be optimized, run
```
bin/main -e <num_eigenvalue>
```

There are other options available, to display them, run
```
bin/main -h
```
