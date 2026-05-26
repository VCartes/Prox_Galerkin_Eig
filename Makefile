MFEM_DIR = /home/vcartes/MFEM/mfem-4.9

-include $(MFEM_DIR)/config/config.mk

CC = $(MFEM_CXX)
CFLAGS = $(MFEM_FLAGS) $(MFEM_LIBS)

SRC = src
BIN = bin

$(BIN)/main: $(SRC)/main.cpp
	$(CC) -o $@ $(SRC)/main.cpp $(CFLAGS)

run: $(BIN)/main
	$(BIN)/main
