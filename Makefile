CONFIG = config/config.mk

MFEM_DIR = $(wildcard ~/MFEM/mfem-4.9)

-include $(CONFIG)
-include $(MFEM_DIR)/config/config.mk

CC = $(MFEM_CXX)
CFLAGS = $(MFEM_FLAGS) $(MFEM_LIBS)

SRC = src
BIN = bin

$(BIN)/main: $(SRC)/main.cpp
	$(CC) -o $@ $(SRC)/main.cpp $(CFLAGS)

.PHONY: config
config:
	@echo 'MFEM_DIR = $(MFEM_DIR)' > $(CONFIG)
	@echo 'Configuration file created correctly'

.PHONY: run
run: $(BIN)/main
	$(BIN)/main
