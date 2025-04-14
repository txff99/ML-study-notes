# Makefile

# Paths
SRC_DIR = runtime/cuda
SRC = $(SRC_DIR)/memory.cu $(SRC_DIR)/mm.cu

# Output shared library
TARGET = $(SRC_DIR)/libcuda.so

# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC -shared

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
