# Makefile

# Paths
SRC_DIR = runtime/cuda
SRC = $(SRC_DIR)/memory.cu $(SRC_DIR)/mm.cu
CUTLASS_LIB = /home/lawlet/Documents/tutorial/cutlass/include
HELPER_LIB = /home/lawlet/Documents/tutorial/dl_impl/third_party/cutlass
# Output shared library
TARGET = $(SRC_DIR)/libcuda.so

# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC -shared  

ifeq ($(USE_CUTLASS),1)
    LIBFLAGS += -I$(CUTLASS_LIB) -I$(HELPER_LIB)
	NVCCFLAGS += -DUSE_CUTLASS
endif

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(LIBFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
