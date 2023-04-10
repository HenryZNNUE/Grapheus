# Makefile for Grapheus project
# Requires CUDA toolkit and a C++17 compiler

# Compiler options
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -fopenmp
NVCCFLAGS = -use_fast_math -std=c++17 -O3 -DNDEBUG

# Libraries
LIBS = -lcublas

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Files
SRCS := $(sort $(shell find $(SRCDIR) -name '*.cu'))
OBJS := $(SRCS:$(SRCDIR)/%.cu=$(OBJDIR)/%.obj)
EXE  := $(BINDIR)/Grapheus

# Targets
all: $(EXE)

$(EXE): $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ $(LIBS) -o $(EXE)

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)
