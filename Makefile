CUDA ?= C:/Cuda13
CUDA_INCLUDE = $(CUDA)/include
NVCC ?= $(CUDA)/bin/nvcc
MSVC ?= 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe'
NVCC_FLAGS = -O0 -g -std=c++14 --expt-relaxed-constexpr --compiler-bindir=$(MSVC)

ALL_GENCODE = -gencode=arch=compute_75,code=sm_75
BINARIES += bin/interview_problem.exe

src = $(wildcard src/*.cu)
obj = $(src:.cu=.o)
obj := $(subst src, obj, $(obj))

all:
	$(MAKE) dirs
	$(MAKE) bin/interview_problem.exe

dirs:
	if [ ! -d bin ]; then mkdir -p bin; fi
	if [ ! -d obj ]; then mkdir -p obj; fi

clean:
	rm -fr bin obj

obj/%.o: src/%.cu $(wildcard src/*.cuh)
	$(NVCC) $(NVCC_FLAGS) $(ALL_GENCODE) -Isrc -I$(CUDA_INCLUDE) -o $@ -c $<

bin/interview_problem.exe: $(obj)
	$(NVCC) $(NVCC_FLAGS) $(ALL_GENCODE) -Isrc -I$(CUDA_INCLUDE) -o $@ $(obj)
