CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc

CUDA_INCLUDE = $(CUDA)/include

NVCC_FLAGS = -O3 -g -std=c++14 --expt-relaxed-constexpr

ALL_GENCODE = \
-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_75,code=sm_75

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
