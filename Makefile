
.PHONY: all clean
.SUFFIXES: .cpp .h .o .cu .cuh .C

DATE := $(shell date +%F)
UNAME := $(shell uname)

ROOTDIR := .
SRCDIR = src
INCLUDE = include
OBJDIR := obj
LOCAL_LIBDIR = ./lib

HT := /usr
HDSO := /usr/lib64

VPATH = src:include

OFILES = \
	CudaDeleter.o \
	Image.o \
	OIIOFiles.o

CUFILES = $(OBJDIR)/CackleRenderer.o

OBJS = $(patsubst %, $(OBJDIR)/%, $(OFILES))

CACKLELIB = $(LOCAL_LIBDIR)/libCackle.a

LIB := -L$(LOCAL_LIBDIR) -lCackle -L/opt/local/lib 

INC_LOCAL := -I./include -I. 
INC_CUDA = -I /usr/local/cuda/samples/common/inc
INC_PYTHON := -I/usr/include/python2.7 -I/usr/lib/python2.7/config
INC_DPA := -I/group/dpa/include

# Extra gencode flags for CUDA

CXX = nvcc 
CUDA_GENCODE = -arch=sm_30

CFLAGS = -g -O2 -Xptxas="-v" -std=c++11 --compiler-options="-fPIC"
COMPILE_FLAGS = -g -O3 -std=c++11 --compiler-options="-fPIC"
OFLAGS = -c $(INC_LOCAL)

OIIOLIB = -ldl -lm -L/group/dpa/lib -lOpenImageIO
CUDALIB = -L/usr/local/cuda/lib64 -lcudart

SWIGLD = $(CXX) -shared
SWIGEXEC = swig

###

all: $(OBJDIR) $(OBJS) $(CUFILES) $(SRCDIR)/CackleRenderer.cu $(INCLUDE)/CackleRenderer.h cuda/raymarchKernel.cu
	ar rv $(CACKLELIB) $(OBJS) $(CUFILES)

$(OBJDIR):
	@if [ ! -d $(OBJDIR) ]; then \
		echo "-----------------------------"; \
		echo "ERROR: Object directory does not exist"; \
		echo "Creating directory: $(OBJDIR)"; \
		echo "-----------------------------"; \
		mkdir $(OBJDIR); \
	fi

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(INCLUDE)/%.h
	$(CXX) -x c++ $< $(CFLAGS) $(OFLAGS) -o $@

$(OBJDIR)/CudaDeleter.o: $(SRCDIR)/CudaDeleter.cu $(INCLUDE)/CudaDeleter.h
	nvcc $(CUDA_GENCODE) $(CFLAGS) $(OFLAGS) $< -o $@ 

$(OBJDIR)/CackleRenderer.o: $(SRCDIR)/CackleRenderer.cu $(INCLUDE)/CackleRenderer.h cuda/raymarchKernel.cu
	nvcc $(CUDA_GENCODE) $(CFLAGS) $(OFLAGS) $(INC_CUDA) $(INC_DPA) $< -o $(OBJDIR)/CackleRenderer.o

$(OBJDIR)/Stitcher.o: $(SRCDIR)/Stitcher.cpp $(INCLUDE)/Stitcher.h
	g++ -g -O2 -std=c++11 -fPIC $(OFLAGS) $(INC_DPA) $< -o $(OBJDIR)/Stitcher.o

clean:
	-rm $(OBJDIR)/*.o $(LOCAL_LIBDIR)/*.a

genswig: swig/cackle.i $(OBJS) $(CUFILES)
	$(SWIGEXEC) -c++ -python -shadow -I./include/ swig/cackle.i
	g++ -g -O2 -fPIC -fopenmp -std=c++11 -c swig/cackle_wrap.cxx  $(INC_LOCAL) $(INC_PYTHON) $(INC_DPA) -o swig/cackle_wrap.o
	g++ -shared -g -O2 -fPIC -fopenmp -std=c++11 swig/cackle_wrap.o -o swig/_cackle.so $(LIB) $(CUDALIB) $(OIIOLIB)

stitcher: swig/stitcher.i $(OBJDIR)/Stitcher.o
	$(SWIGEXEC) -c++ -python -shadow -I./include/ swig/stitcher.i
	g++ -g -O2 -fPIC -fopenmp -std=c++11 -c swig/stitcher_wrap.cxx  $(INC_LOCAL) $(INC_PYTHON) $(INC_DPA) -o swig/stitcher_wrap.o
	g++ -shared -g -O2 -fPIC -fopenmp -std=c++11 swig/stitcher_wrap.o -o swig/_stitcher.so $(OBJDIR)/Stitcher.o $(OIIOLIB)
