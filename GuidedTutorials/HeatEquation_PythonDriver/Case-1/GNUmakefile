# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../../amrex

# For Python support, clone pybind11 as sibling to amrex:
#   git clone -b v3.0.0 https://github.com/pybind/pybind11.git
# Or set PYBIND11_HOME to point to your pybind11 installation
PYBIND11_HOME ?= ../../../../pybind11

DEBUG     = FALSE
USE_MPI   = TRUE
USE_OMP   = FALSE
COMP      = gcc
DIM       = 3

USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

# Python main control flag
ifndef USE_PYTHON_MAIN
  USE_PYTHON_MAIN = FALSE
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# Add pybind11 and Python includes when building for Python
ifeq ($(USE_PYTHON_MAIN),TRUE)
  INCLUDE_LOCATIONS += $(PYBIND11_HOME)/include

  # Add Python headers
  PYTHON_CONFIG ?= python3-config
  CXXFLAGS += $(shell $(PYTHON_CONFIG) --includes)
endif

# Add PIC flag when building for Python
ifeq ($(USE_PYTHON_MAIN),TRUE)
  XTRA_CXXFLAGS += -fPIC
ifeq ($(USE_OMP),TRUE)
  LDFLAGS += -fopenmp
endif
endif

include Make.package
VPATH_LOCATIONS  += .
INCLUDE_LOCATIONS += .

include $(AMREX_HOME)/Src/Base/Make.package

# Job info support
CEXE_sources += AMReX_buildInfo.cpp
INCLUDE_LOCATIONS += $(AMREX_HOME)/Tools/C_scripts

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# Python library building (using pybind11 naming convention)
ifeq ($(USE_PYTHON_MAIN),TRUE)

# In WarpX this is machine-dependent
SHARED_OPTION = -shared

# Pybind11-style naming
PYMODULE_NAME = amrex_heat
PYDIM = $(DIM)d

installamrex_heat: lib$(PYMODULE_NAME).$(PYDIM).so
	cp lib$(PYMODULE_NAME).$(PYDIM).so Python/py$(PYMODULE_NAME)
	cd Python; python3 setup.py install --force --with-lib$(PYMODULE_NAME) $(PYDIM) $(PYINSTALLOPTIONS)

lib$(PYMODULE_NAME).$(PYDIM).a: $(objForExecs)
	@echo Making static library $@ ...
	$(SILENT) $(AR) -crv $@ $^
	$(SILENT) $(RM) AMReX_buildInfo.cpp
	@echo SUCCESS

lib$(PYMODULE_NAME).$(PYDIM).so: $(objForExecs)
	@echo Making dynamic library $@ ...
ifeq ($(USE_CUDA),TRUE)
	$(SILENT) $(CXX) $(LINKFLAGS) $(SHARED_OPTION) -Xlinker=$(SHARED_OPTION) -Xlinker=-fPIC -o $@ $^ $(LDFLAGS) $(libraries)
else
	$(SILENT) $(CXX) $(LINKFLAGS) $(SHARED_OPTION) -fPIC -o $@ $^ $(LDFLAGS) $(libraries)
endif
	$(SILENT) $(RM) AMReX_buildInfo.cpp
	@echo SUCCESS

clean::
	$(SILENT) $(RM) -rf build
	$(SILENT) $(RM) -f lib$(PYMODULE_NAME).?d.a
	$(SILENT) $(RM) -f lib$(PYMODULE_NAME).?d.so
	$(SILENT) $(RM) -f py$(PYMODULE_NAME)/lib$(PYMODULE_NAME).?d.so
	$(SILENT) $(RM) -f $(PYMODULE_NAME).*.so  # Clean pybind11-style names too

endif

# Convenience targets
.PHONY: python_lib python_install inputs

python_lib:
	$(MAKE) USE_PYTHON_MAIN=TRUE lib$(PYMODULE_NAME).$(DIM)d.so

python_install:
	$(MAKE) USE_PYTHON_MAIN=TRUE installamrex_heat

inputs:
	cp -r inputs/* .
