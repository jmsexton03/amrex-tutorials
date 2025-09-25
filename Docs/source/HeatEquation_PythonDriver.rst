.. _guided_heat_python_driver:

Tutorial: Heat Equation - Python Driver
========================================

.. admonition:: **Time to Complete**: 10 mins
   :class: warning

   **PREREQUISITES:**
     - Complete :ref:`guided_heat_simple` tutorial first

   **GOALS:**
     - Create minimal Python interface to AMReX C++ code
     - Implement one-line simulation execution from Python
     - Access simulation results programmatically
     - Understand the function refactoring pattern


This tutorial demonstrates the minimal setup needed to create a Python interface
for AMReX C++ simulation codes. The focus is on a single, clean interface that
allows you to run simulations from Python and access results directly.

The One-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The goal is to enable this simple Python usage pattern:

.. code-block:: python

   import amrex_heat

   # One-line simulation execution with structured results
   result = amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])

   print(f"Max temperature: {result.max_temperature}")
   print(f"Final time: {result.final_time}")
   print(f"Success: {result.success}")

This approach provides direct access to simulation results without subprocess overhead
or complex callback systems.

Key Architecture Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

The essential insight is to refactor the C++ ``main()`` function into a reusable
function that can be called from both the command line and Python:

.. code-block:: cpp

   // Reusable simulation function
   SimulationResult heat_equation_main(int argc, char* argv[]) {
       amrex::Initialize(argc, argv);
       {
           // All simulation logic here
           // ...
           result.max_temperature = phi_new.max(0);
           result.success = true;
       }
       amrex::Finalize();
       return result;
   }

   // C++ entry point
   int main(int argc, char* argv[]) {
       heat_equation_main(argc, argv);
       return 0;
   }

This pattern allows the same simulation code to be used from:

- **Command line**: ``./HeatEquation_PythonDriver inputs``
- **Python**: ``amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])``

Building the Project
~~~~~~~~~~~~~~~~~~~~

Navigate to the directory :code:`amrex-tutorials/GuidedTutorials/HeatEquation_PythonDriver/`
and build the project:

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   cmake --build . -j4

This will create both:

- ``HeatEquation_PythonDriver`` - C++ executable
- ``amrex_heat.cpython-*.so`` - Python module

Required Files
~~~~~~~~~~~~~~

The ``HeatEquation_PythonDriver`` directory contains these essential files:

- ``main.cpp`` - Heat equation solver with ``heat_equation_main()`` function
- ``bindings.cpp`` - Minimal pybind11 interface exposing the one-liner
- ``CMakeLists.txt`` - Build configuration with pybind11 support
- ``pybind11.cmake`` - Pybind11 infrastructure (copied from pyamrex)
- ``inputs`` - Simulation parameters (``n_cell``, ``dt``, etc.)
- ``test.py`` - Example Python usage script
- ``README.md`` - Documentation and usage instructions

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Result Structure
^^^^^^^^^^^^^^^^

The simulation returns a simple struct with essential information:

.. code-block:: cpp

   struct SimulationResult {
       double max_temperature;
       int final_step;
       double final_time;
       bool success;
   };

This struct is automatically exposed to Python through pybind11, allowing direct
access to all fields.

Minimal Python Bindings
^^^^^^^^^^^^^^^^^^^^^^^^

The Python interface is implemented with minimal pybind11 code in ``bindings.cpp``.
The key components are:

1. **Forward declaration** of the ``SimulationResult`` struct from ``main.cpp``
2. **Function declaration** for ``heat_equation_main()``
3. **Pybind11 module** that exposes both the struct and function

.. code-block:: cpp

   // Forward declarations from main.cpp
   struct SimulationResult {
       double max_temperature;
       int final_step;
       double final_time;
       bool success;
   };

   SimulationResult heat_equation_main(int argc, char* argv[]);

   PYBIND11_MODULE(amrex_heat, m) {
       m.doc() = "Minimal AMReX Heat Equation Python Interface";

       // Expose SimulationResult struct
       py::class_<SimulationResult>(m, "SimulationResult")
           .def_readonly("max_temperature", &SimulationResult::max_temperature)
           .def_readonly("final_step", &SimulationResult::final_step)
           .def_readonly("final_time", &SimulationResult::final_time)
           .def_readonly("success", &SimulationResult::success);

       // Main simulation function - one-liner interface
       m.def("run_simulation", [](py::list args) {
           // Convert Python list to C++ argc/argv with proper lifetime management
           std::vector<std::string> args_str;
           for (auto item : args) {
               args_str.push_back(py::str(item));
           }

           std::vector<char*> args_cstr;
           for (auto& s : args_str) {
               args_cstr.push_back(const_cast<char*>(s.c_str()));
           }
           args_cstr.push_back(nullptr);  // Null terminate

           return heat_equation_main(static_cast<int>(args_cstr.size() - 1), args_cstr.data());
       }, "Run the heat equation simulation and return results");
   }

The argument conversion ensures proper lifetime management of the C++ strings and
null-terminates the argument array as expected by ``argc/argv`` conventions.

CMake Integration
^^^^^^^^^^^^^^^^^

The ``CMakeLists.txt`` integrates pybind11 using the pyamrex infrastructure. The key
elements are the pybind11 integration and building both targets from the same source:

.. code-block:: cmake

   # Use pyamrex pybind11 infrastructure
   include(pybind11.cmake)

   # Add the main executable
   add_executable(HeatEquation_PythonDriver main.cpp)

   # Add the pybind11 module including main simulation logic
   pybind11_add_module(amrex_heat bindings.cpp main.cpp)

   # Link AMReX to both targets
   target_link_libraries(HeatEquation_PythonDriver PRIVATE AMReX::amrex)
   target_link_libraries(amrex_heat PRIVATE AMReX::amrex)

The ``pybind11.cmake`` file is copied from the pyamrex repository and provides the necessary
pybind11 infrastructure without requiring a separate pyamrex installation.

Running the Examples
~~~~~~~~~~~~~~~~~~~~

C++ Executable
^^^^^^^^^^^^^^

Test the traditional C++ interface:

.. code-block:: bash

   cd build
   ./HeatEquation_PythonDriver inputs

This runs the simulation and prints progress to the terminal.

Python Interface
^^^^^^^^^^^^^^^^

Test the new Python interface:

.. code-block:: bash

   cd build
   python ../test.py

The Python script demonstrates accessing simulation results:

.. code-block:: python

   import amrex_heat

   print("Running simulation...")
   result = amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])

   print(f"Simulation Results:")
   print(f"  Success: {result.success}")
   print(f"  Final step: {result.final_step}")
   print(f"  Final time: {result.final_time:.6f}")
   print(f"  Max temperature: {result.max_temperature:.6f}")

Expected output:

.. code-block::

   Heat Equation Python Driver Test
   ========================================
   Running simulation...
   Advanced step 1
   Advanced step 2
   ...
   Advanced step 1000

   Simulation Results:
     Success: True
     Final step: 1000
     Final time: 0.010000
     Max temperature: 1.089070

   ✓ Simulation completed successfully!

Adapting This Pattern to Your Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To apply this pattern to your own AMReX simulation, follow these steps:

1. **Choose your output data**: Decide what simulation results to return (max values, final state, convergence info, etc.)

2. **Choose return format**: Select struct, dictionary, or numpy array based on your Python workflow needs

3. **Create the pybind11 module**: Write ``bindings.cpp`` that exposes your chosen interface

4. **Wrap existing main**: Refactor your ``main()`` function into a reusable function like ``simulation_main()`` or ``your_code_main()``

5. **Create Python test**: Write a test script to exercise the new interface

6. **Update CMake**: Add pybind11 library target and include ``pybind11.cmake``

This systematic approach ensures you maintain your existing C++ code while adding clean Python access.

Benefits of This Approach
~~~~~~~~~~~~~~~~~~~~~~~~~

**Simplicity**
^^^^^^^^^^^^^^
- Only ~50 lines of additional code
- Easy to understand and modify
- Minimal dependencies

**Performance**
^^^^^^^^^^^^^^^
- No subprocess overhead
- Direct function calls
- Same performance as C++ executable

**Flexibility**
^^^^^^^^^^^^^^^
- Same code for command line and Python
- Easy to extend with more return data
- Foundation for complex workflows

**Integration**
^^^^^^^^^^^^^^^
- Works with existing build systems
- Compatible with pyamrex infrastructure
- Follows established patterns from WarpX/Nyx

Use Cases
~~~~~~~~~

This pattern is ideal for:

- **Parameter sweeps**: Run multiple simulations with different inputs
- **Optimization workflows**: Use simulation results in optimization loops
- **Data analysis pipelines**: Process simulation outputs immediately
- **Jupyter notebooks**: Interactive simulation and visualization
- **Machine learning**: Generate training data or run inference

Next Steps
~~~~~~~~~~

This minimal Python interface provides the foundation for more advanced features:

1. **Add more return data**: Include arrays, MultiFab statistics, etc.
2. **Parameter setting**: Allow modification of simulation parameters from Python
3. **Progress monitoring**: Add callback system for real-time updates
4. **Full pyamrex integration**: Access MultiFab data structures directly
5. **Workflow automation**: Build complex simulation pipelines
6. **Generic naming**: Replace heat equation-specific names (``amrex_heat``, ``max_temperature``) with generic equivalents (``amrex_sim``, ``max_value``) for reusability across different simulation types
7. **Numpy-compatible results**: Add options to return data as dictionaries, numpy arrays, or other formats that integrate well with the scientific Python ecosystem

Potential improvements for generic usage:

.. code-block:: cpp

   // Generic module and function names
   PYBIND11_MODULE(amrex_sim, m) {
       // Option 1: Return as dictionary for numpy compatibility
       m.def("run_dict", [](py::list args) {
           auto result = simulation_main(argc, argv);
           py::dict d;
           d["success"] = result.success;
           d["max_value"] = result.max_value;  // Generic field name
           d["final_time"] = result.final_time;
           return d;
       });

       // Option 2: Return numerical data as numpy array
       m.def("run_array", [](py::list args) {
           auto result = simulation_main(argc, argv);
           py::array_t<double> data = py::array_t<double>(3);
           // Fill array with [final_step, final_time, max_value]
           return py::make_tuple(result.success, data);
       });
   }

The key insight is that this simple pattern scales naturally to support more
complex use cases while maintaining the clean one-line interface.