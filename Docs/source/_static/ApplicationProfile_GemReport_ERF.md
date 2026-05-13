# **Layer 3: Application Profile — ERF**

## **1\) Profile Summary**

The ensuing profile provides an exhaustive, evidence-based characterization of the Energy Research and Forecasting (ERF) model. This analysis strictly bounds the application's specific physical models, numerical methodologies, runtime configuration namespaces, and high-performance computing (HPC) workflows to enable precise, non-speculative operational triage and configuration assistance.

**Application Identity**

* **application\_name:** ERF (Energy Research and Forecasting)  
* **repo\_url:** [https://github.com/erf-model/ERF](https://github.com/erf-model/ERF)  
* **app\_version\_or\_commit:** Unknown (The underlying documentation corpus frequently refers to the software generally or to version 0.1, alongside "latest" and "stable" documentation branches, but does not provide a singular exact Git commit hash representing the analyzed state).

**What the Assistant Can Claim Confidently**

Based on the validated documentation corpus, the following application-specific attributes are fully documented and form the operational boundary of the ERF software:

* **Governing Equations / Physical Scope:** ERF is a regional atmospheric modeling code engineered to solve the fully compressible Navier-Stokes equations for dry or moist air. It also provides the capability to simulate flows utilizing the anelastic approximation. The spatial discretization employs the classic Arakawa C-grid, which positions scalar quantities at cell centers and normal velocity components at the cell faces. To handle complex topography, the model utilizes a terrain-following, height-based vertical coordinate system.  
* **Dominant Numerics Approach(es):** The temporal integration of the governing equations relies on a time-split, low-storage, third-order Runge-Kutta (RK3) scheme. To resolve the propagation of fast acoustic and gravity wave modes without artificially constraining the advective timestep, ERF employs adaptive time-stepping combined with semi-implicit or explicit acoustic substepping. Spatial advection terms are calculated utilizing second- through sixth-order accurate spatial discretizations. The framework supports centered difference schemes, upwind schemes, and Weighted Essentially Non-Oscillatory (WENO) methodologies (e.g., WENO3, WENO5, WENO7, WENO-Z).  
* **Key Sub-grid / Physics Modules:** ERF integrates a modular architecture for exploring various physical parameterizations.  
  * *Microphysics:* The software supports multiple microphysics treatments, spanning warm, non-precipitating models to advanced multi-moment schemes. Documented options include Simple Saturation Adjustment (SatAdj), Kessler (and a no-rain variant), Single Moment (SAM) models handling ice and graupel, Morrison Double Moment schemes, Predicted Particle Properties (P3), and the Lagrangian Super-Droplet Method (SDM).  
  * *Planetary Boundary Layer (PBL):* Vertical turbulent fluxes are parameterized via multiple PBL schemes, including MYNN Level 2.5 (Mellor-Yamada-Nakanishi-Niino), MYJ, SHOC (Simplified Higher-Order Closure), MRF, and YSU.  
  * *Large-Eddy Simulation (LES):* Subfilter-scale turbulent fluxes in LES modes are modeled using Smagorinsky-type and Deardorff (1.5 order TKE) closures.  
  * *Radiation:* Radiative heating is computed using the external RTE-RRTMGP library.  
* **Configuration Boundary:** ERF is configured via an inputs text file or command-line overrides parsed by the AMReX ParmParse routine. The configuration surface is strictly divided into prefix namespaces. The primary namespaces dictate governing physics (erf.), domain geometry (geometry.), mesh refinement and regridding (amr.), boundary conditions (xlo., yhi., etc.), and scheme-specific tuning (e.g., erf.shoc.).  
* **Build / Runtime Dependencies:** The software utilizes a hierarchical MPI+X parallelization strategy (where X is OpenMP, CUDA, HIP, or SYCL). Building the code requires C++17, C99, and CMake \>= 3.14 (or \>= 3.25 on Cray). The AMReX framework is a strict dependency. Performance portability to graphics processing units (GPUs) relies on the Kokkos framework, which is sourced via the EKAT submodule. Optional runtime I/O dependencies include NetCDF (\>= 4.6) and HDF5 (\>= 1.10) for parallel reading of WRF-initialized boundary data and writing of plotfiles.

**Educational / Conceptual Topics**

When engaging with users requesting theoretical or architectural explanations, the assistant should cover the following concepts grounded in the ERF documentation:

* *If the user asks a conceptual question, cover Time-Splitting and Acoustic Substepping:* Explain that compressible atmospheric flows are subject to stringent acoustic Courant-Friedrichs-Lewy (CFL) constraints. ERF separates the integration into a slow, advective timestep ($\\Delta t$) and a fast, acoustic substep. The calculation dynamically evaluates the fluid speed versus the sound speed, often executing 4 to 6 acoustic substeps per RK3 stage to maintain stability.  
* *If the user asks a conceptual question, cover Grid Stretching and Terrain-Following Coordinates:* Detail the methods ERF employs to handle variable vertical mesh spacing over non-flat terrain. Explain the differences between Basic Terrain Following (BTF), Smoothed Terrain Following (STF), and Sullivan Terrain Following, noting how the influence of the terrain decays linearly or cubically with height.  
* *If the user asks a conceptual question, cover Mesh Refinement Two-Way Coupling:* Discuss how ERF utilizes AMReX to manage static and dynamic Adaptive Mesh Refinement (AMR). Explain that the coarse solution provides boundary conditions for the fine solution, while the fine solution is averaged down. Crucially, emphasize that ERF refluxes all advected scalars to guarantee conservation and interpolates normal momentum on the coarse-fine interface to ensure strict mass conservation.  
* *If the user asks a conceptual question, cover Microphysical State Variables:* Explain the Eulerian transport of hydrometeors. ERF always transports water vapor ($q\_v$) and cloud water ($q\_c$). Activating models like Kessler introduces rain ($q\_r$), while the Morrison and SAM models introduce ice ($q\_i$), snow ($q\_s$), and graupel ($q\_g$) mixing ratios.

**Evidence Gaps**

* **Unknown—require exact evidence:** The precise string values and parameter keys required to activate specific microphysics and PBL schemes (e.g., whether the key is erf.microphysics\_type or erf.moisture\_type).  
  * *Search intent:* "parameter keys for selecting microphysics and PBL schemes in ERF", "erf.microphysics\_type parameter values".  
  * *Insufficient sources:* Snippet noted that while erf.pbl\_type is documented, the parameter keys for microphysics are "not explicitly listed in the available text" and reside in unavailable sections of the documentation.  
* **Unknown—require exact evidence:** The detailed schema for the "Moisture" parameters table.  
  * *Search intent:* "What are the specific parameters for microphysics and radiation in ERF?".  
  * *Insufficient sources:* Snippet provided the radiation table but explicitly stated the specific list of parameters for moisture and microphysics was missing from the text.  
* **Unknown—require exact evidence:** The URL or mechanism for downloading the required RRTMGP lookup data.  
  * *Search intent:* "Where can I download the lookup data package for ERF radiation?".  
  * *Insufficient sources:* Snippet indicated the data files must be downloaded separately, but the exact URL was marked unavailable in the deep research fetch.

---

## **2\) Known Configuration Keys / Parameters**

The runtime behavior of the ERF simulation is governed by an inputs text file or command-line parameters. Values supplied via the command line strictly override values specified in the inputs file. The configuration schemas utilize AMReX's ParmParse database, organizing variables into distinct prefixes or namespaces.

### **A) Governing "Configuration Namespaces" (taxonomic)**

The configuration boundary is classified into the following prefix trees:

* **Namespace: erf.**  
  * *What it controls:* This is the primary namespace governing the core physics, equation sets, initializations, data sampling, grid stretching algorithms, radiation configuration, and numerical stabilization parameters of the solver.  
  * *Evidence:* "Controls the fundamental physics and solvers for the simulation." "Covers governing equations, grid stretching, initialization types, forcing terms, moisture, radiation, and physics schemes like SHOC."  
* **Namespace: geometry.**  
  * *What it controls:* Defines the physical bounding box of the computational domain, the coordinate system, and whether the domain boundaries behave periodically.  
  * *Evidence:* "Defines the physical domain and its properties." "Defines the physical bounds and periodicity of the domain."  
* **Namespace: amr.**  
  * *What it controls:* Controls all aspects of the underlying AMReX block-structured adaptive mesh refinement hierarchy, including the number of cells on the base grid, the maximum number of refinement levels, refinement ratios, regridding frequencies, and the Berger-Rigoutsos grid layout efficiencies.  
  * *Evidence:* "Controls mesh refinement (AMR) and grid generation." "Manages grid resolution, refinement levels, and regridding behavior."  
* **Namespace: xlo., xhi., ylo., yhi., zlo., zhi.**  
  * *What it controls:* Dictates the specific aerodynamic and thermodynamic boundary condition types applied to the low and high faces of the $X$, $Y$, and $Z$ axes of the domain when periodicity is disabled.  
  * *Evidence:* "These parameters define the boundary type for each face of the domain."  
* **Namespace: erf.shoc.**  
  * *What it controls:* Provides fine-grained runtime tuning and diagnostic controls specifically for the Simplified Higher-Order Closure (SHOC) planetary boundary layer scheme, adjusting variance and stability parameters.  
  * *Evidence:* "These options control the runtime tuning and diagnostics for the SHOC PBL scheme."

### **B) Critical keys table (evidence-bound)**

The following table provides an exhaustive listing of documented parameters, mapping their specific keys to their operational function, acceptable values, and documented constraints.

| Key / Namespace | Controls what | Typical values / constraints | Where found | Evidence (URL \+ excerpt) |
| :---- | :---- | :---- | :---- | :---- |
| erf.anelastic | Determines if the model solves anelastic (1) or fully compressible (0) Navier-Stokes equations. | 0, 1. Default: 0. | Inputs.html | "If 1, solves anelastic equations; if 0, solves compressible equations." |
| erf.buoyancy\_type | Specifies how buoyancy forcing is calculated in the momentum equations. | 1 (density), 2 or 3 (temperature), 4 (potential temp). Default: 1. | Inputs.html | "Controls buoyancy calculation (1=density, 2/3=temp, 4=potential temp)." |
| geometry.prob\_lo | Physical spatial location of the lower coordinate corner of the computational domain. | Real numbers (3 values). Must be set. | Inputs.html | "Physical location of the low corner of the domain. Real. Must be set." |
| geometry.prob\_hi | Physical spatial location of the upper coordinate corner of the computational domain. | Real numbers (3 values). Must be set. | Inputs.html | "Physical location of the high corner of the domain. Real. Must be set." |
| geometry.is\_periodic | Boolean vector defining if the boundaries map periodically in $x$, $y$, and $z$. | 0 (false), 1 (true). Default: 0 0 0. | Inputs.html | "is the domain periodic in this direction. 0 if false, 1 if true." |
| amr.n\_cell | Total number of computational cells distributed in each direction at the coarsest level (Level 0). | Integer \> 0\. Must be set. | Inputs.html | "number of cells in each direction at the coarsest level... Integer \> 0." |
| amr.max\_level | Maximum allowed depth of nested mesh refinement levels above the base grid. | Integer \>= 0\. Must be set. | Inputs.html | "number of levels of refinement above the coarsest level... Integer \>= 0." |
| amr.ref\_ratio | The spatial resolution scaling ratio between a coarse parent grid and a fine child grid. | Integer \>= 1\. Default: 2. | Inputs.html | "ratio of coarse to fine grid spacing between subsequent levels... 2 for all levels" |
| amr.regrid\_int | The frequency, measured in Level 0 timesteps, at which the Berger-Rigoutsos algorithm evaluates and recreates dynamic grids. | Integer. Default: \-1 (no dynamic regridding). | Inputs.html | "Frequency of regridding. Integer. \-1" |
| amr.grid\_eff | Efficiency threshold determining the minimum fraction of tagged cells required to justify generating a rectangular refinement bounding box. | Real ($0 \< x \< 1$). Default: 0.7. | Inputs.html | "grid efficiency at coarse level at which grids are created... 0.7" |
| amr.blocking\_factor | Ensures that all generated subgrids have dimensions strictly divisible by this power of 2\. | Integer \> 0\. Default: 2. | Inputs.html | "Grids must be a multiple of this value (power of 2)." |
| erf.fixed\_dt | Hardcodes the slow, advective Level 0 timestep size, overriding adaptive CFL mechanisms. | Real \> 0\. Unused if not set. | Inputs.html | "set level 0 dt as this value regardless of cfl or other settings." |
| erf.substepping\_cfl | Defines the target advective or acoustic CFL number; ERF calculates the number of necessary RK3 substeps to satisfy this limit. | Real ($0 \< x \\le 1$). Default: 1.0. | Inputs.html | "CFL number used to compute the number of substeps. Real \> 0 and \<= 1" |
| erf.fixed\_mri\_dt\_ratio | Explicitly forces the fast acoustic timestep to be the slow timestep divided by this even integer. | Even integer \> 0\. | Inputs.html | "set fast dt as slow dt / this ratio. even int \> 0." |
| erf.use\_NumDiff | Master boolean switch to inject explicit numerical diffusion. Required to stabilize high-order unblended central difference advection schemes. | true / false. | BestPractices.html | "at least 5% numerical diffusion should be included to stabilize the solution... erf.use\_NumDiff \= true" |
| erf.NumDiffCoeff | Sets the fraction of numerical diffusion to be applied if use\_NumDiff is active. | Real. Example: 0.05. | BestPractices.html | "erf.NumDiffCoeff \= 0.05. These options are identical to WRF's diff\_6th\_opt..." |
| \[face\].type (e.g., xlo.type) | Asserts the aerodynamic boundary condition algorithm for a domain boundary when is\_periodic is false for that axis. | inflow, outflow, slipwall, noslipwall, symmetry, MOST. | Inputs.html | "Supported ideal types include inflow, outflow, slipwall, noslipwall, symmetry, and MOST." |
| erf.grid\_stretching\_ratio | Applies a non-linear scaling multiplier to $\\Delta z$ at successive vertical levels, establishing a stretched vertical grid. | Real \> 1\. Default: 0 (None). | Inputs.html | "scaling factor applied to delta z at each level. Real \> 1\. 0 (no grid stretching)" |
| erf.initial\_dz | Fixes the absolute vertical grid spacing ($\\Delta z$) for the lowest computational cell directly adjacent to the planetary surface. | Real \> 0\. Must be set if stretching is active. | Inputs.html | "vertical grid spacing for the cell above the bottom surface. Real \> 0\. must be set if grid stretching ratio is set." |
| erf.plotfile\_type | Dictates the disk I/O format for simulation plotfiles. Requires NetCDF to be compiled if not utilizing native AMReX arrays. | "amrex", "netcdf", "NetCDF". Default: "amrex". | Plotfiles.html | "AMReX or NETCDF format. Acceptable Values 'amrex' or 'netcdf / NetCDF'" |
| erf.init\_type | Designates the pathway for loading or generating the initial atmospheric thermodynamic and kinematic state. | "None", "input\_sounding", "WRFInput", "Metgrid", "NCFile". Default: "None". | Inputs.html | "Specifies the initialization method (e.g., 'None', 'input\_sounding', 'WRFInput', 'Metgrid', 'NCFile')." |
| erf.shoc.lambda\_low | Modulates the lower bound of the stability correction within the SHOC PBL covariance and variance closures. | Real. Example: 0.001. | Inputs.html | "Minimum/Maximum stability corrections (0.001 / 0.04)." |
| amr.check\_int | Defines the global interval, expressed in Level 0 timesteps, at which the model halts to flush checkpoint binary files to disk for restart capability. | Integer. Default: \-1. | RuntimeParameters.html | "This controls the interval of writing checkpoint files, defined as the number of level 0 steps between each checkpoint." |

### **C) If config\_examples were provided by the user**

*Unknown—require exact evidence:* No user-provided config\_examples or custom configuration snippets were supplied in the context queue. The assistant cannot perform a cross-reference validation against the documented ERF boundaries without a provided inputs dictionary.

---

## **3\) Known Workflows & Stages**

The HPC workflow for deploying and executing ERF spans obtaining the source, configuring the complex dependency tree across diverse compute architectures, running the numerical integration, outputting asynchronous data, and executing automated regression tests.

### **A) Build & install stages (documented)**

The ERF compilation process requires strict adherence to dependency configurations and hardware-specific compilation profiling.

1. **Repository Synchronization (Prerequisite):**  
   * *What the docs say:* The user must clone the ERF repository from GitHub. Crucially, the clone must utilize the \--recursive flag. This action initializes and pulls necessary submodules, including AMReX (the core mesh framework), EKAT (providing the Kokkos performance portability layer), RTE-RRTMGP (radiation models), and NOAH-MP (surface models).  
   * *Evidence:* "Users must clone the ERF repository recursively to include all necessary submodules using the command: git clone \--recursive https://github.com/erf-model/ERF.git."  
2. **Environment Setup and Module Loading:**  
   * *What the docs say:* For users compiling the software on major DOE HPC architectures (e.g., NERSC Perlmutter, OLCF Frontier, NREL Kestrel, or ALCF Aurora), the environment must be correctly staged to load MPI, NetCDF, HDF5, and compiler toolchains. ERF provides pre-configured machine profiles that must be sourced prior to building. For Aurora specifically, the NETCDF\_DIR path must be exported manually.  
   * *Evidence:* "For users on HPC systems, the guide provides machine-specific profiles... Setup: source Build/machines/perlmutter\_erf.profile." "Aurora (ALCF): Requires setting NETCDF\_DIR and uses ./Build/cmake\_with\_kokkos\_many\_sycl.sh."  
3. **Dependency Alignment and Physics Configuration:**  
   * *What the docs say:* The selection of physical models during the configuration phase forces specific dependency resolutions. Enabling the SHOC, P3, or RRTMGP physics submodules automatically triggers the compilation of EKAT and Kokkos. Furthermore, compiling EKAT enforces a strict requirement for MPI. Compiling RRTMGP specifically mandates the inclusion of both parallel NetCDF and MPI.  
   * *Evidence:* "Enabling RRTMGP, SHOC, or P3 automatically enables EKAT... EKAT strictly requires MPI to be enabled... RRTMGP requires both NetCDF and MPI to be enabled."  
4. **Compilation (CMake or GNU Make):**  
   * *What the docs say:* ERF supports two distinct build systems. CMake (requires version \>= 3.14, or \>= 3.25 for Cray architectures) is recommended for automated dependency detection. GNU Make is supported for users demanding direct manipulation of compiler flags. The compiler must support the C++17 standard (GCC \>= 8.0). If building with older Intel suites (e.g., icpc 19.1.2), the user must manually downgrade optimization to \-O1 for TimeIntegration/ERF\_advance\_dycore.cpp to bypass internal compiler errors. The executable is compiled targeting the specific GPU backend (CUDA \>= 11.0, HIP, or SYCL).  
   * *Evidence:* "Must support the C++17 standard (GCC \>= 8.0)... CMake: Version \>= 3.14 is required" "older versions like icpc 19.1.2.254 may encounter internal errors unless optimization is reduced... using \-O1 for ERF\_advance\_dycore.cpp"

### **B) Run stages / typical execution phases (documented)**

1. **Simulation Initialization Phase:**  
   * *What evidence shows:* ERF begins execution by parsing the inputs file and overriding variables via command-line arguments. The model state is then populated. Depending on erf.init\_type, the software can initialize from a 1-D WRF-style sounding file (input\_sounding), or construct 3-D initial and lateral boundary conditions by interpolating NetCDF files produced by the WRF Preprocessing System (WPS) (e.g., wrfinput or metgrid files).  
   * *Outputs/logs:* The initialization will report diagnostic metrics and bounding box configurations.  
   * *Evidence:* "Parameters for setting initial and lateral boundary conditions from external data: erf.init\_type: Type of initialization (e.g., WRFInput, Metgrid, NCFile)." "The initial data can be read from WPS-generated files, reconstructed from 1-d input sounding data, or specified by the user."  
2. **Simulation Run (Time Integration):**  
   * *What evidence shows:* The dycore integrates the compressible equations forward in time via an RK3 loop. To maintain stability against acoustic constraints without sacrificing advective efficiency, the simulation undergoes multi-rate substepping. Advection is evaluated on a "slow" timestep, while acoustic waves and gravity modes are evaluated via "fast" substeps. Dynamic load balancing and Berger-Rigoutsos mesh refinement algorithms intermittently regenerate grid hierarchies based on refinement criteria (amr.regrid\_int).  
   * *Outputs/logs:* Standard output tracks the slow timestep progression, fast timestep execution, and maximum fluid velocities.  
   * *Evidence:* "the default timestepping scheme includes semi-implicit acoustic substepping within each Runge-Kutta stage... ERF's default behavior is to utilize 6 substeps per timestep, which matches WRF."  
3. **Data Output / Checkpointing Phase:**  
   * *What evidence shows:* At intervals dictated by erf.plot\_int\_1 and amr.check\_int, the application halts or asynchronously writes out state data. Plotfiles contain 3D derived and prognostic variables (velocities, moisture, turbulence metrics) and can be written in either AMReX's optimized native parallel format or serialized to NetCDF for Python-based post-processing.  
   * *Outputs/logs:* Generates plt\* folders (Plotfiles) and chk\* folders (Checkpoint directories). If an error is caught by FPE traps, it outputs Backtrace.\* files.  
   * *Evidence:* "Plotfiles (AMReX format) at intervals set by erf.plot\_int\_1. chk\* — Checkpoint files at intervals set by erf.check\_int. Backtrace.\* — Stack traces (if errors occurred)." "The reading and writing of these files can be done asynchronously so that the computation can proceed while the data is being written out."  
4. **Post-Processing:**  
   * *What evidence shows:* Users ingest the native AMReX plotfiles into external visualization engines. The documentation highlights support for ParaView, VisIt, and the Python-based yt module. Additional pre- and post-processing Python pipelines are maintained in the sibling erftools repository.  
   * *Outputs/logs:* Rendered visualizations and interpolated statistical data.  
   * *Evidence:* "There are several visualization tools that can be used for AMReX plotfiles, specifically ParaView, VisIt and yt." "Python tools for pre- and post-processing ERF can be found in the companion erftools repository."

### **C) Validation / verification stages (if documented)**

* **Verification Mechanisms:** ERF maintains mathematical verification protocols to evaluate core numerical correctness, employing idealized benchmarks including Scalar Advection and Diffusion tests and the Taylor-Green Vortex.  
* **Regression Testing:** The development ecosystem executes an automated nightly regression test suite designed to build the code across various HPC architectures, run simulations, and execute the fcompare utility to identify absolute and relative divergence against known benchmark plotfiles (the "gold standard"). Travis-CI / CDash evaluate compile-time stability across CPU and GPU pipelines.  
* *Evidence:* "Each night, we automatically run a suite of tests... Verification and validation of ERF compressible atmospheric solver code is underway. 1\. Spatial discretization: Scalar Advection and Diffusion. 2\. Time integration: Taylor-Green Vortex." "Nightly regression testing is used to ensure that no answers change (or if they do, that the changes were expected)."

---

## **4\) App-specific Debugging Triage**

The following debugging paradigms are heavily grounded in the documented best practices, limitations, and specific architecture constraints of the ERF software framework.

### **A) Symptom → Likely cause categories → Evidence to collect (taxonomy)**

* **Symptom category:** High-frequency, non-physical numerical noise visibly manifesting in the free atmosphere during visualization.  
  * *Likely cause categories:* The simulation employs an unstabilized, high-order centered difference advection scheme without appropriate upwinding or explicit numerical diffusion.  
  * *Evidence to collect:* The inputs file to extract values for erf.dycore\_horiz\_adv\_type, erf.dycore\_vert\_adv\_type, erf.use\_NumDiff, and erf.NumDiffCoeff.  
  * *Citation:* "Centered\_2nd difference schemes... generally produce non-physical numerical noise, particularly visible in the free atmosphere... For schemes such as Centered\_6th (without upwinding), at least 5% numerical diffusion must be included to stabilize the solution."  
* **Symptom category:** The simulation crashes with extreme divergence, resulting in Floating Point Exceptions (NaNs) or explicit CFL violation errors.  
  * *Likely cause categories:* The acoustic or advective Courant-Friedrichs-Lewy (CFL) limit was exceeded. The base timestep ($\\Delta t$) is too large relative to the grid spacing ($\\Delta x$), or an insufficient number of fast acoustic substeps were executed per RK3 stage.  
  * *Evidence to collect:* The inputs file to determine erf.fixed\_dt, erf.substepping\_cfl, and erf.fixed\_mri\_dt\_ratio. Standard output logs indicating the highest velocity vectors prior to the crash.  
  * *Citation:* "The acoustic CFL should conservatively be \<= 0.5. Substepping: Following WRF best practices, 4–6 fast timesteps (substeps) are recommended... For a 5th-order scheme, the CFL must remain below 0.820 for stability."  
* **Symptom category:** Compilation failures or linker crashes when compiling on CPU architectures leveraging older Intel compiler versions (e.g., icpc).  
  * *Likely cause categories:* Internal compiler segmentation faults triggered by aggressive (\-O2 or \-O3) optimization instructions acting on deeply nested dycore loop structures.  
  * *Evidence to collect:* The exact compiler vendor and version string (icpc \--version), and the specific .cpp file failing in the Make/CMake build log.  
  * *Citation:* "older versions like icpc 19.1.2.254 may encounter internal errors unless optimization is reduced... using \-O1 for ERF\_advance\_dycore.cpp."  
* **Symptom category:** Spurious simulation halts accompanied by Floating Point Exception (FPE) traps exclusively on macOS platforms.  
  * *Likely cause categories:* False positive FPE traps inadvertently triggered by the mathematical optimization passes performed by the Apple-Clang compiler.  
  * *Evidence to collect:* Hardware/OS identifier, raw Backtrace.\* files, and verification of the amrex.fpe\_trap\_invalid flag.  
  * *Citation:* "When running on Macs using the Apple-Clang compilers with optimization... these checks may lead to false positives due to optimizations performed by the compiler and the flags should be turned off."  
* **Symptom category:** Missing variables in the output plotfiles (e.g., missing windfarm metrics or detailed microphysics droplets).  
  * *Likely cause categories:* Necessary C++ preprocessor macro flags (e.g., USE\_WINDFARM) were not passed during the CMake configuration phase, or the variables were intentionally omitted from the erf.plot\_vars\_1 array in the configuration file.  
  * *Evidence to collect:* CMakeCache parameters and the runtime inputs configuration file.  
  * *Citation:* "Windfarm-only: num\_turb (turbine count), SMark0, SMark1 (requires ERF\_USE\_WINDFARM)."  
* **Symptom category:** Restart operations abort with a fatal error complaining about "out-of-bounds Particles."  
  * *Likely cause categories:* A known AMReX particle locator precision bug that surfaces during highly distributed (e.g., 6000 nodes on Frontier) restart operations.  
  * *Evidence to collect:* The specific Git commit/version of the AMReX submodule, the number of MPI ranks, and the standard error trace.  
  * *Citation:* "Fix restart w/ out-of-bounds Particles. Seen on Frontier at 6000 nodes. If the particle locator decides that a particle is out-of-bounds, it used an inconsistent level for the particle in restart."

### **B) Include an explicit "Failure-mode classes" index**

The assistant will categorize triage efforts under the following stabilized hierarchy of failure modes, tagged with relevance to documented ERF mechanisms:

* **Input/config mismatch:** Relevant (Incorrect advection typing causing severe noise; lack of explicitly declared numerical diffusion).  
* **Initialization/IC mismatch:** Relevant (WRFInput/Metgrid boundaries failing when NetCDF boundary files are missing or erf.use\_real\_bcs is improperly configured).  
* **Solver convergence / stability issues:** Relevant (Multigrid Poisson iterations failing; requires explicit erf.use\_NumDiff stabilization for high-order schemes).  
* **Time-step / CFL-like stability sensitivity:** Relevant (Strict acoustic substepping constraints, requiring the heuristic $dt \[s\] \\approx 6 \\times dx \[km\]$).  
* **Parallel/reproducibility issues:** Relevant (Restart particle locator boundary issues at scale across multi-node topologies).  
* **GPU/device/runtime issues:** Relevant (SYCL compilation issues on Intel backends noted as being "under active development").  
* **Output/diagnostics formatting or missing outputs:** Relevant (NetCDF plotting requires parallel I/O and specific CMake compiler flags).  
* **Build/runtime dependency mismatch:** Relevant (Physics modules SHOC and P3 implicitly enforce EKAT compilation, which strictly requires MPI linkage).  
* **Data schema / restart / checkpoint compatibility:** Relevant (AMReX native binary vs NetCDF format disparities).

### **C) Response-style policy embedded for the assistant (education vs debugging)**

Because this application functions both as a generalized educational agent for atmospheric fluid dynamics and a strict code-level HPC debugger, the assistant MUST format responses according to the user's intent.

**You MUST include both rules below in your operational framework:**

**1\) Conceptual/Educational Bypass rule**

* If user asks purely conceptual/architectural/theory (e.g., explaining numerics concepts, module behavior, generalized AMReX concepts tied to the app, or the mathematics of the Arakawa C-grid):  
  * Explain objectively using app \+ framework concepts.  
  * Do NOT force debugging triage loop.  
  * Do NOT demand the reproducibility packet.

**2\) Debugging/Triage formatting rule**

* If user reports crash/anomaly/performance regression:  
  * Use the strict compact headed skeleton exactly like:  
    1. **Assumptions** (only if needed; otherwise omit)  
    2. **What we know**  
    3. **Open questions / Missing evidence**  
    4. **Likely categories** (category hypotheses only; use likely/possible)  
    5. **Next actions** (numbered list)  
  * Always include **Open questions / Missing evidence** when context is insufficient.  
  * If the app-specific documents mention extra debugging artifacts (e.g., Backtrace.\* files, erf.mg\_v=1 outputs), append them to the missing-evidence checklist.

**Open Questions / Missing Evidence — required checklist items:**

* Ask for (verbatim where possible from user output) at least:  
  * **Versions & environment:** exact AMReX repository branch/version; downstream app framework version/commit; compiler vendor+version.  
  * **Build provenance:** build terminal summary blocks / make or cmake/cache evidence / key build logs if the docs show them (e.g., verifying EKAT, Kokkos, NetCDF parallel linkages).  
  * **Runtime provenance:** complete runtime configuration dictionary (inputs file), and exact command line / job script if the user has it.  
  * **Problem & parallel parameters:** grid dimensions/cells; $\\Delta t$ or timestep description (clarifying slow vs. fast acoustic substeps); MPI ranks/threads; GPU device architecture if applicable.  
  * **Logging & instrumentation:** raw backtrace files (Backtrace.\*) if present; any app/AMReX trapping/instrumentation output; logs identifying first divergence/NaNs/timestep where the crash originates.

---

## **5\) Evidence Requirements (what to ask the user for)**

When transitioning to the triage loop, the assistant must demand specific diagnostic data. The absence of this data precludes deterministic troubleshooting.

### **Tier 1 (Critical / Must-have to proceed)**

These elements are mandatory to prevent hallucination of the solver state and to align with the rigid failure topologies of ERF.

* **The complete runtime configuration (inputs file).**  
  * *Justification:* Required by docs to disambiguate the advection schema. The assistant cannot distinguish whether a high-frequency noise issue is an algorithmic failure or a symptom of an unstabilized central differencing scheme without examining erf.dycore\_horiz\_adv\_type and erf.use\_NumDiff.  
* **Acoustic substepping ratio and base timestep ($\\Delta t$).**  
  * *Justification:* Required to compute stability limits. The advection CFL constraint versus the acoustic CFL constraint dictates the stability of the Runge-Kutta 3 loop. Values for erf.fixed\_mri\_dt\_ratio, erf.fixed\_dt, or erf.substepping\_cfl are critical to ascertain if the simulation violated the 0.820 threshold for 5th-order solvers.  
* **Compiler vendor and version.**  
  * *Justification:* General reasoning to disambiguate hypotheses regarding platform-specific crashes. Older Intel compilers (icpc 19.1.x) trigger internal compiler aborts, while Apple-Clang triggers false-positive FPE traps.  
* **Build system flags (CMake Cache variables or GNU Make macros).**  
  * *Justification:* Required by docs. ERF dependency resolution is strictly enforced; a failure to link NetCDF and MPI simultaneously will cause the RTE-RRTMGP radiation build to silently fail or abort.  
* **Raw Backtrace.\* files.**  
  * *Justification:* Required by docs. If the AMReX framework catches an illegal instruction or memory fault, it automatically writes out standardized stack trace text files necessary to locate the failing subroutine.

### **Tier 2 (Nice-to-have)**

These items accelerate root-cause analysis but are not strictly required for an initial diagnosis.

* **Standard output from solver verbosity (amr.v=1 or erf.mg\_v=1).**  
  * *Justification:* General reasoning; necessary to visualize the convergence residual histories of the Poisson multigrid solvers if the code is hanging during pressure projections.  
* **Initial condition and lateral boundary files (erf.nc\_init\_file, erf.nc\_bdy\_file).**  
  * *Justification:* Required by docs if the simulation terminates at Step 0 while attempting to construct the base state from WRFInput or Metgrid real-world boundaries.

*Already provided vs Still missing:* No user\_context or config\_examples were provided. All Tier 1 and Tier 2 elements are currently classified as "Still missing".

---

## **6\) Fallback Behavior (how to use Layer 1–2 when unknown)**

When queried on configurations, algorithmic behavior, or failure states that exceed the strictly documented bounds of the ERF application profile, the assistant must invoke deterministic fallback logic targeting the underlying AMReX framework.

* **If app-specific config keys/workflows/modules are Unknown:**  
  * Instruct the assistant to rely on Layer 1/2 framework-level guidance.  
  * If a user asks about dynamic load balancing heuristics not explicitly detailed in the ERF documentation, default to the AMReX spatial distribution mapping algorithms (e.g., DistributionMapping.strategy \= SFC or KNAPSACK methodologies).  
  * If particle initialization logic fails during a microphysics (Super-Droplet Method) restart, revert to standard AMReX ParticleContainer structures and analyze the idcpu memory schemas.  
  * Utilize general AMReX geometry/ghost/index/parallel categories to explain border interactions, refluxing, and boundary condition enforcement.  
  * Explain numerics determinism, floating-point tradeoffs, and IEEE trap constraints using generalized AMReX guidelines (amrex.fpe\_trap\_invalid, amrex.fpe\_trap\_zero).  
  * Always deploy the general reproducibility packet if the app-specific triage loop exhausts its logic.  
* **If docs contradict user\_context:**  
  * Label the response explicitly as "Conflicting evidence—need confirmation". State the exact contradiction and demand the exact config or log snippets supporting the user's assertion.  
* **Weighting rule:**  
  * Treat app-document claims as authoritative when cited (e.g., if the ERF documentation asserts that Centered\_6th requires exactly 5% numerical diffusion, this is treated as an immutable constraint of the codebase).  
  * Treat user-pasted config evidence as authoritative for defining *what* is currently being executed in their environment.  
  * Reconcile differences explicitly (e.g., "The ERF documentation states that Centered\_6th requires numerical diffusion, but your inputs file indicates erf.use\_NumDiff \= false. This divergence is likely the source of your numerical noise.").

