# **Layer 3: ERF Application Profile Report**

## **1\) Profile Summary**

The application under investigation is formally identified as ERF, an acronym denoting Energy Research and Forecasting. The primary source code repository for this application is publicly hosted and version-controlled at the specific URL: https://github.com/erf-model/ERF.git. At the time of this architectural profiling and diagnostic synthesis, the exact application release version, software tag, or specific commit hash is entirely unstated within the provided evidentiary inputs; therefore, the application version is explicitly characterized as unknown.

The ERF application operates as a highly specialized, high-performance computing (HPC) software system engineered explicitly for atmospheric modeling. Its inception is deeply intertwined with the United States Department of Energy's (DOE) strategic initiatives, particularly those originating from the Office of Energy Efficiency & Renewable Energy and the Wind Energy Technologies Office (WETO). The foundational capability claim of ERF centers on simulating multiscale environmental interactions that profoundly influence the entire lifecycle of wind power generation. This comprehensive lifecycle spans initial atmospheric resource characterization, the structural and aerodynamic design of wind plants, the operational logistics of energy production, and the ultimate integration of generated power into broader energy grids. The application is strategically positioned to assimilate procedural logic and mathematical formalisms from related WETO projects, including Mesoscale-Microscale Coupling and Offshore Wind Resource Science, thereby establishing itself as a foundational code base across numerous operational centers and renewable energy sectors.

A critical historical context frames ERF's architectural design. The original developmental strategy in 2018 sought to refactor the legacy Weather Research and Forecasting (WRF) model to support efficient multiscale computations, targeting Intel's Many Integrated Core (MIC) architectures. However, as the HPC hardware ecosystem experienced a rapid paradigm shift toward Graphics Processing Units (GPUs) between 2018 and 2019, a consensus was reached that WRF's fundamental software architecture was fundamentally incompatible with modern GPU-accelerated environments. Consequently, ERF was conceived as an entirely new code base constructed fundamentally upon the AMReX framework. AMReX provides the essential infrastructure for performance-portable, block-structured adaptive mesh refinement (AMR), enabling ERF to deploy efficiently across multiple emerging Exascale heterogeneous architectures. This integration ensures that ERF can seamlessly couple with AMRWind, a microscale wind plant code similarly built upon AMReX, facilitating high-fidelity simulations spanning diverse atmospheric flow regimes. Furthermore, the ERF repository maintains flexibility in its core dependency management, allowing developers to utilize an internal AMReX submodule located at Submodules/AMReX or, alternatively, to link against a pre-installed, external compilation of the AMReX framework.

Beyond these high-level architectural mandates and Exascale deployment strategies, ERF application specifics unknown; use Layer 1–2 guidance. The provided inputs lack documentation regarding the internal fluid dynamics equations, turbulence parameterizations, thermodynamic solvers, and specific discretization schemes utilized within the atmospheric model.

To ensure comprehensive evidence discipline, the following diagnostic and contextual components are explicitly identified as missing:

* No config\_examples were provided; cannot name ERF keys or verify specific boundary condition and physics parameterizations.  
* No known\_symptoms or error logs were provided; cannot identify the specific failure mode category or current runtime anomalies experienced by the user.  
* No explicit diagnostic outputs or parallel layout configurations were detailed; cannot ascertain the scaling efficiency or domain decomposition parameters.  
* No system architecture or exact build provenance was supplied by the user; cannot verify the specific GPU backend or compiler toolchain currently in use.

## **2\) Known Configuration Keys / Parameters (from inputs only)**

Given the absence of user-provided configuration examples and the limited scope of the documentation snippets, ERF-specific internal physical parameter keys (such as those controlling atmospheric physics, thermodynamics, or mesh resolution) are currently absent. No configuration keys were provided in inputs that dictate ERF-specific atmospheric physics; unknown.

However, the provided documentation extensively details framework-level configuration keys, environment variables utilized during the build process, and specialized AMReX function pointers utilized for runtime mathematical parameterization. These parameters are essential for establishing the compilation environment and initializing the underlying adaptive mesh infrastructure. The known parameters, derived strictly from the verbatim inputs, are synthesized in the following table.

| Key / Parameter | Controlled Behavior | Evidence Location Hint |
| :---- | :---- | :---- |
| AMREX\_DEFAULT\_INIT | An environment variable that establishes the lowest-precedence method for setting site-wide or machine-specific framework defaults. It allows HPC administrators to inject parameters in job scripts without modifying the application's core input files or its command-line execution arguments. | Appears in the provided AMReX documentation snippet discussing parameter value precedence. |
| ERF\_BUILD\_DIR | An optional build-system environment variable utilized to customize the destination directory for intermediate compilation artifacts and object files during the CMake configuration process. | Appears in the provided config snippet section around the generic installation instructions. |
| ERF\_SOURCE\_DIR | An optional build-system environment variable used to explicitly map the location of the ERF source code tree, allowing out-of-source builds to correctly identify header files and submodule dependencies. | Appears in the provided config snippet section around the generic installation instructions. |
| ERF\_INSTALL\_DIR | An optional build-system environment variable that defines the final output directory where the fully compiled erf\_exec binary and associated executable assets will be deployed upon successful compilation. | Appears in the provided config snippet section around the generic installation instructions. |
| ERF\_HOME | A highly critical environment variable systematically used across all documented machine profiles to dynamically set the root working directory of the project, universally initialized to $(pwd) prior to invoking the Kokkos CMake scripts. | Appears systematically across all HPC machine profile workflows in the documentation snippets. |
| NETCDF\_DIR | A mandatory dependency mapping variable required specifically on the Aurora (ALCF) HPC system to locate the site's external NetCDF library installation, enabling parallel atmospheric data I/O. | Appears strictly within the Aurora build instructions in the provided snippets. |
| erf | A mathematical function pointer that can be passed directly to amrex::Initialize. It evaluates the standard mathematical error function and serves as a high-precedence parameter injection point. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| jn(n,x) | A specialized mathematical initialization function defining the Bessel function of the first kind of order $n$. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| yn(n,x) | A specialized mathematical initialization function defining the Bessel function of the second kind of order $n$. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| comp\_ellint\_1(k) | A mathematical function pointer representing the complete elliptic integral of the first kind. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| comp\_ellint\_2(k) | A mathematical function pointer representing the complete elliptic integral of the second kind. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| heaviside(x1) | A mathematical function defining the Heaviside step function, often used in boundary or initial condition logic within the framework. | Appears in the provided AMReX documentation snippet defining initialization functions. |
| inputs\_most | The verbatim name of a specific input configuration file invoked during the execution of the Canonical Atmospheric Boundary Layer (ABL) tests. The internal keys of this file are Unknown—require exact config/docs evidence. | Appears in the generic execution and job submission commands. |

A comprehensive understanding of these keys requires an in-depth analysis of the parameter precedence hierarchy dictated by the AMReX framework. In Exascale environments, debugging anomalous behavior often traces back to configuration conflicts where lower-precedence inputs are silently overridden. The inputs define a strict, four-tier precedence hierarchy for AMReX parameter evaluation. At the highest tier of precedence, any function pointer passed directly to amrex::Initialize supersedes all other declarations. This allows developers to embed unalterable logic directly into the source execution path. The second tier of precedence belongs to command-line arguments, which are parsed immediately upon binary invocation, allowing users to override configuration files on a per-run basis. The third tier encompasses the standard input file settings (e.g., configurations populated within inputs\_most), which represent the user's intended baseline simulation state. Finally, the lowest tier is governed by the AMREX\_DEFAULT\_INIT environment variable. This specific environmental key is of profound importance for HPC system administrators; it provides a non-intrusive mechanism to set site-wide or machine-specific topological defaults within job scripts without interfering with user-defined application inputs or command-line parameters. When an application fails to exhibit the behavior defined in inputs\_most, diagnostic workflows must rigorously audit this hierarchy to ensure that AMREX\_DEFAULT\_INIT or command-line flags are not creating undocumented overrides.

## **3\) Known Workflows & Stages (from inputs only)**

The deployment and execution of ERF on Exascale computational architectures is governed by a highly structured, machine-specific compilation workflow. Based strictly on the provided evidence, the methodology for initiating the application spans four primary stages, progressing from repository acquisition through hardware-specific configuration, multi-backend compilation, and ultimately, distributed execution.

1. **Stage 1: Source Acquisition and Submodule Initialization**  
   * **Stage name:** Repository Cloning and Framework Preparation.  
   * **Entry conditions:** The user must have a functioning git environment with access to external repositories.  
   * **Typical outputs/evidence mentioned:** Generation of the local ERF directory containing the primary source files and dynamically linked submodules.  
   * **Common mismatch points mentioned in inputs:** The process explicitly requires the \--recursive flag during the clone command (git clone \--recursive https://github.com/erf-model/ERF.git). Given that ERF relies fundamentally on the AMReX framework, which is situated by default as an internal submodule located at Submodules/AMReX , failure to invoke recursive cloning results in an incomplete source tree, precipitating immediate build failures due to missing foundational mesh logic.  
2. **Stage 2: HPC Architecture Environment Profiling**  
   * **Stage name:** Machine Profile and Environment Setup.  
   * **Entry conditions:** Navigation into the root ERF directory and identification of the target HPC architecture.  
   * **Typical outputs/evidence mentioned:** The population of machine-specific environment variables and the loading of optimized compiler toolchains dictated by .profile scripts. The architecture also establishes the ERF\_HOME path initialized to $(pwd).  
   * **Common mismatch points mentioned in inputs:** The user must perfectly align the sourced profile with the physical hardware allocation. The evidence explicitly details four supported Exascale environments :  
     * Sourcing Build/machines/perlmutter\_erf.profile for NERSC infrastructure.  
     * Sourcing Build/machines/kestrel\_erf.profile for NREL infrastructure.  
     * Sourcing Build/machines/frontier\_erf.profile for OLCF infrastructure.  
     * Sourcing Build/machines/aurora\_erf.profile for ALCF infrastructure. A specific mismatch point for Aurora involves the requirement to manually export the NETCDF\_DIR=\<path-to-netcdf\> variable; omission of this step guarantees linkage failure during the subsequent compilation phase.  
3. **Stage 3: Multi-Backend Compilation via CMake**  
   * **Stage name:** Kokkos-Enabled CMake Build Generation.  
   * **Entry conditions:** A correctly profiled environment where ERF\_HOME is verified, and optional build directories (ERF\_BUILD\_DIR, etc.) are established.  
   * **Typical outputs/evidence mentioned:** Deposition of the successfully linked application binary, named erf\_exec, typically routed to the install/bin directory.  
   * **Common mismatch points mentioned in inputs:** ERF relies heavily on Kokkos to abstract computational kernels across disparate GPU backends. The user must invoke the precise shell script corresponding to the hardware's native accelerator. A mismatch point occurs if a user invokes ./Build/cmake\_with\_kokkos\_many\_cuda.sh on Frontier (which requires the AMD HIP backend via \_hip.sh) or executes ./Build/cmake\_with\_kokkos\_many\_sycl.sh on Perlmutter (which relies on NVIDIA CUDA). General workstation builds must fallback to ./Build/cmake\_with\_kokkos\_many.sh.  
4. **Stage 4: Application Execution and Job Submission**  
   * **Stage name:** Distributed Launch and Job Scheduling.  
   * **Entry conditions:** Successful verification of the erf\_exec binary within the install/bin path.  
   * **Typical outputs/evidence mentioned:** The initiation of the application utilizing input parameter files (e.g., ../../Exec/CanonicalTests/ABL/inputs\_most). Output logs are delegated to standard scheduler outputs or terminal stdout/stderr.  
   * **Common mismatch points mentioned in inputs:** The execution command must match the system's distributed resource manager. Generic MPI executions utilize mpiexec \-n 4./erf\_exec. However, HPC deployments mandate the submission of specific batch scripts :  
     * Perlmutter, Kestrel, and Frontier require the use of the sbatch command pointing to specialized .sbatch files located in the ../../Docs/sphinx\_doc/scripts/quickstart/ directory.  
     * Aurora strictly mandates the qsub command targeting a .pbs script (submit\_erf\_aurora.pbs). Interchanging sbatch and qsub will result in immediate workload manager rejection.

## **4\) App-specific Debugging Triage (ERF-focused evidence categories)**

Because no user-described behaviors, runtime errors, or diagnostic logs were provided in the known inputs, an explicit mapping of distinct user symptoms to internal ERF mathematical or algorithmic faults cannot be constructed. Any specific claim regarding atmospheric instability, microscale turbulence modeling failure, or boundary condition divergence within the ERF application remains Unknown—require exact config/docs evidence.

However, based strictly on the mandated generic categories and the documented ERF integration with Exascale HPC environments, AMReX adaptive mesh infrastructure, and multi-GPU Kokkos backends, the following theoretical debugging triage matrix is established to guide subsequent diagnostic efforts.

### **Conflicts / Ambiguities in ERF application evidence**

No conflicts detected in provided inputs. The provided snippets present a unified and logically consistent framework detailing ERF’s structural dependence on AMReX, its Exascale deployment strategies, and its multi-backend Kokkos compilation paths.

### **Symptom: Application Compilation Failure or Library Linkage Halt**

* **Symptom description:** The invocation of the CMake build scripts (e.g., ./Build/cmake\_with\_kokkos\_many\_cuda.sh) terminates prematurely. The terminal outputs compiler errors, CMake dependency resolution failures, or Kokkos backend architecture mismatches.  
* **Likely cause categories:**  
  * Build/runtime environment mismatch category.  
* **Evidence to collect:**  
  * Exact config snippet(s) used, specifically any user-defined environment variables such as ERF\_BUILD\_DIR or NETCDF\_DIR.  
  * Full error log excerpt including the first failure point emitted by the compiler or the CMake configuration process.  
  * Any diagnostic output described in the provided snippets, specifically the terminal text verifying the successful sourcing of the machine profile (e.g., source Build/machines/perlmutter\_erf.profile).  
* **Next tests:**  
  * Isolate components: Verify the integrity of the source tree by re-running the recursive git submodule initialization to ensure the Submodules/AMReX path is fully populated.  
  * Repeat runs to probe reproducibility: Execute the generic ./Build/cmake\_with\_kokkos\_many.sh without specialized hardware flags to determine if the failure is strictly isolated to the Kokkos GPU mapping layer.

### **Symptom: Immediate Abort Upon Job Scheduler Submission**

* **Symptom description:** The erf\_exec binary is successfully compiled, but upon executing the sbatch, qsub, or mpiexec command, the application immediately aborts prior to entering the primary time-stepping loop. The failure occurs during parameter parsing or MPI initialization.  
* **Likely cause categories:**  
  * Configuration/initialization mismatch category.  
  * Parallel/reproducibility category.  
* **Evidence to collect:**  
  * Exact config snippet(s) used, specifically the contents of the inputs\_most file or the user-modified equivalent.  
  * Full error log excerpt including the first failure point, heavily prioritizing standard error (stderr) logs generated by the workload manager.  
  * Parallel layout / hardware description as provided by the user, detailing the number of MPI ranks (e.g., \-n 4) and the expected GPU-to-CPU binding logic.  
* **Next tests:**  
  * Smaller problem size / reduced case: Execute the binary with a singular MPI rank (mpiexec \-n 1) to explicitly isolate domain decomposition and MPI communication overhead from fundamental application initialization faults.  
  * Check environmental parameter injection: Audit the system for any predefined AMREX\_DEFAULT\_INIT variables that may be silently overriding the physical configurations defined in the input text files.

### **Symptom: Solver Divergence or Non-Physical Outputs During Execution**

* **Symptom description:** The ERF simulation initializes successfully and begins time-stepping, but subsequently generates non-physical numerical outputs (e.g., NaNs in the velocity or pressure fields) or experiences a catastrophic solver crash associated with grid refinement or atmospheric coupling operations.  
* **Likely cause categories:**  
  * Numerical instability category.  
  * Boundary/geometry/ghost handling category.  
* **Evidence to collect:**  
  * Exact config snippet(s) used to define the atmospheric initial conditions, boundary constraints, and spatial resolution.  
  * Full error log excerpt including the first failure point, focusing on the specific simulation timestamp or iteration cycle where the divergence occurred.  
  * Any convergence/residual/stability indicators if present in the logs, particularly those emitted by the AMReX linear solvers or fluid dynamics routines.  
* **Next tests:**  
  * Timestep sensitivity check: Systematically reduce the simulation timestep ($\\Delta t$) to evaluate if the instability is a result of violating the Courant–Friedrichs–Lewy (CFL) condition.  
  * Isolate components: If the simulation involves multiscale coupling with the AMRWind microscale code , temporarily decouple the components to determine if the numerical instability originates within the mesoscale ERF domain or specifically at the coupling interface.

## **5\) Evidence Requirements (what to ask the user for)**

To facilitate a precise and deterministic resolution of ERF application anomalies, the abstraction layer must strictly enforce an evidence-driven diagnostic methodology. Speculative parameter adjustment within Exascale atmospheric models is fundamentally counterproductive. Therefore, the user must be prompted to supply explicit evidence, stratified into the following two priority tiers.

### **Tier 1 (Must-have to avoid guessing)**

This tier encompasses the critical path data required to identify the fundamental nature of the computational fault. Without these artifacts, root-cause analysis cannot proceed beyond theoretical profiling.

* **The exact ERF config snippet(s) relevant to the symptom (verbatim paste):** The user must provide the explicit contents of the configuration file utilized during the execution (e.g., the exact parameters housed within inputs\_most or customized application inputs). Without this file, the specific atmospheric physics, grid resolution, and numerical solver toggles activated during the simulation remain categorically Unknown—require exact config/docs evidence.  
* **The full error log excerpt:** The user must supply the raw terminal output or batch scheduler log files. This must explicitly include the first occurrence of the error/divergence (as earliest timestamp/message). Submitting only the final segmentation fault or the concluding MPI abort message obscures the precipitating error, particularly in Kokkos/GPU asynchronous execution environments where failures cascade rapidly.  
* **Reproduction info:**  
  * *Execution string:* The user must provide exactly how they ran it. This includes the verbatim command line categories, specifying whether they utilized mpiexec, sbatch, or qsub, and the exact paths provided to the binary.  
  * *Parallel layout description:* The user must detail the parallel hardware allocation, including the specific number of ranks, thread counts, and the precise GPU count and architecture allocated per node.  
* **Environment/build provenance as text:** The user must paste the exact terminal history demonstrating their compilation workflow. If the user possesses it, they must provide the text detailing which machine profile was sourced (e.g., source Build/machines/frontier\_erf.profile), the specific CMake script invoked, and the explicit paths assigned to required variables like ERF\_HOME and NETCDF\_DIR. If they do not have this history, ask them to paste any version/build headers emitted by the erf\_exec output at startup.

**Current Evidence Status:**

* **Already provided:** None.  
* **Still missing:** Exact config snippets, full error logs, reproduction execution strings, parallel layout descriptions, and environment/build provenance text.

### **Tier 2 (Nice-to-have)**

This tier encompasses secondary diagnostic data that, while not strictly mandatory for initiating triage, significantly accelerates the isolation of complex, non-deterministic bugs inherent in highly parallel adaptive mesh applications.

* **Additional runs showing variability:** If nondeterminism is suspected—such as race conditions in Kokkos GPU kernels or anomalous MPI message ordering—the user should provide logs from $N$ repeat runs utilizing identical parameters to verify if the failure point fluctuates or remains perfectly deterministic.  
* **Convergence history metrics:** Any captured convergence history, residual decay rates, or thermodynamic stability metrics recorded in the AMReX standard output prior to the simulation crashing.  
* **Minimal case attempt description:** A structured summary of what the user has already attempted to isolate the issue. The user must clearly articulate what variables they changed (e.g., lowering the timestep, disabling AMRWind coupling) and what parameters stayed fixed during their internal debugging process.

**Current Evidence Status:**

* **Already provided:** None.  
* **Still missing:** Variability logs from repeat runs, solver convergence history, and minimal case attempt descriptions.

## **6\) Fallback Behavior (how to use Layer 1–2 when unknown)**

The ERF application relies on a deeply integrated, complex stack of mathematical frameworks, ranging from foundational multiscale physics concepts to extreme-scale computing infrastructure. Because the user inputs provided herein lack specific details regarding internal atmospheric configurations, proprietary workflows, and user-defined simulation metrics, it is statistically probable that queries will rapidly transcend the explicitly known parameters detailed in this profile.

To maintain strict evidence discipline and prevent the generation of hallucinated application workflows or fabricated configuration keys, the diagnostic compiler must rigidly enforce the following deterministic fallback rules.

1. **When ERF-specific elements are absent:** If you cannot name any ERF-specific keys or workflow stages based on the limited profile established in Sections 2 and 3, you must instruct the Layer 0 compiler to have the assistant seamlessly delegate diagnostic authority to foundational analytical layers:  
   * **Use Layer 1** for all structural debugging categories inherently related to the AMReX framework. This includes the resolution of block-structured adaptive mesh refinement (AMR) anomalies, distributed memory ghost cell exchange failures, multiscale geometry discretization logic, and overarching parallel layout and MPI/GPU synchronization issues. Because ERF is strictly an AMReX-based code, the underlying framework documentation provides the definitive truth for these infrastructural mechanics.  
   * **Use Layer 2** for all anomalies related to mathematical numerics, solver non-determinism, floating-point exception handling, and generic fluid dynamics triage flows. If the application is experiencing CFL violations or spatial numerical divergence, the universal principles of computational fluid dynamics apply independently of the specific ERF implementation details.  
2. **Handling partial ERF references:** If the user inputs provide partial ERF hints—such as referencing undocumented internal variable names, unverified algorithmic parameterizations, or assumed atmospheric physics behaviors not explicitly corroborated by Snippets through —treat those hints as fundamentally unverified until they can be confirmed by the exact config/log excerpts provided by the user in Tier 1 evidence.

### **Mandatory Weighting Rule**

The following weighting rule must govern the synthesis of all subsequent instructions and responses derived from this profile:

**"Treat ERF application-specific hints as unverified unless confirmed by the exact provided snippets/config/logs."**

