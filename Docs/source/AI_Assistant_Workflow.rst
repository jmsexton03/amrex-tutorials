Building an AMReX Knowledge Assistant
======================================

Standard Large Language Models (LLMs) often hallucinate application-specific physics when asked about the AMReX framework. To mitigate this, we employ an "Architecture-First" mental model and a strict "Evidence Discipline" protocol. This ensures the AI assistant prioritizes the framework's structural mechanics over guessed implementations.

The Knowledge Layers
-------------------

The foundation of a reliable assistant requires generating two deep research context files. These files provide the "ground truth" regarding how AMReX operates independently of any specific physics application.

**Deep Research Prompt 1: Framework Core**

.. code-block:: text

   Generate a comprehensive report on the AMReX Framework Core (Layer 1). Focus strictly on index spaces (Box, BoxArray), data distribution (MultiFab), parallel execution (MFIter), and ghost cell management (ngrow). Do not include application-specific physics.

**Deep Research Prompt 2: HPC Numerics and Debugging**

.. code-block:: text

   Generate a report on Portable HPC Numerics and Debugging Core (Layer 2). Focus on floating-point non-associativity, parallel reduction determinism, and generic stability tradeoffs in block-structured AMR.

The Persona Blueprint
--------------------

Once the context documents are generated, the AI must be initialized with a specific persona that enforces the evidence boundary.

.. code-block:: text

   Act as the AMReX Knowledge Assistant. Your goal is to explain how the AMReX framework works using an Architecture-First approach. 

   Constraints: 
   1. Evidence Boundary: If a detail is not in the provided code or logs, state: 'Unknown—this appears to be application-specific logic.'
   2. Educational Philosophy: Always explain the Index Space, Data Distribution, Execution, and Boundary Handling before discussing tutorial specifics.
   3. No hallucinated ParmParse keys.

The Reproducible Workflow
-------------------------

To instantiate your AMReX Knowledge Assistant, follow these steps:

1. **Generate Context**: Run the two Deep Research prompts in a capable LLM to create the Knowledge Layer documents.
2. **Initialize Session**: Open a new LLM session and upload the resulting Context Documents.
3. **Set Persona**: Paste the Instruction Prompt (Persona Blueprint) into the chat.
4. **Provide Source**: Paste the specific ``amrex-tutorials`` source code you are analyzing (e.g., ``main.cpp`` and the ``inputs`` file).

.. note::
   **The Application Boundary**: It is critical to distinguish between what AMReX handles (the "plumbing" such as Boxes, MultiFabs, and ghost cell exchange) and what the specific tutorial implements (the "physics" or custom computational kernels). The assistant is trained to identify this boundary.
