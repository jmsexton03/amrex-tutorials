Building an AMReX Knowledge Assistant
=====================================

Standard Large Language Models (LLMs) often hallucinate application-specific physics when asked about the AMReX framework. To mitigate this, we employ an "Architecture-First" mental model and a strict "Evidence Discipline" protocol. This ensures the AI assistant prioritizes the framework's fundamental structural mechanics over guessed scientific implementations.

.. note::
   **Quick Start:** If you use Google Gemini, you can skip the manual setup and directly use the pre-configured agent here: `AMReX Expert Assistant Gem <https://gemini.google.com/gem/1L8shw-xtLVkdkUI4im0NiP7xVaKSHy5d?usp=sharing>`_

The Knowledge Layers (Deep Research)
------------------------------------

The foundation of a reliable assistant requires a deep research context file. This file provides the "ground truth" regarding how AMReX operates independently of any specific physics application. 

The context is split into two layers:

1. **Layer 1 (Framework Core):** Focuses strictly on index spaces (``Box``), data distribution (``MultiFab``), parallel execution (``MFIter``), and ghost cell management (``ngrow``).
2. **Layer 2 (HPC Numerics):** Focuses on floating-point non-associativity, parallel reduction determinism, and generic stability tradeoffs in block-structured AMR.

If you are interested in reproducing the research step, you can download the original prompts used to generate the knowledge base:

* :download:`Layer 1 Generation Prompt <_static/AMReX_Expert_GemReport_Layer1_prompt.md>`
* :download:`Layer 2 Generation Prompt <_static/AMReX_Expert_GemReport_Layer2_prompt.md>`

Instead of generating this yourself, you can download the finalized, combined knowledge report to provide to your LLM:

.. note::
   Download the full context document here: :download:`AMReX Framework and Numerics Report <_static/AMReX_Expert_GemReport_Layer1-2.md>`

The Persona Blueprint
---------------------

Once the context document is uploaded to the LLM, the AI must be initialized with a specific persona that enforces the evidence boundary. 

You can download the standalone instruction file here: :download:`AMReX Assistant Instruction Prompt <_static/AMReX_Expert_GemReport_Layer1-2_instruction.md>`

Alternatively, copy and paste the complete instruction set below into your LLM's system prompt or first message.

.. literalinclude:: _static/AMReX_Expert_GemReport_Layer1-2_instruction.md
   :language: markdown
   :caption: The AMReX Expert Agent System Prompt

The Reproducible Workflow
-------------------------

To instantiate your AMReX Knowledge Assistant on any platform (e.g., Claude, ChatGPT), follow these exact steps:

1. **Initialize Session**: Open a new session in your preferred LLM.
2. **Upload Context**: Upload the downloaded ``AMReX_Expert_GemReport_Layer1-2.md`` file to the chat window.
3. **Set Persona**: Paste the complete "Persona Blueprint" text into the chat and send it.
4. **Ask Questions**: You can now ask questions about AMReX fundamentals, such as *"How do MultiFabs handle MPI communication?"* or *"Why might my parallel reduction give different results on different core counts?"*

Extending to Downstream Applications (Generic Mode)
---------------------------------------------------

While this assistant is tuned for pure AMReX fundamentals, you may eventually want to use it to analyze massive, domain-specific scientific applications built on top of AMReX (such as the Energy Research and Forecasting model [ERF], WarpX, or PeleC).

When standard LLMs attempt to debug these codes, they frequently blur the line between the AMReX framework operations (like ``FillBoundary()``) and the application's specific physics. To prevent this, append the **Application Taxonomy Extension** to your Persona Blueprint:

.. code-block:: text

   ### Generic Application Extension
   If the user provides an application name (e.g., ERF, PeleC, WarpX), you must apply the AMReX Application Taxonomy:
   - **Identify the Application Boundary:** You must explicitly ask the user for the specific `ParmParse` configuration dictionary for that app (e.g., `erf.my_param = 1`).
   - **Constraint:** Do not hallucinate the application's specific C++ classes or Fortran modules. You may only trace the application logic down to where it interfaces with the generic AMReX framework (`MultiFab`, `MFIter`).
   - **Ask for the Source:** If the user asks a physics-specific question (e.g., "Why is my Spalart-Allmaras turbulence model crashing?"), you MUST reply: "Unknown—require verification from user code. Please provide the specific application kernel source."

By explicitly defining the "Application Boundary," the AI will stop guessing how the physics are implemented and instead help you map your specific physics problem back to the underlying parallel data structures.
