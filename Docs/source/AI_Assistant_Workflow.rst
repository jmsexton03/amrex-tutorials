Building an AMReX Knowledge Assistant
=====================================

Standard Large Language Models (LLMs) often hallucinate application-specific physics when asked about the AMReX framework. To mitigate this, we employ an "Architecture-First" mental model and a strict "Evidence Discipline" protocol. This ensures the AI assistant prioritizes the framework's fundamental structural mechanics over guessed scientific implementations.

.. note::
   **Quick Start:** If you use Google Gemini, you can skip the manual setup and directly use the pre-configured agent here: `AMReX Expert Assistant Gem <https://gemini.google.com/gem/1L8shw-xtLVkdkUI4im0NiP7xVaKSHy5d?usp=sharing>`_ or the `ERF Specialized Assistant <https://gemini.google.com/gem/1CYi43osCZtA-pqmuBOyw6AZQJpkQBHox?usp=sharing>`_.


The Knowledge Layers (Deep Research)
------------------------------------

The foundation of a reliable assistant requires a deep research context file. This file provides the "ground truth" regarding how AMReX operates independently of any specific physics application.

The context is split into two layers:

1. **Layer 1 (Framework Core):** Focuses strictly on index spaces (``Box``), data distribution (``MultiFab``), parallel execution (``MFIter``), and ghost cell management (``ngrow``).
2. **Layer 2 (HPC Numerics):** Focuses on floating-point non-associativity, parallel reduction determinism, and generic stability tradeoffs in block-structured AMR.

.. important::
   **Reusability:** The Layer 1 and Layer 2 reports are universal. Once generated or downloaded, you can reuse this exact same knowledge base to debug *any* AMReX-based application, from astrophysics to wind energy.

If you are interested in reproducing the research step, you can download the original prompts used to generate the knowledge base:

* :download:`Layer 1 Generation Prompt <_static/AMReX_Expert_GemReport_Layer1_prompt.md>`
* :download:`Layer 2 Generation Prompt <_static/AMReX_Expert_GemReport_Layer2_prompt.md>`

Instead of generating this yourself, you can download the finalized, combined knowledge report to provide to your LLM. *(Note: All pre-compiled reports provided in this tutorial were generated using **Gemini 3.1 Pro with Deep Research**)*.

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

There are two ways to use this workflow. Creating a dedicated Custom GPT or Gem is **highly recommended**, as it allows you to permanently save the AMReX Knowledge base and Instruction Prompt for all future debugging.

.. tabs::

   .. tab:: Custom GPT / Gem (Recommended)

      This method uses Retrieval-Augmented Generation (RAG) to build a permanent tool.

      1. **Create a New Agent:** Navigate to the "Create a Gem" (`Gemini <https://gemini.google.com/gems/create>`_) or "Create a Custom GPT" (`ChatGPT <http://chatgpt.com/gpts/editor>`_) builder interface.
      2. **Set the Instructions:** Copy the "Persona Blueprint" text above and paste it into the *Instructions* or *System Prompt* box.
      3. **Upload Knowledge (RAG):** Upload the downloaded ``AMReX_Expert_GemReport_Layer1-2.md`` file to the *Knowledge* or *Files* section of the builder.
      4. **Save and Share:** Save your agent. You now have a permanent AMReX assistant that will automatically cross-reference the framework rules before answering your questions!

   .. tab:: Individual Chat Sessions

      This method is for quick, one-off analysis in standard chat windows.

      1. **Initialize Session**: Open a new chat session in your preferred LLM.
      2. **Upload Context**: Drag and drop the downloaded ``AMReX_Expert_GemReport_Layer1-2.md`` file into the chat window.
      3. **Set Persona**: Paste the complete "Persona Blueprint" text into the chat and send it as your first message.
      4. **Ask Questions**: You can now ask questions about AMReX fundamentals, such as *"How do MultiFabs handle MPI communication?"*

Extending to Downstream Applications (Layer 3)
----------------------------------------------

While the base assistant is tuned for pure AMReX fundamentals (Layers 1 and 2), you can extend it to analyze massive, domain-specific scientific applications built on top of AMReX. We call this the "Layer 3 Application Profile."

Because LLMs frequently hallucinate application-specific parameter keys, you must extract a strict, evidence-based profile of the application before asking for debugging help. You **do not** need to regenerate the AMReX framework fundamentals—Layers 1 and 2 are fully reusable! You simply swap out the Persona Blueprint for your new Layer 3 profile.

Generating a Custom Layer 3 Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a profile for *any* AMReX-based application (such as Pele, WarpX, or Castro), use the generic Layer 3 generation prompts. These prompts force the LLM to read your application's README, inputs, or documentation snippets, and summarize them into a strict ruleset. 

.. seealso::
   **WarpX Users:** The WarpX team maintains their own official guide for integrating LLM assistance. See the `LLM-Assisted WarpX Development <https://warpx.readthedocs.io/en/latest/developers/llm_assisted_warpx_development.html>`_ documentation.

Depending on your preferred LLM, download the appropriately optimized prompt:

* :download:`Download Generic Layer 3 Prompt optimized for ChatGPT (Prompt A) <_static/Generic_Layer3_prompt_A.md>`
* :download:`Download Generic Layer 3 Prompt optimized for Gemini (Prompt B) <_static/Generic_Layer3_prompt_B.md>`

**To use these prompts:**
1. Open a new session in your target LLM.
2. Paste the Generic Layer 3 Prompt.
3. Fill in the ``INPUTS`` block with your application's name and paste in any relevant documentation or configuration files you have.
4. Save the LLM's output as your new **Layer 3 Context Report**.

Example: The ERF Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working with the **Energy Research and Forecasting (ERF)** model, we have already executed the generation prompt and created a specialized Layer 3 profile. 

.. seealso::
   **ERF Users:** For deeper integration and workflow automation tools, refer to the official `ERF Agentic Workflow <https://erf.readthedocs.io/en/latest/AgenticWorkflow.html>`_ in the ERF User Guide.

You can download the ERF-specific assets here *(Generated via Gemini 3.1 Pro with Deep Research)*:

* :download:`ERF Layer 3 Generation Prompt <_static/ERF_Expert_GemReport_Layer3_prompt.md>`
* :download:`ERF Layer 3 Context Report <_static/ERF_Expert_GemReport_Layer3.md>`
* :download:`ERF Layer 3 Instruction Prompt <_static/ERF_Expert_GemReport_Layer3_instruction.md>`

Initializing a Layer 3 Assistant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build your application-specific agent (e.g., in the Custom GPT/Gem builder):

1. **Upload Context:** Upload BOTH the generic ``AMReX_Expert_GemReport_Layer1-2.md`` document AND your specific Layer 3 Context Document into the Knowledge/Files section.
2. **Set Persona:** Paste your Application's Instruction Prompt (e.g., the ERF Layer 3 Instruction) into the Instructions box *instead* of the generic AMReX one.

This multi-layered RAG approach guarantees the AI understands both the foundation of the AMReX framework and the exact boundaries of your specific physics application.
