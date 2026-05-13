Building an AMReX Knowledge Assistant
=====================================

Standard Large Language Models (LLMs) often hallucinate application-specific physics when asked about the AMReX framework. To prevent this, we use an **“Architecture-First”** mental model combined with a strict **“Evidence Discipline”** protocol. This makes the assistant prioritize AMReX’s fundamental structural mechanics (the framework’s real behavior) rather than guessed implementations. These examples are currently developmental and subject to change.

.. note::
   **TL;DR — Pick your path:**

   **Zero setup:** Use a pre-configured agent directly:
   `AMReX Expert Assistant Gem <https://gemini.google.com/gem/1L8shw-xtLVkdkUI4im0NiP7xVaKSHy5d?usp=sharing>`_ .

   **Permanent setup (recommended):** Build your own Custom Gem or GPT — download the
   :download:`knowledge base <_static/AMReX_Expert_GemReport_Layer1-2.md>` and
   :download:`instruction prompt <_static/AMReX_Expert_GemReport_Layer1-2_instruction.md>`,
   upload both to a new Gem (`Gemini <https://gemini.google.com/gems/create>`_) or
   GPT (`ChatGPT <http://chatgpt.com/gpts/editor>`_). Done.

   **Per-session:** Drag and drop the same two files into any chat window as your first message.

   **Extend to your application specifics:** See below — or jump straight to other solutions like using Context7 as described in the `WarpX LLM guide <https://warpx.readthedocs.io/en/latest/developers/llm_assisted_warpx_development.html>`_ or trying out the `ERF Specialized Assistant <https://gemini.google.com/gem/1CYi43osCZtA-pqmuBOyw6AZQJpkQBHox?usp=sharing>`_ or
   `ERF Agentic Workflow <https://erf.readthedocs.io/en/latest/AgenticWorkflow.html>`_ if you use those codes.

.. dropdown:: Extending to a specific application (Layer 3)

   Download :download:`Generic Layer 3 Prompt B <_static/Generic_Layer3_prompt_B.md>`, then choose your approach:

   .. tabs::

      .. tab:: Quick (2-word replace)

         Replace ``Generic`` with your application name in the title line and opening sentence. You can do this manually, or run:

         .. code-block:: bash

            APP="YourApp"
            sed "s/Generic Application Profile/${APP} Application Profile/; \
                 s/for an AMReX-based application/for ${APP}, an AMReX-based application/" \
                source/_static/Generic_Layer3_prompt_B.md > source/_static/${APP}_Layer3_prompt_B.md

      .. tab:: Full Context (Recommended)

         Fill in the ``INPUTS`` block in the downloaded prompt with everything you have before running:

         .. code-block:: text

            INPUTS (some may be missing)
            - application_name      (string; e.g., "ERF", "PeleC", "WarpX")
            - repo_url              (string; optional)
            - docs_url_list         (list of strings; optional)
            - app_version_or_commit (string; optional but strongly preferred)
            - user_context          (text; optional notes, symptoms, constraints)
            - config_examples       (pasted config snippets; optional)
            - known_symptoms        (optional)
            - known_config_keys     (optional)

   *Note: Because generating these context reports requires analyzing extensive documentation, you must run the resulting prompt through a Deep Research or reasoning-capable LLM.* Save the output as your **Layer 3 Context Report**, upload it alongside the AMReX knowledge base, and swap in your application's instruction prompt.


Detailed Build Explanation
==========================

If you are interested in reproducing the research step or want to understand how the assistant's boundaries are constructed, the following sections detail the complete architecture.

The Knowledge Layers (Deep Research)
---------------------------------------

To make the assistant reliable, it requires a deep research context file that acts as the **ground truth** for how AMReX operates—independent of any particular physics application.

We split the knowledge base into two reusable layers:

1. **Layer 1 (Framework Core):** Covers index spaces (``Box``), data distribution (``MultiFab``), parallel execution (``MFIter``), and ghost cell management (``ngrow``).
2. **Layer 2 (HPC Numerics):** Covers floating-point non-associativity, parallel reduction determinism, and generic stability tradeoffs in block-structured AMR.

.. important::
   **Reusability:** The Layer 1 and Layer 2 reports are universal. Once generated or downloaded, you can reuse this exact same knowledge base to debug *any* AMReX-based application.

If you want to run the research step yourself, you can download the original prompts:

* :download:`Layer 1 Generation Prompt <_static/AMReX_Expert_GemReport_Layer1_prompt.md>`
* :download:`Layer 2 Generation Prompt <_static/AMReX_Expert_GemReport_Layer2_prompt.md>`

*(Note: All pre-compiled reports provided in this tutorial were generated using **Gemini 3.1 Pro with Deep Research**).*


The Persona Blueprint (Instruction Boundary)
---------------------------------------------

After providing the context document to the LLM, you must initialize the model with a specific persona that enforces the evidence boundary. This prevents the LLM from making assumptions when it lacks context.

You can download the standalone instruction file here:
:download:`AMReX Assistant Instruction Prompt <_static/AMReX_Expert_GemReport_Layer1-2_instruction.md>`

Alternatively, copy and paste the complete instruction set below into your LLM's system prompt or first message.

.. literalinclude:: _static/AMReX_Expert_GemReport_Layer1-2_instruction.md
   :language: markdown
   :caption: The AMReX Expert Agent System Prompt


Extending to Specific Applications (Layer 3)
-------------------------------------------------

Once the base assistant is tuned for pure AMReX fundamentals (Layers 1 and 2), you can extend it to analyze large, domain-specific scientific applications built on top of AMReX. We call this the **“Layer 3 Application Profile.”**

Because LLMs frequently hallucinate application-specific parameter keys, you must extract a strict, evidence-based profile of the application before asking for debugging help. You **do not** need to regenerate the AMReX framework fundamentals—Layers 1 and 2 are fully reusable. You simply swap out the Persona Blueprint for your new Layer 3 profile.


Generating a Custom Layer 3 Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a profile for *any* AMReX-based application (such as Pele, WarpX, or Castro), use the generic Layer 3 generation prompts. These prompts force the LLM to read your application's README, inputs, or documentation snippets, then summarize them into a strict ruleset.

.. seealso::
   **WarpX Users:** The WarpX team maintains their own official guide for integrating LLM assistance. See the `LLM-Assisted WarpX Development <https://warpx.readthedocs.io/en/latest/developers/llm_assisted_warpx_development.html>`_ documentation.

Depending on your preferred LLM, download the appropriately optimized prompt:

* :download:`Download Generic Layer 3 Prompt optimized for ChatGPT (Prompt A) <_static/Generic_Layer3_prompt_A.md>`
* :download:`Download Generic Layer 3 Prompt optimized for Gemini (Prompt B) <_static/Generic_Layer3_prompt_B.md>`

**To use these prompts:**
1. Open a new session in your target LLM (ideally using a Deep Research or reasoning model).
2. Paste the Generic Layer 3 Prompt.
3. Fill in the ``INPUTS`` block with your application's name and paste in any relevant documentation or configuration files you have.
4. Save the LLM's output as your new **Layer 3 Context Report**.


Example: The ERF Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working with the **Energy Research and Forecasting (ERF)** model, we have already executed the Layer 3 generation prompt and created a specialized profile.

You can download the ERF-specific assets here *(Generated via Gemini 3.1 Pro with Deep Research)*:

* :download:`ERF Layer 3 Generation Prompt <_static/ERF_Expert_GemReport_Layer3_prompt.md>`
* :download:`ERF Layer 3 Context Report <_static/ERF_Expert_GemReport_Layer3.md>`
* :download:`ERF Layer 3 Instruction Prompt <_static/ERF_Expert_GemReport_Layer3_instruction.md>`

Initializing a Layer 3 Assistant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build your application-specific agent (e.g., in a Custom GPT/Gem builder):

1. **Upload Context:** Upload BOTH the generic ``AMReX_Expert_GemReport_Layer1-2.md`` document AND your specific Layer 3 Context Document (e.g., ERF's context report).
2. **Set Persona:** Paste your Application's Instruction Prompt (e.g., the ERF Layer 3 Instruction) into the Instructions box *instead* of the generic AMReX one.

This multi-layered approach guarantees the AI understands both the foundation of the AMReX framework and the exact boundaries of your specific physics application.


Prompt References & Instructions
--------------------------------

The following tables consolidate all prompts, reports, and instructions referenced throughout this guide for easy access.

**Generation Prompts and Context Reports:**

.. list-table::
   :widths: 20 30 30 20
   :header-rows: 1

   * - Layer/Profile
     - Interactive View (Sources)
     - Downloadable Prompt
     - Downloadable Report
   * - AMReX Framework Core
     - `View <https://gemini.google.com/share/0250e6cc7db3>`_
     - :download:`Prompt <_static/AMReX_Expert_GemReport_Layer1_prompt.md>`
     - :download:`Report <_static/AMReX_Expert_GemReport_Layer1-2.md>`
   * - AMReX Framework Core (Alt)
     - `View <https://gemini.google.com/share/c9f115cb0599>`_
     - ``N/A``
     - :download:`Report <_static/AMReX_Expert_GemReport_Layer1-2.md>`
   * - Portable HPC / Numerics
     - `View <https://gemini.google.com/share/8539c27d4752>`_
     - :download:`Prompt <_static/AMReX_Expert_GemReport_Layer2_prompt.md>`
     - :download:`Report <_static/AMReX_Expert_GemReport_Layer1-2.md>`
   * - Portable HPC / Numerics (Alt)
     - `View <https://gemini.google.com/share/c1d72c161e2e>`_
     - ``N/A``
     - :download:`Report <_static/AMReX_Expert_GemReport_Layer1-2.md>`
   * - ERF Generation Prompt
     - `View <https://gemini.google.com/share/7d1585e7b9af>`_
     - :download:`Prompt <_static/ERF_Expert_GemReport_Layer3_prompt.md>`
     - :download:`Report <_static/ERF_Expert_GemReport_Layer3.md>`
   * - ERF Application Profile
     - `View <https://gemini.google.com/share/14f97271bb23>`_
     - :download:`Prompt <_static/Generic_Layer3_prompt_B_2wordsreplaceERF.md>`
     - :download:`Report <_static/ApplicationProfile_GemReport_ERF.md>`

**Final Persona Blueprints (Instruction Prompts):**

These are the final instructions used to initialize the assistant.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Assistant
     - Instruction Prompt
   * - Generic AMReX Expert
     - :download:`Download <_static/AMReX_Expert_GemReport_Layer1-2_instruction.md>`
   * - ERF Specialized Expert
     - :download:`Download <_static/ERF_Expert_GemReport_Layer3_instruction.md>`
