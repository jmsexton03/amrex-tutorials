LAYER 3 PROMPT (Generic Layer 3: Application Profile)

You are generating a “Layer 3: Application Profile” report for a specific code/application.
This layer must be application-aware ONLY when the user provides enough evidence (name/repo/snippets/config).
Otherwise it must explicitly fall back to Layer 1–2 guidance.

GOAL:
Create an application-specific profile report that another prompt (Layer 0) can compile into final gem instructions, while never inventing application-specific config keys, APIs, or workflows.

INPUTS (only these are guaranteed; others may be empty/omitted):
- application_name (optional TEXT; e.g., user’s label/name for the app)
- repo_url (optional TEXT; a URL string provided by the user)
- user_context (optional TEXT; may contain pasted notes, errors, or descriptions)
- config_examples (optional TEXT; user-provided config snippets)
- app_docs_snippets (optional TEXT; pasted documentation excerpts)
- known_symptoms (optional TEXT; user-described behavior/errors)
- known_config_keys (optional TEXT; list provided by user)

HARD CONSTRAINTS (must follow):
1) No invention:
   - Do not fabricate application-specific parameter names/config keys/workflow steps/log identifiers.
   - If you cannot confirm a detail from inputs, write: “Unknown—require exact config/docs evidence.”
2) Evidence-first:
   - Only mark things “Known” if they appear explicitly in user_context/config_examples/app_docs_snippets/known_config_keys.
3) No external research:
   - Do not open repo_url or consult external sources.
4) Conflict handling:
   - If inputs disagree, note “Conflicting evidence—need confirmation” and list the conflict.
5) Fallback requirement:
   - If application-specific evidence is insufficient, produce a minimal profile that delegates to Layer 1–2.

TASK:
Write a structured application profile report with the REQUIRED headings, in this exact order.

REQUIRED OUTPUT STRUCTURE (Markdown with these headings):
1) Profile Summary
2) Known Configuration Keys / Parameters (from inputs only)
3) Known Workflows & Stages (from inputs only)
4) App-specific Debugging Triage
5) Evidence Requirements (what to ask the user for)
6) Fallback Behavior (how to use Layer 1–2 when unknown)

CONTENT REQUIREMENTS:

1) Profile Summary
- Include:
  - application_name: (value or “unknown”)
  - repo_url: (value or “none provided”)
- What you can claim:
  - Only state that application-specific details are uncertain unless supported by inputs.
- If minimal inputs:
  - If only application_name or repo_url is provided and no config/snippets/errors:
    - state: “Application-specific details unknown; use Layer 1–2 guidance.”
- Explicitly list “What’s missing” as bullets (e.g., “No config snippets provided; cannot name keys”).

2) Known Configuration Keys / Parameters (from inputs only)
- If config_examples/app_docs_snippets/known_config_keys/user_context include any config keys:
  - list them verbatim as a bullet list or table.
- For each key, include:
  - Key name (verbatim)
  - What it appears to control (only what is literally described in inputs; otherwise “Unknown—require exact docs/config evidence”)
  - Where to look in evidence (e.g., “check the provided snippet around this key”) if you can only infer location from the snippet.
- If none are present:
  - write: “No configuration keys were provided in inputs; unknown.”

3) Known Workflows & Stages (from inputs only)
- If inputs mention workflow steps/stages:
  - provide a numbered list of stages, each with:
    - Stage name (as given or plainly labeled)
    - Entry conditions (only if stated in inputs; else “Unknown—require evidence”)
    - Outputs/evidence mentioned in inputs (files/logs/diagnostics if provided; else “Unknown”)
    - Common mismatch points mentioned in inputs (else “Unknown”)
- If no workflow info is present:
  - Provide ONLY a clearly labeled generic skeleton, such as:
    1. Build (unknown app-specific steps)
    2. Configure/run (unknown app-specific steps)
    3. Diagnose/analyze (unknown app-specific steps)
  - Explicitly label it as “generic/approximate; not app-specific.”

4) App-specific Debugging Triage
- Create a symptom → evidence → test plan map.
- Use only symptoms present in known_symptoms or described in user_context.
- For each symptom (or “generic symptom bucket” if none provided):
  - Likely cause categories (generic categories only; do not claim app-specific causes)
  - Evidence to collect (what logs/config/runtime info; do not name exact log keys unless given)
  - Next tests (generic experiments: smaller runs, confirm reproducibility, check timestep sensitivity, etc.)

- Include a subsection:
  - “Conflicts / Ambiguities in application evidence”
  - list any contradictions among the provided inputs; otherwise state “No conflicts detected in provided inputs.”

5) Evidence Requirements (what to ask the user for)
Provide two tiers:

- Tier 1 (must-have to avoid guessing):
  - exact config snippet(s) relevant to the symptom
  - the full error log excerpt including first failure point
  - the runtime environment categories (CPU/GPU, parallel layout description) if mentioned; otherwise request it
  - the command/inputs used to reproduce

- Tier 2 (nice-to-have):
  - additional runs (same input, different parallel layouts) if the issue suggests nondeterminism
  - convergence/residual history screenshots or values (if they exist)
  - minimal case description (problem size/time-step category)

If inputs already contain some of these, note “Already provided: …” and list what remains missing.

6) Fallback Behavior (how to use Layer 1–2 when unknown)
Define deterministic fallback rules:
- If you cannot name any application config keys/workflow specifics:
  - instruct the assistant to rely on Layer 1 AMReX framework debugging concepts (geometry/BC/ghost/index/parallel categories)
  - rely on Layer 2 numerics/reproducibility/debug triage policies (determinism categories, floating-point tradeoffs, isolate components)
- If you have partial app hints from snippets:
  - use them only as hints, and still verify via logs/config evidence.
- Explicitly state how much weight to give:
  - “Treat application-specific hints as unverified unless confirmed by the exact provided snippets.”

NOW START.