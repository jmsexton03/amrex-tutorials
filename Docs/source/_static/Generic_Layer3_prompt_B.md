LAYER 3 PROMPT (Layer 3: AMReX Application Profile)

You are generating a “Layer 3: AMReX Application Profile” report for a specific application. 

GOAL:
Create an application-specific “profile” that another prompt (Layer 0) can compile into final gem instructions. You must ONLY use information provided in the INPUTS block below. If information is missing, you must mark unknowns explicitly and define what evidence is needed.

=============================================================================
INPUTS (User provided):
- application_name: [INSERT APPLICATION NAME HERE, e.g., ERF or AMR-Wind]
- repo_url: [INSERT REPO URL HERE, or leave blank]
- user_provided_context_and_snippets: 
"""
[INSERT ANY CONFIG EXAMPLES, README EXCERPTS, OR CODY EMAIL REPORTS HERE. 
IF NONE, LEAVE BLANK.]
"""
=============================================================================

HARD CONSTRAINTS:
1) No invention / No hallucination:
   - NEVER invent application-specific parameter names, config keys, directory names, log identifiers, or workflow steps based on your pre-training data.
   - If a specific config key or workflow is not explicitly present in the `user_provided_context_and_snippets` above, you must write: “Unknown—require exact config/docs evidence.”
2) Evidence discipline:
   - Only promote statements to “Known” if grounded in the provided INPUTS. 
   - Even if you recognize the repo_url, do not pull in outside knowledge.
3) Minimal profile if needed:
   - If the user_provided_context is empty, produce a minimal report that clearly delegates back to Layer 1 (AMReX Core) and Layer 2 (Numerics/Debug).
4) No deep research:
   - Do not attempt to browse the repo_url to find new facts if you have a browsing tool. Rely strictly on the pasted text.

TASK:
Write a structured application profile report that includes the following sections in this exact order.

REQUIRED OUTPUT STRUCTURE (Markdown with exactly these headings):
## Profile Summary
## Known Configuration Keys / Parameters
## Known Workflows & Stages
## App-specific Debugging Triage
## Evidence Requirements
## Fallback Behavior

CONTENT REQUIREMENTS BY SECTION:

## Profile Summary
Include:
- Application identity: application_name and repo_url.
- What the assistant can confidently claim about this app based ONLY on the inputs.
- Where the profile is uncertain (e.g., “No config examples provided; cannot name exact parameters.”)

## Known Configuration Keys / Parameters
Create a list of any configuration keys/parameters that appear in the INPUTS.
For each key/parameter you list, include:
- Key name (verbatim from inputs)
- What it appears to control (only as described in inputs)
- What evidence to look for in logs
If no keys are present in the inputs, state: “No configuration keys were provided in inputs; unknown.”

## Known Workflows & Stages
Provide an ordered workflow reflecting typical run stages ONLY if mentioned in the inputs.
For each stage include:
- Stage name
- Outputs/evidence (what files/logs are mentioned)
If no workflow info is provided, provide a generic skeleton (e.g., build → configure → run) but label it explicitly as generic and not app-specific.

## App-specific Debugging Triage
Build a “symptom → likely cause categories → evidence to collect” map based ONLY on the inputs.
- Only include symptom patterns explicitly mentioned in the inputs. 
- “Likely cause categories” must be phrased generically unless inputs explicitly connect a symptom to a cause.
- Never claim exact causes without evidence.

## Evidence Requirements
List the minimum evidence the assistant should ask the user for when troubleshooting this app.
- Tier 1 (must-have): e.g., exact config file, full error log excerpt.
- Tier 2 (nice-to-have): input parameters for reproducibility, runtime environment info.
(Use generic categories for these requests unless the inputs give specific file names).

## Fallback Behavior
Define deterministic rules for falling back:
- Instruct the final assistant: "If you cannot name exact keys or steps for this application, fall back to Layer 1 framework concepts (ghost cells, geometry, parallel sync) and Layer 2 numerics policies."
- If Layer 3 provides partial hints, state: “Use these as hints, but verify using user logs/configs.”

NOW START.
```