'''
This file is to run the experiments for the RFC PSM Benchmark.
'''

import re
import json
from typing import Optional, Any, Union, List
import glob
import os
import requests

import time
import anthropic

client = anthropic.Anthropic()

# claude key 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

# def build_fsm_extraction_prompt(protocol_name: str,
#                                 section_title: str,
#                                 section_text: str) -> str:
#     """
#     Returns a complete prompt for extracting FSM components from one section.
#     """
#     template = """
# You will be given the section "{section_title}" of an RFC document for protocol "{protocol_name}".

# **RESPONSE FORMAT (MANDATORY)**
# - Your reply must consist **exclusively** of the JSON object representing the state machine.
# - That JSON must be wrapped in <json> and </json> tags.
# - Do **not** include any extra text, explanation, code fences, or formatting.

# <section>
# {section_text}
# </section>

# Steps:
# 1. Determine if this section has any FSM information (states, transitions, diagrams).
# 2. If **none**, reply exactly:
#    <json>None</json>
# 3. Otherwise, extract the partial FSM and format it **exactly** as:

# {{
#   "states": ["state1", "state2", "state3"],
#   "transitions": [
#     {{
#       "from": "state1",
#       "requisite": "conditionX",
#       "to": "state2",
#       "actions": ["action1"],
#       "response": "response1"
#     }},
#     {{
#       "from": "state2",
#       "requisite": "conditionY",
#       "to": "state3",
#       "actions": ["action2"],
#       "response": "response2"
#     }}
#   ]
# }}

# and then wrap that in <json>…</json> with nothing else.

# Remember: **ONLY** the <json>…</json> block should appear in your final output.
# """
#     return template.format(
#         protocol_name=protocol_name,
#         section_title=section_title,
#         section_text=section_text
#     )


# This is updated prompt version to add more constraints
def build_fsm_extraction_prompt(protocol_name: str,
                                section_title: str,
                                section_text: str) -> str:
    """
    Returns a complete prompt for extracting FSM components from one section,
    with strict formatting and naming guidelines for consistency.
    """
    template = """
You will be given the section "{section_title}" of an RFC document for protocol "{protocol_name}".

**RESPONSE FORMAT (MANDATORY)**
- Your reply must consist **exclusively** of the JSON object representing the state machine.
- That JSON must be wrapped in <json> and </json> tags.
- Do **not** include any extra text, explanation, code fences, or formatting.

<section>
{section_text}
</section>

Steps:
1. Determine if this section has any FSM-related information (states, transitions, diagrams, reply codes, sequences).
2. If **none**, reply exactly:
   <json>None</json>
3. If there is FSM information, extract it and return a structured JSON in the following format (strictly):

{{
  "states": ["state1", "state2", "state3"],
  "transitions": [
    {{
      "from": "state1",
      "event": "recv COMMAND",
      "action": "reply CODE",
      "to": "state2"
    }},
    ...
  ]
}}

**FSM Field Constraints:**

🔹 `"states"`:
- List of all states appearing in `"from"` or `"to"` fields.
- Each state must:
  - Be 1 to 3 words (max 30 characters)
  - Use `CamelCase` or `snake_case`
  - Describe a protocol phase, status, or role (e.g., `Authenticated`, `WaitingForReply`)
  - Contain no punctuation, spaces, or free-form descriptions

Good: `"AwaitingPassword"`, `"transfer_in_progress"`  
Bad: `"State 1"`, `"waiting for command"`, `"cmd?"`

🔹 `"from"` / `"to"`:
- Same naming rules as above

🔹 `"event"`:
- Describes the trigger that causes the transition
- Must begin with a fixed prefix:
  - `"recv "` for received command
  - `"send "` for sent response
  - `"timeout "` for timing event
  - `"cond "` for internal condition

Examples: `"recv USER"`, `"send 230"`, `"timeout 5s"`, `"cond valid_credentials"`

🔹 `"action"`:
- Describes what the system does in response
- Must start with an action verb from this fixed set: `reply`, `send`, `set`, `log`, `reset`, `transition`, `close`
- Followed by one or two short arguments (max 4 words total)

Examples: `"reply 230"`, `"log failure"`, `"set authenticated true"`

---

Important:
- Do not generate free-text descriptions in any field.
- Each transition must contain **exactly**: `from`, `event`, `action`, `to`.
- Do not invent vague or inconsistent state or event names.

---

**OUTPUT RULES:**
- Wrap the JSON in **<json>…</json>** only.
- Do not include Markdown, explanations, comments, or extra text.
- If nothing is found, return exactly `<json>None</json>`.

"""
    return template.format(
        protocol_name=protocol_name,
        section_title=section_title,
        section_text=section_text
    )
    

def extract_json_content(response: str) -> Optional[str]:
    """
    Extracts the raw JSON string wrapped inside <json>...</json> in `response`.
    Returns the inner string, or None if:
      - no such block is found
      - the block is empty/whitespace
      - the block literally contains "None" (case-insensitive)
    """
    pattern = re.compile(r'<json>(.*?)</json>', re.DOTALL | re.IGNORECASE)
    m = pattern.search(response)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner or inner.lower() == "none":
        return None
    return inner


def parse_json_from_response(response: str) -> Union[Any, None]:
    """
    Extracts the JSON block and parses it into a Python object.
    Returns:
      - the parsed object, or
      - None if extraction yields None (no block, empty, or "None")
    Raises:
      - ValueError if the extracted text is non-empty/non-"None" but invalid JSON.
    """
    json_text = extract_json_content(response)
    if json_text is None:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        return json_text  # Return the raw text if JSON parsing fails



# def build_fsm_combination_prompt(partial_fsms: List[Union[str, dict]]) -> str:
#     """
#     Given a list of partial FSMs (either JSON strings or dicts),
#     returns a single prompt string ready to send to the LLM.
#     """
#     # Normalize each partial to a JSON string and wrap in <partial> tags
#     fsm_blocks = []
#     for p in partial_fsms:
#         block = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, indent=2)
#         fsm_blocks.append(f"<partial>\n{block}\n</partial>")

#     partials_block = "\n\n".join(fsm_blocks)

#     # Use an f-string and double all JSON braces to escape them!
#     prompt = f"""
# You will be provided with multiple **partial protocol state machines** extracted from different sections of an RFC. Each partial state machine is a JSON object with these fields:
# - "states": list of state names  
# - "transitions": list of transition objects, each with "from", "to", and optional "requisite", "actions", "response".

# Your task is to **merge** them into one **global** state machine. Please respond _only_ with a single JSON object (no extra commentary) in the following format:

# <json>
# {{
#   "states": ["state1", "state2", ...],
#   "initial_state": "stateX",
#   "final_states": ["stateY", ...],
#   "transitions": [
#     {{
#       "from": "state1",
#       "requisite": "conditionX",
#       "to": "state2",
#       "actions": ["action1"],
#       "response": "response1"
#     }}
#   ]
# }}
# </json>

# **Instructions**
# 1. **Combine** all unique states (treat synonyms like "Init" vs "Initialization State" as one).  
# 2. **Merge** transitions, dropping duplicates (even if phrased differently).  
# 3. **Standardize** state names for consistency.  
# 4. **Determine**  
#    - "initial_state": the state with no incoming transitions  
#    - "final_states": any state with no outgoing transitions  
# 5. **Fill** missing "requisite"/"actions"/"response" with "" or [] as appropriate.

# Here are the partial machines (each wrapped in `<partial>...</partial>`):

# {partials_block}

# Please output **only** the final merged JSON in the `<json>…</json>` block.
# """
#     return prompt



def build_fsm_combination_prompt(partial_fsms: List[Union[str, dict]]) -> str:
    """
    Given a list of partial FSMs (either JSON strings or dicts),
    returns a single prompt string ready to send to the LLM,
    with enforced field structure, naming rules, and formatting.
    """
    fsm_blocks = []
    for p in partial_fsms:
        block = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, indent=2)
        fsm_blocks.append(f"<partial>\n{block}\n</partial>")

    partials_block = "\n\n".join(fsm_blocks)

    prompt = f"""
You will be provided with multiple **partial protocol state machines** extracted from different sections of an RFC. Each partial state machine is a JSON object with the following fields:

- "states": list of state names  
- "transitions": list of transition objects with these required fields:
  - "from": source state name  
  - "event": trigger (e.g., received command, condition)  
  - "action": response or internal action  
  - "to": target state name  

Each partial is wrapped in `<partial>...</partial>`. Some may be `<json>None</json>` — ignore those.

---

Your task is to **merge all valid partial FSMs into one global FSM** and return a single well-structured JSON object in the following format (wrapped in `<json>...</json>`):

<json>
{{
  "states": ["state1", "state2", ...],
  "initial_state": "stateX",
  "final_states": ["stateY", ...],
  "transitions": [
    {{
      "from": "state1",
      "event": "recv COMMAND",
      "action": "reply CODE",
      "to": "state2"
    }}
  ]
}}
</json>

---

### FSM Construction Constraints

**State Naming (`states`, `from`, `to`)**:
- Must be concise, meaningful, and consistent
- Format: 1 to 3 words, `CamelCase` or `snake_case`, no spaces or punctuation
- Examples: `"Authenticated"`, `"AwaitingPassword"`, `"TransferReady"`

**Events (`event`)**:
- Format: 1 to 3 words
- Must begin with: `recv`, `send`, `timeout`, or `cond`
- Examples: `"recv USER"`, `"timeout 10s"`, `"cond valid_credentials"`

**Actions (`action`)**:
- Start with a verb from this list: `reply`, `send`, `log`, `set`, `close`, `open`, `transition`
- Followed by a short phrase (≤ 4 words)
- Examples: `"reply 230"`, `"set authenticated true"`, `"log failure"`

---

### FSM Merging Rules

1. **Unify states**: Standardize naming (e.g., merge `"Init"` and `"Initialization"` into one state).
2. **Remove duplicates**: Transitions that differ only in phrasing should be merged.
3. **Preserve meaning**: If two similar states clearly serve different roles, retain both.
4. **Determine**:
   - `"initial_state"`: The state with **no incoming transitions**
   - `"final_states"`: All states with **no outgoing transitions**

---

Please return **only** the merged FSM in the required format, wrapped inside `<json>...</json>`. Do **not** include explanations, commentary, or Markdown.

Here are the partial FSMs to merge:

{partials_block}

"""
    return prompt


# def call_ollama(model, prompt, temperature=0.0, max_tokens=10000):
#     """
#     Calls the local Ollama HTTP API and returns the generated text.
#     """
#     url = "http://localhost:11434/v1/completions"
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "temperature": temperature,
#         "max_tokens": max_tokens
#     }
#     resp = requests.post(url, json=payload)
#     resp.raise_for_status()  # will raise an HTTPError if the call failed
#     data = resp.json()
#     # Ollama uses the OpenAI-compatible response format:
#     # { "choices": [ { "text": "..." } ], ... }
#     return data["choices"][0]["text"]

def call_api(model, prompt, temperature=0.0, max_tokens=8192):
    """
    Calls the chatgpt API and returns the generated text.
    """
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You are a world-class poet. Respond only with short poems.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # message.content is a list of TextBlock objects
    blocks = message.content  
    poem = blocks[0].text
    return poem


# ── Workflow ──────────────────────────────────────────────────────────────────

def process_protocol(protocol_dir, model, verbose=True):
    name = os.path.basename(protocol_dir.rstrip("/"))
    seg_files = glob.glob(os.path.join(protocol_dir, "*_segments.json"))
    if not seg_files:
        raise FileNotFoundError(f"No segments file in {protocol_dir}")
    segments = json.load(open(seg_files[0]))

    partials = []
    sections = []

    # make sure output directory exists
    os.makedirs("output", exist_ok=True)

    for idx, seg in enumerate(segments, start=1):
        tag, content = seg["tag"], seg["content"]
        prompt = build_fsm_extraction_prompt(
            protocol_name=name,
            section_title=tag,
            section_text=content
        )

        if verbose:
            print(f"\n--- Section {idx}/{len(segments)}: {tag} ---")
            #print("PROMPT:\n")

        resp = call_api(model, prompt, temperature=0.0, max_tokens=8192)
        time.sleep(60) # to bypass the claude rate limit
        #if verbose:
            #print("RAW RESPONSE:\n", resp)

        # fsm = parse_json_from_response(resp)

        # if verbose:
        #     print("PARSED FSM:\n", json.dumps(fsm, indent=2))

        # store per‐section details
        sections.append({
            "tag": tag,
            "prompt": prompt,
            "response": resp
            # "partial_fsm": fsm
        })
        partials.append(resp)

        # # optional: write each partial to its own file
        # partial_file = f"output/{name}_{model}_partial_{idx}.json"
        # with open(partial_file, "w") as pf:
        #     json.dump(fsm, pf, indent=2)
        # if verbose:
        #     print(f"→ saved partial FSM to {partial_file}")

    # combine step
    combine_prompt = build_fsm_combination_prompt(partial_fsms=partials)

    final_resp = call_api(model, combine_prompt, temperature=0.0, max_tokens=8192)
    if verbose:
        print("COMBINED RAW RESPONSE:\n", final_resp)

    final_fsm = parse_json_from_response(final_resp)
    if verbose:
        print("FINAL MERGED FSM:\n", json.dumps(final_fsm, indent=2))

    # write full output
    out = {
        "protocol": name,
        "model": model,
        "sections": sections,
        "final_response": final_resp,
        "final_fsm": final_fsm
    }
    outfile = f"output/{name}_{model}_output.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(f"\n► saved full output to {outfile}")

    return outfile


if __name__ == "__main__":
    
    
    # examples = [
    #     "No JSON here.",
    #     "<json>None</json>",
    #     "<json>{\"states\":[]}</json>"
    # ]
    # for resp in examples:
    #     print("Response:", repr(resp))
    #     print("extract_json_content →", extract_json_content(resp))
    #     print("parse_json_from_response →", parse_json_from_response(resp))
    #     print("---")
        
    # # → then send `prompt` to your LLM client
    # partials = [
    #     {"states": ["Idle"], "transitions": [{"from":"Idle","requisite":"","to":"Init","actions":[],"response":""}]},
    #     {"states": ["Init","Connected"], "transitions":[{"from":"Init","requisite":"","to":"Connected","actions":["open"],"response":""}]}
    # ]

    # prompt = build_fsm_combination_prompt(partials)
    # print(prompt)
    # "DCCP","DHCP", "FTP","IMAP",
    #             "NNTP", "POP3",
    protocols = [  "RTSP", "SIP", "SMTP", "TCP"]
    protocols = ["DCCP"]
    # model = "deepseek-r1:14b"
    # models = ["deepseek-r1:32b","qwen3:32b","gemma3:27b"]
    # models = ["mistral-small3.1"] # this is 24b
    # models = ["qwq"] # 32b
    
    # models = ["deepseek-reasoner"]
    # models = ["gpt-4o-mini"]
    models = ["claude-3-7-sonnet-20250219"]
    for m in models:
        for d in protocols:
            process_protocol(d, m, verbose=True)
    
        
