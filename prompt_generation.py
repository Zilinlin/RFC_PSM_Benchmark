from typing import Optional, Any, Union, List


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
- Maybe begin with a fixed prefix:
  - `"receive "` for received command
  - `"send "` for sent response
  - `"timeout "` for timing event
  - `"cond "` for internal condition
  - or other words 

Examples: `"receive USER"`, `"send 230"`, `"timeout 5s"`, `"cond valid_credentials"`

🔹 `"action"`:
- Describes what the system does in response
- It's best start with an action verb from this fixed set or other verbs if needed: `reply`, `send`, `set`, `log`, `reset`, 
`close`, 'collect', `open`, 'record', 'stop'
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
- Maybe begin with: `receive`, `send`, `timeout`, or `cond`
- Examples: `"receive USER"`, `"timeout 10s"`, `"cond valid_credentials"`

**Actions (`action`)**:
- Start with a verb from this list: `reply`, `send`, `set`, `log`, `reset`, 
`close`, 'collect', `open`, 'record', 'stop' or other verbs if needed
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

