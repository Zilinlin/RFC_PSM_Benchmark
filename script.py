'''
This file is to run the experiments for the RFC PSM Benchmark.
'''

import re
import json
from typing import Optional, Any, Union, List



def build_fsm_extraction_prompt(protocol_name: str,
                                section_title: str,
                                section_text: str) -> str:
    """
    Returns a complete prompt for extracting FSM components from one section.
    """
    template = """You will be given the section "{section_title}" of an RFC document for protocol "{protocol_name}".  
Please respond _only_ with JSON (or the word None) wrapped in <json>...</json>.

<section>
{section_text}
</section>

First, check whether this section contains any information related to a protocol state machine, such as:
- State definitions
- State transitions
- Diagrams or tables describing state flows

If no relevant information is found, simply return:
None

If the section contains partial information about the protocol state machine, extract and organize it using this JSON format:

<template>
{{
  "states": ["state1", "state2", "state3"],
  "transitions": [
    {{
      "from": "state1",
      "requisite": "conditionX",
      "to": "state2",
      "actions": ["action1"],
      "response": "response1"
    }},
    {{
      "from": "state2",
      "requisite": "conditionY",
      "to": "state3",
      "actions": ["action2"],
      "response": "response2"
    }}
  ]
}}
</template>

If some fields like "requisite", "actions", or "response" are not mentioned, set them to an empty string (`""`) or empty list (`[]`) as appropriate.
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
        raise ValueError(f"Invalid JSON extracted: {e}")


def build_fsm_combination_prompt(partial_fsms: List[Union[str, dict]]) -> str:
    """
    Given a list of partial FSMs (either JSON strings or dicts),
    returns a single prompt string ready to send to the LLM.
    """
    # Normalize each partial to a JSON string and wrap in <partial> tags
    fsm_blocks = []
    for p in partial_fsms:
        block = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, indent=2)
        fsm_blocks.append(f"<partial>\n{block}\n</partial>")

    partials_block = "\n\n".join(fsm_blocks)

    # Use an f-string and double all JSON braces to escape them!
    prompt = f"""
You will be provided with multiple **partial protocol state machines** extracted from different sections of an RFC. Each partial state machine is a JSON object with these fields:
- "states": list of state names  
- "transitions": list of transition objects, each with "from", "to", and optional "requisite", "actions", "response".

Your task is to **merge** them into one **global** state machine. Please respond _only_ with a single JSON object (no extra commentary) in the following format:

<json>
{{
  "states": ["state1", "state2", ...],
  "initial_state": "stateX",
  "final_states": ["stateY", ...],
  "transitions": [
    {{
      "from": "state1",
      "requisite": "conditionX",
      "to": "state2",
      "actions": ["action1"],
      "response": "response1"
    }}
  ]
}}
</json>

**Instructions**
1. **Combine** all unique states (treat synonyms like "Init" vs "Initialization State" as one).  
2. **Merge** transitions, dropping duplicates (even if phrased differently).  
3. **Standardize** state names for consistency.  
4. **Determine**  
   - "initial_state": the state with no incoming transitions  
   - "final_states": any state with no outgoing transitions  
5. **Fill** missing "requisite"/"actions"/"response" with "" or [] as appropriate.

Here are the partial machines (each wrapped in `<partial>...</partial>`):

{partials_block}

Please output **only** the final merged JSON in the `<json>…</json>` block.
"""
    return prompt


if __name__ == "__main__":
    
    
    examples = [
        "No JSON here.",
        "<json>None</json>",
        "<json>{\"states\":[]}</json>"
    ]
    for resp in examples:
        print("Response:", repr(resp))
        print("extract_json_content →", extract_json_content(resp))
        print("parse_json_from_response →", parse_json_from_response(resp))
        print("---")
        
    # → then send `prompt` to your LLM client
    partials = [
        {"states": ["Idle"], "transitions": [{"from":"Idle","requisite":"","to":"Init","actions":[],"response":""}]},
        {"states": ["Init","Connected"], "transitions":[{"from":"Init","requisite":"","to":"Connected","actions":["open"],"response":""}]}
    ]

    prompt = build_fsm_combination_prompt(partials)
    print(prompt)
        
