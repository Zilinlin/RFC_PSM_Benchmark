'''
This code is to parse the output to get the FSM from the response of LLM.
'''

import os
import json
from typing import Dict, List, Union, Optional,Any
import re

from openai import OpenAI
from prompt_generation import build_fsm_combination_prompt
client = OpenAI(
  api_key="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"
)


# def build_fsm_combination_prompt(partial_fsms: List[Union[str, dict]]) -> str:
#     """
#     Given a list of partial FSMs (JSON strings or dicts), returns a prompt
#     instructing a GPT model to merge them into one global FSM.
#     """
#     # Normalize each partial to a JSON string and wrap in <partial> tags
#     blocks = []
#     for p in partial_fsms:
#         text = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, indent=2)
#         blocks.append(f"<partial>\n{text}\n</partial>")
#     partials_block = "\n\n".join(blocks)

#     prompt = f"""
# You have multiple partial protocol state machines resposne extracted from different sections of an RFC. Each one is wrapped in `<partial>...</partial>`.

# Your task: **merge** them into one **global** state machine. Follow these rules exactly:

# 1. **states**: list all unique state names.
# 2. **initial_state**: the state with no incoming transitions.
# 3. **final_states**: any state with no outgoing transitions.
# 4. **transitions**: list all transitions, removing duplicates.  
#    Each transition must be an object with keys:
#    - `"from"` (string)
#    - `"to"` (string)
#    - `"requisite"` (string; `""` if none)
#    - `"actions"` (list of strings; `[]` if none)
#    - `"response"` (string; `""` if none)
# 5. Standardize synonyms (e.g. `"Init"` vs `"Initialization"`).
# 6. Do **not** include any explanations, markdown, or code fences—output **only** the JSON object.

# Output **exactly** in this format:

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

# Here are the partial machines:

# {partials_block}
# """.strip()

#     return prompt



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
    
    

def call_api(prompt, temperature=0.0, max_tokens=8192):
    """
    Calls the chatgpt API and returns the generated text.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def extract_final_fsm(directory: str, model: str, protocol: str, output_dir: str) -> dict:
    """
    Locate and load the JSON output file for the given protocol & model.
    If 'final_fsm' is present and non-null, return it;
    otherwise regenerate via o4-mini using partial section responses.

    In all cases, write the final FSM to a JSON file in output_dir named
    "{protocol}_{model}_final_fsm.json".
    """
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        if model in fname and protocol in fname:
            fullpath = os.path.join(directory, fname)
            with open(fullpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            final = data.get('final_fsm')
            # if final_fsm already present
            if final:
                out_file = os.path.join(output_dir, f"{protocol}_{model}_final_fsm.json")
                with open(out_file, 'w', encoding='utf-8') as fo:
                    json.dump(final, fo, indent=2)
                return final

            print(f"Regenerating FSM for {protocol} with model {model}...")
            # Otherwise regenerate from partial responses
            sections = data.get('sections', [])
            responses = [sec.get('response', '') for sec in sections]
            prompt = build_fsm_combination_prompt(responses)
            merged = call_api(prompt)
            new_fsm = parse_json_from_response(merged)

            if new_fsm:
                # write regenerated FSM to output_dir
                out_file = os.path.join(output_dir, f"{protocol}_{model}_final_fsm.json")
                with open(out_file, 'w', encoding='utf-8') as fo:
                    json.dump(new_fsm, fo, indent=2)
                # update original file
                data['final_fsm'] = new_fsm
                with open(fullpath, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, indent=2)
                return new_fsm

    raise FileNotFoundError(
        f"No JSON file in {directory!r} matching model={model!r} and protocol={protocol!r}"
    )



if __name__ == "__main__":
    protocols = ["DCCP","DHCP", "FTP", "IMAP", 
                  "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP",
                  "MQTT", 'PPP', "PPTP", "BGP"]
    # protocols = ["MQTT", 'PPP', "PPTP", "BGP"]
    #protocols = ["SIP", "NNTP", "FTP"]
    # close_models = ["deepseek-reasoner","gpt-4o-mini", "claude-3-7-sonnet-20250219","gemini-2.0-flash"]
    close_models = ["deepseek-chat"]
    directory = "output"
    fsm_dir = "fsm"
    
    for model in close_models:
        for protocol in protocols:
            try:
                final_fsm = extract_final_fsm(directory, model, protocol, fsm_dir)
                print(f"Final FSM for {protocol} with model {model} extracted successfully.")
            except FileNotFoundError as e:
                print(e)