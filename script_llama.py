
import re
import json
from typing import Optional, Any, Union, List
import glob
import os
import requests
from openai import OpenAI

from prompt_generation import build_fsm_extraction_prompt, build_fsm_combination_prompt

from groq import Groq

client = Groq(
    api_key="api-key",
)


# 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

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key='9cd8b4fe-21e6-48bc-aeed-e925275b83ee',
#     base_url="https://api.llmapi.com/"
# )


def call_llama(model_name, prompt, temperature=0.0, max_tokens=10000):
    """
    Calls the Hugging Face Transformers model and returns the generated text.
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract and return the generated text
    # print(response)
    return response.choices[0].message.content


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

        resp = call_llama(model, prompt, temperature=0.0, max_tokens=10000)

        #if verbose:
            #print("RAW RESPONSE:\n", resp)

        #fsm = parse_json_from_response(resp)

        #if verbose:
        #    print("PARSED FSM:\n", json.dumps(fsm, indent=2))

        # store per‐section details
        sections.append({
            "tag": tag,
            "prompt": prompt,
            "response": resp,
            #"partial_fsm": fsm
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

    final_resp = call_llama(model, combine_prompt, temperature=0.0, max_tokens=10000)
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
    
    protocols = ["DCCP","DHCP", "FTP", "IMAP", 
                 "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP",
                 "MQTT", "PPP", "PPTP", "BGP"]
    
    # model = "deepseek-r1:14b"
    # models = ["deepseek-r1:32b","qwen3:32b","gemma3:27b"]
    # models = ["mistral-small3.1"] # this is 24b
    
    # models = ["qwq","qwen3:32b","gemma3:27b","mistral-small3.1"] # 32b
    models = ["llama-3.3-70b-versatile"]
    for m in models:
        for d in protocols:
            process_protocol(d, m, verbose=True)
    
    
