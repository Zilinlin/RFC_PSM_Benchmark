# Protocol State Machine Extraction Benchmark

This repository contains the codebase for our NeurIPS 2025 submission, which benchmarks large language models (LLMs) on extracting protocol state machines (PSMs) from RFC documents. It includes dataset, protocol-specific processing scripts, extraction pipelines, evaluation metrics.

## Supported Protocols


| Protocol Name | RFC Document |
|---------------|--------------|
| RTSP (Real Time Streaming Protocol) | [RFC 7826](https://datatracker.ietf.org/doc/html/rfc7826) |
| FTP (File Transfer Protocol) | [RFC 959](https://datatracker.ietf.org/doc/html/rfc959) |
| SIP (Session Initiation Protocol) | [RFC 3261](https://datatracker.ietf.org/doc/html/rfc3261) |
| SMTP (Simple Mail Transfer Protocol) | [RFC 5321](https://datatracker.ietf.org/doc/html/rfc5321) |
| DCCP (Datagram Congestion Control Protocol) | [RFC 4340](https://datatracker.ietf.org/doc/html/rfc4340) |
| TCP (Transmission Control Protocol) | [RFC 9293](https://datatracker.ietf.org/doc/html/rfc9293) |
| DHCP (Dynamic Host Configuration Protocol for IPV4) | [RFC 2131](https://datatracker.ietf.org/doc/html/rfc2131) |
| IMAP (Internet Message Access Protocol) | [RFC 9051](https://datatracker.ietf.org/doc/html/rfc9051) |
| POP3 (Post Office Protocol v3) | [RFC 1939](https://www.ietf.org/rfc/rfc1939.txt) |
| NNTP (Network News Transfer Protocol) | [RFC 3977](https://datatracker.ietf.org/doc/html/rfc3977) |
| MQTT (Message Queuing Telemetry Transport) | [RFC 9431](https://datatracker.ietf.org/doc/html/rfc9431#name-reduced-protocol-interactio) |
| PPTP (Point-to-Point Tunneling Protocol) | [RFC 2637](https://datatracker.ietf.org/doc/html/rfc2637#autoid-1) |
| BGP (Border Gateway Protocol 4) | [RFC 4271](https://datatracker.ietf.org/doc/html/rfc4271) |
| PPP (Point-to-Point Protocol) | [RFC 1661](https://datatracker.ietf.org/doc/html/rfc1661#autoid-5) |


## Repository Structure

```
├── $protocol_name$/                        # One folder per protocol
│   ├── *_state_machine.json                # Ground-truth protocol state machine (PSM)
│   └── *_segments.json                     # Preprocessed RFC segments used for LLM input
│
├── fsm/                                    # Extracted protocol state machines (PSMs)
│   ├── $protocol$_$model$_final_fsm.json   # Final FSM output for each model and protocol
│
├── draw_state_machine.py                   # Script to render FSM figures from JSON
│
├── eval_fsm_sim.py                         # Computes similarity between extracted and ground-truth PSMs
│
├── output_parser_fsm.py                    # Parses LLM output into a clean FSM structure
│
├── prompt_generation.py                    # Utility functions to generate model prompts from RFC segments
│
├── rfc_preprocess.py                       # Script to chunk and clean RFC documents into segments
│
├── script_claude_api.py                    # Interface to call Claude model API
├── script_deepseek_api.py                  # Interface to call DeepSeek model API
├── script_gemini_api.py                    # Interface to call Gemini model API
├── script_gpt_api.py                       # Interface to call OpenAI GPT model API
├── script.py                               # Calls Ollama-compatible open-source models
│
└── state_machine_format.json               # JSON schema used for protocol state machines
```


## How to Run

### i. RFC Preprocessing

To preprocess RFC documents for a specific protocol, use the `rfc_preprocess.py` script. This script:

1. Cleans the original RFC raw text (`*_raw.txt`)
2. Removes table-of-content (TOC) lines
3. Extracts level-two sections
4. Splits the document into token-limited segments
5. Saves the final segments to a JSON file (`*_segments.json`)

#### 📦 Input File Format
Place the RFC raw file in a folder named the protocol. The file should be named as:
```
<something>_raw.txt
```
 in the protoocl file
for example: 
```
PPP/
├── rfc1661_raw.txt        # Input raw RFC file
```
To run the preprocessing for a single protocol (e.g., `PPP`), execute:
```bash
python3 rfc_preprocess.py FTP
```

This will generate:
- `FTP/<name>_cleaned.txt` – cleaned version of the raw RFC
- `FTP/<name>_no_toc.txt` – cleaned text with TOC lines removed
- `FTP/<name>_segments.json` – final segmented output for model input

---
### ii. Running LLMs on RFC Segments

After generating segmented RFC files (`*_segments.json`), you can extract protocol state machines using various LLMs. Each model has a dedicated script that processes a list of protocols using a shared function.


#### ▶️ Run LLM Extraction

Execute one of the following scripts depending on the model you want to use, before each use, please replace the api_key with your `api_key`.

```bash
python3 script_claude_api.py      # Claude 3 Sonnet
python3 script_deepseek_api.py   # DeepSeek Chat & Reasoner
python3 script_gemini_api.py     # Gemini 2.0 Flash
python3 script_gpt_api.py        # GPT-4o Mini (OpenAI)
python3 script.py                # Ollama-compatible open-source models
```


#### ⚙️ What Happens Internally

Each script:
- Loads the protocol's `*_segments.json` file
- Builds prompts per section and queries the model
- Collects section-level responses
- Merges them into a final FSM using a combination prompt
- Saves everything to:  
  ```
  output/<protocol>_<model>_output.json
  ```

Each output file includes:
- `protocol`: protocol name  
- `model`: model identifier  
- `sections`: per-section prompts & raw responses  
- `final_response`: merged full response  
- `final_fsm`: parsed state machine in JSON format  


#### 📄 Supported Scripts & Models

| Script                 | Models Used                                      |
|------------------------|--------------------------------------------------|
| `script_claude_api.py` | `claude-3-7-sonnet-20250219`                     |
| `script_deepseek_api.py` | `deepseek-chat`, `deepseek-reasoner`          |
| `script_gemini_api.py` | `gemini-2.0-flash`                               |
| `script_gpt_api.py`    | `gpt-4o-mini`                                    |
| `script.py`            | `mistral-small3.1`, `qwq`, `gemma3:27b`, `qwen3:32b` |


> ✅ The segmented files (e.g., `FTP/ftp_segments.json`) already exist in differnt protocol file.

---

### iii. Extracting Final FSMs from LLM Responses

After generating raw output files using LLM scripts (e.g., `output/FTP_gpt-4o-mini_output.json`), you can extract or regenerate the final protocol state machine (FSM) using the provided script.



#### 📄 Script: `output_parser_fsm.py`

This script loads `*_output.json` files produced by LLM calls and does the following:

- ✅ If a `final_fsm` is already present in the output file, it writes it to `fsm/<protocol>_<model>_final_fsm.json`.
- 🔁 If not, it rebuilds the final FSM by combining section-level responses using an LLM (via `gpt-4o-mini`) and saves the result.
- 🧠 All FSMs are saved in the `fsm/` folder in standardized JSON format.


#### ▶️ How to Run

You can run the script as-is to batch process multiple protocols and models:

```bash
python3 output_parser_fsm.py
```

This will:
- Read from `output/`
- Write final FSMs to `fsm/`
- Attempt to process all combinations of:
  - Models:  
    `deepseek-reasoner`, `deepseek-chat`, `gpt-4o-mini`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash`,  
    `mistral-small3.1`, `qwen3:32b`, `gemma3:27b`, `qwq`
  - Protocols:  
    `DCCP`, `DHCP`, `FTP`, `IMAP`, `NNTP`, `POP3`, `RTSP`, `SIP`, `SMTP`, `TCP`, `MQTT`, `PPP`, `PPTP`, `BGP`

You can modify the `protocols` and `models` lists at the bottom of the script to run on a subset.

#### 📂 Output

Each successfully extracted FSM is saved as:
```
fsm/<protocol>_<model>_final_fsm.json
```

Example:
```
fsm/FTP_gpt-4o-mini_final_fsm.json
```
The current `fsm` folder already includes the extracted state machines for testing.

---

### iv FSM Similarity Evaluation

This script evaluates the similarity between **extracted FSMs** (from LLMs) and **ground-truth FSMs**, using **fuzzy semantic matching** for both states and transitions. It provides detailed per-protocol metrics and generates tables for reporting.


Script: `eval_fsm_sim.py`


#### ▶️ How to Run

Run the script directly:
```bash
python3 eval_fsm_sim.py
```


#### 🧾 Requirements
- Extracted FSMs must be saved in: `fsm/<protocol>_<model>_final_fsm.json`
- Ground-truth FSMs must be located at: `<protocol>/*_state_machine.json`


#### ✅ Outputs

- 📊 Console output of per-protocol state and transition metrics
- 📁 `transitions_match_results_whole.csv`: summary of transition matching


### 📁 Output Format Sample

| Protocol | Model       | TotalExtracted | TotalGT | Matched | Precision | Recall | F1-Score |
|----------|-------------|----------------|---------|---------|-----------|--------|----------|
| FTP      | Gpt4o-Mini  | 8              | 10      | 7       | 0.875     | 0.700  | 0.778    |

---

