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
├── $protocol_name$/                      # Folder with protocol name as folder name
│   ├── *_state_machine.json/             # ground-truth state machine
│   └── *_segments.json/                  # processed RFC chunks
│
├── fsm/                       # folder including extracted PSMs
│   ├── $protocol$_$model$_final_fsm.json     # LLM extracted PSM
│
├── draw_state_machine.py        # template to draw state machine figure
|
├── eval_fsm_sim.py      # evaluate PSM similarity between extracted PSM and ground truth
|   
├── output_parser_fsm.py  # extracte the PSM from LLM response 
|
├── prompt_generation.py       # util functions to generate the prompts
│
├── rfc_preprocess.py         # preprocess RFC file to segments
|
├── script_claude_api.py     # call Claude model api
|
├── script_deepseek_api.py     # call deepseek models api
|
├── script_gemini_api.py     # call gemini model api
|
├── script_gpt_api.py     # call gpt model api
|
├── script.py     # call ollama open-sources model
|
└── state_machine_format.json     # PSM format in json
```
