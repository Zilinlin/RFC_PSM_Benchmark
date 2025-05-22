'''This file is to evaluate the FSM similarity between two FSMs using fuzzy matching on both transitions and states.'''

import os
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


model = SentenceTransformer('all-MiniLM-L6-v2')

model_name_mapping = {
    "deepseek-reasoner": "DS-R1",
    "gpt-4o-mini": "Gpt4o-Mini",
    "claude-3-7-sonnet-20250219": "Claude3",
    "gemini-2.0-flash": "Gemini2",
    "deepseek-chat": "DS-V3",
    "qwq": "QWQ",
    "qwen3:32b": "QWen3",
    "gemma3:27b": "Gemma3",
    "mistral-small3.1": "Mistral"
}

def preprocess_state_name(state_name):
    # Convert to lowercase, remove punctuation, and split compound words
    state_name = state_name.lower()
    state_name = re.sub(r"[\W_]+", " ", state_name)  # Replace non-alphanumerics with space
    state_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", state_name)  # Split camel case
    state_name = state_name.strip()
    return state_name



def match_states(source_states, target_states, threshold):
    # Preprocess the state names
    source_clean = [preprocess_state_name(s) for s in source_states]
    target_clean = [preprocess_state_name(t) for t in target_states]

    source_embeddings = model.encode(source_clean, normalize_embeddings=True)
    target_embeddings = model.encode(target_clean, normalize_embeddings=True)

    # Compute pairwise cosine similarities
    similarity_matrix = util.pytorch_cos_sim(source_embeddings, target_embeddings).numpy()

    # Find the best matches
    matches = []
    for i, src in enumerate(source_states):
        best_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_idx]
        best_match = target_states[best_idx]
        if best_score >= threshold:
            matches.append((src, best_match, best_score))
        else:
            # Optionally include unmatched states
            matches.append((src, None, best_score))

    # Optionally print unmatched states
    unmatched = [t for i, t in enumerate(target_states) if i not in np.argmax(similarity_matrix, axis=0)]
    return matches, unmatched

# this function is to match all states between the ground truth and extracted FSMs
def match_all_states(models, protocols, fsm_dir, threshold=0.5):
    all_matches = {}
    summary_stats = []

    for protocol in protocols:
        # Load ground truth FSM
        protocol_dir = os.path.join(protocol)
        gt_file = None
        for file in os.listdir(protocol_dir):
            if file.endswith("_state_machine.json"):
                gt_file = os.path.join(protocol_dir, file)
                break

        if not gt_file:
            continue

        with open(gt_file, 'r') as f:
            gt_fsm = json.load(f)
        gt_states = gt_fsm.get('states', [])

        for model in models:
            # Load extracted FSM
            extracted_file = None

            # Find the extracted FSM file for the protocol and model
            for file in os.listdir(fsm_dir):
                if protocol in file and model in file and file.endswith(".json"):
                    extracted_file = os.path.join(fsm_dir, file)
                    break

            if not extracted_file:
                continue

            with open(extracted_file, 'r') as f:
                ext_fsm = json.load(f)
            ext_states = ext_fsm.get('states', [])

            # Match states
            matches, unmatched = match_states(gt_states, ext_states, threshold=threshold)
            matched_count = len([m for m in matches if m[1] is not None])
            unmatched_gt = len(gt_states) - matched_count
            unmatched_extracted = len(ext_states) - matched_count
            precision = round(matched_count / len(ext_states), 3) if len(ext_states) > 0 else 0
            recall = round(matched_count / len(gt_states), 3) if len(gt_states) > 0 else 0
            f1_score = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0

            # Store results
            all_matches[(protocol, model)] = {
                "matches": matches,
                "unmatched_gt": unmatched_gt,
                "unmatched_extracted": unmatched_extracted,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "total_gt_states": len(gt_states),
                "total_extracted_states": len(ext_states),
                "matched": matched_count
            }

            # Append summary stats for later table generation
            summary_stats.append([protocol, model, len(ext_states), len(gt_states), matched_count, precision, recall, f1_score])

    # Return both detailed matches and summary stats for table generation
    return all_matches, pd.DataFrame(summary_stats, columns=["Protocol", "Model", "Total Extracted", "Total GT", "Matched","Precision", "Recall", "F1-Score"])

def compute_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return sim



def match_transitions_combined_event_action(trans1, trans2, threshold=0.5,
                                            if_partial=True):
    """
    Compare transitions from two FSMs using a precomputed state_match_map for 'from' and 'to' states,
    and a single combined similarity score for event+action.
    
    Args:
      trans1: list of transition dicts from extracted FSM
      trans2: list of transition dicts from ground truth FSM
      state_match_map: dict mapping extracted state name -> matched GT state name
      threshold: float threshold for combined event/action similarity
    
    Returns:
      matched: list of (t1, t2) pairs that fully match
      unmatched: list of t1 that did not match
      fully_correct: count of fully correct transitions
    """
    matched = []
    unmatched = []
    fully_correct = 0
    trans2_copy = trans2.copy()
    
    for t1 in trans1:
        found = False
        for t2 in list(trans2_copy):
            # 1) State match via precomputed map
            # from_match = state_match_map.get(t1["from"]) == t2["from"]
            # to_match   = state_match_map.get(t1["to"])   == t2["to"]
            from_score = compute_similarity(t1.get("from", ""), t2.get("from", ""))
            to_score   = compute_similarity(t1.get("to", ""), t2.get("to", ""))
            from_match = from_score >= threshold
            to_match   = to_score >= threshold
            
            # 2) Compute combined event+action similarity
            # ev_score     = compute_similarity(t1.get("event", ""),  t2.get("event", ""))
            # action_score = compute_similarity(t1.get("action", ""), t2.get("action", ""))
            # combined_score = (ev_score + action_score) / 2
            if if_partial:
                ev_score     = compute_similarity(t1.get("event", ""),  t2.get("event", ""))
                action_score = compute_similarity(t1.get("action", ""), t2.get("action", ""))
                combined_score = max(ev_score,action_score)
            else:
                label1 = t1.get("event", "") + t1.get("action", "")
                label2 = t2.get("event", "") + t2.get("action", "")
                combined_score = compute_similarity(label1, label2)
            
            # 3) Full match if both states match and combined >= threshold
            if from_match and to_match and combined_score >= threshold:
                matched.append((t1, t2))
                fully_correct += 1
                trans2_copy.remove(t2)
                found = True
                break
        
        if not found:
            unmatched.append(t1)
    
    return matched, unmatched, fully_correct



def compute_match_metrics(matched, gt_trans, ex_trans):
    tp = len(matched)
    fn = len(gt_trans) - tp
    fp = len(ex_trans) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


# def evaluate_fsm_similarity(fsm1_json, fsm2_json, allow_partial=True, threshold=0.5):
#     matched, unmatched, fully_correct, partially_correct = match_transitions_fuzzy_states(
#         fsm1_json["transitions"], fsm2_json["transitions"], threshold=threshold, allow_partial=allow_partial)
#     precision, recall, f1 = compute_match_metrics(matched, fsm1_json["transitions"], fsm2_json["transitions"])
#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#         "matched": len(matched),
#         "fully_correct": fully_correct,
#         "partially_correct": partially_correct,
#         "unmatched": len(unmatched),
#     }


def batch_evaluate_transitions_combined(
    protocols, models, fsm_dir, if_partial=True, state_threshold=0.5, trans_threshold=0.5
):
    """
    For each (protocol, model) pair:
      1. Load ground-truth FSM and extracted FSM.
      2. match_states() to get state_match_map.
      3. match_transitions_combined_event_action() for transitions.
      4. Compute metrics and collect into a DataFrame.
    """
    rows = []
    for protocol in protocols:
        print("current protocol:", protocol)
        # 1) Locate GT FSM
        gt_path = None
        for fn in os.listdir(protocol):
            if fn.endswith("_state_machine.json"):
                gt_path = os.path.join(protocol, fn)
                break
        if not gt_path:
            continue
        with open(gt_path) as f:
            gt_fsm = json.load(f)
        gt_states      = gt_fsm.get("states", [])
        gt_transitions = gt_fsm.get("transitions", [])

        for model in models:
            print("current model:", model)
            # 2) Locate extracted FSM
            ext_path = None
            for fn in os.listdir(fsm_dir):
                if protocol in fn and model in fn and fn.endswith(".json"):
                    ext_path = os.path.join(fsm_dir, fn)
                    break
            if not ext_path:
                continue
            with open(ext_path) as f:
                ext_fsm = json.load(f)
            ext_states      = ext_fsm.get("states", [])
            ext_transitions = ext_fsm.get("transitions", [])

            # 3) Build state_match_map via your existing match_states()
            state_matches, _ = match_states(ext_states, gt_states, threshold=state_threshold)
            state_match_map = {src: tgt for src, tgt, score in state_matches if tgt is not None}

            # 4) Match transitions with combined event+action similarity
            matched, unmatched_ext, full_cnt = match_transitions_combined_event_action(
                ext_transitions,
                gt_transitions,
                # state_match_map,
                threshold=trans_threshold,
                if_partial=if_partial
            )

            # 5) Compute metrics
            total_ext = len(ext_transitions)
            total_gt  = len(gt_transitions)
            matched_count = full_cnt
            unmatched_gt   = total_gt - matched_count
            unmatched_ex   = total_ext - matched_count
            precision = matched_count / total_ext if total_ext > 0 else 0
            recall    = matched_count / total_gt  if total_gt  > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision+recall)>0 else 0

            rows.append({
                "Protocol": protocol,
                "Model": model,
                "TotalExtracted": total_ext,
                "TotalGT": total_gt,
                "Matched": matched_count,
                "UnmatchedGT": unmatched_gt,
                "UnmatchedExtracted": unmatched_ex,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1-Score": round(f1, 3)
            })

    return pd.DataFrame(rows)



if __name__ == "__main__":
    #batch_evaluate_fsm_similarity()
    protocols = ["IMAP", "POP3", "MQTT","PPP","PPTP", "BGP",
                 "SIP", "RTSP", "DCCP", "DHCP", "FTP", "NNTP", "SMTP", "TCP"]
    models = ["deepseek-reasoner", "gpt-4o-mini", "claude-3-7-sonnet-20250219", 
              "gemini-2.0-flash", "deepseek-chat",
              "qwq", "qwen3:32b","gemma3:27b","mistral-small3.1"]
    

    '''get the state matches results'''
    all_matches, summary_df = match_all_states(models, protocols, fsm_dir="fsm", threshold=0.5)
    summary_df['Model'] = summary_df['Model'].replace(model_name_mapping)
    print("All Matches:\n", all_matches)
    print("summary_df:\n", summary_df)
    import pandas as pd

    '''The following is to get the LaTeX table for each protocol'''
    # # Assuming summary_df is already loaded
    # protocol_dfs = {protocol: group.reset_index(drop=True) for protocol, group in summary_df.groupby("Protocol")}

    # # Generate LaTeX code for each protocol
    # for protocol, df in protocol_dfs.items():
    #     # Generate the LaTeX code
    #     latex_code = df[['Protocol', 'Model', 'Total Extracted', 'Total GT', 'Matched', 'Precision', 'Recall', 'F1-Score']].to_latex(
    #         index=False,
    #         float_format="%.3f",
    #         column_format="llcccccc",
    #         longtable=False,
    #         caption=f"{protocol} Protocol States Extraction Metrics",
    #         label=f"tab:{protocol.lower()}-states-matching-metrics",
    #         escape=False
    #     )
        
    #     # Save to .tex file
    #     file_name = f"{protocol}_states_metrics.tex"
    #     with open(file_name, "w") as f:
    #         f.write(latex_code)
        
    #     print(f"Generated LaTeX table for {protocol} and saved to {file_name}")

    # # Add booktabs formatting
    # latex_code = latex_code.replace("\\toprule", "\\toprule\n\\textbf{Protocol} & \\textbf{Model} & \\textbf{Total Extracted} & \\textbf{Total GT} & \\textbf{Matched} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\\n\\midrule")

    # print(latex_code)
    
    '''The following is to get the LaTeX table for each model of all protocols'''
    '''for the state metching results'''
    
    # # Group by Model and calculate the sum for Total Extracted, Total GT, and Matched
    # model_summary = summary_df.groupby("Model")[["Total Extracted", "Total GT", "Matched"]].sum().reset_index()

    # # Calculate overall Precision, Recall, and F1-Score for each model
    # model_summary["Precision"] = model_summary["Matched"] / model_summary["Total Extracted"]
    # model_summary["Recall"] = model_summary["Matched"] / model_summary["Total GT"]
    # model_summary["F1-Score"] = 2 * (model_summary["Precision"] * model_summary["Recall"]) / (model_summary["Precision"] + model_summary["Recall"])

    # latex_code = model_summary.to_latex(
    #     index=False,
    #     float_format="%.3f",
    #     column_format="|l|c|c|c|c|c|c|",
    #     longtable=False,
    #     caption="Overall Model Performance on States Matching of Different Protocols",
    #     label="tab:model-performance-summary",
    #     escape=False
    # )
    
    
    # print("Model performance summary generated successfully.\n", latex_code)
    
    '''start the trasistion matching'''
    transition_matches_df = batch_evaluate_transitions_combined(protocols=protocols, 
                                                                models=models, fsm_dir="fsm", if_partial=False)
    transition_matches_df.to_csv("transitions_match_results_whole.csv", index=False)
    print("transitio match results:", transition_matches_df)
    