'''This file is to evaluate the FSM similarity between two FSMs using fuzzy matching on both transitions and states.'''

import os
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def label_sim(a, b, threshold=0.5):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return sim > threshold, sim

def compare_action_lists(a1, a2, threshold=0.5):
    if not a1 and not a2:
        return "full"
    if isinstance(a1, str):
        a1 = [a1]
    if isinstance(a2, str):
        a2 = [a2]
    matches = 0
    for item1 in a1:
        for item2 in a2:
            match, _ = label_sim(item1, item2, threshold)
            if match:
                matches += 1
                break
    if matches == len(a1) == len(a2):
        return "full"
    elif matches > 0:
        return "partial"
    else:
        return "none"

def match_transitions_fuzzy_states(trans1, trans2, threshold=0.5, allow_partial=True):
    matched = []
    unmatched = []
    fully_correct = 0
    partially_correct = 0
    trans2_copy = trans2.copy()
    for t1 in trans1:
        found = False
        for t2 in trans2_copy:
            from_match, _ = label_sim(t1["from"], t2["from"], threshold)
            to_match, _ = label_sim(t1["to"], t2["to"], threshold)
            req_match, _ = label_sim(t1["requisite"], t2["requisite"], threshold)
            act_match_type = compare_action_lists(t1["actions"], t2["actions"], threshold)
            if from_match and to_match and req_match:
                if act_match_type == "full" or (allow_partial and act_match_type == "partial"):
                    matched.append((t1, t2))
                    if act_match_type == "full":
                        fully_correct += 1
                    elif act_match_type == "partial":
                        partially_correct += 1
                    trans2_copy.remove(t2)
                    found = True
                    break
        if not found:
            unmatched.append(t1)
    return matched, unmatched, fully_correct, partially_correct

def compute_match_metrics(matched, gt_trans, ex_trans):
    tp = len(matched)
    fn = len(gt_trans) - tp
    fp = len(ex_trans) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def evaluate_fsm_similarity(fsm1_json, fsm2_json, allow_partial=True, threshold=0.5):
    matched, unmatched, fully_correct, partially_correct = match_transitions_fuzzy_states(
        fsm1_json["transitions"], fsm2_json["transitions"], threshold=threshold, allow_partial=allow_partial)
    precision, recall, f1 = compute_match_metrics(matched, fsm1_json["transitions"], fsm2_json["transitions"])
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "matched": len(matched),
        "fully_correct": fully_correct,
        "partially_correct": partially_correct,
        "unmatched": len(unmatched),
    }

def batch_evaluate_fsm_similarity():
    protocols = ["DCCP", "DHCP", "FTP", "IMAP", "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP"]
    close_models = ["deepseek-reasoner", "gpt-4o-mini", "claude-3-7-sonnet-20250219", "gemini-2.0-flash"]
    fsm_dir = "fsm"
    results = {}

    for protocol in protocols:
        protocol_dir = os.path.join(protocol)
        gt_file = None
        for file in os.listdir(protocol_dir):
            if file.endswith("_state_machine.json"):
                gt_file = os.path.join(protocol_dir, file)
                break
        if not gt_file:
            continue

        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)

        for model in close_models:
            extracted_file = None
            for file in os.listdir(fsm_dir):
                if protocol in file and model in file and file.endswith(".json"):
                    extracted_file = os.path.join(fsm_dir, file)
                    break
            if not extracted_file:
                continue

            with open(extracted_file, 'r') as f:
                extracted = json.load(f)

            results_key = f"{protocol}_{model}"
            results[results_key] = {
                "with_partial": evaluate_fsm_similarity(ground_truth, extracted, allow_partial=True),
                "no_partial": evaluate_fsm_similarity(ground_truth, extracted, allow_partial=False)
            }
            print(f"Evaluated {protocol} with model {model}")
            print(f"Results for {results_key}:")
            print(results[results_key])

    with open("fsm_evaluation_results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)

if __name__ == "__main__":
    batch_evaluate_fsm_similarity()