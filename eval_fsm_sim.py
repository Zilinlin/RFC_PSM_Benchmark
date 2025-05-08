'''This file is to evaluate the FSM similarity between two FSMs using fuzzy matching on both transitions and states.'''

import os
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return sim



def match_transitions_fuzzy_states(trans1, trans2, threshold=0.5, allow_partial=True):
    """
    Compare transitions from two FSMs and classify matches as fully correct, partially correct, or unmatched.
    """
    matched = []
    unmatched = []
    fully_correct = 0
    partially_correct = 0
    trans2_copy = trans2.copy()

    for idx1, t1 in enumerate(trans1):
        found = False
        for idx2, t2 in enumerate(trans2_copy):
            from_score = compute_similarity(t1.get("from", ""), t2.get("from", ""))
            to_score = compute_similarity(t1.get("to", ""), t2.get("to", ""))
            event_score = compute_similarity(t1.get("event", ""), t2.get("event", ""))
            action_score = compute_similarity(t1.get("action", ""), t2.get("action", ""))

            from_match = from_score >= threshold
            to_match = to_score >= threshold
            event_match = event_score >= threshold
            action_match = action_score >= threshold

            # Debugging output
            print(f"\nComparing Transition {idx1} (trans1) with Transition {idx2} (trans2):")
            print(f"  From: '{t1.get('from')}' vs '{t2.get('from')}' | Match: {from_match} | Score: {from_score:.2f}")
            print(f"  To: '{t1.get('to')}' vs '{t2.get('to')}' | Match: {to_match} | Score: {to_score:.2f}")
            print(f"  Event: '{t1.get('event')}' vs '{t2.get('event')}' | Match: {event_match} | Score: {event_score:.2f}")
            print(f"  Action: '{t1.get('action')}' vs '{t2.get('action')}' | Match: {action_match} | Score: {action_score:.2f}")

            if from_match and to_match:
                if event_match and action_match:
                    matched.append((t1, t2))
                    fully_correct += 1
                    trans2_copy.remove(t2)
                    found = True
                    print("  --> Fully Matched\n")
                    break
                elif allow_partial and (event_match or action_match):
                    matched.append((t1, t2))
                    partially_correct += 1
                    trans2_copy.remove(t2)
                    found = True
                    print("  --> Partially Matched\n")
                    break
        if not found:
            unmatched.append(t1)
            print("  --> Not Matched\n")

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
    # IMAP ground truth file is missing, need to handle this case.
    # "IMAP", 
    # "POP3" has bugs
    #protocols = ["IMAP", "POP3"]
    protocols = ["NNTP","SIP","FTP"] # check for testing
    #protocols = ["SIP", "RTSP", "DCCP", "DHCP", "FTP", "NNTP", "SMTP", "TCP"]
    # close_models = ["claude-3-7-sonnet-20250219"]
    # , "gemini-2.0-flash"
    close_models = ["deepseek-reasoner", "gpt-4o-mini", "claude-3-7-sonnet-20250219"]
    fsm_dir = "fsm"
    results = {}

    # Iterate over each protocol
    for protocol in protocols:
        protocol_dir = os.path.join(protocol)
        gt_file = None

        # Find the ground truth FSM file for the protocol
        for file in os.listdir(protocol_dir):
            if file.endswith("_state_machine.json"):
                gt_file = os.path.join(protocol_dir, file)
                break

        if not gt_file:
            continue

        # Load the ground truth FSM
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)

        # Iterate over each model
        for model in close_models:
            extracted_file = None

            # Find the extracted FSM file for the protocol and model
            for file in os.listdir(fsm_dir):
                if protocol in file and model in file and file.endswith(".json"):
                    extracted_file = os.path.join(fsm_dir, file)
                    break

            if not extracted_file:
                continue

            # Load the extracted FSM
            with open(extracted_file, 'r') as f:
                extracted = json.load(f)

            # Evaluate the similarity between ground truth and extracted FSMs
            results_key = f"{protocol}_{model}"
            results[results_key] = {
                "with_partial": evaluate_fsm_similarity(ground_truth, extracted, allow_partial=True),
                "no_partial": evaluate_fsm_similarity(ground_truth, extracted, allow_partial=False)
            }

            # Print the evaluation results
            print(f"Evaluated {protocol} with model {model}")
            print(f"Results for {results_key}:")
            print(results[results_key])

            # Write the current results to the output file
            with open("fsm_evaluation_results.json", "w") as outfile:
                json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    batch_evaluate_fsm_similarity()