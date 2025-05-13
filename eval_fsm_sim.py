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

def preprocess_state_name(state_name):
    # Convert to lowercase, remove punctuation, and split compound words
    state_name = state_name.lower()
    state_name = re.sub(r"[\W_]+", " ", state_name)  # Replace non-alphanumerics with space
    state_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", state_name)  # Split camel case
    state_name = state_name.strip()
    return state_name



def match_states(source_states, target_states, threshold=0.5):
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
    model_obj = SentenceTransformer('all-MiniLM-L6-v2')
    all_matches = {}

    for protocol in protocols:
        # Load ground truth FSM
        gt_file = os.path.join(fsm_dir, protocol, f"gt_{protocol}_state_machine.json")
        if not os.path.exists(gt_file):
            print(f"Warning: No ground truth file for protocol {protocol}")
            continue

        with open(gt_file, 'r') as f:
            gt_fsm = json.load(f)
        gt_states = gt_fsm.get('states', [])

        for model in models:
            # Load extracted FSM
            ext_file = os.path.join(fsm_dir, f"{model}_{protocol}.json")
            if not os.path.exists(ext_file):
                print(f"Warning: No extracted FSM file for model {model} and protocol {protocol}")
                continue

            with open(ext_file, 'r') as f:
                ext_fsm = json.load(f)
            ext_states = ext_fsm.get('states', [])

            # Match states
            matches, unmatched = match_states(gt_states, ext_states, model_obj, threshold=threshold)
            all_matches[(protocol, model)] = {
                "matches": matches,
                "unmatched": unmatched
            }

    return all_matches

def compute_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return sim


def extract_initial_state(fsm_file):
    with open(fsm_file, 'r') as f:
        fsm_data = json.load(f)
    return fsm_data.get('initial_state', None)


# this file is to calculate the similarity of the initial state between the ground truth and extracted FSMs
def compare_initial_states(models, protocols, fsm_dir, threshold=0.5):
    
    score_matrix = pd.DataFrame(index=protocols, columns=models)

    results = []
    
    # Iterate over each protocol
    for protocol in protocols:
        # Find the ground truth FSM file for the protocol
        protocol_dir = os.path.join(protocol)
        gt_file = None
        for file in os.listdir(protocol_dir):
            if file.endswith("_state_machine.json"):
                gt_file = os.path.join(protocol_dir, file)
                break

        if not gt_file:
            continue

        gt_initial = preprocess_state_name(extract_initial_state(gt_file))

        # Iterate over each model
        for model in models:
            extracted_file = None

            # Find the extracted FSM file for the protocol and model
            for file in os.listdir(fsm_dir):
                if protocol in file and model in file and file.endswith(".json"):
                    extracted_file = os.path.join(fsm_dir, file)
                    break

            if not extracted_file:
                continue

            ext_initial = preprocess_state_name(extract_initial_state(extracted_file))

            # Calculate similarity
            model_name = model
            protocol_name = protocol
            sim_score = compute_similarity(gt_initial, ext_initial)

            match = (gt_initial, ext_initial, sim_score) if sim_score >= threshold else (gt_initial, None, sim_score)
            results.append((protocol_name, model_name, match))
            # Add score to matrix
            score_matrix.loc[protocol, model] = sim_score
            

    return results, score_matrix

def plot_score_matrix(score_matrix, threshold=0.5, output_file='score_matrix.png'):
    # Ensure the score matrix is numeric
    numeric_matrix = score_matrix.astype(float).copy()
    plt.figure(figsize=(18, 12))
    # Custom color map with yellow for <=0.5 and green for >0.5
    cmap = sns.diverging_palette(65, 145, s=85, l=65, sep=20, as_cmap=True)
    # Ensure the score matrix is numeric
    numeric_matrix = score_matrix.astype(float).copy()
    plt.figure(figsize=(18, 12))
    # Custom color map with green for >0.5 and yellow for <=0.5
    colors = [(0.8, 0.8, 0.2), (0.2, 0.8, 0.2)]  # yellow to green
    cmap = sns.blend_palette(colors, as_cmap=True)
    # Ensure the score matrix is numeric
    numeric_matrix = score_matrix.astype(float).copy()
    plt.figure(figsize=(18, 12))
    cmap = plt.get_cmap('YlGn')
    # Use different color intensities for scores above and below the threshold
    ax = sns.heatmap(numeric_matrix, annot=True, cmap=cmap, linewidths=0.5, center=threshold, cbar_kws={"label": "Similarity Score"}, vmin=0, vmax=1)
    plt.title("Initial State Matching Scores with Summary Rows")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Score matrix saved as {output_file}")
    plt.show()
    

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
    protocols = ["MQTT","PPP","PPTP", "BGP"] # check for testing
    #protocols = ["SIP", "RTSP", "DCCP", "DHCP", "FTP", "NNTP", "SMTP", "TCP"]
    # close_models = ["claude-3-7-sonnet-20250219"]
    # , "gemini-2.0-flash"
    close_models = ["deepseek-reasoner", "gpt-4o-mini", "claude-3-7-sonnet-20250219", "gemini-2.0-flash"]
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
    #batch_evaluate_fsm_similarity()
    protocols = ["IMAP", "POP3", "MQTT","PPP","PPTP", "BGP",
                 "SIP", "RTSP", "DCCP", "DHCP", "FTP", "NNTP", "SMTP", "TCP"]
    models = ["deepseek-reasoner", "gpt-4o-mini", "claude-3-7-sonnet-20250219", "gemini-2.0-flash", "deepseek-chat"]
    
    initial_state_results, score_matrix = compare_initial_states(models, protocols, fsm_dir="fsm", threshold=0.5)
    print("Initial State Results:", initial_state_results)
    print("Score Matrix:\n", score_matrix)
    
    plot_score_matrix(score_matrix, threshold=0.5, output_file='initial_state_score_matrix.png')
    
    
    