from sentence_transformers import SentenceTransformer, util
import numpy as np
import re


def preprocess_state_name(state_name):
    # Convert to lowercase, remove punctuation, and split compound words
    state_name = state_name.lower()
    state_name = re.sub(r"[\W_]+", " ", state_name)  # Replace non-alphanumerics with space
    state_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", state_name)  # Split camel case
    state_name = state_name.strip()
    return state_name


def match_states(source_states, target_states, threshold=0.5, model_name="all-MiniLM-L6-v2"):
    # Preprocess the state names
    source_clean = [preprocess_state_name(s) for s in source_states]
    target_clean = [preprocess_state_name(t) for t in target_states]

    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
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


# Example usage:
source_states = ["Idle", "Connection_Established", "Data_Transfer", "Error"]
target_states = ["Idle State", "Connected", "Data Exchange", "Disconnected", "Failure"]
matches, unmatched = match_states(source_states, target_states, threshold=0.5)

print("Matched States:")
for src, tgt, score in matches:
    print(f"{src} -> {tgt} (score: {score:.4f})")

print("\nUnmatched Target States:")
print(unmatched)
