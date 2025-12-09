import os
import numpy as np
import itertools
import tiktoken
from typing import List, Tuple
from litellm import embedding

from NiW.constants import API_BASE_URL
from NiW.claim_extraction import summarize_article

# openai.api_key = "" 

EMBED_MODEL = "text-embedding-3-small"

def get_embeddings(sentences: List[str]) -> np.ndarray:
    for i, sentence in enumerate(sentences):
        if len(tiktoken.get_encoding("cl100k_base").encode(sentence)) > 8192:
            sentences[i] = summarize_article(sentence)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return np.array([])
    for i in range(10):
        try:
            response = embedding(
                input=sentences,
                model=EMBED_MODEL,
                api_base=API_BASE_URL,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            break
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            continue
    embeddings = [np.array(e["embedding"]) for e in response["data"]]
    return np.vstack(embeddings)

def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1)
    return norm

# def greedy_dissimilar_selection(distance_matrix: np.ndarray, k: int) -> List[int]:
#     n = distance_matrix.shape[0]
#     if k >= n:
#         return list(range(n))
    
#     selected = [np.random.randint(n)]
#     while len(selected) < k:
#         remaining = list(set(range(n)) - set(selected))
#         max_avg_dist = -1
#         next_idx = -1
#         for i in remaining:
#             avg_dist = np.mean([distance_matrix[i][j] for j in selected])
#             if avg_dist > max_avg_dist:
#                 max_avg_dist = avg_dist
#                 next_idx = i
#         selected.append(next_idx)
#     return selected

def sort_sentence_by_dissimilarity(sentences: List[str], center: str) -> List[str]:
    embeddings = get_embeddings(sentences)
    dist_matrix = compute_distance_matrix(embeddings)
    center_embedding = get_embeddings([center])[0]
    selected_indices = np.argsort(np.linalg.norm(embeddings - center_embedding, axis=1))
    return [sentences[i] for i in selected_indices]


if __name__ == "__main__":
    #read sentences
    test_sentences = []
    with open("claims.md", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_sentences.append(line)
    with open("news_content.md", "r") as f:
        summary = f.read()
    top_k = 7
    dissimilar = find_most_dissimilar_group(test_sentences, summary, group_size=top_k)
    for i, sentence in enumerate(dissimilar):
        print(sentence)
    print(f"Selected {len(dissimilar)} most dissimilar sentences.")
