import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    return np.array(get_model().encode(texts, convert_to_numpy=True), dtype='float32')

def build_faiss(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def save_faiss(index, id_map, base_path):
    faiss.write_index(index, str(base_path) + ".index")
    with open(str(base_path) + ".json", "w") as f:
        json.dump(id_map, f)

def load_faiss(base_path):
    try:
        index = faiss.read_index(str(base_path) + ".index")
        with open(str(base_path) + ".json") as f:
            id_map = json.load(f)
        id_map = {int(k): v for k, v in id_map.items()}
        return index, id_map
    except:
        return None, {}
