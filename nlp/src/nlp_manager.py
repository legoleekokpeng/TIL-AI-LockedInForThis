"""Manages the NLP model using SentenceTransformer dense embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer

class NLPManager:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.loaded = False

    def load_corpus(self, documents: list) -> None:
        """Chunks documents and pre-computes dense neural network embeddings."""
        self.loaded = False
        self.chunks = []

        raw_list = []
        if isinstance(documents, dict):
            raw_list = documents.get("documents") or documents.get("corpus") or documents.get("data") or []
            if not raw_list:
                raw_list = list(documents.values())
        elif isinstance(documents, list):
            raw_list = documents
        else:
            raw_list = [documents]

        # Process text chunks
        for doc in raw_list:
            if not doc:
                continue
                
            text = doc.get("document") or doc.get("text") or doc.get("content") or "" if isinstance(doc, dict) else str(doc)
            
            # Chunk into sentences
            sentences = text.split('.')
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) < 15:  # Skip empty lines or stray characters
                    continue
                    
                chunk_text = clean_sentence + "."
                self.chunks.append({
                    "text": chunk_text,
                    "vector": None
                })

        if self.chunks:
            texts_to_embed = [chunk["text"] for chunk in self.chunks]
            embeddings = self.model.encode(texts_to_embed, normalize_embeddings=True)
            
            for i, emb in enumerate(embeddings):
                self.chunks[i]["vector"] = emb

        self.loaded = True

    def qa(self, question: str) -> str:
        """Calculates semantic similarity using a dense vector dot product."""
        if not self.chunks:
            return ""

        query_vector = self.model.encode(question, normalize_embeddings=True)

        best_chunk_text = self.chunks[0]["text"]
        max_similarity = -1.0

        for chunk in self.chunks:
            if chunk["vector"] is None:
                continue
            
            similarity = float(np.dot(query_vector, chunk["vector"]))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_chunk_text = chunk["text"]

        return best_chunk_text
