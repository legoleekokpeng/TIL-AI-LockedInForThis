import logging
import math
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPManager:
    loaded = False

    def __init__(self):
        logger.info("Initializing Pure-Python BM25 Engine...")
        self.corpus_docs = []
        self.corpus_ids = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.df = {}
        self.idf = {}
        self.doc_term_freqs = []
        
        # BM25 Hyperparameters
        self.k1 = 1.5
        self.b = 0.75
        self.loaded = False
        logger.info("✅ Initialization complete.")

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        """Loads the corpus of documents following the exact test_nlp.py schema."""
        logger.info(f"Loading corpus with {len(documents)} documents...")
        try:
            self.corpus_docs = []
            self.corpus_ids = []
            self.doc_lengths = []
            self.doc_term_freqs = []
            self.df = {}
            self.idf = {}

            for doc in documents:
                doc_id = doc.get("id")
                doc_text = doc.get("document")
                
                if doc_id is not None and doc_text is not None:
                    doc_id_str = str(doc_id)
                    doc_text_str = str(doc_text)
                    
                    self.corpus_ids.append(doc_id_str)
                    self.corpus_docs.append(doc_text_str)
                    
                    tokens = re.findall(r'\w+', doc_text_str.lower())
                    self.doc_lengths.append(len(tokens))
                    
                    tf = {}
                    for token in tokens:
                        tf[token] = tf.get(token, 0) + 1
                    self.doc_term_freqs.append(tf)
                    
                    for token in tf:
                        self.df[token] = self.df.get(token, 0) + 1

            N = len(self.corpus_docs)
            if N > 0:
                self.avg_doc_len = sum(self.doc_lengths) / N
                for token, freq in self.df.items():
                    # Standard BM25 IDF variant formulation
                    self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
                logger.info(f"✅ Successfully processed {N} documents into BM25 indexes.")
            else:
                self.avg_doc_len = 0.0
                logger.warning("Loaded corpus contains zero valid document objects.")

            self.loaded = True

        except Exception as e:
            logger.error(f"Error encountered during load_corpus: {str(e)}")
            self.loaded = True

    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Performs precise document ranking and answer sentence extraction for L1/L2."""
        if not self.loaded or not self.corpus_docs:
            return {"documents": [], "answer": ""}

        try:
            q_tokens = re.findall(r'\w+', question.lower())
            if not q_tokens:
                return {"documents": self.corpus_ids[:3], "answer": ""}

            scores = []
            for i in range(len(self.corpus_docs)):
                score = 0.0
                tf_dict = self.doc_term_freqs[i]
                d_len = self.doc_lengths[i]
                
                for token in q_tokens:
                    if token in tf_dict:
                        tf = tf_dict[token]
                        idf = self.idf.get(token, 0.0)
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                        score += idf * (numerator / denominator)
                scores.append(score)

            # Extract the top 3 document indices sorted by score descending
            top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:3]
            top_doc_ids = [self.corpus_ids[idx] for idx in top_indices]

            # Locate the highest scoring document to perform sentence context extraction
            best_doc_idx = top_indices[0]
            best_doc_text = self.corpus_docs[best_doc_idx]

            # Segment the target document cleanly into sentences
            sentences = re.split(r'(?<=[.!?])\s+', best_doc_text)
            best_sentence = ""
            max_sentence_score = -1.0

            for sentence in sentences:
                s_tokens = re.findall(r'\w+', sentence.lower())
                if not s_tokens:
                    continue
                
                # Rank sentence match strength based on Term-Frequency weightings
                s_score = 0.0
                s_tf = {}
                for tok in s_tokens:
                    s_tf[tok] = s_tf.get(tok, 0) + 1
                for token in q_tokens:
                    if token in s_tf:
                        s_score += self.idf.get(token, 1.0) * (s_tf[token] / (s_tf[token] + 1.0))
                
                if s_score > max_sentence_score:
                    max_sentence_score = s_score
                    best_sentence = sentence

            # Fallback to the first sentence if no keyword intersections were found
            if not best_sentence and sentences:
                best_sentence = sentences[0]

            return {
                "documents": top_doc_ids,
                "answer": best_sentence.strip()
            }

        except Exception as e:
            logger.error(f"Inference execution fallback triggered: {str(e)}")
            return {"documents": self.corpus_ids[:3], "answer": ""}
