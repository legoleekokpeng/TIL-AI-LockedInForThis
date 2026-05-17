"""Manages the NLP model."""


class NLPManager:
    loaded = False

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        self.corpus: list = []
        pass

    def load_corpus(self, documents: list) -> None:
        """Loads the corpus of documents for RAG QA."""
        # Your corpus loading code goes here.
        self.loaded = False
        self.corpus = []
        for doc in documents:
            if not doc:
                continue
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("doc_id") or "UNKNOWN"
                doc_text = doc.get("document") or doc.get("text") or ""
                
            else:
                self.corpus.append({
                    "id": str(doc_id),
                    "text": str(doc).strip()
                })
                
        self.loaded = True

    def qa(self, question: str) -> str:
        """Performs question answering on an image of a document.

        Args:
            question: The question to answer.

        Returns:
            A string containing the answer to the question.
        """

        # Your inference code goes here.
        if not self.corpus:
            return ""

        question_words = [w.strip("?,.!\'\"").lower() for w in question.split() if w]
        if not question_words:
            return "No question provided."

        keywords = set(question_words)
        
        best_doc_text = self.corpus[0]["text"]
        max_overlap = -1
        
        for doc in self.corpus:
            doc_words = set(doc["text"].lower().split())
            # Count common words shared between the question and the document
            overlap = len(keywords.intersection(doc_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_doc_text = doc["text"]

        sentences = best_doc_text.split('.')
        best_sentence = best_doc_text  # Fallback to full text if splitting is messy
        max_sentence_overlap = -1

        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
            sentence_words = set([w.strip("?,.!\'\"").lower() for w in clean_sentence.split()])
            sentence_overlap = len(keywords.intersection(sentence_words))
            
            if sentence_overlap > max_sentence_overlap:
                max_sentence_overlap = sentence_overlap
                best_sentence = clean_sentence + "."
                
        return best_sentence
