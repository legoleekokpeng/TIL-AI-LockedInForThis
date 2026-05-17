"""Manages the NLP model."""


class NLPManager:
    loaded = False

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        self.corpus: list[dict] = []
        pass

    def load_corpus(self, documents: list[dict]) -> None:
        """Loads the corpus of documents for RAG QA."""
        # Your corpus loading code goes here.
        self.corpus = []
        for doc in documents:
            self.corpus.append({
                "id": doc.get("id", "UNKNOWN"),
                "text": doc.get("document", "").strip()
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
            return "The database core is empty."

        question_words = set(question.lower().split())
        
        best_doc_text = self.corpus[0]["text"]
        max_overlap = -1
        
        for doc in self.corpus:
            doc_words = set(doc["text"].lower().split())
            # Count common words shared between the question and the document
            overlap = len(question_words.intersection(doc_words))
            
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
            sentence_words = set(clean_sentence.lower().split())
            sentence_overlap = len(question_words.intersection(sentence_words))
            
            if sentence_overlap > max_sentence_overlap:
                max_sentence_overlap = sentence_overlap
                best_sentence = clean_sentence + "."
                
        return best_sentence
