"""Manages the NLP model."""


class NLPManager:
    loaded = False

    def __init__(self): -> None:
        # This is where you can initialize your model and any static configurations.
        self.corpus = []
        pass

    def load_corpus(self, documents) -> None:
        """Loads the corpus of documents for RAG QA."""
        # Your corpus loading code goes here.
        self.loaded = False
        self.corpus = []
        raw_list = []
        try:
            if isinstance(documents, dict):
                raw_list = documents.get("documents") or documents.get("corpus") or documents.get("data") or []
                if not raw_list and isinstance(documents, dict):
                    raw_list = list(documents.values())
            elif isinstance(documents, list):
                raw_list = documents
            else:
                raw_list = [documents]

            for doc in raw_list:
                if not doc:
                    continue
                
                if isinstance(doc, dict):
                    doc_text = doc.get("document") or doc.get("text") or doc.get("content") or ""
                    self.corpus.append({"text": str(doc_text).strip()})
                else:
                    self.corpus.append({"text": str(doc).strip()})

        except Exception as e:
            print(f"[CRITICAL ERROR DURING PARSING]: {str(e)}")
            
        finally:
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

        f not self.corpus:
            return ""

        keywords = set([w.strip("?,.!\'\"").lower() for w in question.split() if w])
        if not keywords:
            return "No question context identified."

        best_sentence = self.corpus[0]["text"] if self.corpus else ""
        max_overlap = -1
        
        for doc in self.corpus:
            doc_text = doc["text"]
            sentences = doc_text.split('.')
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if not clean_sentence:
                    continue
                sentence_words = set([w.strip("?,.!\'\"").lower() for w in clean_sentence.split()])
                overlap = len(keywords.intersection(sentence_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = clean_sentence + "."

        return best_sentence
