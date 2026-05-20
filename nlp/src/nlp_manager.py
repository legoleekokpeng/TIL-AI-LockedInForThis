class NLPManager:
    loaded = False

    def __init__(self) -> None:
        self.corpus: list = []
        self.loaded = False

    def load_corpus(self, payload: dict) -> dict:
        self.loaded = False
        self.corpus = []
        
        instances = payload.get("instances", [])
        for instance in instances:
            documents_list = instance.get("documents", [])
            for doc_identifier in documents_list:
                self.corpus.append(str(doc_identifier).strip())
                    
        self.loaded = True
        return {"predictions": ["loaded"]}

    def _score_id_relevance(self, query: str, doc_id: str) -> float:
        query_words = set([w.strip("?,.!\'\"()[]{}").lower() for w in query.split() if w])
        doc_clean = doc_id.lower().replace(".txt", "").replace("_", " ").replace("-", " ")
        
        score = 0.0
        for word in query_words:
            if word in doc_clean:
                score += 1.0
        return score

    def qa(self, question_payload: dict) -> dict:
        if not self.corpus:
            return {"predictions": []}

        instances = question_payload.get("instances", [])
        final_predictions = []

        for instance in instances:
            question_text = instance.get("question", "")

            scored_docs = []
            for doc_id in self.corpus:
                score = self._score_id_relevance(question_text, doc_id)
                scored_docs.append((doc_id, score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            top_3 = [doc[0] for doc in scored_docs[:3]]
            
            while len(top_3) < 3 and self.corpus:
                top_3.append(self.corpus[0])

            final_predictions.append(top_3)
            
        return {"predictions": final_predictions}
