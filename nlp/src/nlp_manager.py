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
            for doc in documents_list:
                self.corpus.append({
                    "text": str(doc).strip()
                })
                    
        self.loaded = True
        
        return {"predictions": ["loaded"]}

    def _score_relevance(self, query: str, context: str) -> float:
        """Calculates keyword token density matching rules."""
        query_words = set([w.strip("?,.!\'\"()[]{}").lower() for w in query.split() if w])
        if not query_words:
            return 0.0
            
        context_lower = context.lower()
        score = 0.0
        for word in query_words:
            if word in context_lower:
                score += context_lower.count(word)
        return score

    def qa(self, question_payload: dict) -> dict:
        """Unpacks the question instances and packages answers into the required predictions format."""
        if not self.corpus:
            return {"predictions": []}

        instances = question_payload.get("instances", [])
        answers_list = []

        for instance in instances:
            question_text = instance.get("question", "")
            
            best_context = self.corpus[0]["text"] if self.corpus else ""
            max_score = -1.0

            for doc in self.corpus:
                score = self._score_relevance(question_text, doc["text"])
                if score > max_score:
                    max_score = score
                    best_context = doc["text"]

            answers_list.append(str(best_context).strip())

        return {"predictions": answers_list}
