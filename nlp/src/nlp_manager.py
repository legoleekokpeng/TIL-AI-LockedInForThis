class NLPManager:
    loaded = False

    def __init__(self) -> None:
        self.corpus: list = []
        self.loaded = True

    def load_corpus(self, documents: list) -> None:
        """Ingests the document corpus array matching the template keys exactly."""
        self.corpus = []
        for doc in documents:
            self.corpus.append({
                "id": doc.get("id"),
                "text": doc.get("document", "").strip()
            })
        self.loaded = True

    def _score_relevance(self, query: str, context: str) -> float:
        """Calculates keyword match density score between a query and a text body."""
        query_words = set([w.strip("?,.!\'\"()[]{}").lower() for w in query.split() if w])
        if not query_words:
            return 0.0
            
        context_lower = context.lower()
        score = 0.0
        for word in query_words:
            if word in context_lower:
                score += context_lower.count(word)
        return score

    def qa(self, question: str) -> str:
        """Locates the absolute best document source block and returns its text directly."""
        if not self.corpus:
            return ""

        best_context = self.corpus[0]["text"]
        max_score = -1.0

        for doc in self.corpus:
            score = self._score_relevance(question, doc["text"])
            if score > max_score:
                max_score = score
                best_context = doc["text"]

        return str(best_context).strip()
