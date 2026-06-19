from transformers import pipeline

class TransactionMonitor:
    def __init__(self):
        # Load weights once during API startup to save memory
        self.classifier = pipeline("text-classification", model="ProsusAI/finbert")

    def analyze(self, amount: float, description: str) -> dict:
        if amount > 10000:
            return {"status": "FLAGGED", "reason": "High transaction value"}

        ai_result = self.classifier(description)[0]
        if ai_result['label'] == 'negative' and ai_result['score'] > 0.7:
            return {"status": "FLAGGED", "reason": f"AI Risk Alert ({ai_result['score']:.2f})"}

        return {"status": "CLEAN", "reason": "Passed all checks"}
