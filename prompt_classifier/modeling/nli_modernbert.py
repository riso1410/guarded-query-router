from transformers import pipeline


class ModernBERTNLI:
    def __init__(self, domain: str) -> None:
        self.classifier = pipeline("zero-shot-classification", model="tasksource/ModernBERT-base-nli", device=0, truncation=True, max_length=256, batch_size=32)
        self.label = [domain, "general"]

    def predict(self, text: str) -> dict:
        prediction = self.classifier(text, self.label)
        if prediction == "general":
            return 0
        return 1
