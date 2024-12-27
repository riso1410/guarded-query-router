from transformers import pipeline


class ModernBERTNLI:
    def __init__(self, domain: str) -> None:
        self.classifier = pipeline("zero-shot-classification", model="tasksource/ModernBERT-base-nli")
        self.label = [domain, "general"]

    def predict(self, text: str) -> dict:
        return self.classifier(text, self.label)
