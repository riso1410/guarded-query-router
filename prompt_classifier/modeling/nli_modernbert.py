from transformers import pipeline


class ModernBERTNLI:
    def __init__(self, domain: str, threshold: float = 0.7) -> None:
        self.domain = domain
        self.threshold = threshold
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="tasksource/ModernBERT-base-nli", 
            device=0, 
            truncation=True, 
            max_length=256, 
            batch_size=32
        )
        self.labels = [f"related to {domain}", f"not related to {domain}"]

    def predict(self, text: str) -> int:
        # The pipeline returns a dictionary with 'labels' and 'scores'
        prediction = self.classifier(text, self.labels)
        
        # Get the index of the "related" label
        related_idx = prediction['labels'].index(f"related to {self.domain}")
        
        # Get the confidence score for the "related" label
        confidence = prediction['scores'][related_idx]
        
        # Apply threshold
        if confidence >= self.threshold:
            return 1
        else:
            return 0