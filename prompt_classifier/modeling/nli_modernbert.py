from typing import List

from transformers import pipeline


class ModernBERTNLI:
    """BERT model fine-tuned for Natural Language Inference classification tasks.

    Uses zero-shot classification to determine if text belongs to a specific domain.

    Attributes:
        domain (str): Target domain for classification
        threshold (float): Confidence threshold for classification
        classifier: Transformer pipeline for zero-shot classification
        labels (List[str]): Classification labels
    """

    def __init__(self, domain: str, threshold: float = 0.7) -> None:
        self.domain = domain
        self.threshold = threshold
        self.classifier = pipeline(
            "zero-shot-classification",
            model="tasksource/ModernBERT-base-nli",
            device=0,
            truncation=True,
            max_length=256,
            batch_size=32,
        )
        self.labels = [f"related to {domain}", f"not related to {domain}"]

    def predict(self, text: str) -> int:
        """Predict if text belongs to the specified domain.

        Args:
            text (str): Input text to classify

        Returns:
            int: 1 if text belongs to domain, 0 otherwise
        """
        # The pipeline returns a dictionary with 'labels' and 'scores'
        prediction = self.classifier(text, self.labels)

        # Get the index of the "related" label
        related_idx = prediction["labels"].index(f"related to {self.domain}")

        # Get the confidence score for the "related" label
        confidence = prediction["scores"][related_idx]

        # Apply threshold
        if confidence >= self.threshold:
            return 1
        else:
            return 0
