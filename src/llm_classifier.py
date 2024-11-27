import re
import dspy
import pandas as pd
from utilities import *
from sklearn import metrics
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy import LM, Example, configure
import matplotlib.pyplot as plt

class LMClassifier:
    """Setup for LM configuration and initialization."""

    def __init__(self, api_key: str, proxy_url: str, domain: str, model_name: str, 
                 temperature=0.2, train_size=100, test_size=200):
        self.api_key = api_key
        self.proxy_url = proxy_url
        self.domain = domain
        self.model_name = model_name
        self.temperature = temperature
        self.train_size = train_size
        self.test_size = test_size

        self.lm = LM(
            api_key=self.api_key,
            model=self.model_name,
            api_base=self.proxy_url,
            temperature=self.temperature
        )
        configure(lm=self.lm)
        
    @staticmethod
    def create_example(domain: str, row: pd.Series) -> Example:
        return Example(
            domain=domain,
            prompt=row['question'],
            label=row['label'],
        ).with_inputs("prompt","domain")

    def load_data(self, train_data, test_data):
        train_data = [self.create_example(self.domain, row) for _, row in train_data.iterrows()]
        test_data = [self.create_example(self.domain, row) for _, row in test_data.iterrows()]

        self.train_data = train_data[:self.train_size]
        self.test_data = test_data[:self.test_size]
        
        print(f"Train data: {len(self.train_data)}")
        print(f"Test data: {len(self.test_data)}")


class ClassificationSignature(dspy.Signature):
    """Classify if a text is specific for a domain or not."""
    
    domain = dspy.InputField(desc="The target domain to classify the prompt against.")
    prompt = dspy.InputField(desc="The prompt to classify.")
    
    #explanation = dspy.OutputField(desc="Reasoning behind the classification.")
    label = dspy.OutputField(desc="1, if the input text belong to domain, 0 otherwise.")
    

class ClassificationModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(ClassificationSignature)
        
    def forward(self, prompt: str, domain: str) -> ClassificationSignature:        
        try:
            prediction = self.prog(prompt=prompt, domain=domain)

        except Exception as e:
            print(f"An error occurred while classifying the prompt: {e}\nPrompt: {prompt}")
            prediction = ClassificationSignature(label=0)

        return prediction


class LMTrainer:
    """Handles model optimization using few-shot learning with BootstrapFewShotWithRandomSearch."""
    def __init__(self, train_data):
        self.train_data = train_data
        self.optimized_model = None

    @staticmethod
    def parse_answer(answer) -> bool:
        """Parse answers into a consistent binary format."""
        if isinstance(answer, str) and re.match(r"^[01]$", answer.strip()):
            return bool(int(answer))
        elif isinstance(answer, int) and answer in [0, 1]:
            return bool(answer)
        else:
            print(f"Unexpected non-binary label found: {answer}")
            return False

    def evaluate(self, predictions, true_labels):
        predicted_labels = [self.parse_answer(pred) for pred in predictions]
        actual_labels = [self.parse_answer(label) for label in true_labels]

        matrix = metrics.confusion_matrix(actual_labels, predicted_labels)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [0, 1])
        cm_display.plot()
        plt.show() 

        f1 = metrics.f1_score(actual_labels, predicted_labels)
        accuracy = metrics.accuracy_score(actual_labels, predicted_labels)
        recall = metrics.recall_score(actual_labels, predicted_labels)
        precision = metrics.precision_score(actual_labels, predicted_labels)

        return f1, accuracy, recall, precision

    def comparison_metric(self, example, pred, trace=None) -> bool:
        """Metric function for comparing predicted label with actual label, using parse_answer for consistency."""
        parsed_example_label = self.parse_answer(example.label)
        parsed_pred_label = self.parse_answer(pred.label)

        return parsed_example_label == parsed_pred_label

    def optimize_model(self):
        """Optimize the model using few-shot learning."""
        fewshot_optimizer = BootstrapFewShotWithRandomSearch(
            metric=self.comparison_metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            max_rounds=1,
            num_candidate_programs=4,
        )
        
        compiled_classification = fewshot_optimizer.compile(ClassificationModule(), trainset=self.train_data)
        self.optimized_model = compiled_classification

        return compiled_classification
    
    def predict(self, prompt: str, domain: str):
        """Predict the label for a given prompt."""
        return self.optimized_model(prompt=prompt, domain=domain).label
    
    def save_model(self, model_path: str):
        """Save the optimized model."""
        self.optimized_model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the optimized model."""
        self.optimized_model = ClassificationModule()
        self.optimized_model.load(model_path)
        print(f"Model loaded from {model_path}")