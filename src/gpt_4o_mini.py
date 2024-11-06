import re
import dspy
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy import LM, Example, configure

class GPT4Model:
    """Setup for GPT-4 model configuration and initialization."""

    def __init__(self, api_key: str, proxy_url: str, domain: str, model_name: str, temperature=0.2, train_size=100, test_size=200, seed=22):
        self.api_key = api_key
        self.proxy_url = proxy_url
        self.domain = domain
        self.model_name = model_name
        self.temperature = temperature
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed

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
            target=domain,
            prompt=row["question"],
            completion=row["answer"],
            label=row["label"],
        ).with_inputs("prompt")

    def load_data(self, open_path: str, specific_path: str):
        open_data = pd.read_csv(open_path)
        specific_data = pd.read_csv(specific_path)

        open_examples = [self.create_example(self.domain, row) for _, row in open_data.iterrows()]
        specific_examples = [self.create_example(self.domain, row) for _, row in specific_data.iterrows()]

        final_data = open_examples + specific_examples
        random.seed(self.seed)
        random.shuffle(final_data)

        self.train_data = final_data[:self.train_size]
        self.test_data = final_data[:self.test_size]
        
        print(f"Train data: {len(self.train_data)}")
        print(f"Test data: {len(self.test_data)}")


class ClassificationSignature(dspy.Signature):
    """Classify if a text is specific for a domain or not."""
    
    target = dspy.OutputField(desc="The target domain to classify the prompt against.")
    prompt = dspy.InputField(desc="The prompt to classify.")
    
    #explanation = dspy.OutputField(desc="Reasoning behind the classification.")
    label = dspy.OutputField(desc="1, if the input text belong to domain, 0 otherwise.")
    

class ClassificationModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(ClassificationSignature)
        
    def forward(self, prompt: str) -> ClassificationSignature:
        prediction = self.prog(prompt=prompt)
        return prediction


class Trainer:
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

    def evaluate_model(self, predictions, true_labels):
        parsed_preds = [self.parse_answer(pred) for pred in predictions]
        parsed_labels = [self.parse_answer(label) for label in true_labels]
        return accuracy_score(parsed_labels, parsed_preds)

    def comparison_metric(self, example, pred, trace=None) -> bool:
        """Metric function for comparing predicted label with actual label, using parse_answer for consistency."""
        parsed_example_label = self.parse_answer(example.label)
        parsed_pred_label = self.parse_answer(pred.label)
        return parsed_example_label == parsed_pred_label

    def optimize_model(self):
        """Optimize the model using few-shot learning."""
        fewshot_optimizer = BootstrapFewShotWithRandomSearch(
            metric=self.comparison_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=5,
            max_rounds=1,
            num_candidate_programs=5,
        )

        compiled_classification = fewshot_optimizer.compile(ClassificationModule(), trainset=self.train_data)
        self.optimized_model = compiled_classification

        return compiled_classification
    
    def test_model(self, test_data):
        """Test the optimized model on the test data."""
        predictions = [self.optimized_model(prompt=example.prompt).label for example in test_data]
        true_labels = [example.label for example in test_data]

        return self.evaluate_model(predictions, true_labels)

    def save_model(self, model_path: str):
        """Save the optimized model."""
        self.optimized_model.save(model_path)
        print(f"Model saved to {model_path}")
