import re
import dspy
import random
import tiktoken
import pandas as pd
from sklearn.metrics import accuracy_score
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy import LM, Example, configure
from utilities import Config

class GPT4Model:
    """Setup for GPT-4 model configuration and initialization."""

    def __init__(self, api_key: str, proxy_url: str, model_name="gpt-4o-mini", temperature=0.2,
                 price_per_million_tokens_input=0.15, price_per_million_tokens_output=0.60):
        self.api_key = api_key
        self.proxy_url = proxy_url
        self.model_name = model_name
        self.temperature = temperature
        self.price_per_million_tokens_input = price_per_million_tokens_input
        self.price_per_million_tokens_output = price_per_million_tokens_output

        if not self.api_key:
            raise ValueError("API key not found. Please check your environment variables.")
        
        self.encoder = tiktoken.encoding_for_model(self.model_name)
        self.lm = LM(
            api_key=self.api_key,
            model=self.model_name,
            api_base=self.proxy_url,
            temperature=self.temperature
        )
        configure(lm=self.lm)


class DataPreparation:
    """Handles data loading and processing for training and testing."""

    def __init__(self, config: Config):
        self.train_size = config.train_size
        self.test_size = config.test_size
        self.seed = config.seed

    @staticmethod
    def create_example(row: pd.Series) -> Example:
        return Example(
            prompt=row["question"],
            completion=row["answer"],
            label=row["label"],
        ).with_inputs("prompt")

    def load_data(self, open_path: str, specific_path: str):
        open_data = pd.read_csv(open_path)
        specific_data = pd.read_csv(specific_path)

        open_examples = [self.create_example(row) for _, row in open_data.iterrows()]
        specific_examples = [self.create_example(row) for _, row in specific_data.iterrows()]

        final_data = open_examples + specific_examples
        random.seed(self.seed)
        random.shuffle(final_data)

        self.train_data = final_data[:self.train_size]
        self.test_data = final_data[:self.test_size]
        
        print(f"Train data: {len(self.train_data)}")
        print(f"Test data: {len(self.test_data)}")


class ClassificationSignature(dspy.Signature):
    """Classify if a text is specific for a domain or not. Target domain is law."""

    prompt = dspy.InputField(desc="The prompt to classify.")
    
    #explanation = dspy.OutputField(desc="Reasoning behind the classification.")
    label = dspy.OutputField(desc="1, if the input text is law domain, 0 otherwise.")
    

class ClassificationModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(ClassificationSignature)
        
    def forward(self, prompt: str) -> ClassificationSignature:
        prediction = self.prog(prompt=prompt)
        return prediction


class Trainer:
    """Handles model optimization using few-shot learning with BootstrapFewShotWithRandomSearch."""

    def __init__(self, module_class, train_data, evaluator: 'Evaluator'):
        self.module_class = module_class
        self.train_data = train_data
        self.evaluator = evaluator

    def comparison_metric(self, example, pred, trace=None) -> bool:
        """Metric function for comparing predicted label with actual label, using parse_answer for consistency."""
        parsed_example_label = self.evaluator.parse_answer(example.label)
        parsed_pred_label = self.evaluator.parse_answer(pred.label)
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

        # Compile the classification module with optimized settings
        compiled_classification = fewshot_optimizer.compile(self.module_class(), trainset=self.train_data)
        return compiled_classification


class Evaluator:
    """Model evaluation and cost calculation."""

    def __init__(self, gpt_model: GPT4Model):
        self.encoder = gpt_model.encoder
        self.price_per_million_tokens_input = gpt_model.price_per_million_tokens_input
        self.price_per_million_tokens_output = gpt_model.price_per_million_tokens_output

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

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def calculate_price(self, token_count: int) -> float:
        return (token_count / 1_000_000) * self.price_per_million_tokens_input
