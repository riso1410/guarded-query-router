from typing import List, Tuple, Optional, Any
import re
import time

import dspy
import pandas as pd
from dspy import LM, Example, configure
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from tqdm import tqdm

from prompt_classifier.metrics import calculate_cost


class LlmClassifier:
    """Large Language Model classifier using DSPy framework.
    
    Handles training, optimization and prediction for domain classification tasks.
    
    Attributes:
        api_key (str): API key for LLM service
        api_base (str): Base URL for API
        domain (str): Target domain for classification
        model_name (str): Name of the LLM model
        train_data (List[Example]): Training examples
        test_data (List[Example]): Test examples
        cost (float): Accumulated API cost
        lm: Language model instance
        optimized_model: Optimized classification module
    """

    def __init__(self, model_name: str, train_data: pd.DataFrame, 
                 test_data: pd.DataFrame, domain: str,
                 api_base: str, api_key: str) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.domain = domain
        self.model_name = model_name

        self.train_data = [
            dspy.Example(
                domain=str(domain),
                prompt=str(row['prompt']),
                label=str(int(row['label']))
            ).with_inputs("domain", "prompt")
            for _, row in train_data.iterrows()
        ]

        self.test_data = [
            dspy.Example(
                domain=str(domain),
                prompt=str(row['prompt']),
                label=str(int(row['label']))
            ).with_inputs("domain", "prompt")
            for _, row in test_data.iterrows()
        ]
        self.cost = 0.0

        self.lm = LM(
            api_key=self.api_key,
            model=self.model_name,
            api_base=self.api_base,
            temperature=0.0,
        )
        configure(lm=self.lm)

    @staticmethod
    def create_example(domain: str, row: pd.Series) -> Example:
        """Create a DSPy example from input data.
        
        Args:
            domain (str): Target domain
            row (pd.Series): Input data row
            
        Returns:
            Example: DSPy example instance
        """
        return Example(
            domain=domain,
            prompt=row['prompt'],
            label=row['label'],
        ).with_inputs("prompt","domain")

    @staticmethod
    def parse_answer(answer: Any) -> bool:
        """Parse model answers into binary format.
        
        Args:
            answer: Raw model output
            
        Returns:
            bool: Parsed binary prediction
        """
        answer = answer.strip()
        if isinstance(answer, str) and re.search(r"[01]", answer):
            return bool(int(re.search(r"[01]", answer).group()))
        elif isinstance(answer, int) and answer in [0, 1] or (answer in ["True", "False"]):
            return bool(answer)
        else:
            print(f"Unexpected non-binary label found: {answer}")
            return False

    def comparison_metric(self, example: Example, pred: Example, 
                         trace: Optional[Any] = None) -> bool:
        """Metric function for comparing predicted label with actual label, using parse_answer for consistency."""
        parsed_example = self.parse_answer(example.label)
        parsed_pred = self.parse_answer(pred.label)
        return parsed_example == parsed_pred

    def optimize_model(self) -> dspy.Module:
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

    def predict(self) -> Tuple[List[int], List[int], List[int]]:
        """Predict labels for test data.
        
        Returns:
            Tuple containing:
                - List[int]: Predictions
                - List[int]: Actual labels 
                - List[int]: Prediction times in nanoseconds
        """
        predictions = []
        actuals = []
        prediction_times = []

        for example in tqdm(self.test_data, total=len(self.test_data)):
            self.cost += calculate_cost(example.prompt, input=True)
            start_time = time.perf_counter_ns()
            pred = self.optimized_model(prompt=example.prompt, domain=example.domain)
            end_time = time.perf_counter_ns()
            prediction_times.append(end_time - start_time)
            self.cost += calculate_cost(pred.label, input=False)
            predictions.append(int(self.parse_answer(pred.label)))
            actuals.append(int(self.parse_answer(example.label)))

        return predictions, actuals, prediction_times

    def predict_single(self, prompt: str) -> Tuple[int, int]:
        """Predict the label for a single prompt.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            Tuple containing:
                - int: Prediction
                - int: Prediction time in nanoseconds
        """
        self.cost += calculate_cost(prompt, input=True)
        start_time = time.perf_counter_ns()
        pred = self.optimized_model(prompt=prompt, domain=self.domain)
        end_time = time.perf_counter_ns()
        self.cost += calculate_cost(pred.label, input=False)
        prediction_time = end_time - start_time
        return int(self.parse_answer(pred.label)), prediction_time
    

    def save_model(self, model_path: str) -> None:
        """Save the optimized model."""
        self.optimized_model.save(model_path)

    def load_model(self, model_path: str) -> None:
        """Load the optimized model."""
        self.optimized_model = ClassificationModule()
        self.optimized_model.load(model_path)


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
        """Forward method for classification module.
        
        Args:
            prompt (str): Input prompt
            domain (str): Target domain
            
        Returns:
            ClassificationSignature: Prediction signature
        """
        try:
            prediction = self.prog(prompt=prompt, domain=domain)

        except Exception as e:
            print(f"An error occurred while classifying the prompt: {e}\nPrompt: {prompt}")
            prediction = ClassificationSignature(label=0)

        return prediction
