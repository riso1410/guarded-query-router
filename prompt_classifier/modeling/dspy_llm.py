import re
import time

import dspy
import pandas as pd
from dspy import LM, Example, configure
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from tqdm import tqdm

from prompt_classifier.metrics import calculate_cost


class LlmClassifier:
    def __init__(self, model_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                 domain: str, api_base: str, api_key: str) -> None:
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
        return Example(
            domain=domain,
            prompt=row['prompt'],
            label=row['label'],
        ).with_inputs("prompt","domain")

    @staticmethod
    def parse_answer(answer) -> bool:
        """Parse answers into a consistent binary format."""
        answer = answer.strip()
        if isinstance(answer, str) and re.search(r"[01]", answer):
            return bool(int(re.search(r"[01]", answer).group()))
        elif isinstance(answer, int) and answer in [0, 1] or (answer in ["True", "False"]):
            return bool(answer)
        else:
            print(f"Unexpected non-binary label found: {answer}")
            return False

    def comparison_metric(self, example, pred, trace = None) -> bool:
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

    def predict(self) -> bool:
        """Predict the label for a given prompt."""
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
        try:
            prediction = self.prog(prompt=prompt, domain=domain)

        except Exception as e:
            print(f"An error occurred while classifying the prompt: {e}\nPrompt: {prompt}")
            prediction = ClassificationSignature(label=0)

        return prediction
