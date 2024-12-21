from transformers import pipeline

class NLI:
    def __init__(self, premise):
        self.nli = pipeline("zero-shot-classification")
        self.premise = str()
        self.candidate_labels = [premise, "other"]

    def predict(self, premise, hypothesis, candidate_labels):
        classifier = pipeline("zero-shot-classification", model='cointegrated/rubert-base-cased-nli-threeway')
        res = classifier(hypothesis, candidate_labels)

        print(res)
        print(f'Text: {hypothesis}')
        print(f'Labels: {res["labels"][0]}')
        return self.nli(premise, hypothesis, candidate_labels)
    