import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from utilities import Config
import joblib


class DataPreparationSVM:
    """Handles data loading, processing, and splitting with TF-IDF vectorization."""

    def __init__(self, config: Config):
        self.config = config
        self.vectorizer = TfidfVectorizer()

    def load_data(self, open_path: str, specific_path: str):
        """Load and combine data from two CSV files."""
        open_data = pd.read_csv(open_path)
        specific_data = pd.read_csv(specific_path)
        combined_data = pd.concat([open_data, specific_data], ignore_index=True)
        return combined_data

    def prepare_data(self, data):
        """Shuffle, split, and vectorize the data using TF-IDF."""
        data = data.sample(frac=1, random_state=self.config.seed).reset_index(drop=True) # Shuffle the data

        train_data = data[:self.config.train_size]
        test_data = data[self.config.train_size:self.config.train_size + self.config.test_size]

        print(f'Train size: {len(train_data)}')
        print(f'Test size: {len(test_data)}')

        X_train = self.vectorizer.fit_transform(train_data['question'])
        X_test = self.vectorizer.transform(test_data['question'])

        y_train = train_data['label']
        y_test = test_data['label']

        return X_train, X_test, y_train, y_test


class SVMClassifier:
    """SVM classifier using TF-IDF vectorization."""

    def __init__(self, config: Config):
        self.config = config
        self.model = SVC(C=0.1, random_state=self.config.seed)
        self.model_name = "SVM_TFIDF"

    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation and return scores."""
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }
        scores = cross_validate(self.model, X, y, cv=cv, scoring=scoring)
        return scores

    def train(self, X_train, y_train):
        """Train the SVM model on the training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return accuracy and a classification report."""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        return accuracy, precision, recall
    
    def save_model(self, model_path: str):
        """Save the SVM model and TF-IDF vectorizer to disk."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the SVM model and TF-IDF vectorizer from disk."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
