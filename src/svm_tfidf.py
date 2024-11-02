import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

        if 'question' not in open_data.columns or 'label' not in open_data.columns:
            raise ValueError("Data must contain 'question' and 'label' columns.")

        if 'question' not in specific_data.columns or 'label' not in specific_data.columns:
            raise ValueError("Data must contain 'question' and 'label' columns.")

        combined_data = pd.concat([open_data, specific_data], ignore_index=True)
        return combined_data

    def prepare_data(self, data):
        """Shuffle, split, and vectorize the data using TF-IDF."""
        # Shuffle data
        data = data.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

        # Split data
        train_data = data[:self.config.train_size]
        test_data = data[self.config.train_size:self.config.train_size + self.config.test_size]

        # TF-IDF vectorization
        X_train = self.vectorizer.fit_transform(train_data['question'])
        X_test = self.vectorizer.transform(test_data['question'])

        y_train = train_data['label']
        y_test = test_data['label']

        return X_train, X_test, y_train, y_test


class SVMClassifier:
    """SVM classifier using TF-IDF vectorization."""

    def __init__(self, config: Config):
        self.config = config
        self.model = SVC(kernel='linear', random_state=self.config.seed)
        self.model_name = "SVM_TFIDF"

    def train(self, X_train, y_train):
        """Train the SVM model on the training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return accuracy and a classification report."""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    
    def save_model(self, model_path: str):
        """Save the SVM model and TF-IDF vectorizer to disk."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the SVM model and TF-IDF vectorizer from disk."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
