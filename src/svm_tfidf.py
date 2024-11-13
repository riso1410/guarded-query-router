import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
from sklearn.metrics import make_scorer


class SVMClassifier:
    """SVM classifier with TF-IDF, cross-validation, and evaluation functionality."""
    def __init__(self, config):
        # Initialize configuration settings
        self.C = config.get('C')
        self.train_size = config.get('train_size')
        self.test_size = config.get('test_size')
        self.seed = config.get('seed')
        self.model = SVC(C=self.C)
        self.vectorizer = TfidfVectorizer()
        self.model_name = "SVM_TFIDF"

    def prepare_data(self, open_path: str, specific_path: str):
        """Load, shuffle, split, and vectorize data for training and testing."""
        data = pd.concat([pd.read_csv(open_path), pd.read_csv(specific_path)], ignore_index=True)
        data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)  # Shuffle the data

        # Split into train and test data
        train_data, test_data = data[:self.train_size], data[self.train_size:self.train_size + self.test_size]
        X_train = self.vectorizer.fit_transform(train_data['question'])
        X_test = self.vectorizer.transform(test_data['question'])
        y_train, y_test = train_data['label'], test_data['label']

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the SVM model on the training data."""
        self.model.fit(X_train, y_train)

    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation and return average accuracy, precision, and recall scores."""
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='binary', zero_division=1),
            'recall': make_scorer(recall_score, average='binary', zero_division=1)
        }

        # Stratified cross-validation to maintain class balance across folds
        skf = StratifiedKFold(n_splits=cv)
        cv_results = cross_validate(self.model, X, y, cv=skf, scoring=scoring)
        # Calculate and return average scores
        return {
            'average_accuracy': cv_results['test_accuracy'].mean(),
            'average_precision': cv_results['test_precision'].mean(),
            'average_recall': cv_results['test_recall'].mean()
        }

    def evaluate(self, X_test, y_test):
        """Evaluate the model with accuracy, precision, recall, and a classification report."""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=1)
        recall = recall_score(y_test, predictions, zero_division=1)

        # Display the classification report
        print("Classification Report:\n", classification_report(y_test, predictions, zero_division=1))
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

    def predict(self, text: str):
        """Predict the label of a single text."""
        vectorized_text = self.vectorizer.transform([text])
        return self.model.predict(vectorized_text)[0]
    
    def save_model(self, model_path: str):
        """Save the SVM model and TF-IDF vectorizer."""
        joblib.dump((self.model, self.vectorizer), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the SVM model and TF-IDF vectorizer."""
        self.model, self.vectorizer = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
