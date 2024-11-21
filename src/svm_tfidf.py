import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

class SVMClassifier:
    """SVM classifier with TF-IDF, cross-validation, and evaluation functionality."""
    def __init__(self, config):
        # Initialize configuration settings
        self.C = config.get('C')
        self.train_size = config.get('train_size')
        self.test_size = config.get('test_size')
        self.seed = config.get('seed')
        self.model = SVC(C=self.C)
        self.vectorizer = config.get('embedding')
        self.model_name = "SVM_TFIDF"

    def prepare_data(self, open_path: str, specific_path: str):
        """Load, shuffle, split, and vectorize data for training and testing."""
        data = pd.concat([pd.read_csv(open_path), pd.read_csv(specific_path)], ignore_index=True)
        data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)  # Shuffle the data

        # Split into train and test data
        train_data, test_data = data[:self.train_size], data[self.train_size:self.train_size + self.test_size]
        X_train = list(self.vectorizer.embed(train_data['question']))
        X_test = list(self.vectorizer.embed(test_data['question']))
        y_train, y_test = train_data['label'], test_data['label']

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the SVM model on the training data."""
        self.model.fit(X_train, y_train)

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation and return the mean score."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores: {scores}")
        print(f"Mean cross-validation score: {scores.mean()}")
        return scores.mean()
    
    def predict(self, text: str):
        """Predict the label of a single text."""
        vectorized_text = list(self.vectorizer.embed(text))
        return self.model.predict(vectorized_text)

    def evaluate(self, X, y):
        """Evaluate the model with accuracy, precision, recall, and a classification report."""
        predictions = self.model.predict(X)
        matrix = metrics.confusion_matrix(y, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [0, 1])
        cm_display.plot()
        plt.show() 

        f1 = metrics.f1_score(y, predictions)
        accuracy = metrics.accuracy_score(y, predictions)
        recall = metrics.recall_score(y, predictions)
        precision = metrics.precision_score(y, predictions)
        return f1, accuracy, recall, precision
    
    def save_model(self, model_path: str):
        """Save the SVM model and TF-IDF vectorizer."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the SVM model and TF-IDF vectorizer."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
