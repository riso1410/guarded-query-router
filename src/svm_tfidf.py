import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from utilities import preprocess_data

class SVMClassifier:
    """SVM classifier with TF-IDF, cross-validation, and evaluation functionality."""
    def __init__(self, config):
        # Initialize configuration settings
        self.C = config.get('C')
        self.model = SVC(C=self.C)
        self.embedding_model = config.get('embedding_model')
        self.model_name = "SVM_TFIDF"

    def train(self, X_train, y_train):
        """Train the SVM model on the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, text: str):
        """Predict the label of a single text."""
        vectorized_text = list(self.embedding_model.embed(text))
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
        """Save the SVM model."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load the SVM model."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
