from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

import logging
logging.basicConfig(level=logging.INFO)


class XGBTextClassifier:
    """
    Train and evaluate classification model with XGBoost model
    """
    def __init__(self, dataset, embedding_column='embeddings', predict_column='class',
                 test_size=0.25, random_state=1):
        self.dataset = dataset
        self.embedding_column = embedding_column
        self.predict_column = predict_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = XGBClassifier()
        self.le = LabelEncoder()

    def prepare_data(self):
        X = self.dataset[self.embedding_column].values.tolist()
        Y = self.dataset[self.predict_column]
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(X, Y, test_size=self.test_size,
                               random_state=self.random_state)
        self.y_train = self.le.fit_transform(self.y_train)

    def train(self):
        # A dataset is not balanced. It might a sense to use this info to train the model
        self.model.fit(self.X_train, self.y_train)

    def predict(self) -> list:
        # Current implementation is simple. It might have
        # (at least for the inference to use .predict_proba method here)
        y_pred = self.model.predict(self.X_test)
        y_pred = self.le.inverse_transform(y_pred)
        return y_pred

    def evaluate(self, y_pred) -> None:
        report = classification_report(self.y_test, y_pred,
                                       target_names=self.le.classes_)
        logging.info(report)

    def save_model(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path: str) -> None:
        with open(path, 'rb') as file:
            self.model = pickle.load(file)

    def save_label_encoder(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self.le, file)

    def load_label_encoder(self, path: str) -> None:
        with open(path, 'rb') as file:
            self.le = pickle.load(file)