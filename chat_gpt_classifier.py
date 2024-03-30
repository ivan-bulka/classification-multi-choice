import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from text_classifier import XGBTextClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import logging
logging.basicConfig(level=logging.INFO)

prompts_template = ("You are text classification tool. Based on the user input, "
            "return one of classes: Physics, Chemistry, Maths, Biology "
            "the most closer to the user message. "
            "The output should be a single word."
            "User message: {message}")


class GPTTextClassifier:
    def __init__(self, template=prompts_template, model_name="gpt-4"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Open AI key should be configured")
        self.template = template
        self.model = ChatOpenAI(model=model_name)
        self.output_parser = StrOutputParser()

    def prompt(self, message) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(self.template.format(message=message))

    def classify(self, message: str) -> str:
        chain = (
                {"message": RunnablePassthrough()}
                | self.prompt(message)
                | self.model
                | self.output_parser
        )
        response = chain.invoke(message)
        return response


class GPTTextEvaluator(XGBTextClassifier):

    def __init__(self, dataset, embedding_column='embeddings',
                 predict_column='class', test_size=0.25, random_state=1,
                 input_column='input'):
        super().__init__(dataset, embedding_column, predict_column, test_size,
                         random_state)
        self.input_column = input_column
        self.responses = []

    def prepare_data(self):
        X = self.dataset[self.input_column].values.tolist()
        Y = self.dataset[self.predict_column]
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(X, Y, test_size=self.test_size,
                               random_state=self.random_state)
        self.y_train = self.le.fit_transform(self.y_train)

    def predict(self):
        return [GPTTextClassifier().classify(inp) for inp in self.X_test]

    def evaluate(self, responses):
        report = classification_report(self.y_test, responses,
                                       target_names=self.le.classes_)
        logging.info(report)
