import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

import logging
logging.basicConfig(level=logging.INFO)

prompts_template = ("You are text classification tool. Based on the user input, "
            "return one of classes: 'Physics', 'Chemistry', 'Maths', 'Biology' "
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