
from abc import ABC, abstractmethod
import os
from openai import OpenAI

from transformers import pipeline

# Interface base
class LLMClient(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

# Implementação do ChatGPT via OpenAI
class ChatGPT(LLMClient):
    def __init__(self):
        self.client = OpenAI(api_key="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")

    def get_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro ao acessar a API do ChatGPT: {e}"

# Implementação do HuggingFace QA com contexto fixo
class HuggingFaceLLM(LLMClient):
    def __init__(self):
        self.qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.context = (
            "O Brasil foi descoberto pelos portugueses em 1500. "
            "Sua independência foi proclamada em 1822. "
            "Hoje é um dos maiores países da América do Sul."
        )

    def get_response(self, prompt: str) -> str:
        try:
            result = self.qa(question=prompt, context=self.context)
            return result["answer"]
        except Exception as e:
            return f"Erro ao responder com HuggingFace: {e}"

# Fábrica de modelos
class LLMFactory:
    @staticmethod
    def create(model_name: str) -> LLMClient:
        if model_name == "chatgpt":
            return ChatGPT()
        elif model_name == "huggingface":
            return HuggingFaceLLM()
        else:
            raise ValueError(f"Modelo '{model_name}' não é suportado.")
