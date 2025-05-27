
from abc import ABC, abstractmethod
import os
from openai import OpenAI
import dotenv
from transformers import pipeline

# Criando uma interface que obriga toda IA a ter um método chamado get_response
class LLMClient(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

#classe para usar o ChatGPT da OpenAI
class ChatGPT(LLMClient):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("KEY_API")) # API direta para facilitar o teste

    def get_response(self, prompt: str) -> str:
        try: 
            # Envia a pergunta para o modelo e pega a resposta
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

# Classe para usar um modelo do HuggingFace
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

# Classe que funciona como fábrica para criar os modelos de forma automatica
class LLMFactory:
    @staticmethod
    def create(model_name: str) -> LLMClient:
        if model_name == "chatgpt":
            return ChatGPT()
        elif model_name == "huggingface":
            return HuggingFaceLLM()
        else:
            raise ValueError(f"Modelo '{model_name}' não é suportado.")
