import numpy as np
import textstat
from sentence_transformers import SentenceTransformer, util
from factory import ChatGPT
from abc import ABC, abstractmethod


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

class EnhancedEvaluationStrategy:
    def __init__(self):
        self.observers = []  # Lista de observadores

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def remove_observer(self, observer) -> None:
        self.observers.remove(observer)

    def notify_observers(self, best_response, explanation) -> None:
        for observer in self.observers:
            observer.update(best_response, explanation)

    def evaluate(self, responses, prompt):
   
        def is_valid_response(response):
            if "no, no, no" in response.lower():
                return False
            if len(response.split()) < 5 or response.lower() == 'não sei':
                return False
            return True
        
        valid_responses = {k: v for k, v in responses.items() if is_valid_response(v)}
        
        if not valid_responses:
            return {"best_response": "Nenhuma resposta válida encontrada.", "best_model": "N/A", "explanation": "Todas as respostas foram consideradas inválidas."}

        #  Similaridade semântica
        similarities = np.array([semantic_similarity(prompt, response) for response in valid_responses.values()])

        # Legibilidade (Flesch-Kincaid readability)
        readability_scores = np.array([textstat.flesch_reading_ease(response) for response in valid_responses.values()])
        readability_scores = np.clip(readability_scores, 0, 100)  # Normalização

      
        scores = (
            0.7 * similarities +  
            0.3 * (readability_scores / 100)  
        )

        best_index = np.argmax(scores)
        best_response = list(valid_responses.values())[best_index]
        best_model = list(valid_responses.keys())[best_index]

        explanation = (
            f"A resposta foi escolhida com base em:\n"
            f"- Similaridade semântica: {similarities[best_index]:.2f}\n"
            f"- Legibilidade (Flesch-Kincaid): {readability_scores[best_index]:.2f}\n"
        )

        # Notificar todos os observadores
        self.notify_observers(best_response, explanation)

        return {
            "best_response": best_response,
            "best_model": best_model,
            "explanation": explanation
        }