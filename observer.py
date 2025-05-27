from abc import ABC, abstractmethod

# Classe base (abstrata) que define como um Observer deve funcionar
class Observer(ABC):
    @abstractmethod
    def update(self, best_response: str, explanation: str) -> None:
        pass

# Observer que mostra no terminal qual foi a melhor resposta escolhida
class ConsoleObserver(Observer):
    def update(self, best_response: str, explanation: str) -> None:
        print("\n=== Atualização da Resposta ===")
        print(f"Melhor resposta escolhida: {best_response}")
        print(f"Explicação: {explanation}")