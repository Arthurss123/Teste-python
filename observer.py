from abc import ABC, abstractmethod

# Classe abstrata que define o contrato para os Observers
class Observer(ABC):
    @abstractmethod
    def update(self, best_response: str, explanation: str) -> None:
        pass


# Implementação de um Observer que imprime as atualizações no console
class ConsoleObserver(Observer):
    def update(self, best_response: str, explanation: str) -> None:
        print("\n=== Atualização da Resposta ===")
        print(f"Melhor resposta escolhida: {best_response}")
        print(f"Explicação: {explanation}")