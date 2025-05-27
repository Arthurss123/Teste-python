from factory import ChatGPT, HuggingFaceLLM
from commands import QueryLLMsCommand
from strategy import EnhancedEvaluationStrategy
from observer import ConsoleObserver

def main():


    # Um dicionario com os modelos que usamos
    llms = {
        "ChatGPT": ChatGPT(),
        "HuggingFace": HuggingFaceLLM()
    }

    # Criando o observador
    console_observer = ConsoleObserver()

    # Cria a estratégia de avaliação e conecta o observador nela
    strategy = EnhancedEvaluationStrategy()
    strategy.add_observer(console_observer)

    #Loop principal do menu
    while True:
        print("1. Usar a LLM")
        print("2. Sair")
        choice = input("Digite sua opção:  ")

        if choice == "1":
            user_prompt = input("\nFaça sua pergunta ")

            command = QueryLLMsCommand(llms, user_prompt)
            responses = command.execute()

            print("\nRespostas dos LLMs:")
            for name, response in responses.items():
                print(f"{name}: {response}")

            # Avaliando as respostas e notificando o observador
            strategy.evaluate(responses, user_prompt)


        elif choice == "2":
            print("Saindo...")
            break

        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()