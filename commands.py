class QueryLLMsCommand:
    def __init__(self, llms, prompt):
        # Inicializa o comando com um dicionário de LLMs e um prompt
        self.llms = llms
        self.prompt = prompt

    def execute(self):

        responses = {}

        # Itera sobre cada LLM e obtém a resposta com o prompt fornecido
        for name, llm in self.llms.items():
            responses[name] = llm.get_response(self.prompt)
        return responses