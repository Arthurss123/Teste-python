# LLM

Este é um projeto em Python que compara respostas geradas por dois modelos de linguagem (LLMs): o ChatGPT da OpenAI e um modelo da HuggingFace. A aplicação envia perguntas para os dois modelos, avalia as respostas com base em critérios como similaridade semântica e legibilidade, e exibe a melhor resposta ao usuário.

##  Tecnologias usadas

- Python 3.10+
- [OpenAI API](https://platform.openai.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Sentence Transformers
- textstat
- Padrões de Projeto: Factory, Command, Strategy e Observer

##  Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/llm-evaluator.git
cd llm-evaluator
```
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Defina sua chave da OpenAI:
```bash
# API
set OPENAI_API_KEY=sua-chave-aqui
```

## Como usar

Execute o arquivo principal:

```bash
python main.py

```

2. Use o Menu do terminal
- "1" Para fazer uma pergunta
- "2" Para sair do programa

### Funcionalidades:
- Escolher uma pergunta para enviar aos modelos.
- Ver todas as respostas.
- Ver qual foi considerada a melhor com base em:
  - Similaridade semântica com a pergunta.
  - Facilidade de leitura (legibilidade).
