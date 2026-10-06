# 🤖 LLM-and-ODC
Avaliação de grandes modelos de linguagem (LLMs) para classificar correções de vulnerabilidade de software usando classificação ortogonal de defeitos (ODC)

## 📚 Bibliotecas necessárias de Python
Na bash, faz:
  `pip install -r requirements.txt`

Este comando instalará todas as bibliotecas necessárias nas respetivas versões.

## 🤖 Requisitos para usar modelos LLM
Este projeto utiliza [Ollama](https://ollama.com) para correr modelos de linguagem localmente.

### 🛠️ Instalação do Ollama
1. Vai a [https://ollama.com/download](https://ollama.com/download) e instala o Ollama para o teu sistema operativo.
2. Para funcionar, a aplicação ollama tem de estar aberta.
3. Instala os modelos das LLM ao teu gosto utilizando o comando: `ollama pull "modelo"`

   Ex: `ollama pull gemma3`

## ▶️ Execução
Na bash, faz: `python main.py`
