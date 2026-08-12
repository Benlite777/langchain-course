from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler
from langfuse import get_client

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

llm = ChatOllama(model="gemma4:12b")


def main():
    pass


if __name__ == "__main__":
    main()
