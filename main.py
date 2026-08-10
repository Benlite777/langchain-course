from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langfuse import get_client
from langfuse.langchain import CallbackHandler


def main():
    langfuse = get_client()
    langfuse_handler = CallbackHandler()

    information = """Virat Kohli is an Indian international cricketer and former all-format captain of the Indian national cricket team. He is a right-handed batter and an occasional right-arm medium-pace bowler. Considered one of the greatest batters in limited-overs cricket, he has been acclaimed for his batting skills and records."""

    summary_prompt_template = PromptTemplate.from_template(
        """
        Given the following information {information} about a person, I want you to create:
        1. A short summary of the person
        2. Two interesting facts about them
        """
    )

    llm = ChatOllama(model="gemma4:12b", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(
        {"information": information},
        config={"callbacks": [langfuse_handler]},
    )
    print(response.content)

    langfuse.flush()


if __name__ == "__main__":
    main()