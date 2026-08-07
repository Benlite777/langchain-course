from dotenv import load_dotenv
import os
def main():
    load_dotenv()
    print("Hello from langchain-course!")
    
    print(os.getenv("GOOGLE_API_KEY"))

if __name__ == "__main__":
    main()
