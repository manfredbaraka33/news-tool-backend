import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "news_articles"
CHROMA_DIR = "./chroma_store"
