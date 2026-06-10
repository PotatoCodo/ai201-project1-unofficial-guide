import os
from dotenv import load_dotenv

load_dotenv()

#LLM
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#N_RESULTS
N_RESULTS = 4

#Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#Vector store
CHROMA_COLLECTION = "unofficial"
CHROMA_PATH = "./chroma_db"

#Documents
DOC_PATH = "./documents"