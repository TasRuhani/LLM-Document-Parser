import os
from dotenv import load_dotenv
load_dotenv()

def get_openrouter_keys():
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    return OPENROUTER_API_KEY, OPENROUTER_MODEL