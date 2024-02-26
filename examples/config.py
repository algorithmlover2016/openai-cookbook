import os
import dotenv
dotenv.load_dotenv() # load environment variables from .env file

# Azure OpenAI
api_key = os.getenv("OPENAI_API_KEY_GPT4")
api_version = os.getenv("API_VERSION", "2023-07-01-preview")
azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://kslabopenai2.openai.azure.com")
GPT_MODEL = os.getenv("AZURE_GPT_MODEL", "gpt-4-32k")

azure_embedding_model = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-openai")