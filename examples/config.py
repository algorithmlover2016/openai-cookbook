import os
import logging
import dotenv
dotenv.load_dotenv() # load environment variables from .env file
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s %(process)d %(pathname)s:%(lineno)d %(message)s]',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Azure OpenAI
api_key = os.getenv("OPENAI_API_KEY_GPT4", "<your OpenAI API key if not set as env var>")
api_version = os.getenv("API_VERSION", "2023-07-01-preview")
azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://kslabopenai2.openai.azure.com")
GPT_MODEL = os.getenv("AZURE_GPT_MODEL", "gpt-4-32k")

azure_embedding_model = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-openai")