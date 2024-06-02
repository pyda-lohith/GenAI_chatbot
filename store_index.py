from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = "default"

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV,
    REGION= 'us-east-1'
)


index_name="chatbotmedicalexplore"
docsearch= PineconeVectorStore.from_documents(text_chunks,embeddings,index_name=index_name)
