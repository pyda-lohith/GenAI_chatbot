from langchain import PromptTemplate
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain  
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from src.prompt import * 
import os 
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

pc = Pinecone(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV,
    REGION= 'us-east-1'
)


index_name="chatbotmedicalexplore"

docsearch= pc.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)




@app.route("/")
def index():
    return render_template('chat.html')

​
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)