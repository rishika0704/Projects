import os
import warnings
from pathlib import Path as p
from pprint import pprint
from dotenv import load_dotenv

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

class RAGChatbot:
    def __init__(self):
        # Placeholder for the generative AI model
        self.model = None

    def initialize_model(self, api_key, model_name="gemini-pro"):
        """Initialize the Google Generative AI model."""
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        print("Model initialized successfully.")

    def process_pdf(self, pdf_path):
        """Load and split the PDF into smaller text chunks."""
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context = "\n\n".join(str(p.page_content) for p in pages)
        texts = text_splitter.split_text(context)
        return texts

    def create_retriever(self, texts, embeddings_model_name):
        """Create a retriever using Chroma vector store."""
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embeddings_model_name, 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        retriever = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_data").as_retriever(search_kwargs={"k": 5})
        return retriever

    def generate_response(self, prompt, retriever):
        """Generate a response using the retrieval-based QA chain."""
        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain({"query": prompt})
        return response['result']

# Instantiate the chatbot and initialize the model
if __name__ == "__main__":
    chatbot = RAGChatbot()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    chatbot.initialize_model(api_key=GOOGLE_API_KEY)
