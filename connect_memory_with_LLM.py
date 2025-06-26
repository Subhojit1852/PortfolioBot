import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import certifi
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

os.environ['SSL_CERT_FILE'] = certifi.where()

# Load the .env file
load_dotenv()

# Set the OpenRouter API key and endpoint
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# === Step 1: Load FAISS vectorstore ===
def load_vectorstore(path="vectorstore/db_faiss"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# === Step 2: Define prompt template ===
def get_prompt_template():
    return ChatPromptTemplate.from_template(
        """You are a helpful assistant trained to answer questions from resumes.

Context (from resume):
{context}

Question:
{question}

Answer as clearly and briefly as possible based on the resume:
"""
    )

# === Step 3: Create QA chain ===
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = get_prompt_template()
    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        temperature=0.3
    )

   
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
qa_chain = create_qa_chain(load_vectorstore())

# === Step 4: Run the system ===
# def main():
#     vectorstore = load_vectorstore()
#     qa_chain = create_qa_chain(vectorstore)
#     question = input("Ask something about the resume: ")
#     answer = qa_chain.run(question)
#     print("\n🤖 Answer:", answer)
#     return qa_chain

# if __name__ == "__main__":
#     main()