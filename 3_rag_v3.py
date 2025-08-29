from dotenv import load_dotenv
import os

from langsmith import traceable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"

PDF_PATH="islr.pdf"

@traceable(name="load_pdf")
def load_pdf(path:str):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )
    vs = FAISS.from_documents(splits, embeddings)
    return vs

@traceable(name="setup_pipeline",tags=["setup"])
def setup_pipeline(pdf_path:str,chunk_size=1000, chunk_overlap=200):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. if not found, say you don't know."),
    ("human","Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path:str, question:str):
    
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=200)
    
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 4})
    
    parallel = RunnableParallel({
        "context":retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    chain = parallel | prompt | llm | StrOutputParser()
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)


if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)