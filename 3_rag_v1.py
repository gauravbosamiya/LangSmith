from dotenv import load_dotenv
import os
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

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)
vs = FAISS.from_documents(splits, embeddings)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. if not found, say you don't know."),
    ("human","Question: {question}\n\nContext:\n{context}")
])

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

print("PDF RAG ready. Ask a question (or CTRL+C to exit).")
q = input("\nQ:")
ans = chain.invoke(q.strip())
print("\nA:",ans)
