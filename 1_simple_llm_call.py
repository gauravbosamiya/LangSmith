from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

chain=prompt | llm | parser
response = chain.invoke({"question":"which was the best PM in india till today"})

print(response)