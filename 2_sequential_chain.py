from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

os.environ["LANGCHAIN_PROJECT"] = 'Sequential LLM App'

load_dotenv()
llm1 = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

llm2 = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)

prompt1 = PromptTemplate(
    template='Generate detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()

chain= prompt1 | llm1 | parser | prompt2 | llm2 | parser

config = {
    'run_name':'sequential chain',
    'tags':['llm app','report generation','summarization'],
    'metadata':{'model1':'openai/gpt-oss-20b', 'model1_temp':0.7,'parser':'stringoutputparser'}
}
response = chain.invoke({"topic":"Finance market in india"}, config=config)

print(response)