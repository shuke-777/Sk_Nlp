from langchain.agents import initialize_agent,AgentType
from app.agent_tools import *
from langchain_openai import ChatOpenAI
from app.faiss_store import *
faq_store = FAQVectorstore(768)
faq_store.load()
api_key = 'bf339b5cb24e450d9514d70967887ec9.FzdFGKaisWelqvhg'
model = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key="sk-4b3eff7415da4c88be44ad4f642dc425",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True
)
agent = initialize_agent(
    tools,
    model,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
query='糖尿病还会进行遗传吗？'
response = agent.invoke(query)
print(response)

