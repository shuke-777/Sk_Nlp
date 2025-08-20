from langchain.tools import Tool
from app.db import *
from app.faiss_store import *
vector_store = FAQVectorstore(768)
def faq_search_tool(query):
    result = vector_store.search(query,top_k=3)
    db = sessionLocal()
    answer = []
    for faq_id,score in result:
        faq = db.query(FAQ).filter(FAQ.id==faq_id).first()
        answer.append(f'Q:{faq.question},A:{faq.answer},score:{score:.4f}')
    db.close()
    return '\n'.join(answer) if answer else 'None answer'
def sql_query_tool(query):
    db=sessionLocal()
    #contain是模糊匹配，不要求一次不差，只要里面有部分内容即可
    faqs = db.query(FAQ).filter(FAQ.ask.contains(query)).all()
    if faqs:
        return '\n'.join([f'Q:{faq.question}\nA:{faq.answer}'for faq in faqs])
    return 'SQL没有找到相关的FAQ'

tools=[
    Tool(
        name = 'FAQ_Search',
        func=faq_search_tool,
        description='用于回答医学相关的FAQ问题，输入用户问题，返回最匹配的问答'
    ),
    Tool(
        name = 'SQL_Query',
        func=sql_query_tool,
        description='先用这个方法查找，用于查询本地SQL数据库中的问答条目，回答医学问题，'

    )
]