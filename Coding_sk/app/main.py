import torch
from fastapi import FastAPI
from app.db import *
from faiss_store import *
from utils.utils import *
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
import uvicorn
from fastapi.responses import JSONResponse
vector_store = FAQVectorstore(768)
csv_path = '/Users/salutethedawn/Desktop/编程用文件夹/编程用422/深度学习/个人项目/1_RAG(langchain+agent+sql)/data/Chinese-medical-dialogue-data-master/样例_内科5000-6000.csv'
@asynccontextmanager
async def lifespan(app:FastAPI):
    init_db()
    import_faq(csv_path,vector_store,num=10)
    yield
app = FastAPI(lifespan=lifespan)
@app.get('/')
def root():
    return {'message': '欢迎使用FAQ检索接口！'}
@app.get('/search')
def search(query):
    db=sessionLocal()
    result = vector_store.search(query,top_k=3)
    answer=[]
    for faq_id,score in result:
        faq = db.query(FAQ).filter(FAQ.id==faq_id).first()
        answer.append({
            #'question':faq.title,
            'answer':faq.answer,
            'score':score
        })
    db.close()
    return JSONResponse(content=answer)
def search_logic(query):
    db=sessionLocal()
    result = vector_store.search(query,top_k=3)
    print(result)
    answer=[]
    for faq_id,score in result:
        faq = db.query(FAQ).filter(FAQ.id == faq_id).first()
        print('faq_id',faq_id)
        print('FAQ_ID',FAQ.id)
        answer.append({
            #'question': faq.title,
            'answer': faq.answer,
            'score': score
        })
    db.close()
    return answer

if __name__ == '__main__':
    init_db()
    vector_store=FAQVectorstore(768)
    print("FAISS向量数:", vector_store.index.ntotal)
    #uvicorn.run('main:app', host='127.0.0.1', port=8001, reload=True)




