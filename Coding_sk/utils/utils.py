import os
import sys
from app.db import *
from app.faiss_store import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
def import_faq(csv_path,vector_store,num=None):
    db=sessionLocal()
    with open(csv_path,'r',encoding='gbk') as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(reader):
            if i >=num:
                break
            faq = FAQ(department=row['department'],
                      title=row['title'],
                      ask=row['ask'],
                      answer=row['answer'])
            db.add(faq)
            print(f'第{i}条数据已经添加到数据库中')
            db.commit()
            db.refresh(faq)
            vector_store.add(faq.id,row['ask'])
        print(f"总共处理了 {i} 条数据")
        db.close()


if __name__ == '__main__':
    init_db()
    vector_store= FAQVectorstore(dim=768)
    csv_path = '/Users/salutethedawn/Desktop/编程用文件夹/编程用422/深度学习/个人项目/1_RAG(langchain+agent+sql)/data/Chinese-medical-dialogue-data-master/样例_内科5000-6000.csv'
    import_faq(csv_path,vector_store,num=20)