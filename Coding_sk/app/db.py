from sqlalchemy import Column,create_engine,Integer,String
from sqlalchemy.orm import sessionmaker,declarative_base
base = declarative_base()
url = 'sqlite:///./qb_sk.db'
db=create_engine(url=url)

sessionLocal=sessionmaker(bind=db)
class FAQ(base):
    __tablename__='faq'
    id = Column(Integer,primary_key=True,index=True)
    department = Column(String)
    title = Column(String)
    ask = Column(String)
    answer = Column(String)

def init_db():
    base.metadata.create_all(db)

if __name__ == '__main__':
    init_db()
    FAQ1=FAQ()
    print(FAQ1.id)
