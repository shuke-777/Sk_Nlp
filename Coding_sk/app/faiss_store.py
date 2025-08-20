import faiss
import torch
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
save_path='/Users/salutethedawn/Desktop/编程用文件夹/编程用422/深度学习/个人项目/1_RAG(langchain+agent+sql)/Coding_sk/save'
class FAQVectorstore:
    def __init__(self,dim):
        self.embed_model = SentenceTransformer('/Users/salutethedawn/Desktop/model/M3E')
        self.index = faiss.IndexFlatL2(dim)
        self.id_map=[]
        FAQ_FILENAME = 'faq'
        self.pkl_path = os.path.join(save_path, FAQ_FILENAME + '.pkl')
        self.faiss_path = os.path.join(save_path, FAQ_FILENAME + '.index')
    def save(self):
        #serialize_index = faiss.serialize_index(self.index)
        # torch.save({
        #     'index': serialize_index,
        #     'id_map': self.id_map
        # },self.save_path)
        faiss.write_index(self.index, self.faiss_path)
        # 保存元数据到PKL文件
        with open(self.pkl_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map
            }, f)
            print('faiss文件和元文件保存成功')
    def add(self,id,query):
        embed_query =self.embed(query)
        #这里就是添加的np格式的
        self.index.add(np.array([embed_query]))
        self.id_map.append(id)
        self.save()
    def embed(self,text):
        embed_text=self.embed_model.encode(text)
        if not isinstance(embed_text,np.ndarray):
           embed_text=embed_text.detach().cpu().numpy()
        return embed_text.astype(np.float32)

    def load(self):
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)

        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.id_map = data['id_map']
    def search(self,query,top_k):
        vector = self.embed(query)
        #D是距离，I是索引
        D,I=self.index.search(np.array([vector]),top_k)
        return [(self.id_map[d],float(D[0][i])) for i,d in enumerate(I[0]) if d !=-1]
if __name__ == '__main__':
    FAQ=FAQVectorstore(768)
    FAQ.add(1, "肉毒素多久有效？")
    FAQ.add(2, "玻尿酸可以维持多久？")
    FAQ.add(3, "激光祛斑的效果如何？")
    print(FAQ.index)
    result = FAQ.search('玻尿酸可以维持多久',top_k=3)
    print(result)
    print(FAQ.id_map)