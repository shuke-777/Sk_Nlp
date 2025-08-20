import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
class FAQVectorstore:
    def __init__(self, dim, faiss_path, meta_path):
        self.embed_model = '/Users/salutethedawn/Desktop/model/M3E'  # 你的嵌入模型
        self.index = faiss.IndexFlatL2(dim)
        self.faiss_path = faiss_path  # 用于保存索引
        self.meta_path = meta_path  # 用于保存元信息
        self.id_map = []

    def save(self):
        # 保存FAISS索引
        faiss.write_index(self.index, self.faiss_path)
        # 保存元信息，比如 id_map
        torch.save({'id_map': self.id_map}, self.meta_path)

    def load(self):
        # 加载FAISS索引
        self.index = faiss.read_index(self.faiss_path)
        # 加载元信息
        meta = torch.load(self.meta_path)
        self.id_map = meta['id_map']

    def add(self, id, query):
        embed_query = self.embed(query)
        # 使用 add_with_ids 可以指定索引ID
        self.index.add_with_ids(np.array([embed_query]), np.array([id], dtype=np.int64))
        self.id_map.append(id)  # 同步保存到元信息
        self.save()

    def embed(self, text):
        embed_text = self.embed_model.encode(text)
        if not isinstance(embed_text, np.ndarray):
            embed_text = embed_text.detach().cpu().numpy()
        return embed_text.astype(np.float32)

    def search(self, query, top_k):
        vector = self.embed(query)
        D, I = self.index.search(np.array([vector]), top_k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]


if __name__ == '__main__':
    faiss_path='./faq_index.faiss'
    meta_path='./faq_index.pkl'
    FAQ = FAQVectorstore(768,faiss_path=faiss_path,meta_path=meta_path)

    # 添加 FAQ
    FAQ.add(1, "肉毒素多久有效？")
    FAQ.add(2, "玻尿酸可以维持多久？")
    FAQ.add(3, "激光祛斑的效果如何？")

    # 查询
    result = FAQ.search("玻尿酸可以维持多久", top_k=3)
    print(result)  # 结果会直接返回 (faq_id, 距离)

    # 保存并重新加载
    FAQ.save()
    FAQ.load()
    result2 = FAQ.search("激光祛斑", top_k=3)
    print(result2)
