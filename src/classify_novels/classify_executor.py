import pandas as pd
from tqdm import tqdm

import classify_rag_system as rag_system

class executor:
    def __init__(self, splited_texts) -> None:
        self.splited_texts = splited_texts
        self.question_db = pd.read_csv("./question/query.csv")
    
    def run(self):
        data = {"index": [], "answer": [], "reason": []}

        embeddings = rag_system.get_embeddings(self.splited_texts)
        
        for _, row in tqdm(self.question_db.iterrows(), total=len(self.question_db), desc="回答生成"):
            index = row['index']
            query = row['problem']
            
            # RAGシステムの応答を取得
            answer, reason = rag_system.run_rag_system(query, self.splited_texts, embeddings)
            answer = answer.replace("\n", "")
            
            # データをリストに追加
            data["index"].append(index)
            data["answer"].append(answer)
            data["reason"].append(reason)
            print(answer)

        prediction_db = pd.DataFrame(data)
        
        prediction_db.to_csv("./storage/classify/classify.csv", index=False, header=False)