import pandas as pd
from tqdm import tqdm

from rag_system import rag_system

class executor:
    def __init__(self, splited_texts) -> None:
        self.splited_texts = splited_texts
        self.question_db = pd.read_csv("./question/query.csv")
    
    def run(self):
        data = {"index": [], "answer": [], "reason": []}
        all_data = {"index": [], "answer": [], "reason": []}
        reason_num_data = {"index": [], "answer": [], "reason_num": []}

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
            data["reason"].append("なし")

            all_data["index"].append(index)
            all_data["answer"].append(answer)
            all_data["reason"].append(reason)

            reason_num_data["index"].append(index)
            reason_num_data["answer"].append(answer)
            reason_num_data["reason_num"].append(len(reason))
        
        prediction_db = pd.DataFrame(data)
        prediction_contain_reason_db = pd.DataFrame(all_data)
        prediction_contain_reason_num_db = pd.DataFrame(reason_num_data)
        
        prediction_db.to_csv("./evaluation/submit/predictions.csv", index=False, header=False)
        prediction_contain_reason_db.to_csv("./evaluation/submit/predictions_contain_reason.csv", index=False, header=False)
        prediction_contain_reason_num_db.to_csv("./evaluation/submit/predictions_contain_reason_num.csv", index=False, header=False)