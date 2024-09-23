import pandas as pd
from tqdm import tqdm

class executor:
    def __init__(self,qa) -> None:
        self.qa = qa
        self.question_db = pd.read_csv("./question/query.csv")
    
    def rag_response(self,query):
        return self.qa.invoke(query)
    
    def run(self):
        # データを格納するためのリストを作成
        data = {"index": [], "answer": [], "reason": []}
        
        for _, row in tqdm(self.question_db.iterrows(), total=len(self.question_db), desc="回答生成"):
            index = row['index']
            problem = row['problem']
            
            # RAGシステムの応答を取得
            response = self.rag_response(problem)
            reason = response["source_documents"]
            result = response["result"].replace("\n", "")

            # 返答が45字を超える場合、45字以内にする.
            if len(response) > 45:
                response = response[:45]
            
            # データをリストに追加
            data["index"].append(index)
            data["answer"].append(result)
            # data["reason"].append(reason)
            data["reason"].append("なし")
        
        prediction_db = pd.DataFrame(data)
        
        prediction_db.to_csv("./evaluation/submit/predictions.csv", index=False, header=False)