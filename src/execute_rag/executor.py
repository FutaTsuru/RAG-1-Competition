import pandas as pd
from tqdm import tqdm
import numpy as np

from rag_system import rag_system
from classify_novels.Title_teller import Title_teller
from classify_novels.summary_teller import Summary_teller
from extract_answer import extract_answer
from config import setting

class executor:
    def __init__(self, splited_texts_db: pd.DataFrame) -> None:
        self.splited_texts_db = splited_texts_db
        self.question_db = pd.read_csv("./question/query.csv")
    
    def run(self):
        data = {"index": [], "answer": [], "reason": []}
        all_data = {"index": [], "answer": [], "reason": []}
        reason_num_data = {"index": [], "answer": [], "reason_num": []}

        splited_texts = self.splited_texts_db["chunk"].to_list()

        # 指定したテキストをベクトル化して、指定した保存先に保存する関数
        # rag_system.make_and_save_embeddings(splited_texts, setting.CHUNK_EMBEDDINGS_PATH)

        # チャンクのベクトルを読み込む
        embeddings = np.load(setting.CHUNK_EMBEDDINGS_PATH)

        title_teller = Title_teller()
        summary_teller = Summary_teller()
        
        for _, row in tqdm(self.question_db.iterrows(), total=len(self.question_db), desc="回答生成"):
            index = row['index']
            query = row['problem']
            
            # RAGシステムの応答を取得 (質問と小説を紐づけられたら、引数のsplited_textsはその分縮小して渡す！)
            title = title_teller.connect_query_to_novel(index) # ここ引数indexでお願いしますby 進
            if title != "分かりません":
                target_splited_texts = self.splited_texts_db[self.splited_texts_db["title"]==title]["chunk"].to_list()
                index_list = self.splited_texts_db[self.splited_texts_db["title"]==title].index.tolist()
                target_embeddings = np.take(embeddings, index_list, axis=0)  
                target_summary = "まず，ある小説の要旨を示します．" +  summary_teller.connect_title_to_summary(title)
            else:
                target_splited_texts = splited_texts
                target_embeddings = embeddings
                target_summary = "以下に示すragの検索結果は，複数の小説が混在している可能性があるので注意してください．"

            answer, reason = rag_system.run_rag_system(query, target_splited_texts, target_embeddings, target_summary)
            answer = extract_answer.extract_answer(answer)
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