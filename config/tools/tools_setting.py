# コメントアウト部のように、FunctionCallingの対象となる各関数に対して設定する。

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks_by_keyword",
            "description": "指定したkeywordを含むチャンクを抽出する関数。ユーザーからの質問に答える際に重要な単語があり、その単語が含まれるチャンクが回答生成時に必要だと判断した際に呼び出されるべき関数です。特に人名や地名など、その文章特有の固有名詞が質問文に含まれる場合は有効かもしれません。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "回答生成時に最も重要だと判断した単語。ユーザーの質問文に含まれても、Assistantが重要だと考える単語を生成してもよい。",
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
        "function": {
            "name": "get_keyword_counts",
            "description": "指定したkeywordが、文章中に何回出現するかを返す関数。ユーザーからの質問で、単語の出現回数等を聞かれた場合に呼び出されるべき関数です。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "出現回数が知りたい単語が格納される。",
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            }
        },
        # "function": {
        #     "name": "retrieve_similar_chunks",
        #     "description": "ユーザーの質問文のベクトルと類似度が高いベクトルを持つチャンクを抽出する関数。ユーザーの質問への回答を生成する際に、類似度による紐づいたチャンクを基に良い回答が生成できると判断した場合に呼び出されるべき関数です。",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "retrieval_num": {
        #                 "type": "int",
        #                 "description": "類似度検索で抽出するチャンクの数が格納される。",
        #             },
        #         },
        #         "required": ["retrieval_num"],
        #         "additionalProperties": False,
        #     }
        # }
    }
]