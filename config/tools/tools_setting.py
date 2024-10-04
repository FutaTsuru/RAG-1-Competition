# コメントアウト部のように、FunctionCallingの対象となる各関数に対して設定する。

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks_by_keyword",
            "description": "指定したkeywordを含むチャンクを抽出する関数。ユーザーからの質問に答える際に、重要度が高い単語があると判断し、それを含むチャンク情報が回答生成に役立つと考えたときに積極的に呼び出されるべき関数です。特に人名や地名など、小説特有の固有名詞を含むチャンクは有益な情報を持っている確率が高いです。",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_keyword_counts",
            "description": "指定したkeywordが、文章中に何回出現するかを返す関数。ユーザーからの質問で、固有名詞の出現回数等を聞かれた場合にのみ呼び出されるべき関数です。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "出現回数が知りたい単語が格納される。必ず質問文に含まれる単語を代入してください。文章を代入してはいけません。",
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    # 必要に応じてコメントアウトされた関数も追加
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "retrieve_similar_chunks",
    #         "description": "ユーザーの質問文のベクトルと類似度が高いベクトルを持つチャンクを抽出する関数。ユーザーの質問への回答を生成する際に、類似度による紐づいたチャンクを基に良い回答が生成できると判断した場合に呼び出されるべき関数です。",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "retrieval_num": {
    #                     "type": "int",
    #                     "description": "類似度検索で抽出するチャンクの数が格納される。",
    #                 },
    #             },
    #             "required": ["retrieval_num"],
    #             "additionalProperties": False,
    #         },
    #     },
    # },
]

function_list = ["retrieve_chunks_by_keyword", "get_keyword_counts"]

def remove_tool_by_function_name(tools, function_name):
    return [tool for tool in tools if tool['function']['name'] != function_name]