
import os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from adapters import AutoAdapterModel
import ipdb
import pickle
import re
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import numpy as np
import pymssql
import textwrap
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


latex_model_path = r"C:\vscode_project\latex_tokenizer\output\right_model\checkpoint-1101"
latex_embedding_path = r"C:\vscode_project\latex_tokenizer\latex_embeddings"
topic_embedding_path = r"C:\vscode_project\latex_tokenizer\topic_embeddings"
paper_info_path = r"C:\vscode_project\latex_tokenizer\discrete_data\all_paper_info.jsonl"
NUM_SELECT = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
latex_model = SentenceTransformer(latex_model_path, device=device)
topic_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
topic_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
topic_model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
topic_model.to(device)

def get_topic_embeddings(id_list):
    result = []
    with open(os.path.join(topic_embedding_path, "abstract_id_list.pkl"), "rb") as fr:
        abstract_id_list = pickle.load(fr)
    with open(os.path.join(topic_embedding_path, "body_id_list.pkl"), "rb") as fr:
        body_id_list = pickle.load(fr)
    abstract_embeddings = torch.load(os.path.join(topic_embedding_path, "abstract_embedding.pt"))

    for idx, item in enumerate(abstract_id_list):
        if item in id_list:
            result.append({"id": item, "abstract_embedding": abstract_embeddings[idx]})
    file_list = []
    for f in os.listdir(topic_embedding_path):
        if "embeddings_part" in f:
            file_list.append(f)
    sorted_file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
    topic_embeddings = []
    for file in sorted_file_list:
        embeddings = torch.load(os.path.join(topic_embedding_path, file))
        topic_embeddings.append(embeddings)
    
    topic_embeddings = torch.cat(topic_embeddings, dim=0)
    for id in id_list:
        start = 0
        end = 0
        for idx, item in enumerate(body_id_list):
            if item["id"] == id:
                start = idx
                while body_id_list[idx]["id"] == id:
                    idx += 1
                end = idx
                break
        for idx, res in enumerate(result):
            if res["id"] == id:
                result[idx].update({"topic_embeddings": topic_embeddings[start:end]})
    return result


def merge_embedding(query, keys, values=None):
    """
    基于 Cross-Attention 的无参向量融合
    :param query:  [1, d] 查询向量
    :param keys:   [n, d] 多个段落向量
    :param values: [n, d] 默认与 keys 相同
    :return:       [1, d] 融合后的向量
    """
    if values is None:
        values = keys

    d = query.size(-1)
    # 注意力权重计算
    scores = torch.matmul(query, keys.T) / d**0.5   # [1, n]
    attn_weights = F.softmax(scores, dim=-1)        # [1, n]
    # 加权求和
    context = torch.matmul(attn_weights, values)    # [1, d]
    return context


def resort(query_text, id_list):
    result = []
    inputs = topic_tokenizer([query_text], padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = topic_model(**{k: v.to("cuda") for k, v in inputs.items()})
    query_embeddings = output.last_hidden_state[:, 0, :].cpu()
    for item in id_list:
        merge_emb = merge_embedding(query_embeddings, item["topic_embeddings"])
        abstract_score = F.cosine_similarity(query_embeddings, item["abstract_embedding"])
        body_score = F.cosine_similarity(query_embeddings, merge_emb)
        score = (abstract_score + body_score) / 2
        result.append({"id": item["id"], "score": score.item()})
    return result


id_list = ["2209.00195", "2209.00198"]
query_text = "I'm going to find a book about newton's third law"
hh = get_topic_embeddings(id_list)
res = resort(query_text, hh)
print(res)

# for item in hh:
#     print(item["abstract_embedding"].shape, item["topic_embeddings"].shape,)