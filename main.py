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

latex_file_path = r"C:\vscode_project\latex_tokenizer\math_info"
latex_model_path = r"C:\vscode_project\latex_tokenizer\output\right_model\checkpoint-1101"
latex_embedding_path = r"C:\vscode_project\latex_tokenizer\latex_embeddings"
topic_embedding_path = r"C:\vscode_project\latex_tokenizer\topic_embeddings"
paper_info_path = r"C:\vscode_project\latex_tokenizer\discrete_data\all_paper_info.jsonl"
NUM_SELECT = 20


print("正在加载模型")
device = "cuda" if torch.cuda.is_available() else "cpu"
latex_model = SentenceTransformer(latex_model_path, device=device)
topic_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
topic_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
topic_model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
topic_model.to(device)
print("模型加载成功")
    

def initial_selection(kw_exp, topk):
    with open(os.path.join(latex_embedding_path, "math_id_list.pkl"), "rb") as fr:
        math_id_list = pickle.load(fr)
    org_embedding = latex_model.encode(kw_exp, convert_to_tensor=True, device=device)
    candidates = []
    previous_index = 0
    file_ls = os.listdir(latex_embedding_path)[:-1]
    sorted_file_list = sorted(file_ls, key=lambda x: int(re.search(r'\d+', x).group()))
    for file in tqdm(sorted_file_list, desc="load latex embeddings"):
        embeddings = torch.load(os.path.join(latex_embedding_path, file))
        embeddings = embeddings.to(device)
        similarities = latex_model.similarity(org_embedding, embeddings)
        topk_values, indices = torch.topk(torch.squeeze(similarities), k=topk, sorted=True)
        for v, i in zip(topk_values, indices):
            candidates.append({"math_id": math_id_list[i + previous_index], "similarity": v})
        previous_index += embeddings.shape[0]
    out_sim = []
    result = []
    for item in candidates:
        out_sim.append(item["similarity"])
    out_topk, out_indices = torch.topk(torch.tensor(out_sim), k=topk, sorted=True)
    for v, i in zip(out_topk, out_indices):
        result.append({"math_id": candidates[i]["math_id"], "similarity": v.item()})
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

def get_topic_embeddings(id_list):
    result = []
    with open(os.path.join(topic_embedding_path, "abstract_id_list.pkl"), "rb") as fr:
        abstract_id_list = pickle.load(fr)
    with open(os.path.join(topic_embedding_path, "body_id_list.pkl"), "rb") as fr:
        body_id_list = pickle.load(fr)
    abstract_embeddings = torch.load(os.path.join(topic_embedding_path, "abstract_embedding.pt"))

    for idx, item in enumerate(abstract_id_list):
        if item in id_list:
            result.append({"id": item, "abstract_embedding": abstract_embeddings[idx].unsqueeze(0)})
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


def fatch_infomation(outcome, score):
    with open(paper_info_path, "r") as fr:
        paper_info_list = [json.loads(line) for line in fr.readlines()]
    
    latex_info = []
    file_list = os.listdir(latex_file_path)
    for fl in tqdm(file_list, desc="load latex data"):
        with open(os.path.join(latex_file_path, fl), "r") as fr:
            latex_info.extend([json.loads(line) for line in fr.readlines()])

    paper_id = []
    math_id = []
    latex_exp = []
    latex_sim = []
    title = []
    keywords = []
    topic_score = []
    inline = []
    for item in outcome:
        paper_id.append(item["math_id"][:10])
        math_id.append(item["math_id"])
        latex_sim.append(item["similarity"])
        id = item["math_id"][:10]
        for it in paper_info_list:
            if it["id"] == id:
                title.append(it["title"])
                keywords.append(it["keywords"])
                break
        for it in score:
            if it["id"] == id:
                topic_score.append(it["score"])
                break
        
        for it in latex_info:
            if it["math_id"] == item["math_id"]:
                latex_exp.append(it["latex_expression"])
                inline.append("yes" if it["is_inline"] == 1 else "no")
                break

    date = []
    for item in paper_id:
        date.append("20" + item[:2] + "年" + item[2:4] + "月")
    # candidates = {"date": date, "math_id": math_id, "latex_expression": latex_exp, "exp_score": latex_sim, "is_inline": inline,  "title": title, "keywords": keywords, "topic_score": topic_score}
    candidates = {"日期": date, "表达式编号": math_id, "latex表达式": latex_exp, "表达式相似度": latex_sim, "是否在行内": inline,  "题目": title, "关键词": keywords, "主题相关度": topic_score}
    return candidates

def normalize_format(latex_expr):
    latex_format_symbols = {
        # 格式符
        '\\textbf', '\\textit', '\\texttt', '\\emph', '\\underline', '\\overline', '\\overbrace', '\\underbrace', '\\left', '\\right',
        '\\big', '\\Big', '\\bigg', '\\Bigg', '\\textit', '\\textsf', '\\textrm', '\\mathcal', '\\mathbb', '\\mathfrak', '\\mathscr', '\\displaystyle', '\\textrm',
    }
    for f in latex_format_symbols:
        latex_expr = re.sub("\\" + f, "", latex_expr)

    variables = sorted(set(re.findall(r'\b[a-zA-Z]\b', latex_expr)))
    replacement_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    var_mapping = {}
    for i, var in enumerate(variables):
        if i < len(replacement_letters):
            var_mapping[var] = replacement_letters[i]
    for var, replacement in var_mapping.items():
        latex_expr = re.sub(rf'\b{var}\b', replacement, latex_expr)

    return latex_expr


def main():

    query_expression = r"a^2+b^2=c^2"
    query_text = "we are going to find a paper about triangle"

    query_expression = normalize_format(query_expression)
    first_outcome = initial_selection(query_expression, NUM_SELECT)
    id_list = []
    for item in first_outcome:
        id_list.append(item["math_id"][:10])
    topic_emb = get_topic_embeddings(id_list)
    topic_score = resort(query_text, topic_emb)

    all_info = fatch_infomation(first_outcome, topic_score)

    show_table = pd.DataFrame(all_info)
    # show_table["paper_id"] = show_table["paper_id"].astype(str)
    # show_table = show_table.sort_values("topic_score", ascending=False)
    show_table = show_table.sort_values("主题相关度", ascending=False)
    show_table = show_table.applymap(lambda x: textwrap.fill(str(x), width=20))
    print("查询结果为：")
    print(tabulate(show_table, headers='keys', tablefmt='psql', showindex=False))


if __name__ == "__main__":
    main()