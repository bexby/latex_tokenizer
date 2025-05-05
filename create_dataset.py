import random
import Levenshtein
import re
import json
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("OleehyO/latex-formulas", "raw_formulas", split="train")
formulas = list({item['latex_formula'] for item in dataset if item['latex_formula']})


def extract_latex_keywords(expr):
    """
    提取 LaTeX 公式中的结构性关键词，如 \frac、\sqrt、\alpha 等。
    """
    # 常见结构命令或符号前缀：以反斜杠开头的 LaTeX 命令
    pattern = r"\\[a-zA-Z]+"
    return set(re.findall(pattern, expr))


def jaccard_similarity(set1, set2):
    """
    计算两个集合之间的 Jaccard 相似度。
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def is_structurally_similar(f1, f2, threshold=0.3):
    """
    判断两个 LaTeX 公式在结构关键词上是否相似。
    使用 Jaccard 相似度衡量。
    """
    keywords1 = extract_latex_keywords(f1)
    keywords2 = extract_latex_keywords(f2)
    sim = jaccard_similarity(keywords1, keywords2)
    return sim >= threshold


def build_train_pairs(formulas, num_pairs=5000, edit_dist_limit=10):
    pairs = []
    for _ in tqdm(range(num_pairs)):
        a = random.choice(formulas)

        # 正样本：结构类似的公式
        candidates = [f for f in random.sample(formulas, 10000) if f != a and is_structurally_similar(a, f)]
        if candidates:
            b = random.choice(candidates)
            pairs.append(
                {
                "texts": [a, b],
                "label": 1.0
                }
            )

        # 负样本：编辑距离差异大
        cands = [f for f in random.sample(formulas, 10000) if f != a and Levenshtein.distance(a, f) > edit_dist_limit]
        if cands:
            c = random.choice(cands)
            pairs.append(
                {
                "texts": [a, c],
                "label": 0.0
                }
            )

    return pairs

def save_latex_train_data(pairs, save_path="latex_pairs.json"):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

pairs = build_train_pairs(formulas, num_pairs=10000)
save_latex_train_data(pairs, "latex_pairs_20000.json")
