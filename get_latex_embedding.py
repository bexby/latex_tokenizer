from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import tqdm
import torch
import pickle

model = SentenceTransformer(r"C:\vscode_project\latex_tokenizer\output\right_model\checkpoint-1101", device="cuda")
latex_path = r"C:\vscode_project\latex_tokenizer\math_info"
save_path = r"C:\vscode_project\latex_tokenizer\latex_embeddings"

def write_tensor(path, save_path):
    math_id_ls = []
    latex = []
    part_index = 1
    latex_list = os.listdir(path)
    for file in tqdm(latex_list, desc="imported files"):
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                d = json.loads(line)
                latex.append(d["nor_expression"])
                math_id_ls.append(d["math_id"])
        if len(latex) > 80000:
            tensor_embeddings = model.encode(latex, convert_to_tensor=True)
            torch.save(tensor_embeddings, os.path.join(save_path, "embeddings_part" + str(part_index) + ".pt"))
            latex = []
            part_index += 1
    
    with open(os.path.join(save_path, "math_id_list.pkl"), "wb") as fw:
        pickle.dump(math_id_ls, fw)


write_tensor(latex_path, save_path)