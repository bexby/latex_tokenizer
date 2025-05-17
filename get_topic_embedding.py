from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import tqdm
import torch
import pickle


tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model = model.to("cuda")
body_folder = r"C:\vscode_project\latex_tokenizer\paper_body"
paper_file = r"C:\vscode_project\latex_tokenizer\discrete_data\all_paper_info.jsonl"
save_path = r"C:\vscode_project\latex_tokenizer\body_embedding"
batch_size = 256

def write_body_tensor(path, save_path):
    id_ls = []
    body = []
    part_index = 1
    file_list = os.listdir(path)
    for file in tqdm(file_list, desc="imported files", position=0):
        res_emb = []
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                d = json.loads(line)
                body.append(d["paragraph"])
                id_ls.append({d["id"], d["paragraph_id"]})

        # inputs = tokenizer(body, padding=True, truncation=True,
        #                            return_tensors="pt", return_token_type_ids=False, max_length=512)
        with torch.no_grad():
            # import ipdb
            # ipdb.set_trace()
            # output = model(**{k: v.to("cuda") for k, v in inputs.items()})
            for s in tqdm(range(len(body) // batch_size), position=1):
                batch_inputs = tokenizer(body[s:s + batch_size], padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
                
                output = model(**{k: v.to("cuda") for k, v in batch_inputs.items()})
                tensor_embeddings = output.last_hidden_state[:, 0, :]
                res_emb.append(tensor_embeddings.cpu())
            
            torch.save(torch.stack(res_emb), os.path.join(save_path, "embeddings_part" + str(part_index) + ".pt"))
            body = []
            part_index += 1
    
    with open(os.path.join(save_path, "body_id_list.pkl"), "wb") as fw:
        pickle.dump(id_ls, fw)

def write_paper_tensor(path, save_path):
    abstract = []
    id_list = []
    with open(path, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            d = json.loads(line)
            prompt = f'Title: {d["title"]}\n Keywords: {d["keywords"]}\n Abstract: {d["abstract"]}'
            abstract.append(prompt)
            id_list.append(d["id"])
    
        inputs = tokenizer(abstract, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)
        with torch.no_grad():
            output = model(**{k: v.to("cuda") for k, v in inputs.items()})
            tensor_embeddings = output.last_hidden_state[:, 0, :]
            torch.save(tensor_embeddings, os.path.join(save_path, "abstract_embedding.pt"))

        with open(os.path.join(save_path, "abstract_id_list.pkl"), "wb") as fw:
            pickle.dump(id_list, fw)


write_body_tensor(body_folder, save_path)
# write_paper_tensor(paper_file, save_path)