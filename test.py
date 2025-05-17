from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
import torch
import torch.nn.functional as F
import json

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
#other possibilities: allenai/specter2_<classification|regression|adhoc_query>

with open(r"C:\vscode_project\latex_tokenizer\discrete_data\all_paper_info.jsonl", "r") as fr:
    data = [json.loads(line) for line in fr.readlines()]

paragraphs = []
for i in range(5):
    paragraphs.append(data[i]["abstract"])
paragraphs.append("today is a nice day")
# paragraphs = [
#     "Deep learning has revolutionized NLP in recent years.",
#     "Transformers are the current state-of-the-art in many NLP tasks.",
#     "This paper proposes a novel architecture that builds on BERT.",
#     "I'm looking for a paper about Newton's Third law",
# ]

inputs = tokenizer(paragraphs, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
# take the first token in the batch as the embedding
embeddings = output.last_hidden_state[:, 0, :]

# print(embeddings[0])
# print(embeddings[0].shape)
result = []
for i in range(embeddings.shape[0]):
    result.append(F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings))

ss = torch.stack(result, dim=0)
print(ss)







