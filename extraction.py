from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import unicodedata
import html

def clean_text(text):
    # 清洗函数：保留 Unicode，去格式标记、转义字符串等
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'start_[A-Z_]+', '', text)
    text = re.sub(r'end_[A-Z_]+', '', text)
    text = re.sub(r'\b(?:italic|bold|math|symbol|script|cal|frak|roman)_([a-zA-Z0-9Δ-]+)\b', r'\1', text)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    return ' '.join(text.split())

def split_long_text(text, tokenizer, max_tokens=512):
    """
    使用指定 tokenizer 将长文本分段，每段不超过 max_tokens。
    每段是 token 对齐的，不会截断 token。
    """
    # Tokenize 并获取所有 token ID
    encoded = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for i in range(0, len(encoded), max_tokens):
        chunk_ids = encoded[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks

def parse_paper(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html5lib')

    body = soup.get_text()
    intro_index = body.find('Introduction')
    ref_index = body.find('Reference')
    if intro_index != -1 and ref_index != -1 and intro_index < ref_index:
        body = body[intro_index:ref_index].strip()

    text = re.sub(r'\n{2,}', '[PARA]', body)
    text = text.replace('\n', '')
    cleaned_text = text.replace('[PARA]', '\n')
    cleaned_text = clean_text(cleaned_text)
    return cleaned_text

def load_file_name(root_path) -> dict[str, list[str]]:
    ls_dir = os.listdir(root_path)
    result = dict()
    for dir in ls_dir:
        ls_file = os.listdir(os.path.join(root_path, dir))
        tem = []
        for f in ls_file:
            tem.append(f)
        result.update({dir: tem})
    return result

def main():
    data_path = r"C:\Users\86159\Downloads\ar5iv_1710-2209\ar5iv"
    save_path = r"C:\vscode_project\latex_tokenizer\paper_body"
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    ff = load_file_name(data_path)
    avaliable_file = []
    paper_info = []
    file_index = 1
    for folder, file_ls in tqdm(ff.items(), desc="processed folders", position=0):
        for re_file in tqdm(file_ls, desc="files in folder", position=1):
            file = os.path.join(data_path, folder, re_file)
            if os.path.getsize(file) < 50000:
                continue
            avaliable_file.append(file)
            body = parse_paper(file)
            body_split = split_long_text(body, tokenizer)
            paper_id = re_file.split(".")[0] + "." + re_file.split(".")[1]
            
            with open(os.path.join(save_path, "paper_body_part" + str(file_index) + ".jsonl"), "a") as fw:       
                for pid, bs in enumerate(body_split):
                    item = {"id": paper_id, "paragraph_id": pid, "paragraph": bs}
                    paper_info.append(item)
                    fw.write(json.dumps(item))
                    fw.write("\n")

            if len(paper_info) > 50000:
                paper_info = []
                file_index += 1
            


if __name__ == '__main__':
    main()
