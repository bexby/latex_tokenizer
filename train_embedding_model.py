from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import load_dataset

# 加载 JSON 文件为 Hugging Face Dataset
dataset = load_dataset("json", data_files="train_latex_pairs.json", split="train")

# 查看前两个样本
print(dataset[:2])
# 加载预训练模型
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# 定义损失函数
loss = CosineSimilarityLoss(model)

# 设置训练参数
training_args = SentenceTransformerTrainingArguments(
    output_dir="output/latex-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    evaluation_strategy="no",  # 如果没有验证集
    save_strategy="epoch",
    logging_steps=100,
    fp16=True  # 如果您的硬件支持 FP16
)

# 创建 Trainer 并开始训练
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss
)

trainer.train()
