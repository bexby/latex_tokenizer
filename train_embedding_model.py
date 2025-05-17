from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset

dataset = load_dataset("json", data_files="latex_pairs_20000.jsonl", split="train")
dataset_dict = dataset.train_test_split(0.1)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

# import ipdb
# ipdb.set_trace()

model = SentenceTransformer("./init_model")
# model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

loss = CosineSimilarityLoss(model)
evaluator = EmbeddingSimilarityEvaluator(eval_dataset["text1"], eval_dataset["text2"], eval_dataset["label"], name='latex_eval')

training_args = SentenceTransformerTrainingArguments(
    output_dir="output/right_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    evaluation_strategy="steps",  # 如果没有验证集
    eval_steps=100,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True  # 如果您的硬件支持 FP16
)

# 创建 Trainer 并开始训练
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    evaluator=evaluator,
    loss=loss
)

trainer.train()
