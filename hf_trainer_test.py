import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_datasets = datasets.load_dataset("imdb")


def tokenize_function(sample):
    return tokenizer(sample["text"], max_length=300, truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="imdb_test_trainer",  # 指定输出文件夹，没有会自动创建
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_ratio=0.2,
    logging_dir="./imdb_train_logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=5,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,  # 在定义了tokenizer之后，其实这里的data_collator就不用再写了，会自动根据tokenizer创建
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
