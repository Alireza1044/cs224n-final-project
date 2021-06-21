import torch
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
from datasets import Dataset


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_data():
    with open(f'michael.txt', 'r') as f:
        data = [d.strip() for d in f.readlines()]

    with open(f'dwight.txt', 'r') as f:
        data2 = [d.strip() for d in f.readlines()]

    dct = {"text": data + data2, "label": [0] * len(data) + [1] * len(data2)}

    d = Dataset.from_dict(dct)

    d = d.train_test_split(test_size=0.1, train_size=0.9)

    return d


if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_data()
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    full_train_dataset = tokenized_datasets["train"].shuffle()
    full_eval_dataset = tokenized_datasets["test"].shuffle()
    metric = load_metric("accuracy")
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,
        eval_dataset=full_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
