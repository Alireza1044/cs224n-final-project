from datasets import load_dataset
from transformers import AutoModelForCausalLM
import math
import torch
from transformers import pipeline
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
import sys

sys.path.append("..")
import config
import argparse


def load_data(char):
    data_files = {}
    train_file = config.data_path(char)
    if train_file is not None:
        data_files["train"] = train_file
    extension = train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    dataset = load_dataset(extension, data_files=data_files)['train']
    dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)
    return dataset


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--char', type=str, default="michael")
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--text', type=str, default="hi")
    parser.add_argument('--length', type=int, default=10)
    args = parser.parse_args()

    if not args.predict:
        datasets = load_data(args.char)
        model_checkpoint = "distilgpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
        block_size = 128

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

        training_args = TrainingArguments(
            "test-clm",
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
        )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Training is using:\n\t{device}")

        model.to(device)
        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        model_name = f"{args.char}_bert_lm"
        model.push_to_hub(model_name, use_temp_dir=True)
        tokenizer.push_to_hub(model_name, use_temp_dir=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(f"Alireza1044/{args.char}_bert_lm")
        tokenizer = AutoTokenizer.from_pretrained(f"Alireza1044/{args.char}_bert_lm", use_fast=True)
        device = 0 if torch.cuda.is_available() else -1
        text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

        prefix_text = args.text
        text_generation(prefix_text, max_length=args.length)
