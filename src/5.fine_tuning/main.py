import torch.cuda
from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split

# raw_datasets = load_dataset("imdb")
metric = load_metric('accuracy')

def tokenize_function(examples):
    t = tokenizer(examples, padding=True, truncation=True, return_tensors="pt")
    return t


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", return_offsets_mapping=True)
    data = []
    with open('../../data/clean/cleaned_broken_sentences/michael.txt', 'r') as f:
        data = [l.strip() for l in f.readlines()]
    # max_len = max([len(d) for d in data])
    # tokenized_datasets = [tokenize_function(d) for d in data]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Trainer is using {device}")

    train_set, test_set = train_test_split(data, test_size=0.2, train_size=0.8)
    test_set, validation_set = train_test_split(test_set, test_size=0.5, train_size=0.5)
    train_set = tokenize_function(train_set).to(device)
    test_set = tokenize_function(test_set).to(device)
    validation_set = tokenize_function(validation_set).to(device)

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)

    training_args = TrainingArguments("test_trainer", evaluation_strategy='epoch')

    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=validation_set)
    trainer.train()
