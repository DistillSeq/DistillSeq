import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import random
import os

# Config the training parameters
config = {
    "seed": TODO,
    "data_path": "TODO",
    "model_name": "TODO",
    "epochs": TODO,
    "learning_rate": TODO,
    "batch_size": TODO,
    "max_length": TODO,
    "save_path": "TODO",
}

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(config["seed"])

def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    texts = df['text'].values
    labels = df['label'].values
    return train_test_split(texts, labels, test_size=.2)

train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(config["data_path"])

tokenizer = BertTokenizer.from_pretrained(config["model_name"])

def encode_texts(tokenizer, texts, labels, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',  # 更新此处
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = encode_texts(tokenizer, train_texts, train_labels, config["max_length"])
val_input_ids, val_attention_masks, val_labels = encode_texts(tokenizer, val_texts, val_labels, config["max_length"])

def create_dataloader(input_ids, attention_masks, labels, batch_size, is_train=True):
    data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(data) if is_train else SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)

train_dataloader = create_dataloader(train_input_ids, train_attention_masks, train_labels, config["batch_size"])
validation_dataloader = create_dataloader(val_input_ids, val_attention_masks, val_labels, config["batch_size"], is_train=False)

model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=len(np.unique(np.concatenate([train_labels, val_labels]))))  # 更新此处
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-8)
total_steps = len(train_dataloader) * config["epochs"]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return predictions, true_labels

for epoch_i in range(config["epochs"]):
    print(f'Epoch {epoch_i + 1}/{config["epochs"]}')
    train_loss = train(model, train_dataloader, optimizer, scheduler, device)
    print(f'Train loss: {train_loss}')
    predictions, true_labels = evaluate(model, validation_dataloader, device)
    # 这里可以根据predictions和true_labels计算准确率等指标

if not os.path.exists(config["save_path"]):
    os.makedirs(config["save_path"])
model.save_pretrained(config["save_path"])
tokenizer.save_pretrained(config["save_path"])
print(f'Model saved to {config["save_path"]}')
