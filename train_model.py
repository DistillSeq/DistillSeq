import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split

# Config training parameters
MODEL_NAME = 'TODO'
BATCH_SIZE = "TODO"
EPOCHS = "TODO"
MAX_LEN = 512
LEARNING_RATE = "TODO"
DATA_PATH = "TODO"


df = pd.read_csv(DATA_PATH)
texts = df['text'].tolist()
labels = df['label'].unique().tolist()
label_dict = {label: i for i, label in enumerate(labels)}
encoded_labels = [label_dict[label] for label in df['label']]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoded_labels, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=MAX_LEN,
                                  truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }


train_dataset = TextDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        labels = batch['labels'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_loader)            
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")


model_save_path = f'{MODEL_NAME}_trained_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")
