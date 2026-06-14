import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay)
import torch
from transformers import (AutoTokenizer,AutoModelForSequenceClassification)
from torch.utils.data import (Dataset,DataLoader)
import os

os.makedirs(
    "day5_distilbert_sentiment/reports",
    exist_ok=True
)

df=pd.read_csv("day5_distilbert_sentiment/data/clean_data.csv")
df = df.drop(
    columns=["Unnamed: 0"]
)
print(df["label"].value_counts(normalize=True))

tokenizer=AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

class ReviewDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, key):
        item={
            k:v[key]
            for k,v
            in self.encodings.items()
        }
        item['labels']=torch.tensor(
            self.labels[key],
            dtype=torch.long
        )
        return item



X_train,X_test,y_train,y_test=train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

train_encoding=tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)
test_encoding=tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

train_dataset=ReviewDataset(
    train_encoding,
    y_train.tolist()
)

test_dataset=ReviewDataset(
    test_encoding,
    y_test.tolist()
)

train_loader=DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)
test_loader=DataLoader(
    test_dataset,
    batch_size=16,
)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)
loss_history = []
epochs=3
for epoch in range(epochs):
    model.train()
    total_loss=0
    
    for batch in train_loader:
        outputs=model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss=outputs.loss
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1},Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds=[]
    true_labels=[]
    with torch.no_grad():
        for batch in test_loader:
            outputs=model(
                input_ids=batch['input_ids'],
                attention_mask=batch["attention_mask"]
            )

            logits=outputs.logits
            predictions=torch.argmax(
                logits,
                dim=1
            )
            preds.extend(
                predictions.cpu().numpy()
            )
            true_labels.extend(
                batch["labels"].cpu().numpy()
            )
    epoch_loss = total_loss / len(train_loader)

    loss_history.append(
        epoch_loss
    )
model.save_pretrained(
    "day5_distilbert_sentiment/saved_model"
)

tokenizer.save_pretrained(
    "day5_distilbert_sentiment/saved_model"
)
plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig(
    "day5_distilbert_sentiment/reports/loss_curve.png"
)

plt.close()
accuracy = accuracy_score(
    true_labels,
    preds
)

print(
    f"Accuracy: {accuracy:.4f}"
)   

report = classification_report(
    true_labels,
    preds,
    target_names=[
        "Negative",
        "Neutral",
        "Positive"
    ]
)
with open(
    "day5_distilbert_sentiment/reports/classification_report.txt",
    "w"
) as f:

    f.write(
        "Classification REPORT\n"
    )

    f.write(
        report
    )


cm = confusion_matrix(
    true_labels,
    preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative","Neutral","Positive"]
)
disp.plot()
plt.savefig("day5_distilbert_sentiment/reports/confusion_matrix.jpg")
plt.close()