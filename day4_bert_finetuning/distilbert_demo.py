import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

tokenizer=AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

model.train()

text="The food was amaizing and the service was excelient"

inputs=tokenizer(
    text,
    padding=True,
    trucation=True,
    return_tensors="pt"
)

labels=torch.tensor([2])

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

outputs=model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=labels
)

loss=outputs.loss

print("Loss: ",loss.item())

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("One training step completed!")

prediction=torch.argmax(
    outputs.logits,
    dim=1
)
print("Predicted Class: ",prediction.item())