import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassifiction
)

tokenizer=AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model=AutoModelForSequenceClassifiction.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

model.train()

text="The found was amazing and the service was exellent"

inputs=tokenizer(
    text,
    padding=True,
    truncation=True,
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
    token_type_ids=inputs["token_type_ids"],
    labels=labels
)

loss=outputs.loss

print("Loss: ",loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()
print("One training step completed!")