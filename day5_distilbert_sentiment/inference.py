import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Load saved tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "saved_model"
)

# Load saved model
model = AutoModelForSequenceClassification.from_pretrained(
    "saved_model"
)

model.eval()

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

while True:

    text = input(
        "\nEnter financial news text (or 'quit'): "
    )

    if text.lower() == "quit":
        break

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0][prediction].item()
            * 100
        )

    print(
        f"\nPrediction : {label_map[prediction]}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print("\nClass Probabilities:")

    for idx, label in label_map.items():

        print(
            f"{label}: "
            f"{probabilities[0][idx].item()*100:.2f}%"
        )

