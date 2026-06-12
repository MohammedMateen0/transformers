from transformers import AutoTokenizer

x="I am learning AI"
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

encoded=tokenizer(
    x,
    padding=True,
    trancation=True
    )
for key,val in encoded.items():
  print(key,
        val,
        "\n")