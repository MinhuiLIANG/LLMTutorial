import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

model.eval()

input_text = "Hello GPT"

inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型生成的文本：")
print(generated_text)