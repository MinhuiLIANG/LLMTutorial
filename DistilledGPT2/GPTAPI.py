from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class promptRequest(BaseModel):
    prompt: str

app = FastAPI()
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)

@app.post("/generate")
def generate_text(request: promptRequest):
    input_text = request.prompt
    if not input_text:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}