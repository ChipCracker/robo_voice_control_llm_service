from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("ChipCracker/qwen3-0.6B_robo_voice_control_finetune")
mdl = AutoModelForCausalLM.from_pretrained("ChipCracker/qwen3-0.6B_robo_voice_control_finetune",
                                           torch_dtype="auto", device_map="auto")
print("→", next(mdl.parameters()).device)

app = FastAPI()

class Prompt(BaseModel):
    messages: list[dict]

@app.post("/generate")
def generate(p: Prompt):
    text = tok.apply_chat_template(p.messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    ids = mdl.generate(**inputs, max_new_tokens=8)
    return {"text": tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)}
