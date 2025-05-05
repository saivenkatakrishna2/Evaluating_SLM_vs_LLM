from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load model checkpoints
scratch_path = "./slm_from_scratch_model"
finetune_path = "./fine_tune_slm_model"
llm_path = "google/flan-t5-large"

# Load models
slm_scratch_tokenizer = AutoTokenizer.from_pretrained(scratch_path)
slm_scratch_model = AutoModelForSeq2SeqLM.from_pretrained(scratch_path)

slm_finetune_tokenizer = AutoTokenizer.from_pretrained(finetune_path)
slm_finetune_model = AutoModelForSeq2SeqLM.from_pretrained(finetune_path)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_path)

def generate_summary(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summaries = {}
    if request.method == "POST":
        text = request.form["paragraph"]
        summaries["SLM Scratch"] = generate_summary(slm_scratch_model, slm_scratch_tokenizer, text)
        summaries["SLM Fine-Tuned"] = generate_summary(slm_finetune_model, slm_finetune_tokenizer, text)
        summaries["LLM"] = generate_summary(llm_model, llm_tokenizer, text)
    return render_template("index.html", summaries=summaries)

if __name__ == "__main__":
    app.run(debug=True)
