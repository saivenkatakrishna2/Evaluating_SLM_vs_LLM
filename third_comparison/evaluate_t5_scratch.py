import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Load your fine-tuned model locally
model = T5ForConditionalGeneration.from_pretrained("slm_from_scratch_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("slm_from_scratch_model")

# Load evaluation dataset
val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")

# Load metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")
bertscore = evaluate.load("bertscore")

predictions, references = [], []
model.eval()

for item in val_dataset:
    inputs = tokenizer("summarize: " + item["article"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(decoded)
    references.append(item["highlights"])

# Compute metrics
rouge_result = rouge.compute(predictions=predictions, references=references)
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")

print("\n Evaluation Results (Training from Scratch):")
print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"BLEU: {bleu_result['score']:.2f}")
print(f"BERTScore F1: {sum(bert_result['f1']) / len(bert_result['f1']):.4f}")
