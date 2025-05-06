import time
import torch
import psutil
import evaluate  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


rouge = evaluate.load("rouge")

def run_model(model_name, user_input, reference_summary):
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    prompt = f"summarize: {user_input}"

    start_time = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    result = summarizer(prompt, max_length=100, min_length=20, do_sample=False)
    generated_summary = result[0]['generated_text']

    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    end_time = time.time()

    rouge_scores = rouge.compute(predictions=[generated_summary], references=[reference_summary])

    return {
        "output": generated_summary,
        "rouge": rouge_scores,
        "time_taken_sec": round(end_time - start_time, 2),
        "memory_used_mb": round(mem_after - mem_before, 2)
    }

user_input = input("Enter the paragraph you want to summarize:\n")
reference_summary = input("Enter the reference summary:\n")  

slm_model = "google/flan-t5-small"   
llm_model = "google/flan-t5-xl"      

print("\n--- SLM: google/flan-t5-small ---")
slm_result = run_model(slm_model, user_input, reference_summary)
print("Summary:", slm_result["output"])
print("ROUGE:", slm_result["rouge"])
print("Time (s):", slm_result["time_taken_sec"])
print("Memory Used (MB):", slm_result["memory_used_mb"])

print("\n--- LLM: google/flan-t5-xl ---")
llm_result = run_model(llm_model, user_input, reference_summary)
print("Summary:", llm_result["output"])
print("ROUGE:", llm_result["rouge"])
print("Time (s):", llm_result["time_taken_sec"])
print("Memory Used (MB):", llm_result["memory_used_mb"])
