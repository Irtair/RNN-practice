import os
from transformers import pipeline
import evaluate
from tqdm import tqdm
from transformers import pipeline, logging
logging.set_verbosity_error()

generator = pipeline("text-generation", model="distilgpt2")

# Примеры генерации модели
generation_prompts = ["I am about", "What is going on with", "Don't you mind if", "Let us deal with", "Deep Learning is"]

print("Примеры генерации:\n")
for num, prompt in enumerate(generation_prompts):
    result = generator(
        prompt,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        pad_token_id=50256
    )

    text = result[0]['generated_text'].strip().replace('\n', ' ')
    print(f"{num+1}) {text}")



processed_path = "data\\dataset_processed.txt"
#Вычисление метрики ROUGE
if os.path.exists(processed_path):
    rouge = evaluate.load("rouge")

    with open(processed_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned = [line.strip() for line in lines[-500:] if line.strip()]

    prompts = []
    references = []

    for seq in cleaned:
        words = seq.split()
        if len(words) < 4: continue

        split_idx = int(0.75 * len(words))

        prompts.append(" ".join(words[:split_idx]))
        references.append(" ".join(words[split_idx:]))

    generated = []

    for prompt, ref in tqdm(zip(prompts, references), total=len(prompts)):
        ref_len = max(1, len(ref.split()))

        output = generator(
            prompt,
            max_new_tokens=ref_len,
            do_sample=True,
            top_k=50,
            pad_token_id=50256
        )

        full_text = output[0]['generated_text']
        continuation = full_text[len(prompt):].strip()

        continuation_words = continuation.split()
        continuation = " ".join(continuation_words[:ref_len])

        generated.append(continuation)

    scores = rouge.compute(predictions=generated, references=references)

    print(f"ROUGE-1: {scores['rouge1']:.4f}")
