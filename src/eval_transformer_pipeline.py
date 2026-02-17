import os
import evaluate
from tqdm import tqdm
from transformers import pipeline, logging
logging.set_verbosity_error()
import yaml

path = "./configs/config.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

def transformer_generate(generator, generation_prompts):
    print("Примеры генерации:")
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



processed_path = config["data"]["processed_path"]

def rouge_calc_for_transformer(generator):
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
