import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.lstm_model import AutoCompleteLSTM
import evaluate
import yaml

path = "./configs/config.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)


# Объявление основных переменных, используемых при обучении
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Путь для сохранения модели
save_dir = config["data"]["model_save_dir"]
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, config["model"]["best_model_file"])

#Создание модели
model = AutoCompleteLSTM().to(device)

optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)
criterion = CrossEntropyLoss(ignore_index=-100)

#Проверка на наличие сохраненной модели
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
#Переменные для Early Stopping
best_val_loss = float("inf")
patience = config["training"]["early_stopping"]["patience"]
patience_counter = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

rouge = evaluate.load("rouge")

def evaluate_rouge_on_val(test_dataloader, tokenizer):
    model.eval()
    
    predictions = []
    references = []
    
    print("\n📊 Вычисление ROUGE...")
    
    with torch.no_grad():
        num_samples = 0
        max_samples = 100

        for batch in tqdm(test_dataloader, desc="ROUGE evaluation", leave=False):
            lines = batch['lines'].to(device)
            lengths = batch['lengths']
            
            for i in range(lines.size(0)):
                seq_len = lengths[i].item()
                full_seq = lines[i, :seq_len]

                split_idx = int(0.75 * seq_len)

                prompt = full_seq[:split_idx].unsqueeze(0)
                target = full_seq[split_idx:]

                logits, hidden = model(prompt)

                generated_tokens = []

                for _ in range(len(target)):
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)

                    generated_tokens.append(next_token.item())

                    logits, hidden = model(next_token.unsqueeze(0), hidden=hidden)

                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                ref_text = tokenizer.decode(target.tolist(), skip_special_tokens=True)

                predictions.append(pred_text)
                references.append(ref_text)

                sample_count += 1

            if num_samples >= max_samples:
                break
    
    scores = rouge.compute(predictions=predictions, references=references)
    print(f"ROUGE-1: {scores['rouge1']:.4f}")

def training(train_dataloader):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        lines = batch["lines"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad()
        logits, _ = model(lines, lengths=lengths)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        total_loss += loss.item()
   
    avg_training_loss = total_loss / len(train_dataloader)
    print(f"Train Loss: {avg_training_loss:.4f}")


def validation(val_dataloader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            lines = batch['lines'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch["lengths"]

            logits, _ = model(lines, lengths=lengths)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            val_loss += loss.item()

    avg_validation_loss = val_loss / len(val_dataloader)
    scheduler.step(avg_validation_loss)
    print(f"Val Loss: {avg_validation_loss:.4f}")

    should_stop = False
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        patience_counter = 0

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, model_path)
        print("✅ Модель сохранена!")
    else:
        patience_counter += 1
        print(f"⚠️Валидационная ошибка не уменьшилась ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("🛑 Early stopping")
            should_stop = True
    
    return should_stop


def generate(tokenizer):
    print("\nГенерации:")

    for prompt in ["i love", "thinking about", "where is"]:
        enc = tokenizer.encode(prompt)
        input_ids = enc.ids
        generated = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        hidden = None

        with torch.no_grad():
            # прогоняем весь prompt один раз
            logits, hidden = model(generated)

            for _ in range(20):
                next_token_logits = logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                eos_id = tokenizer.token_to_id("<eos>")
                if next_token.item() == eos_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

                # теперь подаём только последний токен
                logits, hidden = model(next_token, hidden=hidden)

        generated = tokenizer.decode(generated[0].tolist())
        print(f"'{prompt}' → {generated}")