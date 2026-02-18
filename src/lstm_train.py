import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.lstm_model import AutoCompleteLSTM
from tokenizers import Tokenizer
import evaluate


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Пути
        self.save_dir = config["data"]["model_save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.save_dir, config["model"]["best_model_file"])

        self.tokenizer = Tokenizer.from_file(config["data"]["tokenizer_file_name"])

        # Модель
        self.model = AutoCompleteLSTM().to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        self.rouge = evaluate.load("rouge")

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = config["training"]["early_stopping"]["patience"]

        # Загрузка чекпоинта
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Модель загружена из чекпоинта")

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            lines = batch["lines"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"]

            self.optimizer.zero_grad()

            logits, _ = self.model(lines, lengths=lengths)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Train Loss: {avg_loss:.4f}")

    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                lines = batch["lines"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"]

                logits, _ = self.model(lines, lengths=lengths)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        self.scheduler.step(avg_loss)

        print(f"Val Loss: {avg_loss:.4f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0

            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }, self.model_path)

            print("Модель сохранена")
            return False

        else:
            self.patience_counter += 1
            print(f"Val loss не уменьшилась ({self.patience_counter}/{self.patience})")

            if self.patience_counter >= self.patience:
                print("Early stopping")
                return True

            return False

    def evaluate_rouge(self, test_dataloader, max_samples=100):
        self.model.eval()

        predictions = []
        references = []
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="ROUGE", leave=False):
                lines = batch["lines"].to(self.device)
                lengths = batch["lengths"]

                for i in range(lines.size(0)):
                    if num_samples >= max_samples:
                        break

                    seq_len = lengths[i].item()
                    full_seq = lines[i, :seq_len]

                    split_idx = int(0.75 * seq_len)

                    prompt = full_seq[:split_idx].unsqueeze(0)
                    target = full_seq[split_idx:]

                    logits, hidden = self.model(prompt)

                    generated_tokens = []

                    for _ in range(len(target)):
                        next_token = torch.argmax(
                            logits[:, -1, :],
                            dim=-1
                        )

                        generated_tokens.append(next_token.item())

                        logits, hidden = self.model(
                            next_token.unsqueeze(0),
                            hidden=hidden
                        )

                    pred_text = self.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )

                    ref_text = self.tokenizer.decode(
                        target.tolist(),
                        skip_special_tokens=True
                    )

                    predictions.append(pred_text)
                    references.append(ref_text)

                    num_samples += 1

                if num_samples >= max_samples:
                    break

        scores = self.rouge.compute(
            predictions=predictions,
            references=references
        )

        print(f"ROUGE-1: {scores['rouge1']:.4f}")

    def generate(self, max_len=20):
        self.model.eval()

        prompts = ["i love", "thinking about", "where is"]

        for prompt in prompts:
            enc = self.tokenizer.encode(prompt)
            input_ids = enc.ids

            generated = torch.tensor(
                input_ids,
                dtype=torch.long
            ).unsqueeze(0).to(self.device)

            hidden = None

            with torch.no_grad():
                logits, hidden = self.model(generated)

                for _ in range(max_len):
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    eos_id = self.tokenizer.token_to_id("<eos>")
                    if next_token.item() == eos_id:
                        break

                    generated = torch.cat([generated, next_token], dim=1)

                    logits, hidden = self.model(
                        next_token,
                        hidden=hidden
                    )

            text = self.tokenizer.decode(generated[0].tolist())
            print(f"{prompt} → {text}")