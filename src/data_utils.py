import os
import torch
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer


# Регулярные выражения
re_user = re.compile(r'@\w+\s*')
re_links = re.compile(r'https?://\S+|www\.\S+')
re_all_but_base = re.compile(r'[^a-zA-Z0-9\s]')
re_double_spaces = re.compile(r'\s{2,}')


def clear_text_line(line):
    line = line.casefold()
    line = re_user.sub('', line)
    line = re_links.sub('', line)
    line = re_all_but_base.sub('', line)
    line = re_double_spaces.sub(' ', line)
    return line.strip()


class TweeterDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        seq = self.text[idx]
        return {
            "line": torch.tensor(seq[:-1], dtype=torch.long),
            "label": torch.tensor(seq[1:], dtype=torch.long),
        }


data_dir = "data"
processed_path = os.path.join(data_dir, "dataset_processed.txt")
raw_path = os.path.join(data_dir, "tweets.txt")

print("Загрузка/обработка данных...")

if os.path.exists(processed_path):
    with open(processed_path, 'r', encoding='utf-8') as f:
        cleaned = [line.strip() for line in f if line.strip()]
else:
    os.makedirs(data_dir, exist_ok=True)

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_lines = [line for line in f]

    cleaned = list(tqdm(
        map(clear_text_line, raw_lines),
        total=len(raw_lines),
        desc="Очистка текста"
    ))

    with open(processed_path, 'w', encoding='utf-8') as f:
        for line in cleaned:
            if line:
                f.write(line + "\n")


print("Начало токенизации...")
tokenized_path = os.path.join(data_dir, "tokenized.pt")

if os.path.exists(tokenized_path):
    print("📦 Загрузка токенизированных данных из кэша...")
    tokenized = torch.load(tokenized_path)
    print(f"✅ Загружено {len(tokenized)} последовательностей")
else:
    from tokenizer import train_tokenizer
    train_tokenizer()

    tokenizer = Tokenizer.from_file("wordlevel.json")

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    encodings = tokenizer.encode_batch(cleaned)

    tokenized = []
    for enc in tqdm(encodings, desc="Токенизация"):
        ids = enc.ids[:62]
        ids = [bos_id] + ids + [eos_id]

        if len(ids) >= 3:
            tokenized.append(ids)

    torch.save(tokenized, tokenized_path)
    print(f"✅ Токенизация завершена и сохранена. Всего: {len(tokenized)}")


def collate_fn(batch):
    lines = [item['line'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = torch.tensor([len(line) for line in lines], dtype=torch.long)

    padded_lines = pad_sequence(
        lines,
        batch_first=True,
        padding_value=pad_id
    )

    padded_labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        'lines': padded_lines,
        'labels': padded_labels,
        'lengths': lengths,
    }


# Разбиение 80/10/10
tokenized_ds_len = len(tokenized)

train_lines = tokenized[:int(0.8 * tokenized_ds_len)]
val_lines = tokenized[int(0.8 * tokenized_ds_len):int(0.9 * tokenized_ds_len)]
test_lines = tokenized[int(0.9 * tokenized_ds_len):]

print(f"📊 Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")

train_dataset = TweeterDataset(train_lines)
val_dataset = TweeterDataset(val_lines)
test_dataset = TweeterDataset(test_lines)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=320,
    shuffle=True,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=128,
    collate_fn=collate_fn,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=256,
    collate_fn=collate_fn,
)