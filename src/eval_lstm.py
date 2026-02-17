import os
import torch
from lstm_model import AutoCompleteLSTM
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("wordlevel.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "models"
model_path = os.path.join(save_dir, "best_model.pt")
model = AutoCompleteLSTM().to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

model.eval()
input = input("Enter prompt: ")

enc = tokenizer.encode(input)
input_ids = enc.ids
generated = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

hidden = None

with torch.no_grad():
    # прогоняем весь prompt один раз
    logits, hidden = model(generated)

    for _ in range(50):
        next_token_logits = logits[:, -1, :] / 1.1
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        eos_id = tokenizer.token_to_id("<eos>")
        if next_token.item() == eos_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

        # теперь подаём только последний токен
        logits, hidden = model(next_token, hidden=hidden)

generated = tokenizer.decode(generated[0].tolist())

print(f"'{input}' → {generated}")