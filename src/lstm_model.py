import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tokenizers import Tokenizer
import yaml

path = "./configs/config.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

tokenizer = Tokenizer.from_file(config["data"]["tokenizer_file_name"])

class AutoCompleteLSTM(nn.Module):
    def __init__(
            self, 
            vocab_size=tokenizer.get_vocab_size(), 
            embedding_dim=config["model"]["embedding_dim"], 
            hidden_dim=config["model"]["hidden_dim"], 
            num_layers=config["model"]["num_layers"], 
            dropout=config["model"]["dropout"]
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.token_to_id("<pad>"))
        self.dropout_emb = nn.Dropout(dropout)
        
        # Многослойная LSTM с dropout
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def forward(self, x, lengths=None, hidden=None):
        emb = self.embedding(x)
        emb = self.dropout_emb(emb) 
        
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.lstm(packed, hidden)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, hidden = self.lstm(emb, hidden)
        
        out = self.norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)