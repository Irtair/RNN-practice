from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence, NFD, StripAccents

def train_tokenizer():
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Нормализация: lowercase + удаление акцентов
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # Разбивка по пробелам
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=12000,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2
    )

    tokenizer.train(files=["data\\dataset_processed.txt"], trainer=trainer)
    tokenizer.save("wordlevel.json")