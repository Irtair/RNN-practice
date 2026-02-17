import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence, NFD, StripAccents

def train_tokenizer(processed_path, filename):
    if os.path.exists(filename):
        print("Токенайзер уже прошел тренировку")
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

        # Нормализация: lowercase + удаление акцентов
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

        # Разбивка по пробелам
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            vocab_size=12000,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2,
            show_progress=True
        )

        tokenizer.train(files=[processed_path], trainer=trainer)
        tokenizer.save(filename)
        
        print("Тренировка завершена")