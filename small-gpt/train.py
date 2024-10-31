# from model import Transformer
# took model from /mlthings/transformer

from utils import get_loader, train

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers.normalizers import BertNormalizer

# Normal and tokenizer
normalizer = BertNormalizer(lowercase=True, strip_accents=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Defining Parameters (Using base(C) from paper)
num_of_layers = 6
d_model = 256
d_ff = 2048
h = 8
dropout = 0.1
src_vocab_size = tokenizer.vocab_size+1 # for padding token
tgt_vocab_size = tokenizer.vocab_size+1  # for padding token
seq_len = 32 # dataset has too small sentences

epochs  =100
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating DataLoader
train_loader = get_loader('/kaggle/input/custom-squadv2/final_train.csv', tokenizer, normalizer, batch_size)
test_loader = get_loader('/kaggle/input/custom-squadv2/final_validation.csv', tokenizer, normalizer, batch_size)

#  Defining Model
model = Transformer(num_of_layers, d_model, h, dropout, d_ff, src_vocab_size, tgt_vocab_size, seq_len)

# Train Model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
