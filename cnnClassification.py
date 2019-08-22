import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtext import data, vocab
import random
import torch.optim as optim

SEED = 1234
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
GLOVE = 'glove.840B.300d.txt'

EMBEDDING_PATH = GLOVE_PATH
MAX_SIZE = 120000
MAX_LEN = 70
SPLIT_RATIO = 0.9
BATCH_SIZE = 512
HIDDEN_DIM = 32
N_LAYERS = 2

ID = data.Field()
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_fields = [('id', None), ('text', TEXT), ('target', LABEL)]
test_fields = [('id', ID), ('text', TEXT)]
train_data = data.TabularDataset(path=TRAIN_PATH,format='csv',skip_header=True,fields=train_fields)
test_data = data.TabularDataset(path=TEST_PATH,format='csv',skip_header=True,fields=test_fields)
train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO, random_state=random.seed(SEED))

vec = vocab.Vectors(EMBEDDING_PATH)
TEXT.build_vocab(train_data, vectors=vec, max_size=MAX_SIZE, unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)
word_embeddings = TEXT.vocab.vectors
ID.build_vocab(test_data)
txt_len = len(TEXT.vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True, device=device)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_fil, fil_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_fil, kernel_size=(fs, embedding_dim)) for fs in fil_sizes])
        self.fc = nn.Linear(len(fil_sizes) * n_fil, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in self.convs]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

model.embedding.weight.data.copy_(word_embeddings)
UNK = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters())
loss_crit = nn.BCEWithLogitsLoss()
model = model.to(device)
loss_crit = loss_crit.to(device)


def bin_acc(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, loss_crit):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss_crit = loss_crit(predictions, batch.label)
        acc = bin_acc(predictions, batch.label)
        loss_crit.backward()
        optimizer.step()
        epoch_loss += loss_crit.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc . len(iterator)


def evaluate(model, iterator, loss_crit):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = loss_crit(predictions, batch.label)
            acc = bin_acc(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

N_EPOCHS = 10
best_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iter, optimizer, loss_crit)
    valid_loss, valid_acc = evaluate(model, valid_iter, loss_crit)
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('tut4-model.pt'))
test_loss, test_acc = evaluate(model, test_iter, loss_crit)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')