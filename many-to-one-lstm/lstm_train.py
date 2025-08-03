import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os, re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import warnings

import nltk
#nltk.download('stopwords') 
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

data_dir = "../data"

def pre_process_text(all_files):
    all_words, seq_len = [], []
    for f_name in all_files:
        text = open(f_name).readlines()[0].lower()
        text = re.sub ( r'[^\w\s]', '', text)
        words = text.split(" ")
        words = [w for w in words if (w not in stopwords) and (len(w) >=0) ]
        all_words+=words
        seq_len.append(len(words))
    return (all_words, seq_len)

train_dir = f"{data_dir}/aclImdb/train"

all_files = ([ os.path.join ( train_dir,  f"pos/{f_name}") for f_name in  os.listdir(f"{train_dir}/pos")] + 
             [ os.path.join ( train_dir,  f"neg/{f_name}") for f_name in  os.listdir(f"{train_dir}/neg")] )

train_words, sentence_len = pre_process_text(all_files)


print(f"avg sentence length: {np.mean(sentence_len)}")


## Tokenizer
bog = dict(Counter(train_words))
words = sorted([key for (key,value) in bog.items() if value > 500])

words.append("<UNK>")
words.append("<PAD>")

w2i = {w: i for i, w in enumerate(words)}
i2w = {i: w for i, w in enumerate(words)}


## Data loader
class IMDBDataLoader(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len = 200):
        self.tokenizer = w2i
        self.max_seq_len = max_seq_len
        self.data_files = ([ os.path.join ( data_path,  f"pos/{f_name}") for f_name in  os.listdir(f"{data_path}/pos")] + 
             [ os.path.join ( data_path,  f"neg/{f_name}") for f_name in  os.listdir(f"{data_path}/neg")] )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]

        def get_sample(f_name):
            text = open(f_name).readlines()[0].lower()
            text = re.sub ( r'[^\w\s]', '', text)
            words = text.split(" ")
            return [w for w in words if (w not in stopwords) and (len(w) >=0) ]

        def get_tokenzied_sample_seq(f_name):
            sample = get_sample(f_name)
            # if more than seq_len, trim it
            if len(sample) > self.max_seq_len:
                rand_start_idx = np.random.randint( (len(sample) - self.max_seq_len) )
                sample = sample[rand_start_idx: (rand_start_idx + self.max_seq_len) ]

            ## tokenized result
            tokenized = []
            for w in sample:
                if w in self.tokenizer:
                    tokenized.append(self.tokenizer[w])
                else:
                    tokenized.append(self.tokenizer["<UNK>"])
            
            sample = torch.tensor(tokenized)
            return sample

        sample = get_tokenzied_sample_seq(file_path)

        ## label
        label = 1
        if "neg" in file_path: label = 0
        return sample, label

def data_collator(batch):
    word_tokens, labels = [], []
    for token, label in batch:
        labels.append(label)
        word_tokens.append(token)

    labels = torch.tensor(labels)
    
    word_tokens = nn.utils.rnn.pad_sequence(word_tokens, batch_first=True, padding_value=w2i["<PAD>"])
    return word_tokens, labels

train_ds = IMDBDataLoader(train_dir, tokenizer=w2i)

train_data_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=True, collate_fn=data_collator)

for (s, l) in train_data_loader:
    # (B x T x C)
    print(s, l)
    break

## Defining LSTM 
class LSTMNet(nn.Module):
    def __init__(self, emb_dim, hidden_size, vocab_size, n_layers, n_outs):
        super().__init__()
        self.emb_dim = emb_dim          # input to emb
        self.hidden_size = hidden_size  # LSTM internal NN size
        self.vocab_size = vocab_size    # len(tokenizer)
        self.n_layers = n_layers        # stacked lstm layers
        self.n_outs = n_outs            # output class

        ## project input
        self.embs = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_dim)
        ## define lstm
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)
        ## out classifier
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, self.n_outs)

    def forward(self, x):
        embs = self.embs(x)  # B x seq_len x emb_dim
        ## pass through lstm
        outputs, (hn, cn) = self.lstm(embs)
        # Take last out and return out class
        last_neuron = outputs[:, -1, :] # last step neuron ; B x hddn_size
        out = self.dropout(last_neuron)
        out = self.fc(out) # B x 1 x hddn -> B x vocab_size
        return out 
    
model = LSTMNet(emb_dim=8, hidden_size=10, vocab_size=len(w2i), n_layers=2, n_outs=2)
print(model)
for x, label in train_data_loader:
    print(x.shape, label.shape)
    outs = model(x)
    print(outs)
    break

### 
## define dataset
train_ds = IMDBDataLoader("../data/aclImdb/train", tokenizer=w2i)
val_ds = IMDBDataLoader("../data/aclImdb/test", tokenizer=w2i)
train_data_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, collate_fn=data_collator)
val_data_loader = DataLoader(val_ds, batch_size=128, shuffle=True, collate_fn=data_collator)


## defining pre-train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
batch_size = 128

## model
model = LSTMNet(
    emb_dim=128,
    hidden_size=256,
    vocab_size=len(w2i), 
    n_layers=2,
    n_outs=2
)
model = model.to(DEVICE)
print(f"{model = }")

optm = optim.Adam(params=model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()


## Defining training loop
def train(epochs, model, train_data_loader, val_data_loader, optm, loss_fn):
    logs = {
        "epoch" : [],
        "training_loss": [],
        "val_loss": [],
        "training_acc": [],
        "val_acc": [],
    }
    save_path = "best_model.pt"
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}/ {epochs}")
        train_acc, val_acc, train_loss, val_loss = [], [], [], []

        model.train()
        for x, labels in tqdm(train_data_loader, desc = "Training"):
            ## data through model
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optm.zero_grad()
            outputs = model(x)

            ## calc acc
            preds = torch.argmax(outputs, axis=1)
            acc = ( (preds == labels).sum().float() )/ len(preds)
            train_acc.append(acc.item())

            ## Loss 
            loss = loss_fn(outputs, labels)
            train_loss.append(loss.item())
            loss.backward()

            ## clip exp gradiants
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optm.step()

        model.eval()
        for x, labels in tqdm(val_data_loader, desc = "Validation"):
            ## data through model
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                ## forward
                outputs = model(x)

                ## calc acc
                preds = torch.argmax(outputs, axis=1)
                acc = ( (preds == labels).sum().float() )/ len(preds)
                val_acc.append(acc.item())

                ## Loss 
                loss = loss_fn(outputs, labels)
                val_loss.append(loss.item())

        logs["epoch"].append(epoch)
        logs["training_acc"].append(np.mean(train_acc))
        logs["val_acc"].append(np.mean(val_acc))
        logs["training_loss"].append(np.mean(train_loss))
        logs["val_loss"].append(np.mean(val_loss))

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {logs['training_loss'][-1]:.4f} | Val Loss: {logs['val_loss'][-1]:.4f} | "
            f"Train Acc: {logs['training_acc'][-1]:.4f} | Val Acc: {logs['val_acc'][-1]:.4f}"
        )

    if logs["val_loss"][-1] < best_val_loss:
        best_val_loss = logs["val_loss"][-1]
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")

    return logs, model


## train the model
train_logs, model = train(
    epochs=15,
    model=model,
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    optm=optm,
    loss_fn=loss_fn
)

torch.save(model.state_dict(), "final_model.pt")
