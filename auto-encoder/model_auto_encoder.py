import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import torch.optim as optim
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


transform = transforms.Compose(
    [
        transforms.Resize((32,32)), 
        transforms.ToTensor()
    ]
)

train_set = MNIST("../data/mnist/", train=True, transform=transform)
test_set = MNIST("../data/mnist/", train=False, transform=transform)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels = 1, bottleneck = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d( in_channels=in_channels, out_channels= 8, kernel_size=3, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d( in_channels=8, out_channels= 16, kernel_size=3, padding=1, stride=2, bias=False ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d( in_channels=16, out_channels= bottleneck, kernel_size=3, padding=1, stride=2),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( in_channels = bottleneck, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d( in_channels = 16, out_channels=8, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d( in_channels = 8, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward_enc(self, x):
        return self.encoder(x)

    def forward_dec(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

model = ConvAutoEncoder().to(DEVICE)
optm = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train_model(max_steps, model, train_data_loader, val_data_loader, optm, loss_fn):
    logs = {
        "step" : [],
        "training_loss": [],
        "val_loss": [],
    }
    eval_iter, continue_train = 1, True
    train_loss, eval_loss, step_counter = [], [], 0
    encoded_evals = []
    pbar = tqdm(range(max_steps))

    while continue_train:
        for x, labels in train_data_loader:
            step_counter += 1

            x = x.to(DEVICE)
            enc, dec = model(x)
            loss = loss_fn(dec, x)
            optm.zero_grad()
            loss.backward()
            optm.step()
            train_loss.append(loss.item())

            if step_counter % eval_iter == 0:
                logs["training_loss"].append(np.mean(train_loss))
                logs["step"].append(step_counter)
                encoded_data = []

                with torch.no_grad():
                    for x, labels in val_data_loader:
                        x = x.to(DEVICE)
                        enc, dec = model(x)
                        loss = criterion(dec, x)
                        eval_loss.append(loss.item())

                        enc, labels = enc.cpu().flatten(1), labels.reshape(-1,1)
                        encoded_data.append(torch.cat((enc, labels), axis=-1))
                
                encoded_evals.append(torch.concatenate(encoded_data))
                logs["val_loss"].append(np.mean(eval_loss))

                train_loss, eval_loss = [], []

            print(
                f"Step {step_counter}/ {max_steps} | "
                f"Train Loss: {logs['training_loss'][-1]:.4f} | Val Loss: {logs['val_loss'][-1]:.4f} "
            )

            pbar.update(1)
            if step_counter >= max_steps:
                print("Training completed")
                continue_train = False
                break
        
        return model, logs, encoded_evals

trained_model, logs, encoded_evals = train_model(model=model, max_steps=1,
                    train_data_loader=train, val_data_loader=val, optm=optm, loss_fn=criterion)