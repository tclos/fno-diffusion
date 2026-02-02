# fno_diffusion/train.py

import torch
import torch.nn as nn


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device).permute(0, 2, 1)
        y = y.to(device).permute(0, 2, 1)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device).permute(0, 2, 1)
        y = y.to(device).permute(0, 2, 1)

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

    return total_loss / len(loader)
