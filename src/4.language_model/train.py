import argparse
import torch
import os, sys, inspect
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src import config

def train(dataset, model, device, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        state_h.to(device)
        state_c.to(device)
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                optimizer.zero_grad()
                x.to(device)
                y.to(device)

                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                # print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--char', type=str, default="michael")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    args = parser.parse_args()
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Training is using: \n \t {device}")
    print()
    dataset = Dataset(args, device)
    model = Model(dataset, device)
    model.to(device)

    train(dataset, model, device, args)
    save_path = "tmp"
    save_folder = os.path.join(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(model.state_dict(), os.path.join(config.model_save_path, f"{args.char}.language_model.pth"))
