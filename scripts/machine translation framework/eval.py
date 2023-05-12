import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuda import get_memory_usage
from dataset import OurDS as Dataset
from logger import Logger
from models import OurModel as Model

def setup_save_dir(args) -> Path:
    save_dir = Path(args.save_dir)
    assert save_dir.is_dir()
    return save_dir


def setup_logger(save_dir: Path) -> None:
    if (save_dir / 'eval.log').is_file():
        (save_dir / 'eval.log').unlink()
    sys.stdout = Logger(sys.stdout, save_dir / 'eval.log')


def setup_dataloader(args) -> DataLoader:
    testloader = torch.utils.data.DataLoader(
        Dataset(mode = "val"),
        batch_size=None, shuffle=False, num_workers=1)
    return testloader


def setup_model(save_dir, args) -> Module:
    model = Model()
    model.load_state_dict(torch.load(save_dir / 'weights.pt'))
    model.to(args.device)
    return model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def indices_to_text(indices, field):
    tokens = [field.vocab.itos[i] for i in indices]
    # Remove <sos> and <eos> tokens
    tokens = [token for token in tokens if token not in ['<sos>', '<eos>']]
    text = ' '.join(tokens)
    return text

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    for batch in tqdm(iterator, total=len(iterator), desc="Evaluating"):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        output = model(src, trg, 0) # Turn off teacher forcing
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        output_text = indices_to_text(torch.argmax(output, dim=1).cpu().numpy(), TRG)
        trg_text = indices_to_text(trg.cpu().numpy(), TRG)
        
        scores = criterion(output_text, trg_text)
        loss = (scores['rouge1_fmeasure'] + scores['rouge2_fmeasure'] + scores['rougeL_fmeasure']) / 3
        epoch_loss += loss

    avg_loss = epoch_loss / len(iterator)
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, help='saving directory')
    parser.add_argument(
        '--device', default='cpu',
        help='the device on which a torch.Tensor is or will be allocated')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    save_dir = setup_save_dir(args)
    setup_logger(save_dir)
    testloader = setup_dataloader(args)
    model = setup_model(save_dir, args)
    df = eval(model, testloader, save_dir, args)
    df.to_csv(save_dir / 'eval.csv', index=False)

    s = f'pid = {os.getpid()}\n'
    s += f'GPU_memory_usage = {get_memory_usage()}\n'
    for c in ['Dice', 'BertScore', 'NRMSE']: # List of our metrics
        s += f"Input_{c} = {np.mean(df['Input'][c].to_list())} ± {np.std(df['Input'][c].to_list())}\n"
        s += f"Prediction_{c} = {np.mean(df['Prediction'][c].to_list())} ± {np.std(df['Prediction'][c].to_list())}\n"
    print(s)
