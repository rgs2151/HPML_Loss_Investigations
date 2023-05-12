import argparse
import os
from pathlib import Path
import sys

import numpy as np
from skimage import metrics
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuda import get_memory_usage
from dataset import OurDataset
from logger import Logger
from models import OurModel as Model


def setup_save_dir(args) -> Path:
    save_dir = Path(args.save_dir)
    if args.resume:
        assert save_dir.is_dir()
    else:
        save_dir.mkdir(exist_ok=True)
    return save_dir


def setup_logger(resume: bool, save_dir: Path) -> None:
    if not resume and (save_dir / 'train.log').is_file():
        (save_dir / 'train.log').unlink()
    sys.stdout = Logger(sys.stdout, save_dir / 'train.log')


def setup_dataloader(args) -> tuple[DataLoader, DataLoader]:
    trainloader = torch.utils.data.DataLoader(
        OurDataset(mode='train'),
        batch_size=1, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(
        OurDataset(mode='val'),
        batch_size=None, shuffle=False, num_workers=1)
    return trainloader, valloader


def setup_model(save_dir: Path, args) -> Module:
    model = Model()
    if args.resume:
        model.load_state_dict(torch.load(save_dir / 'weights.pt'))
    else:
        model.aft.load_state_dict(torch.load(args.AFT))
        model.rescunet.load_state_dict(torch.load(args.ResCUNet))
    model.to(args.device)
    return model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def setup_optimizer(model: Module):
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-3},])
    return optimizer


def setup_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=1 / 10**.5, patience=2, verbose=True)
    return scheduler


def setup_loss_fun(args):
    loss_func = getattr(torch.nn.functional, args.loss_func)
    return loss_func


class TqdmExtraFormat(tqdm):
    """
    Provides a `total_time` format parameter
    Ref: https://github.com/tqdm/tqdm#description-and-additional-stats
    """
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time))
        return d

def indices_to_text(indices, field):
    tokens = [field.vocab.itos[i] for i in indices]
    # Remove <sos> and <eos> tokens
    tokens = [token for token in tokens if token not in ['<sos>', '<eos>']]
    text = ' '.join(tokens)
    return text

def train_epoch(model, iterator, optimizer, criterion, scorer, clip=1):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, total=len(iterator), desc="Training"):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        if scorer == "CrossEntropyLoss":
            pass

        elif scorer == "BERTScore":
            # Convert output and targets to text strings
            output_text = indices_to_text(torch.argmax(output, dim=1).cpu().numpy(), TRG)
            trg_text = indices_to_text(trg.cpu().numpy(), TRG)
            # print(f'Output: {output_text}')
            # print(f'Target: {trg_text}')

            P, R, F1 = bert_based_scorer.score([output_text], [trg_text])
            reward = F1.mean().item()
            # print("previous loss:", loss)
            # print("previous loss:", loss.item())
            loss *= (1 - reward)
            # print("post loss:", loss.item())

        elif scorer == "BLEU":
            output_text = indices_to_text(torch.argmax(output, dim=1).cpu().numpy(), TRG)
            trg_text = indices_to_text(trg.cpu().numpy(), TRG)

            # Compute BLEU score
            bleu_score = sentence_bleu([trg_text.split()], output_text.split())
            loss *= (1 - bleu_score)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()
        
        # break

    avg_loss = epoch_loss / len(iterator)
    return avg_loss


def val_epoch(model: Module, valloader: DataLoader, args):
    model.eval()
    input_ssim = list()
    pred_ssim = list()
    with tqdm(total=len(valloader), dynamic_ncols=True) as pbar, torch.no_grad():
        for x,y  in valloader:
            pred = torch.cat([x, y], dim=1)

            input_ssim.append(
                metrics.ourmetric(y_hat, y)
            )
            pred_ssim.append(
                metrics.ourmetric(y_hat, y)
            )

            pbar.update()
    return input_ssim, pred_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, help='saving directory, create if it does not exist')
    parser.add_argument('--loss_func', default='mse_loss', choices=['mse_loss', 'l1_loss'])
    parser.add_argument('--device', default='cpu', help='the device on which a torch.Tensor is or will be allocated')
    parser.add_argument('--resume', action='store_true', help='continue')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    save_dir = setup_save_dir(args)
    setup_logger(args.resume, save_dir)
    trainloader, valloader = setup_dataloader(args)
    model = setup_model(save_dir, args)
    optimizer = setup_optimizer(model)
    scheduler = setup_scheduler(optimizer)
    loss_func = setup_loss_fun(args)

    best_metric = 0
    for epoch in range(1, 2):
        all_loss = train_epoch(model, trainloader, loss_func, optimizer, args)
        input_ssim, pred_ssim = val_epoch(model, valloader, args)

        if np.mean(pred_ssim) > np.mean(best_metric):
            torch.save(model.state_dict(), save_dir / 'weights.pt')
            best_metric = pred_ssim

        s = f'Epoch {epoch:05d}:\n'
        s += f'    loss = {np.mean(all_loss)} ± {np.std(all_loss)}\n'
        s += f'    pid = {os.getpid()}\n'
        s += f'    GPU_memory_usage = {get_memory_usage()}\n'
        s += f'    input_ssim = {np.mean(input_ssim)} ± {np.std(input_ssim)}\n'
        s += f'    pred_ssim = {np.mean(pred_ssim)} ± {np.std(pred_ssim)}\n'
        s += f'    best_ssim = {np.mean(best_metric)} ± {np.std(best_metric)}\n'
        print(s)

        scheduler.step(np.mean(pred_ssim))
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break

        if args.debug:
            break
