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


def eval(model: Module, testloader: DataLoader, save_dir: Path, args) -> pd.DataFrame:
    model.eval()
    all_df = list()
    with tqdm(total=len(testloader), dynamic_ncols=True) as pbar, torch.no_grad():
        for x,y in testloader:

            input_metrics = dict()
            pred_metrics = dict()
            input_metrics['Score1'] = # Our implementations of score
            pred_metrics['Score2'] = # Our implementations of score
            input_metrics['Score1'] = # Our implementations of score
            pred_metrics['Score2'] = # Our implementations of score
            input_metrics['Score2'] = # Our implementations of score
            pred_metrics['Score3'] = # Our implementations of score

            input_df = pd.DataFrame.from_dict(input_metrics, orient='index').T
            pred_df = pd.DataFrame.from_dict(pred_metrics, orient='index').T
            all_df.append(pd.concat([input_df, pred_df], axis=1, keys=['Input', 'Prediction']))

            pbar.update()
            if args.debug and pbar.n == 2:
                break

    return pd.concat(all_df)


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
