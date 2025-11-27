#!/usr/bin/env python3
"""
Plot training curves (loss and accuracy) by parsing run log files.
Searches for lines like:
  [ Train | 001/200 ] loss = 0.7780, acc = 0.1844
and plots loss and accuracy vs epoch.

Usage:
  python plot_training_from_runlogs.py /path/to/log1.log /path/to/log2.log ...

Saves PNGs into a `plots/` directory next to the given log files.
"""
import re
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('matplotlib not available:', e)
    print('Please install matplotlib (e.g. pip install matplotlib)')
    sys.exit(1)

TRAIN_RE = re.compile(r"\[\s*Train\s*\|\s*(\d+)/(\d+)\s*\]\s*loss\s*=\s*([0-9.eE+-]+)\s*,\s*acc\s*=\s*([0-9.eE+-]+)")
VAL_RE = re.compile(r"\[\s*Val\s*\|\s*(\d+)/(\d+)\s*\]\s*loss\s*=\s*([0-9.eE+-]+)\s*,\s*acc\s*=\s*([0-9.eE+-]+)")


def parse_log(path):
    train_epochs = []
    train_loss = []
    train_acc = []
    val_epochs = []
    val_loss = []
    val_acc = []
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            m = TRAIN_RE.search(line)
            if m:
                e = int(m.group(1))
                train_epochs.append(e)
                train_loss.append(float(m.group(3)))
                train_acc.append(float(m.group(4)))
                continue
            m2 = VAL_RE.search(line)
            if m2:
                e = int(m2.group(1))
                val_epochs.append(e)
                val_loss.append(float(m2.group(3)))
                val_acc.append(float(m2.group(4)))
                continue
    return {
        'train_epochs': train_epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_epochs': val_epochs,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }


def plot_results(results, logpath):
    run_dir = os.path.dirname(logpath)
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.basename(logpath).replace('.log','')

    # plot loss
    if results['train_epochs'] and results['train_loss']:
        plt.figure(figsize=(8,4))
        plt.plot(results['train_epochs'], results['train_loss'], marker='o', label='train')
        if results['val_epochs'] and results['val_loss']:
            plt.plot(results['val_epochs'], results['val_loss'], marker='x', label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'Loss vs epoch ({base})')
        plt.grid(True)
        plt.legend()
        outp = os.path.join(plots_dir, f'{base}_loss_vs_epoch.png')
        plt.tight_layout()
        plt.savefig(outp)
        plt.close()
        print('saved', outp)
    else:
        print('no training loss rows found in', logpath)

    # plot accuracy
    if results['train_epochs'] and results['train_acc']:
        plt.figure(figsize=(8,4))
        plt.plot(results['train_epochs'], results['train_acc'], marker='o', label='train')
        if results['val_epochs'] and results['val_acc']:
            plt.plot(results['val_epochs'], results['val_acc'], marker='x', label='val')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(f'Accuracy vs epoch ({base})')
        plt.grid(True)
        plt.legend()
        outp = os.path.join(plots_dir, f'{base}_acc_vs_epoch.png')
        plt.tight_layout()
        plt.savefig(outp)
        plt.close()
        print('saved', outp)
    else:
        print('no training acc rows found in', logpath)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python plot_training_from_runlogs.py /path/to/log1.log [/path/to/log2.log ...]')
        sys.exit(1)
    paths = sys.argv[1:]
    for p in paths:
        if not os.path.exists(p):
            print('not found', p)
            continue
        print('parsing', p)
        res = parse_log(p)
        plot_results(res, p)
