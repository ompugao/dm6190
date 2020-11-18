#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from scipy.signal import lfilter

# exponential weighted moving average
def ema_filter(x, alpha):
    y,zf = lfilter([alpha], [1,alpha-1], x, zi=[x[0]*(1-alpha)])
    return y

parser = argparse.ArgumentParser()
parser.add_argument("-eventpath", help="log event file", type=str)
parser.add_argument("-eventpaths", help="log event files", type=str)
parser.add_argument("-imgpath", help="path to save image", type=str)
parser.add_argument("-imgdir", help="dir to save image", type=str)
args = parser.parse_args()

event_acc = EventAccumulator(args.eventpath, size_guidance={'scalars': 0})
event_acc.Reload() # ログファイルのサイズによっては非常に時間がかかる

scalars = {}
for tag in event_acc.Tags()['scalars']:
    events = event_acc.Scalars(tag)
    scalars[tag] = [event.value for event in events]

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')
#sns.set_context("paper", 10, {"lines.linewidth": 4})
plt.rcParams["font.size"] = 36
plt.rcParams['figure.figsize'] = (20,10)


def plot(scalars, imgpath=None):
    #sns.lineplot(range(len(scalars['Loss/train_epoch_loss'])), scalars['Loss/train_epoch_loss'])
    #plt.plot(scalars['Loss/train_epoch_loss'])
    #plt.plot(scalars['Loss/train_epoch_loss'])
    plt.plot(ema_filter(scalars['Loss/train_epoch_loss'], 0.6), label='train loss', linewidth=3.0)
    plt.plot(ema_filter(scalars['Loss/val_epoch_loss'], 0.6), label='validation loss', linewidth=3.0)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)#, fontsize=40)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if imgpath is None:
        plt.show()
    else:
        plt.savefig(imgpath, dpi=300)

if args.imgpath is '' and args.imgdir is '':
    plot(scalars)
elif args.imgpath is not '':
    plot(scalars, args.imgpath)
elif args.imgdir is not '':
    plot(scalars, args.imgdir + '/loss.png')
else:
    print('???')


