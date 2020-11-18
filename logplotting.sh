#!/bin/bash
source /home/leus/.pyenv/versions/dm6190/bin/activate
python log_plotter.py -eventpath='logs/unet_freeze_decoder_2020-09-15 15-35-27.405352' -imgdir='logs/unet_freeze_decoder_2020-09-15 15-35-27.405352'
python log_plotter.py -eventpath='logs/unet_unfreeze_decoder_2020-09-15 16-02-05.704776' -imgdir='logs/unet_unfreeze_decoder_2020-09-15 16-02-05.704776'
python log_plotter.py -eventpath='logs/fpn_freeze_decoder_2020-09-15 16-27-39.525568' -imgdir='logs/fpn_freeze_decoder_2020-09-15 16-27-39.525568'
python log_plotter.py -eventpath='logs/fpn_unfreeze_decoder_2020-09-15 16-50-28.006504' -imgdir='logs/fpn_unfreeze_decoder_2020-09-15 16-50-28.006504'
python log_plotter.py -eventpath='logs/deeplabv3_freeze_decoder_2020-09-15 17-13-17.112977' -imgdir='logs/deeplabv3_freeze_decoder_2020-09-15 17-13-17.112977'
python log_plotter.py -eventpath='logs/deeplabv3_unfreeze_decoder_2020-09-16 00-51-21.956395' -imgdir='logs/deeplabv3_unfreeze_decoder_2020-09-16 00-51-21.956395'
python log_plotter.py -eventpath='logs/deeplabv3plus_freeze_decoder_2020-09-16 08-29-26.554970' -imgdir='logs/deeplabv3plus_freeze_decoder_2020-09-16 08-29-26.554970'
python log_plotter.py -eventpath='logs/deeplabv3plus_unfreeze_decoder_2020-09-16 09-10-49.252822' -imgdir='logs/deeplabv3plus_unfreeze_decoder_2020-09-16 09-10-49.252822'
python log_plotter.py -eventpath='logs/pan_freeze_decoder_2020-09-16 09-52-11.379602' -imgdir='logs/pan_freeze_decoder_2020-09-16 09-52-11.379602'
python log_plotter.py -eventpath='logs/pan_unfreeze_decoder_2020-09-16 10-33-12.988805' -imgdir='logs/pan_unfreeze_decoder_2020-09-16 10-33-12.988805'

