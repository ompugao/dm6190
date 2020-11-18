#!/bin/bash
source /home/leus/.pyenv/versions/pytorch/bin/activate
python test.py -model=unet -logdir='logs/unet_freeze_decoder_2020-09-15 15-35-27.405352' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=unet -logdir='logs/unet_unfreeze_decoder_2020-09-15 16-02-05.704776' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=fpn -logdir='logs/fpn_freeze_decoder_2020-09-15 16-27-39.525568' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=fpn -logdir='logs/fpn_unfreeze_decoder_2020-09-15 16-50-28.006504' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=deeplabv3 -logdir='logs/deeplabv3_freeze_decoder_2020-09-15 17-13-17.112977' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=deeplabv3 -logdir='logs/deeplabv3_unfreeze_decoder_2020-09-16 00-51-21.956395' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=deeplabv3plus -logdir='logs/deeplabv3plus_freeze_decoder_2020-09-16 08-29-26.554970' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=deeplabv3plus -logdir='logs/deeplabv3plus_unfreeze_decoder_2020-09-16 09-10-49.252822' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=pan -logdir='logs/pan_freeze_decoder_2020-09-16 09-52-11.379602' -augmentation_version=0 -imgheight=320 -imgwidth=480
python test.py -model=pan -logdir='logs/pan_unfreeze_decoder_2020-09-16 10-33-12.988805' -augmentation_version=0 -imgheight=320 -imgwidth=480

python test.py -model=unet -logdir='logs/unet_freeze_decoder_2020-09-16 11-14-03.715160' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=unet -logdir='logs/unet_unfreeze_decoder_2020-09-16 12-27-43.885046' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=fpn -logdir='logs/fpn_freeze_decoder_2020-09-16 13-41-20.060483' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=fpn -logdir='logs/fpn_unfreeze_decoder_2020-09-16 14-43-17.875869' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=pan -logdir='logs/pan_freeze_decoder_2020-09-17 11-27-21.809318' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=pan -logdir='logs/pan_unfreeze_decoder_2020-09-17 11-28-05.393323' -augmentation_version=0 -imgheight=640 -imgwidth=960


