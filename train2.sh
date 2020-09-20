#!/bin/bash
#set -eu
source /home/leus/.pyenv/versions/pytorch/bin/activate
echo python train.py -model=unet -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=unet_freeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=unet -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=unet_freeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=unet -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=unet_unfreeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=unet -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=unet_unfreeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=fpn -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=fpn_freeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=fpn -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=fpn_freeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=fpn -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=fpn_unfreeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=fpn -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=fpn_unfreeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3_freeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3_freeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3_unfreeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3_unfreeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3plus_freeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3plus_freeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3plus_unfreeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=deeplabv3plus_unfreeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=pan -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=pan_freeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=pan -freeze_decoder=1 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=pan_freeze_decoder_aug1_480_320_ -gpu=0
echo python train.py -model=pan -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=pan_unfreeze_decoder_aug1_480_320_ -gpu=0
python train.py -model=pan -freeze_decoder=0 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdirprefix=pan_unfreeze_decoder_aug1_480_320_ -gpu=0
