#!/bin/bash
#set -eu
source /home/leus/.pyenv/versions/pytorch/bin/activate
echo python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/unet_freeze_decoder_aug1_960_640_2020-09-17 15-03-43.588931/' -gpu=1
python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/unet_freeze_decoder_aug1_960_640_2020-09-17 15-03-43.588931/' -gpu=1
echo python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/unet_unfreeze_decoder_aug1_960_640_2020-09-17 22-02-50.093149/' -gpu=1
python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/unet_unfreeze_decoder_aug1_960_640_2020-09-17 22-02-50.093149/' -gpu=1
echo python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/fpn_freeze_decoder_aug1_960_640_2020-09-18 04-56-35.049183/' -gpu=1
python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/fpn_freeze_decoder_aug1_960_640_2020-09-18 04-56-35.049183/' -gpu=1
echo python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/fpn_unfreeze_decoder_aug1_960_640_2020-09-18 11-46-27.800628/' -gpu=1
python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/fpn_unfreeze_decoder_aug1_960_640_2020-09-18 11-46-27.800628/' -gpu=1
echo python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3_freeze_decoder_aug1_960_640_2020-09-18 18-35-17.814975/' -gpu=1
python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3_freeze_decoder_aug1_960_640_2020-09-18 18-35-17.814975/' -gpu=1
echo python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3_unfreeze_decoder_aug1_960_640_2020-09-18 18-35-28.045851/' -gpu=1
python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3_unfreeze_decoder_aug1_960_640_2020-09-18 18-35-28.045851/' -gpu=1
echo python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3plus_freeze_decoder_aug1_960_640_2020-09-18 18-35-36.620226/' -gpu=1
python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3plus_freeze_decoder_aug1_960_640_2020-09-18 18-35-36.620226/' -gpu=1
echo python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3plus_unfreeze_decoder_aug1_960_640_2020-09-19 03-18-01.296558/' -gpu=1
python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/deeplabv3plus_unfreeze_decoder_aug1_960_640_2020-09-19 03-18-01.296558/' -gpu=1
echo python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/pan_freeze_decoder_aug1_960_640_2020-09-19 12-02-29.994840/' -gpu=1
python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/pan_freeze_decoder_aug1_960_640_2020-09-19 12-02-29.994840/' -gpu=1
echo python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/pan_unfreeze_decoder_aug1_960_640_2020-09-19 20-43-36.208259/' -gpu=1
python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir='logs/pan_unfreeze_decoder_aug1_960_640_2020-09-19 20-43-36.208259/' -gpu=1
