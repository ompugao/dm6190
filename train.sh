#!/bin/bash
set -eu
source /home/leus/.pyenv/versions/pytorch/bin/activate
echo python train.py -model=unet -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=unet_freeze_decoder_
python train.py -model=unet -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=unet_freeze_decoder_
echo python train.py -model=unet -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=unet_unfreeze_decoder_
python train.py -model=unet -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=unet_unfreeze_decoder_
echo python train.py -model=fpn -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=fpn_freeze_decoder_
python train.py -model=fpn -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=fpn_freeze_decoder_
echo python train.py -model=fpn -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=fpn_unfreeze_decoder_
python train.py -model=fpn -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=fpn_unfreeze_decoder_
echo python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3_freeze_decoder_
python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3_freeze_decoder_
echo python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3_unfreeze_decoder_
python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3_unfreeze_decoder_
echo python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3plus_freeze_decoder_
python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3plus_freeze_decoder_
echo python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3plus_unfreeze_decoder_
python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=deeplabv3plus_unfreeze_decoder_
echo python train.py -model=pan -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=pan_freeze_decoder_
python train.py -model=pan -freeze_decoder=1 -imgwidth=480 -imgheight=320 -logdirprefix=pan_freeze_decoder_
echo python train.py -model=pan -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=pan_unfreeze_decoder_
python train.py -model=pan -freeze_decoder=0 -imgwidth=480 -imgheight=320 -logdirprefix=pan_unfreeze_decoder_





echo python train.py -model=unet -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=unet_freeze_decoder_
python train.py -model=unet -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=unet_freeze_decoder_
echo python train.py -model=unet -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=unet_unfreeze_decoder_
python train.py -model=unet -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=unet_unfreeze_decoder_
echo python train.py -model=fpn -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=fpn_freeze_decoder_
python train.py -model=fpn -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=fpn_freeze_decoder_
echo python train.py -model=fpn -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=fpn_unfreeze_decoder_
python train.py -model=fpn -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=fpn_unfreeze_decoder_
echo python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3_freeze_decoder_
python train.py -model=deeplabv3 -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3_freeze_decoder_
echo python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3_unfreeze_decoder_
python train.py -model=deeplabv3 -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3_unfreeze_decoder_
echo python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3plus_freeze_decoder_
python train.py -model=deeplabv3plus -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3plus_freeze_decoder_
echo python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3plus_unfreeze_decoder_
python train.py -model=deeplabv3plus -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=deeplabv3plus_unfreeze_decoder_
echo python train.py -model=pan -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=pan_freeze_decoder_
python train.py -model=pan -freeze_decoder=1 -imgwidth=960 -imgheight=640 -logdirprefix=pan_freeze_decoder_
echo python train.py -model=pan -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=pan_unfreeze_decoder_
python train.py -model=pan -freeze_decoder=0 -imgwidth=960 -imgheight=640 -logdirprefix=pan_unfreeze_decoder_
