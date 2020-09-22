#!/bin/bash
source /home/leus/.pyenv/versions/pytorch/bin/activate
echo python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/unet_freeze_decoder_aug1_480_320_2020-09-17 15-03-34.890921/' -gpu=0
python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/unet_freeze_decoder_aug1_480_320_2020-09-17 15-03-34.890921/' -gpu=0
echo python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/unet_unfreeze_decoder_aug1_480_320_2020-09-17 21-14-48.742430/' -gpu=0
python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/unet_unfreeze_decoder_aug1_480_320_2020-09-17 21-14-48.742430/' -gpu=0
echo python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/fpn_freeze_decoder_aug1_480_320_2020-09-18 03-22-41.124707/' -gpu=0
python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/fpn_freeze_decoder_aug1_480_320_2020-09-18 03-22-41.124707/' -gpu=0
echo python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/fpn_unfreeze_decoder_aug1_480_320_2020-09-18 09-38-52.856183/' -gpu=0
python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/fpn_unfreeze_decoder_aug1_480_320_2020-09-18 09-38-52.856183/' -gpu=0
echo python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3_freeze_decoder_aug1_480_320_2020-09-18 15-54-46.205066/' -gpu=0
python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3_freeze_decoder_aug1_480_320_2020-09-18 15-54-46.205066/' -gpu=0
echo python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3_unfreeze_decoder_aug1_480_320_2020-09-19 05-15-37.396124/' -gpu=0
python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3_unfreeze_decoder_aug1_480_320_2020-09-19 05-15-37.396124/' -gpu=0
echo python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3plus_freeze_decoder_aug1_480_320_2020-09-19 18-36-23.957823/' -gpu=0
python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3plus_freeze_decoder_aug1_480_320_2020-09-19 18-36-23.957823/' -gpu=0
echo python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3plus_unfreeze_decoder_aug1_480_320_2020-09-20 01-07-17.577715/' -gpu=0
python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/deeplabv3plus_unfreeze_decoder_aug1_480_320_2020-09-20 01-07-17.577715/' -gpu=0
echo python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/pan_freeze_decoder_aug1_480_320_2020-09-20 07-26-47.115591/' -gpu=0
python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/pan_freeze_decoder_aug1_480_320_2020-09-20 07-26-47.115591/' -gpu=0
echo python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/pan_unfreeze_decoder_aug1_480_320_2020-09-20 13-22-00.710588/' -gpu=0
python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir='logs/pan_unfreeze_decoder_aug1_480_320_2020-09-20 13-22-00.710588/' -gpu=0


# leus@hpcs:~/3rdparty/github.com/ompugao/dm6190/1_segmentationreview$ bash checkmodels2.sh
# python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/unet_freeze_decoder_aug1_480_320_2020-09-17 15-03-34.890921/ -gpu=0
# Mean IoU: 0.322500, Pixel Accuracy Sum: 64.557721, mean: 0.806972
# python test.py -model=unet -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/unet_unfreeze_decoder_aug1_480_320_2020-09-17 21-14-48.742430/ -gpu=0
# Mean IoU: 0.350000, Pixel Accuracy Sum: 65.690228, mean: 0.821128
# python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/fpn_freeze_decoder_aug1_480_320_2020-09-18 03-22-41.124707/ -gpu=0
# Mean IoU: 0.492500, Pixel Accuracy Sum: 69.742057, mean: 0.871776
# python test.py -model=fpn -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/fpn_unfreeze_decoder_aug1_480_320_2020-09-18 09-38-52.856183/ -gpu=0
# Mean IoU: 0.437500, Pixel Accuracy Sum: 68.175736, mean: 0.852197
# python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/deeplabv3_freeze_decoder_aug1_480_320_2020-09-18 15-54-46.205066/ -gpu=0
# Mean IoU: 0.472500, Pixel Accuracy Sum: 67.705671, mean: 0.846321
# python test.py -model=deeplabv3 -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/deeplabv3_unfreeze_decoder_aug1_480_320_2020-09-19 05-15-37.396124/ -gpu=0
# Mean IoU: 0.525000, Pixel Accuracy Sum: 70.484258, mean: 0.881053
# python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/deeplabv3plus_freeze_decoder_aug1_480_320_2020-09-19 18-36-23.957823/ -gpu=0
# Mean IoU: 0.461250, Pixel Accuracy Sum: 68.122207, mean: 0.851528
# python test.py -model=deeplabv3plus -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/deeplabv3plus_unfreeze_decoder_aug1_480_320_2020-09-20 01-07-17.577715/ -gpu=0
# Mean IoU: 0.491250, Pixel Accuracy Sum: 69.540378, mean: 0.869255
# python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/pan_freeze_decoder_aug1_480_320_2020-09-20 07-26-47.115591/ -gpu=0
# Mean IoU: 0.240000, Pixel Accuracy Sum: 58.732096, mean: 0.734151
# python test.py -model=pan -imgwidth=480 -imgheight=320 -augmentation_version=1 -logdir=logs/pan_unfreeze_decoder_aug1_480_320_2020-09-20 13-22-00.710588/ -gpu=0
# Mean IoU: 0.250000, Pixel Accuracy Sum: 56.751686, mean: 0.709396
