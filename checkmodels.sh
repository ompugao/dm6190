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

# Mean IoU: 0.205153, Pixel Accuracy Sum: 67.187676, mean: 0.839846
# Mean IoU: 0.187193, Pixel Accuracy Sum: 66.241602, mean: 0.828020
# Mean IoU: 0.215810, Pixel Accuracy Sum: 67.717376, mean: 0.846467
# Mean IoU: 0.213285, Pixel Accuracy Sum: 67.560085, mean: 0.844501
# Mean IoU: 0.221303, Pixel Accuracy Sum: 68.337969, mean: 0.854225
# Mean IoU: 0.217960, Pixel Accuracy Sum: 68.064935, mean: 0.850812
# Mean IoU: 0.186710, Pixel Accuracy Sum: 65.891452, mean: 0.823643
# Mean IoU: 0.200851, Pixel Accuracy Sum: 66.833034, mean: 0.835413
# Mean IoU: 0.163800, Pixel Accuracy Sum: 63.735495, mean: 0.796694
# Mean IoU: 0.186507, Pixel Accuracy Sum: 64.505397, mean: 0.806317


python test.py -model=unet -logdir='logs/unet_freeze_decoder_2020-09-16 11-14-03.715160' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=unet -logdir='logs/unet_unfreeze_decoder_2020-09-16 12-27-43.885046' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=fpn -logdir='logs/fpn_freeze_decoder_2020-09-16 13-41-20.060483' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=fpn -logdir='logs/fpn_unfreeze_decoder_2020-09-16 14-43-17.875869' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=pan -logdir='logs/pan_freeze_decoder_2020-09-17 11-27-21.809318' -augmentation_version=0 -imgheight=640 -imgwidth=960
python test.py -model=pan -logdir='logs/pan_unfreeze_decoder_2020-09-17 11-28-05.393323' -augmentation_version=0 -imgheight=640 -imgwidth=960


# Mean IoU: 0.243431, Pixel Accuracy Sum: 69.757715, mean: 0.871971
# Mean IoU: 0.255911, Pixel Accuracy Sum: 70.520202, mean: 0.881503
# Mean IoU: 0.268006, Pixel Accuracy Sum: 71.170475, mean: 0.889631
# Mean IoU: 0.253208, Pixel Accuracy Sum: 70.479209, mean: 0.880990
# Mean IoU: 0.240053, Pixel Accuracy Sum: 69.698047, mean: 0.871226
# Mean IoU: 0.195554, Pixel Accuracy Sum: 60.221577, mean: 0.752770



leus@hpcs:~/3rdparty/github.com/ompugao/dm6190/1_segmentationreview$ bash checkmodels.sh
Mean IoU: 0.417500, Pixel Accuracy Sum: 67.187676, mean: 0.839846
Mean IoU: 0.366250, Pixel Accuracy Sum: 66.241602, mean: 0.828020
Mean IoU: 0.451250, Pixel Accuracy Sum: 67.717376, mean: 0.846467
Mean IoU: 0.433750, Pixel Accuracy Sum: 67.560085, mean: 0.844501
Mean IoU: 0.485000, Pixel Accuracy Sum: 68.337969, mean: 0.854225
Mean IoU: 0.460000, Pixel Accuracy Sum: 68.064935, mean: 0.850812
Mean IoU: 0.368750, Pixel Accuracy Sum: 65.891452, mean: 0.823643
Mean IoU: 0.406250, Pixel Accuracy Sum: 66.833034, mean: 0.835413
Mean IoU: 0.276250, Pixel Accuracy Sum: 63.735495, mean: 0.796694
Mean IoU: 0.343750, Pixel Accuracy Sum: 64.505397, mean: 0.806317
Mean IoU: 0.523750, Pixel Accuracy Sum: 69.757715, mean: 0.871971
Mean IoU: 0.561250, Pixel Accuracy Sum: 70.520202, mean: 0.881503
Mean IoU: 0.576250, Pixel Accuracy Sum: 71.170475, mean: 0.889631
Mean IoU: 0.550000, Pixel Accuracy Sum: 70.479209, mean: 0.880990
Mean IoU: 0.497500, Pixel Accuracy Sum: 69.698047, mean: 0.871226
Mean IoU: 0.387500, Pixel Accuracy Sum: 60.221577, mean: 0.752770

