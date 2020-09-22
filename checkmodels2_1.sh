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



# python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/unet_freeze_decoder_aug1_960_640_2020-09-17 15-03-43.588931/ -gpu=1
# Mean IoU: 0.466250
# Pixel Accuracy Sum: 69.109692, mean: 0.863871
# python test.py -model=unet -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/unet_unfreeze_decoder_aug1_960_640_2020-09-17 22-02-50.093149/ -gpu=1
# Mean IoU: 0.396250
# Pixel Accuracy Sum: 67.584704, mean: 0.844809
# python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/fpn_freeze_decoder_aug1_960_640_2020-09-18 04-56-35.049183/ -gpu=1
# Mean IoU: 0.518750
# Pixel Accuracy Sum: 70.589950, mean: 0.882374
# python test.py -model=fpn -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/fpn_unfreeze_decoder_aug1_960_640_2020-09-18 11-46-27.800628/ -gpu=1
# Mean IoU: 0.532500
# Pixel Accuracy Sum: 70.438617, mean: 0.880483
# python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/deeplabv3_freeze_decoder_aug1_960_640_2020-09-18 18-35-17.814975/ -gpu=1
# Traceback (most recent call last):
#   File "test.py", line 214, in <module>
#     test_model(model, args.logdir, test_loader, device, numclasses)
#   File "test.py", line 81, in test_model
#     model.load_state_dict(torch.load(logdir+'/model.pth', map_location="cuda:0"))
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 571, in load
#     with _open_file_like(f, 'rb') as opened_file:
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 229, in _open_file_like
#     return _open_file(name_or_buffer, mode)
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 210, in __init__
#     super(_open_file, self).__init__(open(name, mode))
# FileNotFoundError: [Errno 2] No such file or directory: 'logs/deeplabv3_freeze_decoder_aug1_960_640_2020-09-18 18-35-17.814975//model.pth'
# python test.py -model=deeplabv3 -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/deeplabv3_unfreeze_decoder_aug1_960_640_2020-09-18 18-35-28.045851/ -gpu=1
# Traceback (most recent call last):
#   File "test.py", line 214, in <module>
#     test_model(model, args.logdir, test_loader, device, numclasses)
#   File "test.py", line 81, in test_model
#     model.load_state_dict(torch.load(logdir+'/model.pth', map_location="cuda:0"))
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 571, in load
#     with _open_file_like(f, 'rb') as opened_file:
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 229, in _open_file_like
#     return _open_file(name_or_buffer, mode)
#   File "/home/leus/.pyenv/versions/3.7.9/envs/pytorch/lib/python3.7/site-packages/torch/serialization.py", line 210, in __init__
#     super(_open_file, self).__init__(open(name, mode))
# FileNotFoundError: [Errno 2] No such file or directory: 'logs/deeplabv3_unfreeze_decoder_aug1_960_640_2020-09-18 18-35-28.045851//model.pth'
# python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/deeplabv3plus_freeze_decoder_aug1_960_640_2020-09-18 18-35-36.620226/ -gpu=1
# Mean IoU: 0.540000
# Pixel Accuracy Sum: 70.879442, mean: 0.885993
# python test.py -model=deeplabv3plus -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/deeplabv3plus_unfreeze_decoder_aug1_960_640_2020-09-19 03-18-01.296558/ -gpu=1
# Mean IoU: 0.496250
# Pixel Accuracy Sum: 68.057280, mean: 0.850716
# python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/pan_freeze_decoder_aug1_960_640_2020-09-19 12-02-29.994840/ -gpu=1
# Mean IoU: 0.375000
# Pixel Accuracy Sum: 64.037301, mean: 0.800466
# python test.py -model=pan -imgwidth=960 -imgheight=640 -augmentation_version=1 -logdir=logs/pan_unfreeze_decoder_aug1_960_640_2020-09-19 20-43-36.208259/ -gpu=1
# Mean IoU: 0.226250
# Pixel Accuracy Sum: 58.636019, mean: 0.732950
