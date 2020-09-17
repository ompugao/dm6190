import torch
import torchvision
import dataset as ds
import time
import copy
import tqdm
import numpy as np

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
from torch.functional import F
import logging
import matplotlib.pyplot as plt

import numpy as np
import torch


console = logging.StreamHandler()
#console_formatter = logging.Formatter('[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')
#console.setFormatter(console_formatter)
#console.setLevel(logging.INFO)

logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
#logging.getLogger(__name__).addHandler(console)

classes = "background, tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle".replace(' ', '').split(',')


# taken from https://www.pythonf.cn/read/100230 and modified a little
def iou_mean(pred, target, n_classes=1):
    # pred, target: torch.Tensor
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    #pred = torch.from_numpy(pred)
    pred = pred.view(-1) #flatten
    #target = np.array(target)
    #target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from [1,n_classes) -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + \
            target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            # If there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum/n_classes

# taken from https://github.com/pytorch/pytorch/issues/1249#issuecomment-339904369
def dice_loss(input, target):
    smooth = 1.
    loss = 0.
    for c in range(1, n_classes):
        iflat = input[:, c ].view(-1)
        tflat = target[:, c].view(-1)
        intersection = (iflat * tflat).sum()

        # Where class_weights is a list containing the weight for each class and input and target are shaped as (n_batches, n_classes, height, width). target is assumed to be one-hot encoded.
        w = class_weights[c]
        loss += w*(1 - ((2. * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth)))
    return loss

def test_model(model, logdir, dataloader, device, numclasses):
    model.to(device)
    model.load_state_dict(torch.load(logdir+'/model.pth'))
    model.eval()

    # ignore background(0)
    class_correct = list(0. for i in range(1, numclasses))
    class_total = list(0. for i in range(1, numclasses))
    sum_miou = 0.0
    sum_pixacc = 0.0
    with torch.no_grad():
        for data in dataloader:
            gimgs = data['image'].to(device, dtype=torch.float)
            gmasks = data['mask'].to(device, dtype=torch.long)
            outputs = model(gimgs)
            predicted = torch.argmax(outputs, dim=1)
            miou = iou_mean(predicted, gmasks, numclasses)
            sum_miou += miou
            c = (predicted == gmasks).squeeze()
            #for i in range(1, numclasses):
            #    label = gmasks[i]
            #    class_correct[label] += c[i].item()
            #    class_total[label] += 1
            pixacc = np.count_nonzero(np.array(c.cpu())) / torch.numel(c)
            sum_pixacc += pixacc


    #for i in range(1, numclasses):
    #    print('Accuracy of class %d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
    print("Mean IoU: %f"%sum_miou)
    print("Pixel Accuracy Sum: %f, mean: %f"%(sum_pixacc, sum_pixacc/len(dataloader)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="segmentation model", default="unet", type=str)
    parser.add_argument("-imgwidth", help="img width", default=480, type=int)
    parser.add_argument("-imgheight", help="img height", default=320, type=int)
    parser.add_argument("-augmentation_version", help="augmentation_version", default=0, type=int)
    parser.add_argument("-gpu", help="gpu device id", default=0, type=int)
    parser.add_argument("-logdir", help="logdir", type=str)
    args = parser.parse_args()

    import random
    random.seed(0) # fix seed for albumentations
    if args.augmentation_version == 0:
        data_transforms = albumentations.Compose([
            albumentations.Flip(),
            albumentations.RandomBrightness(0.2),
            albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
            albumentations.Normalize(),
            ToTensorV2()
            ])
        testdataset = ds.Dataset(root='./semantic_drone_dataset', train=False, imgsize=(args.imgwidth, args.imgheight), transforms=data_transforms)
    elif args.augmentation_version == 1:
        data_transforms = albumentations.Compose([
            albumentations.RandomSizedCrop([1000, 4000], args.imgheight, args.imgwidth),
            albumentations.OneOf([
                albumentations.RandomBrightness(0.1, p=1),
                albumentations.RandomContrast(0.1, p=1),
                albumentations.RandomGamma(p=0.5)
                ], p=0.3),
            albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
            albumentations.Cutout(p=0.5),
            ToTensorV2()
            ])
        testdataset = ds.Dataset(root='./semantic_drone_dataset', train=False, transforms=data_transforms)

    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    if args.model == "unet":
        model = smp.Unet('resnet101', classes=26, activation=None) #, activation='softmax2d') # in pytorch, nn.CrossEntropy computes softmax
    elif args.model == "fpn":
        model = smp.FPN('resnet101', classes=26, activation=None) #, activation='softmax2d') # in pytorch, nn.CrossEntropy computes softmax
    elif args.model == "deeplabv3":
        model = smp.DeepLabV3('resnet101', classes=26, activation=None)
    elif args.model == "deeplabv3plus":
        model = smp.DeepLabV3Plus('resnet101', classes=26, activation=None)
    elif args.model == "pan":
        model = smp.PAN("resnet101", classes=26, activation=None)


    numclasses = 23 # 22 + 1(background)
    #from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
    test_model(model, args.logdir, test_loader, device, numclasses)

    """
    img, mask = testdataset[0].values()
    with torch.no_grad():
        out = model(img.reshape(1, *img.shape).to(device, torch.float))
        loss = criterion(out, mask.reshape(1, *mask.shape).to(device, torch.long))
        #loss.detach().item()
        ds.imshow([img.permute(1,2,0), img.permute(1,2,0)], [torch.argmax(out.detach().cpu(),1).permute(1,2,0), mask.reshape(*mask.shape, 1)])
        #gridimg = torchvision.utils.make_grid([torch.argmax(out.detach().cpu(), 1), mask.reshape(1, *mask.shape)]) #img.permute(1, 2, 0), 
        plt.imshow(gridimg.permute(1,2,0))
        plt.show()

    model.load_state_dict(best_state)
    """
    # model.eval()
    # testdataset = ds.Dataset(root='./semantic_drone_dataset', train=False, imgsize=(480, 320), transforms=data_transforms)
    # img, mask = testdataset[0].values()
    # import matplotlib.pyplot as plt
    # with torch.no_grad():
    #     out = model(img.reshape(1, *img.shape).to(device, torch.float))
    #     loss = criterion(out, mask.reshape(1, *mask.shape).to(device, torch.long))
    #     #loss.detach().item()
    #     ds.imshow([img.permute(1,2,0), img.permute(1,2,0)], [torch.argmax(out.detach().cpu(),1).permute(1,2,0), mask.reshape(*mask.shape, 1)])
    #     #gridimg = torchvision.utils.make_grid([torch.argmax(out.detach().cpu(), 1), mask.reshape(1, *mask.shape)]) #img.permute(1, 2, 0), 
    #     plt.imshow(gridimg.permute(1,2,0))
    #     plt.show()
