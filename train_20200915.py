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
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.functional import F

console = logging.StreamHandler()
#console_formatter = logging.Formatter('[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')
#console.setFormatter(console_formatter)
#console.setLevel(logging.INFO)

logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
#logging.getLogger(__name__).addHandler(console)

def train(model, criterion, optimizer, dataloaders, device, writer=None, num_epochs=30, print_freq=1, past_epochs=0):
    starttime = time.time()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 1e13

    for epoch in range(num_epochs):
        iepoch = past_epochs + epoch + 1
        log.info('Epoch {}/{}'.format(iepoch, num_epochs))
        log.info('-' * 10)

        loss_history = {"train": [], "val": []}

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            for data in tqdm.tqdm(iter(dataloaders[phase])):
                gimgs = data['image'].to(device, dtype=torch.float)
                gmasks = data['mask'].to(device, dtype=torch.long)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(gimgs)
                    loss = criterion(outputs, gmasks)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            epoch_loss = np.float(loss.data)
            if writer is not None:
                writer.add_scalar("Loss/%s_epoch_loss"%phase, epoch_loss, iepoch)
            if (epoch + 1) % print_freq == 0:
                log.info("Epoch: [%d/%d], Loss: %.4f" %(iepoch, num_epochs, epoch_loss))
                loss_history[phase].append(epoch_loss)

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - starttime
    log.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    log.info("Best val Acc: {:4f}".format(best_loss))

    model.load_state_dict(best_state)

    return model, loss_history, past_epochs + num_epochs
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="segmentation model", default="unet", type=str)
    parser.add_argument("-freeze_decoder", help="segmentation model", default=0, type=int)
    parser.add_argument("-imgwidth", help="img width", default=480, type=int)
    parser.add_argument("-imgheight", help="img height", default=320, type=int)
    parser.add_argument("-logdirprefix", help="logdir prefix", default="", type=str)
    args = parser.parse_args()

    data_transforms = albumentations.Compose([
        albumentations.Flip(),
        albumentations.RandomBrightness(0.2),
        albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
        albumentations.Normalize(),
        ToTensorV2()
        ])

    dataset = ds.Dataset(root='./semantic_drone_dataset', train=True, imgsize=(args.imgwidth, args.imgheight), transforms=data_transforms)
    #dataset = ds.Dataset(root='./semantic_drone_dataset', train=True, imgsize=(480, 320), transforms=data_transforms)
    #dataset = ds.Dataset(root='./semantic_drone_dataset', train=True, imgsize=(960, 640), transforms=data_transforms)
    numtrain = int(len(dataset) * 0.7)
    numval = len(dataset) - numtrain
    trainset, valset = torch.utils.data.random_split(dataset, [numtrain, numval])
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True )
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True )

    dataloaders = {"train": train_loader, "val": val_loader}
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

    def _enablegrad(n, b):
        for param in n.parameters():
            param.require_grad = b
    def freeze(n):
        _enablegrad(n, True)
    def unfreeze(n):
        _enablegrad(n, False)

    freeze(model.encoder)
    if args.freeze_decoder > 0:
        freeze(model.decoder)

    criterion = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(params)

    import datetime
    #logdir="./logs/unet-resnet101-{}".format(str(datetime.datetime.now()))
    logdir="./logs/{}{}".format(args.logdirprefix, str(datetime.datetime.now()))
    logdir = logdir.replace(':', '-')
    writer = SummaryWriter(log_dir=logdir)

    model.to(device)
    total_epoch = 0
    # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
    model, loss_history, total_epoch = train(model, criterion, optimizer, dataloaders, device, writer, num_epochs=100, print_freq=1, past_epochs=total_epoch)
    torch.save(model.state_dict(), logdir+'/model.pth')
