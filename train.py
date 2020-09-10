import torch
import torchvision
import dataset
import time
import copy
import tqdm

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import logging

console = logging.StreamHandler()
#console_formatter = logging.Formatter('[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')
#console.setFormatter(console_formatter)
#console.setLevel(logging.INFO)

logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
#logging.getLogger(__name__).addHandler(console)

def train(model, criterion, optimizer, dataloaders, device, writer=None, num_epochs=30, print_freq=1):
    starttime = time.time()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = 1e13

    for epoch in range(num_epochs):
        log.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        log.info('-' * 10)

        loss_history = {"train": [], "val": []}

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            for data in tqdm.tqdm(iter(dataloaders[phase])):
                imgs = data['image']
                masks = data['mask']
                imgs = imgs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)

                optimizer.zero_grad()

                #with torch.set_grad_enabled(phase == "train"):
                outputs = model(imgs)
                loss = criterion(outputs["out"], msks)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            epoch_loss = np.float(loss.data)
            if (epoch + 1) % print_freq == 0:
                log.info("Epoch: [%d/%d], Loss: %.4f" %(epoch+1, num_epochs, epoch_loss))
                loss_history[phase].append(epoch_loss)

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - starttime
    log.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    log.info("Best val Acc: {:4f}".format(best_loss))

    model.load_state_dict(best_state)

    return model, loss_history



if __name__ == '__main__':

    data_transforms = albumentations.Compose([
        albumentations.Flip(),
        albumentations.RandomBrightness(0.2),
        albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
        albumentations.Normalize(),
        ToTensorV2()
        ])

    dataset = dataset.Dataset(root='./semantic_drone_dataset', train=True, small=True, transforms=data_transforms)
    numtrain = int(len(dataset) * 0.7)
    numval = len(dataset) - numtrain
    trainset, valset = torch.utils.data.random_split(dataset, [numtrain, numval])
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True )
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True )

    dataloaders = {"train": train_loader, "val": val_loader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = smp.Unet('resnet101', classes=20, activation=None) #, activation='softmax2d') # in pytorch, nn.CrossEntropy computes softmax
    def _enablegrad(n, b):
        for param in n.parameters():
            param.require_grad = b
    def freeze(n):
        _enablegrad(n, True)
    def unfreeze(n):
        _enablegrad(n, False)

    freeze(model.encoder)
    freeze(model.decoder)

    criterion = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params)

    import datetime
    writer = SummaryWriter(log_dir="./logs/{}".format(str(datetime.datetime.now())))

    model.to(device)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
    #train(model, criterion, optimizer, dataloaders, device, writer, num_epochs=30, print_freq=1)
