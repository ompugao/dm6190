import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import pathlib
import pandas as pd
import numpy as np

from skimage.color import label2rgb

def imshow(imgs, masks):
    if type(imgs) is not list:
        imgs = [imgs]
    if type(masks) is not list:
        masks = [masks]

    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    totensor = transforms.ToTensor()
    #ptimg = make_grid([totensor(img) for img in imgs])
    ptimg = make_grid([totensor(label2rgb(np.array(mask), np.array(img), bg_label=0)) for img, mask in zip(imgs, masks)])
    plt.imshow(np.transpose(ptimg.numpy(), (1,2,0)), interpolation='nearest')
    plt.show()

class Dataset(VisionDataset):
    def __init__(self, root, train=None, small=False, transforms=None, transform=None, target_transform=None):
        super(Dataset, self).__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        if train:
            csvfile = 'train.csv'
        else:
            csvfile = 'test.csv'
        abscsvpath = pathlib.Path(root).resolve().parent / csvfile
        df = pd.read_csv(abscsvpath, header=None)
        self.filenames = ['%03d'%d for d in df[0].values]

        self.small = small

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        root = pathlib.Path(self.root).resolve()
        filename = self.filenames[index]
        imgdir = 'original_images'
        maskdir = 'label_images_semantic' 
        if self.small:
            imgdir = 'small_images'
            maskdir = 'small_label_images_semantic' 
        imgfile = root / imgdir  / (filename +'.jpg')
        maskfile  = root / maskdir / (filename +'.png')
        img = np.array(default_loader(imgfile))
        mask = np.array(default_loader(maskfile).convert('L')) #convert to grayscale again
        return self.transforms(image=img, mask=mask)

if __name__ == '__main__':
    dataset = Dataset(root='./semantic_drone_dataset', train=True, small=True)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

