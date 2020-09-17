import numpy as np
import torch
imgsize=(10,10)
nclasses = 25
batchsize = 1
output = np.random.rand(batchsize, nclasses+1, *imgsize)
mask = np.random.randint(0, nclasses+1, (batchsize, 1, *imgsize))
output = torch.from_numpy(output)
mask = torch.from_numpy(mask)
predicted = torch.argmax(output, 1)

# taken from https://www.pythonf.cn/read/100230
def iou_mean(pred, target, n_classes=1):
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
    for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
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

print(iou_mean(predicted, mask, nclasses))
from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

