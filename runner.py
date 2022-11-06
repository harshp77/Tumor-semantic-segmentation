import os
from glob import glob
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import first
from utilities import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm

cases = sorted(glob(os.path.join('cases/data/', "*")))

image_data = []
seg_data = []
for path in cases:
  if len(glob(os.path.join(str(path)+'/', "*"))) < 2:
    continue
  image_data.append(path+'/imaging.nii.gz')
  seg_data.append(path+'/segmentation.nii.gz')

train_img = image_data[:150]
train_seg =seg_data[:150]

test_img = image_data[150:209]
test_seg = seg_data[150:209]

train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(train_img, train_seg)]
test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(test_img, test_seg)]

spatial_size=[128,128,64]
train_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=-336.1, a_max=412.1, b_min=0.0, b_max=1.0, clip=True), 
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
        ToTensord(keys=["vol", "seg"]),

    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=-336.1, a_max=412.1,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
        ToTensord(keys=["vol", "seg"]),
    ]
)

# train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
# train_loader = DataLoader(train_ds, batch_size=1)

# test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
# test_loader = DataLoader(test_ds, batch_size=1)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)


model_dir = 'result/' 

device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, train_loader, test_loader, loss_function, optimizer, 600, model_dir)