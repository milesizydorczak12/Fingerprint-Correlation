import shap

import numpy as np
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

data_dir= '/data/therealgabeguo/fingerprint_data/sd302_oldFingerprintExperiments'
batch_size=1
device = 'cuda:0'
model_wts_path = 'resnet_fingerprint01'

data_transforms = {
    'train': transforms.Compose([
	#transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
	#transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.2,1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	#transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=16)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

background = next(iter(dataloaders['train']))[0]
print(background.size())
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load(model_wts_path))
model.eval()

test_images = next(iter(dataloaders['val']))[0]
print(test_images.size())

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

print(shap_values.shape)
