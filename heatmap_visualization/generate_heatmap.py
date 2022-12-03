import shap
from shap.plots import colors

import numpy as np
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

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
background = background.to(device)
print(background.size())

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load(model_wts_path))
model.eval()
model = model.to(device)

test_images = next(iter(dataloaders['val']))[0]
test_images = test_images.to(device)
print(test_images.size())

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

print(len(shap_values), shap_values[0].shape)

# need to sum over
the_heatmap = np.reshape(shap_values[0], (3, 224, 224))
the_heatmap = np.transpose(the_heatmap, (1, 2, 0))
the_heatmap = np.sum(the_heatmap, axis=2)

positive_heatmap = np.abs(the_heatmap * (the_heatmap > 0))
negative_heatmap = np.abs(the_heatmap * (the_heatmap < 0))
max_val = max(np.max(positive_heatmap), np.max(negative_heatmap))
positive_heatmap /= max_val
negative_heatmap /= max_val

full_heatmap = np.stack([positive_heatmap, np.zeros(positive_heatmap.shape), negative_heatmap], axis=2)
#the_heatmap = (the_heatmap - np.min(the_heatmap)) / np.ptp(the_heatmap)
#the_heatmap = np.transpose(np.reshape(shap_values[0], (3, 224, 224)), (1, 2, 0))

the_image = test_images[0].permute((1, 2, 0)).cpu().numpy()

plt.imshow(the_image, cmap='gray')
plt.show()

plt.imshow(the_heatmap, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
plt.show()

plt.imshow(full_heatmap)
plt.show()

plt.imsave('the_heatmap.png', the_heatmap, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)

print('red is more certain')
print('blue is less certain')
