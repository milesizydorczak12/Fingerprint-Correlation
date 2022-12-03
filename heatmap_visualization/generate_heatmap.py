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
train_batch_size=32
test_batch_size=4
device = 'cuda:0'
model_wts_path = 'resnet_fingerprint01'

# Credit:
#   https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

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
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
batch_sizes = {'train':train_batch_size, 'val':test_batch_size}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=True, num_workers=16)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# get background

background, background_labels, background_paths = next(iter(dataloaders['train']))
background = background.to(device)

# load model

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load(model_wts_path))
model.eval()
model = model.to(device)

# get test images

test_images, test_labels, test_paths = next(iter(dataloaders['val']))
test_images = test_images.to(device)

# explain

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)
# shap_values has length 200, shap_values[i].shape = (1, 3, 224, 224)

print(len(shap_values))
print(shap_values[0].shape)

# show predictions for all test images

for i in range(test_batch_size):
    # heatmap for true label and random label
    for curr_person in test_labels:
        the_heatmap = np.reshape(shap_values[curr_person][i], (3, 224, 224))
        the_heatmap = np.transpose(the_heatmap, (1, 2, 0))
        the_heatmap = np.sum(the_heatmap, axis=2)

        max_val = max(abs(np.max(the_heatmap)), abs(np.min(the_heatmap)))

        the_image = test_images[i].permute((1, 2, 0)).cpu().numpy()[:,:,0]
        the_image = (the_image - np.min(the_image)) / np.ptp(the_image)

        #plt.imshow(the_image, cmap='gray', vmin=np.min(the_image), vmax=np.max(the_image))
        #plt.show()

        plt.imshow(the_heatmap, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        the_filename = test_paths[i].split('/')[-1]
        plt.imsave('shap_outputs/' + the_filename[:-4] + '_shap{}'.format(class_names[curr_person]) + '.png', the_heatmap, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        
        plt.imshow(the_image, cmap='gray', vmin=np.min(the_image), vmax=np.max(the_image))
        plt.imsave('shap_outputs/' + the_filename, the_image, cmap='gray', vmin=np.min(the_image), vmax=np.max(the_image))

print('red is more certain')
print('blue is less certain')
