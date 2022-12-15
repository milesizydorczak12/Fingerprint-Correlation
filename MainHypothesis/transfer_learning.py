
# Thank https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys

#sys.path.append('/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/')
sys.path.append('../')

from fileProcessingUtil import *

def get_pid_fgrp(filename):
    return (get_id(filename), get_fgrp(filename))

# Data augmentation and normalization for training
# Just normalization for validation
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

def get_finger_left_out(class_names, data_dir):
	an_img = os.listdir(os.path.join(data_dir, 'val', class_names[0]))[0]
	_, fgrp = get_pid_fgrp(an_img)
	return fgrp

plt.ioff()

def train_model(model, criterion, optimizer, scheduler, dataloaders, class_names, \
        data_dir, results_dir, \
        device, dataset_sizes, features_examined, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_top3_acc = 0.0
    best_top5_acc = 0.0
    epoch_of_best_acc = 0
    epoch_of_best_top3 = 0
    epoch_of_best_top5 = 0

    print('-' * 70)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_top3 = 0
            running_top5 = 0

            # Iterate over data.
            # each batch
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, top3preds = torch.topk(outputs, 3, dim=1)
                    _, top5preds = torch.topk(outputs, 5, dim=1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                labels_list = labels.data.tolist()
                top3preds_list = top3preds.tolist()
                top5preds_list = top5preds.tolist()

                for i in range(len(labels_list)):
                    if (labels_list[i] in set(top3preds_list[i])):
                        running_top3 += 1

                for i in range(len(labels_list)):
                    if (labels_list[i] in set(top5preds_list[i])):
                        running_top5 += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_top3 = float(running_top3) / dataset_sizes[phase]
            epoch_top5 = float(running_top5) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('{} Top 3: {:.4f} Top 5: {:.4f}'.format(
                phase, epoch_top3, epoch_top5))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                epoch_of_best_acc = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and epoch_top3 > best_top3_acc:
                best_top3_acc = epoch_top3
                epoch_of_best_top3 = epoch

            if phase == 'val' and epoch_top5 > best_top5_acc:
                best_top5_acc = epoch_top5
                epoch_of_best_top5 = epoch

        print()

    time_elapsed = time.time() - since

    results_dict = dict()
    results_dict['train_time'] = '{:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    results_dict['top1'] = (best_acc, epoch_of_best_acc)
    results_dict['top3'] = (best_top3_acc, epoch_of_best_top3)
    results_dict['top5'] = (best_top5_acc, epoch_of_best_top5)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Epoch of best val acc:', epoch_of_best_acc)
    print('Best val Acc: {:4f}'.format(best_acc))

    print('Epoch of best top3 acc:', epoch_of_best_top3)
    print('Best top3 val Acc: {:4f}'.format(best_top3_acc))

    print('Epoch of best top5 acc:', epoch_of_best_top5)
    print('Best top5 val Acc: {:4f}'.format(best_top5_acc))

    output_filename = '{}_exclude_{}.txt'.format(features_examined, get_finger_left_out(class_names, data_dir))
    output_path = os.path.join(results_dir, output_filename)

    with open(output_path, 'w') as fout:
        fout.write('Data source: {}'.format(data_dir))
        fout.write('\n\nIdentifying FGRP {}'.format(get_finger_left_out(class_names, data_dir)))
        fout.write('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        fout.write('\n\nEpoch of best val acc: {}'.format(epoch_of_best_acc))
        fout.write('\nBest val Acc: {:4f}'.format(best_acc))

        fout.write('\n\nEpoch of best top3 acc: {}'.format(epoch_of_best_top3))
        fout.write('\nBest top3 val Acc: {:4f}'.format(best_top3_acc))

        fout.write('\n\nEpoch of best top5 acc: {}'.format(epoch_of_best_top5))
        fout.write('\nBest top5 val Acc: {:4f}\n'.format(best_top5_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results_dict

def run_deep_learning(BATCH_SIZE=16, LEARNING_RATE=0.01, NUM_EPOCHS=25, LR_DECAY_FACTOR=0.8, \
        LR_DECAY_STEP=2, MOMENTUM=0.9, WEIGHT_DECAY=5e-5, IS_PRETRAINED=True, CUDA_DEVICE_IDS=[0], \
        data_dir='/data/therealgabeguo/fingerprint_data/sd302_oldFingerprintExperiments', \
        results_dir = '/data/therealgabeguo/old_fingerprint_correlation_results', \
        features_examined='vanilla'):
 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print('leave out:', get_finger_left_out(class_names, data_dir))

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    model_ft = models.resnet18(pretrained=IS_PRETRAINED)
    num_ftrs = model_ft.fc.in_features

    if len(CUDA_DEVICE_IDS) == 1:
        device = 'cuda:' + str(CUDA_DEVICE_IDS[0])
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    else: 
        ids = "cuda:" + ','.join([str(i) for i in CUDA_DEVICE_IDS])
        print(ids)
        device = torch.device(ids if torch.cuda.is_available() else "cpu")
        model_ft = nn.DataParallel(model_ft, device_ids=CUDA_DEVICE_IDS)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)

    model_ft, results_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, class_names,
                            data_dir, results_dir, device, dataset_sizes, features_examined, \
                            num_epochs=NUM_EPOCHS)

    finger_left_out = get_finger_left_out(class_names, data_dir)
    model_dir = os.path.join(results_dir, 'models_{}'.format(features_examined))
    os.makedirs(model_dir, exist_ok=True)
    model_name = 'resnet_fingerprint' + finger_left_out
    torch.save(model_ft.state_dict(), os.path.join(model_dir, model_name))

    results_dict['val_finger'] = finger_left_out

    return results_dict

if __name__ == '__main__':
    run_deep_learning()
