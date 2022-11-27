'''

correlationDriver.py
Given a pre-trained deep learning model on 224p x 224p images and a feature
extraction, uses the SHAP library to extract important pixels from the deep
learned images and determines the correlation between the deep learned features
and the features extracted with the given extraction


'''

# TODO: WRITE UNIT TESTS

import sys, getopt, os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from FingerprintImageEnhancer import FingerprintImageEnhancer
from FingerprintFeatureExtractor import FingerprintFeatureExtractor
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
import shap
import warnings
import contextlib
from matplotlib.colors import LinearSegmentedColormap
#import shap.plots.colors._colors as colors
from shap.plots import colors
import time

#red_blue_transparent = LinearSegmentedColormap('red_blue_no_bounds', {"red": reds, "green": greens, "blue": blues, "alpha": [(a[0], 0.5, 0.5) for a in alphas]})

# Sources:
#   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#   https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Preconditions:
#   images have already been cropped
#   dataset folder structure produced via gather_nist_data.py and train_test_split.py
# Returns:
#   ResNet18 model pretrained on fingerprint dataset; GPU to use
def load_model(model_wts_pth, dataset_pth, class_names):
    # setup GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # create resnet model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Alternatively, size of output sample can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # give it the weights & set to evaluation mode
    model.load_state_dict(torch.load(model_wts_pth))
    model.eval()

    model = model.to(device)

    return model, device

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

# Returns the path storing the conventional feature map
# for the image referenced in this shap filepath
# See docs for get_feature_path, get_shap_path
def get_corresponding_featureMap_filepath(shap_filepath, level, output_dir):
    subdirectories = shap_filepath.split('/')
    img_id, pid = subdirectories[-2], subdirectories[-3]

    #print(subdirectories)

    corresponding_featureMap_filepath = get_feature_path_helper(output_dir=output_dir, img_id=img_id,
     level=level, class_name=pid)
    return corresponding_featureMap_filepath

# Helper method for overloaded get_feature_path; see other get_feature_path
# [img_id] is the original image's filename, without the '.png'
def get_feature_path_helper(output_dir, img_id, level, class_name):
    feature_name = level_to_featureName(level)
    new_filename = img_id + '_' + feature_name + '.png'
    return os.path.join(output_dir, 'level_' + level + '_features', class_name, new_filename)

'''
# Returns the path where the data for the current feature map will be written
# in format [output_dir]/[feature_name]/[class_name]/[feature map filename]
# Ex: output/level_2_features/00002223/00002223_01_dryrun-B_500_roll_05_minutiae.png
def get_feature_path(subject, data_dir, output_dir, paths, class_name, feature_name):
    curr_filepath = paths[subject]
    old_filename = curr_filepath.split('/')[-1]
    output_path = get_feature_path_helper(output_dir=output_dir, img_id=old_filename[:-4],
     feature_name=feature_name, class_name=class_name)

    assert old_filename.split('_')[0] == class_name

    return output_path
'''

# Returns the path where the data for the current SHAP heatmap will be written
# in format [output_dir]/shap/[actual_class]/[img_id]/[new_filename]
# SHAP heatmap filename will be same as [img_id] + '_SHAP_' + [curr_class_name] + '.png'
# Ex: output/shap/00002223/00002223_01_dryrun-B_500_roll_05/00002223_01_dryrun-B_500_roll_05_SHAP_00002237.png
def get_shap_path(subject, data_dir, output_dir, paths, actual_class, curr_class_name):
    curr_filepath = paths[subject]
    img_id = curr_filepath.split('/')[-1][:-4] # last token, get rid of png extension
    new_filename = img_id + '_' + 'SHAP' + '_' + curr_class_name + '.png'
    output_path = os.path.join(output_dir, 'shap', actual_class, img_id, new_filename)

    assert img_id.split('_')[0] == actual_class # make sure we have correct class names

    return output_path

# Returns [output_dir]/correlations/[level]/[pid]/[img_id][shap_class].txt
# See calcCorrelationScores()
def get_correlation_filepath(output_dir, level, shap_filepath):
    subdirs = shap_filepath.split('/')
    img_id = subdirs[-2]
    shap_filename = subdirs[-1]
    pid = shap_filename.split('_')[0]
    shap_class = shap_filename.split('_')[-1][:-4]

    return os.path.join(output_dir, 'correlations', level, pid, img_id, shap_class + '.txt')

# Preconditions:
#   images have already been cropped
#   data_dir folder structure produced via gather_nist_data.py and train_test_split.py
# Outcome:
#   Calculates the SHAP red-blue heatmaps for the preprocessed images in the data_dir
#   Saves these heatmaps to [output_dir]/shap, in a directory structure mirroring data_dir (e.g. folders for each PID)
#   Returns SHAP heatmaps as dictionary: filepath -> SHAP numpy array
# Credit:
#   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#   https://github.com/slundberg/shap/blob/master/notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.ipynb
def shap_by_batch(model_weights, data_dir, output_dir, batch_sizes={'val': 2, 'train': 16}):
    path2shap = dict()

    # TODO: play with hyperparmeters (train size)

    # set up DL
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), transforms.ToTensor())
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                 shuffle=True, num_workers=batch_sizes[x])
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['val'].classes

    model, device = load_model(model_weights, data_dir, class_names)

    print('device:', device)

    # TODO: reconsider background choice?
    # I made it a random sample of train data
    background = next(iter(dataloaders['train']))[0]
    background = background.to(device)

    # Go through batches
    for batch_num, (test_images, labels, paths) in enumerate(dataloaders['val']):
        print('\nbatch {} of {}'.format(batch_num, len(dataloaders['val'])))

        since = time.time()

        test_images = test_images.to(device)
        labels = labels.to(device)

        #print('images type:', test_images.get_device())
        #print('labels type:', labels.get_device())

        assert len({len(labels), len(test_images), len(paths), batch_sizes['val']}) == 1    # all the same size

        # Release the SHAP!
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(test_images)

        # shap_values.shape = (# classes, test batch size, img height, img width, # rgb channels)
        shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        # test_numpy.shape = (test batch size, img height, img width, # rgb channels)
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        # plot the feature attributions
        #shap.image_plot(shap_numpy, test_numpy)

        assert len(shap_values) == len(class_names)
        print('number of classes:', len(shap_values))
        print('class labels:', class_names)

        assert test_numpy.shape[0] == batch_sizes['val']

        for subject in range(test_numpy.shape[0]): # for each datapoint in the batch
            if len(shap_values[0][subject].shape) == 2: # grayscale
                abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
            else: # rgb
                abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
            max_val = np.nanpercentile(abs_vals, 99.9)
            for a_class in range(len(shap_values)): # for all the classes
                sv = shap_values[a_class][subject] if len(shap_values[a_class][subject].shape) == 2 else shap_values[a_class][subject].sum(-1)

                actual_class = class_names[labels[subject]]
                curr_class_name = class_names[a_class]  # TODO: need way to verify this is correct

                assert actual_class in paths[subject]   # the supposed class name matches the class shown in the filepath

                img_path = get_shap_path(subject=subject, data_dir=data_dir, output_dir=output_dir, paths=paths, actual_class=actual_class, curr_class_name=curr_class_name)
                os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)

                plt.imsave(img_path, sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
                img_as_np = plt.imread(img_path)

                assert actual_class in img_path
                assert curr_class_name in img_path
                assert (curr_class_name in paths[subject] and curr_class_name == actual_class) \
                 or (curr_class_name not in paths[subject] and curr_class_name != actual_class)

                path2shap[img_path] = img_as_np

        time_elapsed = time.time() - since
        print('Time elapsed: {}m, {}s'.format(int(time_elapsed // 60), int(time_elapsed % 60)))

    return path2shap

# See load_model, shap_by_batch
def analyzeDeepFeatures(model_weights, data_dir, output_dir):
    return shap_by_batch(model_weights, data_dir, output_dir)

# Saves level 2 features as .png in feature_dir;
# See extractConventionalFeatures for more details
def extractLevel2Features(data_dir, feature_dir):
    print(data_dir)
    path2array = dict()
    image_enhancer = FingerprintImageEnhancer()
    feature_extractor = FingerprintFeatureExtractor()

    for pid in os.listdir(data_dir):
        results_person_dir = os.path.join(feature_dir, pid)
        os.makedirs(results_person_dir, exist_ok = True)
        pid_src_dir = os.path.join(data_dir, pid)
        assert os.path.exists(pid_src_dir)
        for file in os.listdir(pid_src_dir):
            img = cv2.imread(os.path.join(pid_src_dir, file))
            # Convert image to grayscale
            if(len(img.shape) > 2):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance image and extract minutiae features
            out = image_enhancer.enhance(img, resize=False)
            FeaturesTerminations, FeaturesBifurcations = feature_extractor.extractMinutiaeFeatures((out * 255), 10)
            result_matrix = np.empty(feature_extractor._skel.shape)

            for feat in FeaturesTerminations:
                result_matrix[feat.locX][feat.locY] = 1.0
            for feat in FeaturesBifurcations:
                result_matrix[feat.locX][feat.locY] = 1.0

            # Apply gaussian filter to results
            result_matrix = gaussian_filter(result_matrix, sigma=5)

            # save output
            filename = file.split('.')[0] + '_minutiae' + '.png'
            filepath = os.path.join(results_person_dir, filename)

            plt.axes([0,0,1,1])
            plt.axis("off")
            plt.imsave(filepath, result_matrix, cmap='gray')

            '''
            result_matrix = (result_matrix * 255).astype(np.uint8)
            img = Image.fromarray(result_matrix).convert('RGB')
            img.save(filepath)
            '''

            path2array[filepath] = result_matrix

    return path2array


# Preconditions:
#   images have already been cropped
#   data_dir folder structure produced via gather_nist_data.py and train_test_split.py
# Outcome:
#   Extracts level features
#   Saves the feature map in [output_dir]/level_[level]_features
#   Returns these feature maps as dictionary: filepath -> numpy arrays
#   Note that it only does this for validation set!!! (Since no SHAP to compare to for train set)
def extractConventionalFeatures(level, data_dir, output_dir):

    # Make sure that we are only in the validation dataset!!! Doesn't make sense to compare train set
    if '/val' not in data_dir:
        data_dir = os.path.join(data_dir, 'val')
        assert os.path.exists(data_dir)

    feature_dir = os.path.join(output_dir, 'level_' + level + '_features')
    os.makedirs(feature_dir, exist_ok = True)

    if level is '1':
        return None
    elif level is '2':
        return extractLevel2Features(data_dir, feature_dir)
    elif level is '3':
        return None
    else:
        return None

# loads precalculated feature maps from file system
# as dict: path -> image as numpy array
def load_matrices(output_dir, map_name):
    path2file = dict()
    shap_dir = os.path.join(output_dir, map_name)
    for (root, dirs, files) in os.walk(shap_dir):
        for file in files:
            if '.png' in file:
                img_path = os.path.join(root, file)
                img_as_np = plt.imread(img_path)
                path2file[img_path] = img_as_np
    return path2file

# returns output of analyzeDeepFeatures by reading from file system (assumes already created)
def load_deep_matrices(output_dir):
    return load_matrices(output_dir, 'shap')

# returns output of extractConventionalFeatures by reading from file system (assumes already created)
def load_conventional_matrices(output_dir, level):
    return load_matrices(output_dir, 'level_' + str(level) + '_features')

# translates fingerprint feature level # to the conventional name (ex: minutiae, ridges)
def level_to_featureName(level):
    if int(level) == 2:
        return 'minutiae'
    return level

# Preconditions:
#   deep_matrices, conventional_matrices are outputs of analyzeDeepFeatures, extractConventionalFeatures
#   deep_matrices, conventional_matrices are dictionaries with filepaths as keys
# Outcome:
#   writes these correlation scores to [output_dir]/correlations/[level]/[pid]/[img_id]/[shap_class].txt
#   returns map: filenames -> correlation scores between deep_matrices & conventional_matrices
# How calculated:
#   (dot product of red shap values with minutiae) - (dot product of blue shap values with minutiae)
def calcCorrelationScores(deep_matrices, conventional_matrices, output_dir, level):
    if deep_matrices is None:
        deep_matrices = load_deep_matrices(output_dir)
    if conventional_matrices is None:
        conventional_matrices = load_conventional_matrices(output_dir, level)

    correlationScores = dict()
    for shap_filepath in deep_matrices:
        shap_matrix = deep_matrices[shap_filepath]
        #print('shape of shap matrix:', shap_matrix.shape)
        assert len(shap_matrix.shape) == 3

        corresponding_minutiae_filepath = get_corresponding_featureMap_filepath(shap_filepath=shap_filepath, level=level, output_dir=output_dir)
        minutiae_matrix = conventional_matrices[corresponding_minutiae_filepath]
        #print('shape of minutiae matrix:', minutiae_matrix.shape)
        assert len(minutiae_matrix.shape) == 3

        for r in range(minutiae_matrix.shape[0]):
            for c in range(minutiae_matrix.shape[1]):
                assert minutiae_matrix[r][c][0] == minutiae_matrix[r][c][1]
                assert minutiae_matrix[r][c][1] == minutiae_matrix[r][c][2]

        minutiae_matrix = minutiae_matrix[:,:,0]

        #print('shape of minutiae matrix:', minutiae_matrix.shape)
        assert len(minutiae_matrix.shape) == 2

        positive_shap_values, negative_shap_values = shap_matrix[:,:,0], shap_matrix[:,:,2] # red, blue

        correlationScore = np.dot(minutiae_matrix.flatten(), (positive_shap_values - negative_shap_values).flatten())
        correlation_filepath = get_correlation_filepath(output_dir, level, shap_filepath)
        correlationScores[correlation_filepath] = correlationScore

        #print(type(correlationScore))

        assert isinstance(correlationScore, np.float32)

        os.makedirs('/'.join(correlation_filepath.split('/')[:-1]), exist_ok=True)
        with open(correlation_filepath, 'w') as fout:
            fout.write(str(correlationScore))
    return correlationScores

def main(argv):
    USAGE_MSG = "Usage: correlationDriver.py \n\t [-h | -? | --help]" \
    + "\n\t --model_weights - weights of pretrained deep learning model" \
    + "\n\t --level - which fingerprint features (1, 2, 3)" \
    + "\n\t --data_dir - path of test data (produced via gather_nist_data.py, train_test_split.py)" \
    + "\n\t --output_dir - the root dir for the SHAP and feature extraction output" \
    + "\n\t --no_shap - don't run shap" \
    + "\n\t --no_ftr_extraction - don't run conventional feature extraction" \
    + "\n\t --no_correlation - don't calculate correlation scores"

    level = '2'
    model_weights = None
    DATA_DIR = ''    # assume that DATADIR was produced via gather_nist_data.py, train_test_split.py
    OUTPUT_DIR = ''


    run_shap = run_ftr_extraction = run_correlation = True

    try:
        opts, args = getopt.getopt(argv,'h?m:l:d:o:sfc',['help', 'model_weights=', 'level=', 'data_dir=', 'output_dir=',
         'no_shap', 'no_ftr_extraction', 'no_correlation'])
    except getopt.GetoptError:
        print(USAGE_MSG)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(USAGE_MSG)
            sys.exit(0)
        elif opt in ('--model_weights', '-m'):
            model_weights = arg
        elif opt in ('--level', '-l'):
            level = arg
        elif opt in ('--data_dir', '-d'):
            DATA_DIR = arg
        elif opt in ('--output_dir', '--output', '-o'):
            OUTPUT_DIR = arg
        elif opt in ('--no_shap', '-s'):
            run_shap = False
        elif opt in ('--no_ftr_extraction', '--no_feature_extraction', '-f'):
            run_ftr_extraction = False
        elif opt in ('--no_correlation', '-c'):
            run_correlation = False
        else:
            print('invalid option', opt, arg)
    shap_maps = feature_maps = correlations = None
    if run_shap:
        shap_maps = analyzeDeepFeatures(model_weights=model_weights, data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    if run_ftr_extraction:
        feature_maps = extractConventionalFeatures(level=level, data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    if run_correlation:
        correlations = calcCorrelationScores(deep_matrices=shap_maps, conventional_matrices=feature_maps, output_dir=OUTPUT_DIR, level=level)
    return

if __name__ == '__main__':
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main(sys.argv[1:])
