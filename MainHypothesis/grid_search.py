from transfer_learning import run_deep_learning
import itertools
import os

def generate_possible_output_filename(i, features):
    return 'grid_search_output/' + features + '/grid_search_output_attempt' + '{:02d}'.format(i) + '.txt'

def get_output_filename(features):
    i = 0
    while os.path.exists(generate_possible_output_filename(i, features)):
        i += 1
    return generate_possible_output_filename(i, features)

def grid_search(batch_sizes, lrs, lr_factors, lr_steps, weight_decays, num_epochs, features, is_pretraineds):
    with open(get_output_filename(features), 'w') as fout:
        for batch_size, lr, lr_factor, lr_step, weight_decay, is_pretrained in \
         itertools.product(batch_sizes, lrs, lr_factors, lr_steps, weight_decays, is_pretraineds):
            print('num_epochs = {}'.format(num_epochs))
            print('batch_size = {}'.format(batch_size))
            print('learning_rate = {}'.format(lr))
            print('lr_factor = {}'.format(lr_factor))
            print('lr_step = {}'.format(lr_step))
            print('weight_decay = {}'.format(weight_decay))
            print('is_pretrained = {}'.format(is_pretrained))

            results_dict = run_deep_learning(BATCH_SIZE=batch_size, LEARNING_RATE=lr, \
             LR_DECAY_FACTOR=lr_factor, \
             LR_DECAY_STEP=lr_step, WEIGHT_DECAY=weight_decay, NUM_EPOCHS=num_epochs, \
             IS_PRETRAINED=is_pretrained)

            fout.write('num_epochs = {}\n'.format(num_epochs))
            fout.write('batch_size = {}\n'.format(batch_size))
            fout.write('learning_rate = {}\n'.format(lr))
            fout.write('lr_factor = {}\n'.format(lr_factor))
            fout.write('lr_step = {}\n'.format(lr_step))
            fout.write('weight_decay = {}\n'.format(weight_decay))
            fout.write('is_pretrained = {}\n'.format(is_pretrained))

            fout.write(str(results_dict))
            fout.write('\n\n')

batch_sizes = [8, 16, 32, 64, 128, 256]
lrs = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
lr_factors = [0.10, 0.25, 0.50, 0.75, 0.90]
lr_steps = [2, 4, 7]
weight_decays = [1e-3, 1e-4, 1e-5, 1e-6]
num_epochs = 25
features = 'vanilla'
is_pretraineds=[True, False]

grid_search(batch_sizes, lrs, lr_factors, lr_steps, weight_decays, num_epochs, features, is_pretraineds)

print('finished grid search on ' + features)
