import os, sys

sys.path.append('/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/')
root_dir = '/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/MainHypothesis/TransferLearning'

if __name__ == '__main__':
    import general_train_test_split
    import transfer_learning

    TYPE_FILENAME = 'datatype.txt'
    HYPERPARAM_FILENAME = 'hyperparams.txt'

    BATCH_SIZE=16
    LEARNING_RATE=0.001
    NUM_EPOCHS=25
    LR_DECAY_FACTOR=0.8
    LR_DECAY_STEP=2
    MOMENTUM=0.9
    WEIGHT_DECAY=5e-5
    IS_PRETRAINED=True

    with open('fingerprint_data/' + TYPE_FILENAME, 'r') as fin:
        datatype = fin.readline().strip()
    with open('fingerprint_data/' + HYPERPARAM_FILENAME, 'r') as fin:
        for line in fin.readlines():
            tokens = [item.strip() for item in line.split('=')]
            print(tokens)
            hyperparam_name, value = tokens[0], tokens[1]
            if hyperparam_name == 'num_epochs':
                NUM_EPOCHS = int(value)
            elif hyperparam_name == 'batch_size':
                BATCH_SIZE = int(value)
            elif hyperparam_name == 'learning_rate':
                LEARNING_RATE = float(value)
            elif hyperparam_name == 'lr_factor':
                LR_DECAY_FACTOR = float(value)
            elif hyperparam_name == 'lr_step':
                LR_DECAY_STEP = int(value)
            elif hyperparam_name == 'weight_decay':
                WEIGHT_DECAY = float(value)
            elif hyperparam_name == 'pretrained':
                IS_PRETRAINED = bool(value)
            elif hyperparam_name == 'momentum':
                MOMENTUM = float(value)

    print('datatype:', datatype)

    with open('deep_learning_results/' + datatype.strip() + '.txt', 'w') as fout:
        for i in range(1, 10 + 1):
            print('running train test split to val', i)
            general_train_test_split.main(root_dir=root_dir, train_fgrps=[j for j in range(1, 10 + 1) if i != j], val_fgrps = [i])
            #os.system('python3 train_test_split.py ' + str(i))
            #os.system('python3 transfer_learning.py')
            print('\nfinished train test split, now running deep learning\n')

            fout.write('val fgrp ' + str(i) + '\n\n')

            fout.write('BATCH_SIZE={}\n'.format(BATCH_SIZE))
            fout.write('LEARNING_RATE={}\n'.format(LEARNING_RATE))
            fout.write('NUM_EPOCHS={}\n'.format(NUM_EPOCHS))
            fout.write('LR_DECAY_FACTOR={}\n'.format(LR_DECAY_FACTOR))
            fout.write('LR_DECAY_STEP={}\n'.format(LR_DECAY_STEP))
            fout.write('MOMENTUM={}\n'.format(MOMENTUM))
            fout.write('WEIGHT_DECAY={}\n'.format(WEIGHT_DECAY))
            fout.write('IS_PRETRAINED={}\n'.format(IS_PRETRAINED))

            results_dict = transfer_learning.run_deep_learning(BATCH_SIZE=BATCH_SIZE, \
                LEARNING_RATE=LEARNING_RATE, \
                NUM_EPOCHS=NUM_EPOCHS, \
                LR_DECAY_FACTOR=LR_DECAY_FACTOR, \
                LR_DECAY_STEP=LR_DECAY_STEP, \
                MOMENTUM=MOMENTUM, \
                WEIGHT_DECAY=WEIGHT_DECAY, \
                IS_PRETRAINED=IS_PRETRAINED)

            fout.write(str(results_dict))
            fout.write('\n\n\n********************')
