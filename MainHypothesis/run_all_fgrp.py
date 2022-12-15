import os, sys

#sys.path.append('/home/gabeguo/Biometric_Research/Fingerprint/REU-Biometrics-1F/')
sys.path.append('../')

results_dir = '/data/therealgabeguo/old_fingerprint_correlation_results'
os.makedirs(results_dir, exist_ok=True)

vanilla_img_dir = '/data/therealgabeguo/fingerprint_data/sd302_oldFingerprintExperiments'
enhanced_img_dir = '/data/therealgabeguo/fingerprint_data/img_l2_feature_extractions/enhance'
minutiae_img_dir = '/data/therealgabeguo/fingerprint_data/img_l2_feature_extractions/minutiae'
ridgeFreq_img_dir = '/data/therealgabeguo/fingerprint_data/img_l2_feature_extractions/img_l1_feature_extractions/freq'
ridgeOrient_img_dir = '/data/therealgabeguo/fingerprint_data/img_l2_feature_extractions/img_l1_feature_extractions/orient'

root_dir = vanilla_img_dir

if __name__ == '__main__':
    import general_train_test_split
    import transfer_learning

    BATCH_SIZE=16
    LEARNING_RATE=0.001
    NUM_EPOCHS=25
    LR_DECAY_FACTOR=0.8
    LR_DECAY_STEP=2
    MOMENTUM=0.9
    WEIGHT_DECAY=5e-5
    IS_PRETRAINED=True

    assert root_dir[-1] != '/' # can't end path with a /, because feature name won't be properly read
    features_examined = root_dir.split('/')[-1]

    print('features examined:', features_examined)

    with open(os.path.join(results_dir, features_examined + '.txt'), 'w') as fout:
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
                IS_PRETRAINED=IS_PRETRAINED, \
                data_dir=root_dir, \
                results_dir=results_dir, \
                features_examined=features_examined)

            fout.write(str(results_dict))
            fout.write('\n\n\n********************')
