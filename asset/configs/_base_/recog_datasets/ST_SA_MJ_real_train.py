# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, SynthAdd, Syn90k
# Real Dataset: IC11, IC13, IC15, COCO-Test, IIIT5k

train_prefix = 'data/mixture'

dataset_type = 'OCRDataset'

train_prefix1 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/content/vin_data/images/'

train_ann_file1 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/content/vin_data/train_label.txt'

train1 = dict(
    type=dataset_type,
    img_prefix=train_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_prefix2 = '/content/drive/Shareddrives/Google Drive/AIClub/HCM_AI_CHALLENGE/Data/TempoRun/Format/Crop/'
train_ann_file2 = '/content/drive/Shareddrives/Google Drive/AIClub/HCM_AI_CHALLENGE/Data/TempoRun/Format/tempo.txt'

train2 = dict(
    type=dataset_type,
    img_prefix=train_prefix2,
    ann_file=train_ann_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_prefix3 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100K_TestSetA/'
train_ann_file3 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100K_TestSetA/gt_testsetA.txt'

train3 = dict(
    type=dataset_type,
    img_prefix=train_prefix3,
    ann_file=train_ann_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_prefix4 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100k_V2/'
train_ann_file4 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100k_V2/test.txt'

train4 = dict(
    type=dataset_type,
    img_prefix=train_prefix4,
    ann_file=train_ann_file4,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_prefix5 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100k/'
train_ann_file5 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100k/gt.txt'

train5 = dict(
    type=dataset_type,
    img_prefix=train_prefix5,
    ann_file=train_ann_file5,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_prefix6 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100K_TestSetA/'
train_ann_file6 = '/content/drive/MyDrive/HCM_AI_CHALLENGE/SyntheticData/TextReg/Horizontal_100K_TestSetA/gt_wrong_2.txt'

train6 = dict(
    type=dataset_type,
    img_prefix=train_prefix6,
    ann_file=train_ann_file6,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_list = [train1, train2, train3, train4, train5, train6]
