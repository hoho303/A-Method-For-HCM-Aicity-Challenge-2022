dataset_type = 'IcdarDataset'
data_root = '/content/drive/MyDrive/HCM_AI_CHALLENGE/Data/Vin_text/Icdar2015_FullTest/'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]

test_list = [test]
