test_prefix = '/content/drive/Shareddrives/Google Drive/AIClub/HCM_AI_CHALLENGE/Data/Vin_text/Reg/Crop/'

test_ann_file ='/content/drive/Shareddrives/Google Drive/AIClub/HCM_AI_CHALLENGE/Data/Vin_text/Reg/test.txt'

test = dict(
    type='OCRDataset',
    img_prefix=test_prefix,
    ann_file=test_ann_file,
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

test_list = [test]
