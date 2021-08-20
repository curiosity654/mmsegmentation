# dataset settings
dataset_type = 'ICIPRGBDDataset'
data_root = 'data/ICIP'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 808.17865614], std=[58.395, 57.12, 57.375, 943.49172682], to_rgb=True)
img_scale = (640, 480)
crop_size = (640, 480)
train_pipeline = [
    dict(type='LoadRGBDFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='RGBDNormalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadRGBDFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='RGBDNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/rgb',
            ann_dir='train/mask',
            depth_dir='train/depth', 
            depth_suffix='.png',
            # split='split/train.txt',
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/rgb',
        ann_dir='train/mask',
        depth_dir='train/depth', 
        depth_suffix='.png',
        split='train/split/val.txt',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='stage_two/rgb',
        depth_dir='stage_two/depth', 
        depth_suffix='.png',
        pipeline=test_pipeline),
        test_outpath='/root/code/mmsegmentation/work_dirs/pspnet_icip_all/output')