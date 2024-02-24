_base_ = [
    'mmdet::_base_/default_runtime.py', 'mixpl_coco_detection.py'
]

custom_imports = dict(
    imports=['projects.MixPL.mixpl'], allow_failed_imports=False)

detector = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='ckpt/resnet50-0676ba61.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

model = dict(
    type='MixPL',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        least_num=1,
        cache_size=8,
        mixup=True,
        mosaic=True,
        mosaic_shape=[(400, 400), (800, 800)],
        mosaic_weight=0.5,
        erase=True,
        erase_patches=(1, 20),
        erase_ratio=(0, 0.1),
        erase_thr=0.7,
        cls_pseudo_thr=0.4,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10-unlabeled.json'
labeled_dataset.data_prefix = dict(img='train2017/')
unlabeled_dataset.data_prefix = dict(img='train2017/')

train_dataloader = dict(
    batch_size=5,
    num_workers=5,
    sampler=dict(batch_size=5, source_ratio=[1, 4]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 90k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=90000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume=True
