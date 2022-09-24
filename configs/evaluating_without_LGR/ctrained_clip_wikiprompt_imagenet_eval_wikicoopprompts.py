# # 8 GPU
# cfg = dict(
#     model='CVLP_r50',
#     desc_path='data/imagenet',
#     pretrained_clip='checkpoints/pretrain_r50_original/checkpoint_50.pth',
#     context_length=75,
#     pretrain_cvlp=False,
#     # pretrain_cvlp_path='checkpoints/pretrain_r50_original',
#     loss_type="smoothCE",

#     data_set='IMNET_LT',
#     drop_last=True,

#     weight_sample=True,
#     use_sqrt_freq=True,
#     train_mode=False,

#     lr=5e-5,
#     min_lr=0.,

#     epochs=50,
#     batch_size=256,

#     repeated_aug=False,
#     mixup=0.,
#     cutmix=0.,
#     clip_ms=True,
#     distillation_beta=0.5,
#     distillation_type='logits',

#     eval= True,
#     eval_pretrain=True,
#     test=True,
#     select=False,
#     include_wiki= True,
#     prompts='coop',
#     run=' '
# )

# 8 GPU
cfg = dict(
    model='CVLP_r50',
    desc_path='data/imagenet',
    pretrained_clip='pretrained/RN50.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    data_set='IMNET_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,
    train_mode=False,

    lr=5e-5,
    min_lr=0.,

    epochs=50,
    batch_size=256,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',

    eval_pretrain=True,
    test=True,
    include_wiki= True,
    prompts='coop',
    select= True,
    run=' '
    # run='cvlp_coopprompts_completeimagenetLT_ep10'
)