# #Use this config with singlegpu_eval to select anchor sentences
# cfg = dict(
#     model='CVLP_r50',
#     desc_path='data/imagenet',
#     pretrained_clip='checkpoints/ctrain_clip_imagenet_80prompts/checkpoint_50.pth',#'pretrained/RN50.pt',
#     context_length=75,
#     pretrain_cvlp=False,
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

#     eval_pretrain=True,
#     test=True,
#     select=True,
#     include_wiki= True,
#     prompts='all',
# )

# Next, Use this config with multigpu_dist_train to train the LGR and evaluate
# 8 GPU
cfg = dict(
    model='LGR_r50',
    desc_path='data/imagenet',
    pretrained_clip='pretrained/RN50.pt',
    context_length=75,
    pretrain_cvlp=False,
    pretrain_cvlp_path='checkpoints/ctrained_clip_wikiprompt_imagenet_eval_wikicoopprompts/',
    loss_type="CE",
    two_branch=True,

    data_set='IMNET_LT',
    drop_last=True,

    weight_sample=True,
    use_sqrt_freq=True,

    lr=1e-3,
    min_lr=0,
    warmup_epochs=0,
    text_lr=1e-6,

    epochs=50,
    batch_size=128,

    repeated_aug=False,
    clip_ms=True,
    test=True,

    include_wiki= True,
    prompts='coop',
    run='LGR_train_wikipromptcvlpclip_imagenet_wikicoopprompt_dualbranch'

)