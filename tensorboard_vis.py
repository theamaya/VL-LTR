import main
import utils
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
from timm.models import create_model
from datasets import build_dataset

print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda')
print(device)

def tensorboard_visualization(args):
    utils.init_distributed_mode(args)
    # args.test = False
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # cudnn.benchmark = True

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/tensorboard_vis_experiment_1')

    dataset_train, args.nb_classes = build_dataset(split="train", args=args)
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        dataset=dataset_train,
        args=args
    )

    model.to(device)

    writer.add_graph(model)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VL-LTR tensorboard visualization code', parents=[main.get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)

    tensorboard_visualization(args)