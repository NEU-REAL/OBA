import os
import random
import numpy as np
import torch
import torch_pruning as tp
import torchvision
import argparse
import yaml
import registry
from engine.utils.imagenet_utils import presets
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### training settings
world_size = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vit_b_16", choices=["resnet50", "vit_b_16", "vgg16_bn", "vgg19_bn"])
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--distributed", type=bool, default=False)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay ratio for optimizers")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float, help="learning rate decay ratio for optimizers")
parser.add_argument("--sl_weight_decay", default=1e-4, type=float, help="weight decay ratio for optimizers in sparsity learning")
parser.add_argument("--sl-num-epochs", type=int, default=90, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.001, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="30,60", type=str, help="milestones for sparsity learning")
parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
parser.add_argument('--use_wandb', default=True, type=bool, help='whether to record data on wandb')
parser.add_argument('--seed', default=1, type=int, help='global seed')
parser.add_argument("--master_port", default="12221")
parser.add_argument('--save_path_prefix', default='checkpoints', type=str, help='the prefix of save path')
parser.add_argument('--iters_per_step', default=350, type=int, help='the number of iterative steps for each pruning')
parser.add_argument('--iterative_steps', default=50, type=int, help='the number of iterative steps for pruning')
parser.add_argument('--importance_type', default='OBA', type=str, choices=['OBA', 'fastOBA', 'Weight', 'Taylor', 'C-OBS', 'Kron-OBS',
                                                                              'C-OBD', 'Kron-OBD', 'Eigen'],
                    help='the type of importance')
parser.add_argument('--max_pruning_ratio', default=0.95, type=float, help='the max channel sparsity')
parser.add_argument('--normalizer', default="max", type=str, help='the normalizer of importance scores of each layer')
parser.add_argument('--self_unit_weight', default=False, type=bool, help='whether to use self unit weight')
parser.add_argument('--other_unit_weight', default=False, type=bool, help='whether to use other unit weight')
parser.add_argument('--delta', default=0., type=float, help='the delta for OBA')
parser.add_argument('--upward_delta', default=1., type=float, help='the delta for upward direct connectivity importance')
parser.add_argument('--downward_delta', default=1., type=float, help='the delta for downward direct connectivity importance')
parser.add_argument('--parallel_delta', default=1., type=float, help='the delta for parallel connectivity importance')
parser.add_argument('--multistep', default=True, type=bool, help='multistep pruning')
parser.add_argument('--fastoba_delta', default=1., type=float, help='the delta for fast OBA importance')
parser.add_argument('--order', default=2, type=int, help='the order of fast OBA')
parser.add_argument('--ops_ratios', default=0.51, nargs='+', type=float, help='The target FLOPs ratios')
parser.add_argument('--multivariable', default=True, type=bool, help='whether to use multivariable when calculating the importance scores')
args = parser.parse_args()
args.log_name = "Structured{}".format(args.importance_type)

# Function to apply custom initialization

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

seed_everything(args.seed)
save_dir = os.path.join(args.save_path_prefix, args.dataset, args.model, args.log_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
args.num_classes = 1000


def get_dataset(args):
    # Data loading code
    with open('./dataset.yaml', 'r', encoding='utf-8') as f:
        dataset_dirs = yaml.load(f.read(), Loader=yaml.FullLoader)
    imagenet_dir = dataset_dirs["imagenet"]
    traindir = os.path.join(imagenet_dir, 'train')
    valdir = os.path.join(imagenet_dir, 'val')
    print("Loading data...")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    print("Loading training data...")

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                          random_erase_prob=random_erase_prob))


    print("Loading validation data...")
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))

    print("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



if __name__ == "__main__":
    prune_batch_size = args.batch_size // 8
    train, test, train_sampler, test_sampler = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(train, batch_size=prune_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

    ### model initialization
    model = registry.get_model(args.model, num_classes=args.num_classes, pretrained=True, target_dataset=args.dataset)
    model = model.to(device)

    cross_entropy = torch.nn.CrossEntropyLoss()
    example_inputs = torch.randn(prune_batch_size, 3, 224, 224).to(device)
    model.eval()
    ignored_layers = []
    # DO NOT prune the final classifier!
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.Conv2d) and "vit" in args.model:# for vit
            ignored_layers.append(m)


    ops_ratio_list = args.ops_ratios if isinstance(args.ops_ratios, list) else [args.ops_ratios]
    target_ops_ratio = ops_ratio_list[0]
    wrapped_pruner = tp.WrappedPruner(args, train_loader, test_loader, model, example_inputs, ignored_layers,
                               1 - target_ops_ratio * 0.9, device)
    ## sparse prune training
    def get_learning_rate(epoch, milestones, lr, lr_decay_gamma=args.lr_decay_gamma):
        for milestone in milestones:
            if epoch < milestone:
                return lr * (lr_decay_gamma ** milestones.index(milestone))
        return lr * (lr_decay_gamma ** len(milestones))

    prune_step = 0
    save_path = os.path.join(save_dir, "pruned_model.pth")
    print("save_path")
    print(save_path)
    while prune_step < len(ops_ratio_list):
        if args.multistep:
            ops_percent, params_percent = wrapped_pruner.iterative_prune_step(args.order)
            wrapped_pruner.pruner.remove_hooks()
            wrapped_pruner = tp.WrappedPruner(args, train_loader, test_loader, model, example_inputs, ignored_layers,
                                              1 - target_ops_ratio * 0.9, device, base_ops=wrapped_pruner.base_ops, base_params=wrapped_pruner.base_params)
            print("ops_percent", ops_percent)
        else:
            ops_percent, params_percent = wrapped_pruner.onepass_prune_step(ops_ratio_list[prune_step], args.order)
        if wrapped_pruner.pruner.current_step >= args.iterative_steps - 10:
            break
        prune_ratio = ops_percent
        if prune_ratio <= ops_ratio_list[prune_step]:
            prune_step += 1
    wrapped_pruner.remove_hooks()
    args.ops_percent = ops_percent
    args.params_percent = params_percent
    checkpoint = {
        "model": model,
        "args": args,
    }
    torch.save(checkpoint, save_path)