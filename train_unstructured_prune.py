import os
import random
import numpy as np
import torch
import torch_pruning as tp
import torchvision
import yaml
import wandb
import argparse
import registry
import torch.nn.init as init
from torchvision import transforms as T
from torch_pruning.pruner.kfac_utils.network_utils import get_network
from engine.utils.imagenet_utils import presets
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### training settings
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--dataset", type=str, default="imagenet", choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--num_epochs', default=200, type=int, help='number of total training epochs')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument("--lr-decay-milestones", default="60,100,150", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float, help="learning rate decay ratio for optimizers")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay ratio for optimizers")
parser.add_argument("--sl_weight_decay", default=1e-4, type=float, help="weight decay ratio for optimizers in sparsity learning")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--sl-num-epochs", type=int, default=150, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.001, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="80,120", type=str, help="milestones for sparsity learning")
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--test_batch_size', default=128, type=int, help='batch size')
parser.add_argument('--use_wandb', default=True, type=bool, help='whether to record data on wandb')
parser.add_argument('--seed', default=5, type=int, help='global seed')
parser.add_argument('--save_path_prefix', default='checkpoints', type=str, help='the prefix of save path')
parser.add_argument('--iters_per_step', default=200, type=int, help='the number of iterative steps for each pruning')
parser.add_argument('--iterative_steps', default=49, type=int, help='the number of iterative steps for pruning')
parser.add_argument('--importance_type', default='OBA', type=str, choices=['OBA'], help='the type of importance')
# Regarding unstructured pruning, we only implement OBA Pruner. If you want to implement different importance types, you can modify the unstructured_prune method of OBAPruner class.
parser.add_argument('--max_pruning_ratio', default=0.95, type=float, help='the max channel sparsity')
parser.add_argument('--normalizer', default='max', type=str, help='the normalizer of importance scores of each layer')
parser.add_argument('--self_unit_weight', default=False, type=bool, help='whether to use self unit weight')
parser.add_argument('--other_unit_weight', default=False, type=bool, help='whether to use other unit weight')
parser.add_argument('--delta', default=0., type=float, help='the delta for OBA')
parser.add_argument('--upward_delta', default=1., type=float, help='the delta for upward direct connectivity importance')
parser.add_argument('--downward_delta', default=1., type=float, help='the delta for downward direct connectivity importance')
parser.add_argument('--parallel_delta', default=1., type=float, help='the delta for parallel connectivity importance')
parser.add_argument('--multivariable', default=True, type=bool, help='whether to use multivariable when calculating the importance scores')
parser.add_argument('--pruning_ratio', default=0.1, type=float, help='the pruning ratio')
args = parser.parse_args()
args.log_name = "unstructuredOBA_{}_{}".format(args.dataset, args.model)
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}

# Function to apply custom initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        init.normal_(m.weight, std=0.01)
        init.constant_(m.bias, 0)

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
save_path = os.path.join(args.save_path_prefix, args.dataset, args.model, "best_pretrain.pth")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.dataset == "cifar10":
    example_inputs = torch.randn(args.batch_size, 3, 32, 32).to(device)
    args.num_classes = 10
elif args.dataset == "cifar100":
    example_inputs = torch.randn(args.batch_size, 3, 32, 32).to(device)
    args.num_classes = 100
elif args.dataset == "imagenet":
    example_inputs = torch.randn(args.batch_size, 3, 224, 224).to(device)
    args.num_classes = 1000


if args.use_wandb:
    wandb.init(project="OBA", name='{}_{}_{}'.format(args.dataset, args.model, args.log_name), config=args)


def get_dataset(dataset):
    with open('./dataset.yaml', 'r', encoding='utf-8') as f:
        dataset_dirs = yaml.load(f.read(), Loader=yaml.FullLoader)
    if dataset == "cifar10":
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[args.dataset]),
        ])
        train = torchvision.datasets.CIFAR10(root=dataset_dirs['cifar10'], train=True, transform=val_transform, download=True)
        test = torchvision.datasets.CIFAR10(root=dataset_dirs['cifar10'], train=False, transform=val_transform, download=True)
        return train, test
    elif dataset == "cifar100":
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[args.dataset]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[args.dataset]),
        ])
        train = torchvision.datasets.CIFAR100(root=dataset_dirs['cifar100'], train=True, transform=val_transform, download=True)
        test = torchvision.datasets.CIFAR100(root=dataset_dirs['cifar100'], train=False, transform=val_transform, download=True)
        return train, test
    elif dataset == "imagenet":
        # Data loading code
        data_dir = dataset_dirs['imagenet']
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        print("Loading data...")
        resize_size, crop_size = (256, 224)

        print("Loading training data...")

        auto_augment_policy = "ta_wide"
        random_erase_prob = getattr(args, "random_erase", 0.0)
        train = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))


        print("Loading validation data...")
        test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        return train, test
    else:
        raise NotImplementedError

train, test = get_dataset(args.dataset)
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size, num_workers=args.num_workers)

### model initialization
def get_model(args):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.model == "vgg19":
            model = get_network(network="vgg", depth=19, dataset=args.dataset)
        elif args.model == "resnet32":
            model = get_network(network="resnet", depth=32, dataset=args.dataset)
        elif args.model == "resnet20":
            model = get_network(network="resnet", depth=20, dataset=args.dataset)
        else:
            raise NotImplementedError
    else:
        model = registry.get_model(args.model, num_classes=args.num_classes, pretrained=True, target_dataset=args.dataset)
    return model



model = get_model(args)

model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=args.lr_decay_gamma
)
model.eval()
ignored_layers = []
# DO NOT prune the final classifier!
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
        ignored_layers.append(m)


if not os.path.exists(save_path) and args.dataset != "imagenet":
    ## dense training
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        print("Current Epoch {}".format(epoch))
        ## train stage
        right = 0
        total = 0
        total_loss = 0
        model.train()
        for img, labels in train_loader:
            img = img.to(device)
            labels = labels.to(device)
            output = model(img)
            loss = torch.nn.functional.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right += (output.argmax(1) == labels).float().sum().item()
            total += output.shape[0]
            total_loss += loss.item()
        train_accuracy = right / total
        train_loss = total_loss / len(train_loader)
        print("train_accuracy", train_accuracy)
        scheduler.step()
        ## test stage
        right = 0
        total = 0
        total_loss = 0
        model.eval()
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output = model(img)
                loss = torch.nn.functional.cross_entropy(output, labels, reduction="sum")
            right += (output.argmax(1) == labels).float().sum().item()
            total += output.shape[0]
            total_loss += loss.item()
        test_accuracy = right / total
        test_loss = total_loss / len(test_loader)
        print("test_accuracy", test_accuracy)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), save_path)
        if args.use_wandb:
            wandb.log({"dense/test_accuracy": test_accuracy,
                       "dense/test_loss": test_loss,
                       "dense/train_accuracy": train_accuracy,
                       "dense/train_loss":train_loss,
                       "dense/epoch": epoch})
if args.dataset != "imagenet":
    model.load_state_dict(torch.load(save_path), strict=False)

wrapped_pruner = tp.WrappedPruner(args, train_loader, test_loader, model, example_inputs, ignored_layers,
                           args.pruning_ratio, device)

# ori_acc, ori_val_loss = eval(model, test_loader, device=device)
# print("original accuracy: {}".format(ori_acc))
## sparse prune training
milestones = [int(ms) for ms in args.sl_lr_decay_milestones.split(",")]
def get_learning_rate(epoch, milestones, lr, lr_decay_gamma=args.lr_decay_gamma):
    for milestone in milestones:
        if epoch < milestone:
            return lr * (lr_decay_gamma ** milestones.index(milestone))
    return lr * (lr_decay_gamma ** len(milestones))

best_test_acc = 0
prune_step = 0

total_steps = args.iterative_steps
ratio = args.pruning_ratio ** (1 / total_steps)
pruning_steps = torch.tensor([ratio ** i for i in range(total_steps)])
record_steps = []
for i in range(9):
    prune_ratio = 0.1 * (i + 1)
    prune_idx = torch.argmin(torch.abs(pruning_steps - prune_ratio)).item()
    record_steps.append(prune_idx)
record_steps = torch.tensor(record_steps)
test_accuracy = 1
for step in range(total_steps):
    ## exponential
    pruning_ratio = 1 * ratio ** (step + 1)
    ## linear
    # pruning_ratio = 1 - (1 - args.pruning_ratio) * (step + 1) / total_steps
    mask_dict = wrapped_pruner.onepass_unstructured_prune_step(pruning_ratio)
    if step in record_steps:
        ## test once
        right = 0
        total = 0
        total_loss = 0
        model.eval()
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output = model(img)
                loss = torch.nn.functional.cross_entropy(output, labels, reduction="sum")
            right += (output.argmax(1) == labels).float().sum().item()
            total += output.shape[0]
            total_loss += loss.item()
        test_accuracy = right / total
        test_loss = total_loss / len(test_loader)
        save_path = os.path.join(save_dir, "pruned_{}.pth".format(pruning_ratio))
        torch.save(model.state_dict(), save_path)
    print("step {}, test_accuracy".format(step + 1), test_accuracy)
    logdict = {"nofinetune/test_accuracy": test_accuracy,
               "step": step + 1,
               "pruning_ratio": pruning_ratio}
    if args.use_wandb:
        wandb.log(logdict)


