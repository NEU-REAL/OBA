import os
import random
import numpy as np
import torch
import torch_pruning as tp
import torchvision
from tqdm import tqdm
import thop
import wandb
import argparse
import registry
import torch.nn.init as init
from torchvision import transforms as T
from torch_pruning.pruner.kfac_utils.network_utils import get_network, get_bottleneck_builder
import yaml

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### training settings
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vgg19")
parser.add_argument("--dataset", type=str, default="cifar10", choices=['cifar10', 'cifar100'])
parser.add_argument('--num_epochs', default=200, type=int, help='number of total training epochs')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument("--lr-decay-milestones", default="60,100,150", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float, help="learning rate decay ratio for optimizers")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay ratio for optimizers")
parser.add_argument("--sl_weight_decay", default=1e-4, type=float, help="weight decay ratio for optimizers in sparsity learning")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--sl-num-epochs", type=int, default=150, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.001, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="80,120", type=str, help="milestones for sparsity learning")
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--use_wandb', default=True, type=bool, help='whether to record data on wandb')
parser.add_argument('--seed', default=0, type=int, help='global seed')
parser.add_argument('--save_path_prefix', default='checkpoints', type=str, help='the prefix of save path')
parser.add_argument('--iters_per_step', default=200, type=int, help='the number of iterative steps for each pruning')
parser.add_argument('--iterative_steps', default=50, type=int, help='the number of iterative steps for pruning')
parser.add_argument('--multistep', default=True, type=bool, help='whether to use multistep pruning')
parser.add_argument('--importance_type', default='OBA', type=str, choices=['OBA', 'fastOBA', 'Weight', 'Taylor', 'C-OBS', 'Kron-OBS',
                                                                              'C-OBD', 'Kron-OBD', 'Eigen'],
                    help='the type of importance')
parser.add_argument('--ops_ratios', default=0.14, nargs='+', type=float, help='The target FLOPs ratios')
parser.add_argument('--max_pruning_ratio', default=0.95, type=float, help='the max channel sparsity')
parser.add_argument('--normalizer', default="max", type=str, help='the normalizer of importance scores of each layer')
parser.add_argument('--self_unit_weight', default=False, type=bool, help='whether to use self unit weight')
parser.add_argument('--other_unit_weight', default=False, type=bool, help='whether to use other unit weight')
parser.add_argument('--delta', default=0., type=float, help='the delta for OBA')
parser.add_argument('--upward_delta', default=1., type=float, help='the delta for upward direct connectivity importance')
parser.add_argument('--downward_delta', default=1., type=float, help='the delta for downward direct connectivity importance')
parser.add_argument('--parallel_delta', default=1., type=float, help='the delta for parallel connectivity importance')
parser.add_argument('--fastoba_delta', default=1., type=float, help='the delta for fast OBA importance')
parser.add_argument('--order', default=2, type=int, help='the order of the fast OBA')
parser.add_argument('--multivariable', default=True, type=bool, help='whether to use multivariable when calculating the importance scores')
args = parser.parse_args()
args.log_name = "Structured{}".format(args.importance_type)
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

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            out = model(data)
            loss += torch.nn.functional.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
        correct += (pred == target).sum()
        total += len(target)
    return (correct / total).item(), (loss / total).item()

seed_everything(args.seed)
save_dir = os.path.join(args.save_path_prefix, args.dataset, args.model, args.log_name)
save_path = os.path.join(args.save_path_prefix, args.dataset, args.model, "best_pretrain.pth")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.dataset == "cifar10":
    args.num_classes = 10
elif args.dataset == "cifar100":
    args.num_classes = 100
if args.use_wandb:
    wandb.init(project="OBA", name='{}_{}_{}'.format(args.dataset, args.model, args.log_name), config=args)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize( **NORMALIZE_DICT[args.dataset] ),
])
val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize( **NORMALIZE_DICT[args.dataset] ),
])
def get_dataset(dataset):
    with open('./dataset.yaml', 'r', encoding='utf-8') as f:
        dataset_dirs = yaml.load(f.read(), Loader=yaml.FullLoader)
    if dataset == "cifar10":
        train = torchvision.datasets.CIFAR10(root=dataset_dirs['cifar10'], train=True, transform=train_transform,
                                             download=True)
        test = torchvision.datasets.CIFAR10(root=dataset_dirs['cifar10'], train=False, transform=val_transform,
                                            download=True)
        return train, test
    elif dataset == "cifar100":
        train = torchvision.datasets.CIFAR100(root=dataset_dirs['cifar100'], train=True, transform=train_transform,
                                              download=True)
        test = torchvision.datasets.CIFAR100(root=dataset_dirs['cifar100'], train=False, transform=val_transform,
                                             download=True)
        return train, test
train, test = get_dataset(args.dataset)
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

### model initialization
def get_model(args):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.model == "vgg19":
            model = get_network(network="vgg", depth=19, dataset=args.dataset)
        elif args.model == "resnet32":
            model = get_network(network="resnet", depth=32, dataset=args.dataset)
        else:
            raise NotImplementedError
    else:
        model = registry.get_model(args.model, num_classes=args.num_classes, pretrained=True, target_dataset=args.dataset)
    return model

model = get_model(args)
builder = get_bottleneck_builder(network=args.model)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=args.lr_decay_gamma
)
cross_entropy = torch.nn.CrossEntropyLoss()
example_inputs = torch.randn(args.batch_size, 3, 32, 32).to(device)
model.eval()
ignored_layers = []
# DO NOT prune the final classifier!
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
        ignored_layers.append(m)

if not os.path.exists(save_path):
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
            loss = cross_entropy(output, labels)
            # loss = torch.nn.functional.cross_entropy(output, labels)
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
                # loss = cross_entropy(output, labels)
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
model.load_state_dict(torch.load(save_path), strict=False)
ops_ratio_list = args.ops_ratios if isinstance(args.ops_ratios, list) else [args.ops_ratios]
ori_acc, ori_loss = eval(model, test_loader, device=device)
print("original accuracy: {}".format(ori_acc))
## sparse prune training
milestones = [int(ms) for ms in args.sl_lr_decay_milestones.split(",")]
def get_learning_rate(epoch, milestones, lr, lr_decay_gamma=args.lr_decay_gamma):
    for milestone in milestones:
        if epoch < milestone:
            return lr * (lr_decay_gamma ** milestones.index(milestone))
    return lr * (lr_decay_gamma ** len(milestones))

best_test_acc = 0
prune_step = 0

wrapped_pruner = tp.WrappedPruner(args, train_loader, test_loader, model, example_inputs, ignored_layers,
                                              1 - ops_ratio_list[prune_step] * 0.9, device, builder=builder)
while prune_step < len(ops_ratio_list):
    if args.multistep:
        ops_percent, params_percent = wrapped_pruner.iterative_prune_step(args.order)
    else:
        ops_percent, params_percent = wrapped_pruner.onepass_prune_step(ops_ratio_list[prune_step], args.order)
    ## test stage
    test_accuracy, test_loss = eval(model, test_loader, device=device)
    prune_ratio = ops_percent
    print(f"prune ratio:{prune_ratio}, test accuracy:{test_accuracy}")
    if prune_ratio <= ops_ratio_list[prune_step]:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.sl_lr, momentum=0.9, weight_decay=args.sl_weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_decay_gamma
        )
        for epoch in range(args.sl_num_epochs):
            print("Current Fine-tune Epoch {}".format(epoch))
            ## train stage
            right = 0
            total = 0
            total_loss = 0
            for img, labels in train_loader:
                model.train()
                img = img.to(device)
                labels = labels.to(device)
                output = model(img)
                loss = cross_entropy(output, labels)
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
            test_accuracy, test_loss = eval(model, test_loader, device=device)
            # if test_accuracy > best_test_acc:
            #     best_test_acc = test_accuracy
            #     model.zero_grad()
            #     torch.save(model, os.path.join(save_dir, "best_pruned.pth"))
            if args.use_wandb:
                wandb.log({"{}ratio/test_accuracy".format(prune_step): test_accuracy,
                           "{}ratio/test_loss".format(prune_step): test_loss,
                           "{}ratio/train_accuracy".format(prune_step): train_accuracy,
                           "{}ratio/train_loss".format(prune_step):train_loss,
                           "{}ratio/epoch".format(prune_step): epoch,
                           "{}ratio/params_percent".format(prune_step): params_percent,
                           "{}ratio/ops_percent".format(prune_step): ops_percent, })
        prune_step += 1

        model = get_model(args)
        model.load_state_dict(torch.load(save_path), strict=False)
        model = model.to(device)
        if prune_step < len(ops_ratio_list):
            wrapped_pruner = tp.WrappedPruner(args, train_loader, test_loader, model, example_inputs, ignored_layers,
                                              1 - ops_ratio_list[prune_step] * 0.9, device, builder=builder)