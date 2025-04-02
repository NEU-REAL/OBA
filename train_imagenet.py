import os
import random
import numpy as np
import torch
import torchvision
import warnings
from tqdm import tqdm
import wandb
import argparse
import yaml
from engine.utils.imagenet_utils import presets, utils
import torch.distributed as dist
import torch.multiprocessing as mp

### training settings
world_size = torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

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

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


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



def distributed_train(rank, save_dir, model_name):
    if os.path.exists(os.path.join(save_dir, model_name + "_newest.pth")):
        checkpoint = torch.load(os.path.join(save_dir, model_name + "_newest.pth"))
        start_epoch = checkpoint["epoch"]
        optim_state = checkpoint["optimizer"]
        lr_scheduler_state = checkpoint["lr_scheduler"]
    else:
        checkpoint = torch.load(os.path.join(save_dir, model_name + ".pth"))
        start_epoch = 0
        optim_state = None
        lr_scheduler_state = None

    args = checkpoint["args"]
    args.sl_num_epochs = 100
    args.batch_size = 64
    args.use_wandb = True
    args.lr_decay_milestones = "30, 60, 80"
    args.auto_augment = "ta_wide"
    seed_everything(args.seed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    device = torch.device(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=world_size, rank=rank)
    train, test, train_sampler, test_sampler = get_dataset(args)
    cross_entropy = torch.nn.CrossEntropyLoss()
    ### model initialization
    # model = registry.get_model(args.model, num_classes=args.num_classes, pretrained=True, target_dataset=args.dataset)
    model = checkpoint["model"]
    model = model.to(device)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)
    if args.use_wandb and rank == 0:
        wandb.init(project="OBA", name=args.log_name, config=args)
    parameters = utils.set_weight_decay(
        model,
        args.sl_weight_decay,
        norm_weight_decay=None,
        custom_keys_weight_decay=None,
    )
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module
    optimizer = torch.optim.SGD(parameters, lr=args.sl_lr, momentum=0.9, weight_decay=args.sl_weight_decay)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_decay_gamma
    )
    if lr_scheduler_state is not None:
        scheduler.load_state_dict(lr_scheduler_state)
    for epoch in range(start_epoch, args.sl_num_epochs):
        train_sampler.set_epoch(epoch)
        print("Current Fine-tune Epoch {}".format(epoch))
        ## train stage
        right = 0
        total = 0
        total_loss = 0
        for img, labels in tqdm(train_loader):
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
        test_accuracy, test_loss = evaluate(model, cross_entropy, test_loader, device=device)
        if rank == 0:
            checkpoint = {
                "model": model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(checkpoint, os.path.join(save_dir, model_name + "_newest.pth"))
            if args.use_wandb:
                wandb.log({"test_accuracy": test_accuracy,
                           "test_loss": test_loss,
                           "train_accuracy": train_accuracy,
                           "train_loss": train_loss,
                           "epoch": epoch,})

if __name__ == "__main__":
    mp.spawn(distributed_train, nprocs=world_size, join=True, args=(args.save_dir, args.model_name))