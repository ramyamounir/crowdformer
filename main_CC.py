import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torchvision import transforms
from pvt_v2 import pvt_v2_b5
from prep_CC.image import load_data
from prep_CC.dataset import listDataset
import numpy as np
import os, math, random
import utils

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# === DATA === #

dataset_name = 'NOV'
beta = 7
batch_size = 6


root = 'data/CraneCounting/November/npydata'
if dataset_name == 'NOV':
    train_file = f'{root}/november_train.npy'
    test_file = f'{root}/november_test.npy'

with open(train_file, 'rb') as outfile:
    train_list = np.load(outfile).tolist()

with open(test_file, 'rb') as outfile:
    val_list = np.load(outfile).tolist()

def pre_data(train_list, percentage=100):
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    # Calculate the number of items to sample
    sampled_keys = random.sample(list(data_keys.keys()), int(len(data_keys) * percentage / 100))
    sampled_dict = {key_id: data_keys[key] for key, key_id in zip(sampled_keys, range(len(sampled_keys)))}

    return sampled_dict



# === TRAINING LOOPS === #

def train(Pre_data, model, criterion, optimizer, epoch, scheduler, writer):

    train_loader = torch.utils.data.DataLoader(
        listDataset(Pre_data, root, shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=batch_size,
                            num_workers=0),
        batch_size=batch_size, drop_last=False)

    model.train()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        # to cuda device
        img = img.cuda()

        # forward pass
        pred_count = model.get_all_feats(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        loss = criterion(pred_count, gt_count)
        writer(loss)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()


def validate(Pre_data, model):

    test_loader = torch.utils.data.DataLoader(
        listDataset(Pre_data, root,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(test_loader):


        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred_count = model.get_all_feats(img)
            count = torch.sum(pred_count).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

    mae = mae * 1.0 / (len(test_loader) * 1)
    mse = math.sqrt(mse / (len(test_loader)) * 1)

    return mae, mse





def train_gpu(gpu, args):
    print(f'gpu {gpu} spawned')

    # Initialize distributed environment
    args["gpu"] = gpu
    args["rank"] += gpu
    args["device"] = torch.device("cuda:{}".format(args["rank"]))

    dist.init_process_group(backend='nccl', init_method=args["dist_url"], world_size=args["world_size"], rank=args["rank"])
    fix_random_seeds()
    torch.cuda.set_device(args["gpu"])
    cudnn.benchmark = True
    dist.barrier()

    args["main"] = (args["rank"] == 0)
    setup_for_distributed(args["main"])


    # === MODEL === #
    model = pvt_v2_b5().cuda()

    if 0 <= args["rank"] < 4:
        ckpt = torch.load('out/weights/pvt_v2_b5.pth')
    elif 4<= args["rank"] < 8:
        ckpt = torch.load('out/SHB/epoch_best_8.1.pth')

    model.load_state_dict(ckpt, strict=False)
    print('weights loaded')


    # loss function
    class Loss(nn.Module):
        def __init__(self, beta=1):
            super().__init__()
            self.beta = beta

        def forward(self, preds, labels):

            self.diff = torch.abs(preds-labels).mean()
            if self.diff <= self.beta:
                return ((self.diff**2) * 0.5) / (self.beta)
            else:
                return self.diff- (0.5*self.beta)

    criterion = Loss(beta=beta)

    # optimizer
    optimizer = torch.optim.Adam( [{'params': model.parameters(), 'lr': 1e-5}, ], lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1, last_epoch=-1)

    # data
    if args["rank"] == 0 or args["rank"] == 4:
        train_data = pre_data(train_list, percentage=25)
    elif args["rank"] == 1 or args["rank"] == 5:
        train_data = pre_data(train_list, percentage=50)
    elif args["rank"] == 2 or args["rank"] == 6:
        train_data = pre_data(train_list, percentage=75)
    elif args["rank"] == 3 or args["rank"] == 7:
        train_data = pre_data(train_list, percentage=100)

    val_data = pre_data(val_list, percentage=100)

    # logging
    utils.checkdir(f'out/checkpoints/{dataset_name}/{args["rank"]}')
    writer = utils.get_writer(f'out/logs/{dataset_name}/{args["rank"]}')
    training_loss_writer = utils.TBWriter(writer, 'scalar', 'loss/training')
    test_mae_writer = utils.TBWriter(writer, 'scalar', 'Accuracy/MAE')
    test_mse_writer = utils.TBWriter(writer, 'scalar', 'Accuracy/MSE')



    # === TRAINING === #
    best_mae = math.inf
    for epoch in range(500):
        train(train_data, model, criterion, optimizer, epoch, scheduler, training_loss_writer)
        mae, mse = validate(val_data, model)
        test_mae_writer(mae)
        test_mse_writer(mse)

        if mae < best_mae:
            torch.save(model.state_dict(), f'out/checkpoints/{dataset_name}/{args["rank"]}/epoch_best_{mae}.pth')
            best_mae = mae

        scheduler.step()


if __name__ == "__main__":

    args = {}
    args["port"] = random.randint(49152,65535)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args["rank"] = 0
    args["dist_url"] = f'tcp://localhost:{args["port"]}'
    args["world_size"] = torch.cuda.device_count()

    mp.spawn(train_gpu, args = (args,), nprocs = args["world_size"])




