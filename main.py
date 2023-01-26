import torch
import torch.nn as nn
from torchvision import transforms
from pvt_v2 import pvt_v2_b5
from prep_data.image import load_data
from prep_data.dataset import listDataset
import numpy as np
import os, math
import utils

# === DATA === #
dataset_name = 'SHA'

root = 'data/ShanghaiTech/npydata/'
if dataset_name == 'SHA':
    train_file = f'{root}/ShanghaiA_train.npy'
    test_file = f'{root}/ShanghaiA_test.npy'
elif dataset_name == 'SHB':
    train_file = f'{root}/ShanghaiB_train.npy'
    test_file = f'{root}/ShanghaiB_test.npy'

with open(train_file, 'rb') as outfile:
    train_list = np.load(outfile).tolist()
with open(test_file, 'rb') as outfile:
    val_list = np.load(outfile).tolist()

def pre_data(train_list):
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

    return data_keys



# === TRAINING LOOPS === #

def train(Pre_data, model, criterion, optimizer, epoch):

    train_loader = torch.utils.data.DataLoader(
        listDataset(Pre_data, root, shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=1,
                            num_workers=0),
        batch_size=1, drop_last=False)

    model.train()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        # to cuda device
        img = img.cuda()

        # forward pass
        pred_count = model.get_all_feats(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        loss = criterion(pred_count, gt_count)
        training_loss_writer(loss)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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



# === MODEL === #

# model = pvt_v2_b5(drop_path_rate=0.3).cuda()
model = pvt_v2_b5().cuda()
ckpt = torch.load('out/weights/pvt_v2_b5.pth')
model.load_state_dict(ckpt, strict=False)
print('weights loaded')

# loss function
class Loss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, preds, labels):

        self.diff = torch.abs(preds-labels)
        if self.diff <= self.beta:
            return ((self.diff**2) * 0.5) / (self.beta)
        else:
            return self.diff- (0.5*self.beta)

criterion = Loss(beta=1)

# optimizer
optimizer = torch.optim.Adam( [{'params': model.parameters(), 'lr': 1e-5}, ], lr=1e-5, weight_decay=1e-5)

# data
train_data = pre_data(train_list)
val_data = pre_data(val_list)

# logging
utils.checkdir(f'out/checkpoints/{dataset_name}')
writer = utils.get_writer(f'out/logs/{dataset_name}')
training_loss_writer = utils.TBWriter(writer, 'scalar', 'loss/training')
test_mse_writer = utils.TBWriter(writer, 'scalar', 'Accuracy/MAE')
test_mae_writer = utils.TBWriter(writer, 'scalar', 'Accuracy/MSE')



# === TRAINING === #

best_mae = math.inf
for epoch in range(500):
    train(train_data, model, criterion, optimizer, epoch)
    mae, mse = validate(val_data, model)
    test_mae_writer(mae)
    test_mse_writer(mse)

    if mae < best_mae:
        torch.save(model.state_dict(), f'out/checkpoints/{dataset_name}/epoch_best_{mae}.pth')
        best_mae = mae





