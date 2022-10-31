import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.dataset import ICDADataset
import config.configuration as config
from Module.ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss

random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 30
lr = 1e-3
resume_epoch = 0


def save_checkpoints(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'v3_ctpn_ep{epoch:02d}'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')
    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save checkpoint to {}'.format(check_path))
    print('save checkpoint to {}'.format(check_path))


def weight_init(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        model.weight.data.normal_(0., 0.2)
    elif class_name.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = config.pretrained_weight
    print('exists pretrained', os.path.exists(checkpoints_weight))
    if os.path.exists(checkpoints_weight):
        pretrained = False

    dataset = ICDADataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CTPN_Model()
    model.to(device)

    if os.path.exists(checkpoints_weight):
        print('using pretrained weight : {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weight_init)

    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)

    criterion_cls = RPN_CLS_Loss(device)
    criterion_regr = RPN_REGR_Loss(device)

    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(resume_epoch + 1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#' * 50)
        epoch_size = len(dataset) // 1
        model.train()

        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)

        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)

            optimizer.zero_grad()

            out_cls, out_regr = model(imgs)
            loss_cls = criterion_cls(out_cls, clss)
            loss_regr = criterion_regr(out_regr, regrs)

            loss = loss_cls + loss_regr
            loss.backward()
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()

            mmp = batch_i + 1

            print(f'EPOCH : {epoch} / {epochs - 1}---'
                  f'Batch : {batch_i}/{epoch_size}\n'
                  f'batch: loss_cls : {loss_cls.item():.4f} --- loss_regr:{loss_regr.item():.4f}---loss : {loss.item():.4f}\n'
                  f'Epoch: loss_cls : {epoch_loss_cls / mmp:.4f}---loss_regr : {epoch_loss_regr / mmp:.4f}---'
                  f'loss : {epoch_loss / mmp:.4f}\n')

        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size

        print(f'Epoch:{epoch}---{epoch_loss_cls:.4f}---{epoch_loss_regr:.4f}---{epoch_loss:.4f}')

        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_cls = epoch_loss_cls
            best_loss_regr = epoch_loss_regr
            best_model = model
            save_checkpoints({'model_state_dict': best_model.state_dict(),
                              'epoch': epoch}, epoch, best_loss_cls, best_loss_regr, best_loss)
