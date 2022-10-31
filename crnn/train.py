import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

import config.Config as config
from crnn.data.dataset import CRNNDataset, alignCollate, resizeNormalize
from model.Model import CRNN
from utils.Utils import strLabelConverter, averager

random.seed(config.random_seed)
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
train_dataset = CRNNDataset(info_file=config.info_dir, image_dir=config.image_dir)
assert train_dataset

train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                          shuffle=True, collate_fn=alignCollate(config.imgH, config.imgW, keep_ratio=config.keep_ratio))

converter = strLabelConverter(config.alphabet)

criterion = CTCLoss()
best_loss = float('inf')


def save_checkpoints(state, epoch, loss, ext='pth'):
    check_path = os.path.join(config.save_path, f'crnn_ep{epoch:02d}'
                                                f'_{loss:.4f}.{ext}')
    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save model to {}'.format(check_path))
    print('save model to {} successfully'.format(check_path))


def weight_init(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


model = CRNN(config.imgH, config.nc, config.nclass, config.nh)

if config.pretrained_model_path != '' and os.path.exists(config.pretrained_model_path):
    model.load_state_dict(torch.load(config.pretrained_model_path))
else:
    model.apply(weight_init)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

loss_avg = averager()

if config.adam:
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(model.parameters(), lr=config.lr)


def model_eval():
    with open(config.eval_dir, 'r', encoding='utf-8') as f:
        content = f.readlines()
        num_all = 0
        num_acc = 0
        for line in content:
            line.strip()
            fname, label = line.split('\t')
            img = cv2.imread(os.path.join(config.eval_image_dir, fname))
            res = predict(img)
            res = res.strip()
            label = label.strip()
            if res == label:
                num_acc += 1
            num_all += 1
    return num_acc, num_all


def predict(img):
    transorm = resizeNormalize((config.imgW, config.imgH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(np.uint8(img))
    img = transorm(img)
    img = img.view(1, *img.size())
    img = Variable(img)
    img = img.cuda()
    preds = model(img)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def evaluate(model, epoch):
    print('Evaluation Starting...')
    model.eval()
    num_acc, num_all = model_eval()
    accuracy = num_acc / float(num_all)

    print('Accuracy in Validation is {:.4f}'.format(accuracy))


def train_batch(model, criterion, optimizer):
    data = train_iter.next()
    images, tests = data
    batch_size = images.size(0)
    image = images.to(device)

    text, length = converter.encode(tests)

    preds = model(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size

    if torch.isnan(cost):
        print(batch_size, tests)
    else:
        model.zero_grad()
        cost.backward()
        optimizer.step()

    return cost


for epoch in range(config.epochs):
    loss_avg.reset()
    print('epoch {} [{}/{}] starting....'.format(epoch + 1, epoch + 1, config.epochs))
    train_iter = iter(train_loader)
    i = 0
    n_batch = len(train_loader)

    while i < n_batch:
        model.train()
        cost = train_batch(model, criterion, optimizer)
        loss_avg.add(cost)
        loss_avg.add(cost)
        i += 1
    print('Training Loss: {:.4f}'.format(loss_avg.val()))
    # TODO maybe can use tensorbord to display training state
    if loss_avg.val() < best_loss:
        best_loss = loss_avg.val()
        save_checkpoints(model.state_dict(), epoch + 1, best_loss)
    evaluate(model, epoch + 1)
