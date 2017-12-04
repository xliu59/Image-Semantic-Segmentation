#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
import os
import os.path as osp
import argparse
# from __future__ import print_function
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import FCN8


parser = argparse.ArgumentParser(description="Save or load models.")
parser.add_argument('-e', '--epoch', type=int, default=10,
                    help='Number of iteration over the dataset to train')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-tb', '--test_batch_size', type=int, default=16,
                    metavar='N', help='test mini-batch size (default: 16)')
parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--disable_training', action='store_true', default=False,
                    help='Disable training')
parser.add_argument('--enable_testing', action='store_true', default=False,
                    help='Enable testing')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-s', '--save', type=str, help='save the model weights')
parser.add_argument('-l', '--load', type=str, help='load the model weights')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
class_num = 21

class VOC12(Dataset):
    def __init__(self, root_dir, txt_file, input_transform=None, target_transform=None):
        self.name_list = self.__readfile__(txt_file)
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'JPEGImages',self.name_list[idx]+".jpg")
        label_path = os.path.join(self.root_dir, 'SegmentationClass',self.name_list[idx]+".png")

        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(label_path, 'rb') as f:
            label = Image.open(f).convert('P')
        # print(np.shape(image))
        # print(np.shape(label))
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            label = np.array(label, dtype=np.int32)
            label[label==255] = -1
            label = torch.from_numpy(label).long()
        sample = {'image': image, 'label': label}
        return sample

    def __readfile__(self, txt_file):
        name_list = []
        with open(txt_file, 'r') as f:
            for line in f:
                data = line.strip()
                data = data.split(' ')
                name_list.append(data[0])
        return name_list

    def show_pair(self, idx):
        print('length of the dataset: ', len(self))
        sample = self[idx]
        img1 = np.transpose(sample['image'].numpy(), (1, 2, 0))
        img2 = np.transpose(sample['label'].numpy(), (1, 2, 0))
        # print(np.shape(img1))
        # print(np.shape(img2))
        plt.subplot(1, 2, 1)
        plt.imshow(img1, interpolation='nearest')
        plt.subplot(1, 2, 2)
        plt.imshow(img2[:,:,0], interpolation='nearest')
        plt.show()
        plt.tight_layout()
        plt.axis('off')

    def visualization(self, img, lbl, lp):  # TODO
        # jm: img transpose to PIL.image, lbl doesn't change
        img = np.array(transforms.ToPILImage()(img))
        img = img.astype(np.uint8)
        lbl = lbl.numpy().astype(np.uint8)
        lp = lp.numpy().astype(np.uint8)
        plt.subplot(131)
        plt.imshow(img, interpolation='nearest')
        plt.subplot(132)
        plt.imshow(lbl[:,:], interpolation='nearest', vmin = 0, vmax = 24)
        plt.subplot(133)
        plt.imshow(lp[:,:], interpolation='nearest', vmin = 0, vmax = 24)
        plt.show()
        plt.tight_layout()
        plt.axis('off')

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class fcn_4(nn.Module):
    def __init__(self, class_num=21):
        super(fcn_4, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, class_num, 1)
        self.score_pool2 = nn.Conv2d(128, class_num, 1)
        self.score_pool3 = nn.Conv2d(256, class_num, 1)
        self.score_pool4 = nn.Conv2d(512, class_num, 1)

        self.upscore2 = nn.ConvTranspose2d(class_num, class_num, 4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(class_num, class_num, 4, stride=2, bias=False)
        self.upscore8_ = nn.ConvTranspose2d(class_num, class_num, 4, stride=2, bias=False)
        self.upscore = nn.ConvTranspose2d(class_num, class_num, 8, stride=4, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        pool2 = h

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,9:9 + upscore_pool4.size()[2],9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8
        # spatial size 38
        h = self.upscore8_(h)
        # spatial size
        upscore_pool8 = h

        h = self.score_pool2(pool2)
        h = h[:, :, 14:14 + upscore_pool8.size()[2], 14:14 + upscore_pool8.size()[3]]
        score_pool2c = h

        h = upscore_pool8 + score_pool2c  # 1/8
        # spatial size 38
        h = self.upscore(h)
        # spatial size 312
        h = h[:, :, 40:40 + x.size()[2], 40:40 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn8(self, fcn8):
        for name, l1 in fcn8.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

    def transfer_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())


def train(epoch):
    model.train()
    # TODO: is ADAM really the best?
    # TODO: maybe adjust learning rate in training? http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    plot_x = []
    plot_y = []
    for i, data in enumerate(train_loader):
        images = data['image']
        labels = data['label']
        images, labels = Variable(images), Variable(labels)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(images)
        # labels = labels.type('torch.LongTensor').cuda()
        loss =  cross_entropy2d(output, labels)  # TODO: find out the difference between this and F.cross_entropy. Seems identical.
        loss /= len(output)  # normalizing when training in batches
        plot_x.append(len(plot_x)+len(train_loader)*epoch + 1)
        plot_y.append(loss.data[0])
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.data[0]))
        if (i==(len(train_loader)-1)):
            training_loss = 'FCN4_trainloss.txt'
            with open(training_loss, 'a') as f:
                for i in range(0, len(plot_x)):
                    f.write(" ".join([str(plot_x[i]), str(plot_y[i])]))
                    f.write('\n')

# evaluation tools
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=21):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def test(test_loader):
    model.eval()
    label_trues, label_preds = [], []
    print('Start testing')
    for i, data in enumerate(test_loader):
        images = data['image']
        labels = data['label']
        images, labels = Variable(images, volatile=True), Variable(labels)
        # print(images.size(), labels.size())
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        output = model(images)
        imgs = images.data.cpu()
        # lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_pred = output.data.max(1)[1].cpu()
        lbl_true = labels.data.cpu()
        # if i==0:
        #     print("bincount pre:",np.bincount(lbl_pred.numpy().flatten()))
        #     print("bincount true:",np.bincount(lbl_true.type('torch.LongTensor').numpy().flatten()))
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            # test_loader.dataset.visualization(img, lt, lp)
            lt = lt.numpy()
            lp = lp.numpy()
            label_trues.append(lt)
            label_preds.append(lp)
            #print(np.shape(label_trues), np.shape(label_preds))
    metrics = label_accuracy_score(label_trues, label_preds, n_class=class_num )
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))

if __name__ == "__main__":

    trans_image = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trans_target = transforms.Compose([transforms.Scale((224, 224))])
    train_data_root_dir = './VOC2012'
    train_data_txt_dir = './VOC2012/ImageSets/Segmentation/train.txt'
    train_set = VOC12(train_data_root_dir, train_data_txt_dir, input_transform=trans_image, target_transform=trans_target)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_data_root_dir = './VOC2012'
    test_data_txt_dir = './VOC2012/ImageSets/Segmentation/val.txt'
    test_set = VOC12(test_data_root_dir, test_data_txt_dir, input_transform=trans_image, target_transform=trans_target)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=2)
    # train_set.show_pair(10)

    if args.load:
        # load pretrained fcn_32 network
        load_path = args.load
        print('Load weights at {}'.format(load_path))
        fcn_8 = FCN8.fcn_8(class_num=class_num)
        fcn_8.load_state_dict(torch.load(load_path))
        # fcn_16 instance
        model = fcn_4(class_num=class_num)
        # copy params from vgg16
        model.copy_params_from_fcn8(fcn_8)
    else:
        # load pretrained vgg16 network
        vgg16 = models.vgg16(pretrained=True)
        # fcn_32 instance
        model = fcn_4(class_num=class_num)
        # copy params from vgg16
        model.transfer_from_vgg16(vgg16)

    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
        model.cuda()

    for epoch in range(0, args.epoch):  # loop over the dataset multiple times
        if not args.disable_training:
            train(epoch)
        if args.enable_testing:
            print('Train_set testing result:')
            test(train_loader)
            print('Test_set testing result:')
            test(test_loader)

    if args.save is not None:
        save_path = args.save
        print('Saving weights at {}'.format(save_path))
        torch.save(model.state_dict(), save_path)
