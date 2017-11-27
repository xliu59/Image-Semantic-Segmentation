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

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy


parser = argparse.ArgumentParser(description="Save or load models.")
parser.add_argument('-e', '--epoch', type=int, default=100,
                    help='Number of iteration over the dataset to train')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-tb', '--test_batch_size', type=int, default=1,
                    metavar='N', help='test mini-batch size (default: 1)')
parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--disable_training', action='store_true', default=False,
                    help='Disable training')
parser.add_argument('--enable_testing', action='store_true', default=False,
                    help='Enable testing')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-s', '--save', type=str, help='save the model weights')
parser.add_argument('-l', '--load', type=str, help='load the model weights')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


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


trans = transforms.Compose([transforms.Scale((227,227)),transforms.ToTensor()])
train_data_root_dir = '/media/jm/000B48300008D6EB/datasets/VOCdevkit/VOC2012'
train_data_txt_dir = '/media/jm/000B48300008D6EB/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
train_set = VOC12(train_data_root_dir, train_data_txt_dir,input_transform=trans, target_transform=trans)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
# TODO: divide train and test set
test_set = VOC12(train_data_root_dir, train_data_txt_dir,input_transform=trans, target_transform=trans)
test_loader = DataLoader(train_set, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
# trainset.show_pair(50)


class fcn_32(nn.Module):
    def __init__(self, class_num=21):
        super(fcn_32, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

            # conv4
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        self.f_conv = nn.Sequential(
            # fully convolutional 1
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),  # TODO: Does dropout probability matter?

            # fully convolutional 2
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.up_sampling = nn.Sequential(
            nn.Conv2d(4096, class_num, 1),
            nn.ConvTranspose2d(class_num, class_num, 64,
                               stride=32, bias=False)
        )

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
        h = self.conv(h)
        h = self.f_conv(h)
        h = self.up_sampling(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        return h

    def transfer_from_vgg16(self, vgg16):
        for l1, l2 in zip(vgg16.features, self.conv):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for l1, l2 in zip(vgg16.classifier, self.f_conv):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data.view(l2.weight.size())
                l2.bias.data = l1.bias.data.view(l2.bias.size())


torch.manual_seed(1)
# load pretrained vgg16 network
vgg16 = models.vgg16(pretrained=True)
# fcn_32 instance
model = fcn_32()
# copy params from vgg16
model.transfer_from_vgg16(vgg16)
if args.cuda:
    torch.cuda.manual_seed(1)
    model.cuda()

if args.load:
    load_path = args.load
    print('Loading weights from {}'.format(load_path))
    model.load_state_dict(torch.load(load_path))


# TODO: is ADAM really the best?
# TODO: maybe adjust learning rate in training? http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def train(epoch):
    model.train()
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
        labels = labels.type('torch.LongTensor')
        loss =  cross_entropy2d(output, labels)  # TODO: find out the difference between this and F.cross_entropy. Seems identical.
        loss /= len(output)  # normalizing when training in batches
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))





for epoch in range(1):  # loop over the dataset multiple times
    if not args.disable_training:
        train(epoch)
    if args.enable_testing:
        test()


if args.save is not None:
    save_path = args.save
    print('Saving weights at {}'.format(save_path))
    torch.save(model.state_dict(), save_path)