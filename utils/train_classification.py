# 加上这句之后 即使在python2.X，使用print就得像python3.X那样加括号使用。python2.X中print不需要括号，而在python3.X中则需要。
from __future__ import print_function
import argparse  # 实现命令行中输入参数的传递
import os
import random
import torch
import torch.nn.parallel
# 优化器模块
import torch.optim as optim
# 处理数据集的模块
import torch.utils.data
# 从pointnet.pytorch/pointnet/dataset.py和pointnet.pytorch/pointnet/model.py中导入库
# 数据进行预处理的库
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
# pointnet的模型结构库
from pointnet.model import PointNetCls, feature_transform_regularizer
# 封装好的类
import torch.nn.functional as F
# 展示进度条的模块
from tqdm import tqdm

# 使用argparse 的第一步是创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser()
# 添加程序参数信息
# 终端键入batchsize大小
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
# 默认的数据集每个点云是2500个点
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
# 加载数据的进程数目
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
# epoch，训练多少轮 默认250
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
# 输出文件夹名称
parser.add_argument('--outf', type=str, default='cls', help='output folder')
# 预训练模型路径
parser.add_argument('--model', type=str, default='', help='model path')
# 这里，数据集的路径必须手动设置
parser.add_argument('--dataset', type=str, required=True, help="dataset path", default='.\modelnet40_normal_resampled')
# 数据集类型
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
# 是否进行特征变换
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

# 解析参数
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

# 返回1～10000间的一个整数，作为随机种子 opt的类型为：<class 'argparse.Namespace'>
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
# 保证在有种子的情况下生成的随机数都是一样的
random.seed(opt.manualSeed)
# 设置一个用于生成随机数的种子，返回的是一个torch.Generator对象
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))