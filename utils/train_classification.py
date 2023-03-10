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
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
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

# 调用pointnet.pytorch/pointnet/dataset.py中的ShapeNetDataset类，创建针对shapenet数据集的类对象
if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(  # 训练集
        root=opt.dataset,
        classification=True,  # 打开分类的选项
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(  # 测试集
        root=opt.dataset,
        classification=True,
        split='test',  # 标记为测试
        npoints=opt.num_points,
        data_augmentation=False)
# 调用pointnet.pytorch/pointnet/dataset.py中的ModelNetDataset类，创建针对modelnet40数据集的类对象
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

# 用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,  # 将数据集的顺序打乱
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))  # 12137 2874
num_classes = len(dataset.classes)
print('classes', num_classes)  # classes 16

# 创建文件夹，若无法创建，进行异常检测
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 调用model.py的PointNetCls定义分类函数
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

# 如果有预训练模型，将预训练模型加载
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# 优化器：adam-Adaptive Moment Estimation(自适应矩估计)，利用梯度的一阶矩和二阶矩动态调整每个参数的学习率
# betas：用于计算梯度一阶矩和二阶矩的系数
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# 学习率调整：每个step_size次epoch后，学习率x0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# 将所有的模型参数移到GPU中
classifier.cuda()

# 计算batch的数量
num_batch = len(dataset) / opt.batchSize

# 开始一趟一趟的训练
for epoch in range(opt.nepoch):
    scheduler.step()
    # 将一个可遍历对象组合为一个索引序列，同时列出数据和数据下标,(0, seq[0])...
    # __init__(self, iterable, start=0)，参数为可遍历对象及起始位置
    for i, data in enumerate(dataloader, 0):
        points, target = data  # 读取待训练对象点云与标签
        target = target[:, 0]  # 取所有行的第0列
        points = points.transpose(2, 1)  # 改变点云的维度
        points, target = points.cuda(), target.cuda()  # tensor转到cuda上
        optimizer.zero_grad()  # 梯度清除，避免backward时梯度累加
        classifier = classifier.train()# 训练模式，使能BN和dropout
        pred, trans, trans_feat = classifier(points)  # 网络结果预测输出
        # 损失函数：负log似然损失，在分类网络中使用了log_softmax，二者结合其实就是交叉熵损失函数
        loss = F.nll_loss(pred, target)
        # 对feature_transform中64X64的变换矩阵做正则化，满足AA^T=I
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward() # loss反向传播
        optimizer.step() # 梯度下降，参数优化
        pred_choice = pred.data.max(1)[1] # max(1)返回每一行中的最大值及索引,[1]取出索引（代表着类别）
        correct = pred_choice.eq(target.data).cpu().sum() # 判断和target是否匹配，并计算匹配的数量
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        # 每10次batch之后，进行一次测试
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()  # 测试模式，固定住BN和dropout
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
    # 保存权重文件在cls/cls_model_1.pth
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

# 在测试集上验证模型的精度
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
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
