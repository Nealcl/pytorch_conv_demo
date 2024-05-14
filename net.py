
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.multiprocessing import freeze_support
import matplotlib.pyplot as plt
import os
torch.manual_seed(1)
# 设置超参数
epoches = 2
batch_size = 50
learning_rate = 0.001


###################################################################################################
#cifa10数据集
#数据集加载
#对训练集及测试集数据的不同处理组合
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([     
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径   
train_data = datasets.CIFAR10(root=os.getcwd(), train=True,transform=transform_train,download=True)
test_data =datasets.CIFAR10(root=os.getcwd(),train=False,transform=transform_test,download=True)

#数据分批

#使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
#shuffle表示在每个epoch开始的时候，对数据进行重新排序
#数据分批之前：torch.Size([3, 32, 32])：Tensor[[32*32][32*32][32*32]],每一个元素都是归一化之后的RGB的值；数据分批之后：torch.Size([64, 3, 32, 32])
#数据分批之前：train_data([50000[3*[32*32]]])
#数据分批之后：train_loader([50000/64*[64*[3*[32*32]]]])
train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=2)
test_loader = Data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True,num_workers=2)

#模型加载，有多种内置模型可供选择
model = torchvision.models.densenet201(pretrained=False)

#定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
criterion = nn.CrossEntropyLoss()
#torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#模型和输入数据都需要to device
mode  = model.to(device)

#模型训练
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('cifar-10')
for epoch in range(epoches):
    for i,data in enumerate(train_loader):
        #取出数据及标签
        inputs,labels = data
        #数据及标签均送入GPU或CPU
        inputs,labels = inputs.to(device),labels.to(device)
        
        #前向传播
        outputs = model(inputs)
        #计算损失函数
        loss = criterion(outputs,labels)
        #清空上一轮的梯度
        optimizer.zero_grad()
        
        #反向传播
        loss.backward()
        #参数更新 init
        optimizer.step()
        #利用tensorboard，将训练数据可视化
        if  i%50 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch*len(train_loader)+i)
        print('it’s training...{}'.format(i))
    print('epoch{} loss:{:.4f}'.format(epoch+1,loss.item()))

#保存模型参数
torch.save(model,'cifar10_densenet161.pt')
print('cifar10_densenet161.pt saved')

#模型加载
model = torch.load('cifar10_densenet161.pt')
#测试
#model.eval()
model.train()

correct,total = 0,0
for j,data in enumerate(test_loader):
    inputs,labels = data
    inputs,labels = inputs.to(device),labels.to(device)
    #前向传播
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data,1)
    total =total+labels.size(0)
    correct = correct +(predicted == labels).sum().item()
    #准确率可视化
    if  j%20 == 0:
        writer.add_scalar("Train/Accuracy", 100.0*correct/total, j)
        
print('准确率：{:.4f}%'.format(100.0*correct/total))
####################################################################################


if __name__ == "__main__":
    # main()
    freeze_support()
    print(1)
