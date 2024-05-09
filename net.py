
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

# # 搭建CNN
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()   # 继承__init__功能
#         ## 第一层卷积
#         self.conv1 = nn.Sequential(
#             # 输入[1,28,28]
#             nn.Conv2d(
#                 in_channels=1,    # 输入图片的通道数
#                 out_channels=16,  # 输出图片的通道数
#                 kernel_size=5,    # 5x5的卷积核，相当于过滤器
#                 stride=1,         # 卷积核在图上滑动，每隔一个扫一次
#                 padding=2,        # 给图外边补上0
#             ),
#             # 经过卷积层 输出[16,28,28] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[16,14,14] 传入下一个卷积
#         )
#         ## 第二层卷积
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,    # 同上
#                 out_channels=32,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2
#             ),
#             # 经过卷积 输出[32, 14, 14] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
#         )
#         ## 输出层
#         self.output = nn.Linear(in_features=32*7*7, out_features=10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)           # [batch, 32,7,7]
#         x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
#         output = self.output(x)     # 输出[50,10]
#         return output
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
        #参数更新
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
    # if  j%20 == 0:
    #     writer.add_scalar("Train/Accuracy", 100.0*correct/total, j)
        
print('准确率：{:.4f}%'.format(100.0*correct/total))
####################################################################################


# # 下载MNist数据集
# train_data = torchvision.datasets.MNIST(
#     root="./datasets/",  # 训练数据保存路径
#     train=True,
#     transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
#     download=False,  # 是否需要下载
# )
# print(train_data.train_data.size())   # [60000,28,28]
# print(train_data.train_labels.size())  # [60000]
# # plt.imshow(train_data.train_data[0].numpy())
# # plt.show()
# print(123)

# test_data = torchvision.datasets.MNIST(root="./datasets/", train=False)
# print(test_data.test_data.size())    # [10000, 28, 28]
# # print(test_data.test_labels.size())  # [10000, 28, 28]
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
# test_y = test_data.test_labels[:2000]

# # 装入Loader中
# train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)


# def main():
#     # cnn 实例化
#     cnn = CNN()
#     print(cnn)

#     # 定义优化器和损失函数
#     optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#     loss_function = nn.CrossEntropyLoss()

#     # 开始训练
#     for epoch in range(epoches):
#         print("进行第{}个epoch".format(epoch))
#         for step, (batch_x, batch_y) in enumerate(train_loader):
#             output = cnn(batch_x)  # batch_x=[50,1,28,28]
#             # output = output[0]
#             loss = loss_function(output, batch_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if step % 50 == 0:
#                 test_output = cnn(test_x)  # [10000 ,10]
#                 pred_y = torch.max(test_output, 1)[1].data.numpy()
#                 # accuracy = sum(pred_y==test_y)/test_y.size(0)
#                 accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#                 print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


#     test_output = cnn(test_x[:10])
#     pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
#     print(pred_y)
#     print(test_y[:10])

if __name__ == "__main__":
    # main()
    freeze_support()
    print(1)
