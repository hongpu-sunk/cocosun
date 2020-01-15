"""
模块导入
"""
import numpy 
import torch
from torch import nn

#---------------------------------------------------------------------------------------------------------
#数据预处理
data_length = 30
seq_length = 3

number = [i for i in range(data_length)]
li_x = []
li_y = []
for i in range(0, data_length - seq_length):
    x = number[i: i+seq_length]
    y = number[i + seq_length]
    li_x.append(x)
    li_y.append(y)
    
data_x = numpy.reshape(li_x,(len(li_x),1,seq_length))
data_x = torch.from_numpy(data_x / float(data_length)).float()
#将输入数据进行归一化
data_y = torch.zeros(len(li_y),data_length).scatter_(1, torch.tensor(li_y).unsqueeze_(dim=1), 1).float()


#----------------------------------------------------------------------------------------------------------
#定义网络模型
class net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layer):
        super(net, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.layer3 = nn.Softmax()
        
    def forward(self,x):
        x,_ = self.layer1(x)
        sample, batch, hidden = x.size()
        x = x.reshape(-1,hidden)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = net(seq_length, 32, data_length, 4)

#---------------------------------------------------------------------------------------
#定义损失函数和优化器
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#----------------------------------------------------------------------------------------
#训练模型
for _ in range(500):
    output = model(data_x)
    loss = loss_fun(data_y, output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (_ + 1)%50 == 0:
        print('Epoch:{}, Loss:{}'.format(_, loss.data))
        
#------------------------------------------------------------------------------------
#预测结果
result = model(data_x)
for target, pred in zip(data_y, result):
    print("accurate result:{}, predict:{}".format(target.argmax().data, pred.argmax().data))
        
        
        
        
        
