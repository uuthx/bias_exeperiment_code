from functools import reduce
from torch import nn,optim
import torch
from load_data import load_data
import numpy as np


def rate_to_observe(data):
    data[data > 0 ] = 1
    return data

def load_datas():
    train,test = load_data()
    observe = rate_to_observe(train)
    return train,test,observe

class UPMF(nn.Module):
    def __init__(self,user_num,item_num,O,embed_k = 4):
        super(UPMF,self).__init__()
        self.Uemb = nn.Embedding(user_num,embed_k)
        self.Vemb = nn.Embedding(item_num,embed_k)

        

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.observe = torch.Tensor(O)

    def forward(self,user_index,item_index):
        U = self.Uemb(user_index)
        V = self.Vemb(item_index)
        out = U.mul(V).sum(1)

        return out
    def train(self,row,col,data,epochs,lr = 0.001,batch_size= 64,weight_decay = 0.1):
        optimize = optim.Adam(self.parameters(),lr=lr,weight_decay= weight_decay)
        loss_func = nn.MSELoss()

        epoch_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0
            p_item = torch.mean(self.observe,0)
            total_batch = len(row) // batch_size + 1

            optimize.zero_grad()
            for batch_id in range(total_batch):
                user_index = row[batch_id * batch_size : (batch_id + 1) * batch_size]
                item_index = col[batch_id * batch_size : (batch_id + 1) * batch_size]
                # print(p_item[item_index].size())
                pre = model(torch.LongTensor(user_index),torch.LongTensor(item_index)) * p_item[item_index]
                pre = self.relu(pre)
                loss = loss_func(pre,torch.FloatTensor(data[user_index,item_index]))
                loss.backward()
                optimize.step()

                epoch_loss += loss.item()

            print('epoch {} loss is {}'.format(epoch,epoch_loss))

            # self.observe = rate_to_observe(pre.detach())
    
if __name__ =="__main__":
    train,test,observe = load_datas()

    row,col = np.nonzero(train)
    model = UPMF(train.shape[0],train.shape[1],observe)
    model.train(row,col,observe,300)
    
    test_rol,test_col = np.nonzero(test)
    pre = model(torch.LongTensor(test_rol),torch.LongTensor(test_col)).detach().numpy()
    mse_func = lambda x,y: np.mean((x-y)**2)
    mse = mse_func(pre,test[test_rol,test_col])
    print(mse)