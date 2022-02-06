import copy
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from load_data import load_data,rating_mat_to_sample


class PMF(nn.Module):
    def __init__(self,O,user_size,item_size,latent_size = 10,momuntum = 0.8):
        super(PMF,self).__init__()
        self.momuntm = momuntum
        self.U = nn.Parameter(torch.rand((user_size,latent_size),requires_grad=True))
        self.V = nn.Parameter(torch.rand((item_size,latent_size),requires_grad=True))
        self.O = O

    def forward(self,x):
        user = self.U[x[:,0]]
        item = self.V[x[:,1]]

        out = user.mm(item.T)

        return out


    def compute_P(self,x,y,y_ips = None):
        if y_ips is None:
            one_over_zl = torch.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = torch.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
    
    def fit(self,x,y,alpha = 0.01,batch_size = 128,epoch_nums = 1000,lr = 0.001,tol = 1e-4,opitm = None,loss = None):
        loss_all = []
        optimize = torch.optim.SGD(self.parameters(),lr= lr,momentum=self.momuntm)
        
        last_loss = 1e9

        nums_sample = len(x)
        total_batch = nums_sample // batch_size

        early_stop = 0

        for epoch in range(epoch_nums):
            all_idx = np.arange(nums_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            ips = self.compute_P(x,self.O)
            print(ips)
            return

            for batch_id in tqdm(range(total_batch),desc='epoch {}:'.format(epoch),mininterval=1):
                data = x[batch_id * batch_size:( batch_id + 1) * batch_size,:]
                ips_cur = ips[batch_id * batch_size:( batch_id + 1) * batch_size]
                rate_estimate = self.forward(data)
                rate_true = torch.Tensor(y[batch_id * batch_size:( batch_id + 1) * batch_size])
                loss = torch.sum( ips_cur * (rate_true - rate_estimate) ** 2) + alpha * torch.sum(torch.square(self.U)) + alpha * torch.sum(torch.square(self.V))
                # loss = bce_func(rate_estimate,rate_true) + alpha * torch.sum(torch.square(self.U)) + alpha * torch.sum(torch.square(self.V))

                optimize.zero_grad()
                loss.backward()
                optimize.step()

                epoch_loss += loss.item()
            print('the epoch {} loss is {}'.format(epoch,epoch_loss))
            loss_all.append(epoch_loss)
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[PMF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
            last_loss = epoch_loss
        
        print('training is end,the final loss is {}'.format(loss_all[-1]))


if __name__ =="__main__":
    x_train,x_test = load_data()
    train_x,train_y = rating_mat_to_sample(x_train)
    test_x,test_y = rating_mat_to_sample(x_test)
    

    nums_user = x_train.shape[0]
    nums_item = x_train.shape[1]
    O = torch.zeros((nums_user,nums_item))
    for id in train_x:
        O[id[0]][id[1]] = 1
    print(O)

    model = PMF(O,nums_user,nums_item)
    print([i for i in model.parameters()]) 
    model.fit(train_x,train_y)

    sigmoid_f = nn.Sigmoid()
    bce_f = nn.BCELoss(reduce=True)
    y_pre = model.forward(train_x)

    mes_func = lambda x,y: torch.mean((x-y)**2)
    
    print('mse is   {}'.format(mes_func(torch.Tensor(train_y),y_pre)))

    
    print('mse is   {}'.format(mes_func(model.forward(test_x),torch.Tensor(test_y))))

    torch.save(model.state_dict,'./pmf_observe.txt')
    
