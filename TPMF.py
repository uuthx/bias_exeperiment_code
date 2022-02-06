import math
from typing import Counter
from numpy.core.fromnumeric import size
from numpy.matrixlib import mat
import torch
import numpy as np
from torch.nn.modules.loss import BCELoss
from tqdm.std import tqdm
import load_data
import utils
import MF
from tqdm import tqdm
from sklearn.metrics import roc_auc_score



class TPMF(torch.nn.Module):
    def __init__(self,user_num,item_num,rate_embeding_U,rate_embeding_V,score_L = 5,epsilon = 0.5,embed_k = 10,observe_alpha = 1,observe_beta = 1):
        super(TPMF,self).__init__()

        #probability of item j is observable to user i
        self.miu = torch.FloatTensor(np.zeros((user_num,item_num)))

        #rate_embeding which is pretraining on MF
        self.rate_embeding_U = rate_embeding_U
        self.rate_embeding_V = rate_embeding_V 

        #user select matrix embeding,using L1 to optimize
        self.select_embeding_G = torch.nn.Embedding(user_num,embedding_dim=embed_k,norm_type=1)
        self.select_embeding_H = torch.nn.Embedding(item_num,embedding_dim=embed_k,norm_type=1)

        #reflect the rating value influence in select
        self.rate_select_A = torch.nn.Embedding(user_num,score_L,norm_type=1)
        self.rate_select_B = torch.nn.Embedding(item_num,score_L,norm_type=1)
        self.select_bias = torch.nn.Parameter(torch.tensor([0.5],requires_grad=True))
        self.epsilon = epsilon
        self.d = embed_k
        self.score_L = score_L
        self.u_num = user_num
        self.i_num = item_num
        self.sigmoid = torch.nn.Sigmoid()
        self.alpha = observe_alpha
        self.beta = observe_beta
        self.relu = torch.nn.ReLU()


    def CDF(self,data:torch.Tensor):
        mat_coin = (2/3.1415926) ** 0.5
        cdf =1 - (torch.exp(-1 * torch.pow( data,2) / 2) * (1 / ( 2 * 3.1415926) ** 0.5)) / (0.7 * data + mat_coin)
        return cdf



    def forward(self,x,is_Training = False,size= 64):
        u_idx = torch.LongTensor(x[:,0])
        i_idx = torch.LongTensor(x[:,1])
        G_emb = self.select_embeding_G(u_idx)
        H_emb = self.select_embeding_H(i_idx)
        A_emb = self.rate_select_A(u_idx)
        B_emb = self.rate_select_B(i_idx)
        U_emb = self.rate_embeding_U(u_idx)
        V_emb = self.rate_embeding_V(i_idx)
        # print('G size {}'.format(G_emb.size()))
        # print('h size {}'.format(H_emb.size()))
        # print('A size {}'.format(A_emb.size()))
        # print('B size {}'.format(B_emb.size()))
        # print('U size {}'.format(U_emb.size()))
        # print('V size {}'.format(V_emb.size()))
        # print('M_I size {}'.format(MF_I.size()))

        #calculate all estimate select,all is zeros except diag
        select =G_emb.mm(H_emb.T) + torch.max(A_emb + B_emb,dim=1)[0] + self.select_bias
        select = self.sigmoid(select)
        # print('select size{}'.format(select.size()))

        #calculate all estimate rates
        # print((G_emb.mul(H_emb).unsqueeze(1) + self.epsilon * MF_I).size())
        sigma = (G_emb.mul(H_emb).unsqueeze(1) + self.epsilon * torch.ones((1,self.d))) / (G_emb.unsqueeze(1).matmul(H_emb.unsqueeze(1).permute(0,2,1)) + self.epsilon * self.d)
        rates = (U_emb.mul(sigma.squeeze(1))).matmul(V_emb.T)
        rates = self.relu(rates)

        
        # print('rate size{}'.format(rates.size()))

        #calculate expection of observe
        p_select = 0.399 * torch.exp(-1 * select.pow(2) / 2)
        rate_ceil = torch.ceil(rates)
        rate_floor = torch.floor(rates)
        p_rate = self.CDF(rate_ceil) - self.CDF(rate_floor)
        miu = self.miu[:,i_idx][u_idx,:]
        calculate = miu.mul(p_rate.mul(p_select))
        observe = calculate / (calculate + 1-miu)
        if is_Training:
            return select,observe,rates
        else:
            return rates.detach().numpy(),observe.detach().numpy()


    def predicted(self,x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def fit(self,x,y,nums_epoch = 1000,batch_size = 64,lr = 0.01,lamb = 0.01,tol = 1e-4,verbose = True):
        optimizer = torch.optim.Adam(self.parameters(),lr=lr,weight_decay=lamb)
        func_loss_bce = torch.nn.BCELoss()
        func_loss_mse = torch.nn.MSELoss()
        last_loss = 1e9
        #initialize miu with item frequence
        count = Counter(x[:,1])
        for j in range(self.i_num):
            p = 0
            if j in count:
                p = count[j] / self.u_num
                for i in range(self.u_num):
                    self.miu[i][j] = p
        self.miu = torch.FloatTensor(self.miu)
        nums_sample = len(x)
        total_batch = nums_sample // batch_size

        rol = -0.5 * math.log(2*math.pi)

        early_stop = 0

        for epoch in range(nums_epoch):
            all_idx = np.arange(nums_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            print('epoch {}  is begin.......'.format(epoch))

            for idx in tqdm(range(total_batch),desc='epoch {}:'.format(epoch),mininterval=10):
                selected_idx = all_idx[batch_size * idx :(idx+ 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx])

                select,observe,rate=self.forward(sub_x,size = batch_size,is_Training= True)
                
                #get item which is truly observed
                rate_true =torch.diag(rate,0)
                select_true = torch.diag(select,0)

                select_estimate = select - torch.diag_embed(select_true)


                # xent_loss = torch.sum((rate_true - sub_y).pow(2)) + torch.sum((select_true - 1).pow(2))
                xent_loss = func_loss_mse(rate_true,sub_y) + func_loss_bce(select_true,torch.ones((batch_size)))
                rate_ceil = torch.ceil(rate)
                rate_floor = torch.floor(rate)
                p_rate = self.CDF(rate_ceil) - self.CDF(rate_floor)
                miss_loss = torch.sum(observe.mul(p_rate.mul(select_estimate.pow(2) / 2 + rol)))
                # L1 = lamb + torch.sum(torch.Tensor([p.abs().sum() for p in self.parameters()]))
                loss =  xent_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().numpy()
                #update miu
                _,observe_back = self.forward(sub_x,is_Training= False,size=batch_size)
                sum = np.sum(observe_back,0)
                observe_back[:,[range(batch_size)]] = sum
                # print(torch.min(observe))
                # print(self.miu[:,sub_x[:,1]][sub_x[:,0],:].size())
                # print('-----------------------------------------------------------')
                # print(self.miu[:,sub_x[:,1]][sub_x[:,0],:])
                for i_id,i in enumerate(sub_x[:,0]):
                    for j_id,j in enumerate(sub_x[:,1]):
                        self.miu[i][j] = observe_back[i_id][j_id] / batch_size
                # self.miu[:,sub_x[:,1]][sub_x[:,0],:] = (self.alpha + observe - 1) / (self.alpha + self.alpha + batch_size - 2)
                # print(self.miu[:,sub_x[:,1]][sub_x[:,0],:])
                # return
                # return


            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss
            print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))


            # if epoch % 10 == 0 and verbose:
            #     print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == nums_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")








if __name__ =="__main__":
    train_mat,test_mat = load_data.load_data()
    x_train,y_train = load_data.rating_mat_to_sample(train_mat)
    x_test,y_test = load_data.rating_mat_to_sample(test_mat)
    # y_test = utils.binarize(y_test)
    # y_train = utils.binarize(y_train)

    nums_user = train_mat.shape[0]
    nums_item = train_mat.shape[1]

    # mf = MF.MF(nums_user,nums_item,embedding_k=10)
    # mf.fit(x_train,y_train,lr = 0.01,lamb=1e-4,tol=1e-5,verbose=False)
    # U,V = mf.get_embeding()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mf = MF.MF(nums_user,nums_item)
    mf.load_state_dict(torch.load('./preTrain_mf'))
    U,V = mf.get_embeding()
    tpmf = TPMF(nums_user,nums_item,U,V)
    print([i for i in tpmf.parameters()])
    # all_idx = np.arange(len(x_test))
    # np.random.shuffle(all_idx)
    # selected_idx = all_idx[64 * 0 :(0+ 1) * 64]
    # sub_x = x_train[selected_idx]
    # tpmf.forward(sub_x,size=64)
    tpmf.fit(x_train,y_train,batch_size=128)
    test_pred = mf.predicted(x_test)
    mse_func = lambda x,y: np.mean((x-y)**2)
    mse_mf = mse_func(y_test,test_pred)
    # auc_score = roc_auc_score(y_test,test_pred)

    torch.save(mf.state_dict(),'./preTrain_mf')
    print('mse--------------',mse_mf)
    print([i for i in tpmf.parameters()])
    # print('auc--------------',auc_score)
    # torch.save(tpmf.state_dict,'./tpmf')
    # for i in tpmf.parameters():
    #     print(i)