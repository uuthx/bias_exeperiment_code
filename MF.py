from torch import nn
import torch
class MF(nn.Module):
    # choose pmf or mf through set the optimizer in fit with the parameter weight_decay 
    def __init__(self,nums_user,nums_item,embedding_k = 10,device = 'cpu',use_mul = False,*args,**kwargs):
        super(MF,self).__init__()
        self.nums_user = nums_user
        self.nums_item = nums_item
        self.emmbedding_k = embedding_k

        self.W = nn.Embedding(self.nums_user,self.emmbedding_k)
        self.H = nn.Embedding(self.nums_item,self.emmbedding_k)
        self.sigmoid = nn.Sigmoid()
        self.xent_func = nn.MSELoss()
        self.device = device
        self.use_mul = use_mul
        self.sigmoid = nn.Sigmoid()

    def forward(self,user_idx,item_idx):

        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb),1)
        
        return self.sigmoid(out)
    def prediect(self,user_idx,item_idx):

        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb),1)
        
        return self.sigmoid(out).detach().numpy()
        
