from torch.utils.data import Dataset
from load_data import *

class DebiasDataset(Dataset):
    def __init__(self,datatype = 'train',name = 'coat') -> None:
        super().__init__()
        self.train,self.test = load_data(name)
        self.datatype = datatype
        self.name = name
    
    def __getitem__(self, index):

        if self.datatype == 'train':
            user,item,rates = self.train[index]
        # if self.datatype == 'test':
        else:
            user,item,rates = self.test[index]
            if self.name == 'yahoo':
                rates = binaary_rates(rates)
        return (user,item),rates
    
    def __len__(self):
        if self.datatype == 'train':
            return len(self.train)
        # if self.datatype == 'test':
        else:
            return len(self.test)
