
import pandas as pd
import numpy as np
from load_data import *
class Popularity():

    def __init__(self ,config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']

    def fit(self,train_data):
        column_names = train_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.counts_df = train_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        self.counts_df["Count"] = 1
        self.counts_df['popularity_score'] = self.counts_df.groupby(["ItemId"])['Count'].transform(sum) / self.num_users
        self.counts_df = self.counts_df[['ItemId','popularity_score']].drop_duplicates().set_index('ItemId').to_dict()


    def predict(self,test_data):
        column_names = test_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.test_data = test_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        prediction = self.test_data['ItemId'].map(self.counts_df['popularity_score']).fillna(1/(self.num_items+1)).values
        return prediction


class Propensity():

    def __init__(self ,config):
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.method = config['propensity_model']
        # if self.method == 'poisson':
        #     self.model = HPF(k=10,check_every=10, ncores=-1, maxiter=150)
        if self.method == 'popularity':
            self.model = Popularity(config)

    def fit(self,train_data):
        column_names = train_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.counts_df = train_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        self.counts_df["Count"] = 1
        self.model.fit(self.counts_df)

    def predict(self,test_data):
        column_names = test_data.columns
        userid = column_names[0]
        itemid = column_names[1]
        ratings = column_names[2]
        self.test_data = test_data.rename(columns = {userid:"UserId",itemid:"ItemId",ratings:"Count"})
        if self.method == 'poisson':
            prediction = self.model.predict(self.test_data["UserId"].values,self.test_data["ItemId"].values)
            return 1 - np.exp(-prediction)
        else:
            prediction = self.model.predict(self.test_data)
            return prediction


    def get_data(self,name = 'coat',data_type = 'test',datas = np.array([])):
        if datas.shape[0]:
            test = datas
        elif data_type == 'test':
            _,test = load_data(name)
        else:
            test,_ = load_data(name)
        if name == 'coat':
            user,item,rates = used_for_property(test)
            data = np.concatenate([user.reshape(-1,1),item.reshape(-1,1),rates.reshape(-1,1)],axis=1)
            data = pd.DataFrame(data,columns=['user','item','rates'])
        else:
            data = pd.DataFrame(test,columns=['user','item','rates'])

        return data

# if __name__ == '__main__':
#     config = {'num_users':290,'num_items':300,'propensity_model':'popularity'}
#     model = Propensity(config)
#     data = model.get_data(data_type='train')
#     print(data.columns)
#     model.fit(data)
#     score = model.predict(data)
