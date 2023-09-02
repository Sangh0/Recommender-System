import random
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.target_tensor[idx]
    


class SampleGenerator(object):
    def __init__(self, ratings):
        assert 'userID' in ratings.columns
        assert 'itemID' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings

        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userID'].unique())
        self.item_pool = set(self.ratings['itemID'].unique())

        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split(self.preprocess_ratings)

    def _binarize(self, ratings):
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings
    
    def _split(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['userID'])['timestamp'].rank(method='first', ascending=False) # sorting with the standard of timestamp
        test = ratings[ratings['rank_latest'] == 1] # the recent data is used for testing
        train = ratings[ratings['rank_latest'] > 1] # the rest is used for training
        assert train['userID'].nunique() == test['userID'].nunique()
        return train[['userID', 'itemID', 'rating']], test[['userID', 'itemID', 'rating']]
    
    def _sample_negative(self, ratings):
        interact_status = ratings.groupby('userID')['itemID'].apply(set).reset_index().rename(
            columns={'itemID': 'interacted_items'}) # grouping items into a single set for each user
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x) # define the remaining items, excluding the interacted items, as negative items
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99)) # randomly sample 99 negative items defined in the above code."
        return interact_status[['userID', 'negative_items', 'negative_samples']]
    
    def _get_train_loader(self, num_negatives, batch_size):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userID', 'negative_items']], on='userID')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))

        for row in train_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(int(row.rating))

            for i in range(num_negatives):
                users.append(int(row.userID))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    
    @property
    def _get_testset(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userID', 'negative_samples']], on='userID')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        
        for row in test_ratings.itertuples():
            test_users.append(int(row.userID))
            test_items.append(int(row.itemID))

            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userID))
                negative_items.append(int(row.negative_samples[i]))

        return [
            torch.LongTensor(test_users),
            torch.LongTensor(test_items),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items),
        ]
    

def get_data_loader(path: str, subset: str='train', batch_size: int=1024, num_negative: int=4):
    ml1m_rating = pd.read_csv(path, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

    sample_generator = SampleGenerator(ml1m_rating)

    if subset == 'train':
        return sample_generator._get_train_loader(num_negatives=num_negative, batch_size=batch_size)
    
    else:
        return sample_generator._get_testset