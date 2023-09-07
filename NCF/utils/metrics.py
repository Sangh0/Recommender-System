import math
import pandas as pd
from typing import *


def _preprocess(subjects: Dict):
    assert len(subjects) == 6
    assert list(subjects.keys()) == ['users', 'items', 'outputs', 'neg_users', 'neg_items', 'neg_outputs']
    
    users, items, outputs = subjects['users'], subjects['items'], subjects['outputs']
    neg_users, neg_items, neg_outputs = subjects['neg_users'], subjects['neg_items'], subjects['neg_outputs']
    
    test = pd.DataFrame({
        'user': users,
        'test_item': items,
        'test_output': outputs,
    })

    full = pd.DataFrame({
        'user': neg_users + users,
        'item': neg_items + items,
        'output': neg_outputs + outputs,
    })
    
    full = pd.merge(full, test, on=['user'], how='left')
    full['rank'] = full.groupby('user')['output'].rank(method='first', ascending=False)
    full.sort_values(['user', 'rank'], inplace=True)
    return full


class NDCG(object):
    
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, subjects: Dict):
        full = _preprocess(subjects)
        top_k = full[full['rank'] <= self.top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()


class HitRate(object):

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, subjects: Dict):
        full = _preprocess(subjects)
        top_k = full[full['rank'] <= self.top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        return len(test_in_top_k) * 1.0 / full['user'].nunique()