def tie_to_dict(users, items, outputs, neg_users, neg_items, neg_outputs):
    return {
        'users': users.view(-1).tolist(), 
        'items': items.view(-1).tolist(), 
        'outputs': outputs.view(-1).tolist(), 
        'neg_users': neg_users.view(-1).tolist(), 
        'neg_items': neg_items.view(-1).tolist(), 
        'neg_outputs': neg_outputs.view(-1).tolist(),
    }