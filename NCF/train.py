import os
import time
import ast
import argparse
import logging
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.utils import tie_to_dict
from utils.metrics import NDCG, HitRate
from utils.scheduler import PolynomialLRDecay, CosineWarmupLR
from utils.callback import CheckPoint, EarlyStopping
from utils.dataset import get_data_loader
from models.neumf import NeuMF
from models.mlp import MLP
from models.gmf import GMF


logger = logging.getLogger('The logs of model training')
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


def train_on_epoch(model, train_loader, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    for batch_idx, (users, items, ratings) in enumerate(train_loader):
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        outputs = model(users, items)
        loss = loss_func(outputs.view(-1), ratings)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (batch_idx+1)


def valid_on_epoch(model, valid_loader, device):
    model.eval()
    with torch.no_grad():
        users, items  = valid_loader[0].to(device), valid_loader[1].to(device)
        neg_users, neg_items = valid_loader[2].to(device), valid_loader[3].to(device)
        
        outputs = model(users, items)
        neg_outputs = model(neg_users, neg_items)

        users, items = users.cpu(), items.cpu()
        neg_users, neg_items = neg_users.cpu(), neg_items.cpu()
        outputs, neg_outputs = outputs.cpu(), neg_outputs.cpu()

        src = tie_to_dict(users, items, outputs, neg_users, neg_items, neg_outputs)
    return src


def fit(
    model, 
    train_loader, 
    valid_loader, 
    metric_top_k: int=10,
    epochs: int=100,
    lr: float=0.001,
    weight_decay: float=5e-4,
    momentum: Optional[float]=None,
    opt_: str='momentum',
    lr_sche_: str='poly',
    lr_scheduling: bool=True,
    check_point: bool=True,
    cp_standard: str='ndcg',
    early_stop: bool=False,
    es_patience: Optional[int]=None,
    prj_: str='exp1',
):
    assert opt_ in ('sgd', 'momentum', 'adam')
    assert lr_sche_ in ('poly', 'cosine')
    assert cp_standard in ('ndcg', 'hitrate')

    log_dir = f'./runs/train/{prj_}'
    os.makedirs(log_dir + '/weights', exist_ok=True)
    cp = CheckPoint(verbose=True)

    es_path = log_dir + '/weights/es_weight.pt'
    patience = es_patience if es_patience is not None else 20
    es = EarlyStopping(verbose=True, patience=patience, path=es_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device is {device}')

    # model
    model = model.to(device)
    logger.info('model loading complete...!')

    # loss funciton
    loss_func = nn.BCELoss()

    # metrics
    ndcg_func = NDCG(top_k=metric_top_k)
    hit_rate_func = HitRate(top_k=metric_top_k)

    # optimizer
    if opt_ == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif opt_ == 'momentum':
        momentum = momentum if momentum is not None else 0.9
        optimizer = optim.SGD(
            model.parameters(),
            momentum=momentum,
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    # schedulers
    if lr_sche_ == 'poly':
        lr_scheduler = PolynomialLRDecay(
            optimizer=optimizer,
            max_decay_steps=epochs,
        )

    else:
        lr_scheduler = CosineWarmupLR(
            optimizer=optimizer,
            epochs=epochs,
            warmup_epochs=int(epochs * 0.1),
        )

    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    loss_list = []
    val_ndcg_list, val_hitrate_list = [], []
    start_training = time.time()
    pbar = tqdm(range(epochs), total=int(epochs))

    for epoch in pbar:
        epoch_time = time.time()

        ############## training ##############
        train_loss = train_on_epoch(
            model,
            train_loader,
            optimizer,
            loss_func,
            device,
        )
        loss_list.append(train_loss)

        ############## validating ##############
        src = valid_on_epoch(
            model,
            valid_loader,
            device,
        )
        ndcg, hit_rate = ndcg_func(src), hit_rate_func(src)
        val_ndcg_list.append(ndcg)
        val_hitrate_list.append(hit_rate)

        logger.info(f'\n{"="*30} Epoch {epoch+1}/{epochs} {"="}*30'
                    f'\ntime: {(time.time() - epoch_time):.2f}s'
                    f'\n    lr = {optimizer.param_groups[0]["lr"]}')
        logger.info(f'\ntrain average loss: {train_loss:.3f}')
        logger.info(f'\nvalid average NDCG: {ndcg:.3f}, Hit Rate: {hit_rate:.3f}')
        logger.info(f'\n{"="*50}')

        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('valid/NDCG', ndcg, epoch)
        writer.add_scalar('valid/Hit Rate', hit_rate, epoch)

        if lr_scheduling:
            lr_scheduler.step()

        std = 1 - ndcg if cp_standard == 'ndcg' else 1 - hit_rate

        if check_point:
            path = log_dir + '/weights/check_point_{:03d}.pt'.format(epoch)
            cp(std, model, path)
            
        if early_stop:
            es(std, model)
            if es.early_stop:
                logger.info('\n##########################\n'
                            '##### Early Stopping #####\n'
                            '##########################')
                break

    logger.info(f'\nTotal training time : {time.time() - start_training:2f}s')
    return {
        'model': model,
        'loss': loss_list,
        'val_ndcg': val_ndcg_list,
        'val_hitrate': val_hitrate_list,
    }


def arg_as_list(param):
    arg = ast.literal_eval(param)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError('Argument \ "%s\" is not a list'%(arg))
    return arg


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Model', add_help=False)

    parser.add_argument('--prj_name', type=str, default='prj1',
                        help='the folder name')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument()

    # model parameters
    parser.add_argument('--num_users', type=int, default=6040,
                        help='the number of users in dataset')
    parser.add_argument('--num_items', typa=int, default=3706,
                        help='the number of items in dataset')
    parser.add_argument('--latent_dim_mf', type=int, default=8,
                        help='the dimension of latent vector in MF network')
    parser.add_argument('--latent_dim_mlp', type=int, default=8,
                        help='the dimension of latent vector in MLP network')
    parser.add_argument('--layers', type=arg_as_list, default=[16, 32, 16, 8],
                        help='the list consisting of the number of layers')
    
    # hyperparameters
    parser.add_argument('--metric_top_k', type=int, default=10,
                        help='the k is rating to measure metric')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum constant')
    parser.add_argument('--opt_type', type=str, default='adam', choices=['sgd', 'momemtum', 'adam'],
                        help='optimizer')
    parser.add_argument('--lr_sche_type', type=str, default='poly', choices=['poly', 'cosine'],
                        help='learning rate scheduler')
    parser.add_argument('--lr_scheduling', action='store_true',
                        help='scheduling the learning rate')
    parser.add_argument('--check_point', action='store_true',
                        help='model check point')
    parser.add_argument('--cp_std', type=str, default='ndcg', choices=['ndcg', 'hitrate'],
                        help='set the standard of metric for check point')
    parser.add_argument('--early_stop', action='store_true',
                        help='early stopping to overcome over-ftting')
    parser.add_argument('--es_patience', type=int, default=10,
                        help='the patience for early stopping')
    
    return parser


def main(args):
    train_loader = get_data_loader(path=args.data_path, subset='train', batch_size=args.batch_size)
    val_data = get_data_loader(path=args.data_path, subset='val')

    model = NeuMF(
        num_users=args.num_users, 
        num_items=args.num_items, 
        latent_dim_mf=args.latent_dim_mf,
        latent_dim_mlp=args.latent_dim_mlp,
        layers=args.layers,
    )

    history = fit(
        model, 
        train_loader, 
        val_data, 
        metric_top_k=args.metric_top_k,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        opt_=args.opt_type,
        lr_sche_=args.lr_sche_type,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        cp_standard=args.cp_std,
        early_stop=args.early_stop,
        es_patience=args.es_patience,
        prj_=args.prj_name,
    )