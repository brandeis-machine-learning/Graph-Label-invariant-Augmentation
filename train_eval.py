import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import pandas as pd
from utils import print_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_train_test(train_dataset,
                      test_dataset,
                      model_func,
                      epochs,
                      batch_size,
                      lr,
                      lr_decay_factor,
                      lr_decay_step_size,
                      weight_decay,
                      epoch_select,
                      with_eval_mode=True):
    assert epoch_select in ['test_last', 'test_max'], epoch_select

    model = model_func(train_dataset).to(device)
    print_weights(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_accs, test_accs = [], []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_loss, train_acc = train(
            model, optimizer, train_loader, device)
        train_accs.append(train_acc)
        test_accs.append(eval_acc(model, test_loader, device, with_eval_mode))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print('Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, train_accs[-1], test_accs[-1]))
        sys.stdout.flush()

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    t_end = time.perf_counter()
    duration = t_end - t_start

    if epoch_select == 'test_max':
        train_acc = max(train_accs)
        test_acc = max(test_accs)
    else:
        train_acc = train_accs[-1]
        test_acc = test_accs[-1]

    return train_acc, test_acc, duration


from copy import deepcopy


def k_fold(dataset, folds, epoch_select, dataset_name, n_percents=1):
    # n_splits = folds - 2

    if n_percents == 10:
        all_indices = torch.arange(0,len(dataset),1, dtype=torch.long)
        return [all_indices], [all_indices], [all_indices], [all_indices]


    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices, train_indices_unlabel = [], [], []
    save_test,  save_train, save_val, save_train_unlabel = [], [], [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))
        if len(save_test) > 0 and len(list(idx)) < len(save_test[0]):
            save_test.append(list(idx) + [list(idx)[-1]])
        else:
            save_test.append(list(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
        save_val = [save_test[i] for i in range(folds)]
        # n_splits += 1
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]
        save_val = [save_test[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train_all = train_mask.nonzero(as_tuple=False).view(-1)

        idx_train = []
        for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), dataset.data.y[idx_train_all]):
            idx_train.append(idx_train_all[idx])
            if len(idx_train) >= n_percents:
                break
        idx_train = torch.concat(idx_train).view(-1)
        
        train_indices.append(idx_train)
        cur_idx = list(idx_train.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train[0]):
            save_train.append(cur_idx + [cur_idx[-1]])
        else:
            save_train.append(cur_idx)

        # train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[train_indices[i].long()] = 0
        idx_train_unlabel = train_mask.nonzero(as_tuple=False).view(-1)
        train_indices_unlabel.append(idx_train_unlabel) # idx_train_all, idx_train_unlabel
        cur_idx = list(idx_train_unlabel.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train_unlabel[0]):
            save_train_unlabel.append(cur_idx + [cur_idx[-1]])
        else:
            save_train_unlabel.append(cur_idx)

    print("Train:", len(train_indices[i]), "Val:", len(val_indices[i]), "Test:", len(test_indices[i]))

    return train_indices, test_indices, val_indices, train_indices_unlabel


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def eval_acc(model, loader, device, with_eval_mode, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(data)
            pred = pred.max(1)[1]

            if suffix == 10:
                out2, _, pred2, _ = model.forward_cl(data, True, get_one_hot_encoding(data, model.n_class), grads=None, eta=eta)
                pred2 = pred2.max(1)[1]
        
        if suffix == 10:
            correct_invariant += pred.eq(pred2.view(-1)).sum().item()
        correct += pred.eq(data.y.view(-1)).sum().item()

    if suffix == 10:
        return correct / len(loader.dataset), correct_invariant / len(loader.dataset)
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _, _ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)




def get_one_hot_encoding(data, n_class):
    y = data.y.view(-1)
    encoding = np.zeros([len(y), n_class])
    for i in range(len(y)):
        encoding[i, int(y[i])] = 1
    return torch.from_numpy(encoding).to(device)


def train(model, optimizer, dataset, device, batch_size, eta):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    for data1, data2 in zip(loader1, loader2):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, None, eta=eta)
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        out2, _, pred2, pred_gcn2 = model.forward_cl(data2, True, None, grads=graph_grad, eta=eta)

        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        loss = model.loss_cl(out1[indices], out2[indices])

        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()

    return total_loss / len(loader1.dataset)



def train_label(model, optimizer, dataset, device, batch_size, eta):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    model.train()
    total_loss = 0
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)

        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, get_one_hot_encoding(data1, model.n_class), eta=eta)
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        out2, _, pred2, pred_gcn2 = model.forward_cl(data2, True, get_one_hot_encoding(data2, model.n_class), grads=graph_grad, eta=eta)

        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        loss = model.loss_cl(out1[indices], out2[indices])

        out, _, hidden, pred_gcn = model(data1)
        loss += (F.nll_loss(pred1, data1.y.view(-1))+F.nll_loss(pred2[indices], data2.y.view(-1)[indices])) #* 0.01

        pred = out.max(1)[1]
        correct += pred.eq(data1.y.view(-1)).sum().item()

        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()
    return total_loss / len(loader1.dataset), correct / len(loader1.dataset)


def cross_validation_with_label(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  aug1=None, aug_ratio1=None,
                                  aug2=None, aug_ratio2=None, suffix=None, eta=1.0, n_percents=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, train_idx_unlabel) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, dataset_name, n_percents=int(n_percents)))):

        train_idx[train_idx < 0] = train_idx[0]
        train_idx[train_idx >= len(dataset)] = train_idx[0]
        test_idx[test_idx < 0] = test_idx[0]
        test_idx[test_idx >= len(dataset)] = test_idx[0]
        val_idx[val_idx < 0] = val_idx[0]
        val_idx[val_idx >= len(dataset)] = val_idx[0]

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_dataset_unlabel = dataset[train_idx_unlabel]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        train_loader_unlabel = DataLoader(train_dataset_unlabel, batch_size, shuffle=True)


        dataset.aug = "none"
        model = model_func(dataset).to(device)
        optimizer_label = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Adam(model.parameters(), lr=lr/5, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs+1):

            train_loss = train(
                model, optimizer, train_dataset_unlabel, device, batch_size, eta)

            train_label_loss, train_acc = train_label(
                model, optimizer_label, train_dataset, device, batch_size, eta)


            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc(
                model, test_loader, device, with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_label_loss': train_label_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

            with open('logs/' + dataset_name + '_' + str(eta) + '_log', 'a+') as f:
                 f.write(str(epoch) + ' ' + str(train_loss) + ' ' + str(train_label_loss))
                 f.write('\n')
        print(fold, "finish run")


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean