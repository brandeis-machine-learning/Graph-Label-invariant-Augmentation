from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from gcn_conv import GCNConv


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        self.n_class = dataset.num_classes
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features
        if collapse:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, dataset.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = GCNConv(hidden_in, hidden, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            if self.res_branch == "resnet":
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
            else:
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
            self.bn_hidden = BatchNorm1d(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, dataset.num_classes)
            self.lin_class2 = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.proj_head2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.collapse:
            return self.forward_collapse(x, edge_index, batch, xg)
        elif self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        elif self.res_branch == "BNReLUConv":
            return self.forward_BNReLUConv(x, edge_index, batch, xg)
        elif self.res_branch == "ConvReLUBN":
            return self.forward_ConvReLUBN(x, edge_index, batch, xg)
        elif self.res_branch == "resnet":
            return self.forward_resnet(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_collapse(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    ### This one!!!
    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, batch))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, batch))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg

        # pred_gcn = self.predictor2(x)


        graph_representation = x
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        hidden_representation = x
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1), graph_representation, hidden_representation, x

    def forward_cl(self, data, aug=False, label=None, grads=None, eta=1.0):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        return self.forward_BNConvReLU_cl(x, edge_index, batch, xg, aug, label, grads, eta)



    def predictor(self, x):
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
    # def predictor2(self, x):
    #     if self.dropout > 0:
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.lin_class2(x)
    #     return F.log_softmax(x, dim=-1)

    def augmentation(self, image, label, grads, aug, eta):
        patient, count = 10, 0
        error_num, error = 99999, 99999
        image_present = torch.mean(torch.sqrt(torch.sum((image-torch.mean(image,dim=0))**2, dim=-1)))#torch.max(torch.sqrt(torch.sum(image**2, dim=-1)))
        epsilon = image_present * eta
        rand_idx = torch.randperm(len(image))

        while True:
            sign_data_grad = torch.rand((50,image.size(1)), dtype=torch.float32).cuda() - 0.5
            sign_data_grad = sign_data_grad/torch.sqrt(torch.sum(sign_data_grad**2, dim=-1)).unsqueeze(1)
            perturbed_images = image.unsqueeze(1) + torch.normal(mean=epsilon, std=epsilon*0.04)*sign_data_grad.unsqueeze(0)

            h_origin = self.predictor(image)
            h_pred = self.predictor(perturbed_images)   # batch*10*class

            cur_error_num, cur_error = 0, torch.zeros([1,])[0].cuda()
            arg_h_pred = torch.argmax(h_pred, dim=-1)   # batch*10
            if label == None:
                arg_h_origin = torch.argmax(h_origin, dim=-1)
                _, index = torch.max(h_origin, dim=-1)

                index_tuple = (torch.arange(h_pred.size(0))[:,None], torch.arange(h_pred.size(1))[None,:], index[:,None])
                cur_pred = h_pred[index_tuple]
            else:
                arg_h_origin = torch.argmax(label, dim=-1)
                cur_pred = h_pred * label.unsqueeze(1)
                cur_pred, _ = torch.min(cur_pred, dim=-1)

            eq = arg_h_pred - arg_h_origin.unsqueeze(1)
            eq_mask = (eq==0)
            neq_mask = (eq_mask==False)

            row_check = torch.any(eq_mask, dim=-1)

            if (label == None) and (torch.all(row_check) == False) and (count < patient):
                # print("row check ----------------")
                # print(row_check)
                # epsilon *= 0.5
                # count += 1
                continue

            if aug:
                cur_pred = cur_pred*eq_mask + (1 - cur_pred)*neq_mask
                cur_pred, index = torch.min(cur_pred, dim=-1)
                index_tuple = (torch.arange(image.size(0))[:], index)
                perturbed_image_final = perturbed_images[index_tuple]
            else:
                perturbed_image_final = image
            break

        return perturbed_image_final


    def forward_BNConvReLU_cl(self, x, edge_index, batch, xg=None, aug=False, label=None, grads=None, eta=1.0):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, batch))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index, batch))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg

        # out2 = x
        # if self.dropout > 0:
        #     out2 = F.dropout(out2, p=self.dropout, training=self.training)
        # out2 = self.proj_head2(out2)
        # pred_gcn = self.predictor2(x)

        if aug:
            x = self.augmentation(x, label, grads, aug, eta)

        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        out = x
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.proj_head(out)
        pred = self.predictor(x)

        pred_gcn = pred
        return out, x, pred, pred_gcn

    def forward_BNReLUConv(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index, batch)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_, edge_index, batch)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        
        # batch_size *= 2
        # x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        '''
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        self_sim = sim_matrix[range(batch_size), list(range(int(batch_size/2), batch_size))+list(range(int(batch_size/2)))]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim - self_sim)
        loss = - torch.log(loss).mean()
        '''
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim #/ (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        
        return loss


    def forward_ConvReLUBN(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index, batch))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x, edge_index, batch))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, edge_index, batch, xg=None):
        # this mimics resnet architecture in cv.
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index, batch)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i*3+0](x_))
            x_ = self.convs[i*3+0](x_, edge_index, batch)
            x_ = F.relu(self.bns_conv[i*3+1](x_))
            x_ = self.convs[i*3+1](x_, edge_index, batch)
            x_ = F.relu(self.bns_conv[i*3+2](x_))
            x_ = self.convs[i*3+2](x_, edge_index, batch)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__