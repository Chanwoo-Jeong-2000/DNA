import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, bpr_loss_our

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class SimGCL_my(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL_my, self).__init__(conf, training_set, test_set)
        args = self.config['SimGCL']
        self.scale = float(args['scale'])
        self.cl_rate = float(args['lambda'])
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.scale)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                pos, neg = self.model.item_bias.weight[pos_idx], self.model.item_bias.weight[neg_idx] 
                
                #rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                rec_loss = bpr_loss_our(user_emb, pos_item_emb, neg_item_emb, pos, neg)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, pos, neg) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                self.aim = self.model.item_bias
            temp = self.fast_evaluation(epoch)
            if temp == 'early_stop':
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)) + self.aim.weight.squeeze().detach()
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, scale):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.scale = scale
        self.item_bias = nn.Embedding(self.data.item_num, 1)
        torch.nn.init.xavier_uniform_(self.item_bias.weight)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        
        item_all_embeddings = F.normalize(item_all_embeddings, dim=-1, p=2) * self.scale
        
        return user_all_embeddings, item_all_embeddings
