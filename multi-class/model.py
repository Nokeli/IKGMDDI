import math
from typing import Hashable
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import get_adj_mat, load_feat, normalize_adj, preprocess_adj, get_train_test_set
import sys
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import gmtime, strftime, localtime
import numpy as np
#import torchsnooper
from torch_geometric.nn import GCNConv, GATConv
from utils import get_adj_mat

'''Model definition'''
import sys
import torch
import random
import numpy as np
import copy
from aggregator import Aggregator


# 构建知识图谱的KGCN网络
class KGCN(torch.nn.Module):
    """
    input：药物的list列表，其中输入的药物list的标号为图谱中实体的编号
    output：药物节点的embedding
    构建参数：
    num_drug_entity: 药物实体的个数
    num_ent : 图谱中实体的个数
    num_rel: 图谱中关系的个数
    kg：构建好的图谱
    args: 参数
    device: CPU or GPU
    """

    def __init__(self, num_drug_entity, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_drug_entity = num_drug_entity
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        # self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.num_drug_entity, self.dim, args.aggregator)

        self._gen_adj()

        self.drug = torch.nn.Embedding(num_drug_entity, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        self.adj_ent = self.adj_ent.to(self.device)
        self.adj_rel = self.adj_rel.to(self.device)

    def forward(self, drug_entity_list):
        '''
        input: drug_entity_list is a list of drug 编号为图谱的实体的参数
        drug_entity_list : [number of drug]
        '''
        self.list_size = drug_entity_list.size(0)
        # change to [batch_size, 1]
        u = drug_entity_list.view((-1, 1))
        #        u = u.to(self.device)

        # [number of drug, dim]
        #       user_embeddings = self.drug(u)
        user_embeddings = self.drug(u).squeeze(dim=1)

        entities, relations = self._get_neighbors(u)

        item_embeddings = self._aggregate(user_embeddings, entities, relations)

        return item_embeddings

    # @torchsnooper.snoop()
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            test = self.adj_ent[entities[h]]
            test = test.view((self.list_size, -1))
            neighbor_entities = (self.adj_ent[entities[h]]).view((self.list_size, -1))
            neighbor_relations = (self.adj_rel[entities[h]]).view((self.list_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.list_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.list_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=entity_vectors[hop],
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.list_size, self.dim))


# Graph Attention network
class GAT(torch.nn.Module):
    def __init__(self, num_nodes, num_outputs, num_hidden, init_feat=64):
        super(GAT, self).__init__()
        self.X = nn.Parameter(torch.randn((num_nodes, init_feat)), requires_grad=True)
        nn.init.xavier_uniform_(self.X)  # Initialize node feature matrix with xavier initialization
        # Define two conv layers
        self.conv1 = GATConv(init_feat, num_hidden, dropout=0.5, heads=2, concat=False)
        self.conv2 = GATConv(num_hidden, num_outputs, dropout=0.5, heads=2, concat=False)

    def forward(self, edge_index):
        x = self.conv1(self.X, edge_index)

        x = self.conv2(x, edge_index)

        return x


# Generator definition with WGAN style
#生成器添加了一个relu层
class Generator(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(num_inputs, 100)
        self.l2 = nn.Linear(100, num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(num_inputs, 100)
        self.l2 = nn.Linear(100, num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.l1(x)

        x = self.l2(x)

        return x


# WGAN Style discriminator
class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(num_inputs, 100)
        self.l2 = nn.Linear(100, num_outputs)
        self.relu = F.relu

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        y = x.mean(0)
        return y


class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(nn.Linear(num_inputs, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
                                    nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                                    nn.Linear(256, num_outputs)
                                    )
        # self.layers = nn.Sequential(nn.Linear(num_inputs, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        #                             nn.Linear(256, num_outputs)
        #                             )

    def forward(self, x):
        output = self.layers(x)
        return output


class Model(nn.Module):
    def __init__(self, num_drug_entity, num_ent, num_rel, kg, args, device, gcn_outputs, num_outputs, attr_dim):
        super(Model, self).__init__()
        # self.GCN = GCN(num_nodes,gcn_outputs,gcn_hidden)
        # self.encoder = GAT(num_nodes,gcn_outputs,gcn_hidden) #Topology structure encoder with GAT
        self.encoder = KGCN(num_drug_entity, num_ent, num_rel, kg, args, device)
        self.g_t = MLP(gcn_outputs, num_outputs)  # Projecting structural embedding to the common space
        self.g_t2a = Generator(num_outputs, attr_dim)  # Generator for structure to attribute
        self.g_a = MLP(attr_dim, num_outputs)  # MLP to encoding drug attribute
        self.g_a2t = Generator(num_outputs, gcn_outputs)  # Generator for attribute to structure
        #self.classifier = DNN(num_outputs * 4, 1)
        self.classifier = DNN(632*2, 86)
        self.sigmod = nn.Sigmoid()

    def forward(self, drug_entity_list, attr_mtx, x, *args):
        # 通过图卷积得到节点特征，维度为(1710,80)
        gcn_out = self.encoder(drug_entity_list)
        # 通过MLP网络将图卷积的特征进行处理得道：（1710,128）
        topo_emd = self.g_t(gcn_out)
        # 通过生成器将图特征映射到药物特征维度：（1710，256）
        t2a_mtx = self.g_t2a(topo_emd)
        # 将预训练好的药物特征进行MLP映射，得到（1710,128）
        attr_emd = self.g_a(attr_mtx)
        # 通过生成器将药物特征映射到图特征空间，得到（1710,80）
        a2t_mtx = self.g_a2t(attr_emd)

        # Concatnation
        embedding = torch.hstack([gcn_out, topo_emd, attr_emd,attr_mtx])
        #embedding = torch.hstack([topo_emd, attr_emd])
        # attr_mtx = 256 gcn_out=80 attr_emd = 128 topo_embed = 128
        self.h_t = topo_emd
        self.h_a = attr_emd
        self.topo_emb = gcn_out
        self.attr_emb = attr_mtx
        self.false_topo = a2t_mtx
        self.false_attr = t2a_mtx

        x = x.long()

        X = torch.hstack([embedding[x[:, 0]], embedding[x[:, 1]]])
        output = self.classifier(X)
        output = output
        return output, gcn_out, t2a_mtx, a2t_mtx


def get_edge_index(edge_list):
    edge_list_image = edge_list[:, [1, 0]]
    edge_list_image
    edge_index = np.vstack([edge_list, edge_list_image])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index.t().contiguous()
