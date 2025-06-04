
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
from utils import get_adj_mat,load_feat,normalize_adj,preprocess_adj,get_train_test_set,get_full_dataset
import sys
import time
import os

from model import Discriminator,Model,Generator,get_edge_index

from sklearn.neighbors import KernelDensity
import random
from sklearn.manifold import TSNE

from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from functools import reduce
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold,train_test_split


'''Code for 5 fold cross-validation'''
parser = argparse.ArgumentParser()
parser.add_argument('--aggregator', type=str, default='concat', help='which aggregator to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=80, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--n_topo_feats', type=int, default=80, help='dim of topology features')
parser.add_argument('--n_hid', type=int, default=256, help='num of hidden features')
parser.add_argument('--n_out_feat', type=int, default=128, help='num of output features')
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
args = parser.parse_args()

# 知识图谱结构特征的维度
n_topo_feats = args.n_topo_feats
# 隐藏层特征的维度
n_hid = args.n_hid
# 输出特征的维度
n_out_feat = args.n_out_feat
#训练的轮数
n_epochs = args.n_epochs
# 批处理的大小
batch_size = args.batch_size
drug_vocab = {}
entity_vocab = []
relation_vocab = []

print(os.path.basename(sys.argv[0]))
print("embedding size: ",n_out_feat)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    
    
def read_kg(file_path: str):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            count +=1
            head,relation,tail = line.strip().split('\t')
            # # undirected graph
            if int(head) in kg:
                kg[int(head)].append((int(relation), int(tail)))
            else:
                kg[int(head)] = [(int(relation), int(tail))]
            if int(tail) in kg:
                kg[int(tail)].append((int(relation), int(head)))
            else:
                kg[int(tail)] = [(int(relation), int(head))]
    print('Logging Info - Constructing adjacency matrix...')
    # adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    # adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    print(len(kg))
    return kg
def generate_dict(entity_file_path,relation_file_path):
    with open(entity_file_path, encoding='utf8') as f:
        for line in f:
            id, entity_name = line.strip().split('\t')
            entity_vocab.append(int(id))
    with open(relation_file_path,encoding='utf8') as t:
        for line in t:
            id, realtion = line.strip().split('\t')
            relation_vocab.append(int(id))
def read_example_file(file_path:str,separator:str):
    print(f'Logging Info - Reading example file: {file_path}')
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            d1,d2,flag=line.strip().split(separator)
            examples.append([int(d1),int(d2),int(flag)])
    print(len(examples))
    # drug_list = []
    # for e in examples:
    #     if e[0] not in drug_list:
    #         drug_list.append(e[0])
    #     if e[1] not in drug_list:
    #         drug_list.append(e[1])
    # drug_list = sorted(drug_list)
    #num_drug = max(drug_list)+1
    examples_matrix=np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    # X=examples_matrix[:,:2]
    # y=examples_matrix[:,2:3]
    # train_data_X, valid_data_X,train_y,val_y = train_test_split(X,y, test_size=0.2,stratify=y)
    # train_data=np.c_[train_data_X,train_y]
    # valid_data_X, test_data_X,val_y,test_y = train_test_split(valid_data_X,val_y, test_size=0.5)
    # valid_data=np.c_[valid_data_X,val_y]
    # test_data=np.c_[test_data_X,test_y]
    return examples_matrix
def load_feat(file_path:str,drug_list):
    loaded_array = np.load(file_path)
    pca = PCA(256)
    result = np.array(loaded_array)
    loaded_array = pca.fit_transform(result)
    return loaded_array
generate_dict('data/new_DRKG/entities.tsv','data/new_DRKG/relations.tsv')
new_example_matrix= read_example_file('data/new_DRKG/DDI_pos_neg.txt','\t')
drug_list = [i for i in range(2322)]
num_drug = 2322
new_example_matrix = np.array(new_example_matrix)
#new_example_matrix = redefine_drug_id(drug_list,example_matrix)
drug_list = np.array(drug_list)
drug_list = torch.from_numpy(drug_list)
drug_list = drug_list.to(torch.int64)
# print(example_matrix)
kg = read_kg('data/new_DRKG/train.tsv')
# true_edges,num_nodes = get_adj_mat()


# all_edges,all_labels = get_full_dataset() #Get all DDIs in dataset


feat_mat = load_feat('data/DRKG/data/DRKG/drug_smiles_deal.npy',drug_list)
feat_mat = torch.from_numpy(feat_mat)
drug_list = drug_list.to(device)
feat_mat = feat_mat.to(device)


class DDIDataset(Dataset):
    '''Customized dataset processing class'''
    def __init__(self,x,y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples


def update_E(net_E, drug_entity_list, feat_mat, batch_edges, labels, D_t2a, D_a2t, loss, trainer_E, device):
    #   model,train_edges_true,train_feat_mat,edges,labels,D_t2a,D_a2t,criterion,trainer_E,device
    '''This function mainly used to optimize parameters of encoders'''

    y_pred, gcn_out, t2a_mtx, a2t_mtx = net_E(drug_entity_list, feat_mat, batch_edges)

    trainer_E.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    # Calcute adversarial loss
    mone = (one * -1).to(device)
    D_a2t.eval()
    D_t2a.eval()
    fake_y_a2t = D_a2t(a2t_mtx)
    fake_y_a2t.backward(one)

    fake_y_t2a = D_t2a(t2a_mtx)
    fake_y_t2a.backward(one)

    trainer_E.step()
    trainer_E.zero_grad()

    # Calcute prediction loss
    y_pred, gcn_out, t2a_mtx, a2t_mtx = net_E(drug_entity_list, feat_mat, batch_edges)
    y_pred = y_pred.reshape(-1)
    labels = labels.to(torch.float32)
    model_loss = loss(y_pred, labels)
    model_loss.backward(retain_graph=True)

    # trainer_E.step()
    trainer_E.step()
    return y_pred, model_loss

def update_D_t2a(net_E, drug_entity_list, feat_mat, batch_edges, D_t2a, loss, trainer_D_t2a, device):


    '''This function mainly used to optimize parameters of discriminator for topology-to-attribute'''
    # edge_index = get_edge_index(edge_list).to(device)

    clamp_lower = -0.01
    clamp_upper = 0.01
    # Perform gradient clip
    for p in D_t2a.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)
    trainer_D_t2a.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _, _, t2a_mtx, _ = net_E(drug_entity_list, feat_mat, batch_edges)
    fake_y = D_t2a(t2a_mtx)
    fake_y.backward(mone)

    real_y = D_t2a(feat_mat)
    real_y.backward(one)

    trainer_D_t2a.step()
    return

def update_D_a2t(net_E, drug_entity_list, feat_mat, batch_edges, D_a2t, loss, trainer_D_a2t, device):
    '''This function mainly used to optimize parameters of discriminator for attribute-to-topology'''
    # edge_index = get_edge_index(edge_list).to(device)

    clamp_lower = -0.01
    clamp_upper = 0.01
    # Perform gradient clip
    for p in D_a2t.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)

    trainer_D_a2t.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _, gcn_out, _, a2t_mtx = net_E(drug_entity_list, feat_mat, batch_edges)
    fake_y = D_a2t(a2t_mtx)

    fake_y.backward(mone)
    real_y = D_a2t(gcn_out)
    real_y.backward(one)

    trainer_D_a2t.step()
    return

results = []  #Record the prediction results for each fold

skf = StratifiedKFold(n_splits=5) #Stratified split policy is adopted
np.random.shuffle(new_example_matrix)
new_example_data = new_example_matrix[:,:2]
new_example_label = new_example_matrix[:,2]
'''Perform 5 fold cross-validation'''
for k,(train_index,test_index) in enumerate(skf.split(new_example_data,new_example_label)):
    train_set,test_vaild_set = new_example_matrix[train_index],new_example_matrix[test_index]
    train_label,test_vaild_set_label = new_example_label[train_index],new_example_label[test_index]
    vaild_set, test_set = train_test_split(test_vaild_set,test_size=0.5,shuffle=False)
    vaild_label,test_label = train_test_split(test_vaild_set_label,test_size=0.5,shuffle=False)
   
    train_dataset = DDIDataset(train_set,train_label)
    vaild_dataset = DDIDataset(vaild_set, vaild_label)
    test_dataset = DDIDataset(test_set, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    vaild_loader = DataLoader(vaild_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = Model(num_drug,len(entity_vocab),len(relation_vocab),kg,args,device,n_topo_feats,n_out_feat,feat_mat.shape[1]).to(device)
    
    D_t2a = Discriminator(feat_mat.shape[1],1)
    D_a2t = Discriminator(n_topo_feats,1)
    model.to(device)
    D_t2a.to(device)
    D_a2t.to(device)
    criterion = nn.BCELoss().to(device)

    trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
    trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(), lr=1e-4)
    trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(), lr=1e-4)

    best_vaild_loss = float('inf')
    n_iterations = len(train_loader)
    start = time.time()
    patience = 5
    running_loss = 0.0
    running_correct = 0.0
    # Training phase
    for epoch in range(n_epochs):
        model.train()
        true_labels,pred_labels = [],[]
        running_loss = 0.0
        running_correct = 0.0
        total_samples = 0
        for i,(edges,labels) in enumerate(train_loader):
            edges, labels = edges.to(device), labels.to(device)
            y_pred, loss = update_E(model, drug_list, feat_mat, edges, labels, D_t2a, D_a2t, criterion, trainer_E,
                                    device)
            update_D_t2a(model, drug_list, feat_mat, edges, D_t2a, loss, trainer_D_t2a, device)
            update_D_a2t(model, drug_list, feat_mat, edges, D_a2t, loss, trainer_D_a2t, device)
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
            y_pred = y_pred.cpu().detach().numpy().reshape(-1).round()
            labels = labels.cpu().numpy()
            total_samples += labels.shape[0]
            true_labels.append(list(labels))
            running_loss += loss.item()
            running_correct += (y_pred == labels).sum().item()

        print(f"epoch {epoch+1}/{n_epochs};trainging loss: {running_loss/n_iterations:.4f}")
        print(f"epoch {epoch+1}/{n_epochs};training set acc: {running_correct/total_samples:.4f}")
    
        def merge(x,y):
            return x + y
        
        true_labels = reduce(merge,true_labels)
        pred_labels = reduce(merge,pred_labels)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        lr_precision, lr_recall, _ = precision_recall_curve(true_labels, pred_labels)
        aupr = auc(lr_recall, lr_precision)
        auroc = roc_auc_score(true_labels,pred_labels)

        with torch.no_grad():
            model.eval()
            vaild_loss = 0
            for i, (edges, labels) in enumerate(vaild_loader):
                edges, labels = edges.to(device), labels.to(device)
                y_pred, _, _, _ = model(drug_list, feat_mat, edges, False)
                y_pred = y_pred.reshape(-1)
                labels = labels.to(torch.float32)
                loss = criterion(y_pred, labels)
                vaild_loss += loss.item()
            vaild_loss /= len(vaild_loader)
            print(f"epoch {epoch + 1}/{n_epochs};vaild_lost: {vaild_loss:.4f}")
            if vaild_loss < best_vaild_loss:
                best_vaild_loss = vaild_loss
                counter = 0
                save_file = "DRKG_neigh_{}.pth".format(k)
                torch.save(model.state_dict(), save_file)
            else:
                counter += 1
                if counter >= patience:
                    print('Early stopping triggered')
                    break
      

    end = time.time()
    elapsed = end-start
    print(f"Training completed in {elapsed//60}m: {elapsed%60:.2f}s.")

    n_test_samples = 0
    n_correct = 0
    total_labels = []
    total_pred = []
    fold_results = []
    # Testing phase
    with torch.no_grad():
        model.load_state_dict(torch.load('DRKG_neigh_{}.pth'.format(k)))
        model.eval()
        for edges,labels in test_loader:
            edges,labels = edges.to(device),labels.to(device)
            y_pred, _, _, _ = model(drug_list, feat_mat, edges, False)
            total_pred.append(y_pred.cpu().reshape(-1))
            y_pred = y_pred.cpu().numpy().reshape(-1).round()
            total_labels.append(labels.cpu())
            labels = labels.cpu().numpy()

            n_test_samples += edges.shape[0]
            n_correct += (y_pred == labels).sum()

        # Calculate evaluation indexes
        acc = 100.0 * n_correct/n_test_samples
        total_pred = torch.cat(total_pred)
        total_labels = torch.cat(total_labels)       
        lr_precision, lr_recall, _ = precision_recall_curve(total_labels,total_pred)
        aupr = auc(lr_recall, lr_precision)
        auroc = roc_auc_score(total_labels,total_pred )
        fold_results.append(k)
        fold_results.append(acc)
        fold_results.append(aupr)
        fold_results.append(auroc)
        print(f"test set accuracy: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
    results.append(fold_results)
  

#Write cross-validation results to file
pd.DataFrame(results,columns=['fold','acc','aupr','auroc']).to_csv('cross_validation.csv')
