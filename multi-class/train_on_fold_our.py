import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
import numpy as np
from sklearn.decomposition import PCA
from utils import get_adj_mat,normalize_adj,preprocess_adj,get_train_test_set
import sys
import time
import os
from sklearn.preprocessing import label_binarize
from model import Discriminator,Model,Generator,get_edge_index
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from functools import reduce
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split,StratifiedKFold
from collections import defaultdict


'''Code for 5 fold cross-validation'''

parser = argparse.ArgumentParser()
parser.add_argument('--aggregator', type=str, default='neigh', help='which aggregator to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=120, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--n_topo_feats', type=int, default=120, help='dim of topology features')
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
            head,tail,relation = line.strip().split(' ')
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
            entity_name,id = line.strip().split('\t')
            entity_vocab.append(int(id))
    with open(relation_file_path,encoding='utf8') as t:
        for line in t:
            realtion,id = line.strip().split('	')
            relation_vocab.append(int(id))
    print(len(entity_vocab))
    print(len(relation_vocab))

# def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict):
#     print(f'Logging Info - Reading entity2id file: {file_path}' )
#     assert len(drug_vocab) == 0 and len(entity_vocab) == 0
#     with open(file_path, encoding='utf8') as reader:
#         count=0
#         for line in reader:
#             if(count==0):
#                 count+=1
#                 continue
#             drug, entity = line.strip().split('\t')
#             drug_vocab[entity]=len(drug_vocab)
#             entity_vocab[entity] = len(entity_vocab)

def read_example_file(file_path:str):
    print(f'Logging Info - Reading example file: {file_path}')
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            d1,d2,flag=line.strip().split(' ')
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
#read_entity2id_file('data/kegg/entity2id.txt',drug_vocab,entity_vocab)
# print(drug_vocab == entity_vocab)
def load_feat(file_path:str,drug_list):
    loaded_array = np.load(file_path)
    pca = PCA(256)
    result = np.array(loaded_array)
    loaded_array = pca.fit_transform(result)
    return loaded_array
def redefine_drug_id(drug_list:list,example_matrix):
    new_example = []
    for e in example_matrix:
        re = []
        e0_id = drug_list.index(e[0])
        e1_id = drug_list.index((e[1]))
        new_example.append([e0_id,e1_id,e[2]])
    new_example = np.array(new_example)
    return new_example
generate_dict('drugbank/entity2id_new.txt','drugbank/relation2id.txt')
#new_example_matrix= read_example_file('drugbank/multi_class_DDI_id_step2.txt')
drug_list = [i for i in range(1704)]
num_drug = 1704
# file = open("data/DRKG/new_approved_example.txt",'r')
# new_example_matrix = []
# file_line = file.readlines()
# for line in file_line:
#     line = line.strip().split(' ')
#     new_example_matrix.append([int(line[0]),int(line[1]),int(line[2])])
#new_example_matrix = np.array(new_example_matrix)
#new_example_matrix = redefine_drug_id(drug_list,example_matrix)
drug_list = np.array(drug_list)
drug_list = torch.from_numpy(drug_list)
drug_list = drug_list.to(torch.int64)
# print(example_matrix)
kg = read_kg('drugbank/train2id_refined.txt')

class DDIDataset(Dataset):
    '''Customized dataset processing class'''

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def merge(x, y):
    return x + y



def get_straitified_data(file_path,ratio=0.2):
    all_tup = []
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split(' ')
            all_tup.append((int(d1), int(d2), int(flag)))
    np.random.shuffle(all_tup)
    tuple_by_type = defaultdict(list)
    for h,t,r in all_tup:
        tuple_by_type[r-1].append((h,t,r-1))
    tuple_by_type.keys().__len__()
    train_edges = []
    test_edges = []
    splits = []
 
    for k in range(0,(int)(1/ratio)):
        edges = []
        for r in tuple_by_type.keys():
            test_set_size = int(len(tuple_by_type[r])*ratio)
            if k<9:
                edges.append(tuple_by_type[r][k*test_set_size:(k+1)*test_set_size])
            else:
                edges.append(tuple_by_type[r][k*test_set_size:])
        splits.append(edges)
    return splits        
   
#Generate training and test set for each fold
def make_data(splits, fold_k):
    test_edges = splits[fold_k * 2]
    vaild_edges = splits[fold_k * 2 + 1]
    train_edges = []
    for i in range(0, len(splits)):
        if i == (fold_k * 2) or i == (fold_k * 2 + 1):
            continue
        train_edges += splits[i]

    def merge(x, y):
        return x + y

    test_tups = np.array(reduce(merge, test_edges))
    vaild_tups = np.array(reduce(merge, vaild_edges))
    train_tups = np.array(reduce(merge, train_edges))

    train_edges = train_tups[:, :2]
    train_labels = train_tups[:, -1]
    vaild_edges = vaild_tups[:, :2]
    vaild_labels = vaild_tups[:, -1]
    test_edges = test_tups[:, :2]
    test_labels = test_tups[:, -1]
    return train_edges, train_labels, vaild_edges, vaild_labels, test_edges, test_labels


testset_ratio = 0.1
splits = get_straitified_data('drugbank/multi_class_DDI_id_step2.txt', testset_ratio)

feat_mat = load_feat('drugbank/mol_emb.npy',drug_list)
drug_list = drug_list.to(device)
print(os.path.basename(sys.argv[0]))
print("embedding size: ",n_out_feat)
print("training set ratio:",(1-testset_ratio))
print('shape of initial feature matrix',feat_mat.shape)
feat_mat = torch.from_numpy(feat_mat)
feat_mat = feat_mat.to(device)
print(feat_mat.shape)

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
    model_loss = loss(y_pred,labels)
    model_loss.backward()

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



results = [] #Record the prediction results for each fold

'''Perform 5 fold cross-validation'''
for k in range(0,5):
    
   
    print("training on fold ",k)
    train_edges, train_labels, vaild_edges, vaild_labels, test_edges, test_labels = make_data(splits, k)
    train_dataset = DDIDataset(train_edges, train_labels)
    vaild_dataset = DDIDataset(vaild_edges, vaild_labels)
    test_dataset = DDIDataset(test_edges, test_labels)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    vaild_loader = DataLoader(dataset=vaild_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = Model(num_drug,len(entity_vocab),len(relation_vocab),kg,args,device,n_topo_feats,n_out_feat,feat_mat.shape[1]).to(device)
    D_t2a = Discriminator(feat_mat.shape[1],1)
    D_a2t = Discriminator(n_topo_feats,1)
    model.to(device)
    D_t2a.to(device)
    D_a2t.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    trainer_E = torch.optim.RMSprop(model.parameters(),lr=1e-3)
    trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(),lr=1e-4)
    trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(),lr=1e-4)
 

    n_iterations = len(train_loader)
    num_epochs = n_epochs
    start = time.time()
   
    running_loss = 0.0
    running_correct = 0.0
    best_vaild_loss = float('inf')
    patience = 5
    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels,pred_labels = [],[]
        running_loss = 0.0
        running_correct = 0.0
        total_samples = 0
        for i,(edges,labels) in enumerate(train_loader):
   
            edges,labels = edges.to(device),labels.to(device)
            y_pred, loss = update_E(model, drug_list, feat_mat, edges, labels, D_t2a, D_a2t, criterion, trainer_E,
                                    device)
            update_D_t2a(model, drug_list, feat_mat, edges, D_t2a, loss, trainer_D_t2a, device)
            update_D_a2t(model, drug_list, feat_mat, edges, D_a2t, loss, trainer_D_a2t, device)
            running_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
            
            labels = labels.cpu().numpy()
            total_samples += labels.shape[0]
            true_labels.append(list(labels))
            running_loss += loss.item()
          

        print(f"epoch {epoch+1}/{num_epochs};trainging loss: {running_loss/n_iterations:.4f}")
        print(f"epoch {epoch+1}/{num_epochs};training set acc: {running_correct/total_samples:.4f}")
       
        # def merge(x,y):
        #     return x + y


        with torch.no_grad():
            model.eval()
            vaild_loss = 0
            for i, (edges, labels) in enumerate(vaild_loader):
                edges, labels = edges.to(device), labels.to(device)
                y_pred, _, _, _ = model(drug_list, feat_mat, edges, False)
                # pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
                # y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                # labels = labels.cpu().numpy()
                loss = criterion(y_pred, labels)
                vaild_loss += loss.item()
            vaild_loss /= len(vaild_loader)
            print(f"epoch {epoch + 1}/{num_epochs};vaild_lost: {vaild_loss:.4f}")
            if vaild_loss < best_vaild_loss:
                best_vaild_loss = vaild_loss
                counter = 0
                save_file = "drugbank_mulitclass_dim120_{}.pth".format(k)
                torch.save(model.state_dict(), save_file)
            else:
                counter += 1
                if counter >= patience:
                    print('Early stopping triggered')
                    break
        

    end = time.time()
    elapsed = end-start
    print(f"Training completed in {elapsed//60}m: {elapsed%60:.2f}s.")

    def roc_aupr_score(y_true, y_score, average="macro"):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            
            return auc(recall, precision)

        def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
            if average == "binary":
                return binary_metric(y_true, y_score)
            if average == "micro":
                y_true = y_true.ravel()
                y_score = y_score.ravel()
            if y_true.ndim == 1:
                y_true = y_true.reshape((-1, 1))
            if y_score.ndim == 1:
                y_score = y_score.reshape((-1, 1))
            n_classes = y_score.shape[1]
            score = np.zeros((n_classes,))
            for c in range(n_classes):
                y_true_c = y_true.take([c], axis=1).ravel()
                y_score_c = y_score.take([c], axis=1).ravel()
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)

        return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

    event_num = 86
    n_test_samples = 0
    n_correct = 0
    total_labels = []
    total_pred = []
    fold_results = []
    def writelabel(filename,pred_labels,true_labels):
        file = open(filename,'w')
        for i in range(len(pred_labels)):
            file.write(str(pred_labels[i])+' '+str(true_labels[i])+'\n')
    # Testing phase
    with torch.no_grad():
        model.load_state_dict(torch.load('drugbank_mulitclass_dim120_{}.pth'.format(k)))
        model.eval()
        for edges, labels in test_loader:
            edges, labels = edges.to(device), labels.to(device)
            y_pred, _, _, _ = model(drug_list, feat_mat, edges, False)
            y_hat = F.softmax(y_pred, dim=1)
            total_pred.append(y_hat.cpu().numpy())
            total_labels.append(labels.cpu().numpy())
            n_test_samples += edges.shape[0]
            n_correct += torch.sum(
                (torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()

        acc = 100.0 * n_correct / n_test_samples

        total_pred = np.vstack(total_pred)
        total_labels = np.concatenate(total_labels)
        pred_type = np.argmax(total_pred, axis=1)
        y_one_hot = label_binarize(total_labels, classes=np.arange(event_num))

        auc = roc_auc_score(y_one_hot, total_pred, average='micro')
        macro_precision = precision_score(total_labels, pred_type, average='macro')
        macro_recall = recall_score(total_labels, pred_type, average='macro')
        macro_f1 = f1_score(total_labels, pred_type, average='macro')
        micro_precision = precision_score(total_labels, pred_type, average='micro')
        micro_recall = recall_score(total_labels, pred_type, average='micro')
        micro_f1 = f1_score(total_labels, pred_type, average='micro')
        fold_results.append(k)
        fold_results.append(acc)
        fold_results.append(auc)
        fold_results.append(macro_precision)
        fold_results.append(macro_recall)
        fold_results.append(macro_f1)
        fold_results.append(micro_precision)
        fold_results.append(micro_recall)
        fold_results.append(micro_f1)
        output_file = 'mulitclass_true_pred{}.txt'.format(k)
        writelabel(output_file,pred_type,total_labels)
        print(f"test set accuracy: {acc}")
        print(f"AUC: {auc}")
        print(f"macro_precision: {macro_precision}")
        print(f"macro_recall: {macro_recall}")
        print(f"macro_f1: {macro_f1}")
        print(f"micro_precision: {micro_precision}")
        print(f"micro_recall: {micro_recall}")
        print(f"micro_f1: {micro_f1}")

    results.append(fold_results)

# Write cross-validation results to file
pd.DataFrame(results, columns=['fold', 'acc', 'auc', 'macro_precision', 'macro_recall', 'macro_f1', 'micro_precision',
                               'micro_recall', 'micro_f1']).to_csv('cross_validation_120.csv')