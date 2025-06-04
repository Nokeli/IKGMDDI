import torch
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''
    
    def __init__(self, num_drug_entity, dim, aggregator):
        super(Aggregator, self).__init__()
        self.num_drug_entity = num_drug_entity
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator
        self.w_1 = torch.nn.Linear(3*140,140,bias=True)
        self.w_2 = torch.nn.Linear(140,1,bias=True)
        self.nn = torch.nn.LeakyReLU()
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        self.num_drug_entity = user_embeddings.size(0)

        #neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        if self.aggregator == 'sum':
            output = self_vectors + neighbors_agg
            output = output.view((-1,self.dim))
            #output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((self.num_drug_entity, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        neigh = user_embeddings.shape[1]
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.num_drug_entity, 1, neigh, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated
    def _mix_neighbor_vectors_attention(self,neighbor_vectors, neighbor_relation, user_embeddings):
        neigh = user_embeddings.shape[1]
        input = torch.empty(neighbor_vectors.shape[0], neighbor_vectors.shape[1], neighbor_vectors.shape[2], 1)
        f_user_embeddings = torch.ones_like(input).to('cuda')
        user_embeddings = user_embeddings.view((self.num_drug_entity, 1, neigh, self.dim))
        f_user_embeddings_hat = f_user_embeddings * user_embeddings
        head_rel = torch.concatenate([f_user_embeddings_hat,neighbor_relation],dim=-1)
        head_rel_tail = torch.concatenate([head_rel,neighbor_vectors],dim=-1)
        c_head_rel_tail = self.w_1(head_rel_tail)
        b_head_rel_tail = self.w_2(c_head_rel_tail)
        b_head_rel_tail = self.nn(b_head_rel_tail)
        attention_score = F.softmax(b_head_rel_tail,dim=-2)
        output = attention_score * c_head_rel_tail
        neighbors_aggregated = output.sum(dim=2)
        return neighbors_aggregated
