import networks as nw
import arguments as ag
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
import numpy as np
import torch
import csv
import os
from scipy.spatial import distance
# import torch_sparse as ts


class EDGE(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        hidden = 128
        self.parser = nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0:self.input_dim]
        x2 = x[:, self.input_dim:]
        h1 = self.parser(x1)
        h2 = self.parser(x2)
        p = (self.cos(h1, h2) + 1)*0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()
    return ((dataset - mean) / std).astype(dtype)


def intensityNormalisationFeatureScaling(dataset, dtype):
    max_value = dataset.max()
    min_value = dataset.min()

    return ((dataset - min_value) / (max_value - min_value)).astype(dtype)


class Data_Loader:
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2
        self.node_ftr = None

    # def load_data(self, connectivity='correlation', atlas='ho'):
    #     subject_IDs = get_ids()
    #     labels = get_subject_score(subject_IDs, score='Group')
    #     num_nodes = len(subject_IDs)
    #     ages = get_subject_score(subject_IDs, score='Age')
    #     genders = get_subject_score(subject_IDs, score='Gender')
    #     y_onehot = np.zeros([num_nodes, self.num_classes])
    #     y = np.zeros([num_nodes])
    #     age = np.zeros([num_nodes], dtype=np.float32)
    #     gender = np.zeros([num_nodes], dtype=np.int32)
    #     for i in range(num_nodes):
    #         y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
    #         y[i] = int(labels[subject_IDs[i]])
    #         age[i] = float(ages[subject_IDs[i]])
    #         gender[i] = genders[subject_IDs[i]]
    #     self.y = y
    #     self.raw_features = ts.get_node_feature()
    #     phonetic_data = np.zeros([num_nodes, 2], dtype=np.float32)
    #     phonetic_data[:, 0] = gender
    #     phonetic_data[:, 1] = age
    #     self.pd_dict['Gender'] = np.copy(phonetic_data[:, 0])
    #     self.pd_dict['Age'] = np.copy(phonetic_data[:, 1])
    #     phonetic_score = self.pd_dict
    #     return self.raw_features, self.y, phonetic_data, phonetic_score

    # def data_split(self, n_folds):
    #     skf = StratifiedKFold(n_splits=n_folds)
    #     cv_splits = list(skf.split(self.raw_features, self.y))
    #     return cv_splits

    def get_PAE_inputs(self, non_img):
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = non_img.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edge_net_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = get_static_affinity_adj(self.node_ftr, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edge_net_input[flatten_ind] = np.concatenate((non_img[i], non_img[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.1)[0]
        edge_index = edge_index[:, keep_ind]
        edge_net_input = edge_net_input[keep_ind]

        return edge_index, edge_net_input

    def get_inputs(self, non_img, embeddings, phonetic_score):
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = non_img.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edge_net_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = get_static_affinity_adj(self.node_ftr, phonetic_score)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edge_net_input[flatten_ind] = np.concatenate((non_img[i], non_img[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.1)[0]
        edge_index = edge_index[:, keep_ind]
        edge_net_input = edge_net_input[keep_ind]

        return edge_index, edge_net_input


def get_subject_score(subject_list, score):
    scores_dict = {}

    phenotype = "/phenotypic_information.csv"
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Image Data ID'][1:] in subject_list:
                scores_dict[row['Image Data ID'][1:]] = row[score]
    return scores_dict


def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(os.path.join("timeseries_subjects_id.txt"), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l not in ["age_category", "sex", "race"]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

    return graph


def get_static_affinity_adj(features, pd_dict):
    pd_affinity = create_affinity_graph_from_scores(list(pd_dict.keys()), pd_dict)
    dist_v = distance.pdist(features, metric='correlation')
    dist = distance.squareform(dist_v)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim

    return adj


class Local_GNN(torch.nn.Module):
    def __init__(self):
        super(Local_GNN, self).__init__()
        self._setup()

    def _setup(self):
        self.graph_convolution_1 = GCNConv(3,64)
        self.graph_convolution_2 = GCNConv(64,20)
        self.index_select_1 = SABP(20, ratio=0.9)
        self.graph_convolution_3 = GCNConv(20,20)

    def forward(self, data):
        node_features, edges, edge_attr = data[0], data[1], data[2]
        node_features_1 = F.relu(self.graph_convolution_1(node_features, edges, edge_attr))
        node_features_2 = F.relu(self.graph_convolution_2(node_features_1, edges, edge_attr))
        pool_features_2, pool_edge_index, pool_edge_attr, batch, perm, mi = self.index_select_1(node_features_2, edges, edge_attr, None)
        node_features_3 = F.relu(self.graph_convolution_3(pool_features_2, pool_edge_index, pool_edge_attr))
        cat_feature = pool_features_2 + node_features_3
        graph_embedding = cat_feature.view(1, -1)
        return graph_embedding, perm, mi


class SABP(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SABP,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.gcn = Conv(in_channels,in_channels)
        self.fc = torch.nn.Linear(in_channels*2, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        score_neg = x[torch.randperm(x.size(0))]
        embed = self.gcn(x,edge_index,edge_attr)
        joint = torch.cat((embed, x),dim = -1)
        margin = torch.cat((embed, score_neg),dim = -1)
        joint = self.fc(joint)
        margin = self.fc(margin)
        joint = F.normalize(joint, dim=1)
        margin = F.normalize(margin, dim=1)
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
        score = self.score_layer(x,edge_index,edge_attr).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm, mi_est


class Global_GNN(nn.Module):
    def __init__(self, args: ag.LG_GNNArgs):
        super().__init__()
        self.dropout_rate = args.dropout_rate
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        hidden_dim = 20
        self.convs.append(ChebConv(1620, hidden_dim, args.k_order, normalization='sym'))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(ChebConv(hidden_dim, hidden_dim, args.k_order, normalization='sym'))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(ChebConv(hidden_dim, hidden_dim, args.k_order, normalization='sym'))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(ChebConv(hidden_dim, hidden_dim, args.k_order, normalization='sym'))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.out_fc = nn.Linear(hidden_dim, args.class_num)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, edges, edge_weight):
        x = features
        layer_out = []
        x = self.convs[0](x, edges)
        x = self.bns[0](x)
        x = F.relu(x, inplace=True)
        layer_out.append(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.convs[1](x, edges)
        x = self.bns[1](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.convs[2](x, edges)
        x = self.bns[2](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.convs[3](x, edges)
        x = self.bns[3](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[2]
        layer_out.append(x)
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        emb = sum(layer_out)
        x = self.out_fc(emb)
        return x


class LG_GNNParams(nw.NetworkParams):
    def __init__(self, non_img, phonetic_score, args: ag.LG_GNNArgs):
        super().__init__(args)
        self.args = args
        self.non_img = non_img
        self.phonetic_score = phonetic_score


class LG_GNN(nw.NetworkBase):
    """
    @article{zhang2022classification,
      title={Classification of brain disorders in rs-fMRI via local-to-global graph neural networks},
      author={Zhang, Hao and Song, Ran and Wang, Liping and Zhang, Lin and Wang, Dawei and Wang, Cong and Zhang, Wei},
      journal={IEEE Transactions on Medical Imaging},
      volume={42},
      number={2},
      pages={444--455},
      year={2022},
      publisher={IEEE}
    }
    """
    def __init__(self, params: LG_GNNParams):
        super().__init__(params)

        self.non_img = params.non_img
        self.phonetic_score = params.phonetic_score

        self.mi_loss = torch.tensor([0.0]).to(ag.Arguments.device)
        input_dim = len(self.phonetic_score)
        self.edge = EDGE(input_dim, params.args.dropout_rate)
        self.graph_level_model = Local_GNN()
        self.hierarchical_model = Global_GNN(params.args)

    def forward(self, all_data):
        dl = Data_Loader()
        embeddings = []
        perms = []
        MI = 0
        for data in all_data:
            embedding, perm, mi = self.graph_level_model.forward(data)
            MI = MI + mi
            perm = perm.cpu().numpy()
            perms.append(perm)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))
        self.mi_loss = MI/len(all_data)
        edge_index, edge_input = dl.get_inputs(self.non_img, embeddings, self.phonetic_score)
        edge_input = (edge_input - edge_input.mean(axis=0)) / edge_input.std(axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(ag.Arguments.device)
        edge_input = torch.tensor(edge_input, dtype=torch.float32).to(ag.Arguments.device)
        edge_weight = torch.squeeze(self.edge.forward(edge_input))
        predictions = self.hierarchical_model.forward(embeddings, edge_index, edge_weight)
        return predictions[self.current_indices]

    def loss(self, scores, labels) -> torch.Tensor:
        final_loss = F.cross_entropy(scores, labels) - 0.1 * self.mi_loss
        return final_loss
