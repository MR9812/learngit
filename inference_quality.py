import os
import json
import numpy as np
from itertools import product, permutations
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
#from dgl.dataloading.neighbor import MultiLayerNeighborSampler
#from dgl.dataloading.pytorch import NodeDataLoader
from dgl.dataloading import MultiLayerNeighborSampler
from dgl.dataloading import NodeDataLoader
from sklearn.metrics import roc_auc_score
import math
import torch.nn as nn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import datetime


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype][:sub_graph.num_dst_nodes()]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                # t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                t = G.dstdata['t'][ntype].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype][:G.num_dst_nodes(ntype)] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class HGT_Origin(nn.Module):
    def __init__(self, node_dict, edge_dict, n_inp_dict, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT_Origin, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp_dict = n_inp_dict
        self.gcs = nn.ModuleList()
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()

        for k,v in sorted(node_dict.items(), key=lambda x: x[1]):
            self.adapt_ws.append(nn.Linear(n_inp_dict[k], n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))

        self.out = nn.Linear(n_hid, n_out)

    def forward(self, blocks, x, out_key):
        h = {}
        for ntype in blocks[0].srctypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](x[ntype]))
            h[ntype] = F.dropout(h[ntype], 0.4, training=self.training)

        for l, (layer, block) in enumerate(zip(self.gcs, blocks)):
            h = layer(block, h)
            
        return torch.sigmoid(self.out(h[out_key])).view([-1])
    

def load_subtensor(node_feats, input_nodes, device='cpu'):
    """
    Copys features and labels of a set of nodes onto device.
    """
    batch_inputs = {}
    for k,v in node_feats.items():
        batch_inputs[k] = node_feats[k][input_nodes[k]].to(device)
    return batch_inputs



#加载数据
class Dataset:
    def __init__(self, date):
        self.date = date
        print("Begin to Process {} features...".format(date))

        self.hdfs_path = 'hdfs://haruna/home/byte_ecom_govern/miaorui/quality_ccr/data/v6/predict/{}'.format(date)
        self.local_path = './data/{}_v2_test_feature_data.csv'.format(date)

        self.features = []

        self.columns = ['shop_id',]

        self.shop_ids = []
    
    def read_data(self, date, base_path, header_columns, item_columns):
        dfs = []
        date_path = base_path.format(date)
        df = pd.read_csv(date_path, delimiter='\t', header=None)
        df.columns = header_columns
        for item in item_columns:
            df['date'] = str(date)
            df[item] = df[item].astype(str) + '_' + df['date']
        dfs.append(df)
        df = pd.concat(dfs)
        return df
    
    def standardization(self, data, mu, sigma):
        #mu = np.mean(data, axis=0)
        #sigma = np.std(data, axis=0)
        return (data - mu) / sigma, mu, sigma

    def get_item_std_features(self, item_type, item_nodes, path):
        features = pd.read_csv(path)
        features = features.fillna(-1)
        clean_shop_features = features.drop(['%s_id' % item_type], axis=1)

        f = features['%s_id' % item_type].to_list()
        f_item_id_mapping = {item:idx for idx, item in enumerate(f)}

        std_features = standardization(clean_shop_features.values)
        std_features = np.nan_to_num(std_features)

        f_orders = []
        for node in item_nodes:
            f_orders.append(f_item_id_mapping[node])
        f_orders = np.asarray(f_orders)

        np_vector = std_features[f_orders]

        return np_vector

    def download_daily_data(self):
        prepare_data_types = [
            'shop_nums',
            'author_features',
            'shop_features',
            'author_shop_edges',
            'shop_prec_edges']

        train_path = {}
        hdfs_base_path = 'hdfs://haruna/home/byte_ecom_govern/miaorui/quality_ccr/data/v6/predict/{date}'
        for data_type in prepare_data_types:
            train_path[data_type] = os.path.join(hdfs_base_path, data_type)
            
        for data_type, hdfs_path in train_path.items():
            hdfs_data_path = hdfs_path.format(date=self.date)
            local_save_path = './data/shipin/v6/predict_{data_type}_{date}.csv'.format(data_type=data_type, date=self.date)
            download_cmd = 'hdfs dfs -getmerge {hdfs_path} {save_path}'.format(hdfs_path=hdfs_data_path, save_path=local_save_path)
            os.system(download_cmd)
            
        node_dict = {}
        node_dict['author'] = 0
        node_dict['shop'] = 1
        
        #shop
        base_path = './data/shipin/v6/predict_shop_nums_{}.csv'
        header_columns = ['shop_id', 'num_14d', 'num_1d', 'main']
        item_columns = ['shop_id']
        shop_seeds = self.read_data(self.date, base_path, header_columns, item_columns)
        #高准边
        base_path = './data/shipin/v6/predict_shop_prec_edges_{}.csv'
        header_columns = ['group_id', 'shop_id', 'parent']
        item_columns = ['group_id', 'shop_id', 'parent']
        shop_prec_edges = self.read_data(self.date, base_path, header_columns, item_columns)
        #商家达人边
        base_path = './data/shipin/v6/predict_author_shop_edges_{}.csv'
        header_columns = ['author_id', 'shop_id']
        item_columns = ['author_id', 'shop_id']
        shop_author_edges = self.read_data(self.date, base_path, header_columns, item_columns)
        #点
        shop_nodes = shop_seeds['shop_id'].drop_duplicates().tolist()
        #import ipdb;ipdb.set_trace()
        print("Loading {} shop nodes...".format(len(shop_nodes)))
        author_nodes = shop_author_edges['author_id'].drop_duplicates().tolist()
        print("Loading {} author nodes...".format(len(author_nodes)))
        
        shop_id_mapping = {shop:idx for idx, shop in enumerate(shop_nodes)}
        id_shop_mapping = {idx:shop for idx, shop in enumerate(shop_nodes)}
        author_id_mapping = {author:idx for idx, author in enumerate(author_nodes)}
        id_author_mapping = {idx:author for idx, author in enumerate(author_nodes)}
        
        num_shops = len(shop_id_mapping)
        num_authors = len(author_id_mapping)
        num_nodes_dict = {}
        num_nodes_dict['shop'] = num_shops
        num_nodes_dict['author'] = num_authors
        
        graph_data = {}
        # 高准团伙边
        s_type = 'shop'
        t_type = 'shop'
        r_type = 'prec'
        s_lack, t_lack = 0, 0
        graph_data[(s_type, r_type, t_type)] = {}
        graph_data[(t_type, 'rev_' + r_type, s_type)] = {}
        u, v = [], []
        for s_id, t_id in shop_prec_edges[['shop_id', 'parent']].values.tolist():
            if str(t_id)[0] == '-':
                continue
            if s_id not in shop_id_mapping:
                s_lack += 1 
                continue
            if t_id not in shop_id_mapping:
                t_lack += 1 
                continue
            s_idx = shop_id_mapping[s_id]
            t_idx = shop_id_mapping[t_id]
            u.append(s_idx)
            v.append(t_idx)
        graph_data[(s_type, r_type, t_type)] = (u, v)
        graph_data[(t_type, 'rev_' + r_type, s_type)] = (v, u)
        
        # 达人商家边
        s_type = 'shop'
        t_type = 'author'
        r_type = 'marketing'

        graph_data[(s_type, r_type, t_type)] = {}
        graph_data[(t_type, 'rev_' + r_type, s_type)] = {}
        u, v = [], []

        for t_id, s_id in shop_author_edges[['author_id', 'shop_id']].values.tolist():

            s_idx = shop_id_mapping[s_id]
            t_idx = author_id_mapping[t_id]

            u.append(s_idx)
            v.append(t_idx)

        graph_data[(s_type, r_type, t_type)] = (u, v)
        graph_data[(t_type, 'rev_' + r_type, s_type)] = (v, u)
        
        G = dgl.heterograph(graph_data)
        
        shop_1d_nums = []
        shop_14d_nums = []
        shop_main = []
        shop_list = []
        shop_seeds['num_14d'] = shop_seeds['num_14d'].replace('\\N', np.nan).astype('float').fillna(0)
        shop_seeds['num_1d'] = shop_seeds['num_1d'].replace('\\N', np.nan).astype('float').fillna(0)
        for shop, num14d, num1d, main in shop_seeds[['shop_id', 'num_14d', 'num_1d', 'main']].values:
            shop_1d_nums.append(int(num1d))
            shop_14d_nums.append(int(num14d))
            shop_main.append(int(main))
        #import ipdb;ipdb.set_trace()
        # 需要改 label
        #shop_labels = [shop2label[node] if fnode in shop2label else 0 for node in list(shop_nodes)]
        #labels = torch.tensor(shop_labels).long()
        
        node_dict, edge_dict = {}, {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in G.etypes:
            edge_dict[etype] = len(edge_dict)
            G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
            
        shop_feature_names = ['shop_id', 'aftersale_fake_ccr_1d','aftersale_fake_ccr_7d','aftersale_fake_ccr_90d','aftersale_fake_ccr_180d','aftersale_fake_ccr_td','im_message_fake_ccr_1d','im_message_fake_ccr_7d','im_message_fake_ccr_60d','im_message_fake_ccr_90d','im_message_fake_ccr_180d','im_message_fake_ccr_td','comment_fake_ccr_7d','comment_fake_ccr_60d','comment_fake_ccr_90d','comment_fake_ccr_180d','comment_fake_ccr_td','aftersale_ccr_7d','aftersale_ccr_14d','aftersale_ccr_30d','aftersale_ccr_60d','aftersale_ccr_90d','aftersale_ccr_180d','aftersale_ccr_td','im_message_ccr_1d','im_message_ccr_7d','im_message_ccr_14d','im_message_ccr_60d','im_message_ccr_90d','im_message_ccr_180d','im_message_ccr_td','comment_ccr_1d','comment_ccr_7d','comment_ccr_14d','comment_ccr_30d','comment_ccr_60d','comment_ccr_90d','comment_ccr_180d','comment_ccr_td','fake_ccr_7d','fake_ccr_30d','fake_ccr_60d','fake_ccr_90d','fake_ccr_180d','fake_ccr_td','ccr_1d','ccr_7d','ccr_14d','ccr_30d','ccr_60d','ccr_90d','ccr_180d','ccr_td','aftersale_fake_ccr_rate_1d','aftersale_fake_ccr_rate_7d','aftersale_fake_ccr_rate_14d','aftersale_fake_ccr_rate_30d','aftersale_fake_ccr_rate_60d','aftersale_fake_ccr_rate_90d','aftersale_fake_ccr_rate_180d','aftersale_fake_ccr_rate_td','im_message_fake_ccr_rate_7d','im_message_fake_ccr_rate_14d','im_message_fake_ccr_rate_30d','im_message_fake_ccr_rate_60d','im_message_fake_ccr_rate_90d','im_message_fake_ccr_rate_180d','im_message_fake_ccr_rate_td','comment_fake_ccr_rate_7d','comment_fake_ccr_rate_14d','comment_fake_ccr_rate_30d','comment_fake_ccr_rate_60d','comment_fake_ccr_rate_90d','comment_fake_ccr_rate_180d','comment_fake_ccr_rate_td','fake_ccr_rate_1d','fake_ccr_rate_7d','fake_ccr_rate_14d','fake_ccr_rate_30d','fake_ccr_rate_60d','fake_ccr_rate_90d','fake_ccr_rate_180d','fake_ccr_rate_td','shop_settle_type','final_rank','final_score','goods_score','service_score','logistics_score','goods_rank','service_rank','logistics_rank','create_product_cnt_td','create_product_cnt_30d','create_product_cnt_7d','create_product_cnt_1d','available_sale_product_cnt_td','total_alliance_promoted_product_cnt','total_alliance_promote_product_cnt','machine_audit_reject_product_cnt_td','good_eval_rate_td','good_eval_rate_30d','good_eval_rate_7d','good_eval_rate_1d','bad_eval_rate_td','bad_eval_rate_30d','bad_eval_rate_7d','bad_eval_rate_1d','product_good_eval_rate_td','product_good_eval_rate_30d','product_good_eval_rate_7d','product_good_eval_rate_1d','shop_service_good_eval_rate_td','shop_service_good_eval_rate_30d','shop_service_good_eval_rate_7d','shop_service_good_eval_rate_1d','logistic_good_eval_rate_td','logistic_good_eval_rate_30d','logistic_good_eval_rate_7d','logistic_good_eval_rate_1d','product_bad_eval_rate_td','product_bad_eval_rate_30d','product_bad_eval_rate_7d','product_bad_eval_rate_1d','shop_service_bad_eval_rate_td','shop_service_bad_eval_rate_30d','shop_service_bad_eval_rate_7d','shop_service_bad_eval_rate_1d','logistic_bad_eval_rate_td','logistic_bad_eval_rate_30d','logistic_bad_eval_rate_7d','logistic_bad_eval_rate_1d','cancel_order_rate_td','cancel_order_rate_30d','cancel_order_rate_7d','cancel_order_rate_1d','saling_refund_order_rate_td','saling_refund_order_rate_30d','saling_refund_order_rate_7d','saling_refund_order_rate_1d','saling_refund_sucess_order_rate_td','saling_refund_sucess_order_rate_30d','saling_refund_sucess_order_rate_7d','saling_refund_sucess_order_rate_1d','saled_refund_order_rate_td','saled_refund_order_rate_30d','saled_refund_order_rate_7d','saled_refund_order_rate_1d','quality_refund_rate_td','before_44d_14d_quality_refund_rate','before_21d_14d_quality_refund_rate','ontime_order_rate_td','ontime_order_rate_30d','ontime_order_rate_7d','ontime_order_rate_1d','fake_ship_order_rate_td','fake_ship_order_rate_30d','fake_ship_order_rate_7d','fake_ship_order_rate_1d','satisfaction_rate_td','satisfaction_rate_30d','satisfaction_rate_7d','satisfaction_rate_1d','complain_rate_td','complain_rate_30d','complain_rate_7d','complain_rate_1d','sale_refund_rate_7d']
        author_feature_names = ['author_id','fake_punish_cnt_1d','fake_punish_cnt_7d','fake_punish_cnt_14d','fake_punish_cnt_30d','fake_punish_cnt_60d','fake_punish_cnt_90d','fake_punish_cnt_180d','fake_punish_cnt_td','punish_cnt_1d','punish_cnt_7d','punish_cnt_14d','punish_cnt_30d','punish_cnt_60d','punish_cnt_90d','punish_cnt_180d','punish_cnt_td','fake_punish_rate_1d','fake_punish_rate_7d','fake_punish_rate_14d','fake_punish_rate_30d','fake_punish_rate_60d','fake_punish_rate_90d','fake_punish_rate_180d','fake_punish_rate_td','aftersale_fake_ccr_1d','aftersale_fake_ccr_7d','aftersale_fake_ccr_14d','aftersale_fake_ccr_30d','aftersale_fake_ccr_60d','aftersale_fake_ccr_90d','aftersale_fake_ccr_180d','aftersale_fake_ccr_td','im_message_fake_ccr_1d','im_message_fake_ccr_7d','im_message_fake_ccr_14d','im_message_fake_ccr_30d','im_message_fake_ccr_60d','im_message_fake_ccr_90d','im_message_fake_ccr_180d','im_message_fake_ccr_td','comment_fake_ccr_1d','comment_fake_ccr_7d','comment_fake_ccr_14d','comment_fake_ccr_30d','comment_fake_ccr_60d','comment_fake_ccr_90d','comment_fake_ccr_180d','comment_fake_ccr_td','aftersale_ccr_1d','aftersale_ccr_7d','aftersale_ccr_14d','aftersale_ccr_30d','aftersale_ccr_60d','aftersale_ccr_90d','aftersale_ccr_180d','aftersale_ccr_td','im_message_ccr_1d','im_message_ccr_7d','im_message_ccr_14d','im_message_ccr_30d','im_message_ccr_60d','im_message_ccr_90d','im_message_ccr_180d','im_message_ccr_td','comment_ccr_1d','comment_ccr_7d','comment_ccr_14d','comment_ccr_30d','comment_ccr_60d','comment_ccr_90d','comment_ccr_180d','comment_ccr_td','fake_ccr_1d','fake_ccr_7d','fake_ccr_14d','fake_ccr_30d','fake_ccr_60d','fake_ccr_90d','fake_ccr_180d','fake_ccr_td','ccr_1d','ccr_7d','ccr_14d','ccr_30d','ccr_60d','ccr_90d','ccr_180d','ccr_td','aftersale_fake_ccr_rate_1d','aftersale_fake_ccr_rate_7d','aftersale_fake_ccr_rate_14d','aftersale_fake_ccr_rate_30d','aftersale_fake_ccr_rate_60d','aftersale_fake_ccr_rate_90d','aftersale_fake_ccr_rate_180d','aftersale_fake_ccr_rate_td','im_message_fake_ccr_rate_1d','im_message_fake_ccr_rate_7d','im_message_fake_ccr_rate_14d','im_message_fake_ccr_rate_30d','im_message_fake_ccr_rate_60d','im_message_fake_ccr_rate_90d','im_message_fake_ccr_rate_180d','im_message_fake_ccr_rate_td','comment_fake_ccr_rate_1d','comment_fake_ccr_rate_7d','comment_fake_ccr_rate_14d','comment_fake_ccr_rate_30d','comment_fake_ccr_rate_60d','comment_fake_ccr_rate_90d','comment_fake_ccr_rate_180d','comment_fake_ccr_rate_td','fake_ccr_rate_1d','fake_ccr_rate_7d','fake_ccr_rate_14d','fake_ccr_rate_30d','fake_ccr_rate_60d','fake_ccr_rate_90d','fake_ccr_rate_180d','fake_ccr_rate_td']
        #import ipdb;ipdb.set_trace()
        npzfile = np.load('np_data_shipin_mu_sigma.npz')
        s_mu = npzfile['arr_0']
        s_sigma = npzfile['arr_1']
        a_mu = npzfile['arr_2']
        a_sigma = npzfile['arr_3']
        #shop feature
        base_path = './data/shipin/v6/predict_shop_features_{}.csv'
        header_columns = shop_feature_names
        item_columns = ['shop_id']
        shop_features = self.read_data(self.date, base_path, header_columns, item_columns)
        shop_features_ids = shop_features[['shop_id']]
        shop_features = shop_features.replace('\\N', np.nan).astype('float').fillna(-1)
        shop_features['shop_id'] = shop_features_ids
        clean_shop_features = shop_features.drop(['shop_id'], axis=1)
        f = shop_features['shop_id'].to_list()
        f_item_id_mapping = {item:idx for idx, item in enumerate(f)}
        std_features, s_mu, s_sigma = self.standardization(clean_shop_features.values, s_mu, s_sigma)
        std_features = np.nan_to_num(std_features)
        f_orders = []
        for node in shop_nodes:
            f_orders.append(f_item_id_mapping[node])
        f_orders = np.asarray(f_orders)
        shop_np_vector = std_features[f_orders]
        
        #author feature
        base_path = './data/shipin/v6/predict_author_features_{}.csv'
        header_columns = author_feature_names
        item_columns = ['author_id']
        shop_features = self.read_data(self.date, base_path, header_columns, item_columns)
        shop_features_ids = shop_features[['author_id']]
        shop_features = shop_features.replace('\\N', np.nan).astype('float').fillna(-1)
        shop_features['author_id'] = shop_features_ids
        clean_shop_features = shop_features.drop(['author_id'], axis=1)
        f = shop_features['author_id'].to_list()
        f_item_id_mapping = {item:idx for idx, item in enumerate(f)}
        std_features, a_mu, a_sigma = self.standardization(clean_shop_features.values, a_mu, a_sigma)
        std_features = np.nan_to_num(std_features)
        f_orders = []
        for node in author_nodes:
            f_orders.append(f_item_id_mapping[node])
        f_orders = np.asarray(f_orders)
        author_np_vector = std_features[f_orders]
        #import ipdb;ipdb.set_trace()
        norm_operator = {
            's_mu': s_mu,
            'a_mu': a_mu,
            's_sigma': s_sigma,
            'a_sigma': a_sigma
        }
        
        n_inp_dict, node_feature = {}, {}
        for item_type, item_features in zip(['shop', 'author'], [shop_np_vector, author_np_vector]):
            node_feature[item_type] = torch.tensor(item_features).float()
            n_inp_dict[item_type] = item_features.shape[1]
        
        n_inp_dict['shop'] = n_inp_dict['shop'] - 1
        n_inp_dict['author'] = n_inp_dict['author'] - 1
        node_feature['shop'] = node_feature['shop'][:,  :-1]
        node_feature['author'] = node_feature['author'][:,  :-1]
            
        save_data = {}
        save_data['G'] = G
        save_data['n_inp_dict'] = n_inp_dict
        save_data['node_dict']  = node_dict
        save_data['edge_dict']  = edge_dict
        save_data['shop_nodes'] = shop_nodes
        save_data['author_nodes'] = author_nodes
        save_data['node_feature'] = node_feature
        
        test_idx = {'shop': torch.tensor(np.arange(len(shop_nodes)))}
        for etype in G.etypes:
            G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

        test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        test_dataloader_out = NodeDataLoader(
            G, test_idx,
            test_sampler,
            batch_size=2048,
            shuffle=True, drop_last=False, num_workers=1)
        
        return test_dataloader_out, node_dict, edge_dict, n_inp_dict, node_feature, shop_nodes, shop_1d_nums, shop_14d_nums, shop_main

    
class PipelineUtils:
    def __init__(self, date):
        self.save_json_path = '{}_output_score.json'.format(date)
        self.upload_path = 'hdfs://haruna/home/byte_ecom_govern/miaorui/quality_ccr/data/v6/predict/res/{}'.format(date)

    def upload_hdfs(self, prob, seeds, shop_nodes, shop_1d_nums, shop_14d_nums, shop_main):
        shop_list = []
        for i in range(len(shop_nodes)):
            a,_,b = shop_nodes[i].partition('_')
            shop_list.append(int(a))
        #import ipdb;ipdb.set_trace()
        #(np.array(prob)>=0.8).tolist()
        preds_df = pd.DataFrame(prob, columns=['quality_score'])
        shop_nodes =  pd.DataFrame(np.asarray(shop_list)[seeds].tolist(), columns=['shop_id'])
        num1d_df = pd.DataFrame(np.asarray(shop_1d_nums)[seeds], columns=['num_1d'])
        num14d_df = pd.DataFrame(np.asarray(shop_14d_nums)[seeds], columns=['num_14d'])
        main_df = pd.DataFrame(np.asarray(shop_main)[seeds], columns=['main'])
        output_df = pd.concat([shop_nodes, preds_df, num1d_df, num14d_df, main_df], axis=1)

        output_values = output_df.values
        output_columns = output_df.columns

        with open(self.save_json_path, 'w') as json_file:
            for value in output_values:
                json_dict = dict(zip(output_columns, [int(value[0])] + [float(v) for v in value[1:]]))
                json_file.write(json.dumps(json_dict) + '\n')
        
        
        upload_path = self.upload_path
        json_path = self.save_json_path

        print("Begin to upload res. Upload path: {}".format(upload_path))
        rm_cmd = 'hdfs dfs -rm -r {}'.format(upload_path)
        create_cmd = 'hdfs dfs -mkdir {}'.format(upload_path)
        upload_cmd = 'hdfs dfs -put {} {}'.format(json_path, upload_path)

        os.system(rm_cmd)
        os.system(create_cmd)
        os.system(upload_cmd)

        flag_file = '_SUCCESS'
        os.system("echo '' > {}".format(flag_file))
        upload_cmd = 'hdfs dfs -put {} {}'.format(flag_file, upload_path)
        os.system(upload_cmd)

        print("UPLOAD SUCCESS...")

def main():
        #加载数据
        ISOT = '%Y%m%d'
        daily_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime(ISOT)
        #daily_date = '20220802'
        dataset = Dataset(date=daily_date)
        test_dataloader, node_dict, edge_dict, n_inp_dict, test_node_feature, test_shop_nodes, shop_1d_nums, shop_14d_nums, shop_main = dataset.download_daily_data()

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        #加载模型
        new_model = HGT_Origin(node_dict, edge_dict, n_inp_dict=n_inp_dict,
                    n_hid=128,
                    n_out=1,
                    n_layers=3,
                    n_heads=16,
                    use_norm = True).to(device)

        download_cmd = 'hdfs dfs -get {}{} {}'.format("hdfs://haruna/home/byte_ecom_govern/miaorui/quality_ccr/data/v6/", "best_quality_test.pt", "best_quality_test.pt")
        print("Begin to download model...")
        os.system(download_cmd)

        #载入模型    
        new_model.load_state_dict(torch.load("best_quality_test.pt", map_location=torch.device('cpu')))
        new_model.eval()
        
        #验证
        prob_new = []
        true_seeds_new = []
        print("training")
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            # forward
            batch_inputs = load_subtensor(test_node_feature, input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            # metric and loss
            val_batch_logits = new_model(blocks, batch_inputs, 'shop')
            #import ipdb;ipdb.set_trace()
            prob_new.extend(list(val_batch_logits.detach().cpu().numpy()))
            true_seeds_new.extend(list(seeds['shop'].cpu().numpy()))
            
        #import ipdb;ipdb.set_trace()
        putils = PipelineUtils(date=daily_date)
        putils.upload_hdfs(prob_new, true_seeds_new, test_shop_nodes, shop_1d_nums, shop_14d_nums, shop_main)
        print(len(prob_new))

if __name__ == '__main__':
    main()