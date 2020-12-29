import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os,json,pickle,collections,time
import multiprocessing
from random import randrange
from numpy.random import choice

def get_matrix(triples,entity,rel):
    ent_size = max(entity)+1
    rel_size = (max(rel) + 1)
    #print(ent_size,rel_size)
    adj_matrix = sp.lil_matrix((ent_size,ent_size))
    adj_features = sp.lil_matrix((ent_size,ent_size))
    rel_in = np.zeros((ent_size,rel_size))
    rel_out = np.zeros((ent_size,rel_size))

    for i in range(max(entity)+1):
        adj_features[i,i] = 1

    for h,r,t in triples:
        adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
        adj_features[h,t] = 1; adj_features[t,h] = 1;
        rel_out[h][r] += 1; rel_in[t][r] += 1

    rel_features = np.concatenate([rel_in,rel_out],axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))    
    return adj_matrix,adj_features,rel_features

def get_hits(vec, test_pair, top_k=(1, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    #print('For each left:')
    record=[]#['L']
    for i in range(len(top_lr)):
        score = top_lr[i] / len(test_pair) * 100
        #print('Hits@%d: %.2f%%' % (top_k[i], score))
        record.append(score)
    
    MRR = MRR_lr / Lvec.shape[0]
    #print('MRR: %.3f' % MRR)  
    record.append(MRR)

    #record.append('R')
    #print('For each right:')
    for i in range(len(top_rl)):
        score = top_rl[i] / len(test_pair) * 100
        #print('Hits@%d: %.2f%%' % (top_k[i], score))
        record.append(score)
        
    MRR = MRR_rl / Lvec.shape[0]
    #print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))
    record.append(MRR)    
    return record

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append([head,r+1,tail])
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair
    
def load_anchor(lang, train_ratio = 0.3,seed=0):
    alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    np.random.seed(seed)
    np.random.shuffle(alignment_pair)
    train_pair,test_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)],\
                           alignment_pair[int(len(alignment_pair)*train_ratio):]
    return np.array(train_pair),np.array(test_pair)

def extension_attr(ent_p,train_pair):
    for i,j in train_pair:
        ent_p[i] = ent_p[i] | ent_p[j]
        ent_p[j] = ent_p[i]
    return ent_p

def load_data(lang,p):
    print(time.ctime(),'\tLoading data...')
    entity1,rel1,triples1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2')

    adj_matrix,adj_features,rel_features = \
        get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))
    ent_prop_vec = ent2prop_ids(entity1.union(entity2),folder=lang,p_=p)

    return adj_matrix,adj_features,rel_features,ent_prop_vec,entity1,entity2

def read_lines(file_path):
    if file_path is None:
        return []
    file = open(file_path, 'r', encoding='utf8')
    return file.readlines()

def read_attrs_range(file_path):
    dic = dict()
    lines = read_lines(file_path)
    for line in lines:
        line = line.strip()
        params = line.split('\t')
        assert len(params) == 2
        dic[params[0]] = int(params[1])
    return dic

def merge_dicts(dict1, dict2):
    for k in dict1.keys():
        vs = dict1.get(k)
        dict2[k] = dict2.get(k, set()) | vs
    return dict2

def read_ents_by_order(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    uri_list = list()
    ids_uris_dict = dict()
    uris_ids_dict = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        uri_list.append(params[1])
        ids_uris_dict[int(params[0])] = params[1]
        uris_ids_dict[params[1]] = int(params[0])
    return uri_list, ids_uris_dict, uris_ids_dict

def read_attrs(attrs_file):

    attrs_dic = dict()
    with open(attrs_file, 'r', encoding='utf8') as file:
        for line in file:
            params = line.strip().strip('\n').split('\t')
            if len(params) >= 2:
                attrs_dic[params[0]] = set(params[1:])
            else:
                print(line)
    return attrs_dic

def get_common(props_list, props_set):
    lprops = len(props_set)
    #print("total props:", lprops)
    #print("total prop frequency:", len(props_list))
    most_frequent_props = collections.Counter(props_list).most_common(lprops)
    common_props_ids,prop_id2freq = dict(),dict()
    for prop, freq in most_frequent_props:
        if freq >= 2 and prop not in common_props_ids:
            idx = len(common_props_ids)
            common_props_ids[prop] = idx
            prop_id2freq[idx] = freq
    return common_props_ids,prop_id2freq,most_frequent_props[0][1]

def ent2prop_ids(entity,folder='../data/dbp15k/zh_en/',p_=0.01):
    ent_size = max(entity)+1
    attrs1 = read_attrs(folder + 'training_attrs_1')
    len_ent1_a = len(attrs1.keys())
    attrs2 = read_attrs(folder + 'training_attrs_2')
    len_ent2_a = len(attrs2.keys())
    attrs_all = merge_dicts(attrs1, attrs2)

    props_set = set()
    props_list = []

    for uri,props in attrs_all.items():
        props_list.extend(list(props))
        props_set |= props

    prop_ids,prop_id2freq,max_count = get_common(props_list, props_set)
    prop_size = len(prop_ids)
    
    id2prop = dict(zip(prop_ids.values(),prop_ids.keys()))
    range_dict = read_attrs_range(folder+'all_attrs_range')
    en_range_dict = read_attrs_range(folder+'en_all_attrs_range')
    
    c=0
    range_vec = list()
    for i in range(prop_size):
        assert i in id2prop
        attr_uri = id2prop[i]
        if attr_uri in range_dict:
            range_vec.append(range_dict.get(attr_uri))
        elif attr_uri in en_range_dict:
            range_vec.append(en_range_dict.get(attr_uri))
        else:
            range_vec.append(5)
            c+=1
    
    uri2prop_idx = dict()
    for ent,props in attrs_all.items():
        prop_indexs = set()
        for p in props:
            if p in prop_ids:
                prop_indexs.add(prop_ids.get(p))
        if len(prop_indexs) < 1:
            continue
        prop_indexs = list(prop_indexs)
        uri2prop_idx[ent] = prop_indexs
        
    ents1, id2uri1, uri2id1 = read_ents_by_order(folder + 'ent_ids_1')
    ents2, id2uri2, uri2id2 = read_ents_by_order(folder + 'ent_ids_2')
    uri2id1.update(uri2id2)
    
    ent2prop_idx = dict()
    for uri,ids in uri2prop_idx.items():
        ent2prop_idx[uri2id1[uri]] = ids
        
    id2f = dict()
    for ent,props in uri2prop_idx.items():
        if ent in ents1:
            l = len_ent1_a
        else:
            l = len_ent2_a
        for p in props:
            id2f[p] = prop_id2freq[p]/l
            

    f_id = [(v,k) for k,v in id2f.items()]
    f_id.sort()
    id2pos = {f_id[0][1]:0}
    i=0
    while i < len(f_id)-1:
        j=1
        while j < len(f_id)-i:
            if (f_id[i+j][0]-f_id[i][0])/f_id[i][0] > p_:
                id2pos[f_id[i+j][1]] = id2pos[f_id[i][1]]+1
                break
            else:
                id2pos[f_id[i+j][1]] = id2pos[f_id[i][1]]
                j+=1
        i+=j
        
    min_freq = 1/max(len_ent1_a,len_ent2_a)
    max_freq = max_count/min(len_ent1_a,len_ent2_a)
    l_prop_vec = (max(id2pos.values())+1) * 4
    vec = sp.lil_matrix((ent_size,l_prop_vec),dtype=np.int32)
    for ent,props in ent2prop_idx.items():
        for p in props:
            idx = id2pos[p] * 4 + range_vec[p]
            vec[ent,idx] = 1
    
    #print(vec.shape)#,vec[0])
    props_count = vec.sum(axis=0)
    props_count = props_count.reshape(-1).tolist()[0]
    props_count = [idx for idx,value in enumerate(props_count) if value > 1]
    vec = vec[:,props_count] 
    
    #print(vec.shape)#,vec[0])
    return vec
    