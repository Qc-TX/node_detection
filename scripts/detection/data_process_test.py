import json
import os
import os.path as osp
import argparse
import torch
import time
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset

null = ''
# groundtruth_uuid = '../train/grountruth/groundtruth_uuid.txt'  # r
# groundtruth_nodeId = 'groundtruth_nodeId.txt'  # w
id_to_uuid = 'id_to_uuid.txt'  # w
feature_path = '../../models/feature.txt'
label_path = '../../models/label.txt'


def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


class TestDatasetA(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDatasetA, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def MyDatasetA(path, model):
    graphId = model
    feature_num = 0
    label_num = 0
    f_feature = open(feature_path, 'r')
    feature_map = {}
    for i in f_feature:
        temp = i.strip('\n').split('\t')
        feature_map[temp[0]] = int(temp[1])
        feature_num += 1
    f_feature.close()

    f_label = open(label_path, 'r')
    label_map = {}
    for i in f_label:
        temp = i.strip('\n').split('\t')
        label_map[temp[0]] = int(temp[1])
        label_num += 1
    f_label.close()

    # f_gt = open(groundtruth_uuid, 'r')
    # ground_truth = {}
    # for line in f_gt:
    #     ground_truth[line.strip('\n')] = 1
    # f_gt.close()

    node_cnt = 0
    nodeType_cnt = 0
    edgeType_cnt = 0
    provenance = []
    nodeType_map = {}
    edgeType_map = {}
    edge_s = []
    edge_e = []
    adj = {}
    adj2 = {}
    data_thre = 1000000
    # fw = open(groundtruth_nodeId, 'w')
    fw2 = open(id_to_uuid, 'w')
    nodeId_map = {}
    cnt = 0
    nodeA = []
    for i in range(1):
        now_path = path
        show(now_path)
        f = open(now_path, 'r')
        for line in f:
            try:
                dict_line = eval(line.strip('\n'))
                info = {}
                test_srclabel = str(list(dict_line['srcmsg'].keys())[0])  # temp[1]
                test_srcuuid = str(list(dict_line['srcmsg'].values())).strip("['").strip("']")  # temp[0]
                test_dstlabel = str(list(dict_line['dstmsg'].keys())[0])  # temp[3]
                test_dstuuid = str(list(dict_line['dstmsg'].values())).strip("['").strip("']")  # temp[2]
                test_feature = dict_line['edge_type']  # temp[4]
                test_time = dict_line['time']  # temp[5]
                info['src_label'] = test_srclabel
                info['src_uuid'] = test_srcuuid
                info['dst_label'] = test_dstlabel
                info['dst_uuid'] = test_dstuuid
                info['feature'] = test_feature
                info['time'] = test_time

                temp = [info['src_uuid'], info['src_label'], info['dst_uuid'], info['dst_label'], info['feature'],
                        info['time']]

                cnt += 1
                if not (temp[1] in label_map.keys()): continue
                if not (temp[3] in label_map.keys()): continue
                if not (temp[4] in feature_map.keys()): continue

                if not (temp[0] in nodeId_map.keys()):
                    nodeId_map[temp[0]] = node_cnt
                    fw2.write(str(str(node_cnt) + '\t' + temp[0] + '\n').strip("'"))

                    # if temp[0] in ground_truth.keys():
                    #     fw.write(str(nodeId_map[temp[0]]) + ' ' + temp[1] + ' ' + temp[0] + '\n')
                    #     nodeA.append(node_cnt)
                    node_cnt += 1

                temp[0] = nodeId_map[temp[0]]

                if not (temp[2] in nodeId_map.keys()):
                    nodeId_map[temp[2]] = node_cnt
                    fw2.write(str(str(node_cnt) + '\t' + temp[2] + '\n').strip("'"))

                    # if temp[2] in ground_truth.keys():
                    #     fw.write(str(nodeId_map[temp[2]]) + ' ' + temp[3] + ' ' + temp[2] + '\n')
                    #     nodeA.append(node_cnt)
                    node_cnt += 1
                temp[2] = nodeId_map[temp[2]]
                temp[1] = label_map[temp[1]]
                temp[3] = label_map[temp[3]]
                temp[4] = feature_map[temp[4]]
                edge_s.append(temp[0])
                edge_e.append(temp[2])
                if temp[2] in adj.keys():
                    adj[temp[2]].append(temp[0])
                else:
                    adj[temp[2]] = [temp[0]]
                if temp[0] in adj2.keys():
                    adj2[temp[0]].append(temp[2])
                else:
                    adj2[temp[0]] = [temp[2]]
                provenance.append(temp)
            except(TypeError):
                continue
        f.close()
    # fw.close()
    fw2.close()

    uuid_list = []
    uuid_to_time = {}
    cnt = 0

    f = open(id_to_uuid, 'r')
    for line in f:
        uuid_list.append(line.strip('\n').split('\t')[1])
        cnt += 1
    f.close()
    os.system('rm ' + id_to_uuid)

    ft = open(path, 'r')
    f = open(id_to_uuid, 'w')
    for line in ft:
        try:
            dict_line = eval(line.strip('\n'))
            info = {}
            info['src_label'] = str(list(dict_line['srcmsg'].keys())[0])  # temp[1]
            info['src_uuid'] = str(list(dict_line['srcmsg'].values())).strip("['").strip("']")  # temp[0]
            info['dst_label'] = str(list(dict_line['dstmsg'].keys())[0])  # temp[3]
            info['dst_uuid'] = str(list(dict_line['dstmsg'].values())).strip("['").strip("']")  # temp[2]
            info['feature'] = dict_line['edge_type']  # temp[4]
            info['time'] = dict_line['time']  # temp[5]

            temp = [info['src_uuid'], info['src_label'], info['dst_uuid'], info['dst_label'], info['feature'],
                    info['time']]

            uuid1 = temp[0]
            uuid2 = temp[2]
            uuid_to_time[uuid1] = temp[5]
            uuid_to_time[uuid2] = temp[5]
        except(TypeError):
            continue
    for i in range(cnt):
        try:
            f.write(str(i) + '\t' + str(uuid_list[i]) + '\t' + str(uuid_to_time[uuid_list[i]]) + '\n')
        except(KeyError):
            f.write(str(i) + '\t' + str(uuid_list[i]) + '\t' + "unknown time" + '\n')
            continue
    f.close()
    ft.close()

    x_list = []
    y_list = []
    train_mask = []
    test_mask = []
    for i in range(node_cnt):
        temp_list = []
        for j in range(feature_num * 2):
            temp_list.append(0)
        x_list.append(temp_list)
        y_list.append(0)
        train_mask.append(True)
        test_mask.append(True)

    for temp in provenance:
        srcId = temp[0]
        srcType = temp[1]
        dstId = temp[2]
        dstType = temp[3]
        edge = temp[4]
        x_list[srcId][edge] += 1
        y_list[srcId] = srcType
        x_list[dstId][edge + feature_num] += 1
        y_list[dstId] = dstType

    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data1 = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)
    feature_num *= 2
    neibor = set()
    _neibor = {}
    for i in nodeA:
        neibor.add(i)
        if not i in _neibor.keys():
            templ = []
            _neibor[i] = templ
        if not i in _neibor[i]:
            _neibor[i].append(i)
        if i in adj.keys():
            for j in adj[i]:
                neibor.add(j)
                if not j in _neibor.keys():
                    templ = []
                    _neibor[j] = templ
                if not i in _neibor[j]:
                    _neibor[j].append(i)
                if not j in adj.keys(): continue
                for k in adj[j]:
                    neibor.add(k)
                    if not k in _neibor.keys():
                        templ = []
                        _neibor[k] = templ
                    if not i in _neibor[k]:
                        _neibor[k].append(i)
        if i in adj2.keys():
            for j in adj2[i]:
                neibor.add(j)
                if not j in adj2.keys(): continue
                for k in adj2[j]:
                    neibor.add(k)
    _nodeA = []
    for i in neibor:
        _nodeA.append(i)

    return [data1], feature_num, label_num, adj, adj2, nodeA, _nodeA, _neibor
