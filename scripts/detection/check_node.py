import os.path as osp
import os
import argparse
from operator import itemgetter
import random

from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
from data_process_test import *
import difflib
from mysql_con import *

apt_type_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def get_apt_type(num_list: list) -> int:
    return num_list.index(max(num_list))


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


# def load_IOC():
#     f_IOC = open('IOC.txt', 'r')
#     IOC = []
#     for IOC_line in f_IOC:
#         if IOC_line.strip(' ').strip('\n') != '':
#             IOC.append(IOC_line.strip('\n'))
#         else:
#             continue
#     return IOC
#
#
# def calculate_IOC_sim(name: str, IOC_list: list, t:float):
#     flag = 0
#     for ioc in IOC_list:
#         if string_similar(name, ioc) > t:
#             flag = 1
#             return flag
#         else:
#             flag = 0
#     return flag

def load_IOC():
    f_IOC = open('classified_IOC.txt', 'r')
    IOC = {}
    for IOC_line in f_IOC:
        IOC[IOC_line.strip('\n').split('\t')[0]] = int(IOC_line.strip('\n').split('\t')[1])
    f_IOC.close()
    return IOC


def calculate_IOC_sim(name: str, IOC_list: dict, t: float):
    flag = 0
    # IOC_list = random_dic(IOC_list)
    for ioc in IOC_list.keys():
        if string_similar(name, ioc) > t:
            flag = 1
            apt_type_cnt[IOC_list[ioc]] += 1
            return flag
        else:
            flag = 0
    return flag


def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='../../my_data/rtf/final-rtf.json')
parser.add_argument('--threatID', type=str, default='f1cabd0d-a011-4975-a8bc-5a0a4aaebbc0')
parser.add_argument('--threatName', type=str, default='')
parser.add_argument('--model', type=str, default='0')
args = parser.parse_args()
threat_id = args.threatID
threat_name = args.threatName
log_path = args.log

b_size = 5000
nodeA = []
path = log_path
show('Start testing graph.')
data1, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neibor = MyDatasetA(path, args.model)

dataset = TestDatasetA(data1)
data = dataset[0]

loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False, add_self_loops=True)

final_result = {}
for i in range(len(data.test_mask)):
    final_result[i] = [-1, -1]


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32, normalize=False, concat=concat)
        self.conv2 = SAGEConv(32, out_channels, normalize=False, concat=concat)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size)
        return F.log_softmax(x, dim=1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

Net = SAGENet

model = Net(feature_num, label_num).to(device)

cnt6_1 = 0
cnt1_6 = 0
thre = 1.5


def test(mask):
    global cnt1_6
    global cnt6_1
    model.eval()

    correct = 0
    total_loss = 0
    num_flag = 0
    for data_flow in loader(mask):
        out = model(data.x.to(device), data_flow.to(device))
        pred = out.max(1)[1]
        pro = F.softmax(out, dim=1)
        pro1 = pro.max(1)
        for i in range(len(data_flow.n_id)):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)
        for i in range(len(data_flow.n_id)):
            if pro1[0][i] / pro2[0][i] < thre:
                pred[i] = 100

        # 维护一个final_result列表，保存模型检测结果
        # final_result[i]=[-1,-1]-->赋值
        # final_result[i]!=[-1,-1] 且 预测与实际不符 --> 后面模型预测成功可以赋值
        # final_result[i]!=[-1,-1] 且 预测与实际相符 --> 不赋值
        for i in range(len(data_flow.n_id)):
            if final_result[num_flag + i][0] == -1:
                final_result[num_flag + i] = [data.y[data_flow.n_id[i]].item(), pred[i].item()]
            elif data.test_mask[i]:
                final_result[num_flag + i] = [data.y[data_flow.n_id[i]].item(), pred[i].item()]
            # elif final_result[num_flag + i][0] != final_result[num_flag + i][1]:
            #     final_result[num_flag + i] = [data.y[data_flow.n_id[i]].item(), pred[i].item()]
            else:
                continue

        # 从0到data_flow.n_id(5000)，逐个比较
        # pred[i]是预测值，data.y[data_flow.n_id[i]]是标签
        # print("data_flow.n_id" + str(len(data_flow.n_id)))
        # 实际都是negative
        for i in range(len(data_flow.n_id)):
            # 实际negative预测值为positive
            if (data.y[data_flow.n_id[i]] != pred[i]):
                fp.append(int(data_flow.n_id[i]))
            # 实际negative命中预测值为negative
            else:
                tn.append(int(data_flow.n_id[i]))
        # print(pred)
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
        num_flag += len(data_flow.n_id)
        # print("correct:" + str(correct))

        loss_func = torch.nn.CrossEntropyLoss()
        total_loss = loss_func(out, data.y[data_flow.n_id].to(device))

    # return total_loss / mask.sum().item(), correct / mask.sum().item()
    return total_loss, correct / mask.sum().item()


# 多模型检测
loop_num = 0
model_map = {0: 0}
for j in range(1):

    test_acc = 0
    args.model = model_map[j]
    while (1):
        if loop_num > 100: break
        model_path = '../../models/model_' + str(loop_num)
        if not osp.exists(model_path):
            loop_num += 1
            continue
        model.load_state_dict(torch.load(model_path))

        fp = []
        tn = []
        loss, test_acc = test(data.test_mask)
        final_acc = (len(data.test_mask) - len(fp)) / len(data.test_mask)
        # print(model_path + ' ' + str(loop_num) + '  loss:{:.4f}'.format(loss) + '  acc:{:.4f}'.format(
        #     test_acc) + '  fp:' + str(len(fp)))
        print(model_path + ' detecting')
        for i in tn:
            data.test_mask[i] = False
        if test_acc == 1: break
        loop_num += 1
    if test_acc == 1: break

# 自定义输出
# 将模型输出结果中id对应到节点id
f = open('id_to_uuid.txt', 'r')
node_map = {}
for line in f:
    line = line.strip('\n').split('\t')
    node_map[int(line[0])] = [line[1], line[2]]
f.close()

ioc_list = load_IOC()

# 将id对应到实际节点类型和预测节点类型
# 维护result列表，result[i] = '1'表示id为i的节点被标记为恶意相关
result = []
tot_node = len(data.test_mask)
for i in range(tot_node):
    result.append('0')
for i in range(tot_node):
    if data.test_mask[i]:
        result[i] = '1'
        neighbor = set()
        if i in adj.keys():
            for j in adj[i]:
                neighbor.add(j)
                if not j in adj.keys(): continue
                for k in adj[j]:
                    neighbor.add(k)
        if i in adj2.keys():
            for j in adj2[i]:
                neighbor.add(j)
                if not j in adj2.keys(): continue
                for k in adj2[j]:
                    neighbor.add(k)
        for j in neighbor:
            try:
                if calculate_IOC_sim(node_map[j][0], ioc_list, 0.9) == 1:
                    result[j] = '1'
            except KeyError:
                continue

# 读label，等下映射
f_label = open('../../models/label.txt', 'r')
label_map = {}
for i in f_label:
    temp = i.strip('\n').split('\t')
    label_map[int(temp[1])] = temp[0]
    label_num += 1
f_label.close()

# 写进alarm
print('Start writing result.')
fw = open('../output/alarm.txt', 'w')
fw.write(str(len(data.test_mask)) + '\n')
for i in range(len(data.test_mask)):
    if data.test_mask[i]:
        neibor = set()
        if i in adj.keys():
            for j in adj[i]:
                neibor.add(j)
                if not j in adj.keys(): continue
                for k in adj[j]:
                    neibor.add(k)
        if i in adj2.keys():
            for j in adj2[i]:
                neibor.add(j)
                if not j in adj2.keys(): continue
                for k in adj2[j]:
                    neibor.add(k)

        if len(neibor) >= 1:
            fw.write('\n')
            fw.write(str(i) + ':')

        for j in neibor:
            try:
                if result[j] == '1':
                    fw.write(' ' + str(j))
            except KeyError:
                continue
fw.close()

# 普通写文件
# 输出为<id, 节点id, 实际类型, 预测类型, 最近活动时间>
fw = open('../output/output.txt', 'w')
for i, x in enumerate(result):
    if x == '1':
        # print()
        try:
            fw.write(str(i) + '  ' + node_map[i][0] + '  ' + label_map[final_result[i][0]] + '  ' + label_map[
                final_result[i][1]] + '  ' + node_map[i][1] + '\n')
        except KeyError:
            try:
                fw.write(str(i) + '  ' + node_map[i][0] + '  ' + label_map[
                    final_result[i][0]] + '  ' + "------ERROR------" + '  ' + node_map[i][1] + '\n')
            except KeyError:
                continue
fw.close()

# # 按活动时间排序，写文件
# # 输出为<id, 节点id, 实际类型, 预测类型, 最近活动时间>
# ft = open('output_time.json', 'w')
# node_list = []
# for i, x in enumerate(result):
#     node_info = {}
#     if x == '1':
#         node_info['id'] = str(i)
#         node_info['uuid'] = node_map[i][0]
#         node_info['type'] = label_map[final_result[i][0]]
#         try:
#             node_info['pred_type'] = label_map[final_result[i][1]]
#         except(KeyError):
#             node_info['pred_type'] = "------ERROR------"
#         node_info['time'] = node_map[i][1]
#         if node_info['type'] != node_info['pred_type']:
#             node_info['wrong'] = '1'
#         else:
#             node_info['wrong'] = '0'
#         node_list.append(node_info)
#
# node_list = sorted(node_list, key=itemgetter('time'), reverse=True)
#
# for node in node_list:
#     ft.write(str(node) + '\n')
# ft.close()

# 按活动时间排序，写文件
# 输出为<id, 节点id, 实际类型, 预测类型, 最近活动时间, 类型是否预测错误, threat id, apt type, ip>
# ft = open('../output/output_time.json', 'w')
node_list = []
for i, x in enumerate(result):
    node_info = {}
    try:
        if x == '1':
            node_info['id'] = str(i)
            node_info['uuid'] = node_map[i][0]
            node_info['type'] = label_map[final_result[i][0]]
            try:
                node_info['pred_type'] = label_map[final_result[i][1]]
            except(KeyError):
                node_info['pred_type'] = "------ERROR------"
            node_info['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(node_map[i][1][0:10])))
            if node_info['type'] != node_info['pred_type']:
                node_info['wrong'] = '1'
            else:
                node_info['wrong'] = '0'
            node_info['threat_id'] = threat_id
            node_info['apt_type'] = get_apt_type(apt_type_cnt)
            node_info['ip'] = '127.0.0.1'
            node_list.append(node_info)
    except KeyError:
        continue

# 按时间顺序写数据库 （其实也不用按时间顺序写，查的时候按时间顺序查就可以
# select * from node_detection order by time desc;
# node_list = sorted(node_list, key=itemgetter('time'), reverse=True)
for one_node in node_list:
    write_node_result(one_node)
# node_list_str = json.dumps(node_list)
# node_json = json.loads(node_list_str)
#
# ft.write(str(node_json))
# ft.close()

# update threat table
write_apt_type(get_apt_type(apt_type_cnt), threat_id)

# show('Finish testing graph ' + str(graphId) + ' in model ' + str(args.model))
print(str(apt_type_cnt))
print(get_apt_type(apt_type_cnt))
show("finished.")
