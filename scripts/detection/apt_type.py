import difflib

apt_type_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def get_apt_type(num_list: list) -> int:
    return num_list.index(max(num_list))


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

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

if __name__ == "__main__":
    # 载入IOC列表，格式参考classified_IOC.txt（可以用parse_IOC.py处理普通IOC列表得到）
    IOC_dict = load_IOC()

    # 载入node列表，表内元素是node的名字，比如"explore.exe"或者"d:\explore.exe"
    node_file = open(node_list_file,"r")
    node_list = []
    for line in node_list:
        node_list.append(line.strip("\n"))

    # 挨个计算node与IOC中的相似度，并维护计数apt_type_cnt列表
    for node in node_list:
        calculate_IOC_sim(node,IOC_dict,0.8)

    # get_apt_type获得apt类型
    print(get_apt_type(apt_type_cnt))
