IOC_dict = {}


def num_write(start: int, end: int, num_IOC_dict: dict, IOC_index: int):
    i = 1
    while i <= len(num_IOC_dict):
        if start <= i <= end:
            num_IOC_dict[list(num_IOC_dict.keys())[i - 1]] = IOC_index
            i += 1
        else:
            i += 1


f = open('IOC.txt', 'r')
fw = open('classified_IOC.txt', 'w')
for line in f:
    if line.strip('\n') in IOC_dict.keys():
        IOC_dict[line.strip('\n')+"_"] = 0
    else:
        IOC_dict[line.strip('\n')] = 0
f.close()

num_write(2, 6, IOC_dict, 1)
num_write(8, 16, IOC_dict, 2)
num_write(18, 28, IOC_dict, 3)
num_write(30, 39, IOC_dict, 4)
num_write(41, 52, IOC_dict, 5)
num_write(54, 69, IOC_dict, 6)
num_write(71, 94, IOC_dict, 7)
num_write(96, 113, IOC_dict, 8)
num_write(115, 119, IOC_dict, 9)


for IOC,clas in IOC_dict.items():
    if clas != 0:
        fw.write(IOC + '\t' + str(clas) + '\n')
fw.close()
