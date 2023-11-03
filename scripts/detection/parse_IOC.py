IOC_dict = {}


def num_write(start: int, end: int, num_IOC_dict: dict, IOC_index: int):
    i = 1
    while i <= len(num_IOC_dict):
        if start <= i <= end:
            num_IOC_dict[list(num_IOC_dict.keys())[i - 1]] = IOC_index
            i += 1
        else:
            i += 1
            continue


f = open('IOC.txt', 'r')
fw = open('classified_IOC.txt', 'w')
for line in f:
    IOC_dict[line.strip('\n')] = 0
f.close()

num_write(2, 17, IOC_dict, 1)
num_write(20, 30, IOC_dict, 2)
num_write(33, 44, IOC_dict, 3)
num_write(48, 71, IOC_dict, 4)
num_write(75, 92, IOC_dict, 5)

for IOC,clas in IOC_dict.items():
    if clas != 0:
        fw.write(IOC + '\t' + str(clas-1) + '\n')
fw.close()
