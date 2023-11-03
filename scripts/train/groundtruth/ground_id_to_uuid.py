
f = open('id_to_uuid.txt', 'r')
node_map = {}
for line in f:
    line = line.strip('\n').split('\t')
    node_map[int(line[0])] = [line[1], line[2]]
f.close()

f_id = open('groundtruth_id.txt', 'r')
node_id = []
for line in f_id:
    node_id.append(line.strip('\n'))
f_id.close()

fw = open('groundtruth_uuid.txt', 'w')
for node in node_map:
    if str(node) in node_id:
        fw.write(node_map[node][0]+'\n')
fw.close()