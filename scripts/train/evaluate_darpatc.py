f_gt = open('groundtruth/groundtruth_nodeId.txt', 'r')
f_alarm = open('../output/alarm.txt', 'r')

eps = 1e-10

f = open('groundtruth/id_to_uuid.txt', 'r')
node_map = {}
for line in f:
	line = line.strip('\n').split('\t')
	node_map[int(line[0])] = line[1]
f.close()

gt = {}
for line in f_gt:
	gt[int(line.strip('\n').split(' ')[0])] = 1

ans = []
ans_f = {}

for line in f_alarm:
	if line == '\n': continue
	if not ':' in line:
		tot_node = int(line.strip('\n'))
		for i in range(tot_node):
			ans.append('tn')
		for i in gt:
			ans[i] = 'fn'
		continue
	line = line.strip('\n')
	a = int(line.split(':')[0])
	b = line.split(':')[1].strip(' ').split(' ')
	flag = 0
	try:
		for i in b:
			ans[int(i)] = 'fp'
			if i == '': continue
			if int(i) in gt.keys():
				ans[int(i)] = 'tp'
				flag = 1
	except ValueError:
		continue

	if a in gt.keys():
		ans[a] = 'tp'
	else:
		if flag == 0:
			ans[a] = 'fp'

# 将模型输出结果alarm中id对应到节点id
f_alarm = open('../output/alarm.txt', 'r')
negative = []
for line in f_alarm:
	if line == '\n': continue
	if not ':' in line:
		tot_node = int(line.strip('\n'))
		for i in range(tot_node):
			negative.append('0')
		continue
	line = line.strip('\n')
	a = int(line.split(':')[0])
	b = line.split(':')[1].strip(' ').split(' ')
	negative[a] = '1'
	for i in b:
		if i == '': continue
		negative[int(i)] = '1'
# cnt = 1
# for i,x in enumerate(negative):
# 	if x =='1':
# 		print("num" + str(cnt) + " " + str(i)+'  ' + node_map[i])
# 		cnt += 1






tn = 0
tp = 0
fn = 0
fp = 0
for i in ans:
	if i == 'tp': tp += 1
	if i == 'tn': tn += 1
	if i == 'fp': fp += 1
	if i == 'fn': fn += 1
print(tp,fp,tn,fn)
# precision = tp/(tp+fp+eps)
# recall = tp/(tp+fn+eps)
precision = (tp+tn)/(tp+fp+tn+fn)
recall = tp/(tp+fn)
fscore = 2*precision*recall/(precision+recall+eps)


print('Precision: ', precision)
print('Recall: ', recall)
print('F-Score: ', fscore)