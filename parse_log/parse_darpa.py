import json

value = {"srcmsg": "", "dstmsg": "", "edge_type": "", "time": ""}
dst_process_action = ['OPEN', 'TERMINATE']
dst_flow_action = ['MESSAGE', 'START']
dst_file_action = ['READ', 'CREATE', 'MODIFY', 'RENAME', 'DELETE', 'WRITE']


def get_dst_type(action):
    if action in dst_process_action: return 'PROCESS'
    if action in dst_flow_action: return 'FLOW'
    if action in dst_file_action:
        return 'FILE'
    else:
        return 'no_type'


f = open('/Users/enchanter/Downloads/part_darpa.json', 'r')
fw = open('test_result.json', 'w')
num = 0
for line in f:
    json_line = json.loads(line)
    try:
        value['edge_type'] = json_line['action']
        src_type = 'PROCESS'
        src_dict = {}
        src_dict[src_type] = json_line['properties']['image_path']
        value['srcmsg'] = src_dict

        dst_dict = {}
        dst_type = get_dst_type(value['edge_type'])
        if dst_type != 'no_type':
            value['dstmsg'] = dst_dict
            if dst_type == 'process':
                dst_dict[dst_type] = str(json_line['properties']['command_line']).split('-').split('  ')[0].strip(
                    ' ').strip(' 0xffffffff').strip('\"')
                value['dstmsg'] = dst_dict
            if dst_type == 'flow':
                dst_dict[dst_type] = json_line['properties']['dest_ip'] + ':' + json_line['properties'][
                    'dest_port']
                value['dstmsg'] = dst_dict
            if dst_type == 'file':
                dst_dict[dst_type] = json_line['properties']['file_path']
                value['dstmsg'] = dst_dict
            value['time'] = json_line['timestamp']
            fw.write(json.dumps(value) + "\n")
        else: continue
    except:
        continue
