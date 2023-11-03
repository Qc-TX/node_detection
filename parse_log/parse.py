import json

# 打开包含 JSON 数据的文件

value = {"srcmsg": "", "dstmsg": "", "edge_type": "", "time": ""}
open_set = ["ProcessStart"]
read_set = ["RegistryOpen", "ProcessDCStart", "FileIORead", "RegistryQuery", "RegistryEnumerateKey",
            "RegistryEnumerateValueKey", "RegistryQueryValue", "RegistryQueryMultipleValue", "FileIOQueryInfo"]
create_set = ["RegistryCreate", "FileIOCreate", "RegistryKCBCreate", "FileIOFileCreate"]
message_set = ["TcpIpSendIPV4", "TcpIpRecvIPV4", "TcpIpDisconnectIPV4", "TcpIpRetransmitIPV4", "TcpIpReconnectIPV4",
               "TcpIpTCPCopyIPV4", "TcpIpAcceptIPV4", "TcpIpFailed"]
modify_set = ["RegistrySetValue", "RegistrySetInformation"]
start_set = ["TcpIpConnectIPV4"]
rename_set = ["FileIORename"]
delete_set = ["RegistryDelete", "RegistryDeleteValue", "RegistryKCBDelete", "FileIODelete", "FileIOFileDelete"]
terminal_set = ["ProcessEnd"]
write_set = ["FileIOWrite"]
unknown_set = ["ThreadCSwitch", "DiskIOWrite", "DiskIOWriteInit", "ProcessDCEnd", "ProcessDefunct", "ThreadEnd",
               "ThreadDCEnd", "RegistryKCBRundownEnd", "FileIOOperationEnd", "FileIOCleanup", "FileIOClose",
               "RegistryKCBRundownBegin", "FileIOSetInfo", "DiskIORead", "DiskIOReadInit", "FileIODirEnum",
               "ProcessPerfCtr", "ProcessPerfCtrRundown", "RegistryVirtualize", "RegistryClose", "RegistryFlush",
               "FileIOFlush", "DiskIOFlushInit", "DiskIOFlushBuffers", "DiskIODrvComplReq", "DiskIODrvComplReqRet",
               "DiskIODrvComplRout", "DiskIODrvMjFnCall", "DiskIODrvMjFnRet", "PerfInfoThreadDPC", "PerfInfoDPC",
               "PerfInfoTimerDPC", "PerfInfoISR", "PerfInfoSysClEnter", "PerfInfoSysClEnter", "ImageLoad",
               "ImageUnload", "ImageDCStart", "ImageDCEnd", "FileIODirNotify", "FileIOFSControl", "FileIOName",
               "FileIOFileRundown", "ALPC-Receive-Message", "ALPC-Send-Message", "ALPC-Unwait",
               "ALPC-Wait-For-New-Message", "ALPC-Wait-For-Reply", "ThreadStart", "ThreadDCStart", "TcpIpSendIPV6",
               "TcpIpRecvIPV6", "TcpIpDisconnectIPV6", "TcpIpRetransmitIPV6", "TcpIpReconnectIPV6", "TcpIpTCPCopyIPV6",
               "TcpIpConnectIPV6", "TcpIpAcceptIPV6"]

registry_set = ["RegistryCreate", "RegistryOpen", "RegistryDelete", "RegistryQuery", "RegistrySetValue",
                "RegistryDeleteValue", "RegistryQueryValue", "RegistryEnumerateKey", "RegistryEnumerateValueKey",
                "RegistryQueryMultipleValue", "RegistrySetInformation", "RegistryFlush", "RegistryKCBCreate",
                "RegistryKCBDelete", "RegistryKCBRundownBegin", "RegistryKCBRundownEnd", "RegistryVirtualize",
                "RegistryClose"]
process_set = ["ProcessStart", "ProcessEnd", "ProcessDCStart", "ProcessDCEnd", "ProcessDefunct", "ProcessPerfCtr",
               "ProcessPerfCtrRundown"]
thread_set = ["ThreadStart", "ThreadEnd", "ThreadDCStart", "ThreadDCEnd", "ThreadCSwitch"]
network_set = ["TcpIpSendIPV4", "TcpIpSendIPV6", "TcpIpRecvIPV4", "TcpIpDisconnectIPV4", "TcpIpRetransmitIPV4",
               "TcpIpReconnectIPV4", "TcpIpTCPCopyIPV4", "TcpIpRecvIPV6", "TcpIpDisconnectIPV6", "TcpIpRetransmitIPV6",
               "TcpIpReconnectIPV6", "TcpIpTCPCopyIPV6", "TcpIpConnectIPV4", "TcpIpAcceptIPV4", "TcpIpConnectIPV6",
               "TcpIpAcceptIPV6", "TcpIpFailed"]
fileio_set = ["FileIOCreate", "FileIODirEnum", "FileIODirNotify", "FileIOSetInfo", "FileIODelete", "FileIORename",
              "FileIOQueryInfo", "FileIOFSControl", "FileIOName", "FileIOFileCreate", "FileIOFileDelete",
              "FileIOFileRundown", "FileIOOperationEnd", "FileIORead", "FileIOWrite", "FileIOCleanup", "FileIOClose",
              "FileIOFlush"]
diskio_set = ["DiskIOWrite", "DiskIORead", "DiskIOReadInit", "DiskIOWriteInit", "DiskIOFlushInit", "DiskIOFlushBuffers",
              "DiskIODrvComplReq", "DiskIODrvComplReqRet", "DiskIODrvComplRout", "DiskIODrvMjFnCall",
              "DiskIODrvMjFnRet"]
alpc_set = ["ALPC-Receive-Message", "ALPC-Send-Message", "ALPC-Unwait", "ALPC-Wait-For-New-Message",
            "ALPC-Wait-For-Reply"]
other_set = ["PerfInfoThreadDPC", "PerfInfoDPC", "PerfInfoTimerDPC", "PerfInfoISR", "PerfInfoSysClEnter",
             "PerfInfoSysClEnter", "ImageLoad", "ImageUnload", "ImageDCStart", "ImageDCEnd"]


def getEdge_type(eventname):
    if eventname in open_set: return "OPEN"
    if eventname in read_set: return "READ"
    if eventname in create_set: return "CREATE"
    if eventname in message_set: return "MESSAGE"
    if eventname in modify_set: return "MODIFY"
    if eventname in start_set: return "START"
    if eventname in rename_set: return "RENAME"
    if eventname in delete_set: return "DELETE"
    if eventname in terminal_set: return "TERMINAL"
    if eventname in write_set: return "WRITE"
    if eventname in unknown_set: return "FALSE"


def getSrcmsg(eventname, event):
    ret = {}
    if eventname in message_set:
        ret["FLOW"] = event["args"]["saddr"] + ":" + str(event["args"]["sport"])
    else:
        ret["PROCESS"] = event["PName"]
    return ret


def getDstmsg(eventname, event):
    ret = {}
    if eventname in message_set:
        ret["FLOW"] = event["args"]["daddr"] + ":" + str(event["args"]["dport"])
    elif eventname in registry_set:
        ret["FILE"] = event["args"]["KeyName"]
    elif eventname in start_set:
        ret["PROCESS"] = event["args"]["ImageFileName"]
    elif eventname in fileio_set:
        try:
            ret["FILE"] = event["args"]["OpenPath"]
        except:
            ret["FILE"] = event["args"]["FileName"]
    return ret


i = 1
with open("test.json", "r") as source_file, open("output.json", "w") as target_file, open("prase_error.txt",
                                                                                          "w") as error_file:
    # 逐行读取源文件
    for line in source_file:
        try:
            data = json.loads(line)
            eventName = data["Event"]
            edge_type = getEdge_type(eventName)
            value["time"] = data["TimeStamp"]
            value["edge_type"] = getEdge_type(eventName)
            value["srcmsg"] = getSrcmsg(eventName, data)
            value["dstmsg"] = getDstmsg(eventName, data)
            if value["dstmsg"] != "{}":
                print(i)
                i = i + 1
                target_file.write(json.dumps(value) + "\n")
            else:
                continue
        except:
            error_file.write(line + "\n")
