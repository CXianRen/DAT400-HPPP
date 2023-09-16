import os
import glob

folder_path = "./sde_log"

parse_files = glob.glob(os.path.join(folder_path,"*.parse"))
parse_files.sort(reverse=True)
print(parse_files)


# return flops and bytes
def get_number_of_flops_bytes(file):
    total_flops=0
    total_byte=0
    lines=open(file).readlines()
    for line in lines:
        if line.find("--->Total FLOPs =")==0:
            # print(line)
            total_flops= line.split(' ')[-1]
        if line.find("--->Total Bytes =")==0:
            # print(line)
            total_byte= line.split(' ')[-1]
    # print(int(total_flops), int(total_byte))
    return int(total_flops), int(total_byte)

def get_execution_time(file):
    time=0
    lines=open(file).readlines()
    for line in lines:
        if line.find("The execution time of main loop is")==0:
            # print(line)
            time= line.split(' ')[-1]
    # print(float(time.replace('.\n','')))
    return float(time.replace('.\n',''))


def calculate_AI(float_num,byte_num):
    return float_num/byte_num


def calculate_perf(floats_num,time):
    return floats_num/time/1024/1024/1024  # convert to GB

def get_thread_data_size(file:str):
    return file.split('_')[2], file.split('_')[4]

header = "thread_num, data_size(MB),total_float,total_bytes,excution_time(sec),AI(float/byte),perf(GFlops/sec)"

print(header)

csv = open("extract.csv",'w+')
csv.write(header)
csv.write("\n")

for parse_file in parse_files:
    t,d = get_thread_data_size(parse_file)
    total_float, total_byte = get_number_of_flops_bytes(parse_file)
    etime = get_execution_time(parse_file.split('.log')[0]+'.terminal')
    AI = calculate_AI(total_float,total_byte)
    perf = calculate_perf(total_float, etime)
    csv_line = "%s,%s, %s,%s,%s,%.3f,%.3f"%(t,d,total_float,total_byte,etime,AI,perf)
    print(csv_line)
    csv.write(csv_line)
    csv.write("\n")

csv.close()
