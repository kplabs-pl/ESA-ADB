from timeeval.metrics.eTaPR_pkg.DataManage import Range
import time
import datetime
import pandas as pd

def load_stream_2_range(stream_data: list, normal_label: int, anomaly_label: int, is_range_name: bool) -> list:
    return_list = []
    start_id = -1
    end_id = -1
    id = 0
    range_id = 1

    prev_val = -2 #Set prev_val as a value different to normal and anomalous labels

    for val in stream_data:
        if val == anomaly_label and (prev_val == normal_label or prev_val == -2): #Enter the anomaly range
            start_id = id
        elif val == normal_label and prev_val == anomaly_label: #Go out the anomaly range
            name_buf = ''
            if is_range_name:
                name_buf = str(range_id)
            end_id = id - 1
            return_list.append(Range.Range(start_id, end_id, name_buf))
            range_id += 1
            #start_id = 0

        id += 1
        prev_val = val
    if start_id > end_id: #start_id != 0 and start_id != -1: #if an anomaly continues till the last point
        return_list.append(Range.Range(start_id, id - 1, str(range_id)))

    return return_list


def load_stream_file(filename: str, normal_label: int, anomaly_label: int, is_range_name: bool) -> list:
    return_list = []
    start_id = -1
    end_id = -1
    id = 0
    range_id = 1
    #is_first = True

    prev_val = -2 #Set prev_val as a value different to normal and anomalous labels

    f = open(filename, 'r', encoding='utf-8', newline='')

    for line in f.readlines():
        val = int(line.strip().split()[0])

        '''
        #skip the first line
        if is_first:
            if val == anomaly_label:
                start_id = id
            prev_val = val
            is_first = False
            continue
        '''

        if val == anomaly_label and (prev_val == normal_label or prev_val == -2): #Enter the anomaly range
            start_id = id
        elif val == normal_label and prev_val == anomaly_label: #Go out the anomaly range
            name_buf = ''
            if is_range_name:
                name_buf = str(range_id)
            end_id = id - 1
            return_list.append(Range.Range(start_id, end_id, name_buf))
            range_id += 1
            #start_id = 0

        id += 1
        prev_val = val
    f.close()
    if start_id > end_id: #start_id != 0 and start_id != -1: #if an anomaly continues till the last point
        return_list.append(Range.Range(start_id, id - 1, str(range_id)))

    return return_list

def load_range_file(filename: str, time_format: str) -> list:
    return_list = []
    #is_first = True

    f = open(filename, 'r', encoding='utf-8', newline='')
    for line in f.readlines():
        # skip the first line
        #if is_first:
            #is_first = False
            #continue

        items = line.strip().split(',')
        if time_format == 'index':
            first_idx = int(items[0])
            last_idx = int(items[1])
        else:
            first_idx = string_to_unixtime(items[0], time_format)
            last_idx = string_to_unixtime(items[1], time_format)

        name_buf = ''
        if len(items) > 2:
            name_buf = str(items[2])

        return_list.append(Range.Range(first_idx, last_idx, name_buf))
    f.close()

    for idx in range(1, len(return_list)):
        if return_list[idx].get_time()[0] <= return_list[idx-1].get_time()[1]:
            print("Error: ranges ({},{}) and ({},{}) are overlapped in {}".format(return_list[idx-1].get_time()[0],
                                                                                  return_list[idx-1].get_time()[1],
                                                                                  return_list[idx].get_time()[0],
                                                                                  return_list[idx].get_time()[1], filename))
            exit(0)

    return return_list


def unixtime_to_string(epoch: int, format: str) -> str:
    return datetime.datetime.fromtimestamp(epoch).strftime(format) #'%Y-%m-%d %I:%M:%S %p'


def string_to_unixtime(timestamp: str, format: str) -> int:
    return int(time.mktime(datetime.datetime.strptime(timestamp, format).timetuple()))


def save_range_list(filename: str, range_list: list) -> None:
    f = open(filename, encoding='utf-8', mode='w')
    for single_range in range_list:
        first, last = single_range.get_time()
        f.writelines(str(first)+','+str(last)+','+single_range.get_name()+'\n')
    f.close()

# Assume that the first line of input files including the information of file format and its corresponding information
# This function handles three types of file format
def load_file(filename: str, filetype: str) -> list:
    assert(filetype == 'range' or filetype == 'stream')

    if filetype == 'stream':
        return load_stream_file(filename, 1, -1, True)
    elif filetype == 'range':
        return load_range_file(filename, 'index')


def make_attack_file(input_files: list, sep: str, label_featname: str, input_normal_label: int, input_anomalous_label: int,
                     output_stream_file: str, output_range_file: str, output_normal_label: int, output_anomalous_label: int) -> None:
    label = []
    for an_input_file in input_files:
        temp_file = pd.read_csv(an_input_file, sep=sep)
        label += temp_file[label_featname].values.tolist()

    with open(output_stream_file, 'w') as f:
        for a_label in label:
            if a_label == input_normal_label:
                f.write('{}\n'.format(output_normal_label))
            elif a_label == input_anomalous_label:
                f.write('{}\n'.format(output_anomalous_label))
            else:
                print("There is an unknown label, " + a_label, flush=True)
                f.close()
                return

    ranges = load_stream_2_range(label, 0, 1, False)
    save_range_list(output_range_file, ranges)

def save_range_2_stream(filename: str, range_list: list, last_idx: int, normal_label: int, anomalous_label: int) -> None:
    f = open(filename, encoding='utf-8', mode='w')
    range_id = 0
    for idx in range(last_idx):
        if idx < range_list[range_id].get_time()[0]:
            f.writelines('{}\n'.format(normal_label))
        elif range_list[range_id].get_time()[0] <= idx <= range_list[range_id].get_time()[1]:
            f.writelines('{}\n'.format(anomalous_label))
        else:
            f.writelines('{}\n'.format(normal_label))
            if range_id < len(range_list) - 1:
                range_id += 1
    f.close()
