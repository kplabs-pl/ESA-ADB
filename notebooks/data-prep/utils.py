from enum import Enum


class AnnotationLabel(Enum):
    NOMINAL = 0
    ANOMALY = 1
    RARE_EVENT = 2
    GAP = 3
    INVALID = 4


def encode_telecommands(param_df, resampling_rule):
    # Encode telecommands as 0-1 peaks ensuring that they are not removed after resampling
    original_timestamps = param_df.index.copy()
    for timestamp in original_timestamps:
        timestamp_before = timestamp - resampling_rule
        if len(param_df.loc[timestamp_before:timestamp]) == 1:
            param_df.loc[timestamp_before] = 0
            param_df = param_df.sort_index()
        timestamp_after = timestamp + resampling_rule
        if len(param_df.loc[timestamp:timestamp_after]) == 1:
            param_df.loc[timestamp_after] = 0
            param_df = param_df.sort_index()

    return param_df

def find_full_time_range(params_dict: dict):
    # Find full dataset time range
    start_time = []
    end_time = []
    for df in params_dict.values():
        if len(df) == 0:
            continue
        start_time.append(df.index[0])
        end_time.append(df.index[-1])
    start_time = min(start_time)
    end_time = max(end_time)

    return start_time, end_time
