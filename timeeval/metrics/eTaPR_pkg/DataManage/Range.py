# To store a single anomaly
class Range:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def get_len(self):
        return self._last_timestamp - self._first_timestamp + 1

    def __eq__(self, other):
        return self._first_timestamp == other.get_time()[0] and self._last_timestamp == other.get_time()[1]

    def distance(self, other_range) -> int:
        if min(self._last_timestamp, other_range.get_time()[1]) - max(self._first_timestamp, other_range.get_time()[0]) > 0:
            return 0
        else:
            return min(abs(self._first_timestamp - other_range.get_time()[1]),
                       abs(self._last_timestamp - other_range.get_time()[0]))

    def compare(self, other_range) -> int:
        if min(self._last_timestamp, other_range.get_time()[1]) - max(self._first_timestamp, other_range.get_time()[0]) > 0:
            return 0
        elif self._last_timestamp - other_range.get_time()[0] < 0:
            return -1
        else:
            return 1

def stream_2_ranges(self, prediction_stream: list) -> list:
    result = []
    for i in range(len(prediction_stream)-1):
        start_time = 0
        if prediction_stream[i] == 0 and prediction_stream[i+1] == 1:
            start_time = i+1
        elif prediction_stream[i] == 1 and prediction_stream[i+1] == 0:
            result.append(Range(start_time, i, ''))
    return result
