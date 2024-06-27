import os
current_dir = os.path.dirname(os.path.realpath(__file__))

data_raw_folder = os.path.abspath(os.path.join(current_dir, "../../data"))
data_processed_folder = os.path.abspath(os.path.join(data_raw_folder, "preprocessed"))
