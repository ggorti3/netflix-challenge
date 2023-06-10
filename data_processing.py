import os.path

def combined_data_to_csv(file_paths, new_file_path):
    with open(new_file_path, "w") as nf:
        for file_path in file_paths:
            with open(file_path, "r") as f:
                line = f.readline()
                while line:
                    movie_id = line.split(":")[0]
                    line = f.readline()
                    while line and line[-2] != ":":
                        nf.write(movie_id + "," + line)
                        line = f.readline()

def train_test_split(all_data_path, probe_data_path, data_dir):
    probe_keys = set()
    with open(probe_data_path, 'r') as f:
        line = f.readline()
        while line:
            mid, uid = line.strip().split(",")
            mid = int(mid)
            uid = int(uid)
            key = (mid, uid)
            probe_keys.add(key)
            line = f.readline()

    with open(os.path.join(data_dir, "train_data.csv"), "w") as f1:
        with open(os.path.join(data_dir, "test_data.csv"), "w") as f2:
            with open(all_data_path, 'r') as f:
                line = f.readline()
                while line:
                    mid, uid, r, d = line.strip().split(",")
                    mid = int(mid)
                    uid = int(uid)
                    key = (mid, uid)
                    
                    data_str = "{},{},{},{}\n".format(mid, uid, r, d)

                    if key in probe_keys:
                        f2.write(data_str)
                    else:
                        f1.write(data_str)

                    line = f.readline()


if __name__ == "__main__":
    # file_paths = [
    #     "data/archive/combined_data_1.txt",
    #     "data/archive/combined_data_2.txt",
    #     "data/archive/combined_data_3.txt",
    #     "data/archive/combined_data_4.txt"
    # ]
    # new_file_path = "data/all_data.csv"
    # combined_data_to_csv(file_paths, new_file_path)
    
    # file_paths = ["data/archive/probe.txt"]
    # new_file_path = "data/probe.csv"
    # combined_data_to_csv(file_paths, new_file_path)

    train_test_split("./data/all_data.csv", "./data/probe.csv", "./data")