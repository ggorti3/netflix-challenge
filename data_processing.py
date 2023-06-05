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

if __name__ == "__main__":
    file_paths = [
        "data/archive/combined_data_1.txt",
        "data/archive/combined_data_2.txt",
        "data/archive/combined_data_3.txt",
        "data/archive/combined_data_4.txt"
    ]
    new_file_path = "data/all_data.csv"
    combined_data_to_csv(file_paths, new_file_path)
    
    file_paths = ["data/archive/probe.txt"]
    new_file_path = "data/probe.csv"
    combined_data_to_csv(file_paths, new_file_path)