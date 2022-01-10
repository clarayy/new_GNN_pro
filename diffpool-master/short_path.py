import numpy as np

read_dictionary = np.load('BA20_short_path.npy', allow_pickle=True).item()
print(read_dictionary[2][3])