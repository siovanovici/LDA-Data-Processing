import numpy as np
import os
from pathlib import Path

# Function checks for new csv data folders* (not individual files) in raw_data_root and coverts them into a npy array
# format 

raw_data_root = 'data/raw_csv'
npy_data_root = 'data/processed_npy'

for path, subdirs, files in os.walk(raw_data_root):
    for name in files:

        if not os.path.exists(npy_data_root + os.path.join(path, name)[12:-16]):
            Path(npy_data_root + os.path.join(path, name)[12:-16]).mkdir(parents=True, exist_ok=True)

            for path2 in os.listdir(os.path.join(path, name)[0:-16]):
                path2 = os.path.join(path, name)[0:-16] + path2

                data = np.genfromtxt(path2, delimiter=';', skip_header=6, dtype='str')[..., 1::]
                header = np.genfromtxt(path2, delimiter=';', skip_header=3, dtype='str', max_rows=1)[1::]

                data = np.char.replace(data, ',', '.').astype(float)
                header = np.char.replace(header, ',', '.')
                header = [x[:-3] for x in header]

                np.save(npy_data_root + os.path.join(path, name)[12:-16] + header[0] + '_' + header[1] + '_' + header[2] + '.npy', data)