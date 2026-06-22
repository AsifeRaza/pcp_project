from pathlib import Path
import numpy as np
import pickle

# Load data
#
def load_data():
    data_folder = Path(r"D:\PCP\data")
    #data_folder = Path(r"D:\PCP\data")
    npz_files = sorted(data_folder.glob("*.npz"))
    # print(f"Found {len(npz_files)} files")
    # print(npz_files)

    #
    # data = np.load(npz_files[0])
    # print(data.files)
    # for key in data.files:
    #     print(key, data[key].shape)

    #
    subjects = []
    for file in sorted(data_folder.glob("*.npz")):
        data = np.load(file)
        X = data["X"]
        eye = data["y"]
        # print(np.shape(X))
        # print(np.shape(eye))

        subjects.append({
            #"subject_id": file.stem,
            "X": X,
            "eye": eye
        })
    #     print(len(subjects))
    #     print("X size MB:", X.nbytes / 1e6)
    # print(f"Loaded {len(subjects)} subjects")

    # print(subjects[0]["X"].shape)
    # print(subjects[0]["eye"].shape)
    return(subjects)
