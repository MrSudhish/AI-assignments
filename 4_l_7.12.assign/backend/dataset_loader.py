import os
import numpy as np
from preprocess import load_and_preprocess

def load_dataset(data_folder):
    X = []
    y = []

    for label in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, label)

        if not os.path.isdir(folder_path):
            continue

        print(f"Loading: {label}")

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                features = load_and_preprocess(img_path)
                X.append(features)
                y.append(label)
            except:
                print("Error:", img_path)

    return np.array(X), np.array(y)
