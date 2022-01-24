import glob
import numpy as np

from buteo.machine_learning import augmentation

folder = "path_to_folder"
inputs = glob(folder + "input_*.npy")
labels = glob(folder + "label_*.npy")

x, y = augmentation(inputs, labels)

np.save(folder + "x_augmented.npy", x)
np.save(folder + "y_augmented.npy", y)
