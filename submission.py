import pandas as pd
import numpy as np

PATH = "output/resnet18_medium_imagenet_2/preds.npy"
data = np.load(PATH)
data = np.argmax(data, axis=1)

df = pd.DataFrame(data)
df.index += 1
df.to_csv("submission3.csv", header=["Category"], index_label="Id")
