import numpy as np
from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset
import PIL
import torchvision.transforms as T
from matplotlib import pyplot as plt
import numpy as np

preds = np.load('output/resnet18/preds.npy')
image, label = CIFAR10Dataset().__getitem__(0)
print(preds[0])
print(image.shape)
# print(tensor_to_image(image))
plt.imshow(np.array(image[0]))
plt.show()