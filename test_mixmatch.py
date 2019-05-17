"""Same as notebook"""
import unittest
import numpy as np

from mixmatch import MixMatchLoss

arr = [1,2,3,4]
img = np.array(arr *4).reshape(4,4)
import torch.nn.functional as F

to_arr = lambda x: x.detach().numpy()
import pickle
from .mixmatch import ArrayDataset, MixupLoader
def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


from torch import nn
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

model = nn.Sequential(
    nn.Conv2d(3, 2, 3, stride=1, padding=1),
    Flatten(),
    nn.Linear(2 * 32 * 32, 10),
)

class TestMismatch(unittest.TestCase):


    def test_mixup_torch(self):
        (X_labeled, y_labeled, X_unlabeled) = pickle_load('cifar_subset.pkl')
        ds = ArrayDataset(X_labeled[:12], y_labeled[:12], X_unlabeled[:12])
        BS = 4
        loader = MixupLoader(ds, batch_size=BS)
        loader.model = model
        loss_fn = MixMatchLoss()
        for xb, yb in loader:
            # print(x.shape,y.shape)
            # print(np.round(to_arr(yb), 3))
            preds = F.softmax(model.forward(xb), dim=1)
            loss = loss_fn(preds, yb, BS // 2)
            print(loss)
            break
