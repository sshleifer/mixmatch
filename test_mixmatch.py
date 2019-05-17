"""Same as notebook"""
import unittest
import numpy as np

arr = [1,2,3,4]
img = np.array(arr *4).reshape(4,4)
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
to_arr = lambda x: x.detach().numpy()
import pickle
from .dataset import ArrayDataset, MixupLoader
def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def cross_ent_continuous(logits, labels):
    # TODO(SS): Call softmax within
    y_cross = labels * logits.log()
    loss = y_cross.sum(dim=1).mean()
    return loss


class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        super().__init__()
        self.lambda_u = lambda_u
    def forward(self, preds, y, n_labeled):
        # This line fails cause y continuous
        labeled_loss = cross_ent_continuous(preds[:n_labeled], y[:n_labeled])
        unlabeled_loss = nn.MSELoss()(preds[n_labeled:], y[n_labeled:])
        return labeled_loss + (self.lambda_u * unlabeled_loss)


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
