"""Same as notebook"""
import unittest
import numpy as np
from .model import get_small_model
from .mixmatch import ArrayDataset, MixupLoader, MixMatchLoss
import torch.nn.functional as F
from torch import nn
import pickle


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


class TestMismatch(unittest.TestCase):
    def test_loader(self):
        (X_labeled, y_labeled, X_unlabeled) = pickle_load('cifar_subset.pkl')
        ds = ArrayDataset(X_labeled[:12], y_labeled[:12], X_unlabeled[:12])
        BS = 4
        model = get_small_model()
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
