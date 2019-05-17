import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ArrayDataset(Dataset):
    def __init__(self, X, y, X_unlabeled):
        super().__init__()
        self.X = X
        self.y = y
        self.X_unlabeled = X_unlabeled
        self.last_labeled = False

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        """Alternate generating labeled and unlabeled."""
        if self.last_labeled:
            self.last_labeled = False
            zeros = np.zeros_like(self.y[0])
            return (self.X_unlabeled[np.random.randint(0, len(self.X_unlabeled), )], zeros)
        else:
            self.last_labeled = True
            idx = np.random.randint(0, len(self.X), )
            return (self.X[idx], self.y[idx])


to_arr = lambda x: x.detach().numpy()


def sharpen(x, T):
    numerator = x ** (1 / T)
    return numerator / numerator.sum(dim=1, keepdim=True)


def mixup(x1, x2, y1, y2, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha).sample((x1.shape[0],))
    beta = torch.max(beta, 1 - beta)
    return linear_comb(x1, x2, beta), linear_comb(y1, y2, beta)


def linear_comb(x1, x2, l):
    # TODO: why doesnt broadcasting work?
    orig = torch.cat([(x1[i] * l[i]).unsqueeze(0) for i in range(len(l))])
    other = torch.cat([(x2[i] * (1 - l[i])).unsqueeze(0) for i in range(len(l))])
    mixed = orig + other
    if len(mixed.shape) == 3: mixed = mixed.unsqueeze(0)  # bs=2
    return mixed





class MixupLoader(DataLoader):

    def __init__(self, ds, batch_size, T=0.5, K=2, alpha=0.75, verbose=False):
        self.bs = batch_size
        assert self.bs % 2 == 0
        self.ds = ds
        self.T = T
        self.K = K
        self.alpha = alpha
        self.verbose = verbose
        super().__init__(ds, collate_fn=self.collate_fn, batch_size=self.bs,
                         num_workers=0)

    def get_pseudo_labels(self, ub):
        preds = self.model(ub) / self.K
        qb = sharpen(preds, self.T).detach()
        return qb

    @staticmethod
    def augment_fn(X):
        # TODO(SS): fix me
        return X

    def collate_fn(self, examples):
        K, T, alpha = self.K, self.T, self.alpha
        C = lambda arrs: np.concatenate(np.expand_dims(arrs, 0))
        X_labeled = C([X for X, y_ in examples if y_.sum() == 1])
        y = torch.Tensor(np.array([y_ for X, y_ in examples if y_.sum() == 1]))
        X_unlabeled = C([X for X, y_ in examples if y_.sum() == 0])

        xb = torch.Tensor(self.augment_fn(X_labeled))
        n_labeled = len(X_labeled)
        ub = torch.cat(
            [torch.Tensor(self.augment_fn(X_unlabeled)) for _ in range(K)])  # unlabeled
        qb = self.get_pseudo_labels(ub)
        Ux = ub
        Uy = torch.cat([qb for _ in range(K)])

        # Shuffled labeled and unlabeled for mixup partners
        indices = torch.randperm(xb.size(0) + Ux.size(0))  # .to(self.device)
        Wx = torch.cat([xb, Ux], dim=0)[indices]
        Wy = torch.cat([y, qb], dim=0)[indices]
        np.testing.assert_allclose(to_arr(Wy).sum(1), 1., 3)

        X, p = mixup(xb, Wx[:n_labeled], y, Wy[:n_labeled], alpha)
        U, q = mixup(Ux, Wx[n_labeled:], Uy, Wy[n_labeled:], alpha)
        X = torch.cat([X, U], dim=0)
        Y = torch.cat([p, q], dim=0)
        if self.verbose:
            print(X_labeled.shape, X_unlabeled.shape)
            print(f'Wx:{Wx.shape}')
            print(f' p: {to_arr(p)}')
            print(f'Returing: x final: {X.shape}, Y_final: {np.round(to_arr(Y), 3)}')
        return X, Y


def cross_ent_continuous(logits, labels):
    # TODO(SS): Call softmax within
    y_cross = labels * logits.log()
    loss = -y_cross.sum(dim=1).mean()
    return loss


class MixMatchLoss(nn.Module):
    def __init__(self, lambda_u=100):
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, preds, y, n_labeled):
        labeled_loss = cross_ent_continuous(preds[:n_labeled], y[:n_labeled])
        unlabeled_loss = nn.MSELoss()(preds[n_labeled:], y[n_labeled:])
        return labeled_loss + (self.lambda_u * unlabeled_loss)
