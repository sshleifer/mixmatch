def mixmatch(X_labeled, y, X_unlabeled, model, augment_fn, T=0.5, K=2, alpha=0.75):
    """Generate labeled and unlabeled batches for mixmatch. Helpers are below."""
    xb = augment_fn(X_labeled)
    n_labeled = len(xb)
    ub = [augment_fn(X_unlabeled) for _ in range(K)]  # unlabeled
    qb = sharpen(sum(map(model, ub)) / K, T)
    C = np.concatenate
    Ux = C(ub, axis=0)
    Uy = C([qb for _ in range(K)], axis=0)
    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = C([Ux, xb], axis=0)[indices]
    Wy = C([qb, y], axis=0)[indices]
    X, p = mixup(xb, Wx[:n_labeled], y, Wy[:n_labeled], alpha)
    U, q = mixup(Ux, Wx[n_labeled:], Uy, Wy[n_labeled:], alpha)
    return C([X, U], axis=1), C([p, q], axis=1), n_labeled


def sharpen(x, T):
    numerator = x ** (1 / T)
    return numerator / numerator.sum(axis=1, keepdims=True)

def lin_comb(a, b, frac_a):
    try:
        return (frac_a * a) + (1 - frac_a) * b
    except ValueError:
        return shit_mult(frac_a, a) + shit_mult(1-frac_a, b)


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, -alpha, x1.shape[0])
    beta = np.maximum(beta, 1 - beta)
    return lin_comb(x1, x2, beta), lin_comb(y1, y2, beta)


class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        super().__init__()
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, preds, y, n_labeled):
        labeled_loss = self.xent(preds[:n_labeled], y[:n_labeled])
        unlabeled_loss = self.mse(preds[n_labeled:], y[n_labeled:])
        return labeled_loss + (self.lambda_u * unlabeled_loss)
