from torch import nn

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def get_small_model():
    return nn.Sequential(
        nn.Conv2d(3, 2, 3, stride=1, padding=1),
        Flatten(),
        nn.Linear(2 * 32 * 32, 10),
    )
