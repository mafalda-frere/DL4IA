'''Define classifiers.
'''
import torch.nn as nn


class ShallowClassifier(nn.Module):     # en entrée : embedding que fournit le transformers
    ''' A shallow classifier with dense layers. '''

    def __init__(self, d_input, d_inner, n_classes):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(d_input, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, n_classes)
        )

    def forward(self, x):
        return self.logits(x)