import torch
import torch.nn as nn
from uer.targets import *


class AlbertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and sentence order prediction (SOP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(AlbertTarget, self).__init__(args, vocab_size)

        self.factorized_embedding_parameterization = True
 