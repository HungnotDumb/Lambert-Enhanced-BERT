import torch
import math
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class WordEmbedding(nn.Module):
    