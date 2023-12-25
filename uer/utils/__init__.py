from uer.utils.tokenizers import CharTokenizer
from uer.utils.tokenizers import SpaceTokenizer
from uer.utils.tokenizers import BertTokenizer
from uer.utils.data import *
from uer.utils.act_fun import *
from uer.utils.optimizers import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "seq2seq": Seq2seqDataset,
            