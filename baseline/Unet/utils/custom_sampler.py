import torch
import random
import numpy as np
from torch.utils.data import Sampler

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.remainder = 1 if len(self.data_source)%5 else 0
        
    def __iter__(self):
        indices = []
        seq_ids = np.arange(0,len(self.data_source)//5 + self.remainder)
        while(len(seq_ids)>(1 if self.remainder else 0)):
            seq_id = random.randint(0,len(seq_ids)-self.remainder-1 )
            title = seq_ids[seq_id]
            for dada_i in range(0,5):
                indices.append(title*5 + dada_i)
            seq_ids = np.delete(seq_ids, seq_id)
        
        if self.remainder:
            for i in range(0, len(self.data_source) - seq_ids[0] * 5):
                indices.append(seq_ids[0]*5 + i)
            
        return iter(indices)
    def __len__(self):
        return len(self.data_source)