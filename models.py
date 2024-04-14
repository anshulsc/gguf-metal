
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import utils
from huggingface_hub import snapshot_download
from  mlx.utils import tree_flatten, tree_unflatten

@dataclass
class ModelArgs:
    hidden_size: int 
    num_hidden_layers: int 
    intermediate_size: int 
    num_attention_heads: int
    rms_norm_eps: int 
    vocab_size: int 
    num_key_value_heads: int 
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
    

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()




class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args 
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
