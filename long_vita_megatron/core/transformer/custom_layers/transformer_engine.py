import torch
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.legacy.model.rms_norm import RMSNorm
from megatron.training import get_args
# from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
# from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
# from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
# from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D

        
class PTNorm:
    """
    Conditional Initialization of Transformer-Engine’s LayerNorm or RMSNorm Instance
    """
    
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        args = get_args()
        if config.normalization == "LayerNorm":
            if args.tp_2d:
                instance = LayerNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
            else:
                instance = nn.LayerNorm(
                    normalized_shape=hidden_size,
                    eps=eps,
                )
        elif config.normalization == "RMSNorm":
            if args.tp_2d:
                instance = RMSNorm2D(
                    hidden_size,
                    eps=eps,
                    last_dim_split_comm_intf=TPYCollectiveComm(),
                )
                instance.use_fused_rmsnorm = False
            else:
                instance = RMSNorm(
                    dim=hidden_size,
                    eps=eps,
                    sequence_parallel=config.sequence_parallel,
                )
                instance.use_fused_rmsnorm = True
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance

class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
