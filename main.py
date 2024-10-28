import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import  Tuple, Optional
from databricks import  dataclass 

@dataclass
class ModelConfig:
    d: int = 100 # Input dimension d as in paper
    D: int =4 # Number of in-context examples
    sigma2:float = 0.1 # Noise variance σ²
    learning_rate :float = 1e-4
    batch_size: int = 256
    n_train_steps:int = 200000


class CoherentTransformer(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.config= config

        # this is based on the Assumption 3.2  in the paper for parameter format
        # WᵀQ WK format with specific structure
        self.W_K == nn.parameter(torch.randn(config.d +2,config.d+2))
        self.W_O == nn.parameter(torch.randn(config.d +2,config.d+2))
        self.W_V == nn.parameter(torch.randn(config.d +2,config.d+2))
        self.W_out = nn.Parameter(torch.randn(config.d + 2, config.d + 2))

        self.initialize_parameters()

    def initialize_parameters(self):
        d = self.config.d

        sigma2 = self.config.sigma2
        v_z = ((d-1)*sigma2 - 2) / ((d-1)*sigma2 + 2)
        v_y = v_z +4/((d-1)*sigma2+2)
        v_x = v_y - v_z

        # Set W_K^T W_Q structure as per paper
        with torch.no_grad():
            self.W_K.zero_()
            self.W_Q.zero_()

            #set blocks for x
            self.W_k[:d,:d] = torch.eye(d) * v_z
            # set v_z and v_y comp
            self.W_K[d,d] = v_z
            self.W_K[d+1, d+1] = v_y

            self.W_Q.copy_(self.W_K) # same struc  for W_q

            self.W_out.zero_()
            self.W_V.zero_()
            self.W_out[ d,d] = 1/v_x
            self.W_out[d+1, d+1] = 1/v_y

    def forward(self,E_CoT: torch.Tensor) ->Tuple[torch.Tensor,torch.Tensor]:
        """
        working on the def 3.2 from the paper
         Args:
            E_CoT: Input prompt matrix [batch_size, d+2, D+1]
        Returns:
            z_q: Predicted intermediate response
            y_q: Predicted final response
        
        """
        batch_size = E_CoT.shape[0]

        # step one  predcit z_q using eq  3 from the paper

        attention_weights = torch.bmm(
            torch.bmm(E_CoT.transpose(1, 2), self.W_K.T),
            torch.bmm(self.W_Q, E_CoT)
        ) / self.config.D

        z_q = torch.bmm(
            torch.bmm(self.W_out, self.W_V),
            torch.bmm(E_CoT, attention_weights)
        )[:, self.config.d, -1]

        # step 2 update the input with what was predciet  z_q
        E_CoT_updated = E_CoT.clone()
        E_CoT_updated[:,self.config.d, -1] = z_q

        attention_weights = torch.bmm(
            torch.bmm(E_CoT_updated.transpose(1, 2), self.W_K.T),
            torch.bmm(self.W_Q, E_CoT_updated)
        ) / self.config.D
        

        y_q = torch.bmm(
            torch.bmm(self.W_out, self.W_V),
            torch.bmm(E_CoT_updated, attention_weights)
        )[:, self.config.d + 1, -1]

        return z_q, y_q





        #predecit y  using the eq 4 from the paper





       





        





