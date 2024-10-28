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
    

class DataGenerator:
    """
    working on createing the assumption from the 3.1 

    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def generate_batch(self,batch_size:int,device:torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Generates data following Assumption 3.1:
        - x ~ N(0, I_d)
        - z = β^T x
        - y = z + ε, ε ~ N(0, σ²)
        """
        d ,D = self.config.d, self.config.D

        # Sample β uniformly from unit sphere as per paper
        beta = torch.randn(batch_size,d)
        beat = F.normalize(beta, dim =1)

        # genreate x n(0,i_d)
        x = torch.randn(batch_size,D+1,d)
        # Generate z = β^T x
        z = torch.bmm(x, beta.unsqueeze(2))
        # gen y =z + ε
        epsilon = torch.randn_like(z) * np.sqrt(self.config.sigma2)
        y = z +epsilon 

        # construc e_Cot as per eq (3) in the paper
        E_CoT = torch.zeros(batch_size, d + 2,D+1,device=device)
        E_CoT[:, :d, :] = x.transpose(1, 2)
        E_CoT[:,d:-1] = z[:,:-1,0]
        E_CoT[:d,d + 1:-1] = y[:,:-1,0]

        return E_CoT.to(device), z[:, -1].to(device), y[:, -1].to(device)
    

def train_model(config: ModelConfig, device: torch.device = torch.device('cuda')):

    """
    Training implementation following paper's optimization objective (6)
    """

    model = CoherentTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    data_generator = DataGenerator(config)

    model.train()
    for step in range(config.n_train_steps):
       E_CoT, z_true, y_true = data_generator.generate_batch(config.batch_size, device)

       z_pred , y_pred = model(E_CoT)

       # compute loss follwing eq
       loss = F.mse_loss(y_pred,y_true.squeeze())

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if step & 1000  ==0:
           print(f"Step {step}, Loss: {loss.item():.6f}")

    return model


def evaluate_sensitivity(model: CoherentTransformer,
                       config: ModelConfig,
                       noise_type: str,
                       noise_std: float,
                       device: torch.device):
    """
    Implements sensitivity analysis from Section 3.3
    """
    model.eval()
    data_generator = DataGenerator(config)
    
    total_loss = 0
    n_eval = 1000

    with torch.no_grad():
        for _ in range(n_eval):
            E_CoT , z_true , y_true = data_generator.generate_batch(1,device)

            # adding some noise
            if noise_type =='y':
                E_CoT[:,config.d+1 :-1] += torch.randn_like(E_CoT[:,config.d+1,:-1]) * noise_std
            elif noise_type == 'x':
                E_CoT[: , :config.d,:] += torch.randn_like(E_CoT[:,config.d,:-1]) * noise_std
            elif noise_type =='z':
                E_CoT[:, config.d,:-1] += torch.randn_like(E_CoT[:,config])

            z_pred,y_pred = model(E_CoT)
            loss = F.mse_loss(y_pred,y_true.squeeze())
            total_loss += loss.item()

        return total_loss / n_eval
    

def main():
    # set a random seedn
    torch.manual_seed(42)

    config= ModelConfig()

    # training the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model= train_model(config,device)

    # Run sensitivity analysis as in paper
    noise_std = 1.0
    for noise_type in ['y', 'x', 'z']:
        sensitivity = evaluate_sensitivity(model, config, noise_type, noise_std, device)
        print(f"Sensitivity to {noise_type}-noise: {sensitivity:.6f}")

if __name__ == "__main__":
    main()





       





        





