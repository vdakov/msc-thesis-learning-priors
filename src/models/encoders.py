import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),                # This provides the nonlinearity
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

LinearEncoder = nn.Linear 
MLPEncoder = SimpleMLP
