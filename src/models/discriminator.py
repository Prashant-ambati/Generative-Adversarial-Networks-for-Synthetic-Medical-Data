"""
Discriminator Network for Conditional GAN
Built by Prashant Ambati
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real from synthetic medical data.
    
    Args:
        input_dim (int): Dimension of input data
        condition_dim (int): Dimension of condition vector
        hidden_dims (list): List of hidden layer dimensions
    """
    
    def __init__(self, input_dim=20, condition_dim=10, hidden_dims=[256, 512, 256]):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # Input dimension is data + condition
        total_input_dim = input_dim + condition_dim
        
        # Build the network layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value for real/fake classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, data, condition):
        """
        Forward pass of the discriminator.
        
        Args:
            data (torch.Tensor): Input data [batch_size, input_dim]
            condition (torch.Tensor): Condition vector [batch_size, condition_dim]
            
        Returns:
            torch.Tensor: Discriminator output [batch_size, 1]
        """
        # Concatenate data and condition
        x = torch.cat([data, condition], dim=1)
        
        # Get discriminator output
        output = self.network(x)
        
        return output
    
    def compute_gradient_penalty(self, real_data, fake_data, condition, device='cpu'):
        """
        Compute gradient penalty for WGAN-GP training.
        
        Args:
            real_data (torch.Tensor): Real data samples
            fake_data (torch.Tensor): Generated fake data samples
            condition (torch.Tensor): Condition vector
            device (str): Device to run on
            
        Returns:
            torch.Tensor: Gradient penalty
        """
        batch_size = real_data.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)
        
        # Interpolated samples
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = self.forward(interpolated, condition)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty