"""
Conditional Wasserstein GAN for Synthetic Medical Data Generation
Built by Prashant Ambati
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .generator import Generator
from .discriminator import Discriminator


class ConditionalWGAN:
    """
    Conditional Wasserstein GAN with Gradient Penalty for medical data synthesis.
    """
    
    def __init__(self, noise_dim=100, condition_dim=10, data_dim=20, 
                 lr_g=0.0001, lr_d=0.0004, lambda_gp=10, device='cpu'):
        """
        Initialize the Conditional WGAN.
        
        Args:
            noise_dim (int): Dimension of noise vector
            condition_dim (int): Dimension of condition vector
            data_dim (int): Dimension of medical data
            lr_g (float): Learning rate for generator
            lr_d (float): Learning rate for discriminator
            lambda_gp (float): Gradient penalty coefficient
            device (str): Device to run on
        """
        self.device = device
        self.lambda_gp = lambda_gp
        
        # Initialize networks
        self.generator = Generator(
            noise_dim=noise_dim,
            condition_dim=condition_dim,
            output_dim=data_dim
        ).to(device)
        
        self.discriminator = Discriminator(
            input_dim=data_dim,
            condition_dim=condition_dim
        ).to(device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'wasserstein_distance': []
        }
    
    def train_step(self, real_data, conditions, n_critic=5):
        """
        Perform one training step.
        
        Args:
            real_data (torch.Tensor): Real medical data
            conditions (torch.Tensor): Condition vectors
            n_critic (int): Number of discriminator updates per generator update
            
        Returns:
            dict: Training losses
        """
        batch_size = real_data.size(0)
        
        # Train Discriminator
        d_losses = []
        for _ in range(n_critic):
            self.optimizer_d.zero_grad()
            
            # Real data loss
            d_real = self.discriminator(real_data, conditions)
            d_real_loss = -torch.mean(d_real)
            
            # Fake data loss
            noise = torch.randn(batch_size, self.generator.noise_dim, device=self.device)
            fake_data = self.generator(noise, conditions)
            d_fake = self.discriminator(fake_data.detach(), conditions)
            d_fake_loss = torch.mean(d_fake)
            
            # Gradient penalty
            gradient_penalty = self.discriminator.compute_gradient_penalty(
                real_data, fake_data, conditions, self.device
            )
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.optimizer_d.step()
            
            d_losses.append(d_loss.item())
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        noise = torch.randn(batch_size, self.generator.noise_dim, device=self.device)
        fake_data = self.generator(noise, conditions)
        g_fake = self.discriminator(fake_data, conditions)
        g_loss = -torch.mean(g_fake)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        # Calculate Wasserstein distance approximation
        wasserstein_dist = torch.mean(d_real) - torch.mean(d_fake)
        
        # Update history
        self.history['g_loss'].append(g_loss.item())
        self.history['d_loss'].append(np.mean(d_losses))
        self.history['wasserstein_distance'].append(wasserstein_dist.item())
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': np.mean(d_losses),
            'wasserstein_distance': wasserstein_dist.item()
        }
    
    def generate_synthetic_data(self, num_samples, conditions):
        """
        Generate synthetic medical data.
        
        Args:
            num_samples (int): Number of samples to generate
            conditions (torch.Tensor): Conditions for generation
            
        Returns:
            torch.Tensor: Generated synthetic data
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.noise_dim, device=self.device)
            synthetic_data = self.generator(noise, conditions)
        
        return synthetic_data
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.history = checkpoint['history']