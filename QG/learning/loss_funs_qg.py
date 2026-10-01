#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""
import tensorflow.keras.backend as kb


#%% L2 + mass conservation loss
def param_loss_2D(alpha):

    def custom_loss(y_actual,y_pred):

        kd = alpha.kd
        delta = alpha.delta
        Ny = alpha.Ny
        Nx = alpha.Nx
        dx = alpha.dx
        dy = alpha.dy
        dy2 = dy*dy
        dx2 = dx*dx
        lam = 1

        # L2-error
        yloss = kb.sum(kb.square(y_actual[:,:,0:Ny*Nx] -y_pred[:,:,0:Ny*Nx]),axis=2)
        yloss = yloss +kb.sum(kb.square(y_actual[:,:,Ny*Nx:2*Nx*Ny] -y_pred[:,:,Ny*Nx:2*Nx*Ny]),axis=2)

        # Mass conservation
        yloss = yloss + lam*kb.abs(kb.sum(y_pred[:,:,0:Ny*Nx],axis=2))
        yloss = yloss + lam*kb.abs(kb.sum(y_pred[:,:,Ny*Nx:2*Ny*Nx],axis=2))


        return yloss
    return custom_loss

#%% L2 + mass conservation + VAE loss
def vae_loss_2D(alpha, lam_kl= 1):

    def custom_loss(y_actual,output):

        lam = 0.0001 # weight for KL divergence loss
        Ny = alpha.Ny
        Nx = alpha.Nx
        used_features = 2*Nx*Ny
        latent_dim = 60

        # VAE output
        y_pred = output[:,:,0:used_features]
        z_mean = output[:,:,used_features:used_features+latent_dim]
        z_sigma = output[:,:,used_features+latent_dim:used_features+2*latent_dim]

        # QG L2-error
        yloss = kb.sum(kb.square(y_actual[:,:,0:Ny*Nx] - y_pred[:,:,0:Ny*Nx]),axis=2)
        yloss = yloss + kb.sum(kb.square(y_actual[:,:,Ny*Nx:2*Nx*Ny] - y_pred[:,:,Ny*Nx:2*Nx*Ny]),axis=2)

        # QG Mass conservation
        yloss = yloss + lam*kb.abs(kb.sum(y_pred[:,:,0:Ny*Nx],axis=2))
        yloss = yloss + lam*kb.abs(kb.sum(y_pred[:,:,Ny*Nx:2*Ny*Nx],axis=2))


        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5*kb.sum( kb.square(z_sigma) + kb.square(z_mean) - 1. - kb.log(1e-8 + kb.square(z_sigma)) )

        # sum
        total_loss = yloss + lam*kl*lam_kl

        return total_loss
    return custom_loss
