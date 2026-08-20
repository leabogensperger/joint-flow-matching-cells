import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch 

def load_ema_model(model, checkpoint_path, device):
    """
    Loads the EMA state dict from a checkpoint, cleaning 'module.' 
    prefixes and skipping metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'ema_state_dict' not in checkpoint:
        raise KeyError(f"No 'ema_state_dict' found in {checkpoint_path}")

    ema_state_dict = checkpoint['ema_state_dict']
    new_state_dict = OrderedDict()
    
    for k, v in ema_state_dict.items():
        if k == "n_averaged":
            continue
        # Strip 'module.' prefix if it exists (usually from DataParallel)
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded EMA model (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint.get('best_val_acc', 0):.4f})")
    return model, checkpoint

def plot_losses(train_history, val_history, val_metrics_history, save_path):
    plt.switch_backend('Agg') 
    
    # 1. Determine if we need the 3rd subplot
    has_extras = any(k in val_metrics_history[0] for k in ['conc_loss', 'adv_loss'])
    n_cols = 3 if has_extras else 2
    
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1: axes = [axes] # Handle edge case
    
    epochs = range(1, len(train_history) + 1)

    # --- Plot 1: Total Loss (Train vs Val) ---
    axes[0].plot(epochs, train_history, label='Train Total', color='#1f77b4', lw=2)
    axes[0].plot(epochs, val_history, label='Val Total', color='#ff7f0e', lw=2, linestyle='--')
    axes[0].set_title('Total VAE Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Core Components (SSIM & KLD) ---
    recon_vals = [m.get('ssim_loss', m.get('recon_loss', 0)) for m in val_metrics_history]
    kld_vals = [m.get('kld_loss', 0) for m in val_metrics_history]
    recon_label = 'SSIM Loss' if 'ssim_loss' in val_metrics_history[0] else 'MSE'
    
    axes[1].plot(epochs, recon_vals, label=recon_label, color='#2ca02c', lw=2)
    axes[1].set_ylabel(recon_label, color='#2ca02c')
    
    ax2_kld = axes[1].twinx()
    ax2_kld.plot(epochs, kld_vals, label='Val KLD', color='#d62728', lw=2)
    ax2_kld.set_ylabel('KLD', color='#d62728')
    
    axes[1].set_title('Reconstruction vs Regularization')
    axes[1].grid(True, alpha=0.3)
    
    # Combine legends for the twin axis
    lns1, lbs1 = axes[1].get_legend_handles_labels()
    lns2, lbs2 = ax2_kld.get_legend_handles_labels()
    axes[1].legend(lns1 + lns2, lbs1 + lbs2, loc='upper right')

    # --- Plot 3: Extra Metrics (Concentration & Adversarial) ---
    if has_extras:
        conc_vals = [m.get('conc_loss', 0) for m in val_metrics_history]
        adv_vals = [m.get('adv_loss', 0) for m in val_metrics_history]
        
        axes[2].plot(epochs, conc_vals, label='Conc. Regressor', color='#9467bd', lw=2)
        axes[2].set_ylabel('MSE (Concentration)', color='#9467bd')
        
        ax3_adv = axes[2].twinx()
        ax3_adv.plot(epochs, adv_vals, label='Adversarial Loss', color='#8c564b', lw=2, linestyle=':')
        ax3_adv.set_ylabel('Adv Loss', color='#8c564b')
        
        axes[2].set_title('Auxiliary Tasks')
        axes[2].grid(True, alpha=0.3)
        
        lns3, lbs3 = axes[2].get_legend_handles_labels()
        lns4, lbs4 = ax3_adv.get_legend_handles_labels()
        axes[2].legend(lns3 + lns4, lbs3 + lbs4, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def setup_experiment(cfg, root_dir, suffix=None):
    """
    Creates a unique directory for this run inside the project root.
    Includes an optional suffix (e.g., '_latent') for experiment tracking.
    """
    arch_name = cfg['model']['name']
    
    # Add suffix if provided (e.g., "VAE_latent")
    if suffix:
        arch_name = f"{arch_name}_{suffix.lstrip('_')}"
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Anchor the path to the root_dir
    run_dir = Path(root_dir) / "experiments" / arch_name / timestamp
    
    # parents=True ensures /experiments/ exists first
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup the config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(cfg, f)

    return run_dir