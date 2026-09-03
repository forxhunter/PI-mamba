
import torch
import sys
import os

# Add package source to path
sys.path.append("/data2/2026_RNAAI/v12_package/src")

from backbone_pi_mamba import PIMambaBackbone

def test_v12_backbone():
    print("Initializing V12 Backbone...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PIMambaBackbone(
        d_model=256,
        n_layers=2,
        d_state=16
    ).to(device)
    
    print("Running Forward Pass...")
    B, L = 2, 50
    coords = torch.randn(B, L, 3, device=device)
    t = torch.rand(B, device=device)
    
    # Check new return signature: velocity, omega_logits
    velocity, omega_logits = model(coords, t)
    
    print(f"Velocity shape: {velocity.shape}")
    print(f"Omega Logits shape: {omega_logits.shape}")
    
    assert velocity.shape == (B, L, 3)
    assert omega_logits.shape == (B, L, 2)
    
    print("Running Sampling (with Projection)...")
    samples = model.sample(length=30, n_samples=1, n_steps=5, device=device)
    print(f"Sample shape: {samples.shape}")
    
    print("SUCCESS: V12 Backbone verified.")

if __name__ == "__main__":
    test_v12_backbone()
