"""
End-to-end forward model: topography -> precipitation -> drying -> δD

Example script demonstrating the complete orographic precipitation model
with isotopic fractionation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fixed imports - match your actual file names
from D2linear import d2_linear
from drying import drying  
from fractionation import fractionation


def main():
    """Run the complete forward model."""
    
    # -------------------- User parameters --------------------
    u0 = 10.0          # m/s zonal wind (from west, positive eastward)
    v0 = 10.0          # m/s meridional wind (from south, positive northward)
    tau = 600.0        # s total delay time (cloud + fallout)
    T0 = 280.0         # K surface temperature
    Nm = 0.005         # 1/s Brunt–Väisälä frequency (stability)
    deltaD0 = -100.0   # ‰ initial vapor isotopic composition (δD₀)
    alpha_eff = 0.985  # effective Rayleigh exponent parameter
    
    # -------------------- Load topography --------------------
    # Handle different ways the script might be run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from examples/
    
    # Try .mat files first since they're more reliable
    mat_paths = [
        os.path.join(project_root, 'data', 'topography.mat'),
        os.path.join('data', 'topography.mat'),
        'topography.mat'
    ]
    
    topo = None
    for path in mat_paths:
        try:
            print(f"Trying to load: {path}")
            mat_data = loadmat(path)
            topo = mat_data['OLYMPICTOPO']
            print(f"Successfully loaded topography from .mat file: {path}")
            print(f"Data type: {type(topo)}, Shape: {topo.shape}")
            
            # Save a clean .npy version for future use
            save_path = os.path.join(project_root, 'data', 'topography_clean.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, topo)
            print(f"Saved clean .npy version to: {save_path}")
            break
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    # If .mat not found, try clean .npy files  
    if topo is None:
        npy_paths = [
            os.path.join(project_root, 'data', 'topography_clean.npy'),
            os.path.join('data', 'topography_clean.npy'),
        ]
        
        for path in npy_paths:
            try:
                topo = np.load(path)
                print(f"Loaded topography from clean .npy file: {path}")
                break
            except FileNotFoundError:
                continue
    
    if topo is None:
        print("Error: Could not find topography data file.")
        print("Searched in these locations:")
        all_paths = mat_paths + [os.path.join(project_root, 'data', 'topography_clean.npy')]
        for path in all_paths:
            print(f"  {os.path.abspath(path)}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    ny, nx = topo.shape
    print(f"Topography shape: {ny} x {nx}")
    
    # -------------------- Build coordinate grid --------------------
    x = np.arange(1, nx + 1)
    y = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x, y)
    
    # -------------------- Linear orographic precipitation --------------------
    print("Computing precipitation...")
    precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
    print(f"Surface saturation mixing ratio (qs0): {qs0:.6f}")
    print(f"Moisture scale height (Hw): {Hw:.1f} m")
    
    # -------------------- Path-integrated drying field --------------------
    print("Computing drying field...")
    dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
    
    # -------------------- Rayleigh fractionation to δD --------------------
    print("Computing isotopic fractionation...")
    deltaD = fractionation(dryingratio, deltaD0, alpha_eff)
    
    # -------------------- Create figures --------------------
    print("Creating plots...")
    
    # Figure 1: δD field
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(deltaD, origin='lower', cmap='RdYlBu_r')
    plt.colorbar(im1, label='δD (‰)')
    plt.title('Predicted δD (‰) — Rayleigh from model drying')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    # Figure 2: Precipitation field
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(precip, origin='lower', cmap='Blues')
    plt.colorbar(im2, label='Precipitation (mm h⁻¹)')
    plt.title('Linear Orographic Precipitation (mm h⁻¹)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    plt.tight_layout()
    
    # Create results directory with proper path handling
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'forward_model_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Topography subset (like the MATLAB example)
    plt.figure(figsize=(12, 4))
    
    subset_range = slice(150, 350)  # Equivalent to MATLAB's 150:350
    
    plt.subplot(1, 3, 1)
    im3 = plt.imshow(topo[subset_range, subset_range], origin='lower', cmap='terrain')
    plt.colorbar(im3, label='Elevation (m)')
    plt.title('Topography Subset')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    plt.subplot(1, 3, 2)
    im4 = plt.imshow(precip[subset_range, subset_range], origin='lower', cmap='Blues')
    plt.colorbar(im4, label='Precipitation (mm h⁻¹)')  
    plt.title('Precipitation Subset')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    plt.subplot(1, 3, 3)
    im5 = plt.imshow(deltaD[subset_range, subset_range], origin='lower', cmap='RdYlBu_r')
    plt.colorbar(im5, label='δD (‰)')
    plt.title('δD Subset')
    plt.xlabel('Grid X') 
    plt.ylabel('Grid Y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'subset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # -------------------- Print summary statistics --------------------
    print("\n" + "="*50)
    print("MODEL RESULTS SUMMARY")
    print("="*50)
    print(f"Precipitation range: {precip.min():.3f} to {precip.max():.3f} mm/h")
    print(f"Mean precipitation: {precip.mean():.3f} mm/h")
    print(f"Drying ratio range: {dryingratio.min():.3f} to {dryingratio.max():.3f}")
    print(f"δD range: {deltaD.min():.1f} to {deltaD.max():.1f} ‰")
    print(f"Mean δD: {deltaD.mean():.1f} ‰")
    
    # Save results with proper path handling
    np.save(os.path.join(results_dir, 'precipitation.npy'), precip)
    np.save(os.path.join(results_dir, 'drying_ratio.npy'), dryingratio) 
    np.save(os.path.join(results_dir, 'deltaD.npy'), deltaD)
    print(f"\nResults saved to {results_dir}")
    print("="*50)


if __name__ == "__main__":
    main()