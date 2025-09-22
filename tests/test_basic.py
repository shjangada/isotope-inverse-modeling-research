"""
Basic tests for the orographic precipitation model.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermodynamics import thermodyn
from linear_model import d2_linear
from drying import drying
from fractionation import fractionation


class TestThermodynamics:
    """Test thermodynamic calculations."""
    
    def test_thermodyn_basic(self):
        """Test basic thermodynamic calculation."""
        Nm = 0.005
        T0 = 280.0
        speed = 10.0
        
        qs0, Hw, senscoef = thermodyn(Nm, T0, speed)
        
        # Basic sanity checks
        assert qs0 > 0, "Surface saturation mixing ratio should be positive"
        assert Hw > 0, "Moisture scale height should be positive"
        assert 0 < senscoef < 1, "Sensitivity coefficient should be between 0 and 1"
        
        # Typical ranges for realistic atmospheric conditions
        assert 0.001 < qs0 < 0.1, f"qs0 = {qs0} seems unrealistic"
        assert 1000 < Hw < 20000, f"Hw = {Hw} seems unrealistic"


class TestLinearModel:
    """Test linear precipitation model."""
    
    def test_d2_linear_basic(self):
        """Test basic linear model functionality."""
        # Create simple synthetic topography
        nx, ny = 64, 64
        topo = np.zeros((ny, nx))
        
        # Add a simple Gaussian mountain
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        center_x, center_y = nx//2, ny//2
        topo = 1000 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * 10**2))
        
        # Model parameters
        u0, v0 = 10.0, 5.0
        tau = 600.0
        T0 = 280.0
        Nm = 0.005
        
        precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
        
        # Basic checks
        assert precip.shape == topo.shape, "Precipitation should have same shape as topography"
        assert np.all(precip >= 0), "Precipitation should be non-negative"
        assert np.max(precip) > 0, "Should have some precipitation"
        assert qs0 > 0 and Hw > 0, "Thermodynamic parameters should be positive"


class TestDrying:
    """Test drying calculations."""
    
    def test_drying_basic(self):
        """Test basic drying functionality."""
        # Simple test case
        nx, ny = 32, 32
        precip = np.random.rand(ny, nx) * 5  # Random precipitation 0-5 mm/h
        
        x = np.arange(1, nx + 1)
        y = np.arange(1, ny + 1)
        X, Y = np.meshgrid(x, y)
        
        u0, v0 = 10.0, 5.0
        qs0 = 0.01
        Hw = 5000.0
        T0 = 280.0
        
        drying_ratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
        
        # Basic checks
        assert drying_ratio.shape == precip.shape, "Drying ratio should match precipitation shape"
        assert np.all(drying_ratio >= 0), "Drying ratio should be non-negative"
        assert np.all(np.isfinite(drying_ratio)), "Drying ratio should be finite"


class TestFractionation:
    """Test isotopic fractionation."""
    
    def test_fractionation_basic(self):
        """Test basic fractionation functionality."""
        # Simple test case
        drying_ratio = np.array([[0.0, 0.1, 0.2], 
                                [0.3, 0.5, 0.7]])
        initialcomp = -100.0  # ‰
        alpha = 0.985
        
        deltaD = fractionation(drying_ratio, initialcomp, alpha)
        
        # Basic checks
        assert deltaD.shape == drying_ratio.shape, "δD should match drying ratio shape"
        assert np.all(deltaD <= initialcomp), "δD should become more negative (or stay same)"
        assert np.all(np.isfinite(deltaD)), "δD should be finite"
        
        # More drying should lead to more negative δD
        assert deltaD[0, 0] >= deltaD[0, 2], "More drying should give more negative δD"


class TestIntegration:
    """Integration tests."""
    
    def test_complete_workflow(self):
        """Test complete model workflow."""
        # Create simple synthetic topography
        nx, ny = 32, 32
        topo = np.zeros((ny, nx))
        
        # Add a simple mountain
        x = np.arange(nx)
        y = np.arange(ny) 
        X_topo, Y_topo = np.meshgrid(x, y)
        center_x, center_y = nx//2, ny//2
        topo = 800 * np.exp(-((X_topo - center_x)**2 + (Y_topo - center_y)**2) / (2 * 8**2))
        
        # Model parameters
        u0, v0 = 15.0, 8.0
        tau = 500.0
        T0 = 285.0
        Nm = 0.004
        deltaD0 = -80.0
        alpha_eff = 0.98
        
        # Build grid
        x = np.arange(1, nx + 1)
        y = np.arange(1, ny + 1)
        X, Y = np.meshgrid(x, y)
        
        # Run complete workflow
        precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
        drying_ratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
        deltaD = fractionation(drying_ratio, deltaD0, alpha_eff)
        
        # Integration checks
        assert precip.shape == (ny, nx), "Precipitation shape mismatch"
        assert drying_ratio.shape == (ny, nx), "Drying ratio shape mismatch"
        assert deltaD.shape == (ny, nx), "δD shape mismatch"
        
        assert np.max(precip) > 0, "Should have some precipitation"
        assert np.max(drying_ratio) > 0, "Should have some drying"
        assert np.min(deltaD) < deltaD0, "Should have some fractionation"
        
        print(f"Integration test passed:")
        print(f"  Max precipitation: {np.max(precip):.2f} mm/h")
        print(f"  Max drying ratio: {np.max(drying_ratio):.3f}")
        print(f"  δD range: {np.min(deltaD):.1f} to {np.max(deltaD):.1f} ‰")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])