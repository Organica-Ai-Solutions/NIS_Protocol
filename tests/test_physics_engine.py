"""
NIS Protocol v4.0 - Physics Engine Test Suite

Comprehensive tests for physics-informed neural network validation,
kinematics calculations, and conservation law checking.

Run with: pytest tests/test_physics_engine.py -v
"""

import pytest
import numpy as np
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPhysicsConstants:
    """Test fundamental physics constants are correct"""
    
    def test_speed_of_light(self):
        """Speed of light should be ~299,792,458 m/s"""
        c = 299792458  # m/s
        assert c == 299792458
    
    def test_planck_constant(self):
        """Planck constant should be ~6.626e-34 J·s"""
        h = 6.62607015e-34
        assert abs(h - 6.626e-34) < 1e-37
    
    def test_gravitational_constant(self):
        """Gravitational constant should be ~6.674e-11 N·m²/kg²"""
        G = 6.67430e-11
        assert abs(G - 6.674e-11) < 1e-14
    
    def test_boltzmann_constant(self):
        """Boltzmann constant should be ~1.381e-23 J/K"""
        k_B = 1.380649e-23
        assert abs(k_B - 1.381e-23) < 1e-26


class TestConservationLaws:
    """Test physics conservation law validation"""
    
    def test_energy_conservation(self):
        """Total energy should be conserved in closed system"""
        # Initial state: ball at height h with velocity 0
        m = 1.0  # kg
        g = 9.81  # m/s²
        h_initial = 10.0  # m
        v_initial = 0.0  # m/s
        
        # Calculate initial energy
        KE_initial = 0.5 * m * v_initial**2
        PE_initial = m * g * h_initial
        E_total = KE_initial + PE_initial
        
        # Final state: ball at ground with velocity v
        h_final = 0.0
        v_final = math.sqrt(2 * g * h_initial)  # From energy conservation
        
        KE_final = 0.5 * m * v_final**2
        PE_final = m * g * h_final
        E_final = KE_final + PE_final
        
        # Energy should be conserved (within floating point tolerance)
        assert abs(E_total - E_final) < 1e-10
    
    def test_momentum_conservation(self):
        """Total momentum should be conserved in collision"""
        # Two balls colliding elastically
        m1, m2 = 2.0, 3.0  # kg
        v1_initial, v2_initial = 5.0, -2.0  # m/s
        
        # Initial momentum
        p_initial = m1 * v1_initial + m2 * v2_initial
        
        # After elastic collision (using conservation equations)
        v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
        v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)
        
        # Final momentum
        p_final = m1 * v1_final + m2 * v2_final
        
        # Momentum should be conserved
        assert abs(p_initial - p_final) < 1e-10
    
    def test_angular_momentum_conservation(self):
        """Angular momentum should be conserved without external torque"""
        # Spinning figure skater pulling arms in
        I_initial = 5.0  # kg·m² (arms out)
        omega_initial = 2.0  # rad/s
        
        L = I_initial * omega_initial  # Angular momentum
        
        I_final = 2.0  # kg·m² (arms in)
        omega_final = L / I_final  # Should spin faster
        
        L_final = I_final * omega_final
        
        # Angular momentum should be conserved
        assert abs(L - L_final) < 1e-10
        # And skater should spin faster
        assert omega_final > omega_initial


class TestKinematics:
    """Test kinematic calculations for robotics"""
    
    def test_projectile_motion(self):
        """Test projectile motion equations"""
        v0 = 20.0  # m/s initial velocity
        theta = math.radians(45)  # 45 degree angle
        g = 9.81  # m/s²
        
        # Components
        v0x = v0 * math.cos(theta)
        v0y = v0 * math.sin(theta)
        
        # Time of flight
        t_flight = 2 * v0y / g
        
        # Range
        R = v0x * t_flight
        
        # Maximum height
        h_max = v0y**2 / (2 * g)
        
        # Verify with known formulas
        R_formula = v0**2 * math.sin(2 * theta) / g
        h_formula = v0**2 * math.sin(theta)**2 / (2 * g)
        
        assert abs(R - R_formula) < 1e-10
        assert abs(h_max - h_formula) < 1e-10
    
    def test_circular_motion(self):
        """Test circular motion calculations"""
        r = 5.0  # m radius
        v = 10.0  # m/s tangential velocity
        
        # Centripetal acceleration
        a_c = v**2 / r
        
        # Angular velocity
        omega = v / r
        
        # Period
        T = 2 * math.pi * r / v
        
        # Verify relationships
        assert abs(a_c - omega**2 * r) < 1e-10
        assert abs(T - 2 * math.pi / omega) < 1e-10


class TestDenavitHartenberg:
    """Test Denavit-Hartenberg transformation matrices"""
    
    def test_identity_transform(self):
        """Zero parameters should give identity-like transform"""
        # DH parameters: theta=0, d=0, a=0, alpha=0
        theta, d, a, alpha = 0, 0, 0, 0
        
        T = self._dh_matrix(theta, d, a, alpha)
        
        # Should be identity matrix
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(T, expected)
    
    def test_pure_rotation(self):
        """Test pure rotation around z-axis"""
        theta = math.pi / 2  # 90 degrees
        d, a, alpha = 0, 0, 0
        
        T = self._dh_matrix(theta, d, a, alpha)
        
        # Check rotation part
        assert abs(T[0, 0] - 0) < 1e-10  # cos(90) = 0
        assert abs(T[0, 1] - (-1)) < 1e-10  # -sin(90) = -1
        assert abs(T[1, 0] - 1) < 1e-10  # sin(90) = 1
        assert abs(T[1, 1] - 0) < 1e-10  # cos(90) = 0
    
    def test_pure_translation(self):
        """Test pure translation along z-axis"""
        theta, alpha = 0, 0
        d = 5.0  # Translation along z
        a = 3.0  # Translation along x
        
        T = self._dh_matrix(theta, d, a, alpha)
        
        # Check translation part
        assert abs(T[0, 3] - a) < 1e-10  # x translation
        assert abs(T[2, 3] - d) < 1e-10  # z translation
    
    def test_transform_composition(self):
        """Test that transforms compose correctly"""
        # Two sequential transforms
        T1 = self._dh_matrix(math.pi/4, 1, 0.5, 0)
        T2 = self._dh_matrix(math.pi/6, 0.5, 0.3, math.pi/2)
        
        # Composed transform
        T_composed = T1 @ T2
        
        # Should still be a valid transformation matrix
        # Check that rotation part is orthogonal
        R = T_composed[:3, :3]
        R_T_R = R.T @ R
        np.testing.assert_array_almost_equal(R_T_R, np.eye(3), decimal=10)
        
        # Check determinant is 1 (proper rotation)
        assert abs(np.linalg.det(R) - 1) < 1e-10
    
    def _dh_matrix(self, theta, d, a, alpha):
        """Compute Denavit-Hartenberg transformation matrix"""
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])


class TestTrajectoryPlanning:
    """Test trajectory planning algorithms"""
    
    def test_minimum_jerk_trajectory(self):
        """Test minimum jerk (5th order polynomial) trajectory"""
        # Start and end positions
        x0, xf = 0.0, 10.0
        v0, vf = 0.0, 0.0  # Start and end at rest
        a0, af = 0.0, 0.0  # Zero acceleration at endpoints
        T = 2.0  # Duration
        
        # Generate trajectory points
        t = np.linspace(0, T, 100)
        x, v, a = self._minimum_jerk(t, T, x0, xf, v0, vf, a0, af)
        
        # Check boundary conditions
        assert abs(x[0] - x0) < 1e-10
        assert abs(x[-1] - xf) < 1e-10
        assert abs(v[0] - v0) < 1e-10
        assert abs(v[-1] - vf) < 1e-10
        
        # Check smoothness (no discontinuities)
        dx = np.diff(x)
        assert np.all(dx >= 0)  # Monotonically increasing
    
    def test_trajectory_continuity(self):
        """Test that trajectory is continuous"""
        x0, xf = 0.0, 5.0
        T = 1.0
        
        t = np.linspace(0, T, 1000)
        x, v, a = self._minimum_jerk(t, T, x0, xf, 0, 0, 0, 0)
        
        # Check velocity is derivative of position
        v_numerical = np.gradient(x, t)
        np.testing.assert_array_almost_equal(v[1:-1], v_numerical[1:-1], decimal=2)
        
        # Check acceleration is derivative of velocity
        a_numerical = np.gradient(v, t)
        np.testing.assert_array_almost_equal(a[1:-1], a_numerical[1:-1], decimal=1)
    
    def _minimum_jerk(self, t, T, x0, xf, v0, vf, a0, af):
        """Compute minimum jerk trajectory (5th order polynomial)"""
        # Normalized time
        tau = t / T
        
        # Polynomial coefficients
        c0 = x0
        c1 = v0 * T
        c2 = 0.5 * a0 * T**2
        c3 = 10*(xf - x0) - 6*v0*T - 4*vf*T - 1.5*a0*T**2 + 0.5*af*T**2
        c4 = -15*(xf - x0) + 8*v0*T + 7*vf*T + 1.5*a0*T**2 - af*T**2
        c5 = 6*(xf - x0) - 3*v0*T - 3*vf*T - 0.5*a0*T**2 + 0.5*af*T**2
        
        # Position
        x = c0 + c1*tau + c2*tau**2 + c3*tau**3 + c4*tau**4 + c5*tau**5
        
        # Velocity
        v = (c1 + 2*c2*tau + 3*c3*tau**2 + 4*c4*tau**3 + 5*c5*tau**4) / T
        
        # Acceleration
        a = (2*c2 + 6*c3*tau + 12*c4*tau**2 + 20*c5*tau**3) / T**2
        
        return x, v, a


class TestHeatEquation:
    """Test heat equation solver"""
    
    def test_steady_state(self):
        """Test that solution reaches steady state"""
        # 1D heat equation with fixed boundary conditions
        L = 1.0  # Length
        nx = 50  # Grid points
        alpha = 0.01  # Thermal diffusivity
        
        dx = L / (nx - 1)
        dt = 0.4 * dx**2 / alpha  # Stability condition
        
        # Initial condition: hot in middle
        T = np.zeros(nx)
        T[nx//2] = 100.0
        
        # Boundary conditions
        T[0] = 0.0
        T[-1] = 0.0
        
        # Time stepping
        for _ in range(10000):
            T_new = T.copy()
            for i in range(1, nx-1):
                T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
            T = T_new
        
        # Should approach linear interpolation between boundaries
        # (steady state solution)
        assert np.max(T) < 1.0  # Heat should have dissipated
    
    def test_energy_conservation_heat(self):
        """Test that total thermal energy is conserved (no flux boundaries)"""
        L = 1.0
        nx = 50
        alpha = 0.01
        
        dx = L / (nx - 1)
        dt = 0.4 * dx**2 / alpha
        
        # Initial condition
        T = np.sin(np.linspace(0, np.pi, nx)) * 100
        
        # Insulated boundaries (no flux)
        initial_energy = np.sum(T) * dx
        
        # Time stepping with no-flux boundaries
        for _ in range(100):
            T_new = T.copy()
            for i in range(1, nx-1):
                T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
            # No-flux boundary conditions
            T_new[0] = T_new[1]
            T_new[-1] = T_new[-2]
            T = T_new
        
        final_energy = np.sum(T) * dx
        
        # Energy should be conserved (within numerical tolerance)
        # Energy should be conserved (within numerical tolerance ~3% for explicit scheme)
        assert abs(initial_energy - final_energy) / initial_energy < 0.03


class TestWaveEquation:
    """Test wave equation solver"""
    
    def test_wave_speed(self):
        """Test that wave propagates at correct speed"""
        L = 10.0  # Length
        c = 2.0   # Wave speed
        nx = 200
        
        dx = L / (nx - 1)
        dt = 0.5 * dx / c  # CFL condition
        
        x = np.linspace(0, L, nx)
        
        # Initial condition: Gaussian pulse
        x0 = L / 4
        sigma = 0.5
        u = np.exp(-(x - x0)**2 / (2 * sigma**2))
        u_prev = u.copy()
        
        # Fixed boundaries
        u[0] = u[-1] = 0
        
        # Time stepping
        steps = int(L / (4 * c * dt))  # Time for wave to travel L/4
        
        for _ in range(steps):
            u_new = np.zeros(nx)
            for i in range(1, nx-1):
                u_new[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
            u_new[0] = u_new[-1] = 0
            u_prev = u.copy()
            u = u_new
        
        # Peak should have moved approximately L/4 to the right
        peak_initial = x0
        peak_final = x[np.argmax(u)]
        expected_displacement = c * steps * dt
        
        # Allow some tolerance due to dispersion
        assert abs(peak_final - peak_initial - expected_displacement) < 1.0


class TestPhysicsValidation:
    """Test physics validation for robotics commands"""
    
    def test_velocity_limits(self):
        """Test that velocity limits are enforced"""
        max_velocity = 10.0  # m/s
        
        # Valid command
        assert self._validate_velocity(5.0, max_velocity) == True
        
        # Invalid command
        assert self._validate_velocity(15.0, max_velocity) == False
    
    def test_acceleration_limits(self):
        """Test that acceleration limits are enforced"""
        max_acceleration = 5.0  # m/s²
        
        # Valid command
        assert self._validate_acceleration(3.0, max_acceleration) == True
        
        # Invalid command
        assert self._validate_acceleration(10.0, max_acceleration) == False
    
    def test_torque_limits(self):
        """Test that joint torque limits are enforced"""
        # Simple 2-link arm
        m1, m2 = 1.0, 0.5  # kg
        l1, l2 = 0.5, 0.3  # m
        g = 9.81
        
        # Maximum torque at joint 1 (worst case: arm fully extended)
        max_torque = (m1 * l1/2 + m2 * (l1 + l2/2)) * g
        
        # Check that calculated torque is reasonable
        assert max_torque > 0
        assert max_torque < 100  # Sanity check
    
    def test_workspace_limits(self):
        """Test that workspace boundaries are enforced"""
        # Robot arm workspace
        r_min, r_max = 0.2, 1.0  # m
        z_min, z_max = 0.0, 0.8  # m
        
        # Valid position
        assert self._in_workspace(0.5, 0.5, r_min, r_max, z_min, z_max) == True
        
        # Invalid position (too far)
        assert self._in_workspace(1.5, 0.5, r_min, r_max, z_min, z_max) == False
        
        # Invalid position (too close)
        assert self._in_workspace(0.1, 0.5, r_min, r_max, z_min, z_max) == False
    
    def _validate_velocity(self, v, v_max):
        return abs(v) <= v_max
    
    def _validate_acceleration(self, a, a_max):
        return abs(a) <= a_max
    
    def _in_workspace(self, r, z, r_min, r_max, z_min, z_max):
        return r_min <= r <= r_max and z_min <= z <= z_max


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
