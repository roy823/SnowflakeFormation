import math
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

def find_six_neighbors(s: np.ndarray):
    """
    Parameters:
        s: (N+2)x(N+2) array
    Returns:
        s1, s2, s3, s4, s5, s6: NxN array
    """
    # Find the eight neighbors (including diagonal ones) in a square grid
    s_u = s[:-2, 1:-1]
    s_d = s[2:, 1:-1]
    s_l = s[1:-1, :-2]
    s_r = s[1:-1, 2:]
    
    s_lu = s[:-2, :-2]
    s_ru = s[:-2, 2:]
    s_ld = s[2:, :-2]
    s_rd = s[2:, 2:]
    
    # Construct the 6 neighbors of a hexagon grid
    s1 = s_u
    s2 = s_d
    s3 = s_l
    s4 = s_r
    
    s5 = np.empty([s.shape[0]-2, s.shape[1]-2])
    s6 = np.empty([s.shape[0]-2, s.shape[1]-2])
    
    s5[:,0::2] = s_ld[:, 0::2]
    s6[:,0::2] = s_rd[:, 0::2]
    
    s5[:,1::2] = s_lu[:, 1::2]
    s6[:,1::2] = s_ru[:, 1::2]
    
    return (s1, s2, s3, s4, s5, s6)

def contrast(s, k:float=0.8, a: float=1):
    # Water vapor part contrast
    vapor_contrast = k*s # s<1
    vapor_color = plt.cm.gray(vapor_contrast)
    
    # Ice crystal part contrast
    ice_contrast = (np.exp(2*(s-0.9)*a) - 1) / (np.exp(2*(s-0.9)*a) + 1)
    ice_color = plt.cm.ocean_r(ice_contrast)
    
    y = np.zeros_like(s)
    y[s<1] = vapor_contrast[s<1]
    y[s>=1] = ice_contrast[s>=1]
    
    cm = np.zeros([s.shape[0], 4]) # 4: rgba
    cm[s<1] = vapor_color[s<1]
    cm[s>=1] = ice_color[s>=1]
    
    return y, cm

class EnhancedReiterCA:
    
    def __init__(self, grid_size: int, alpha: float, beta: float, gamma: float, 
                 seed_type: str = 'single', color_scheme: str = 'default'):
        """
        Enhanced version of Reiter Cellular Automata
        
        Parameters:
            grid_size: Size of the grid
            alpha: Diffusion parameter
            beta: Background water vapor
            gamma: Crystallization parameter
            seed_type: Initial seed type 'single', 'double', 'triple', 'star', 'ring'
            color_scheme: Color scheme 'default', 'winter', 'hot', 'rainbow'
        """
        self.grid_size = grid_size
        self.s = np.ones([grid_size+2, grid_size+2]) * beta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.color_scheme = color_scheme
        self.step_counter = 0
        
        # Set initial pattern
        center = grid_size//2+1
        
        if seed_type == 'single':
            # Single center point
            self.s[center, center] = 1.0
        
        elif seed_type == 'double':
            # Two center points
            offset = grid_size//10
            self.s[center-offset, center-offset] = 1.0
            self.s[center+offset, center+offset] = 1.0
            
        elif seed_type == 'triple':
            # Three points forming a triangle
            offset = grid_size//8
            self.s[center, center] = 1.0
            self.s[center-offset, center+offset] = 1.0
            self.s[center+offset, center+offset] = 1.0
            
        elif seed_type == 'star':
            # Star-shaped initial structure
            self.s[center, center] = 1.0
            for i in range(6):
                angle = i * np.pi / 3
                r = grid_size // 10
                x = int(center + r * np.cos(angle))
                y = int(center + r * np.sin(angle))
                self.s[x, y] = 1.0
                
        elif seed_type == 'ring':
            # Ring-shaped initial structure
            r = grid_size // 10
            for theta in np.linspace(0, 2*np.pi, 12):
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                self.s[x, y] = 1.0
        
        # Create boundary mask to detect edge collision
        self.boundary_mask = np.ones(self.s.shape, dtype=np.int32)
        boundary_size = grid_size // 6
        self.boundary_mask[boundary_size:-boundary_size, boundary_size:-boundary_size] = 0

    def frozen(self):
        return self.s[1:-1, 1:-1] >= 1
    
    def have_frozen_neighbors(self):
        s1, s2, s3, s4, s5, s6 = find_six_neighbors(self.s)
        return np.logical_or.reduce([
            s1>=1,
            s2>=1,
            s3>=1,
            s4>=1,
            s5>=1,
            s6>=1,
        ])
    
    def receptive(self):
        return np.logical_or(self.frozen(), self.have_frozen_neighbors())
    
    def update(self):
        """Update one step of CA simulation"""
        self.step_counter += 1
        
        # Exit if edge is touched
        if self.edge_touched():
            return
            
        receptive = self.receptive() # boolean array
        
        v = np.zeros([self.grid_size+2, self.grid_size+2])
        v[1:-1, 1:-1][receptive] = self.s[1:-1, 1:-1][receptive] + self.gamma
        
        u = np.zeros([self.grid_size+2, self.grid_size+2])
        u[0, :] = self.beta
        u[-1, :] = self.beta
        u[:, 0] = self.beta
        u[:, -1] = self.beta
        u[1:-1, 1:-1][~receptive] = self.s[1:-1, 1:-1][~receptive]
        
        u1,u2,u3,u4,u5,u6 = find_six_neighbors(u)
        u_mean = (u1 + u2 + u3 + u4 + u5 + u6) / 6.0
        
        # Add periodic oscillation to create more interesting crystal patterns
        if self.step_counter % 100 < 50:
            wave_factor = 1.0 + 0.05 * np.sin(self.step_counter / 50 * np.pi)
        else:
            wave_factor = 1.0
            
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + wave_factor * self.alpha * (u_mean - u[1:-1, 1:-1])
        self.s[1:-1, 1:-1] = u[1:-1, 1:-1] + v[1:-1, 1:-1]
    
    def edge_touched(self):
        """Check if edge is touched"""
        if np.any(self.s[self.boundary_mask == 1] >= 1):
            return True
        return False
    
    def diameter(self):
        """Calculate crystal diameter"""
        frozen_grid = self.frozen()
        return np.sum(frozen_grid[:, self.grid_size//2])
    
    def draw(self, ax: plt.Axes, contrast_param: float = 0.5):
        """Draw current CA state"""
        R, C = np.meshgrid(np.arange(self.grid_size+2), np.arange(self.grid_size+2), indexing="ij")
        X = 1.5*C
        Y = math.sqrt(3) * R + (C%2) * math.sqrt(3) / 2
        
        # Choose color scheme
        if self.color_scheme == 'winter':
            _, cm = self._winter_color_scheme(self.s.flatten(), contrast_param)
        elif self.color_scheme == 'hot':
            _, cm = self._hot_color_scheme(self.s.flatten(), contrast_param)
        elif self.color_scheme == 'rainbow':
            _, cm = self._rainbow_color_scheme(self.s.flatten(), contrast_param)
        else:
            # Default scheme
            _, cm = contrast(self.s.flatten(), a=contrast_param)
            
        ax.scatter(X.flatten(), Y.flatten(), c=cm, s=5.0)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _winter_color_scheme(self, s, a: float=1):
        """Blue winter theme color scheme"""
        vapor_contrast = 0.6*s # s<1
        vapor_color = plt.cm.Blues(vapor_contrast)
        
        # s > 1
        ice_contrast = (np.exp(2*(s-0.9)*a) - 1) / (np.exp(2*(s-0.9)*a) + 1)
        ice_color = plt.cm.winter(ice_contrast)
        
        y = np.zeros_like(s)
        y[s<1] = vapor_contrast[s<1]
        y[s>=1] = ice_contrast[s>=1]
        
        cm = np.zeros([s.shape[0], 4]) # 4: rgba
        cm[s<1] = vapor_color[s<1]
        cm[s>=1] = ice_color[s>=1]
        
        return y, cm
    
    def _hot_color_scheme(self, s, a: float=1):
        """Hot warm theme color scheme"""
        vapor_contrast = 0.7*s # s<1
        vapor_color = plt.cm.YlOrRd(vapor_contrast)
        
        # s > 1
        ice_contrast = (np.exp(2*(s-0.9)*a) - 1) / (np.exp(2*(s-0.9)*a) + 1)
        ice_color = plt.cm.hot(ice_contrast)
        
        y = np.zeros_like(s)
        y[s<1] = vapor_contrast[s<1]
        y[s>=1] = ice_contrast[s>=1]
        
        cm = np.zeros([s.shape[0], 4]) # 4: rgba
        cm[s<1] = vapor_color[s<1]
        cm[s>=1] = ice_color[s>=1]
        
        return y, cm
    
    def _rainbow_color_scheme(self, s, a: float=1):
        """Rainbow theme color scheme"""
        vapor_contrast = 0.7*s # s<1
        vapor_color = plt.cm.gist_earth(vapor_contrast)
        
        # s > 1
        ice_contrast = (np.exp(2*(s-0.9)*a) - 1) / (np.exp(2*(s-0.9)*a) + 1)
        ice_color = plt.cm.rainbow(ice_contrast)
        
        y = np.zeros_like(s)
        y[s<1] = vapor_contrast[s<1]
        y[s>=1] = ice_contrast[s>=1]
        
        cm = np.zeros([s.shape[0], 4]) # 4: rgba
        cm[s<1] = vapor_color[s<1]
        cm[s>=1] = ice_color[s>=1]
        
        return y, cm