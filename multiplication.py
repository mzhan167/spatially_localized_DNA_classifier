# author: mzhan167
# date: DEC18, 2023
# multiplication

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt

# defind a reaction system:   
# Input + Helper + Origami -> W*Weight + waste1
# Weight + M probe <-> aW + waste2
# aW + S <-> fS + E

class reaction:
    def __init__(self):

        self.T = 1
        self.W = 0
        self.I = 5e-6
        self.M = 3e-3
        self.aW = 5e-11
        self.S = 3e-3
        self.fS = 5e-11
        
    def rxn(self, t, x):
        I = x[0] # Input
        H = x[1] # Helper
        O = x[2] # DNA-IC CLA
        W = x[3] # Weight
        W1 = x[4] # Waste
        M = x[5] # M probe
        aW = x[6]
        W2 = x[7]
        S = x[8] # S probe
        fS = x[9]
        E = x[10] # E probe
        
        # rate equation
        
        dI = -self.I*I*H*O
        dH = -self.I*I*H*O
        dO = -self.I*I*H*O
        dW = self.I*I*H*(4*self.W*O) -self.T*self.M*(W/(4*self.W))*(M/self.W) +self.T*self.aW*(aW/(4*self.T))*(W2/self.W)
        dW1 = self.I*I*H*O
        dM = -self.T*self.M*(W/(4*self.W))*(M/self.W) +self.T*self.aW*(aW/(4*self.W))*(W2/self.W)
        daW = self.T*self.M*(W/(4*self.W))*(M/self.W) -self.T*self.aW*(aW/(4*self.W))*(W2/self.W) -self.T*self.S*aW*S +self.T*self.fS*fS*E
        dW2 = self.T*self.M*(W/(4*self.W))*(M/self.W) -self.T*self.aW*(aW/(4*self.W))*(W2/self.W)
        dS = -self.T*self.S*aW*S +self.T*self.fS*fS*E
        dfS = self.T*self.S*aW*S -self.T*self.fS*fS*E
        dE = self.T*self.S*aW*S -self.T*self.fS*fS*E
        
        return np.array([ dI, dH, dO, dW, dW1, dM, daW, dW2, dS, dfS, dE,], dtype=np.dtype(float))

    
    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau')
    
    