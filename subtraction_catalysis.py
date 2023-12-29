# author: mzhan167
# date: NOV20, 2023
# subtraction and catalysis

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt

# defind a reaction system:
# N + E + F -> NEF + W1
# E + RE -> WE + Re
# WE + FuelE -> E + W2
# F + RF -> WF + Rf
# WF + FuelF -> F + W3

class reaction:
    def __init__(self):

        self.T = 10 # local concentration factor
        self.NEF = 5e-4
        self.NEFW = 5e-11
        self.ERE = 1e-5
        self.FRF = 1e-5
        self.WERe = 5.55e-15
        self.WFRf = 8.19e-15
        self.WEF = 5e-4
        self.WFF = 5e-4
        self.EW = 5.97e-16
        self.FW = 3.74e-14
        
    def rxn(self, t, x):
        
        N = x[0] # N probe
        E = x[1] # E probe
        F = x[2] # F probe
        NEF = x[3]
        W1 = x[4]
        RE = x[5] # RE probe
        WE = x[6]
        Re = x[7]
        FuelE = x[8]
        W2 = x[9]
        RF = x[10] # RF probe
        WF = x[11]
        Rf = x[12]
        FuelF = x[13]
        W3 = x[14]
        
        # rate equation

        dN = -self.NEF*N*E*F +self.NEFW*NEF*W1
        dE = -self.NEF*N*E*F +self.NEFW*NEF*W1 -self.ERE*self.T*E*RE +self.WERe*WE*Re +self.WEF*WE*FuelE -self.EW*E*W2
        dF = -self.NEF*N*E*F +self.NEFW*NEF*W1 -self.FRF*self.T*F*RF +self.WFRf*WF*Rf +self.WFF*WF*FuelF -self.FW*F*W3
        dNEF = self.NEF*N*E*F -self.NEFW*NEF*W1
        dW1 = self.NEF*N*E*F -self.NEFW*NEF*W1
        dRE = -self.ERE*self.T*E*RE +self.WERe*WE*Re
        dWE = self.ERE*self.T*E*RE -self.WERe*WE*Re -self.WEF*WE*FuelE +self.EW*E*W2
        dRe = self.ERE*self.T*E*RE -self.WERe*WE*Re
        dFuelE = -self.WEF*WE*FuelE +self.EW*E*W2
        dW2 = self.WEF*WE*FuelE -self.EW*E*W2
        dRF = -self.FRF*self.T*F*RF +self.WFRf*WF*Rf
        dWF = self.FRF*self.T*F*RF -self.WFRf*WF*Rf -self.WFF*WF*FuelF +self.FW*F*W3
        dRf = self.FRF*self.T*F*RF -self.WFRf*WF*Rf
        dFuelF = -self.WFF*WF*FuelF +self.FW*F*W3
        dW3 = self.WFF*WF*FuelF -self.FW*F*W3
        
        return np.array([ dN, dE, dF, dNEF, dW1, dRE, dWE, dRe, dFuelE, dW2, dRF, dWF, dRf, dFuelF, dW3,], dtype=np.dtype(float))

    
    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau')
    
    