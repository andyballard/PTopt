import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from math import sqrt, erfc, pi, exp

class TSequenceOptimizer(object):

    def __init__(self, Energies, Entropies, Ndof, ptarget=0.22):
        """ This class predicts a optimal sequence of Temperatures for Parallel Tempering simulation
            based upon a database of configurational minima of the energy landscape.
            Inputs:
                Energies : array of energies for the configurational minima
                Entropies : array of entropies for configurational minima 
                Ndof : number of system degrees of freedom (3*Natoms - 6) usually
                ptarget : target PT acceptance rate
        """
        self.kappa = Ndof * 0.5

        self.ExactGammaAve = False

        self.ptarget = ptarget
        self.g = 1.18331

        self.E = Energies
        self.S = Entropies


    def calcMinimaWeights(self,T):
        """ Calculate equilibrium occupation probabilities for configurational minima
            Inputs : 
                T = Temperature of interest
            returns : 
                normalized array of weights
        """
        F = self.E - T * self.S

        Fmin=np.min(F)
        P=np.exp(-(F-Fmin)/T)
        
        P = P / np.sum(P)

        return P
        
    def calcPacc(self,Ti,Tc):
        """ estimate PT acceptance probability
            Inputs: 
            Pair of temperatures, Ti, Tc, with Ti<Tc
        """
        E = self.E
        if self.ExactGammaAve == True : kappa = self.kappa-1
        else: kappa = self.kappa

        deltaT = Tc - Ti
        gamma = Tc/Ti
        Pacc_est = 0.0
        pacc = np.zeros((len(E),len(E)))

        Pi = self.calcMinimaWeights(Ti)
        Pc = self.calcMinimaWeights(Tc)

        for w in range(len(E)):
            for o in range(len(E)):
                deltaV =E[o]-E[w]

                chi0=sqrt(kappa)*(gamma-1)*sqrt(1./(gamma**2+1))
                chi = chi0 * (1+deltaV/(kappa*deltaT))
                pacc[w,o]=erfc(chi/sqrt(2))
#                 print w,o, pacc[w,o], deltaV,kappa,gamma,deltaT

#                 print w,o, pacc[w,o], chi
#                 print "chi: ", chi, chi0, deltaV, kappa, deltaT

                Pacc_est = Pacc_est + Pi[w]*Pc[o]*pacc[w,o]
        
#         print "in paccest : ", Ti, Tc, kappa, Pi[10], Pc[10], Pacc_est

        return Pacc_est

    def costAndGradient(self,Ti,Tc):
        """ Return cost function and derivative for optimizaiton of T sequence:
            Returns : 
                (Cost, dC/dT) : tuple 
                Cost = C(P_acc(T)) = 0.5 * (P_acc(T) - target)**2
                dC/dT = (P_acc(T)-target) * dP_acc/dT
        """
        E = self.E
        kappa = self.kappa

        paccest = self.calcPacc(Ti,Tc)
        Pi = self.calcMinimaWeights(Ti)
        Pc = self.calcMinimaWeights(Tc)
        Cost = 0.5 * ((paccest-self.ptarget)**2) #+ 1000.0 * (1-np.sign(Tc-Ti))

        deltaT = Tc - Ti
        gamma = Tc/Ti
            
        dPaccdT = 0.0
            
        pacc = np.zeros(shape=(len(E),len(E)))

        # average V for Tc replica needed for derivative below
        aveVc = np.dot(self.E, Pc)+ self.kappa * Tc
        
        for w in range(len(E)):
            for o in range(len(E)):

                deltaV =E[o]-E[w]

                chi0=sqrt(kappa)*(gamma-1)*sqrt(1./(gamma**2+1))
                chi = chi0 * (1+deltaV/(kappa*deltaT))

                dchi0dT = (sqrt(kappa) / Ti) * ((gamma+1)/(gamma**2+1)**(1.5))
                dchidT = dchi0dT*(1+deltaV/(kappa*deltaT)) - chi0*deltaV/(kappa*deltaT**2)

                dPaccdT = dPaccdT + Pi[w]*Pc[o]*(-(2./sqrt(pi))*exp(-(chi**2)/2)) * dchidT/sqrt(2)

                #add term that calculates well probabilities
                pacc[w,o]=erfc(chi/sqrt(2))

                # TODO: simplify below expression and redefine aveVc properly
                dPcdT = Pc[o]*(aveVc-self.kappa*Tc - E[o])  / Tc**2  

                dPaccdT = dPaccdT + Pi[w]*pacc[w,o]*dPcdT
                

        gradE = (paccest-self.ptarget) * dPaccdT
#         print "iteration: ", Tc, Cost, gradE

        return Cost, gradE

    def findNextTemperature(self,Ti):
        """ 
        Use L-BFGS to find next temperature in sequence
        """
        Tc = Ti*(1.+0.001)
        
        """ This function to optimize is the cost function, with Ti held fixed"""
        cost_for_Tc_optimization = lambda x : self.costAndGradient(Ti, x)
        
        Tbest, cost_value, info = fmin_l_bfgs_b(cost_for_Tc_optimization,[Tc],
                            bounds=[(Ti+0.0001,self.g*self.g*Ti)],
                            iprint=0,
                            factr=1e2)

        return Tbest

    def run(self, N=10, T0=0.0125):
        """ Find temperature sequence.
            Inputs : 
                N : Number of temperatures desired
                T0 : Starting temperature
        """
    #     T0=0.2597933583E-01
        Tcmin=np.zeros(N)
        Tcmin[0]=T0

        for i in range(N-1): 
            Ti=Tcmin[i]
            tmin=self.findNextTemperature(Ti)
            Tcmin[i+1]=tmin
            print Tcmin[i], Tcmin[i+1]

        print Tcmin

        return Tcmin

def load_gmin_data_file(filename):
    
    Ndof = 3.*31-6
    E, f, pg = np.loadtxt("min.data",usecols=(0,1,2), unpack=True)
    S = -0.5*f - np.log(pg) 

    return E, S


def test():
    """ Example for LJ31 system """
    Ndof = 3*31 - 6
    
#     E, S = load_gmin_data_file("example_data/min.data")
    """ Load entries of Energy, Entropy values for configurational minima of LJ31 cluster"""
    E, S = np.loadtxt("example_data/LJ31.EandS",unpack=True)
    
    """ run optimizer"""
    p = TSequenceOptimizer(E, S, Ndof)
    optimal_temperatures = p.run(N=5, T0=0.0125)
    
    print "Optimal temperature sequence: "
    for o in optimal_temperatures:
        print o
    
if "__name__== main()":
    test()