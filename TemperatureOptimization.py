import numpy as np
import pele.thermodynamics as pele
import sys
from math import *
from pele.optimize import mylbfgs as opt 
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy import interpolate

class TemperatureEst(object):

    def __init__(self, Energies, Entropies):

        self.kappa = 87*0.5
#         self.kappa = (3*38-6)*0.5
        self.ExactGammaAve = False

        self.ptarget = 0.22
        self.g = 1.18331
        self.Ti=0.0125#*g*g


        data = np.loadtxt("min.data") 
        self.E = Energies
        self.S = Entropies


    def calcPw(self,T):

        F = self.E - T * self.S
#         F=pele.free_energy(self.E,self.pgo,self.vib,T,self.kappa,h=1)
        Fmin=np.min(F)

        P=np.exp(-(F-Fmin)/T)
        
        P = P / np.sum(P)
        
        return P
        
    def calcCostfromlist(self,Tc):

        E = self.E
        kappa = self.kappa
        Ti = self.Ti

        dat = self.calcPaccestfromlist(Tc)
        p = dat[0]
        pprime = dat[1]

        C = 0.5 * ((p-self.ptarget)**2) 
        Cprime = (p-self.ptarget) * pprime

        return C, Cprime

    def calcPaccestfromlist(self,Tc):

#         Ts = Tc[0]
        # calc Pacc
        Ts = Tc
        # calc paccest
        deltaT = Ts-self.Ti
        Vavei = self.calcEavefromlist(Ts)
        Vavec = self.calcEavefromlist(Ts)
        deltaVave = Vavec-Vavei
        Cvidat = self.calcCvfrominterp(self.Ti)
        Cvi = Cvidat[0]
        Cvcdat = self.calcCvfrominterp(Ts)
        Cvc = Cvcdat[0]
        dCvcdT = Cvcdat[1]

        delta = self.kappa*deltaT + deltaVave
        r = sqrt(Cvi*self.Ti**2 + Cvc*Ts**2)
        Pacc = erfc(delta/(sqrt(2)*r)) 

        # calc dPacc/dT 
        dPaccdT = 1. - (delta)/(2*r**2) * (2*Ts + Ts**2 * dCvcdT/Cvc)
        dPaccdT = dPaccdT * Cvc/(sqrt(2)*r) * (- exp(-delta**2/(2*(r**2))))

        return Pacc, dPaccdT

    def calcPaccest(self,Tc):

        Ti = self.Ti
        E = self.E
        if self.ExactGammaAve == True : kappa = self.kappa-1
        else: kappa = self.kappa
        print kappa, "kappa"
        deltaT = Tc - Ti
        gamma = Tc/Ti
        Pacc_est = 0.0
        pacc = np.zeros((len(E),len(E)))

        Pi = self.calcPw(Ti)
        Pc = self.calcPw(Tc)

        for w in range(len(E)):
            for o in range(len(E)):
                deltaV =E[o]-E[w]

                chi0=sqrt(kappa)*(gamma-1)*sqrt(1./(gamma**2+1))
                chi = chi0 * (1+deltaV/(kappa*deltaT))
                pacc[w,o]=erfc(chi/sqrt(2))

                Pacc_est = Pacc_est + Pi[w]*Pc[o]*pacc[w,o]

        return Pacc_est

    def calcPaccest_ww(self,Tlow,Tc,w,o):

        Ti = Tlow
        E = self.E
        kappa = self.kappa
        deltaT = Tc - Ti
        gamma = Tc/Ti
        Pacc_est = 0.0
        pacc = np.zeros((len(E),len(E)))

        Pi = self.calcPw(Ti)
        Pc = self.calcPw(Tc)

        deltaV =E[o]-E[w]

        chi0=sqrt(kappa)*(gamma-1)*sqrt(1./(gamma**2+1))
        chi = chi0 * (1+deltaV/(kappa*deltaT))
        pacc[w,o]=erfc(chi/sqrt(2))

        #return Pi[w]*Pc[o]*pacc[w,o]
        return pacc[w,o]

        

    def pot(self,Tc):
        """ Return cost function and derivative:
            C(P_acc(T)) = (P_acc(T) - target)**2
        """
        E = self.E
        kappa = self.kappa
        Ti = self.Ti

        paccest = self.calcPaccest(Tc)
        Pi = self.calcPw(Ti)
        Pc = self.calcPw(Tc)
        Energy = 0.5 * ((paccest-self.ptarget)**2) #+ 1000.0 * (1-np.sign(Tc-Ti))

        deltaT = Tc - Ti
        gamma = Tc/Ti
            
        dPaccdT = 0.0
            
        pacc = np.zeros(shape=(len(E),len(E)))

        # average V for Tc replica needed for derivative below
        aveVc = self.calcEavefromlist(Tc)

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

                dPcdT = Pc[o]*(aveVc-self.kappa*Tc - E[o])  / Tc**2  

                dPaccdT = dPaccdT + Pi[w]*pacc[w,o]*dPcdT
                

        gradE = (paccest-self.ptarget) * dPaccdT

        return Energy, gradE

    def TempOptfromlist(self,Tlow):

        Tc = Tlow*(1.+0.001)
        self.Ti = Tlow
        g = self.g
        
#         Pi = self.calcPw(Tlow)
#         print "I am here", Tc,Tlow
#         ret = fmin_l_bfgs_b(self.pot,[Tc],bounds=[(Tlow+0.0001,g*g*Tlow)],iprint=1,factr=1e6)
        print "TempOptfromlist> Tc", Tc
#         ret = fmin_l_bfgs_b(self.calcCostfromlist,[Tc],bounds=[(Tlow+0.0001,g*g*Tlow)],iprint=0,factr=1e2)
        a = Tc
        ret = fmin_l_bfgs_b(self.calcCostfromlist,[a],bounds=[(Tlow+0.0001,g*g*Tlow)],iprint=1,factr=1e2)

#         return ret[0]
        return ret

    def TempOpt(self,Ti):

        Tc = Ti*(1.+0.001)
        self.Ti = Ti
        
        ret = fmin_l_bfgs_b(self.pot,[Tc],bounds=[(Ti+0.0001,self.g*self.g*Ti)],iprint=1,factr=1e2)

        return ret

    def paccest_list(self):
        
        temps=np.loadtxt("temperatures")
        pest = np.ndarray(shape=len(temps))
        for i in range(len(temps)-1):
            self.Ti = temps[i] 
            Tc = temps[i+1]
            #print self.Ti, self.calcPaccest(Tc)
            pest[i] = self.calcPaccest(Tc)

        return (temps,pest)

    def iterate(self):

        N=10
        T0=0.0125
    #     T0=0.2597933583E-01
        Tcmin=np.zeros(N)
        Tcmin[0]=T0

        for i in range(N-1): 
            self.Ti=Tcmin[i]
            tmin=self.TempOpt(Tcmin[i])
            Tcmin[i+1]=tmin
            print Tcmin[i], Tcmin[i+1]

        print Tcmin

    """
        alast=0
        Ti=T0
        Tc=Ti+0.001
        while(Tc<Ti+0.020):
        #while(Tc<Ttrue*5):
            alist = pot(Tc)
            a=alist[0]
            print a,alist[1],(a-alast)/0.0001,Tc#,(Tc-Ttrue)/Ttrue
            Tc = Tc+0.0001
            alast = a
    """

def main():
    p = TemperatureEst()
    out = p.paccest_list()
    p.ExactGammaAve= True
    out2 = p.paccest_list()
    print out[0],out2[0]
if "__name__==main()":
    main()