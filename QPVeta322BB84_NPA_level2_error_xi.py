#Program for QPVeta_(3,2,2)-BB84 

#Importing necessary things for the program
import csv
import math
import cmath
from math import cos, pi, acos, floor, exp, sin
import numpy as np
import mosek
import matplotlib.pyplot as plt
from sympy.core import S, expand
                       
from ncpol2sdpa import *


#Fix the parameter xi, which is the probability that the attackers can respond different
#NOTE: For the original case, when no error is allowed, set xi=0
xi=0.001
I=cmath.sqrt(-1)
#We start with the level 1 of the NPA hierarcy
level = 1
#We choose the number of basis as stated in the paper: mu and mv correspond to the number of angles theta and phi, respectively. 
mu=2
mv=2                       
m=mu*mv - (mv - 1)
ket0=np.array([[1],[0]])
ket1=np.array([[0],[1]])

#We define the projective measurements that the verifiers do
def angle(alpha):
    return np.array([acos(2*(floor(alpha/mv)+1)/mu-1),pi/mv*(alpha % mv)])
def ket0tp(a):
    return cos(angle(a)[0]/2)*ket0+cmath.exp(I*angle(a)[1])*sin(angle(a)[0]/2)*ket1
def ket1tp(a):
    return sin(angle(a)[0]/2)*ket0-cmath.exp(I*angle(a)[1])*cos(angle(a)[0]/2)*ket1
def P(x):
    return np.matmul(x,x.transpose().conjugate())
def braket(phi,psi):
    return np.matmul(phi.transpose().conjugate(),psi)
def ketbra(phi,psi):
    return np.matmul(phi,psi.transpose().conjugate())
def V0(a):
    return P(ket0tp(a))
def V1(a):
    return P(ket1tp(a))
V=[]
for a in range(m):
    V.append(np.array([V0(m-1-a),V1(m-1-a)]))
   
def norm(M):
    return np.real(np.linalg.norm(M, ord=2))

#We define the coefficients alpha and beta and their corresponding conjugates
def beta(x,y,th,th1):
    if x==0 and y == 0:
        return braket(ket0tp(th), ket0tp(th1))
    elif x==0 and y == 1:
        return braket(ket0tp(th), ket1tp(th1)) 
    elif x==1 and y == 0:
        return braket(ket1tp(th), ket0tp(th1))
    elif x==1 and y == 1:
        return braket(ket1tp(th), ket1tp(th1))
def cbeta(x,y,th,th1):
    return beta(x,y,th,th1).conjugate()

def alpha(x,y,th,th1):
    if x==0 and y == 0:
        return braket(ket0tp(th1), ket0tp(th))
    elif x==0 and y == 1:
        return braket(ket0tp(th1), ket1tp(th)) 
    elif x==1 and y == 0:
        return braket(ket1tp(th1), ket0tp(th))
    elif x==1 and y == 1:
        return braket(ket1tp(th1), ket1tp(th))
def calpha(x,y,th,th1):
    return alpha(x,y,th,th1).conjugate()

#We define the projector measurements
numA=3
numB=3

A_configuration = [3]*m
B_configuration = [3]*m

#We define the predicate of the game
def predicate(x,y,a,b):
    return (x==y and a == b and a != 2)

#To make it computationally efficient, taking into account that we work with projectors, we consider one of the projectors the identity minus the others
def mVar(mList, i):
    nOutcomes = len(mList)+1
    if i == nOutcomes-1:
        return S.One - sum(mList[j] for j in range(nOutcomes-1))
    else:
        return mList[i]
    


A = generate_measurements(A_configuration, 'A')
B = generate_measurements(B_configuration, 'B')
monomial_substitutions = projective_measurement_constraints(A, B)
# Input distribution over which we optimize predicate
inputs = tuple((x,x) for x in range(m))
distribution = {xy : S.One/len(inputs) for xy in inputs}
objective = S.Zero
for ((x,y), p) in distribution.items():
    for a in range(3):
        for b in range(3):
            if(predicate(x,y,a,b)):
                objective = p * mVar(A[x],a) * mVar(B[y],b)+objective
objective = -expand(objective)
sdp = SdpRelaxation(flatten([A, B]), verbose=0)
#We create empty strings where we will store the data
r = []
r2 = []
#We find the upper boung given by the NPA hierarchy for different p_err and we apply the corresponding inequalities 
for p_err in np.arange(0,0.25,0.005):
    inequalities = []
    inequalities = inequalities + [xi - mVar(A[x],a)*mVar(B[x],b) for x in range(m) for a in range(numA) for b in range(numB) if a!=b]
    for x in range(m):
        for y in range(m):    
            inequalities += [p_err * (4*xi+mVar(A[x],0)*mVar(B[x],0) + mVar(A[x],1)*mVar(B[x],1) + mVar(A[y],0)*mVar(B[y],0) + mVar(A[y],1)*mVar(B[y],1))+8*xi-sum((2-norm(V[x][a]+V[y][b]))*mVar(A[x],a)*mVar(B[y],b) for a in [0,1] for b in [0,1])]
        
    for th in range(m):
        for th1 in range(m): 
            betalist=[]
            for i in [0,1]:
                for j in [0,1]:
                    betalist.append(abs(beta(i,j,th,th1))**2)
            alphalist=[]
            for i in [0,1]:
                for j in [0,1]:
                    alphalist.append(abs(alpha(i,j,th,th1))**2)
            inequalities += [norm((1+abs(beta(0,0,th,th1))**2)*V[th][0]+(1+abs(alpha(0,0,th,th1))**2)*V[th1][0] 
                                  +beta(0,0,th,th1)*cbeta(1,0,th,th1)*ketbra(ket0tp(th),ket1tp(th))+beta(1,0,th,th1)*cbeta(0,0,th,th1)*ketbra(ket1tp(th),ket0tp(th)) 
                                  +alpha(0,0,th,th1)*calpha(1,0,th,th1)*ketbra(ket0tp(th1),ket1tp(th1))+alpha(1,0,th,th1)*calpha(0,0,th,th1)*ketbra(ket1tp(th1),ket0tp(th1)))*mVar(A[th],0)*mVar(B[th1],0)
                             +norm((1+abs(beta(0,1,th,th1))**2)*V[th][0]+(1+abs(alpha(1,0,th,th1))**2)*V[th1][1]
                                    +beta(0,1,th,th1)*cbeta(1,1,th,th1)*ketbra(ket0tp(th),ket1tp(th))+beta(1,1,th,th1)*cbeta(0,1,th,th1)*ketbra(ket1tp(th),ket0tp(th))
                                    +alpha(0,0,th,th1)*calpha(1,0,th,th1)*ketbra(ket0tp(th1),ket1tp(th1))+alpha(1,0,th,th1)*calpha(0,0,th,th1)*ketbra(ket1tp(th1),ket0tp(th1)))*mVar(A[th],0)*mVar(B[th1],1)
                             +norm((1+abs(beta(1,0,th,th1))**2)*V[th][1]+(1+abs(alpha(0,1,th,th1))**2)*V[th1][0]
                                   +beta(0,0,th,th1)*cbeta(1,0,th,th1)*ketbra(ket0tp(th),ket1tp(th))+beta(1,0,th,th1)*cbeta(0,0,th,th1)*ketbra(ket1tp(th),ket0tp(th))
                                   +alpha(0,1,th,th1)*calpha(1,1,th,th1)*ketbra(ket0tp(th1),ket1tp(th1))+alpha(1,1,th,th1)*calpha(0,1,th,th1)*ketbra(ket1tp(th1),ket0tp(th1)))*mVar(A[th],1)*mVar(B[th1],0)
                             +norm((1+abs(beta(1,1,th,th1))**2)*V[th][1]+(1+abs(alpha(1,1,th,th1))**2)*V[th1][1]
                                   +beta(0,1,th,th1)*cbeta(1,1,th,th1)*ketbra(ket0tp(th),ket1tp(th))+beta(1,1,th,th1)*cbeta(0,1,th,th1)*ketbra(ket1tp(th),ket0tp(th))
                                   +alpha(0,1,th,th1)*calpha(1,1,th,th1)*ketbra(ket0tp(th1),ket1tp(th1))+alpha(1,1,th,th1)*calpha(0,1,th,th1)*ketbra(ket1tp(th1),ket0tp(th1)))*mVar(A[th],1)*mVar(B[th1],1)
                             +p_err*((2+max(betalist)[0,0])*sum(mVar(A[th],a)*mVar(B[th],a) for a in [0,1])+(2+max(alphalist)[0,0])*sum(mVar(A[th1],a)*mVar(B[th1],a) for a in [0,1]))
                             -4*sum(mVar(A[x],a)*mVar(B[y],b) for a in [0,1] for b in [0,1])]   
    equalities = []
 #We add the second level of the hierarchy
    AA = [Ai*Aj for Ai in flatten(A) for Aj in flatten(A)]
    AB0=[Ai*Bj for Ai in flatten(A) for Bj in flatten(B)]
    BB=[Bi*Bj for Bi in flatten(B) for Bj in flatten(B)]
    AB=AA+AB0+BB
        
    sdp.get_relaxation(level, objective=objective,
                                     substitutions=monomial_substitutions,
                                     inequalities=inequalities,
                                     equalities=equalities,
                                     extramonomials=AB)
        
     #We solve the SDP  
    sdp.solve(solver='mosek')
      #We store the solution together with its corresponding value of p_err
    r.append(p_err)
    r2.append( -sdp.primal)

print(r)
print(r2)
