#Program for QPVeta_BB84 

#Importing necessary things for the program
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
#We choose the number of basis as stated in the paper: mu and mv correspond to the number of angles theta and phi, respectively. In the following, we recover QPVeta_BB84 from the extention to such a protocol to m bases
mu=2
mv=1                       
m=mu*mv - (mv - 1)
ket0=np.array([[1],[0]])
ket1=np.array([[0],[1]])

def angle(alpha):
    return np.array([acos(2*(floor(alpha/mv)+1)/mu-1),pi/mv*(alpha % mv)])
def ket0tp(a):
    return cos(angle(a)[0]/2)*ket0+cmath.exp(I*angle(a)[1])*sin(angle(a)[0]/2)*ket1
def ket1tp(a):
    return sin(angle(a)[0]/2)*ket0-cmath.exp(I*angle(a)[1])*cos(angle(a)[0]/2)*ket1
def P(x):
    return np.matmul(x,x.transpose().conjugate())
def V0(a):
    return P(ket0tp(a))
def V1(a):
    return P(ket1tp(a))
V=[]
for a in range(m):
    V.append(np.array([V0(m-1-a),V1(m-1-a)]))
   
def norm(M):
    return np.real(np.linalg.norm(M, ord=2))

#We define the projectors and the level of the NPA hierarcy
numA=3
numB=3
level = 1
A_configuration = [3]*m
B_configuration = [3]*m

#We deine the predicte of the game
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
r = []
#We find the upper boung given by the NPA hierarchy for different p_err and we apply the corresponding inequalities 
for p_err in np.concatenate([np.arange(0,0.12,0.005),np.arange(0.12,0.15,0.005)]):
    inequalities = []
    inequalities = inequalities + [xi - mVar(A[x],a)*mVar(B[x],b) for x in range(m) for a in range(numA) for b in range(numB) if a!=b]
        
    
    for x in range(m):
        for y in range(m):    
            inequalities += [p_err * (4*xi+mVar(A[x],0)*mVar(B[x],0) + mVar(A[x],1)*mVar(B[x],1) + mVar(A[y],0)*mVar(B[y],0) + mVar(A[y],1)*mVar(B[y],1))+8*xi-sum((2-norm(V[x][a]+V[y][b]))*mVar(A[x],a)*mVar(B[y],b) for a in [0,1] for b in [0,1])]
          
       
    
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
    
    sdp.solve(solver='Mosek')
    #We store the solution together with its corresponding value of p_err
    r.append((p_err, -sdp.primal))
    print(-sdp.primal)
print(r)





                       
                       
