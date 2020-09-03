# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:44:52 2020

@author: gui_c
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from copy import deepcopy
import math

beta = 0.96
gamma = 2
r = 0.01
w = 1
P=np.array(((0.55, 0.45),(0.15, 0.85))) #transition matrix
epsilon = np.array((0.2, 2.0)) #productivity shocks

R = 1+r

grid_min  = 1e-3
grid_max  = 12
grid_size = 1000
step = (grid_max-grid_min)/grid_size

x_grid = np.linspace(grid_min, grid_max, grid_size)
a = x_grid

def u(x):
    return x**(1-gamma)/(1-gamma)

utility = np.repeat(-math.inf,grid_size*grid_size*2)
utility = utility.reshape(grid_size,grid_size,2)
utility = utility.astype(float)

for i in range(0,grid_size):
    for j in range(0,grid_size):
        for k in range(2):
            if (1+r)*a[i] + w*epsilon[k] - a[j] >= 0:
                utility[i,j,k] = u((1+r)*a[i] + w*epsilon[k] - a[j])
            
g = np.repeat(1,grid_size*2)
g = g.reshape(grid_size,2).astype(np.int64)
v  = np.repeat(0,grid_size*2).reshape(grid_size,2).astype(float)
Tv = np.repeat(0,grid_size*2).reshape(grid_size,2).astype(float)

@jit
def update(v,Tv,g):
    for i in range(grid_size):
        for k in range(2):
            vtemp = np.repeat(-math.inf,grid_size)
            for j in range(grid_size):
                vtemp[j] = utility[i,j,k] + beta*(P[k,0]*v[j,0] + P[k,1]*v[j,1]) 
            Tv[i,k] = max(vtemp)
            temp_list = list(vtemp)
            g[i,k]  = temp_list.index(max(temp_list))
    

tol=1e-4
max_iter=500
counter = 0
error = tol + 1

while counter < max_iter and error > tol:
    update(v,Tv,g)
    error = max(max(abs(v[:,1]-Tv[:,1])),max(abs(v[:,0]-Tv[:,0])))
    counter += 1
    print(counter)
    v = deepcopy(Tv)



policy = np.repeat(1,grid_size*2)
policy = policy.reshape(grid_size,2).astype(float)
for i in range(grid_size):
    for k in range(2):
        policy[i,k] = a[g[i,k]]
            

policy_consumption = np.repeat(1,grid_size*2)
policy_consumption = policy_consumption.reshape(grid_size,2).astype(float)

policy_consumption[:,0] = (1+r)*a + w*epsilon[0] - policy[:,0]
policy_consumption[:,1] = (1+r)*a + w*epsilon[1] - policy[:,1]

plt.plot(a.T,v[:,0].T, label = "Low Productivity")
plt.plot(a.T,v[:,1].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Value Function')
plt.title("Value Function")
plt.legend()
plt.show()


plt.plot(a.T,policy[:,0].T, label = "Low Productivity")
plt.plot(a.T,policy[:,1].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Policy for Assets')
plt.title("Policy for Assets")
plt.legend()
plt.show()

plt.plot(a.T,policy_consumption[:,0].T, label = "Low Productivity")
plt.plot(a.T,policy_consumption[:,1].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Policy for Consumption')
plt.title("Policy for Consumption")
plt.legend()
plt.show()