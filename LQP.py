#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:14:51 2019

@author: Harsha Vaddireddy
"""
#%%
# The CD equation with extrenal forcing(f(x)) and control input(u(x) 
# is solved using LQR and gradeint descent alogrithm.
import numpy as np
from matplotlib import pyplot as plt

## Constants 
n = 101
gamma = 10
ic = 0.1
h = 0.01
#### Construct Q matrix of 198 x 198 (2N x 2N)
q11 = np.eye(n-2,n-2)
q12 = np.zeros([n-2,n-2])
q13 = np.zeros([n-2,n-2])
q14 = np.eye(n-2,n-2)*gamma
q = np.block([[q11,q12],[q13,q14]])
##### Construct A matrix of 99 x 198 (N x 2N)  
c1 = 1.0/h**2
c2 = -2.0/h**2 - 1.0/h
c3 = 1.0/h**2 + 1.0/h
c11 = np.full(n-3,c1)
c12 = np.full(n-2,c2)
c13 = np.full(n-3,c3)
a11 = np.diag(c11,-1)
a12 = np.diag(c12, 0)
a13 = np.diag(c13, 1)
a1 = a11 + a12 + a13
a2  = np.eye(n-2,n-2)*-1.0
a  =  np.concatenate((a1, a2), axis=1)
#%% Construct b matirx
x = np.linspace(0,1,n)
f  = np.sin(np.pi*x)
th_d = np.cos(np.pi*x)
b = np.zeros([n-2,1])
for i in range(n-2):   
    b[i] = f[i+1] - c1*th_d[i] -c2*th_d[i+1] -c3*th_d[i+2]
b[0] = b[0] + c1*th_d[0]
b[n-3] = b[n-3] + c3*th_d[100]




#%%
## Construct the solution
qinv = np.linalg.inv(q)
atrp = np.transpose(a)

t1 = np.matmul(qinv, atrp)

t2 = np.matmul(a, qinv)

t2 = np.linalg.inv(np.matmul(t2, atrp))

z = np.matmul(t1,t2)
z = np.matmul(z,b)

ob = np.matmul(np.matmul(np.transpose(z),q), z)/2

u = z[n-2:,0]
u = np.append(u,[0])
u = np.append([0],u)

theta = z[0:99,] + th_d.reshape(101,1)[1:100,]
theta = np.append(theta,[0]).reshape(100,1)
theta = np.append([0],theta).reshape(101,1)


#%% Numerical solution
z_num = np.full((2*n-4,1),ic)
lam = np.full((n-2,1),ic)

alpha = 0.0001
beta = 0.00001
tol = 5
i = 0
#for i in range(1):
while tol >= 10e-6: 
      z_temp = z_num
      z_num = z_num - alpha*(np.matmul(q,z_num) - np.matmul(atrp, lam))
      lam = lam + beta*(b - np.matmul(a,z_num))
      i = i+1
      print(i)
      tol  = np.linalg.norm((z_temp - z_num))
      
ob_num = np.matmul(np.matmul(np.transpose(z_num),q), z_num)/2
      
u_num = z_num[n-2:,0]
u_num = np.append(u_num,[0])
u_num = np.append([0],u_num)

th_num = z_num[0:n-2,]+ th_d.reshape(101,1)[1:100]      
th_num = np.append(th_num,[0]).reshape(100,1)
th_num = np.append([0],th_num).reshape(101,1)  
    
#%%  Numerical solution with box constraint      
z_num1 = np.full((2*n-4,1),ic)
lam1 = np.full((n-2,1),ic)

alpha = 0.0001
beta = 0.00001
tol1 = 5      
      
k = 0
#for i in range(1):
while tol1 >= 10e-6:
      z_temp1 = z_num1
      z_num1 = z_num1 - alpha*(np.matmul(q,z_num1) - np.matmul(atrp, lam1))
      lam1 = lam1 + beta*(b - np.matmul(a,z_num1))
      for j in range(n-2):
          if z_num1[n-2+j,0] < -1:
             z_num1[n-2+j,0] == -1
          elif z_num1[n-2+j,0] > 1:
               z_num1[n-2+j,0] == 1
      tol1  = np.linalg.norm((z_temp1 - z_num1))
      k = k+1
      print(k)      
ob_num1 = np.matmul(np.matmul(np.transpose(z_num1),q), z_num1)/2
      
u_num1 = z_num1[n-2:,0]
u_num1 = np.append(u_num1,[0])
u_num1 = np.append([0],u_num1)

th_num1 = z_num1[0:n-2,]+ th_d.reshape(101,1)[1:100,]    
th_num1 = np.append(th_num1,[0]).reshape(100,1)
th_num1 = np.append([0],th_num1).reshape(101,1)    

#%% Post Processing 1
plt.figure()
plt.plot(x, theta, 'r' , linewidth=2.0, label='Analytical')
plt.plot(x, th_num, '--b' , linewidth=2.0, label='Numerical')
plt.xlabel('$x$',fontsize=18)
plt.ylabel(r'$\theta(^\circ c)$',fontsize=18)
plt.legend( prop={'size': 15})
plt.grid(color='black', linestyle='dotted')
plt.xlim(0,1)
plt.show()
#plt.savefig('a11.pdf', bbox_inches='tight')


plt.figure()
plt.plot(x, theta,color= 'r' , linewidth=2.0, label='Analytical')
plt.plot(x, th_num1, '--b' , linewidth=2.0, label='Numerical (Box constraint)')
plt.xlabel('x',fontsize=18)
plt.ylabel(r'$\theta(^\circ c)$',fontsize=18)
plt.legend( prop={'size': 15})
plt.grid(color='black', linestyle='dotted')
plt.xlim(0,1)
plt.show()
#plt.savefig('a12.pdf', bbox_inches='tight')

#%% Post Processing 2

plt.figure()
plt.plot(x, u, 'r' , linewidth=2.0, label='Analytical')
plt.plot(x, u_num, '--b' , linewidth=2.0, label='Numerical')
plt.xlabel('$x$',fontsize=18)
plt.ylabel(r'u(x)',fontsize=18)
plt.legend( prop={'size': 15})
plt.grid(color='black', linestyle='dotted')
plt.xlim(0,1)
#plt.ylim(,0)
plt.show()
#plt.savefig('a13.pdf', bbox_inches='tight')


plt.figure()
plt.plot(x, u,color= 'r' , linewidth=2.0, label='Analytical')
plt.plot(x, u_num1, '--b' , linewidth=2.0, label='Numerical (Box constraint)')
plt.xlabel('x',fontsize=18)
plt.ylabel(r'u(x)',fontsize=18)
plt.legend( prop={'size': 15})
plt.grid(color='black', linestyle='dotted')
plt.xlim(0,1)
#plt.ylim(-0.09,0)
plt.show()
#plt.savefig('a14.pdf', bbox_inches='tight')
#

 