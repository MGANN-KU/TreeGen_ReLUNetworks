# -*- coding: utf-8 -*-
"""
Created on Sun November 24,2024 at 22:28:00

@author: Ghafoor
"""

import numpy as np
#Deletion in Tree edit distance

# parameters
eps = 1e-7
C = 1e5
B = 1e3

# Input
d = 3
m = 5
t = [3, 5, 2+m, 2, 4, 4+m, 2+m, 4, 4+m, 3+m]
n = len(t)//2

# First layer is input layer, Given as
x = [1, 3, 0]
X = x  
print("Input:")
print("d:", d)
print("m:", m)
print("t:", t)
print("x:", X)
###################################################################################       
###################################################################################
#                    Padding 2d number of B at the end of E(T)=t
###################################################################################
###################################################################################
# Create t'=t by padding 2*d instances of B to t
t1 = t + [B] * (2 * d)

# print("Original list (t):", t)
# print("Padded list (t'):", t1)
###################################################################################
# To construct weight matrix for second layer/first hidden layer is L1=W1*X+B1
W1 = []

## (rho11, rho 12 to solve delta(x_j, x_k))
for j in range(1, d+1):
  for k in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for l in range(1, d+1):
         w = 0
         if j == l and j!=k:
                w = 1/eps
         if k == l and j!=k:
                 w=-1/eps
         temp_row.append(w)
      W1.append(temp_row)
#####
## (varrho1, varrho 2 to solve delta(x_j, x_k))
for j in range(1, d+1):
  for k in range(1, d+1):
    for q in range(2): 
      temp_row = []
      for l in range(1, d+1):
         w = 0
         if j == l and j!=k:
                w = -1/eps
         if k == l and j!=k:
                 w= 1/eps
         temp_row.append(w)
      W1.append(temp_row)
#####
## (x_j as identity map)
for k in range(1, d+1): 
    temp_row = []
    for j in range(1, d+1):
        w = 0
        if k == j:
            w = 1
        temp_row.append(w)
    W1.append(temp_row)
#####
for l in range(1, 2*n+2*d+1):  # (eta nodes)
    for q in range(2):
        temp_row = []
        for i in range(1, d+1):
            w = 0
            temp_row.append(w)
        W1.append(temp_row)
############
# for i in W1:
#       print(i)
############
# Bias matrix for second layer/first hidden layer
############
B1 = []
# bias matrix for rho 11, 12

for j in range(1, d+1):
  for k in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
         b = 0
         if q==0:
             b=1
         temp_row.append(b)
      B1.append(temp_row)

# bias matrix for varrho 11, 12

for j in range(1, d+1):
  for k in range(1, d+1):
    for q in range(2):
      temp_row = []
      for i in range(1):
         b = 0
         if q==0:
             b=1
         temp_row.append(b)
      B1.append(temp_row) 
      
# top(bias matrix for x_j)

for k in range(1, d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B1.append(temp_row)

# bottom(bias matrix for eta nodes)

for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for j in range(1):
            b = 0
            if q == 0:
                b = ((m-t1[i-1]) / eps)+1
            else:
                b = (m-t1[i-1]) / eps
            temp_row.append(b)
        B1.append(temp_row)
############
# for i in B1:
#     print(i)
##################################
L1 = []  # eta nodes
for i in range(len(W1)):
    temp_row = []
    L1_i_entry = np.maximum((np.dot(W1[i], X)+B1[i]), 0)
    L1.append(L1_i_entry)
############
# print('Printing eta nodes of second layer/first hidden layer')
# for i in L1:
#     print(i)
############
# To construct weight matrix for third layer/second hidden layer is L2=W2*L1+B2
W2 = []
# (for omega_jk that represents delta(x_j,x_k)
A1 = []
for j in range(1, d+1):
  for k in range(1, d+1):
    temp_row = []
    for i in range(1, d+1):# rho 11, 12 nodes
      for l in range(1, d+1):
        for q in range(2):
            w = 0
            if j == i and k==l:
                if q==0:
                  w = 1
                else:
                    w=-1
            temp_row.append(w)
    A1.append(temp_row)
#####
A2 = []
for j in range(1, d+1):
  for k in range(1, d+1):
    temp_row = []
    for i in range(1, d+1):# varrho 11, 12 nodes
      for l in range(1, d+1):
        for q in range(2):
            w = 0
            if j == i and k==l:
                if q==0:
                  w = 1
                else:
                    w=-1
            temp_row.append(w)
    A2.append(temp_row)
##########
A3 = []
for j in range(1, d+1):
  for k in range(1, d+1):
    temp_row = []
    for i in range(1, d+1):# x_j nodes
            w = 0
            temp_row.append(w)
    A3.append(temp_row)
##########
A4 = []
for j in range(1, d+1):
  for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):
        for q in range(2):# eta nodes
            w = 0
            temp_row.append(w)
    A4.append(temp_row)
##########
for i in range(len(A1)):
    concatenated_row = A1[i] + A2[i]+ A3[i] + A4[i]
    W2.append(concatenated_row)
# print("weight matrix for third layer/second hidden layer")
# print(W2)
#######################
# (for x_j nodes as identity map)
A5 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, d+1):# rho 11, 12 nodes
      for l in range(1, d+1):
        for q in range(2):
            w = 0
            temp_row.append(w)
    A5.append(temp_row)
#####
A6 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, d+1):# varrho 11, 12 nodes
      for l in range(1, d+1):
        for q in range(2):
            w = 0
            temp_row.append(w)
    A6.append(temp_row)
#####
A7 = []
for k in range(1, d+1):
    temp_row = []
    for j in range(1, d+1):
        w = 0
        if k == j:
            w = 1
        temp_row.append(w)
    A7.append(temp_row)
#####
A8 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            temp_row.append(w)
    A8.append(temp_row)
##########
for i in range(len(A5)):
    concatenated_row = A5[i] + A6[i] + A7[i] + A8[i]
    W2.append(concatenated_row)
# print("weight matrix for third layer/second hidden layer")
# print(W2)
#######################
# for alpha 11, alpha 12 nodes(These nodes calculates delta(p_i ,0)of eq.2)
A9 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1, d+1):# rho 11, 12 nodes
          for k in range(1, d+1):
            for q in range(2):
              w = 0
              temp_row.append(w)
        A9.append(temp_row)
#####
A10 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1, d+1):# varrho 11, 12 nodes
          for k in range(1, d+1):
            for q in range(2):
              w = 0
              temp_row.append(w)
        A10.append(temp_row)
#####
A11 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1, d+1):
            w = 0
            temp_row.append(w)
        A11.append(temp_row)
#####
A12 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                if i == l:
                    if q == 0:
                        w = 1 / eps
                    else:  # q = 1
                        w = -1 / eps
                temp_row.append(w)
        A12.append(temp_row)
##########
for i in range(len(A9)):
    concatenated_row = A9[i] + A10[i] + A11[i] + A12[i]
    W2.append(concatenated_row)
##############
# (for beta 11, beta 12 nodes)(These nodes calculates delta(p_i ,0)of eq.2)
A13 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1, d+1):# rho 11, 12 nodes
          for k in range(1, d+1):
            for q in range(2):
               w = 0
               temp_row.append(w)
        A13.append(temp_row)
#####
A14 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1, d+1):# varrho 11, 12 nodes
          for k in range(1, d+1):
            for q in range(2):
               w = 0
               temp_row.append(w)
        A14.append(temp_row)
#####
A15 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for j in range(1,d+1):
            w = 0
            temp_row.append(w)
        A15.append(temp_row)
#####
A16 = []
for i in range(1, 2*n+2*d+1):
    for p in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                if i == l:
                    if q == 0:
                        w = -1 / eps
                    else:  # q = 1
                        w = 1 / eps
                temp_row.append(w)
        A16.append(temp_row)
# print(A16)
##########
for i in range(len(A13)):
    concatenated_row = A13[i] + A14[i] + A15[i] + A16[i]
    W2.append(concatenated_row)
##############
# (for gamma1 nodes)(These nodes calculates summation p_k of eq.2)
A17 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# rho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A17.append(temp_row)
#####
A18 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# varrho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A18.append(temp_row)
#####
A19 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):
        w = 0
        temp_row.append(w)
    A19.append(temp_row)
#####
A20 = []
for l in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            if i <= l:
                if q == 0:
                    w = 1
                else:  # q = 1
                    w = -1
            temp_row.append(w)
    A20.append(temp_row)
# print(A8)
##########
for i in range(len(A17)):
    concatenated_row = A17[i] + A18[i] + A19[i] + A20[i]
    W2.append(concatenated_row)
#########
# (for Tau1 nodes)(These nodes calculates r_i of eq.6)
A21 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# rho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A21.append(temp_row)
#####
A22 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# varrho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A22.append(temp_row)
#####
A23 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):
        w = 0
        temp_row.append(w)
    A23.append(temp_row)
#####
A24 = []
for l in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            if i == l:
                if q == 0:
                    w = t1[l-1]
                else:  # q = 1
                    w = -t1[l-1]
            temp_row.append(w)
    A24.append(temp_row)
##########
for i in range(len(A21)):
    concatenated_row = A21[i] + A22[i] + A23[i] + A24[i]
    W2.append(concatenated_row)
##############
# (for Gamma nodes)(These nodes calculates summation s_i of eq.7)
A25 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# rho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A25.append(temp_row)
#####
A26 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# varrho 11, 12 nodes
      for k in range(1, d+1):
        for q in range(2):
           w = 0
           temp_row.append(w)
    A26.append(temp_row)
#####
A27 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):
        w = 0
        temp_row.append(w)
    A27.append(temp_row)
#####
A28 = []
for l in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            if i == l:
                if q == 0:
                    w = -t1[l-1]
                else:  # q = 1
                    w = t1[l-1]
            temp_row.append(w)
    A28.append(temp_row)
##########
for i in range(len(A25)):
    concatenated_row = A25[i] + A26[i] + A27[i] + A28[i] 
    W2.append(concatenated_row)
#####################
# Bias matrix for third layer/second hidden layer

B2 = []
# bias matrix for omega_jk

for j in range(1, d+1):
    for k in range(1, d+1):
       temp_row = []
       for l in range(1):
          b = -1
          temp_row.append(b)
       B2.append(temp_row)
    
# bias matrix for x_j

for k in range(1, d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B2.append(temp_row)
    
# bias matrix for alpha 11,12 nodes

for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for j in range(1):
            b = 0
            if q == 0:
                b = 1
            temp_row.append(b)
        B2.append(temp_row)

# bias matrix for beta 11,12 nodes

for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for j in range(1):
            b = 0
            if q == 0:
                b = 1
            temp_row.append(b)
        B2.append(temp_row)    

# bias matrix for gamma1 nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B2.append(temp_row)

# bias matrix for Tau1 nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B2.append(temp_row)

# bias matrix for Gamma nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = t1[i-1]
        temp_row.append(b)
    B2.append(temp_row)

#################################
L2 = []  # gamma, alpha11,12, beta11,12, tau1, Gamma1, tau nodes
for i in range(len(W2)):
    temp_row = []
    L2_i_entry = np.maximum((np.dot(W2[i], L1)+B2[i]), 0)
    L2.append(L2_i_entry)
    # print('this is index i:', i)
    # print('this is the value L2[i]:', L2_i_entry)
##################################
# print('Printing gamma1, alpha11,12, beta11,12, tau1, Gamma1, tau nodes nodes for third layer/second hidden layer')
# for i in L2:
#     print(i)
###################
# To construct weight matrix for fourth layer/third hidden layer is L3=W3*L2+B3
W3 = []
# (x'_j that removes the repeted input of eq. 21)
A33 = []
for l in range(1,d+1):
    temp_row = []
    for j in range(1, d+1):
        for k in range(1, d+1):
           w = 0
           if l == j and k<j:
              w = -C
           temp_row.append(w)
    A33.append(temp_row)
##########
A34 = []
for l in range(1,d+1):
    temp_row = []
    for j in range(1, d+1):
           w = 0
           if l == j:
              w = 1
           temp_row.append(w)
    A34.append(temp_row)
##########
A35 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # alpha 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A35.append(temp_row)
##########
A36 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # beta 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A36.append(temp_row)
##########
A37 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # gamma1
        w = 0
        temp_row.append(w)
    A37.append(temp_row)
##########
A38 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Tau1
        w = 0
        temp_row.append(w)
    A38.append(temp_row)
##########
A39 = []
for k in range(1, d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Gamma
        w = 0
        temp_row.append(w)
    A39.append(temp_row)
#########
for i in range(len(A33)):
    concatenated_row = A33[i] + A34[i] + A35[i] + A36[i] + A37[i] + A38[i] + A39[i]
    W3.append(concatenated_row)
#######################
# (for gamma2)(These nodes calculates p'_i of eq.2)
A41 = []
for l in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):  # omega nodes
      for k in range(1, d+1):
        w = 0
        temp_row.append(w)
    A41.append(temp_row)
    
##########
A42 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):  # x_j nodes
        w = 0
        temp_row.append(w)
    A42.append(temp_row)
    
##########
A43 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # alpha11,12
        for q in range(2):
            w = 0
            if k == i:
                if q == 0:
                    w = -C
                else:
                    w = C
            temp_row.append(w)
    A43.append(temp_row)
##########
A44 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # beta11,12
        for q in range(2):
            w = 0
            if k == i:
                if q == 0:
                    w = -C
                else:
                    w = C
            temp_row.append(w)
    A44.append(temp_row)
#########

A45 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # gamma1
        w = 0
        if k == i:
            w = 1
        temp_row.append(w)
    A45.append(temp_row)
#####
    
A46 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Tau1
        w = 0
        temp_row.append(w)
    A46.append(temp_row)
##########
A47 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Gamma1
        w = 0
        temp_row.append(w)
    A47.append(temp_row)
##########
for i in range(len(A41)):
    concatenated_row = A41[i] + A42[i] + A43[i] + A44[i] + A45[i] + A46[i] + A47[i]
    W3.append(concatenated_row)
#######################
# (for mu11,12)(These nodes calculates delta(s_i ,r_l +m) of eq.2)
A49 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # omega nodes
              for j in range(1, d+1):
                 w = 0
                 temp_row.append(w)
            A49.append(temp_row)
#####
A50 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # x_j nodes
                w = 0
                temp_row.append(w)
            A50.append(temp_row)
#####
A51 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # alpha11,12
                for p in range(2):
                    w = 0
                    temp_row.append(w)
            A51.append(temp_row)
##########
A52 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for i in range(1, 2*n+2*d+1):  # beta11,12
                for p in range(2):
                    w = 0
                    temp_row.append(w)
            A52.append(temp_row)
##########
A53 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # gamma1
                w = 0
                temp_row.append(w)
            A53.append(temp_row)
##########
A54 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # Tau1
                w = 0
                if k == l:
                    w = -1/eps
                temp_row.append(w)
            A54.append(temp_row)
##########
A55 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # Gamma
                w = 0
                if k == i:
                    w = 1/eps
                temp_row.append(w)
            A55.append(temp_row)
##########
for i in range(len(A49)):
    concatenated_row = A49[i] + A50[i] +A51[i] + A52[i] + A53[i] + A54[i] + A55[i] 
    W3.append(concatenated_row)
#######################
# (for lambda11,12)(These nodes calculates delta(s_i ,r_l +m) of eq.8)
A57 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # omega nodes
              for k in range(1, d+1):
                w = 0
                temp_row.append(w)
            A57.append(temp_row)
#####
A58 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1, d+1):  # x_j nodes
                w = 0
                temp_row.append(w)
            A58.append(temp_row)
#####
A59 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # alpha11,12
                for p in range(2):
                    w = 0
                    temp_row.append(w)
            A59.append(temp_row)
##########
A60 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # beta11,12
                for p in range(2):
                    w = 0
                    temp_row.append(w)
            A60.append(temp_row)
##########
A61 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # gamma1
                w = 0
                temp_row.append(w)
            A61.append(temp_row)
##########
A62 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # Tau1
                w = 0
                if k == l:
                    w = 1/eps
                temp_row.append(w)
            A62.append(temp_row)
##########
A63 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):  # Gamma
                w = 0
                if k == i:
                    w = -1/eps
                temp_row.append(w)
            A63.append(temp_row)
##########
for i in range(len(A57)):
    concatenated_row = A57[i] + A58[i] + A59[i] + A60[i] + A61[i] + A62[i] + A63[i] 
    W3.append(concatenated_row)
#########
# (for mu and mu prime nodes to find H(sj -1))
A65 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, d+1):  # omega nodes
      for k in range(1, d+1):
        w = 0
        temp_row.append(w)
    A65.append(temp_row)
#####
A66 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, d+1):  # x_j nodes
        w = 0
        temp_row.append(w)
    A66.append(temp_row)
#####
A67 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # alpha 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A67.append(temp_row)
##########
A68 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # beta 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A68.append(temp_row)
##########
A69 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # gamma1
        w = 0
        temp_row.append(w)
    A69.append(temp_row)
##########
A70 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Tau1
        w = 0
        temp_row.append(w)
    A70.append(temp_row)
##########
A71 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Gamma
        w = 0
        if k == i:
            w = 1/eps
        temp_row.append(w)
    A71.append(temp_row)
##########
for i in range(len(A65)):
    concatenated_row = A65[i] + A66[i] + A67[i] + A68[i] + A69[i] + A70[i]+ A71[i] 
    W3.append(concatenated_row)
#########
# (for lambda and lambda prime nodes to find H(rj -1))
A73 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, d+1):  # omega nodes
      for k in range(1, d+1):
        w = 0
        temp_row.append(w)
    A73.append(temp_row)
#####
A74 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for j in range(1, d+1):  # x_j nodes
        w = 0
        temp_row.append(w)
    A74.append(temp_row)
#####
A75 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # alpha 11,12
        for p in range(2):
            w = 0
            temp_row.append(w)
    A75.append(temp_row)
##########
A76 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # beta 11,12
        for p in range(2):
            w = 0
            temp_row.append(w)
    A76.append(temp_row)
##########
A77 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # gamma1
        w = 0
        temp_row.append(w)
    A77.append(temp_row)
##########
A78 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Tau1
        w = 0
        temp_row.append(w)
    A78.append(temp_row)
##########
A79 = []
for k in range(1, 2*n+2*d+1):
  for q in range(2):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Gamma
        w = 0
        if k == i:
            w = 1/eps
        temp_row.append(w)
    A79.append(temp_row)
##########
for i in range(len(A73)):
    concatenated_row = A73[i] + A74[i] + A75[i] + A76[i] + A77[i]+ A78[i] + A79[i]  
    W3.append(concatenated_row)
# print("weight matrix for fourth layer/third hidden layer")
# print(W3)
#########
# (for Gamma as identity map)
A81 = []
for l in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):  # omega nodes
      for k in range(1, d+1):
        w = 0
        temp_row.append(w)
    A81.append(temp_row)
#####
A82 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):  # x_j nodes
        w = 0
        temp_row.append(w)
    A82.append(temp_row)
#####
A83 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # alpha 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A83.append(temp_row)
##########
A84 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # beta 11,12
        for q in range(2):
            w = 0
            temp_row.append(w)
    A84.append(temp_row)
##########
A85 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # gamma1
        w = 0
        temp_row.append(w)
    A85.append(temp_row)
##########
A86 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Tau1
        w = 0
        temp_row.append(w)
    A86.append(temp_row)
##########
A87 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for i in range(1, 2*n+2*d+1):  # Gamma
        w = 0
        if k == i:
            w = 1
        temp_row.append(w)
    A87.append(temp_row)
##########
for i in range(len(A81)):
    concatenated_row = A81[i] + A82[i] + A83[i] + A84[i] + A85[i] + A86[i] + A87[i] 
    W3.append(concatenated_row)
# print("weight matrix for fourth layer/third hidden layer")
# print(W3)
#####################
# Bias matrix for fourth layer/third hidden layer

B3 = []
# bias matrix for x'_j

for k in range(1, d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B3.append(temp_row)

# bias matrix for gamma2 nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = C
        temp_row.append(b)
    B3.append(temp_row)

# bias matrix for mu 11,12 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q == 0:
                    b = (-m/eps)+1
                else:
                    b = -m/eps
                temp_row.append(b)
            B3.append(temp_row)

# bias matrix for lambda 11,12 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q == 0:
                    b = (m/eps)+1
                else:
                    b = m/eps
                temp_row.append(b)
            B3.append(temp_row)

# bias matrix for mu and mu prime nodes

for l in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q == 0:
                    b = -(1/eps)+1
                else:
                    b = -1/eps
                temp_row.append(b)
            B3.append(temp_row)

# bias matrix for lambda and lambda prime nodes

for l in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                b = 0
                if q == 0:
                    b = -(1/eps)+1
                else:
                    b = -1/eps
                temp_row.append(b)
            B3.append(temp_row) 
            
# bias matrix for Gamma nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B3.append(temp_row)
# for i in B3:
#     print(i)
##################################
L3 = []  # omega prime, gamma2 , mu11,12, lambda11,12 , mu, mu prime, lambda, lambda prime nodes
for i in range(len(W3)):
    temp_row = []
    L3_i_entry = np.maximum((np.dot(W3[i], L2)+B3[i]), 0)
    L3.append(L3_i_entry)
##################################
# print('Printing omega prime, gamma2 , mu11,12, lambda11,12, mu, mu prime, lambda, lambda prime nodes for fourth layer/third hidden layer')
# for i in L3:
#     print(i)
###################
# To construct weight matrix for fifth layer/fouth hidden layer is L4=W4*L3+B4
W4 = []
# omega prime(correspond to x'_j) as identity map
D1 = []
for j in range(1, d+1):
            temp_row = []
            for k in range(1, d+1):# x'_j nodes
                w = 0
                if k == j:
                    w = 1
                temp_row.append(w)
            D1.append(temp_row)

##########
D2 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# gamma2 nodes
                w = 0
                temp_row.append(w)
            D2.append(temp_row)
##########
D3 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # Mu11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D3.append(temp_row)
##########
D4 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # lambda 11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D4.append(temp_row)
##########
D5 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# mu and mu prime nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D5.append(temp_row)
##########
D6 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# lambda and lambda prime nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D6.append(temp_row)
##########
D7 = []
for j in range(1, d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# Gamma nodes
                w = 0
                temp_row.append(w)
            D7.append(temp_row)
##########
for i in range(len(D1)):
    concatenated_row = D1[i] + D2[i] + D3[i] + D4[i] + D5[i] + D6[i] + D7[i]
    W4.append(concatenated_row)
#######################
# gamma2 as identity map
D8 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
            D8.append(temp_row)

##########
D9 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# gamma2 nodes
                w = 0
                if l == j:
                    w = 1
                temp_row.append(w)
            D9.append(temp_row)
##########
D10 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # Mu11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D10.append(temp_row)
##########
D11 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # lambda 11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D11.append(temp_row)
##########
D12 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# mu and mu prime nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D12.append(temp_row)
##########
D13 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# lambda and lambda prime nodes
              for q in range(2):
                w = 0
                temp_row.append(w)
            D13.append(temp_row)
##########
D14 = []
for j in range(1, 2*n+2*d+1):
            temp_row = []
            for l in range(1, 2*n+2*d+1):# Gamma nodes
                w = 0
                temp_row.append(w)
            D14.append(temp_row)
##########
for i in range(len(D8)):
    concatenated_row = D8[i] + D9[i] + D10[i]+D11[i] + D12[i] + D13[i] + D14[i]
    W4.append(concatenated_row)
#######################
# mu 21,22 (These nodes calculates delta(p'_i ,0) of eq.3)
D15 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
            D15.append(temp_row)

##########
D16 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):
                w = 0
                if i == l:
                    w = 1/eps
                temp_row.append(w)
            D16.append(temp_row)
##########
D17 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # Mu11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D17.append(temp_row)
##########
D18 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # lambda 11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D18.append(temp_row)
##########
D19 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):
                for p in range(2):
                  w = 0
                  temp_row.append(w)
            D19.append(temp_row)
##########
D20 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):
                for p in range(2):
                  w = 0
                  temp_row.append(w)
            D20.append(temp_row)
##########
D21 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
            D21.append(temp_row)
##########
for i in range(len(D15)):
    concatenated_row = D15[i] + D16[i] + D17[i] + D18[i] + D19[i]+ D20[i]+ D21[i]
    W4.append(concatenated_row)
#######################
# lambda 21,22(These nodes calculates delta(p'_i ,0) of eq.3)
D22 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
            D22.append(temp_row)

##########
D23 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):
                w = 0
                if i == l:
                    w = -1/eps
                temp_row.append(w)
            D23.append(temp_row)
##########
D24 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # Mu11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D24.append(temp_row)
##########
D25 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for l in range(1, 2*n+2*d+1):  # lambda 11,12
                for k in range(1, 2*n+2*d+1):
                    for p in range(2):
                        w = 0
                        temp_row.append(w)
            D25.append(temp_row)
##########
D26 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                  w = 0
                  temp_row.append(w)
            D26.append(temp_row)
##########
D27 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                  w = 0
                  temp_row.append(w)
            D27.append(temp_row)
##########
D28 = []
for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
            D28.append(temp_row)
##########
for i in range(len(D22)):
    concatenated_row = D22[i] + D23[i] + D24[i] + D25[i] + D26[i] + D27[i]+ D28[i]
    W4.append(concatenated_row)
#######################
# psi 1(These nodes correspond delta(s_i ,r_l +m) of eq.8)
D29 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, d+1):# x'_j nodes
            w = 0
            temp_row.append(w)
        D29.append(temp_row)

##########
D30 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D30.append(temp_row)
##########
D31 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # Mu11,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    if  i!=1 and i>l and k>j and j == l and i == k:
                        if p == 0:
                            w = 1
                        else:
                            w = -1
                    temp_row.append(w)
        D31.append(temp_row)
##########
D32 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # lambda11,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    if  i!=1 and i>l and k>j and j == l and i == k:
                        if p == 0:
                            w = 1
                        else:
                            w = -1
                    temp_row.append(w)
        D32.append(temp_row)
##########
D33 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
              w = 0
              temp_row.append(w)
        D33.append(temp_row)
##########
D34 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
              w = 0
              temp_row.append(w)
        D34.append(temp_row)
##########
D35 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D35.append(temp_row)
##########
for i in range(len(D29)):
    concatenated_row = D29[i] + D30[i] + D31[i] + D32[i] + D33[i] + D34[i]+ D35[i] 
    W4.append(concatenated_row)
#######################
# psi 2(These nodes correspond H(s_i -1) of eq.8)
D36 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, d+1):# x'_j nodes
            w = 0
            temp_row.append(w)
        D36.append(temp_row)

##########
D37 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D37.append(temp_row)
##########
D38 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # Mu11,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    temp_row.append(w)
        D38.append(temp_row)
##########
D39 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # lambda,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    temp_row.append(w)
        D39.append(temp_row)
##########
D40 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):# mu and mu prime
            for q in range(2):
              w = 0
              if  i!=1 and i==k:
                  if q == 0:
                      w = 1
                  else:
                      w = -1
              temp_row.append(w)
        D40.append(temp_row)
##########
D41 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):# lambda and lambda prime
            for q in range(2):
              w = 0
              temp_row.append(w)
        D41.append(temp_row)
##########
D42 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D42.append(temp_row)
##########
for i in range(len(D36)):
    concatenated_row = D36[i] + D37[i] + D38[i] + D39[i] + D40[i] + D41[i]+ D42[i] 
    W4.append(concatenated_row)
#########
# psi 3(These nodes correspond H(r_i -1) of eq.8)
D43 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, d+1):# x'_j nodes
            w = 0
            temp_row.append(w)
        D43.append(temp_row)

##########
D44 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D44.append(temp_row)
##########
D45 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # Mu11,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    temp_row.append(w)
        D45.append(temp_row)
##########
D46 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):  # lambda,12
            for k in range(1, 2*n+2*d+1):
                for p in range(2):
                    w = 0
                    temp_row.append(w)
        D46.append(temp_row)
##########
D47 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):# mu and mu prime
            for q in range(2):
              w = 0
              temp_row.append(w)
        D47.append(temp_row)
##########
D48 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):# lambda and lambda prime
            for q in range(2):
              w = 0
              if i==k:
                  if q == 0:
                      w = 1
                  else:
                      w = -1
              temp_row.append(w)
        D48.append(temp_row)
##########
D49 = []
for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
        D49.append(temp_row)
##########
for i in range(len(D43)):
    concatenated_row = D43[i] + D44[i] + D45[i] + D46[i] + D47[i] + D48[i]+ D49[i]
    W4.append(concatenated_row)
#########
# Gamma
D50 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, d+1):# x'_j nodes
        w = 0
        temp_row.append(w)
    D50.append(temp_row)

##########
D51 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
        w = 0
        temp_row.append(w)
    D51.append(temp_row)
##########
D52 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):  # Mu11,12
        for j in range(1, 2*n+2*d+1):
            for p in range(2):
                w = 0
                temp_row.append(w)
    D52.append(temp_row)
##########
D53 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):  # lambda 12
        for j in range(1, 2*n+2*d+1):
            for p in range(2):
                w = 0
                temp_row.append(w)
    D53.append(temp_row)
##########
D54 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
      for q in range(2):
        w = 0
        temp_row.append(w)
    D54.append(temp_row)
##########
D55 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
      for q in range(2):
        w = 0
        temp_row.append(w)
    D55.append(temp_row)
##########
D56 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
        w = 0
        if i == k:
            w = 1
        temp_row.append(w)
    D56.append(temp_row)
##########
for i in range(len(D50)):
    concatenated_row = D50[i] + D51[i] + D52[i] + D53[i] + D54[i] + D55[i]+ D56[i] 
    W4.append(concatenated_row)
#####################
# print("weight matrix for fifth layer/fourth hidden layer")
# print(W4)
#####################
# Bias matrix for fifth layer/fourth hidden layer

B4 = []

# bias matrix for # x'_j nodes

for j in range(1, d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row)

# bias matrix for gamma 2

for j in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1):
                b = 0
                temp_row.append(b)
            B4.append(temp_row) 
            
# bias matrix for mu 21,22

for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B4.append(temp_row)

# bias matrix for lambda 21,22

for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1):
                b = 0
                if q == 0:
                    b = 1
                temp_row.append(b)
            B4.append(temp_row)

# bias matrix for psi 1 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1):
            b=0
            if i!=1 and i>l:
              b = -1
            temp_row.append(b)
        B4.append(temp_row)

# bias matrix for psi 2 nodes

for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1):
            b = 0
            temp_row.append(b)
        B4.append(temp_row)

# bias matrix for psi 3 nodes

for l in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1):
            b = 0
            temp_row.append(b)
        B4.append(temp_row)

# bias matrix for Gamma nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B4.append(temp_row)
      
###################################
# print('Printing B4')
# for i in B4:
#     print(i)
###################################
L4 = []  # alpha 21,22, beta 21,22, psi1,psi2,psi3 nodes
for i in range(len(W4)):
    temp_row = []
    L4_i_entry = np.maximum((np.dot(W4[i], L3)+B4[i]), 0)
    L4.append(L4_i_entry)
    # print('this is index i:', i)
    # print('this is the value L4[i]:', L4_i_entry)
###################################
# print('Printing mu 21,22, lambda 21,22, psi1,psi2,psi3 nodes for fifth layer/fouth hidden hidden layer')
# for i in L4:
#     print(i)
##################
# To construct weight matrix for sixth layer/fifth hidden layer is L5=W5*L4+B5
W5 = []
# # x'_j nodes as identity map
E1 = []
for j in range(1, d+1):
       temp_row = []
       for k in range(1, d+1):
                w = 0
                if j==k:
                    w=1
                temp_row.append(w)
       E1.append(temp_row)
##########
E2 = []
for j in range(1, d+1):
       temp_row = []
       for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E2.append(temp_row)

##########
E3 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
             for q in range(2):
                w = 0
                temp_row.append(w)
       E3.append(temp_row)
##########
E4 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for q in range(2):
                w = 0
                temp_row.append(w)
       E4.append(temp_row)
##########
E5 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E5.append(temp_row)
##########
E6 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E6.append(temp_row)
##########
E7 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E7.append(temp_row)
##########
E8 = []
for j in range(1, d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E8.append(temp_row)

##########
for i in range(len(E1)):
    concatenated_row = E1[i] + E2[i] + E3[i] + E4[i] + E5[i] + E6[i] + E7[i] + E8[i] 
    W5.append(concatenated_row)
#######################
# gamma 2 nodes as identity map
E9 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
       E9.append(temp_row)
##########
E10 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, 2*n+2*d+1):
                w = 0
                if j==k:
                    w=1
                temp_row.append(w)
       E10.append(temp_row)

##########
E11 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
             for q in range(2):
                w = 0
                temp_row.append(w)
       E11.append(temp_row)
##########
E12 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for q in range(2):
                w = 0
                temp_row.append(w)
       E12.append(temp_row)
##########
E13 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E13.append(temp_row)
##########
E14 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E14.append(temp_row)
##########
E15 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E15.append(temp_row)
##########
E16 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E16.append(temp_row)

##########
for i in range(len(E9)):
    concatenated_row = E9[i] + E10[i] + E11[i] + E12[i] + E13[i] + E14[i] + E15[i] + E16[i]
    W5.append(concatenated_row)
#######################
# eta 3 nodes (These nodes correspond \max(2n - C(1-\delta(p'_i, 0) ) of eq.3)
E17 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
       E17.append(temp_row)
##########
E18 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E18.append(temp_row)

##########
E19 = []
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
             for q in range(2):
                w = 0
                if i==l:
                    if q==0:
                      w=C
                    else:
                        w=-C
                temp_row.append(w)
       E19.append(temp_row)
##########
E20 = []
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for q in range(2):
                w = 0
                if i==l:
                    if q==0:
                      w=C
                    else:
                        w=-C
                temp_row.append(w)
       E20.append(temp_row)
##########
E21 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
          for k in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E21.append(temp_row)
##########
E22 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E22.append(temp_row)
##########
E23 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E23.append(temp_row)
##########
E24 = []
for j in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
       E24.append(temp_row)

##########
for i in range(len(E17)):
    concatenated_row = E17[i] + E18[i] + E19[i] + E20[i] + E21[i] + E22[i] + E23[i] + E24[i]
    W5.append(concatenated_row)
#######################
# omega 1(corresponding to eq 8 of Thm 2)
E25 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, d+1):# x'_j nodes
                    w = 0
                    temp_row.append(w)
        E25.append(temp_row)
##########
E26 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
        E26.append(temp_row)
##########
E27 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
                    w = 0
                    temp_row.append(w)
        E27.append(temp_row)
##########
E28 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
                    w = 0
                    temp_row.append(w)
        E28.append(temp_row)
##########
E29 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):# psi1
            for k in range(1, 2*n+2*d+1):
               w = 0
               if i!=1 and k!=1 and i>l and k>j and i == k and l == j:
                   w=1
               temp_row.append(w)
        E29.append(temp_row)
##########
E30 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):
                    w = 0
                    if i!=1 and i>l and j >= l+1 and j<i:
                        w = -C
                    temp_row.append(w)
        E30.append(temp_row)
##########
E31 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1, 2*n+2*d+1):
                w = 0
                if i!=1 and i>l and j >= l+1 and j < i:
                    w = C
                temp_row.append(w)
        E31.append(temp_row)
##########
E32 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
        E32.append(temp_row)
##########
for i in range(len(E25)):
    concatenated_row = E25[i] + E26[i] + E27[i] + E28[i] + E29[i] +E30[i]  + E31[i] +E32[i]  
    W5.append(concatenated_row)
#######################
# Gamma as identity map
E33 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, d+1):# x'_j nodes
                w = 0
                temp_row.append(w)
    E33.append(temp_row)
##########
E34 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1, 2*n+2*d+1):
                w = 0
                temp_row.append(w)
    E34.append(temp_row)

##########
E35 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                temp_row.append(w)
    E35.append(temp_row)
##########
E36 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                temp_row.append(w)
    E36.append(temp_row)
##########
E37 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
        for l in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
    E37.append(temp_row)
##########
E38 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
    E38.append(temp_row)
##########
E39 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
    E39.append(temp_row)
##########
E40 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
        w = 0
        if i == k:
            w = 1
        temp_row.append(w)
    E40.append(temp_row)
##########
for i in range(len(E33)):
    concatenated_row =  E33[i] + E34[i] + E35[i] + E36[i] + E37[i]+ E38[i] + E39[i]+ E40[i]
    W5.append(concatenated_row)
# #####################
# print("weight matrix for sixth layer/fifth hidden layer")
# print(W5)
# #####################
# #Bias matrix for sixth layer/fifth hidden layer

B5 = []

# bias matrix for x'_j nodes

for j in range(1, d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B5.append(temp_row)

# bias matrix for gamma 2

for j in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1):
        b = 0
        temp_row.append(b)
      B5.append(temp_row)  

# bias matrix for eta 3

for j in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1):
        b = -2*C+2*n
        temp_row.append(b)
      B5.append(temp_row)   
      
# bias matrix for omega 1 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for j in range(1):
            b = 0
            temp_row.append(b)
        B5.append(temp_row)

# bias matrix for Gamma nodes

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for j in range(1):
        b = 0
        temp_row.append(b)
    B5.append(temp_row)
##################################
L5 = []  # eta 3 and omega 1(eq8) nodes
for i in range(len(W5)):
    temp_row = []
    L5_i_entry = np.maximum((np.dot(W5[i], L4)+B5[i]), 0)
    L5.append(L5_i_entry)
##################################
# print('Printing eta 3 and omega 1(eq8) nodes nodes for sixth layer/fifth hidden layer')
# for i in L5:
#     print(i)
###################
# To construct weight matrix for seventh layer/sixth hidden layer is L6=W6*L5+B6
W6 = []
# alpha 21, 22 nodes(These nodes correspond q_ji of eq.4)
F1 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, d+1):# x'_j nodes
                  w = 0
                  if k == j:
                      w = -1/eps
                  temp_row.append(w)
           F1.append(temp_row)

##########
F2 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# gamma 2 nodes
                  w = 0
                  if k == l:
                      w = 1/eps
                  temp_row.append(w)
           F2.append(temp_row)

##########
F3 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# eta 3 nodes
                  w = 0
                  if k == l:
                      w = 1/eps
                  temp_row.append(w)
           F3.append(temp_row)

##########
F4 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1): # omega1 nodes
               for q in range(1, 2*n+2*d+1):
                  w = 0
                  temp_row.append(w)
            F4.append(temp_row)
##########
F5 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# Gamma nodes
                  w = 0
                  temp_row.append(w)
           F5.append(temp_row)
##########
for i in range(len(F1)):
    concatenated_row = F1[i] + F2[i] + F3[i] + F4[i] + F5[i] 
    W6.append(concatenated_row)
# print(W6)
######################
# beta 21, 22 nodes(These nodes correspond q_ji of eq.4)
F6 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
            temp_row = []
            for k in range(1, d+1):# x'_j nodes
                  w = 0
                  if k == j:
                      w = 1/eps
                  temp_row.append(w)
            F6.append(temp_row)

##########
F7 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# gamma 2 nodes
                  w = 0
                  if k == l:
                      w = -1/eps
                  temp_row.append(w)
           F7.append(temp_row)

##########
F8 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# eta 3 nodes
                  w = 0
                  if k == l:
                      w = -1/eps
                  temp_row.append(w)
           F8.append(temp_row)

##########
F9 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1): # omega1 nodes
               for q in range(1, 2*n+2*d+1):
                  w = 0
                  temp_row.append(w)
            F9.append(temp_row)
##########
F10 = []
for j in range(1, d+1):
    for l in range(1, 2*n+2*d+1): 
         for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# Gamma nodes
                  w = 0
                  temp_row.append(w)
           F10.append(temp_row)
##########
for i in range(len(F6)):
    concatenated_row = F6[i] + F7[i] + F8[i] + F9[i] + F10[i]
    W6.append(concatenated_row)
# print(W6)
######################
# alpha 31,32 nodes(These nodes correspond delta(v_li, 1) of eq.9)
F11 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, d+1):# x'_j nodes
                  w = 0
                  temp_row.append(w)
           F11.append(temp_row)
##########
F12 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, 2*n+2*d+1):# gamma 2 nodes
                  w = 0
                  temp_row.append(w)
           F12.append(temp_row)
##########
F13 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, 2*n+2*d+1):# eta 3 nodes
                  w = 0
                  temp_row.append(w)
           F13.append(temp_row)

##########
F14 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1): # omega1 nodes
               for j in range(1, 2*n+2*d+1):
                  w = 0
                  if j==i and k==l:
                      w=1/eps
                  temp_row.append(w)
            F14.append(temp_row)
##########
F15 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# Gamma nodes
                  w = 0
                  temp_row.append(w)
           F15.append(temp_row)
##########
for i in range(len(F11)):
    concatenated_row = F11[i] + F12[i] + F13[i] + F14[i] + F15[i] 
    W6.append(concatenated_row)
#######################
# beta 31,32 nodes(These nodes correspond delta(v_li, 1) of eq.9)
F16 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, d+1):# x'_j nodes
                  w = 0
                  temp_row.append(w)
           F16.append(temp_row)
##########
F17 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, 2*n+2*d+1):# gamma 2 nodes
                  w = 0
                  temp_row.append(w)
           F17.append(temp_row)
##########
F18 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):  
           temp_row = []
           for k in range(1, 2*n+2*d+1):# eta 3 nodes
                  w = 0
                  temp_row.append(w)
           F18.append(temp_row)
##########
F19 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):
            temp_row = []
            for k in range(1, 2*n+2*d+1): # omega1 nodes
               for j in range(1, 2*n+2*d+1):
                  w = 0
                  if l==k and j==i:
                      w=-1/eps
                  temp_row.append(w)
            F19.append(temp_row)
##########
F20 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      for p in range(2):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# Gamma nodes
                  w = 0
                  temp_row.append(w)
           F20.append(temp_row)
##########
for i in range(len(F16)):
    concatenated_row = F16[i] + F17[i] + F18[i] + F19[i] + F20[i]  
    W6.append(concatenated_row)
#######################
# Gamma nodes as identity map
F21 = []
for l in range(1, 2*n+2*d+1):  
           temp_row = []
           for k in range(1, d+1):# x'_j nodes
                  w = 0
                  temp_row.append(w)
           F21.append(temp_row)
##########
F22 = []
for l in range(1, 2*n+2*d+1):  
           temp_row = []
           for k in range(1, 2*n+2*d+1):# gamma 2 nodes
                  w = 0
                  temp_row.append(w)
           F22.append(temp_row)
##########
F23 = []
for l in range(1, 2*n+2*d+1): 
           temp_row = []
           for k in range(1, 2*n+2*d+1):# eta 3 nodes
                  w = 0
                  temp_row.append(w)
           F23.append(temp_row)
##########
F24 = []
for l in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1, 2*n+2*d+1): # omega1 nodes
               for q in range(1, 2*n+2*d+1):
                  w = 0
                  temp_row.append(w)
            F24.append(temp_row)
##########
F25 = []
for l in range(1, 2*n+2*d+1):
           temp_row = []
           for k in range(1, 2*n+2*d+1):# Gamma nodes
                  w = 0
                  if l==k:
                      w=1
                  temp_row.append(w)
           F25.append(temp_row)
##########
for i in range(len(F21)):
    concatenated_row = F21[i] + F22[i] + F23[i] + F24[i] + F25[i]  
    W6.append(concatenated_row)
######################
# print("weight matrix for seventh layer/sixth hidden layer")
# print(W6)
#####################
# #Bias matrix for seventh layer/sixth hidden layer

B6 = []

# bias matrix alpha 21,22 nodes

for j in range(1, d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1):
                w = 0
                if q == 0:
                      w = 1
                temp_row.append(w)
            B6.append(temp_row)

# bias matrix for beta 21,22 nodes

for j in range(1, d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for k in range(1):
                w = 0
                if q == 0:
                      w = 1
                temp_row.append(w)
            B6.append(temp_row)
            
# bias matrix alpha 31,32 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                w = 0
                if q == 0:
                      w = (-1/eps)+1
                else:
                      w = -1/eps
                temp_row.append(w)
            B6.append(temp_row)

# bias matrix for beta 31,32 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):
            temp_row = []
            for j in range(1):
                w = 0
                if q == 0:
                      w = (1/eps)+1
                else:
                      w = 1/eps
                temp_row.append(w)
            B6.append(temp_row)

# bias matrix for Gamma nodes

for l in range(1, 2*n+2*d+1):
            temp_row = []
            for j in range(1):
                w = 0
                temp_row.append(w)
            B6.append(temp_row)
##################################
# # print('Printing B6')
# # for i in B6:
# #     print(i)
##################################
L6 = []  # alpha 21,22, alpha 21,22, alpha 31,32 and beta 31,32 nodes
for i in range(len(W6)):
    temp_row = []
    L6_i_entry = np.maximum((np.dot(W6[i], L5)+B6[i]), 0)
    L6.append(L6_i_entry)
##################################
# print('Printing alpha 21,22, alpha 21,22, alpha 31,32 and beta 31,32 nodes for seventh layer/sixth hidden layer')
# for i in L6:
#     print(i)
############
# To construct weight matrix for eighth layer/seventh hidden layer is L7=W7*L6+B7
W7 = []
# Tau2(These nodes correspond r'_i. Tau 2 gives the labels of inward edges corresponding to non-zero x_j)
G1 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, d+1):# alpha 21,22
            for l in range(1, 2*n+2*d+1):
              for q in range(2):
                w = 0
                if i == l:
                    if q == 0:
                        w = t1[i-1]
                    else:
                        w = -t1[i-1]
                temp_row.append(w)
        G1.append(temp_row)

##########
G2 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, d+1):# beta 21,22
            for l in range(1, 2*n+2*d+1):
              for q in range(2):
                w = 0
                if i == l:
                    if q == 0:
                        w = t1[i-1]
                    else:
                        w = -t1[i-1]
                temp_row.append(w)
        G2.append(temp_row)
##########
G3 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for l in range(1, 2*n+2*d+1):# alpha 31,32
          for k in range(1, 2*n+2*d+1):
             for q in range(2):
                w = 0
                temp_row.append(w)
        G3.append(temp_row)
##########
G4 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for l in range(1, 2*n+2*d+1):# beta 31,32
          for k in range(1, 2*n+2*d+1):
              for q in range(2):
                w = 0
                temp_row.append(w)
        G4.append(temp_row)
##########
G5 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for l in range(1, 2*n+2*d+1):# Gamma nodes
                w = 0
                temp_row.append(w)
        G5.append(temp_row)

##########
for i in range(len(G1)):
    concatenated_row = G1[i] + G2[i] + G3[i] + G4[i] + G5[i] 
    W7.append(concatenated_row)
# #######################
# zeta 1 nodes(These nodes correspond v'_li of eq.9)
G7 = []
for j in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1): 
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G7.append(temp_row)

##########
G8 = []
for j in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1):
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G8.append(temp_row)
##########
G9 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        if i==j and l==k:
                            if p==0:
                                w=1
                            else:
                                w=-1
                        temp_row.append(w)
            G9.append(temp_row)
##########
G10 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        if l==k and i==j:
                            if p==0:
                                w=1
                            else:
                                w=-1
                        temp_row.append(w)
            G10.append(temp_row)
##########
G11 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                        w = 0
                        temp_row.append(w)
            G11.append(temp_row)
##########
for i in range(len(G7)):
    concatenated_row = G7[i] + G8[i] + G9[i] + G10[i]+ G11[i] 
    W7.append(concatenated_row)
######################
# zeta 2 nodes(These nodes correspond summation v'_ki of eq.10)
G13 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G13.append(temp_row)

##########
G14 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G14.append(temp_row)
##########
G15 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        if  i != 1 and i>l and j != 1 and j>k and i==j and k>=l+1 and k<i:
                            if p==0:
                                w=1
                            else:
                                w=-1
                        temp_row.append(w)
            G15.append(temp_row)
##########
G16 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        if  i != 1 and i>l and j != 1 and j>k and i==j and k>=l+1 and k<i:
                            if p==0:
                                w=1
                            else:
                                w=-1
                        temp_row.append(w)
            G16.append(temp_row)
##########
G17 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                        w = 0
                        temp_row.append(w)
            G17.append(temp_row)
##########
for i in range(len(G13)):
    concatenated_row = G13[i] + G14[i]+ G15[i] + G16[i] + G17[i] 
    W7.append(concatenated_row)
######################
# Gamma nodes
G19 = []
for l in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G19.append(temp_row)

##########
G20 = []
for l in range(1, 2*n+2*d+1): 
            temp_row = []
            for k in range(1, d+1):
                for r in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G20.append(temp_row)
##########
G21 = []
for l in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G21.append(temp_row)
##########
G22 = []
for l in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):  
                     for p in range(2):
                        w = 0
                        temp_row.append(w)
            G22.append(temp_row)
##########
G23 = []
for l in range(1, 2*n+2*d+1):
            temp_row = []
            for k in range(1, 2*n+2*d+1):
                        w = 0
                        if l==k:
                            w=1
                        temp_row.append(w)
            G23.append(temp_row)
##########
for i in range(len(G19)):
    concatenated_row = G19[i] + G20[i] + G21[i] + G22[i] + G23[i]  
    W7.append(concatenated_row)
######################
# print("weight matrix for eighth layer/seventh hidden layer")
# print(W7)
#####################
# #Bias matrix for eighth layer/seventh hidden layer

B7 = []

# bias matrix for Tau2

for i in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1):
         b = -d*t1[i-1]
         temp_row.append(b)
      B7.append(temp_row)
        
# bias matrix for zeta 1 

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1):
            w = -1
            temp_row.append(w)
        B7.append(temp_row)
        
# bias matrix for zeta 2 

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1):
            w=0
            if i != 1 and i>l:
               w = -(i-l-1)
            temp_row.append(w)
        B7.append(temp_row)        

# bias matrix for Gamma 

for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1):
            w=0
            temp_row.append(w)
        B7.append(temp_row)
               
# print('Printing B7')
# for i in B7:
#     print(i)
##################################
L7 = []  # Tau2, zeta 1, zeta 2 nodes
for i in range(len(W7)):
    temp_row = []
    L7_i_entry = np.maximum((np.dot(W7[i], L6)+B7[i]), 0)
    L7.append(L7_i_entry)
##################################
# print('Printing tau2, zeta 1, zeta 2 nodes for eighth layer/seventh hidden layer')
# for i in L7:
#     print(i)
#####################
# To construct weight matrix for ninth layer/eighth hidden layer is L7=W7*L6+B7
W8 = []
# Tau2 nodes as identity map
H1 = []
for i in range(1, 2*n+2*d+1):
          temp_row = []
          for j in range(1, 2*n+2*d+1):
                    w = 0
                    if i == j:
                        w = 1
                    temp_row.append(w)
          H1.append(temp_row)
##########
H2 = []
for i in range(1, 2*n+2*d+1):
          temp_row = []          
          for l in range(1, 2*n+2*d+1):
                for k in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H2.append(temp_row)
##########
H3 = []
for i in range(1, 2*n+2*d+1):
          temp_row = []          
          for l in range(1, 2*n+2*d+1):
                for k in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H3.append(temp_row)
##########
H4 = []
for i in range(1, 2*n+2*d+1):
          temp_row = []          
          for l in range(1, 2*n+2*d+1):# Gamma
                    w = 0
                    temp_row.append(w)
          H4.append(temp_row)
##########
for i in range(len(H1)):
    concatenated_row = H1[i] + H2[i] + H3[i] + H4[i]
    W8.append(concatenated_row)
# print(W8)
######################
# mu31,32 nodes
H13 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2): 
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                    w = 0
                    if i != 1 and i>l and l==k:
                        w=-1/eps
                    temp_row.append(w)
          H13.append(temp_row)
##########
H14 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H14.append(temp_row)
##########
H15 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H15.append(temp_row)
##########
H16 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                    w = 0
                    if i != 1 and i>l and i==k:
                        w=1/eps
                    temp_row.append(w)
          H16.append(temp_row)    
##########
for i in range(len(H13)):
    concatenated_row = H13[i] + H14[i] + H15[i] + H16[i]
    W8.append(concatenated_row)
# print(W8)
######################
# lambda 31,32 nodes
H19 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2): 
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                    w = 0
                    if i != 1 and i>l and l==k:
                        w=1/eps
                    temp_row.append(w)
          H19.append(temp_row)
##########
H20 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H20.append(temp_row)
##########
H21 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H21.append(temp_row)
##########
H22 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
        for q in range(2):     
          temp_row = []
          for k in range(1, 2*n+2*d+1):
                    w = 0
                    if i != 1 and i>l and i==k:
                        w=-1/eps
                    temp_row.append(w)
          H22.append(temp_row)    
##########
for i in range(len(H19)):
    concatenated_row = H19[i] + H20[i] + H21[i] + H22[i]
    W8.append(concatenated_row)
# print(W8)
######################
# omega 2 nodes
H25 = []
for j in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1):
          temp_row = []
          for i in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H25.append(temp_row)
##########
H26 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
          temp_row = []
          for j in range(1, 2*n+2*d+1):
                for k in range(1, 2*n+2*d+1):
                    w = 0
                    if l==j and i==k:
                        w = 1
                    temp_row.append(w)
          H26.append(temp_row)
##########
H27 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
          temp_row = []
          for j in range(1, 2*n+2*d+1):
                for k in range(1, 2*n+2*d+1):
                    w = 0
                    if l==j and i==k:
                        w = -1
                    temp_row.append(w)
          H27.append(temp_row)
##########
H28 = []
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
          temp_row = []
          for j in range(1, 2*n+2*d+1):
                    w = 0
                    temp_row.append(w)
          H28.append(temp_row)    
##########
for i in range(len(H25)):
    concatenated_row = H25[i] + H26[i] + H27[i] + H28[i]
    W8.append(concatenated_row)    
  
# #######################
# print("weight matrix for ninth layer/eighth hidden layer")
# print(W8)
#####################
# #Bias matrix for ninth layer/eighth hidden layer

B8 = []
# bias matrix for Tau2 nodes

for i in range(1, 2*n+2*d+1):  
            temp_row = []
            for k in range(1):
                w = 0
                temp_row.append(w)
            B8.append(temp_row)

# bias matrix for mu 31, 32

for l in range(1, 2*n+2*d+1):
      for i in range(1, 2*n+2*d+1):  
          for p in range(2):
            temp_row = []
            for k in range(1):
                w = 0
                if i != 1 and i>l:
                  if p == 0:
                      w = (-m/eps)+1
                  else:
                      w = -m/eps
                temp_row.append(w)
            B8.append(temp_row)
          
# bias matrix for lambda 31, 32

for l in range(1, 2*n+2*d+1):
      for i in range(1, 2*n+2*d+1):  
          for p in range(2):
            temp_row = []
            for k in range(1):
                w = 0
                if i != 1 and i>l:
                  if p == 0:
                      w = (m/eps)+1
                  else:
                      w = m/eps
                temp_row.append(w)
            B8.append(temp_row)
          
# bias matrix for omega 2

for i in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1):
            w = 0
            temp_row.append(w)
        B8.append(temp_row)

# print('Printing B8')
# for i in B8:
#     print(i)
##################################
L8 = []  # mu31,32, lambda 31,32, omega 2 nodes
for i in range(len(W8)):
    temp_row = []
    L8_i_entry = np.maximum((np.dot(W8[i], L7)+B8[i]), 0)
    L8.append(L8_i_entry)
############################
# print('Printing mu31,32, lambda 31,32, omega 2 nodes for ninth layer/eighth hidden layer')
# for i in L8:
#     print(i)
############################
# To construct weight matrix for tenth layer/ninth hidden layer is L9=W9*L8+B9
W9 = []
# Tau2 nodes as identity map
H43 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Tau2
        w = 0
        if i==l:
            w = 1
        temp_row.append(w)
    H43.append(temp_row)
##########
H45 = []
for k in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# mu 31, 32
        for i in range(1, 2*n+2*d+1):  
            for p in range(2):
              w = 0
              temp_row.append(w)
    H45.append(temp_row)
##########
H46 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# lambda 31, 32
        for i in range(1, 2*n+2*d+1):  
            for p in range(2):
              w = 0
              temp_row.append(w)
    H46.append(temp_row)
##########
H47 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# omega 2
        for k in range(1, 2*n+2*d+1):
          w = 0
          temp_row.append(w)
    H47.append(temp_row)
##########
for i in range(len(H43)):
    concatenated_row = H43[i] + H45[i]+ H46[i] + H47[i] 
    W9.append(concatenated_row)    
#######################

# omega 2 nodes as identity map
H57 = []
for i in range(1, 2*n+2*d+1):
    for j in range(1, 2*n+2*d+1):
      temp_row = []
      for l in range(1, 2*n+2*d+1):# Tau2
          w = 0
          temp_row.append(w)
      H57.append(temp_row)
##########
H59 = []
for i in range(1, 2*n+2*d+1):
    for j in range(1, 2*n+2*d+1):
      temp_row = []
      for l in range(1, 2*n+2*d+1):# mu 31, 32
        for i in range(1, 2*n+2*d+1):  
            for p in range(2):
                w = 0
                temp_row.append(w)
      H59.append(temp_row)
##########
H60 = []
for i in range(1, 2*n+2*d+1):
    for j in range(1, 2*n+2*d+1):
      temp_row = []
      for l in range(1, 2*n+2*d+1):# lambda 31, 32
          for i in range(1, 2*n+2*d+1):  
              for p in range(2):
                w = 0
                temp_row.append(w)
      H60.append(temp_row)
##########
H61 = []
for i in range(1, 2*n+2*d+1):
    for j in range(1, 2*n+2*d+1):
      temp_row = []
      for l in range(1, 2*n+2*d+1):# omega 2
        for k in range(1, 2*n+2*d+1):
          w = 0
          if i==l and j==k:
              w=1
          temp_row.append(w)
      H61.append(temp_row)
##########
for i in range(len(H57)):
    concatenated_row = H57[i] + H59[i]+ H60[i] + H61[i] 
    W9.append(concatenated_row)    
#######################
# zeta3 (These nodes are delta(s_i, r'_l +m))
H63 = []
for l in range(1, 2*n+2*d+1):
   for i in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1, 2*n+2*d+1):# Tau2
         w = 0
         temp_row.append(w)
      H63.append(temp_row)
##########
H64 = []
for i in range(1, 2*n+2*d+1):
   for j in range(1, 2*n+2*d+1):
     temp_row = []   
     for l in range(1, 2*n+2*d+1):# mu31,32 nodes 
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
              w = 0
              if i==l and j==k:
                  if q==0:
                      w=1
                  else:
                      w=-1
              temp_row.append(w)
     H64.append(temp_row)
##########
H65 = []
for i in range(1, 2*n+2*d+1):
   for j in range(1, 2*n+2*d+1):
     temp_row = []   
     for l in range(1, 2*n+2*d+1):# lambda31,32 nodes 
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
              w = 0
              if i==l and j==k:
                  if q==0:
                      w=1
                  else:
                      w=-1
              temp_row.append(w)
     H65.append(temp_row)
##########
H66 = []
for l in range(1, 2*n+2*d+1):
   for i in range(1, 2*n+2*d+1):
      temp_row = []
      for j in range(1, 2*n+2*d+1):# omega 2
        for k in range(1, 2*n+2*d+1):
          w = 0
          temp_row.append(w)
      H66.append(temp_row)
##########
for i in range(len(H63)):
    concatenated_row = H63[i] + H64[i] + H65[i]+ H66[i]  
    W9.append(concatenated_row)    
#######################  
# print("weight matrix for tenth layer/ninth hidden layer")
# print(W9)
#####################
# #Bias matrix for tenth layer/ninth hidden layer

B9 = []
# bias matrix for Tau2

for l in range(1, 2*n+2*d+1):
          temp_row = []
          for k in range(1):
            w = 0
            temp_row.append(w)
          B9.append(temp_row)

# bias matrix for omega2 nodes

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
          temp_row = []
          for k in range(1):
            w = 0
            temp_row.append(w)
          B9.append(temp_row)

# bias matrix for zeta 3 nodes
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1):
         w = -1
         temp_row.append(w)
      B9.append(temp_row)          
##################################         
# print('Printing B9')
# for i in B9:
#     print(i)
##################################
L9 = []  # zeta3 nodes
for i in range(len(W9)):
    temp_row = []
    L9_i_entry = np.maximum((np.dot(W9[i], L8)+B9[i]), 0)
    L9.append(L9_i_entry)
###################################
# print('Printing zeta3 nodes for tenth layer/ninth hidden layer')
# for i in L9:
#     print(i)
##################################
# To construct weight matrix for eleventh layer/tenth hidden layer is L10=W10*L9+B10
W10 = []
I1=[]# tau2 nodes as identity map
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):# tau2 nodes
                  w = 0
                  if i==l:
                     w = 1
                  temp_row.append(w)
       I1.append(temp_row)
##########
I2=[]
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):# omega2 nodes
           for k in range(1, 2*n+2*d+1):
                  w = 0
                  temp_row.append(w)
       I2.append(temp_row)
##########
I3=[]
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):# zeta3 nodes
           for k in range(1, 2*n+2*d+1):
                  w = 0
                  temp_row.append(w)
       I3.append(temp_row)
##########
for i in range(len(I1)):
    concatenated_row = I1[i] + I2[i] + I3[i] 
    W10.append(concatenated_row)
# print(W10)
######################
I4=[]# Omega1 
for i in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, 2*n+2*d+1):# tau2 nodes
                  w = 0
                  temp_row.append(w)
       I4.append(temp_row)
##########
I5=[]
for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
       temp_row = []
       for k in range(1, 2*n+2*d+1):# omega2 nodes
           for j in range(1, 2*n+2*d+1):
                  w = 0
                  if i==j and k != l:
                      w = -1
                  temp_row.append(w)
       I5.append(temp_row)
##########
I6=[]
for i in range(1, 2*n+2*d+1):
    for l in range(1, 2*n+2*d+1):
       temp_row = []
       for j in range(1, 2*n+2*d+1):# zeta3 nodes
          for k in range(1, 2*n+2*d+1):
                  w = 0
                  if i==j and l==k:
                     w = 1
                  temp_row.append(w)
       I6.append(temp_row)
##########
for i in range(len(I4)):
    concatenated_row = I4[i] + I5[i] + I6[i] 
    W10.append(concatenated_row)
#####################
# print("weight matrix for tenth layer/ninth hidden layer")
# print(W10)
#####################
# #Bias matrix for tenth layer/ninth hidden layer

B10 = []
# bias matrix for Tau2

for i in range(1, 2*n+2*d+1):  
       temp_row = []
       for k in range(1):
          w = 0
          temp_row.append(w)
       B10.append(temp_row)
       
# bias matrix for Omega 1

for l in range(1, 2*n+2*d+1):
    for i in range(1, 2*n+2*d+1):
          temp_row = []
          for k in range(1):
            w = 0
            temp_row.append(w)
          B10.append(temp_row)       
##################################       
# print('Printing B10')
# for i in B10:
#     print(i)
##################################
L10 = []  # Omega 1 nodes
for i in range(len(W10)):
    temp_row = []
    L10_i_entry = np.maximum((np.dot(W10[i], L9)+B10[i]), 0)
    L10.append(L10_i_entry)
##################################
# print('Printing Omega 1 nodes for eleventh layer/tenth hidden layer')
# for i in L10:
#     print(i)
############
# # To construct weight matrix for twelfth layer/eleventh hidden layer is L11=W11*L10+B11
W11 = []
# Tau2 nodes as identity map
J1 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Tau2 nodes
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    J1.append(temp_row)
##########
J2 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Omega 1 nodes
       for k in range(1, 2*n+2*d+1):
            w = 0
            temp_row.append(w)
    J2.append(temp_row)    
##########
for i in range(len(J1)):
    concatenated_row = J1[i] + J2[i]
    W11.append(concatenated_row)
#######################
# Omega2 nodes that gives the lable and position of outward edges of required inward edges
J3 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Tau2 nodes
        w = 0
        temp_row.append(w)
    J3.append(temp_row)
##########
J4 = []
for i in range(1, 2*n+2*d+1):
       temp_row = []
       for l in range(1, 2*n+2*d+1):# Omega1
           for k in range(1, 2*n+2*d+1):
             w = 0
             if i==k:
                 w=t1[i-1]
             temp_row.append(w)
       J4.append(temp_row)    
##########
for i in range(len(J3)):
    concatenated_row = J3[i] + J4[i]
    W11.append(concatenated_row)
#######################
# print("weight matrix for twelfth layer/eleventh hidden layer")
# print(W11)
#####################
# #Bias matrix for twelfth layer/eleventh hidden layer

B11 = []
# bias matrix for Tau2
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1):
        w = 0
        temp_row.append(w)
    B11.append(temp_row)
# bias matrix for Omega2
for l in range(1, 2*n+2*d+1):
      temp_row = []
      for k in range(1):
          w = 0
          temp_row.append(w)
      B11.append(temp_row)    
##################################    
# print('Printing B11')
# for i in B11:
#     print(i)
##################################
L11 = []  # Omega 2 nodes
for i in range(len(W11)):
    temp_row = []
    L11_i_entry = np.maximum((np.dot(W11[i], L10)+B11[i]), 0)
    L11.append(L11_i_entry)
##################################
# print('Printing Omega2 nodes for twelfth layer/eleventh hidden layer')
# for i in L11:
#     print(i)
##################################
# To construct weight matrix for thirteenth layer/twelfth hidden layer is L12=W12*L11+B12
W12 = []
# upsilon 1 nodes
N5 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Tau 2 nodes
        w = 0
        if i == l:
            w = 1
        temp_row.append(w)
    N5.append(temp_row)

##########
N6 = []
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):# Omega 3 nodes
            w = 0
            if i==l:
                w=1
            temp_row.append(w)
    N6.append(temp_row)    
##########
for i in range(len(N5)):
    concatenated_row = N5[i] + N6[i]
    W12.append(concatenated_row)
#######################
# print("weight matrix for thirteenth layer/twelfth hidden layer")
# print(W12)
#####################
# #Bias matrix for thirteenth layer/twelfth hidden layer

B12 = []
# bias matrix for upsilon 1

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1):
        w = 0
        temp_row.append(w)
    B12.append(temp_row)

# print('Printing B12')
# for i in B12:
#     print(i)
##################################
L12 = []  # upsilon 1 nodes
for i in range(len(W12)):
    temp_row = []
    L12_i_entry = np.maximum((np.dot(W12[i], L11)+B12[i]), 0)
    L12.append(L12_i_entry)
##################################
# print('Printing upsilon 1 nodes for thirteenth layer/twelfth hidden layer')
# for i in L12:
#     print(i)
##################################
# To construct weight matrix for fourteenth/thirteenth hidden layer is L13=W13*L12+B13
W13 = []
# rho 31,32 nodes to compute delta(P_i, 0) 
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            w = 0
            if i == l:
              w = 1/eps
            temp_row.append(w)
        W13.append(temp_row)

##########
# verrho 31,32 nodes to compute delta(P_i, 0) 
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            w = 0
            if i == l:
              w = -1/eps
            temp_row.append(w)
        W13.append(temp_row)
#####################
# #Bias matrix for fourteenth/thirteenth hidden layer

B13 = []
# bias matrix for rho 31,32 nodes

for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            w = 0
            if q==0:
                w=1
            temp_row.append(w)
        B13.append(temp_row)
    
# bias matrix for verrho 31,32 nodes

for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
            w = 0
            if q==0:
                w=1
            temp_row.append(w)
        B13.append(temp_row)
# print('Printing B13')
# for i in B13:
#     print(i)
##################################
L13 = []  # rho 31,32 and verrho 31,32 nodes
for i in range(len(W13)):
    temp_row = []
    L13_i_entry = np.maximum((np.dot(W13[i], L12)+B13[i]), 0)
    L13.append(L13_i_entry)
##################################
# print('Printing rho 31,32 and verrho 31,32 nodes for fourteenth/thirteenth hidden layer')
# for i in L13:
#     print(i)
##############
# To construct weight matrix for fifteenth/fourteenth hidden layer is L14=W14*L13+B14
W14 = []
# upsilon 2 nodes(Q_i nodes of eq.36)
N7=[]
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                if k == i:
                    if q==0:
                        w=1
                    else:
                        w=-1
                temp_row.append(w)
        N7.append(temp_row)
##########
N8=[]
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1, 2*n+2*d+1):
            for q in range(2):
                w = 0
                if k == i:
                    if q==0:
                        w=1
                    else:
                        w=-1
                temp_row.append(w)
        N8.append(temp_row)
##########
for i in range(len(N7)):
    concatenated_row = N7[i] + N8[i]
    W14.append(concatenated_row)
#######################
# print("weight matrix for fifteenth/fourteenth hidden layer")
# print(W14)
#####################
# #Bias matrix for fifteenth/fourteenth hidden layer

B14 = []
for i in range(1, 2*n+2*d+1):
        temp_row = []
        for k in range(1):
            w = -1
            temp_row.append(w)
        B14.append(temp_row)

# print('Printing B14')
# for i in B14:
#     print(i)
##################################
L14 = []  # upsilon 2 nodes
for i in range(len(W14)):
    temp_row = []
    L14_i_entry = np.maximum((np.dot(W14[i], L13)+B14[i]), 0)
    L14.append(L14_i_entry)
##################################
# print('Printing upsilon 2 nodes for fifteenth/fourteenth hidden layer')
# for i in L14:
#     print(i)
##################
# To construct weight matrix for sixteenth/fifteenth hidden layer is L15=W15*L14+B15
W15 = []
#varsigma1 nodes for summation Q_k
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1, 2*n+2*d+1):
            w = 0
            if k <= i:
                w = 1
            temp_row.append(w)
    W15.append(temp_row)
## rho 41,42 nodes to compute delta(Q_i, 0) 
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            w = 0
            if l == i:
                w = 1/eps
            temp_row.append(w)
        W15.append(temp_row)
## varrho 41,42 nodes to compute delta(Q_i, 0) 
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for l in range(1, 2*n+2*d+1):
            w = 0
            if l == i:
                w = -1/eps
            temp_row.append(w)
        W15.append(temp_row)
#########
# print("weight matrix for sixteenth/fifteenth hidden layer")
# print(W15)
#####################
# #Bias matrix for sixteenth/fifteenth hidden layer

B15 = []

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1):
        w = 0
        temp_row.append(w)
    B15.append(temp_row)
    
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          w = 0
          if q==0:
              w=1
          temp_row.append(w)
        B15.append(temp_row)
       
for i in range(1, 2*n+2*d+1):
    for q in range(2):
        temp_row = []
        for k in range(1):
          w = 0
          if q==0:
              w=1
          temp_row.append(w)
        B15.append(temp_row)       
# print('Printing B15')
# for i in B15:
#     print(i)
##################################
L15 = []  # varsigma 1, rho 41,42, verroh 41,42 nodes
for i in range(len(W15)):
    temp_row = []
    L15_i_entry = np.maximum((np.dot(W15[i], L14)+B15[i]), 0)
    L15.append(L15_i_entry)
##################################
# print('varsigma 1, rho 41,42, verroh 41,42 nodes for sixteenth/fifteenth hidden layer')
# for i in L15:
#     print(i)
##################
# To construct weight matrix for seventeenth/sixteenth hidden layer is L16=W16*L15+B16
W16= []
######
# for varsigma 2 nodes correspondint to R_i 
N9=[]
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):
        w = 0
        if l == i:
            w = B
        temp_row.append(w)
    N9.append(temp_row)
##########
N10=[]
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            if l == i:
                if q==0:
                  w = -C
                else:
                  w= C  
            temp_row.append(w)
    N10.append(temp_row)
##########
N11=[]
for i in range(1, 2*n+2*d+1):
    temp_row = []
    for l in range(1, 2*n+2*d+1):
        for q in range(2):
            w = 0
            if l == i:
                if q==0:
                  w = -C
                else:
                  w= C  
            temp_row.append(w)
    N11.append(temp_row)
##########
for i in range(len(N9)):
    concatenated_row = N9[i] + N10[i] + N11[i]
    W16.append(concatenated_row)
#######################
# print("weight matrix for seventeenth/sixteenth hidden layer")
# print(W16)
#####################
# #Bias matrix for seventeenth/sixteenth hidden layer

B16 = []

for i in range(1, 2*n+2*d+1):
    temp_row = []
    for k in range(1):
        w = C
        temp_row.append(w)
    B16.append(temp_row)
    
# print('Printing B16')
# for i in B16:
#     print(i)
##################################
L16 = []  # varsigma 2 nodes
for i in range(len(W16)):
    temp_row = []
    L16_i_entry = np.maximum((np.dot(W16[i], L15)+B16[i]), 0)
    L16.append(L16_i_entry)
##################################
# print('Printing varsigma 2 nodes for seventeenth/sixteenth hidden layer')
# for i in L16:
#     print(i)
##################
# To construct weight matrix for eighteenth/seventeenth layer hidden layer L17=W17*L16+B17
W17 = []
######
# xi nodes corresponding to eq.38
for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
      for q in range(2):
          temp_row = []
          for l in range(1, 2*n+2*d+1):
            w = 0
            if i+j-1 == l:
                w = 1/eps
            temp_row.append(w)
          W17.append(temp_row)
#######
# chi nodes
for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
      for q in range(2):
          temp_row = []
          for l in range(1, 2*n+2*d+1):
            w = 0
            if i+j-1 == l:
                w = -1/eps
            temp_row.append(w)
          W17.append(temp_row)
##################################
# print("Printing weight matrix for eighteenth/seventeenth layer hidden layer")
# for i in W17:
#     print(i)
##################################
# Bias matrix for eighteenth/seventeenth layer hidden layer
B17 = []
for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
      for q in range(2):
          temp_row = []
          for l in range(1):
            w = 0
            if q == 0:
                w =( -(i*B)/eps )+1
            else:
                w = -(i*B)/eps 
            temp_row.append(w)
          B17.append(temp_row)

for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
      for q in range(2):
          temp_row = []
          for l in range(1):
            w = 0
            if q == 0:
                w =( (i*B+1)/eps )+1
            else:
                w = (i*B+1)/eps 
            temp_row.append(w)
          B17.append(temp_row)
##################################          
# print("Printing B17")
# for i in B17:
#     print(i)
##################################
L17 = []  # Xi and Chi nodes
for i in range(len(W17)):
    #temp_row = []
    L17_i_entry = np.maximum((np.dot(W17[i], L16)+B17[i]), 0)
    L17.append(L17_i_entry)
##################################
# print('Printing Xi and Chi nodes of eighteenth/seventeenth layer hidden layer')
# for i in L17:
#     print(i)
##################################
# To construct weight matrix for ninteenth/eighteenth hidden layer L18=W18*L17+B18
W18 = []
## varpi nodes
N12=[]
for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
            temp_row = []
            for l in range(1, 2*n-2*d+2*d+1):
                for k in range(1, 2*d+2):
                    for q in range(2):
                        w = 0
                        if i == l and j==k:
                            if q==0:
                                w=t1[i+j-2]
                            else:
                                w=-t1[i+j-2]
                        temp_row.append(w)
            N12.append(temp_row)

N13=[]
for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
            temp_row = []
            for l in range(1, 2*n-2*d+2*d+1):
                for k in range(1, 2*d+2):
                    for q in range(2):
                        w = 0
                        if i == l and j==k:
                            if q==0:
                                w=t1[i+j-2]
                            else:
                                w=-t1[i+j-2]
                        temp_row.append(w)
            N13.append(temp_row)
##########
for i in range(len(N12)):
    concatenated_row = N12[i] + N13[i] 
    W18.append(concatenated_row)
#######################
# print("weight matrix for ninteenth/eighteenth hidden layer")
# print(W18)
#######################
# Bias matrix for ninteenth/eighteenth hidden layer
B18 = []

for i in range(1, 2*n-2*d+2*d+1):
    for j in range(1, 2*d+2):
            temp_row = []
            for l in range(1):
                w = -t1[i+j-2]
                temp_row.append(w)
            B18.append(temp_row)
#######################
# print("Printing B18")
# for i in B18:
#     print(i)
#######################
L18 = []  # varpi nodes
for i in range(len(W18)):
    temp_row = []
    L18_i_entry = np.maximum((np.dot(W18[i], L17)+B18[i]), 0)
    L18.append(L18_i_entry)
#######################
# print('Printing varpi nodes of ninteenth/eighteenth hidden layer')
# for i in L18:
#     print(i)
#######################
# To construct weight matrix for Output layer y=W19*L18+B19
W19= []

for i in range(1, 2*n-2*d+2*d+1):
    temp_row = []
    for l in range(1, 2*n-2*d+2*d+1):
        for j in range(1, 2*d+2):
            w = 0
            if l == i:
                w = 1
            temp_row.append(w)
    W19.append(temp_row)
#######################
# print("weight for Output layer")
# print(W19)
#######################
# Bias matrix for Output layer

B19 = []
for i in range(1, 2*n-2*d+2*d+1):
    temp_row = []
    for l in range(1):
        w = 0
        temp_row.append(w)
    B19.append(temp_row)
#######################
# print("Bias for Output layer")
# print(B19)
#######################
Y = []  # Output nodes
for i in range(len(W19)):
    #temp_row = []
    Y_i_entry = np.maximum((np.dot(W19[i], L18)+B19[i]), 0)
    Y.append(Y_i_entry)
output = [int(array[0]) for array in Y]

print("Output string, y:", output)

# Define the threshold (2 million)
threshold = 2*m

# Filter values that are ≤ 2 m
filtered_euler_string = [y for y in output if y <= threshold]

# Print the final output
print("Required Euler string, u:", filtered_euler_string)

