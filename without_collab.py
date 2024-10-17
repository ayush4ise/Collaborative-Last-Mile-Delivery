import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np

data = pd.read_excel("Data\Instances\Small_Instances.xlsx", sheet_name='A1', index_col=0)
data = data[['X', 'Y']]

def EuclideanDistance(x1, y1, x2, y2):
    import math
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

# Make a matrix of distances
distances = pd.DataFrame(index=data.index, columns=data.index)
for i in data.index:
    for j in data.index:
        distances.loc[i, j] = EuclideanDistance(data.loc[i, 'X'], data.loc[i, 'Y'], data.loc[j, 'X'], data.loc[j, 'Y'])

# Create a new model
m = gp.Model("smaller_instances")

# Sets
# Given in Table 3
L = [1, 2] # list of lsps
D = [1, 2] # list of depots # P
S = [3, 4, 5, 6] # list of satellites
C = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # list of customers # Z
# O = [19, 20] # list of collaboration points

T = [1,2] # list of first echelon vehicles
V = [3, 4, 5, 6] # list of second echelon vehicles


DUS = D + S # list of depots and satellites
# SUCUO = S + C + O # list of satellites, customers, and collaboration points
# CUO = C + O # list of customers, and collaboration points
SUC = S + C # list of satellites, customers
ns = len(S) # number of satellites
nc = len(C) # number of customers

DUS_1 = [1, 3, 5] # list of depots and satellites for lsp 1
DUS_2 = [2, 4, 6] # list of depots and satellites for lsp 2

C1 = [8, 10, 12, 14, 16, 18] # list of customers for lsp 1
C2 = [7, 9, 11, 13, 15, 17] # list of customers for lsp 2

S1 = [3, 5] # list of satellites for lsp 1
S2 = [4, 6] # list of satellites for lsp 2

# Parameters
# Given in Table 4
# t_ij
C_ij = np.array(distances) # cost to travel from i to j

F_t = {1:100, 2:100} # fixed cost of first echelon vehicles
F_v = {3:50, 4:50, 5:50, 6:50} # fixed cost of second echelon vehicles

d_c = {7:10, 8:10, 9:10, 10:10, 11:10, 12:10, 13:10, 14:10, 15:10, 16:10, 17:10, 18:10} # demand of customer c
A_s = {3:30, 4:30, 5:30, 6:30} # capacity of satellite s
K1 = 60 # capacity of first echelon vehicles
K2 = 30 # capacity of second echelon vehicles
p_c = {7:2, 8:1, 9:2, 10:1, 11:2, 12:1, 13:2, 14:1, 15:2, 16:1, 17:2, 18:1} # DC to which customer c belongs to



# Assumption
M = 10000 # A sufficiently large constant



# Variables
# Given in Table 5

# 1 if first echelon vehicle ‘t’ is moving from node ‘i’ to ‘j’, 0 otherwise
R_ij_t = m.addVars(DUS, DUS, T, vtype=gp.GRB.BINARY, name="R_ij_t") 

# 1 if second echelon vehicle ‘v’ is moving from node ‘i’ to ‘j’, 0 otherwise
X_ij_v = m.addVars(SUC, SUC, V, vtype=gp.GRB.BINARY, name="X_ij_v")

# 1 if first echelon vehicle ‘t’ is being used, 0 otherwise
U_t = m.addVars(T, vtype=gp.GRB.BINARY, name="U_t")

# 1 if second echelon vehicle ‘v’ is being used, 0 otherwise
U_v = m.addVars(V, vtype=gp.GRB.BINARY, name="Y_v")

# 1 if customer ‘c’ is assigned to first echelon vehicle ‘t’, 0 otherwise
Y_c_t = m.addVars(C, T, vtype=gp.GRB.BINARY, name="Y_c_t")

# 1 if customer ‘c’ is assigned to second echelon vehicle ‘v’, 0 otherwise
Y_c_v = m.addVars(C, V, vtype=gp.GRB.BINARY, name="Y_c_v")

# 1 if customer ‘c’ is assigned to satellite ‘s’, 0 otherwise
Q_c_s = m.addVars(C, S, vtype=gp.GRB.BINARY, name="Q_c_s")



# Objective Function
m.setObjective(
    gp.quicksum(C_ij[i-1, j-1] * R_ij_t[i, j, t] for i in DUS for j in DUS for t in T) +
    gp.quicksum(C_ij[i-1, j-1] * X_ij_v[i, j, v] for i in SUC for j in SUC for v in V) +
    gp.quicksum(F_t[t] * U_t[t] for t in T) + 
    gp.quicksum(F_v[v] * U_v[v] for v in V),
    sense=gp.GRB.MINIMIZE
)



# Constraints
# Constraints in first echelon

# for j in DUS:
#     for t in T:
#         m.addConstr(gp.quicksum(R_ij_t[l, j, t] for l in DUS) + gp.quicksum(R_ij_t[j, l, t] for l in DUS) == 0, name=f"FirstEchelon1_j_{j}_t_{t}")
for j in DUS:
    for t in T:
        m.addConstr(gp.quicksum(R_ij_t[l, j, t] for l in DUS) - gp.quicksum(R_ij_t[j, l, t] for l in DUS) == 0, name=f"FirstEchelon1_j_{j}_t_{t}")


for t in T:
    m.addConstr(gp.quicksum((gp.quicksum(R_ij_t[i, j, t] for j in D)) for i in S) <= U_t[t], name=f"FirstEchelon2_t_{t}")


for c in C:
    for t in T:
        m.addConstr(gp.quicksum(R_ij_t[p_c[c], s, t] for s in S) >= Y_c_t[c, t], name = f"FirstEchelon3_c_{c}_t_{t}")


for t in T:
    m.addConstr(gp.quicksum((d_c[c] * Y_c_t[c, t]) for c in C) <= K1 * U_t[t], name=f"FirstEchelon4_t_{t}")


for c in C:
    m.addConstr(gp.quicksum(Y_c_t[c, t] for t in T) == 1, name=f"FirstEchelon5_c_{c}")


for c in C:
    for s in S:
        for t in T:
            m.addConstr(M * (2 - Y_c_t[c, t] - Q_c_s[c, s]) + gp.quicksum(R_ij_t[s, k, t] for k in DUS) >= 1, name=f"FirstEchelon6_c_{c}_s_{s}_t_{t}")



# Constraints in second echelon

# for j in SUCUO:
#     for v in V:
#         m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUCUO) + gp.quicksum(X_ij_v[j, l, v] for l in SUCUO) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")
for j in SUC:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUC) - gp.quicksum(X_ij_v[j, l, v] for l in SUC) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")


# changed <= to ==
for v in V:
    m.addConstr(gp.quicksum((gp.quicksum(X_ij_v[i, j, v] for j in S)) for i in C) == U_v[v], name=f"SecondEchelon2_v_{v}")


for c in C:
    m.addConstr(gp.quicksum(Y_c_v[c, v] for v in V) == 1, name=f"SecondEchelon3_c_{c}")


# changed == to >=
for c in C:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[c, k, v] for k in SUC) >= Y_c_v[c, v], name=f"SecondEchelon4_c_{c}_v_{v}")


# added 
for c in C:
    for s in S:
        for v in V:
            m.addConstr(gp.quicksum(X_ij_v[c, k, v] for k in SUC) + gp.quicksum(X_ij_v[s, k, v] for k in C) - Q_c_s[c, s] >= Y_c_v[c, v], name = f'SecondEchelonNew1_c_{c}_s_{s}_v_{v}')


# added
for v in V:
    m.addConstr(gp.quicksum(d_c[c] * Y_c_v[c, v] for c in C) <= K2 * U_v[v], name = f'SecondEchelonNew2_v_{v}')



# Optimize model
m.optimize()

# Output the results
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    # for var in m.getVars():
    #     print(f"{var.varName}: {var.x}")

    varInfo = {}
    for v in m.getVars():
        if v.x>0:
            varInfo[v.varName] = v.x

    pd.DataFrame(varInfo, index = ['value']).T.to_excel('solution.xlsx')

else:
    print(f"Optimization ended with status {m.status}.")


# # do IIS if the model is infeasible
# if m.Status == GRB.INFEASIBLE:
#     m.computeIIS()

# m.write('iismodel.ilp')

# # Print out the IIS constraints and variables
# print('\nThe following constraints and variables are in the IIS:')
# for c in m.getConstrs():
#     if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

# for v in m.getVars():
#     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')