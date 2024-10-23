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
# L = [1, 2] # list of lsps
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
w_i_j = m.addVars(C, S, vtype=gp.GRB.BINARY, name="w_i_j")

# dellaert_19
# flow from depot i ot satellite j on first echelon vehicle 't' 
x_ij_t = m.addVars(D, S, T, vtype=gp.GRB.INTEGER, name="x_ij_t")



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


# dellaert_21 - U_t, dellaert_19 - 1
# dellaert_21 - i in S, j in D, dellaert_19 - i in DUS, j in D
for t in T:
    m.addConstr(gp.quicksum((gp.quicksum(R_ij_t[i, j, t] for j in D)) for i in DUS) <= 1, name=f"FirstEchelon2_t_{t}")


# dellaert_19
for t in T:
    for i in S:
        for j in D:
            m.addConstr(K1 * gp.quicksum(R_ij_t[i, k, t] for k in DUS) - x_ij_t[j, i, t] >= 0, name=f"D19_1_t_{t}_i_{i}_j_{j}")


# dellaert_19
for t in T:
    for i in S:
        for j in D:
            m.addConstr(K1 * gp.quicksum(R_ij_t[j, k, t] for k in DUS) - x_ij_t[j, i, t] >= 0, name=f"D19_2_t_{t}_i_{i}_j_{j}")


# dellaert_19
for t in T:
    m.addConstr(gp.quicksum(gp.quicksum(x_ij_t[i, j, t] for j in S) for i in D) <= K1 * U_t[t], name=f"D19_3_t_{t}")


# dellaert_19
for i in C:
    m.addConstr(gp.quicksum(gp.quicksum(X_ij_v[i, j, v] for j in SUC) for v in V) == 1, name=f"D19_4_i_{i}")



# Constraints in second echelon

# for j in SUCUO:
#     for v in V:
#         m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUCUO) + gp.quicksum(X_ij_v[j, l, v] for l in SUCUO) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")
# dellaert_19 - j - S
for j in S:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUC) - gp.quicksum(X_ij_v[j, l, v] for l in SUC) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")


# changed <= to ==
# dellaert_19 - <=1, i in SUC
for v in V:
    m.addConstr(gp.quicksum((gp.quicksum(X_ij_v[i, j, v] for j in S)) for i in SUC) <=1, name=f"SecondEchelon2_v_{v}")


# All dellaert_19 from here on
for i in C:
    for j in S:
        for v in V:
            m.addConstr(gp.quicksum(X_ij_v[i, k, v] for k in SUC) + gp.quicksum(X_ij_v[j, k, v] for k in SUC) - w_i_j[i, j] <=1, name=f"D19_5_i_{i}_j_{j}_v_{v}")


for i in C:
    m.addConstr(gp.quicksum(w_i_j[i, j] for j in S) == 1, name=f"D19_6_i_{i}")


for v in V:
    m.addConstr(gp.quicksum((d_c[i] * gp.quicksum(X_ij_v[i, j, v] for j in SUC)) for i in C) <= K2 * U_v[v], name=f"D19_7_v_{v}")


for k in S:
    m.addConstr(gp.quicksum(gp.quicksum(x_ij_t[i, k, t] for t in T) for i in D) - gp.quicksum(d_c[j] * w_i_j[j, k] for j in C) == 0, name=f"D19_8_k_{k}")


for i in D:
    for j in S:
        for t in T:
            m.addConstr(x_ij_t[i, j, t] >= 0, name=f"D19_9_i_{i}_j_{j}_t_{t}")


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

    pd.DataFrame(varInfo, index = ['value']).T.to_excel('solution_NC_d19.xlsx')

else:
    print(f"Optimization ended with status {m.status}.")