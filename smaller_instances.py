import gurobipy as gp

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
D = [1, 2] # list of depots
S = [3, 4, 5, 6] # list of satellites
C = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # list of customers
O = [19, 20] # list of collaboration points

T = [1,2] # list of first echelon vehicles
V = [3, 4] # list of second echelon vehicles


DUS = D + S # list of depots and satellites
SUCUO = S + C + O # list of satellites, customers, and collaboration points
CUO = C + O # list of customers, and collaboration points

DUS_1 = [1, 3, 5] # list of depots and satellites for lsp 1
DUS_2 = [2, 4, 6] # list of depots and satellites for lsp 2

C1 = [8, 10, 12, 14, 16, 18] # list of customers for lsp 1
C2 = [7, 9, 11, 13, 15, 17] # list of customers for lsp 2

S1 = [3, 5] # list of satellites for lsp 1
S2 = [4, 6] # list of satellites for lsp 2

# Parameters
# Given in Table 4
C_ij = np.array(distances) # cost to travel from i to j

F_t = {1:100, 2:100} # fixed cost of first echelon vehicles
F_v = {3:50, 4:50} # fixed cost of second echelon vehicles

d_c = {7:30, 8:30, 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30, 16:30, 17:30, 18:30} # demand of customer c
A_s = {3:30, 4:30, 5:30, 6:30} # capacity of satellite s
K1 = 60 # capacity of first echelon vehicles
K2 = 30 # capacity of second echelon vehicles
p_c = {7:2, 8:1, 9:2, 10:1, 11:2, 12:1, 13:2, 14:1, 15:2, 16:1, 17:2, 18:1} # DC to which customer c belongs to



# Assumption
M = 10000  # A sufficiently large constant



# Variables
# Given in Table 5

# 1 if first echelon vehicle ‘t’ is moving from node ‘i’ to ‘j’, 0 otherwise
R_ij_t = m.addVars(DUS, DUS, T, vtype=gp.GRB.BINARY, name="R_ij_t") 

# 1 if second echelon vehicle ‘v’ is moving from node ‘i’ to ‘j’, 0 otherwise
X_ij_v = m.addVars(SUCUO, SUCUO, V, vtype=gp.GRB.BINARY, name="X_ij_v")

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

# 1 if vehicle ‘v’ is assigned to satellite ‘s’, 0 otherwise
B_v_s = m.addVars(V, S, vtype=gp.GRB.BINARY, name="B_v_s")

# 1 if vehicle ‘v’ is assigned to satellite ‘s’ and visiting collaboration point ‘o’, 0 otherwise
K_sv_o = m.addVars(V, S, O, vtype=gp.GRB.BINARY, name="K_sv_o")

# Number of goods moving from node ‘i’ to ‘j’ in vehicle ‘v’
H_ij_v = m.addVars(SUCUO, SUCUO, V, vtype=gp.GRB.INTEGER, name="H_ij_v")

# Variables to avoid subtour in first echelon
G_i_t = m.addVars(S, T, vtype=gp.GRB.INTEGER, name="G_i_t")

# Variables to avoid subtour in second echelon
G_i_v = m.addVars(CUO, V, vtype=gp.GRB.INTEGER, name="G_i_v")



# Objective Function
m.setObjective(
    gp.quicksum(C_ij[i, j] * R_ij_t[i, j, t] for i in DUS for j in DUS for t in T) +
    gp.quicksum(C_ij[i, j] * X_ij_v[i, j, v] for i in SUCUO for j in SUCUO for v in V) +
    gp.quicksum(F_t[t] * U_t[t] for t in T) + 
    gp.quicksum(F_v[v] * U_v[v] for v in V),
    sense=gp.GRB.MINIMIZE
)



# Constraints
# Constraints in first echelon

for j in DUS:
    for t in T:
        m.addConstr(gp.quicksum(R_ij_t[l, j, t] for l in DUS) + gp.quicksum(R_ij_t[j, l, t] for l in DUS) == 0, name=f"FirstEchelon1_j_{j}_t_{t}")


for t in T:
    m.addConstr(gp.quicksum((gp.quicksum(R_ij_t[i, j, t] for j in D) for i in S) <= U_t[t] , name=f"FirstEchelon2_t_{t}"))


for c in C:
    for t in T:
        m.addConstr(gp.quicksum(R_ij_t[p_c[c], s, t] for s in S) >= Y_c_t, name = f"FirstEchelon3_c_{c}_t_{t}")


for t in T:
    m.addConstr(gp.quicksum((d_c[c] * Y_c_t[c][t]) for c in C) <= K1 * U_t[t], name=f"FirstEchelon4_t_{t}")


for c in C:
    m.addConstr(gp.quicksum(Y_c_t[c][t] for t in T) == 1, name=f"FirstEchelon5_c_{c}")


for c in C:
    for s in S:
        for t in T:
            m.addConstr(M * (2 - Y_c_t[c][t] - Q_c_s[c][s]) + gp.quicksum(R_ij_t[s, k, t] for k in DUS) >= 1, name=f"FirstEchelon6_c_{c}_s_{s}_t_{t}")


for i in DUS_1:
    for j in DUS_2:
        for t in T:
            m.addConstr(R_ij_t[i, j, t] == 0, name=f"FirstEchelon7_i_{i}_j_{j}_t_{t}")


for c in C1:
    m.addConstr(gp.quicksum(Q_c_s[c][s] for s in S1) == 1, name=f"FirstEchelon8_c_{c}")

for c in C2:
    m.addConstr(gp.quicksum(Q_c_s[c][s] for s in S2) == 1, name=f"FirstEchelon9_c_{c}")