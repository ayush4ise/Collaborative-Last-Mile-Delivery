import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np

from plotting import *

instance = 'A1'

data = pd.read_excel("Data\Instances\Small_Instances.xlsx", sheet_name=instance, index_col=0)
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
V = [3, 4, 5, 6] # list of second echelon vehicles


DUS = D + S # list of depots and satellites
SUCUO = S + C + O # list of satellites, customers, and collaboration points
CUO = C + O # list of customers, and collaboration points
SUC = S + C # list of satellites, customers
ns = len(S) # number of satellites
nc = len(C) + 2 # number of customers + collaboration points

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
F_v = {3:50, 4:50, 5:50, 6:50} # fixed cost of second echelon vehicles

d_c = {7:10, 8:10, 9:10, 10:10, 11:10, 12:10, 13:10, 14:10, 15:10, 16:10, 17:10, 18:10} # demand of customer c
A_s = {3:30, 4:30, 5:30, 6:30} # capacity of satellite s
K1 = 60 # capacity of first echelon vehicles
K2 = 30 # capacity of second echelon vehicles
p_c = {7:2, 8:1, 9:2, 10:1, 11:2, 12:1, 13:2, 14:1, 15:2, 16:1, 17:2, 18:1} # DC to which customer c belongs to



# Assumption
M = 100000 # A sufficiently large constant



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
K_sv_o = m.addVars(S, V, O, vtype=gp.GRB.BINARY, name="K_sv_o")

# Number of goods moving from node ‘i’ to ‘j’ in vehicle ‘v’
H_ij_v = m.addVars(SUCUO, SUCUO, V, vtype=gp.GRB.INTEGER, name="H_ij_v")

# Variables to avoid subtour in first echelon
G_i_t = m.addVars(S, T, vtype=gp.GRB.CONTINUOUS, name="G_i_t")

# Variables to avoid subtour in second echelon
G_i_v = m.addVars(CUO, V, vtype=gp.GRB.CONTINUOUS, name="G_i_v")



# Objective Function
m.setObjective(
    gp.quicksum(C_ij[i-1, j-1] * R_ij_t[i, j, t] for i in DUS for j in DUS for t in T) +
    gp.quicksum(C_ij[i-1, j-1] * X_ij_v[i, j, v] for i in SUCUO for j in SUCUO for v in V) +
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


# for i in DUS_1:
#     for j in DUS_2:
#         for t in T:
#             m.addConstr(R_ij_t[i, j, t] == 0, name=f"FirstEchelon7_i_{i}_j_{j}_t_{t}")
for i in DUS_1:
    for j in DUS_2:
        m.addConstr(gp.quicksum(R_ij_t[i, j, t] for t in T) == 0, name=f"FirstEchelon7_i_{i}_j_{j}")


for c in C1:
    m.addConstr(gp.quicksum(Q_c_s[c, s] for s in S1) == 1, name=f"FirstEchelon8_c_{c}")

for c in C2:
    m.addConstr(gp.quicksum(Q_c_s[c, s] for s in S2) == 1, name=f"FirstEchelon8_c_{c}")


# Constraints in second echelon

# for j in SUCUO:
#     for v in V:
#         m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUCUO) + gp.quicksum(X_ij_v[j, l, v] for l in SUCUO) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")
for j in SUCUO:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[l, j, v] for l in SUCUO) - gp.quicksum(X_ij_v[j, l, v] for l in SUCUO) == 0, name=f"SecondEchelon1_j_{j}_v_{v}")


for v in V:
    m.addConstr(gp.quicksum((gp.quicksum(X_ij_v[i, j, v] for j in S)) for i in CUO) <= U_v[v], name=f"SecondEchelon2_v_{v}")


for c in C:
    m.addConstr(gp.quicksum(Y_c_v[c, v] for v in V) >= 1, name=f"SecondEchelon3_c_{c}")


for c in C:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[c, k, v] for k in SUCUO) == Y_c_v[c, v], name=f"SecondEchelon4_c_{c}_v_{v}")


for v in V:
    m.addConstr(gp.quicksum(B_v_s[v, s] for s in S) == 1, name=f"SecondEchelon5_v_{v}")


for s in S:
    for v in V:
        m.addConstr(gp.quicksum(X_ij_v[s, j, v] for j in CUO) == B_v_s[v, s], name=f"SecondEchelon6_s_{s}_v_{v}")


for v in V:
    m.addConstr(gp.quicksum((gp.quicksum(K_sv_o[s, v, o] for o in O)) for s in S) <= 1, name=f"SecondEchelon7_v_{v}")


for v in V:
    for s in S:
        for o in O:
            m.addConstr(M * (1 - K_sv_o[s, v, o]) + gp.quicksum(X_ij_v[s, j, v] for j in CUO) + gp.quicksum(X_ij_v[i, o, v] for i in SUC) >= 2, name=f"SecondEchelon8_v_{v}_s_{s}_o_{o}")


# for v in V:
#     for s in S:
#         for c in C:
#             m.addConstr(M * (3 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, s]) + gp.quicksum(X_ij_v[i, c, v] for i in SUC) >= 1, name=f"SecondEchelon9_v_{v}_s_{s}_c_{c}")


for v in V:
    for c in C:
        for s in S:
            for e in S:
                if s!=e:
                    m.addConstr(M * (3 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, e]) + gp.quicksum(K_sv_o[e, v, o] for o in O) >= 1, name=f"SecondEchelon10_v_{v}_c_{c}_s_{s}_e_{e}")


for v in V:
    for o in O:
        for c in C1:
            for s in S:
                for e in S:
                    if s!=e:
                        m.addConstr(M * (4 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, e] - K_sv_o[e, v, o]) + gp.quicksum(X_ij_v[f, c, v] for f in CUO)  >= 1, name=f"SecondEchelon11_v_{v}_o_{o}_c_{c}_s_{s}_e_{e}")

for v in V:
    for o in O:
        for c in C2:
            for s in S:
                for e in S:
                    if s!=e:
                        m.addConstr(M * (4 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, e] - K_sv_o[e, v, o]) + gp.quicksum(X_ij_v[f, c, v] for f in CUO)   >= 1, name=f"SecondEchelon12_v_{v}_o_{o}_c_{c}_s_{s}_e_{e}")


for v in V:
    for o in O:
        for c in C:
            for s in S:
                for e in S:
                    if s!=e:
                        m.addConstr(M * (4 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, e] - K_sv_o[e, v, o]) + gp.quicksum(K_sv_o[s, f, o] for f in V) >= 1, name=f"SecondEchelon13_v_{v}_o_{o}_c_{c}_s_{s}_e_{e}")


for s in S:
    m.addConstr(gp.quicksum(Q_c_s[c, s] * d_c[c] for c in C) == gp.quicksum((gp.quicksum(H_ij_v[s, j, v] for j in CUO)) for v in V), name=f"SecondEchelon14_s_{s}")


for c in C:
    m.addConstr(gp.quicksum((gp.quicksum(H_ij_v[i, c, v] for i in SUCUO)) for v in V) - gp.quicksum((gp.quicksum(H_ij_v[c, j, v] for j in SUCUO)) for v in V) == d_c[c], name=f"SecondEchelon15_c_{c}")


for i in SUCUO:
    for j in SUCUO:
        if i!=j:
            for v in V:
                m.addConstr(H_ij_v[i, j, v] <= K2 * X_ij_v[i, j, v], name=f"SecondEchelon16_i_{i}_j_{j}_v_{v}")


for o in O:
    for v in V:
        for k in V:
            if v!=k:
                m.addConstr(M * (2 - gp.quicksum(X_ij_v[i, o, v] for i in SUC) - gp.quicksum(X_ij_v[i, o, k] for i in SUC)) + gp.quicksum(H_ij_v[i, o, v] for i in SUC) >= gp.quicksum(H_ij_v[o, j, k] for j in SUC), name=f"SecondEchelon17_o_{o}_v_{v}_k_{k}")


# for o in O:
#     for v in V:
#         for k in V:
#             if v!=k:
                m.addConstr(M * (2 - gp.quicksum(X_ij_v[i, o, v] for i in SUC) - gp.quicksum(X_ij_v[i, o, k] for i in SUC)) + gp.quicksum(H_ij_v[i, o, k] for i in SUC) >= gp.quicksum(H_ij_v[o, j, v] for j in SUC), name=f"SecondEchelon18_o_{o}_v_{v}_k_{k}")    


for o in O:
    m.addConstr(gp.quicksum((gp.quicksum(X_ij_v[i, o, v] for v in V)) for i in SUC) <= 2, name=f"SecondEchelon19_o_{o}")


for v in V:
    m.addConstr(gp.quicksum((gp.quicksum (H_ij_v[i, j, v] for j in S)) for i in CUO) == 0, name=f"SecondEchelon20_v_{v}")


for s in S:
    m.addConstr(gp.quicksum(Q_c_s[c, s] * d_c[c] for c in C) <= A_s[s], name=f"SecondEchelon21_s_{s}")



for i in S:
    for j in S:
        for t in T:
            m.addConstr(G_i_t[i, t] - G_i_t[j, t] + ns * R_ij_t[i, j, t] <= ns - 1, name=f"SecondEchelon22_i_{i}_j_{j}_t_{t}" )


for i in CUO:
    for j in CUO:
        for v in V:
            m.addConstr(G_i_v[i, v] - G_i_v[j, v] + nc * X_ij_v[i, j, v] <= nc - 1, name=f"SecondEchelon23_i_{i}_j_{j}_v_{v}")



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

    # for c in m.getConstrs():
    #     if c.Slack > 1e-6:
    #         print('Constraint %s is not active at solution point' % (c.ConstrName))

    solution_variables = pd.DataFrame(varInfo, index = ['value']).T
    solution_variables.to_excel(f'solution({instance}) - {m.ObjVal:0.3f}.xlsx')
    route_plot(data, solution_variables, instance)

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