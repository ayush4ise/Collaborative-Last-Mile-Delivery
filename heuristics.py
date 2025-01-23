import gurobipy as gp
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

instance = 'A3'

data = pd.read_excel("Data\Instances\Small_Instances.xlsx", sheet_name=instance, index_col=0)
customers = data[data['designation'] == 'c']
KM = KMeans(n_clusters=3, max_iter=500, random_state=42)
KM.fit(customers[['X', 'Y']])

data = data[['X', 'Y']]
data.loc[19] = KM.cluster_centers_[0]
data.loc[20] = KM.cluster_centers_[1]
data.loc[21] = KM.cluster_centers_[2]

def EuclideanDistance(x1, y1, x2, y2):
    import math
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

def route_plot(data, solution_variables, instance):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    # plot depots [1, 2]
    # square shape, opacity 0.5
    plt.scatter(data['X'][1], data['Y'][1], color='blue', label='Depot', s=400, marker='s', alpha = 0.5)
    plt.scatter(data['X'][2], data['Y'][2], color='orange', label='Depot', s=400, marker='s', alpha = 0.5)

    # plot satellites [3, 4, 5, 6] (3, 5 belong to LSP 1, 4, 6 belong to LSP 2)
    plt.scatter(data['X'][3], data['Y'][3], color='blue', label='Satellite', s=400, marker='^', alpha = 0.8)
    plt.scatter(data['X'][4], data['Y'][4], color='orange', label='Satellite', s=400, marker='^', alpha = 0.8)
    plt.scatter(data['X'][5], data['Y'][5], color='blue', label='Satellite', s=400, marker='^', alpha = 0.8)
    plt.scatter(data['X'][6], data['Y'][6], color='orange', label='Satellite', s=400, marker='^', alpha = 0.8)

    # plot customers [7-18] (even numbers belong to LSP 1, odd numbers belong to LSP 2) (colour according to LSP)
    for i in range(7, 19):
        if i % 2 == 0:
            plt.scatter(data['X'][i], data['Y'][i], color='blue', label='Customer', s=200, alpha = 0.5)
        else:
            plt.scatter(data['X'][i], data['Y'][i], color='orange', label='Customer', s=200, alpha = 0.5)

    # plot collaboration points [19, 20] as green diamonds
    plt.scatter(data['X'][19], data['Y'][19], color='green', label='Collaboration Point', s=200, marker='D')
    plt.scatter(data['X'][20], data['Y'][20], color='green', label='Collaboration Point', s=200, marker='D')
    plt.scatter(data['X'][21], data['Y'][21], color='green', label='Collaboration Point', s=200, marker='D')


    # plt.scatter(data['X'], data['Y'], color='blue')
    for i in data.index:
        plt.annotate(i, (data['X'][i], data['Y'][i]), color='black')

    # first-echelon
    for variable_name in solution_variables.index:
        if variable_name[0] == 'R':
            point_a = int(variable_name[7])
            point_b = int(variable_name[9])
            if point_a % 2 != 0:

                plt.arrow(data['X'][point_a], 
                        data['Y'][point_a], 
                        data['X'][point_b] - data['X'][point_a], 
                        data['Y'][point_b] - data['Y'][point_a], 
                        color='purple',
                        head_width=0.35,
                        lw = 2)

            else:
                plt.arrow(data['X'][point_a],
                        data['Y'][point_a],
                        data['X'][point_b] - data['X'][point_a], 
                        data['Y'][point_b] - data['Y'][point_a], 
                        color='red',
                        head_width=0.35,
                        lw = 2)
            
    # second-echelon
    for variable_name in solution_variables.index:
        if variable_name[0] == 'X':
            point_a = int(variable_name.split(',')[0].split('[')[-1])
            point_b = int(variable_name.split(',')[1])
            vehicle = int(variable_name.split(',')[2][0])
            vehicle_colour_map = {3:'magenta', 4:'turquoise', 5:'green', 6:'black'}
            plt.arrow(data['X'][point_a], 
                    data['Y'][point_a], 
                    data['X'][point_b] - data['X'][point_a], 
                    data['Y'][point_b] - data['Y'][point_a], 
                    color= vehicle_colour_map[vehicle],
                    head_width=0.35)
            
    legend_entries = [
        plt.Line2D([0], [0], color='red', marker='s', markersize=10, label='Depot'),
        plt.Line2D([0], [0], color='red', marker='^', markersize=10, label='Satellite'),
        plt.Line2D([0], [0], color='red', marker='o', markersize=10, label='Customer'),
        plt.Line2D([0], [0], color='red', marker='D', markersize=10, label='Collaboration Point'),
        # add points for LSPs without any coordinates on the plot
        plt.Line2D([0], [0], color='blue', marker='o', markersize=10, label='LSP 1'),
        plt.Line2D([0], [0], color='orange', marker='o', markersize=10, label='LSP 2'),
    ]
    # add legend mentioning that squares are depots, triangles are satellites, circles are customers, and diamonds are collaboration points, arrows are routes, and colours are according to LSPs
    plt.legend(handles=legend_entries)
    plt.title(f'Route Plot for {instance}')
    plt.savefig(f'route_plot({instance}).png')
    plt.close()

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
O = [19, 20, 21] # list of collaboration points

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
            m.addConstr(G_i_v[i, v] - G_i_v[j, v] + (nc+2) * X_ij_v[i, j, v] <= (nc+2) - 1, name=f"SecondEchelon23_i_{i}_j_{j}_v_{v}")



# Optimize model
m.optimize()

# Output the results
if m.status == gp.GRB.OPTIMAL:
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
    solution_variables.to_excel(f'KMeans_solution({instance}) - {m.ObjVal:0.3f}.xlsx')
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