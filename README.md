# Collaborative-Last-Mile-Delivery
 
An implementation of [this paper](https://doi.org/10.1016/j.eswa.2024.124164).


PHASE 2 - clustered satellite allocation seems inefficient. (Just says make satellite pairs which are closest to each other, but that might not be the best way to do it.)


Mistakes-

- subscript mismatch in the objective function (X_ij_v instead of X_ij_t)

- Constraint 2 - negative sign instead of positive sign
Flow conservation requires that the flow into a node is equal to the flow out of the node. Since R_ij_t is a binary variable, the summation of in flows for a given node j should be equal to the summation of out flows for that node, so the equation should be:
R_ij_t - R_ji_t = 0


- Constraint 8 - summation R_ij_t, since for all t was missing 


- Constraint 10 - same as constraint 2


- Constraint 31 - nc should be number of customers + collaboration points instead of just number of customers


- Constraint 27 - SUC instead of SUCUO (X_ij_v <= 2)


Question-

- The value of M effects the optimal solutions. Why?

- Constraint SecondEchelon21 - sum over C Q_c_s * dc <= A_s (but if Q_c_s represents the updated customers after the exchange, then it doesn't work, so maybe it's just the original customers)

- Constraint SecondEchelon14 - same issue as above

- Constraint SecondEchelon4 - gp.quicksum(X_ij_v[c, k, v] for k in SUCUO) == Y_c_v[c, v], but X_ij_v stores the path, hence has exchanged customers, which ultimately points that Y_c_v represents updated customers after the exchange 

- Constraint SecondEchelon9 - wants to have updated Q_c_s


TO SOLVE THE Q_c_s ISSUE-
1. We can have a separate variable to store the updated customers after the exchange
2. We can remove the capacity constraints restricting the number of customers in the vehicle (SecondEchelon21 and SecondEchelon14)


- MORE ISSUES -

- Constraint SE9 and SE10-SE13 do not sit together. 

SE9 uses Big M to ensure 3 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, s], meaning for a given customer, satellite, vehicle (c, s, v), customer c is assigned to satellite s (assumed to be after exchange), vehicle v is assigned to customer c (assumed to be after exchange), and vehicle v is assigned to satellite s (original vehicle of the satellite, since vehicles are not being swapped).

SE10-SE13 used Big M to ensure 3 - Q_c_s[c, s] - Y_c_v[c, v] - B_v_s[v, e], meaning that for a given customer, satellite, vehicle (c, s, v), customer c is assigned to satellite s (assumed to be after exchange), vehicle v is assigned to customer c (assumed to be after exchange), and vehicle v is assigned to satellite e (e not equal to s, meaning vehicle assigned to some other satellite).



Omitted Variables from Original Problem Formulation: (Which the paper takes inspiration from)

- service time of satellite

- time window for each customer [az,bz]

- service time at customer location sz

- synchronization time between first echelon vehicle reaching the satellite and the second echelon vehicle leaving the satellite
