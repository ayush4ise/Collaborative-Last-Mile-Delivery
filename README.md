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



Question-

- The value of M effects the optimal solutions. Why?


Omitted Variables from Original Problem Formulation: (Which the paper takes inspiration from)

- service time of satellite

- time window for each customer [az,bz]

- service time at customer location sz

- synchronization time between first echelon vehicle reaching the satellite and the second echelon vehicle leaving the satellite
