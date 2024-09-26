import pandas as pd
import numpy as np
import random

# Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def algorithm_1(satellites, customers, collaboration_points):
    sat = satellites.copy()
    cust = customers.copy()
    collab_pts = collaboration_points.copy()

    satellite_capacity = 6

    lsps = cust['lsp'].unique()

    origin_satellite_total = pd.DataFrame()

    for lsp in lsps:
        origin_satellite = []
        # filter customers by lsp
        cust_lsp = cust[cust['lsp'] == lsp].copy()
        # filter satellites by lsp
        sat_lsp = sat[sat['lsp'] == lsp].copy()

        #print('LSP:', lsp)
        #print('Customers:', cust_lsp)
        #print('Satellites:', sat_lsp)

        # greedy allocation of customers to satellites
        for i, c in cust_lsp.iterrows():
            compare_distance = {}
            for j, s in sat_lsp.iterrows():
                compare_distance[j] = euclidean_distance(c['X'], c['Y'], s['X'], s['Y'])
            compare_distance = dict(sorted(compare_distance.items(), key=lambda item: item[1]))
            j = list(compare_distance.keys())[0]

            # assign customer to satellite
            origin_satellite.append({'customer': i, 'satellite': j})

        original_satellite = pd.DataFrame(origin_satellite)
        #print('Original Satellite:', original_satellite)

        # reallocation of customer to balance demand
        # there are only 2 satellites per LSP
        capacity = original_satellite['satellite'].value_counts().to_dict()
        #print('Capacity:', capacity)   
        for i, c in capacity.items():
            if c > satellite_capacity:
                # filter customers allocated to satellite
                cust_sat = original_satellite[original_satellite['satellite'] == i].copy()

                # sort customers by distance to satellite
                cust_sat['distance'] = cust_sat.apply(lambda x: euclidean_distance(cust_lsp.loc[x['customer'], 'X'], cust_lsp.loc[x['customer'], 'Y'], sat_lsp.loc[i, 'X'], sat_lsp.loc[i, 'Y']), axis=1)
                cust_sat = cust_sat.sort_values(by='distance', ascending=False)  # Sort by farthest distance
                
                # Get the other satellite for the same LSP
                other_satellite = sat_lsp[sat_lsp.index != i].index[0]

                extra = c - satellite_capacity
                while extra > 0:
                    # Reallocate the farthest customer
                    customer_to_reallocate = cust_sat.iloc[0]
                    original_satellite.loc[original_satellite['customer'] == customer_to_reallocate['customer'], 'satellite'] = other_satellite
                    #print('Updated Satellite:', original_satellite)

                    # Update capacities
                    capacity[i] -= 1
                    capacity[other_satellite] += 1

                    # Remove reallocated customer from the list
                    cust_sat = cust_sat.iloc[1:]
                    extra -= 1
            
        origin_satellite_total = pd.concat([origin_satellite_total, original_satellite])

    # add origin_satellite column to cust, and match the 'customer' column to get index in cust 
    cust['origin_satellite'] = None
    for i, c in origin_satellite_total.iterrows():
        cust.loc[c['customer'], 'origin_satellite'] = c['satellite']
                    
    #return origin_satellite_total
    return cust

def algorithm_2(satellites, customers, collaboration_points):
    sat = satellites.copy()
    cust = customers.copy() # includes origin_satellite
    collab_pts = collaboration_points.copy()

    # Form a pair of satellites 𝑆1 and 𝑆2 such that they are closest to each other but belong to different logistics service providers
    # sat_index = sat.index.tolist()
    sat_pairs = {}
    for i, sat1 in sat.iterrows():
        for j, sat2 in sat.iterrows():
            if sat1['lsp'] != sat2['lsp']:
                distance = euclidean_distance(sat1['X'], sat1['Y'], sat2['X'], sat2['Y'])
                sat_pairs[(i,j)] = distance

    sat_pairs = dict(sorted(sat_pairs.items(), key=lambda x: x[1]))
    #print(sat_pairs)

    final_sat_pairs = []
    all_sats = sat.index.tolist()
    for sat_pair in sat_pairs.keys():
        if sat_pair[0] in all_sats and sat_pair[1] in all_sats:
            final_sat_pairs.append(sat_pair)
            all_sats.remove(sat_pair[0])
            all_sats.remove(sat_pair[1])
    #print(final_sat_pairs)

    # Assign collaboration point 𝑂1 closest to the pair of satellites 𝑆1 and 𝑆2
    collab_pt_o1 = {}
    for sat_pair in final_sat_pairs:
        sat1 = sat.loc[sat_pair[0]]
        sat2 = sat.loc[sat_pair[1]]
        collab_pts['dist_o1'] = collab_pts.apply(lambda x: euclidean_distance(x['X'], x['Y'], sat1['X'], sat1['Y']) + euclidean_distance(x['X'], x['Y'], sat2['X'], sat2['Y']), axis=1)
        collab_pt_o1[sat_pair] = collab_pts['dist_o1'].idxmin()
    #print(collab_pt_o1)

    # Consider customers 𝐶 ∈ {𝐶1, 𝐶2, ..., 𝐶𝑛} whose origin satellites are 𝑆1 and 𝑆2
    # For all customers- 
    # Assign clustered sat S1 or S2 to the customer based on euclidean distance
    # Assign 𝑂1 as collaboration point to the customer

    for sat_pair in final_sat_pairs:
        sat1 = sat.loc[sat_pair[0]]
        sat2 = sat.loc[sat_pair[1]]
        for i,c in cust.iterrows():
            if c['origin_satellite'] == sat_pair[0] or c['origin_satellite'] == sat_pair[1]:
                cust.loc[i, 'clustered_sat'] = sat_pair[0] if euclidean_distance(c['X'], c['Y'], sat1['X'], sat1['Y']) < euclidean_distance(c['X'], c['Y'], sat2['X'], sat2['Y']) else sat_pair[1]
                cust.loc[i, 'collab_pt'] = collab_pt_o1[sat_pair]
    #print(cust)
    
    return {
        'cust': cust,
        'final_sat_pairs': final_sat_pairs
    }


# to calculate the traveled distance of a vehicle visiting multiple points
def travel_distance(points, data):
    distance = 0
    for i in range(len(points)-1):
        distance += euclidean_distance(data.loc[points[i], 'X'], data.loc[points[i], 'Y'], data.loc[points[i+1], 'X'], data.loc[points[i+1], 'Y'])
    return distance

# swap random two points in a list, if the new list has a shorter distance, return the new list
# if the list has 'CP' (collaboration point), split the list into two lists at 'CP', perform swap on either list and merge them back
def swap(points, data):
    if 'CP' in points:
        cp_index = points.index('CP')
        points1 = points[:cp_index]
        points2 = points[cp_index+1:]
        new_points1 = swap(points1, data)
        new_points2 = swap(points2, data)
        new_points = new_points1 + ['CP'] + new_points2
        return new_points
    else:
        new_points = points.copy()
        i, j = random.sample(range(len(points)), 2)
        new_points[i], new_points[j] = new_points[j], new_points[i]
        if travel_distance(new_points, data) < travel_distance(points, data):
            return new_points
        else:
            return points
        

def algorithm_4(vehicle_routes, data, n=5):
    n = n # number of iterations for the algorithm
    # use swap algorithm and replace the route with the new route if the new route is shorter (less traveled distance)
    for i in range(n):
        for vehicle in vehicle_routes:
            vehicle['route'] = swap(vehicle['route'], data)
    return vehicle_routes