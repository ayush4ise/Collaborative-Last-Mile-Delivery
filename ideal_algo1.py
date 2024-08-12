import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def allocate_customers_to_satellites(LSPs, customers, satellites, capacities):
    N = len(LSPs)  # total number of logistics service providers
    
    for i in range(N):
        # Allocate customers to the nearest satellite
        for customer in customers[i]:
            distances = [euclidean_distance(customer, satellite) for satellite in satellites[i]]
            nearest_satellite = np.argmin(distances)
            satellites[i][nearest_satellite].append(customer)
        
        # Check if any satellite exceeds its capacity
        if any(len(sat) > capacities[i] for sat in satellites[i]):
            for customer in customers[i]:
                D = []
                for sat_index in range(len(satellites[i])):
                    for other_sat_index in range(sat_index + 1, len(satellites[i])):
                        d = euclidean_distance(satellites[i][sat_index], customer) - euclidean_distance(satellites[i][other_sat_index], customer)
                        D.append((d, sat_index, other_sat_index))
                
                # Sort satellites according to D
                D.sort()
                
                # Reallocate customers to the satellite where capacity is available
                for _, sat_index, other_sat_index in D:
                    if len(satellites[i][other_sat_index]) < capacities[i]:
                        satellites[i][other_sat_index].append(satellites[i][sat_index].pop())
                        break

    return satellites

# Example usage
LSPs = ['LSP1', 'LSP2']  # Example LSPs
customers = [[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10), (11, 12)]]  # Example customer coordinates per LSP
satellites = [[[(1, 2)], [(3, 4)]], [[(7, 8)], [(9, 10)]]]  # Example satellite coordinates per LSP
capacities = [5, 5]  # Example satellite capacities

allocated_satellites = allocate_customers_to_satellites(LSPs, customers, satellites, capacities)
print(allocated_satellites)
