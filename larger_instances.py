# # Inputs are all dictionaries with the following structure:
# customers = {}
# # customers['A'] = {'demand': 10, 'x': 0, 'y': 0, 'lsp': 1}
# depots = {}
# # depots['D'] = {'x': 0, 'y': 0, 'lsp': 1}
# satellites = {}
# # satellites['S'] = {'x': 0, 'y': 0, 'lsp' = 1}
# collaboration_points = {}
# # collaboration_points['C'] = {'x': 0, 'y': 0}
# vehicles = {}
# # vehicles['V'] = {'capacity': 30, satellite': 'S', 'lsp': 1}


def InputFetcher(path:str, instance:str):
    """
    Fetches the input from the Excel file and returns the dictionaries of customers, depots, satellites, collaboration points

    Parameters
    ----------
    path : str
        The path of the file containing the input
    instance : str
        The name of the instance

    Returns
    -------
    customers : dict
        The dictionary containing the customers
    depots : dict
        The dictionary containing the depots
    satellites : dict
        The dictionary containing the satellites
    collaboration_points : dict
        The dictionary containing the collaboration points
    """
    import pandas as pd

    input_file = pd.read_excel(path, sheet_name=instance, index_col=0)

    # change column names to lowercase
    input_file.columns = input_file.columns.str.lower()

    # Fetching the customers
    customers = input_file[input_file['designation']=='c'].drop(columns=['designation']).to_dict(orient='index')

    # Fetching the depots
    depots = input_file[input_file['designation']=='d'].drop(columns=['designation', 'demand']).to_dict(orient='index')

    # Fetching the satellites
    satellites = input_file[input_file['designation']=='s'].drop(columns=['designation', 'demand']).to_dict(orient='index')

    # Fetching the collaboration points
    collaboration_points = input_file[input_file['designation']=='z'].drop(columns=['designation', 'demand', 'lsp']).to_dict(orient='index')

    return customers, depots, satellites, collaboration_points

def VehicleDict(satellites, instance:str):
    """
    Assigns the vehicles to the satellites based on the instance and returns the dictionary of vehicles

    Parameters
    ----------
    satellites : dict
        The dictionary containing the satellites
    instance : str
        The name of the instance

    Returns
    -------
    vehicles : dict
        The dictionary containing the vehicles
    """

    vehicles = {}
    if instance[0] == 'A': # small instances, 1 vehicle per satellite
        for satellite in satellites:
            vehicles[satellite] = {'capacity': 30, 'satellite': satellite, 'lsp': satellites[satellite]['lsp']}

    elif instance[0] == 'B': # large instances, 2 vehicles per satellite
        for satellite in satellites:
            vehicles[(satellite, 1)] = {'capacity': 30, 'satellite': satellite, 'lsp': satellites[satellite]['lsp']}
            vehicles[(satellite, 2)] = {'capacity': 30, 'satellite': satellite, 'lsp': satellites[satellite]['lsp']}

    return vehicles

def EuclideanDistance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points

    Parameters
    ----------
    x1 : float
        The x-coordinate of the first point
    y1 : float
        The y-coordinate of the first point
    x2 : float
        The x-coordinate of the second point
    y2 : float
        The y-coordinate of the second point

    Returns
    -------
    distance : float
        The Euclidean distance between the two points
    """
    import math

    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return distance

# Algorthm 1 - Phase A
def algorithm1(satellites, customers): 
    """
    An improved greedy algorithm for allocation and balancing
    
    Parameters
    ----------
    satellites : dict
        The dictionary containing the satellites
    customers : dict
        The dictionary containing the customers

    Returns
    -------
    customers : dict
        The dictionary containing the customers with the updated origin satellite                                 
    """
    # set N <- total no. of lsps (2)
    N = 2

    # loop for each lsp
    for i in range(N):
        lsp_customers = {key: value for key, value in customers.items() if value['lsp'] == i+1}
        lsp_satellites = {key: value for key, value in satellites.items() if value['lsp'] == i+1}

        ############################################################################################################
        # lspi_satellite = {Sat1: {'x':, 'y':, 'lsp':}, Sat2: {'x':, 'y':, 'lsp':}}
        # lspi_customers = {Cust1: {'demand':, 'x':, 'y':, 'lsp':}, Cust2: {'demand':, 'x':, 'y':, 'lsp':}}
        ############################################################################################################

        # for all customers of lsp i, assign them to the nearest satellite considering the Euclidean distance
        for customer in lsp_customers:
            min_dist = float('inf')
            for satellite in lsp_satellites:
                dist = EuclideanDistance(lsp_customers[customer]['x'], lsp_customers[customer]['y'], lsp_satellites[satellite]['x'], lsp_satellites[satellite]['y'])
                if dist < min_dist:
                    min_dist = dist
                    lsp_customers[customer]['origin_satellite'] = satellite
            lsp_satellites[lsp_customers[customer]['origin_satellite']]['origin_customers'] = lsp_satellites[lsp_customers[customer]['origin_satellite']].get('origin_customers', []) + [customer]

        ############################################################################################################
        # lspi_customers = {Cust1: {'demand':, 'x':, 'y':, 'lsp':, 'origin_satellite':}, Cust2: {'demand':, 'x':, 'y':, 'lsp':, 'origin_satellite':}}
        # lspi_satellite = {Sat1: {'x':, 'y':, 'lsp':, 'origin_customers':}, Sat2: {'x':, 'y':, 'lsp':, 'origin_customers':}}
        ############################################################################################################

        # if demand assigned to a satellite exceeds its capacity, reallocation of customers to the satellite where capacity is available
        # satellite capacity, As <- 60 (which is 6 customers)
        As = 60

        for satellite in lsp_satellites:
            if sum([lsp_customers[customer]['demand'] for customer in lsp_satellites[satellite].get('origin_customers', [])]) > As:
                # filter customers allocated to satellite
                customers_sat = {key: value for key, value in lsp_customers.items() if key in lsp_satellites[satellite].get('origin_customers', [])}
                # sort customers by distance to satellite
                for customer in customers_sat:
                    customers_sat[customer]['distance'] = EuclideanDistance(customers_sat[customer]['x'], customers_sat[customer]['y'], lsp_satellites[satellite]['x'], lsp_satellites[satellite]['y'])
                customers_sat = dict(sorted(customers_sat.items(), key=lambda item: item[1]['distance'], reverse=True))
                # get the other satellite for the same LSP
                other_satellite = [key for key in lsp_satellites if key != satellite][0]

                extra = sum([lsp_customers[customer]['demand'] for customer in lsp_satellites[satellite].get('origin_customers', [])]) - As

                while extra > 0:
                    # relocate the farthest customer to the other satellite
                    customer = list(customers_sat.keys())[0]
                    lsp_satellites[other_satellite]['origin_customers'] = lsp_satellites[other_satellite].get('origin_customers', []) + [customer]
                    lsp_satellites[satellite]['origin_customers'].remove(customer)
                    lsp_customers[customer]['origin_satellite'] = other_satellite
                    # update capacity
                    extra -= lsp_customers[customer]['demand']
                    customers_sat.pop(customer, None)

        customers.update(lsp_customers)
        satellites.update(lsp_satellites)

    return customers, satellites

# Algorthm 2 - Phase B