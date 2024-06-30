import pandas as pd
from pulp import *
import csv

# Load data
demand = pd.read_csv('Demand.csv')
vehicles = pd.read_csv('Vehicles.csv')
vehicles_fuels = pd.read_csv('Vehicles_fuels.csv')
fuels = pd.read_csv('Fuels.csv')
carbon_emissions = pd.read_csv('Carbon_emissions.csv')

# Define helper functions
def get_resale_value(purchase_year, current_year, purchase_cost):
    age = max(0, current_year - purchase_year)
    if age >= 10:
        return purchase_cost * 0.3  # 30% of purchase cost after 10 years
    resale_percentages = [90, 80, 70, 60, 50, 40, 30, 30, 30, 30]
    return purchase_cost * resale_percentages[age] / 100

def get_insurance_cost(purchase_year, current_year, purchase_cost):
    age = max(0, current_year - purchase_year)
    if age >= 10:
        return purchase_cost * 0.14  # 14% of purchase cost after 10 years
    insurance_percentages = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    return purchase_cost * insurance_percentages[age] / 100

def get_maintenance_cost(purchase_year, current_year, purchase_cost):
    age = max(0, current_year - purchase_year)
    if age >= 10:
        return purchase_cost * 0.19  # 19% of purchase cost after 10 years
    maintenance_percentages = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    return purchase_cost * maintenance_percentages[age] / 100

# Define the problem
prob = LpProblem("Fleet_Transition", LpMinimize)

# Parameters
years = range(2023, 2039)
distance_buckets = ['D1', 'D2', 'D3', 'D4']

# Decision variables
buy = LpVariable.dicts("buy", ((v, y) for v in vehicles['ID'] for y in years), lowBound=0, cat='Integer')
use = LpVariable.dicts("use", ((v, f, d, y) 
                               for v in vehicles['ID'] 
                               for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique() 
                               for d in distance_buckets 
                               for y in years), lowBound=0)
sell = LpVariable.dicts("sell", ((v, y) for v in vehicles['ID'] for y in years), lowBound=0, cat='Integer')
distance = LpVariable.dicts("distance", ((v, f, d, y) 
                                         for v in vehicles['ID'] 
                                         for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique() 
                                         for d in distance_buckets 
                                         for y in years), lowBound=0)

# Objective function components
C_buy = lpSum(buy[v, y] * vehicles.loc[vehicles['ID'] == v, 'Cost ($)'].values[0] for v in vehicles['ID'] for y in years)

C_ins = lpSum(get_insurance_cost(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y, vehicles.loc[vehicles['ID'] == v, 'Cost ($)'].values[0]) * 
               lpSum(buy[v, py] for py in range(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y+1)) 
               for v in vehicles['ID'] for y in years if y >= vehicles.loc[vehicles['ID'] == v, 'Year'].values[0])

C_mnt = lpSum(get_maintenance_cost(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y, vehicles.loc[vehicles['ID'] == v, 'Cost ($)'].values[0]) * 
               lpSum(buy[v, py] for py in range(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y+1)) 
               for v in vehicles['ID'] for y in years if y >= vehicles.loc[vehicles['ID'] == v, 'Year'].values[0])

C_fuel = lpSum(distance[v, f, d, y] * vehicles_fuels.loc[(vehicles_fuels['ID'] == v) & (vehicles_fuels['Fuel'] == f), 'Fuel Consumption (unit_fuel/km)'].values[0] * 
                fuels.loc[(fuels['Fuel'] == f) & (fuels['Year'] == y), 'Cost ($/unit_fuel)'].values[0] 
                for v in vehicles['ID'] for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique() for d in distance_buckets for y in years)

C_sell = lpSum(get_resale_value(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y, vehicles.loc[vehicles['ID'] == v, 'Cost ($)'].values[0]) * sell[v, y] 
                for v in vehicles['ID'] for y in years)

# Objective function
prob += C_buy + C_ins + C_mnt + C_fuel - C_sell

# Constraints
# 1. Demand satisfaction
for y in years:
    for size in demand['Size'].unique():
        for d in distance_buckets:
            demand_value = demand.loc[(demand['Year'] == y) & (demand['Size'] == size) & (demand['Distance'] == d), 'Demand'].sum()
            prob += lpSum(distance[v, f, d, y] for v in vehicles.loc[vehicles['Size'] == size, 'ID'] 
                          for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique()
                          if vehicles.loc[vehicles['ID'] == v, 'Distance'].values[0] >= int(d[1])) >= demand_value, f"Demand_{y}_{size}_{d}"

# 2. Carbon emission limits
for y in years:
    carbon_limit = carbon_emissions.loc[carbon_emissions['Year'] == y, 'Total Carbon emission limit'].values[0]
    prob += lpSum(distance[v, f, d, y] * vehicles_fuels.loc[(vehicles_fuels['ID'] == v) & (vehicles_fuels['Fuel'] == f), 'Fuel Consumption (unit_fuel/km)'].values[0] * 
                  fuels.loc[(fuels['Fuel'] == f) & (fuels['Year'] == y), 'Emissions (CO2/unit_fuel)'].values[0]
                  for v in vehicles['ID'] for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique() for d in distance_buckets) <= carbon_limit, f"Carbon_Limit_{y}"

# 3. Vehicle purchase year constraint
for v in vehicles['ID']:
    for y in years:
        if y != vehicles.loc[vehicles['ID'] == v, 'Year'].values[0]:
            prob += buy[v, y] == 0, f"Buy_Year_{v}_{y}"

# 4. Vehicle lifespan constraint
for v in vehicles['ID']:
    purchase_year = vehicles.loc[vehicles['ID'] == v, 'Year'].values[0]
    for y in range(purchase_year, 2039):
        if y + 10 <= 2038:
            prob += lpSum(sell[v, sy] for sy in range(y, min(y+11, 2039))) == lpSum(buy[v, by] for by in range(purchase_year, y+1)), f"Vehicle_Life_{v}_{y}"

# 5. Fleet sell limit constraint
for y in years:
    prob += lpSum(sell[v, y] for v in vehicles['ID']) <= 0.2 * lpSum(buy[v, py] for v in vehicles['ID'] for py in range(vehicles.loc[vehicles['ID'] == v, 'Year'].values[0], y+1)), f"Fleet_Sell_Limit_{y}"

# 6. Vehicle-fuel compatibility constraint
for v in vehicles['ID']:
    compatible_fuels = set(vehicles_fuels.loc[vehicles_fuels['ID'] == v, 'Fuel'].unique())
    for f in fuels['Fuel'].unique():
        if f not in compatible_fuels:
            for d in distance_buckets:
                for y in years:
                    if (v, f, d, y) in use:
                        prob += use[v, f, d, y] == 0, f"Fuel_Compatibility_{v}_{f}_{d}_{y}"

# 7. Use-distance relationship
for v in vehicles['ID']:
    for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique():
        for d in distance_buckets:
            for y in years:
                prob += distance[v, f, d, y] <= vehicles.loc[vehicles['ID'] == v, 'Yearly range (km)'].values[0] * use[v, f, d, y], f"Use_Distance_{v}_{f}_{d}_{y}"

# 8. Fleet balance constraint
for v in vehicles['ID']:
    purchase_year = vehicles.loc[vehicles['ID'] == v, 'Year'].values[0]
    for y in years:
        if y >= purchase_year:
            prob += lpSum(use[v, f, d, y] for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique() for d in distance_buckets) <= lpSum(buy[v, py] for py in range(purchase_year, y+1)) - lpSum(sell[v, sy] for sy in range(purchase_year, y)), f"Fleet_Balance_{v}_{y}"

# Solve the problem
prob.solve()

# Extract and save results
results = []
for y in years:
    for v in vehicles['ID']:
        # Buy operations
        if value(buy[v, y]) > 0:
            results.append([y, v, value(buy[v, y]), 'Buy', '', '', ''])
        
        # Use operations
        for f in vehicles_fuels[vehicles_fuels['ID'] == v]['Fuel'].unique():
            for d in distance_buckets:
                if value(use[v, f, d, y]) > 0:
                    results.append([y, v, value(use[v, f, d, y]), 'Use', f, d, value(distance[v, f, d, y]) / value(use[v, f, d, y])])
        
        # Sell operations
        if value(sell[v, y]) > 0:
            results.append([y, v, value(sell[v, y]), 'Sell', '', '', ''])

# Save results to CSV
with open('solution.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'])
    writer.writerows(results)

print("Results saved to 'solution.csv'")
