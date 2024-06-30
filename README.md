# Fleet Decarbonization Optimization

## Overview

This project is part of the Shell.ai Hackathon 2024. My goal is to create an optimization model for transitioning vehicle fleets to net-zero emissions while minimizing costs. The model integrates various cost factors, vehicle specifications, fuel consumption metrics, and carbon emission constraints to achieve a balanced, data-driven solution.

## Repository Structure

```
.
├── code/
│   ├── main.py
│   ├── Demand.csv
│   ├── Vehicles.csv
│   ├── Vehicles_fuels.csv
│   ├── Fuels.csv
│   ├── Carbon_emissions.csv
│   └── solution.csv
├── documentation/
│   ├── SHELL ENERGY.pptx
└── README.md
```

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - pandas
  - pulp

You can install the required packages using pip:
```sh
pip install pandas pulp
```

## How to Run the Code

1. **Clone the repository:**
   ```sh
   git clone https://github.com/PASS-CLOUD/ENERGY
   cd ENERGY/code
   ```

2. **Ensure all required CSV files are in the `code/` directory:**
   - `Demand.csv`
   - `Vehicles.csv`
   - `Vehicles_fuels.csv`
   - `Fuels.csv`
   - `Carbon_emissions.csv`

3. **Run the optimization script:**
   ```sh
   python main.py
   ```

4. **Output:**
   - The results will be saved in `solution.csv` in the same directory.

## Files

- **`main.py`**: The main script containing the optimization model and logic.
- **`Demand.csv`**: Data file containing the yearly demand for different vehicle sizes and distances.
- **`Vehicles.csv`**: Data file containing information about the vehicles, including purchase year, cost, size, and yearly range.
- **`Vehicles_fuels.csv`**: Data file detailing the fuel types compatible with each vehicle and their fuel consumption rates.
- **`Fuels.csv`**: Data file containing fuel cost and emission data for different years.
- **`Carbon_emissions.csv`**: Data file with yearly carbon emission limits.
- **`solution.csv`**: Output file generated by the script, detailing the optimized fleet operations.

## Concept Description

### Summary

my concept leverages advanced mathematical optimization models to transition fleets to net-zero emissions cost-effectively. By integrating vehicle data, fuel consumption metrics, and carbon emission constraints, my solution minimizes the total cost of ownership while meeting stringent emission targets. This approach surpasses existing technologies by offering a holistic, data-driven method for fleet management, ensuring both sustainability and operational efficiency.

### Development Process

The concept was developed by analyzing the provided data on vehicle specifications, fuel consumption, and carbon emissions. I identified the need for a balanced approach that minimizes costs and adheres to emission limits. The core idea was to create a model that optimizes vehicle purchases, usage, and resale over time, ensuring demand satisfaction and compliance with carbon emission targets.

### Algorithms Used

The primary algorithm used is linear programming, implemented using the PuLP library. Linear programming is ideal for this problem as it efficiently handles constraints and optimizes a linear objective function. my model considers multiple decision variables and constraints to achieve the optimal fleet composition and usage strategy.

### Algorithm Selection

I chose linear programming due to its robustness in handling complex optimization problems with multiple constraints. Compared to other algorithms like genetic algorithms or simulated annealing, linear programming provides exact solutions and ensures all constraints are strictly adhered to, making it the best fit for my problem.

### Impact of Chosen Algorithm

The use of linear programming significantly impacted my concept by enabling precise optimization of fleet operations. It allowed us to systematically incorporate various cost factors and constraints, leading to a well-balanced solution that minimizes total costs while meeting all demands and emission targets.

### Proposed Architecture

my architecture consists of data preprocessing, optimization model formulation, and result extraction. Data preprocessing involves loading and cleaning the datasets. The optimization model is formulated using PuLP, defining decision variables, objective function, and constraints. Finally, results are extracted and saved to CSV for further analysis and reporting.

### Factors Considered

- **Cost Minimization:** Purchase, insurance, maintenance, and fuel costs.
- **Emission Compliance:** Adhering to yearly carbon emission limits.
- **Demand Satisfaction:** Ensuring fleet meets yearly distance demands.
- **Vehicle Lifespan:** Managing vehicle purchases and resale over time.
- **Fuel Compatibility:** Matching vehicles with compatible fuel types.

### Competitive Landscape

Using a matrix, I compared my concept with existing fleet management solutions. my approach stands out due to its comprehensive optimization model, integrating cost and emission factors, which is not typically addressed in conventional fleet management systems.

### Comparison with Alternatives

my concept offers a unique combination of cost optimization and emission compliance. Unlike traditional fleet management solutions that focus primarily on operational efficiency, my model integrates environmental sustainability as a core component, providing a dual benefit of cost savings and reduced carbon footprint.

### Intellectual Property Landscape

Currently, there are patents on specific algorithms and systems for fleet management and optimization. However, my unique integration of linear programming for simultaneous cost and emission optimization, tailored to fleet decarbonization, presents a novel approach that could be considered for patent protection.

### Competitors

- **Current Competitors:**
  - Conventional Fleet Management Systems (e.g., Fleet Complete, Samsara)
  - Environmental Optimization Software (e.g., Optoro, Locus)
- **Future Competitors:**
  - Emerging AI-driven Fleet Management Solutions
  - Sustainability-focused Tech Startups

### Market Size

- **Total Addressable Market (TAM):** The global fleet management market, valued at approximately $19 billion.
- **Serviceable Addressable Market (SAM):** The market segment focusing on fleet decarbonization, estimated at $5 billion.
- **Serviceable Obtainable Market (SOM):** The realistic portion of the market I can capture, projected at $1 billion.

### Market Size Calculation Method

I calculated the market size using industry reports and market research data. The TAM was derived from the total global fleet management market value. The SAM was estimated based on the proportion of the market interested in sustainability and decarbonization. The SOM was calculated considering our competitive positioning and potential market share.

---
Here are some potential limitations of the code provided:

1. **Data Dependence**:
   - The effectiveness of the model relies heavily on the quality and accuracy of the input data (CSV files). Any inaccuracies or missing data can significantly impact the results.

2. **Static Assumptions**:
   - The model uses static percentages for resale value, insurance costs, and maintenance costs. These values may not reflect real-world variations and changes over time.

3. **Simplified Resale Value, Insurance, and Maintenance Calculations**:
   - The calculations for resale value, insurance, and maintenance are simplified and may not capture all the factors influencing these costs in a real-world scenario.

4. **Fixed Carbon Emission Limits**:
   - The model uses fixed carbon emission limits from the carbon_emissions.csv file. These limits may change over time due to regulatory changes or new environmental policies.

5. **Vehicle-Fuel Compatibility**:
   - The model assumes that vehicle-fuel compatibility is binary (either a vehicle is compatible with a fuel type, or it is not). It does not account for partial compatibility or adaptation costs.

6. **Single Objective Function**:
   - The model optimizes a single objective function, which is the minimization of costs. It does not consider other important factors such as operational efficiency, customer satisfaction, or environmental impact beyond carbon emissions.

7. **Simplified Demand Satisfaction**:
   - The model assumes that the demand for different vehicle sizes and distance buckets can be met exactly as specified in the demand.csv file. It does not account for potential variations or uncertainties in demand.

8. **Limited Time Horizon**:
   - The model is designed to optimize decisions over a fixed time horizon (2023-2038). It does not account for long-term strategic planning beyond this period.

9. **Fleet Sell Limit Constraint**:
   - The constraint on the fleet sell limit may not reflect real-world operational flexibility or constraints. It assumes a maximum of 20% of the fleet can be sold in any given year.

10. **Computational Complexity**:
    - The linear programming model can become computationally intensive as the number of vehicles, fuel types, and years increases, potentially leading to longer solve times or even infeasibility for large-scale problems.

11. **No Consideration for Technological Advancements**:
    - The model does not account for potential technological advancements in vehicle efficiency, fuel types, or emissions reduction technologies that could emerge during the planning period.

12. **Assumption of Constant Costs**:
    - The costs of fuels, maintenance, and insurance are assumed to be constant or follow predefined percentages. These costs can fluctuate due to market dynamics, economic conditions, or changes in policy.
