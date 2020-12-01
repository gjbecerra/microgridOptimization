# Helper functions for solving the energy problem using Gurobi

from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# Function for reading data from simple example data
def readExampleData(label):
    data = pd.read_csv("DemandayPV24DRGEO.csv") 
    col = list(data.get(label))
    return col

# Function for reading load data from loadMeasures dataset
# Data available from 03/01/2017 to 21/12/2017
# date: selects the day for reading the data
# id: selects the location (house number) for the load measurements
def readLoadPVData(date,id):
    # Reads complete load dataset
    loadData = pd.read_csv("loadMeasures.csv", sep=',', index_col="datetime", parse_dates=True)
    # Gets load data for a particular date and id
    loadPower = list(loadData.loc[date.strftime("%Y-%m-%d"),id])
    # Gets load data for all the year and a particular id
    yearLoadData = loadData.loc[date.strftime("%Y")+'-01-01':date.strftime("%Y")+'-12-31',id]

    # Reads complete PV dataset
    pvData = pd.read_csv("PVforecast-measure.csv", sep=',', index_col="datetime", parse_dates=True)
    # Gets PV data for a particular date
    pvPower = list(pvData.loc[date.strftime("%Y-%m-%d"),"PV forecast [kW]"])
    # Gets PV data for all the year
    yearPvData = pvData.loc[date.strftime("%Y")+'-01-01':date.strftime("%Y")+'-12-31','PV forecast [kW]']

    # Joins the complete load and PV yearly data as a single dataframe
    yearLoadPvData = yearPvData.to_frame().join(yearLoadData.to_frame())
    # Computes the average net power flow
    yearPn = abs(yearLoadPvData["PV forecast [kW]"] - yearLoadPvData[id])
    averageYearLoadPower = yearPn.mean()
    averageNetLoadPower = list(yearPn.loc[date.strftime("%Y-%m-%d")])

    return loadPower, pvPower, averageNetLoadPower, averageYearLoadPower

# Function for computing the energy price according to Creg 15-2018
def computeCregPrice(typicalLoadCurve, unitCost):
    Pi = 0.01*np.array(typicalLoadCurve)
    Dt = unitCost
    fch=2

    # Finds the intervals for maximum (x), medium (z) and minimum (y) load
    indx = np.where((Pi>=0.95))[0]
    indz = np.where((Pi<0.95) & (Pi>=0.75))[0]
    indy = np.where((Pi<0.75))[0]
    Hx = len(indx)
    Hz = len(indz)
    Hy = len(indy)
    Px = np.mean(Pi[indx])
    Pz = np.mean(Pi[indz])
    Py = np.mean(Pi[indy])
    sumPi =np.sum(Pi)

    # Builds the matrices
    A = np.array([[Hx*Px/fch,Hz*Pz,fch*Hy*Py],[1/fch,-Px/Pz,0],[1/(fch*fch),0,-Px/Py]])
    b = np.array([[Dt*sumPi],[0],[0]])

    # Solves the linear system 
    x = np.linalg.solve(A, b)

    # CostoVariable=np.sum(x[0]*Curvapu[indx])+np.sum(x[1]*Curvapu[indz])+np.sum(x[2]*Curvapu[indy])
    # CostoFija=np.sum(Dt*Curvapu)

    # Returns the computed energy prices for each interval
    energyCost = np.zeros(24)
    energyCost[indx] = x[0]
    energyCost[indz] = x[1]
    energyCost[indy] = x[2]
    return list(energyCost)

def readMarketPrice(priceDate):
    # energyCost: energy cost
    data = pd.read_csv("precioBolsa.csv")
    data.set_index("Fecha", inplace=True)
    # Gets the hourly price for a given date
    energyCost = list(data.loc[priceDate.strftime("%Y-%m-%d"), :])
    return energyCost

def computeDynamicPrice(priceDate, averageNetLoadPower, averageYearLoadPower, unitCost):
    # Loads market price data from  file:
    # energyCost: energy cost
    data = pd.read_csv("precioBolsa.csv")
    data.set_index("Fecha", inplace=True)
    # Gets the hourly market price for a given date
    marketCost = list(data.loc[priceDate.strftime("%Y-%m-%d"), :])
    # Computes the daily mean for the complete dataset and stores it
    # as the last column of the dataframe
    data['dailymean'] = data.mean(axis=1)
    # Computes the mean market price for all the year
    averageYearCost = data.loc[priceDate.strftime("%Y")+'-01-01':priceDate.strftime("%Y")+'-12-31','dailymean'].mean()
    energyCost = list((unitCost/2)*(marketCost/averageYearCost + averageNetLoadPower/averageYearLoadPower))
    return energyCost

# Builds and solves the optimal problem for the scenario 2:
# System with battery storage, photovoltaic generation, but no Demand Response
def optimalSolutionScenario1(loadPower, pvPower, energyCost):
    # Create a new moded
    m = Model('microgrid')

    # Create paremeters
    batteryNominalPower = 20        # Nominal battery power
    eta_charge = 0.95               # Charge efficiency
    eta_discharge = 0.95            # Discharge efficiency
    gridPower_min = 0               # Minimum power supplied by the grid
    gridPower_max = 50              # Maximum power supplied by the grid
    batteryChargePower_min = 0      # Minimum power charged to the batteries
    batteryChargePower_max = 20     # Maximum power charged to the batteries
    batteryDischargePower_min = 0   # Minimum power discharged from the batteries
    batteryDischargePower_max = 20  # Maximum power discharged from the batteries
    batteryStoredEnergy_min = 0.2*batteryNominalPower*5     # Minimum energy stored in the batteries
    batteryStoredEnergy_max = 0.8*batteryNominalPower*5     # Maximum energy stored in the batteries

    # Create variables
    gridPower             = m.addVars(range(24), lb=gridPower_min,             ub=gridPower_max,             name='Grid_Supplied_Power')      # Power supplied by the grid at time t
    batteryChargePower    = m.addVars(range(24), lb=batteryChargePower_min,    ub=batteryChargePower_max,    name='Battery_Charge_Power')     # Power charged to the batteries at time t
    batteryDischargePower = m.addVars(range(24), lb=batteryDischargePower_min, ub=batteryDischargePower_max, name='Battery_Discharge_Power')  # Power discharged from the batteries at time t
    batteryStoredEnergy   = m.addVars(range(25), lb=batteryStoredEnergy_min,   ub=batteryStoredEnergy_max,   name='Battery_Stored_Energy')    # Energy stored in the batteries at time t
    q                     = m.addVars(range(24), vtype=GRB.BINARY, name="xbat")     # Binary variable that defines the state of the inverter (xbat=1: charging, xbat=0: discharging)

    # Create constraints
    m.addConstrs((batteryStoredEnergy[t] == batteryStoredEnergy[t-1] + eta_charge*batteryChargePower[t-1] - eta_discharge*batteryDischargePower[t-1] for t in range(1,25)), name="Constr2")
    m.addConstrs((gridPower[t] == loadPower[t] + eta_charge*batteryChargePower[t] - eta_discharge*batteryDischargePower[t] - pvPower[t] for t in range(24)), name="Constr3")
    m.addConstr(batteryStoredEnergy[0] == batteryStoredEnergy[24], name="Constr5")
    m.addConstrs((gridPower[t] <= loadPower[t] for t in range(24)), name="Constr9")
    m.addConstrs((batteryChargePower[t] <= q[t]*batteryChargePower_max for t in range(24)))
    m.addConstrs((batteryDischargePower[t] <= (1-q[t])*batteryDischargePower_max for t in range(24)))

    # Create objective function
    m.setObjective(quicksum(energyCost[t]*gridPower[t] for t in range(24)), GRB.MINIMIZE)

    # Save model
    m.update()
    # m.write('microgrid.lp')

    # Run the optimizer
    m.optimize()

    # Get solution
    if m.status == GRB.Status.OPTIMAL:
        gridPower_sol = m.getAttr('x', gridPower)
        batteryChargePower_sol = m.getAttr('x', batteryChargePower)
        batteryDischargePower_sol = m.getAttr('x', batteryDischargePower)
        batteryStoredEnergy_sol = m.getAttr('x', batteryStoredEnergy)

        # Appends the last value to show the last hour in the step plot
        gridPower_res   =  gridPower_sol.values()   + [gridPower_sol.values()[-1]]
        batteryChargePower_res  =  batteryChargePower_sol.values()  + [batteryChargePower_sol.values()[-1]]
        batteryDischargePower_res =  batteryDischargePower_sol.values() + [batteryDischargePower_sol.values()[-1]]
        batteryStoredEnergy_res   =  batteryStoredEnergy_sol.values()

        # Gets the total cost as the optimal value of the objective function
        totalCost = m.objVal
        
        optimSolFound = True
    else:
        m.computeIIS()
        if m.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('The following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        gridPower_res   =  0
        batteryChargePower_res  =  0
        batteryDischargePower_res =  0
        batteryStoredEnergy_res   =  0

        totalCost = 0 
        optimSolFound = False
    
    return optimSolFound, gridPower_res, batteryChargePower_res, batteryDischargePower_res, batteryStoredEnergy_res, totalCost


# Builds and solves the optimal problem for the scenario 3:
# System with battery storage, photovoltaic generation and Demand Response
def optimalSolutionScenario2(loadPower, pvPower, energyCost):
    # Create a new moded
    m = Model('microgrid')

    # Create paremeters
    batteryNominalPower = 20        # Nominal battery power
    eta_charge = 0.95               # Charge efficiency
    eta_discharge = 0.95            # Discharge efficiency
    gridPower_min = 0               # Minimum power supplied by the grid
    gridPower_max = 50              # Maximum power supplied by the grid
    batteryChargePower_min = 0      # Minimum power charged to the batteries
    batteryChargePower_max = 20     # Maximum power charged to the batteries
    batteryDischargePower_min = 0   # Minimum power discharged from the batteries
    batteryDischargePower_max = 20  # Maximum power discharged from the batteries
    batteryStoredEnergy_min = 0.2*batteryNominalPower*5     # Minimum energy stored in the batteries
    batteryStoredEnergy_max = 0.8*batteryNominalPower*5     # Maximum energy stored in the batteries
    Psh_min = 0                 # Minimum power shifted by demand response
    Psh_max = 100               # Maximum power shifted by demand response
    Pcut1_min = 0                 # Minimum demand response power cut 
    Pcut1_max = 10                # Maximum demand response power cut
    Pcut2_min = 0                 # Minimum demand response power cut 
    Pcut2_max = 20                # Maximum demand response power cut
    Pcut3_min = 0                 # Minimum demand response power cut 
    Pcut3_max = 30                # Maximum demand response power cut
    Cdr = [38, 56, 114]         # Price of demand response program for each cut

    # Create variables
    gridPower             = m.addVars(range(24), lb=gridPower_min,             ub=gridPower_max,             name='Grid_Supplied_Power')      # Power supplied by the grid at time t
    batteryChargePower    = m.addVars(range(24), lb=batteryChargePower_min,    ub=batteryChargePower_max,    name='Battery_Charge_Power')     # Power charged to the batteries at time t
    batteryDischargePower = m.addVars(range(24), lb=batteryDischargePower_min, ub=batteryDischargePower_max, name='Battery_Discharge_Power')  # Power discharged from the batteries at time t
    batteryStoredEnergy   = m.addVars(range(25), lb=batteryStoredEnergy_min,   ub=batteryStoredEnergy_max,   name='Battery_Stored_Energy')    # Energy stored in the batteries at time t
    Psh  = m.addVars(range(24),     lb=Psh_min,    ub=Psh_max,    name='PowerShifted')              # Power shifted by demand response program at time t
    Pcut1  = m.addVars(range(24),   lb=Pcut1_min,  ub=Pcut1_max,  name='PowerDemandResponseCut')    # Power cut by demand response program at time t
    Pcut2  = m.addVars(range(24),   lb=Pcut2_min,  ub=Pcut2_max,  name='PowerDemandResponseCut')    # Power cut by demand response program at time t
    Pcut3  = m.addVars(range(24),   lb=Pcut3_min,  ub=Pcut3_max,  name='PowerDemandResponseCut')    # Power cut by demand response program at time t
    xbat                  = m.addVars(range(24), vtype=GRB.BINARY, name="xbat")     # Binary variable that defines the state of the inverter (xbat=1: charging, xbat=0: discharging)

    # Create constraints
    m.addConstrs((batteryStoredEnergy[t] == batteryStoredEnergy[t-1] + eta_charge*batteryChargePower[t-1] - eta_discharge*batteryDischargePower[t-1] for t in range(1,25)), name="Constr2")
    m.addConstrs((gridPower[t] == loadPower[t] + batteryChargePower[t] - batteryDischargePower[t] - pvPower[t] + Psh[t] - Pcut1[t] - Pcut2[t] - Pcut3[t] for t in range(24)), name="Constr3")
    m.addConstr(quicksum(Pcut1[t] + Pcut2[t] + Pcut3[t] for t in range(24)) == quicksum(Psh[t] for t in range(24)), name="Constr4")
    m.addConstr(batteryStoredEnergy[0] == batteryStoredEnergy[24], name="Constr5")
    # m.addConstrs((gridPower[t] <= loadPower[t] for t in range(24)), name="Constr9")
    m.addConstrs((gridPower[t] <= loadPower[t] + Psh[t] - Pcut1[t] - Pcut2[t] - Pcut3[t] for t in range(24)), name="Constr9")
    m.addConstrs((batteryChargePower[t] <= xbat[t]*batteryChargePower_max for t in range(24)))
    m.addConstrs((batteryDischargePower[t] <= (1-xbat[t])*batteryDischargePower_max for t in range(24)))

    # Create objective function
    m.setObjective(quicksum(energyCost[t]*gridPower[t] + Cdr[0]*Pcut1[t] + Cdr[1]*Pcut2[t] + Cdr[2]*Pcut3[t] for t in range(24)), GRB.MINIMIZE)

    # Save model
    m.update()
    # m.write('microgrid.lp')

    # Run the optimizer
    m.optimize()

    # Get solution
    if m.status == GRB.Status.OPTIMAL:
        gridPower_sol = m.getAttr('x', gridPower)
        batteryChargePower_sol = m.getAttr('x', batteryChargePower)
        batteryDischargePower_sol = m.getAttr('x', batteryDischargePower)
        batteryStoredEnergy_sol = m.getAttr('x', batteryStoredEnergy)
        Psh_sol = m.getAttr('x', Psh)
        Pcut1_sol = m.getAttr('x', Pcut1)
        Pcut2_sol = m.getAttr('x', Pcut2)
        Pcut3_sol = m.getAttr('x', Pcut3)

        # Appends the last value to show the last hour in the step plot
        gridPower_res   =  gridPower_sol.values()   + [gridPower_sol.values()[-1]]
        batteryChargePower_res  =  batteryChargePower_sol.values()  + [batteryChargePower_sol.values()[-1]]
        batteryDischargePower_res =  batteryDischargePower_sol.values() + [batteryDischargePower_sol.values()[-1]]
        batteryStoredEnergy_res   =  batteryStoredEnergy_sol.values()
        Psh_res  =  Psh_sol.values()  + [Psh_sol.values()[-1]]
        Pcut1_res  =  Pcut1_sol.values()  + [Pcut1_sol.values()[-1]]
        Pcut2_res  =  Pcut2_sol.values()  + [Pcut2_sol.values()[-1]]
        Pcut3_res  =  Pcut3_sol.values()  + [Pcut3_sol.values()[-1]]

        totalCost = m.objVal
        
        optimSolFound = True
    else:
        m.computeIIS()
        if m.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('The following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        gridPower_res   =  0
        batteryChargePower_res  =  0
        batteryDischargePower_res =  0
        batteryStoredEnergy_res   =  0

        totalCost = 0 
        optimSolFound = False
    
    return optimSolFound, gridPower_res, batteryChargePower_res, batteryDischargePower_res, batteryStoredEnergy_res, Psh_res, Pcut1_res, Pcut2_res, Pcut3_res, totalCost

def prepareFigureScenario1(loadDate, totalCost, energyCost, pvPower, loadPower, gridPower, batteryChargePower, batteryDischargePower, batteryStoredEnergy, typicalLoadCurve):
    # Prepares figure with results
    print(f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}")
   
    fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(10,13))
    figtitle = f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}"

    fig.suptitle(figtitle)
    axs[0].step(range(25),energyCost + [energyCost[-1]], where='post')
    axs[0].legend(["Spot price of Grid Power"])
    axs[0].minorticks_on()
    axs[0].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[0].grid(b=True, which='minor', color='lightgray', linestyle='--')
    if typicalLoadCurve:
        loadCurve = typicalLoadCurve + [typicalLoadCurve[-1]]
        axs2 = axs[0].twinx()
        axs2.step(range(25), loadCurve, where='post', linestyle='--', color='tab:red')
        axs2.tick_params(axis='y', labelcolor='tab:red')
        axs2.legend(["Typical Load Curve"], loc='upper right')
    axs[1].step(range(25),gridPower, where='post', linestyle='-')
    axs[1].step(range(25),pvPower + [pvPower[-1]], where='post', linestyle='-')
    axs[1].step(range(25),loadPower + [loadPower[-1]], where='post', linestyle='--')
    axs[1].legend(["Power supplied by the grid", "Power supplied by the PV system", "Power consumed by the loads"])
    axs[1].minorticks_on()
    axs[1].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[1].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[2].step(range(25),batteryChargePower, where='post', linestyle='-')
    axs[2].step(range(25),batteryDischargePower, where='post', linestyle='--')
    axs[2].legend(["Charge Power in battery", "Discharge Power in battery"])
    axs[2].minorticks_on()
    axs[2].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[2].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[3].plot(range(25),batteryStoredEnergy, linestyle='-')
    axs[3].legend(["Energy in battery"])
    axs[3].minorticks_on()
    axs[3].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[3].grid(b=True, which='minor', color='lightgray', linestyle='--')
    
    # filename = "OptimalSolution_" + loadDate.strftime("%Y-%m-%d")
    filename = "OptimalSolution_"
    filename = filename + datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
    plt.savefig("results/" + filename)
    plt.show()


def prepareFigureScenario2(loadDate, totalCost, energyCost, pvPower, loadPower, gridPower, batteryChargePower, batteryDischargePower, batteryStoredEnergy, Psh, Pcut1, Pcut2, Pcut3, typicalLoadCurve):
    # Prepares figure with results
    print(f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}")
   
    fig, axs = plt.subplots(5, 1, constrained_layout=True, figsize=(10,13))
    figtitle = f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}"

    newLoadPower = [loadPower[t] + Psh[t] - Pcut1[t] - Pcut2[t] - Pcut3[t] for t in range(24)]

    fig.suptitle(figtitle)
    axs[0].step(range(25),energyCost + [energyCost[-1]], where='post')
    axs[0].legend(["Spot price of Grid Power"])
    axs[0].minorticks_on()
    axs[0].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[0].grid(b=True, which='minor', color='lightgray', linestyle='--')
    if typicalLoadCurve:
        loadCurve = typicalLoadCurve + [typicalLoadCurve[-1]]
        axs2 = axs[0].twinx()
        axs2.step(range(25), loadCurve, where='post', linestyle='--', color='tab:red')
        axs2.tick_params(axis='y', labelcolor='tab:red')
        axs2.legend(["Typical Load Curve"], loc='upper right')
    axs[1].step(range(25),gridPower, where='post', linestyle='-')
    axs[1].step(range(25),pvPower + [pvPower[-1]], where='post', linestyle='-')
    axs[1].step(range(25),newLoadPower + [newLoadPower[-1]], where='post', linestyle='--')
    axs[1].legend(["Power supplied by the grid", "Power supplied by the PV system", "Power consumed by the loads"])
    axs[1].minorticks_on()
    axs[1].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[1].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[2].step(range(25),batteryChargePower, where='post', linestyle='-')
    axs[2].step(range(25),batteryDischargePower, where='post', linestyle='--')
    axs[2].legend(["Charge Power in battery", "Discharge Power in battery"])
    axs[2].minorticks_on()
    axs[2].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[2].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[3].plot(range(25),batteryStoredEnergy, linestyle='-')
    axs[3].legend(["Energy in battery"])
    axs[3].minorticks_on()
    axs[3].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[3].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[4].step(range(25),Pcut1, where='post', linestyle='-')
    axs[4].step(range(25),Pcut2, where='post', linestyle='--')
    axs[4].step(range(25),Pcut3, where='post', linestyle='-.')
    axs[4].step(range(25),Psh, where='post', linestyle=':')
    axs[4].legend(["Power cut 1 (Pdr1)", "Power cut 2 (Pdr2)", "Power cut 3 (Pdr3)", "Shifted power (Psh)"])
    axs[4].minorticks_on()
    axs[4].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[4].grid(b=True, which='minor', color='lightgray', linestyle='--')
    
    # filename = "OptimalSolution_" + loadDate.strftime("%Y-%m-%d")
    filename = "OptimalSolution_"
    filename = filename + datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
    plt.savefig("results/" + filename)
   
    plt.show()

def exportResults(fileName, Pcharge, Pdischarge, Pgrid):
    data = {'H': [datetime.time(i,0).strftime("%H:%M") for i in range(24)], 'Pch': Pcharge[:-1], 'Pdch': Pdischarge[:-1], 'Pd': Pgrid[:-1]}
    df = pd.DataFrame(data)
    df.to_csv(fileName, sep=';', index=False, line_terminator=';\n', date_format="%H:%M")

