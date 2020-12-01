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

    # Builds the matrices
    A = np.array([[Hx*Px/fch,Hz*Pz,fch*Hy*Py],[1/fch,-Px/Pz,0],[1/(fch*fch),0,-Px/Py]])
    b = np.array([[Dt*np.sum(Pi)],[0],[0]])

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
def optimalSolutionScenario1(Pl, Ppv, Cd):
    # Create a new moded
    m = Model('microgrid')

    # Create paremeters
    Es_nom = 20        # Nominal battery power
    eta_ch = 0.95               # Charge efficiency
    eta_dch = 0.95            # Discharge efficiency
    Pd_min = 0               # Minimum power supplied by the grid
    Pd_max = 50              # Maximum power supplied by the grid
    Pch_min = 0      # Minimum power charged to the batteries
    Pch_max = 20     # Maximum power charged to the batteries
    Pdch_min = 0   # Minimum power discharged from the batteries
    Pdch_max = 20  # Maximum power discharged from the batteries
    Es_mini = 0.2*Es_nom*5     # Minimum energy stored in the batteries
    Es_max = 0.8*Es_nom*5     # Maximum energy stored in the batteries

    # Create variables
    Pd   = m.addVars(range(24), lb=Pd_min,   ub=Pd_max,   name='Grid_Supplied_Power')      # Power supplied by the grid at time t
    Pch  = m.addVars(range(24), lb=Pch_min,  ub=Pch_max,  name='Battery_Charge_Power')     # Power charged to the batteries at time t
    Pdch = m.addVars(range(24), lb=Pdch_min, ub=Pdch_max, name='Battery_Discharge_Power')  # Power discharged from the batteries at time t
    Es   = m.addVars(range(25), lb=Es_mini,  ub=Es_max,   name='Battery_Stored_Energy')    # Energy stored in the batteries at time t
    q    = m.addVars(range(24), vtype=GRB.BINARY, name="charger_on_off")     # Binary variable that defines the state of the inverter (xbat=1: charging, xbat=0: discharging)

    # Create constraints
    m.addConstrs((Es[t] == Es[t-1] + eta_ch*Pch[t-1] - eta_dch*Pdch[t-1] for t in range(1,25)), name="C(1b)")
    m.addConstrs((Pd[t] == Pl[t] + Pch[t] - Pdch[t] - Ppv[t] for t in range(24)), name="C(1c)")
    m.addConstr(Es[0] == Es[24], name="C(1d)")
    m.addConstrs((Pch[t] <= q[t]*Pch_max for t in range(24)), name="C(1f)")
    m.addConstrs((Pdch[t] <= (1-q[t])*Pdch_max for t in range(24)), name="C(1g)")
    m.addConstrs((Pd[t] <= Pl[t] for t in range(24)), name="C(1h)")

    # Create objective function
    m.setObjective(quicksum(Cd[t]*Pd[t] for t in range(24)), GRB.MINIMIZE)

    # Save model
    m.update()
    # m.write('microgrid.lp')

    # Run the optimizer
    m.optimize()

    # Get solution
    if m.status == GRB.Status.OPTIMAL:
        Pd_sol = m.getAttr('x', Pd)
        Pch_sol = m.getAttr('x', Pch)
        Pdch_sol = m.getAttr('x', Pdch)
        Es_sol = m.getAttr('x', Es)

        # Appends the last value to show the last hour in the step plot
        Pd_res   =  Pd_sol.values()   + [Pd_sol.values()[-1]]
        Pch_res  =  Pch_sol.values()  + [Pch_sol.values()[-1]]
        Pdch_res =  Pdch_sol.values() + [Pdch_sol.values()[-1]]
        Es_res   =  Es_sol.values()

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
        Pd_res   =  0
        Pch_res  =  0
        Pdch_res =  0
        Es_res   =  0

        totalCost = 0 
        optimSolFound = False
    
    return optimSolFound, Pd_res, Pch_res, Pdch_res, Es_res, totalCost


# Builds and solves the optimal problem for the scenario 3:
# System with battery storage, photovoltaic generation and Demand Response
def optimalSolutionScenario2(Pl, Ppv, Cd):
    # Create a new moded
    m = Model('microgrid')

    # Create paremeters
    Es_nom = 20        # Nominal battery power
    eta_ch = 0.95               # Charge efficiency
    eta_dch = 0.95            # Discharge efficiency
    Pd_min = 0               # Minimum power supplied by the grid
    Pd_max = 50              # Maximum power supplied by the grid
    Pch_min = 0      # Minimum power charged to the batteries
    Pch_max = 20     # Maximum power charged to the batteries
    Pdch_min = 0   # Minimum power discharged from the batteries
    Pdch_max = 20  # Maximum power discharged from the batteries
    Es_min = 0.2*Es_nom*5     # Minimum energy stored in the batteries
    Es_max = 0.8*Es_nom*5     # Maximum energy stored in the batteries
    Pcut_min = 0                 # Minimum demand response power cut 
    Pcut_max = 10                # Maximum demand response power cut
    Psh_min = 0                 # Minimum power shifted by demand response
    Psh_max = 100               # Maximum power shifted by demand response
    Ccut = [38, 56, 114]         # Price of demand response program for each cut

    # Create variables
    Pd   = m.addVars(range(24), lb=Pd_min,   ub=Pd_max,   name='Grid_Supplied_Power')      # Power supplied by the grid at time t
    Pch  = m.addVars(range(24), lb=Pch_min,  ub=Pch_max,  name='Battery_Charge_Power')     # Power charged to the batteries at time t
    Pdch = m.addVars(range(24), lb=Pdch_min, ub=Pdch_max, name='Battery_Discharge_Power')  # Power discharged from the batteries at time t
    Es   = m.addVars(range(25), lb=Es_min,   ub=Es_max,   name='Battery_Stored_Energy')    # Energy stored in the batteries at time t
    Pcut = m.addVars(len(Ccut),range(24), lb=Pcut_min, ub=Pcut_max, name='PowerDemandResponseCut')    # Power cut by demand response program at time t
    Psh  = m.addVars(range(24), lb=Psh_min,  ub=Psh_max,  name='PowerShifted')              # Power shifted by demand response program at time t
    q    = m.addVars(range(24), vtype=GRB.BINARY, name="xbat")     # Binary variable that defines the state of the inverter (xbat=1: charging, xbat=0: discharging)

    # Create constraints
    m.addConstrs((Es[t] == Es[t-1] + eta_ch*Pch[t-1] - eta_dch*Pdch[t-1] for t in range(1,25)), name="Constr2")
    m.addConstrs((Pd[t] == Pl[t] + Pch[t] - Pdch[t] - Ppv[t] + Psh[t] - quicksum(Pcut[i,t] for i in range(len(Ccut))) for t in range(24)), name="Constr3")
    m.addConstr(quicksum(quicksum(Pcut[i,t] for i in range(len(Ccut))) for t in range(24)) == quicksum(Psh[t] for t in range(24)), name="Constr4")
    m.addConstr(Es[0] == Es[24], name="Constr5")
    m.addConstrs((Pch[t] <= q[t]*Pch_max for t in range(24)))
    m.addConstrs((Pdch[t] <= (1-q[t])*Pdch_max for t in range(24)))
    m.addConstrs((Pd[t] <= Pl[t] + Psh[t] - quicksum(Pcut[i,t] for i in range(len(Ccut))) for t in range(24)), name="Constr9")

    # Create objective function
    m.setObjective(quicksum(Cd[t]*Pd[t] + quicksum(Ccut[i]*Pcut[i,t] for i in range(len(Ccut))) for t in range(24)), GRB.MINIMIZE)

    # Save model
    m.update()
    # m.write('microgrid.lp')

    # Run the optimizer
    m.optimize()

    # Get solution
    if m.status == GRB.Status.OPTIMAL:
        Pd_sol = m.getAttr('x',Pd)
        Pch_sol = m.getAttr('x',Pch)
        Pdch_sol = m.getAttr('x',Pdch)
        Es_sol = m.getAttr('x',Es)
        Psh_sol = m.getAttr('x',Psh)
        Pcut_sol = m.getAttr('x',Pcut)

        # Appends the last value to show the last hour in the step plot
        Pd_res   =  Pd_sol.values()   + [Pd_sol.values()[-1]]
        Pch_res  =  Pch_sol.values()  + [Pch_sol.values()[-1]]
        Pdch_res =  Pdch_sol.values() + [Pdch_sol.values()[-1]]
        Es_res   =  Es_sol.values()
        Psh_res  =  Psh_sol.values()  + [Psh_sol.values()[-1]]
        Pcut_res  =  Pcut_sol

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
        Pd_res   =  0
        Pch_res  =  0
        Pdch_res =  0
        Es_res   =  0

        totalCost = 0 
        optimSolFound = False
    
    return optimSolFound, Pd_res, Pch_res, Pdch_res, Es_res, Psh_res, Pcut_res, totalCost

def prepareFigureScenario1(loadDate, totalCost, Cd, Ppv, Pl, Pd, Pch, Pdch, Es, typicalLoadCurve):
    # Prepares figure with results
    print(f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}")
   
    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(10,13))
    # figtitle = f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}"
    figtitle = f"Sistema de Alamcenamiento sin DR, Costo total = ${totalCost:.2f}"

    fig.suptitle(figtitle)
    axs[0].step(range(25),Cd + [Cd[-1]], where='post')
    axs[0].legend(["Cd"],loc='upper left')
    axs[0].minorticks_on()
    axs[0].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[0].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[0].set_ylabel('Costo [$]')
    axs[0].set_xlim([0,24])
    loadCurve = typicalLoadCurve + [typicalLoadCurve[-1]]
    axs0b = axs[0].twinx()
    axs0b.step(range(25), loadCurve, where='post', linestyle='--', color='tab:red')
    axs0b.tick_params(axis='y', labelcolor='tab:red')
    axs0b.legend(["Typical Load Curve"], loc='upper right')
    axs0b.set_ylabel('Potencia [%]', color='tab:red')
    axs0b.tick_params(axis='y', colors='tab:red')
    axs0b.set_xlim([0,24])
    axs[1].step(range(25),Pd, where='post', linestyle='-')
    axs[1].step(range(25),Ppv + [Ppv[-1]], where='post', linestyle='-')
    axs[1].step(range(25),Pl + [Pl[-1]], where='post', linestyle='--')
    axs[1].legend(["Pd", "Ppv", "Pl"], loc="upper left", ncol=3) 
    axs[1].minorticks_on()
    axs[1].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[1].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[1].set_ylabel('Potencia [kW]')
    axs[1].set_xlim([0,24])
    axs[2].step(range(25),Pch, where='post', linestyle='-')
    axs[2].step(range(25),Pdch, where='post', linestyle='-')
    axs[2].legend(["Pch", "Pdch"], loc='upper left', ncol=2) 
    axs[2].minorticks_on()
    axs[2].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[2].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[2].set_xlabel('Tiempo [h]')
    axs[2].set_ylabel('Potencia [kW]')
    axs[2].set_xlim([0,24])
    axs2b = axs[2].twinx()
    axs2b.plot(range(25),Es, linestyle='--', color='tab:red')
    axs2b.tick_params(axis='y', labelcolor='tab:red')
    axs2b.legend(["Es"], loc='upper right')
    axs2b.set_ylabel('Energía [kWh]', color='tab:red')
    axs2b.tick_params(axis='y', colors='tab:red')
    axs2b.set_xlim([0,24])
    
    # filename = "OptimalSolution_" + loadDate.strftime("%Y-%m-%d")
    filename = "OptimalSolution_"
    filename = filename + datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
    plt.savefig("results/" + filename + ".svg")
    plt.show()


def prepareFigureScenario2(loadDate, totalCost, Cd, Ppv, Pl, Pd, Pch, Pdch, Es, Psh, Pcut, typicalLoadCurve):
    # Prepares figure with results
    print(f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}")
   
    fig, axs = plt.subplots(4, 1, constrained_layout=True, figsize=(10,13))
    # figtitle = f"Load Date: {loadDate}, Total Energy Cost = ${totalCost:.2f}"
    figtitle = f"Sistema de Alamcenamiento con DR, Costo total = ${totalCost:.2f}"

    loadPower = [Pl[t] + Psh[t] - Pcut[0,t] - Pcut[1,t] - Pcut[2,t] for t in range(24)]

    fig.suptitle(figtitle)
    axs[0].step(range(25),Cd + [Cd[-1]], where='post')
    axs[0].legend(["Cd"],loc='upper left')
    axs[0].minorticks_on()
    axs[0].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[0].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[0].set_ylabel('Costo [$]')
    axs[0].set_xlim([0,24])
    loadCurve = typicalLoadCurve + [typicalLoadCurve[-1]]
    axs0b = axs[0].twinx()
    axs0b.step(range(25), loadCurve, where='post', linestyle='--', color='tab:red')
    axs0b.tick_params(axis='y', labelcolor='tab:red')
    axs0b.legend(["Typical Load Curve"], loc='upper right')
    axs0b.set_ylabel('Potencia [%]', color='tab:red')
    axs0b.tick_params(axis='y', colors='tab:red')
    axs0b.set_xlim([0,24])
    axs[1].step(range(25),Pd, where='post', linestyle='-')
    axs[1].step(range(25),Ppv + [Ppv[-1]], where='post', linestyle='-')
    axs[1].step(range(25), Pl + [Pl[-1]], where='post', linestyle='--')
    axs[1].step(range(25),loadPower + [loadPower[-1]], where='post', linestyle='--')
    axs[1].legend(["Pd", "Ppv", "Pl", "Pl+Psh-sum(Pcut_i)"], loc="upper left", ncol=4)
    axs[1].minorticks_on()
    axs[1].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[1].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[1].set_ylabel('Potencia [kW]')
    axs[1].set_xlim([0,24])
    axs[2].step(range(25),Pch, where='post', linestyle='-')
    axs[2].step(range(25),Pdch, where='post', linestyle='-')
    axs[2].legend(["Pch", "Pdch"], loc='upper left', ncol=2)
    axs[2].minorticks_on()
    axs[2].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[2].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[2].set_ylabel('Potencia [kW]')
    axs[2].set_xlim([0,24])
    axs2b = axs[2].twinx()
    axs2b.plot(range(25),Es, linestyle='--', color='tab:red')
    axs2b.tick_params(axis='y', labelcolor='tab:red')
    axs2b.legend(["Es"], loc='upper right')
    axs2b.set_ylabel('Energía [kWh]', color='tab:red')
    axs2b.tick_params(axis='y', colors='tab:red')
    axs2b.set_xlim([0,24])
    axs[3].step(range(25),Psh, where='post', linestyle='-')
    axs[3].step(range(24),list(Pcut[0,t] for t in range(24)), where='post', linestyle='--')
    axs[3].step(range(24),list(Pcut[1,t] for t in range(24)), where='post', linestyle='-.')
    axs[3].step(range(24),list(Pcut[2,t] for t in range(24)), where='post', linestyle=':')
    axs[3].legend(["Psh", "Pcut1", "Pcut2", "Pcut3"], loc='upper left', ncol=4)
    axs[3].minorticks_on()
    axs[3].grid(b=True, which='major', color='darkgray', linestyle='-')
    axs[3].grid(b=True, which='minor', color='lightgray', linestyle='--')
    axs[3].set_xlabel('Tiempo [h]')
    axs[3].set_ylabel('Potencia [kW]')
    axs[3].set_xlim([0,24])
    
    # filename = "OptimalSolution_" + loadDate.strftime("%Y-%m-%d")
    filename = "OptimalSolution_"
    filename = filename + datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
    plt.savefig("results/" + filename + ".svg")
   
    plt.show()

def exportResults(fileName, Pcharge, Pdischarge, Pgrid):
    data = {'H': [datetime.time(i,0).strftime("%H:%M") for i in range(24)], 'Pch': Pcharge[:-1], 'Pdch': Pdischarge[:-1], 'Pd': Pgrid[:-1]}
    df = pd.DataFrame(data)
    df.to_csv(fileName, sep=';', index=False, line_terminator=';\n', date_format="%H:%M")

