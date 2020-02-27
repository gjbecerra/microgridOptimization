#!/usr/bin/python

# Correa, C.A., Bolanos, R.A., Garces, A. Optimal Operation of Microgrids in the Colombian Energy Market
# Microgrid simulation

import csv
from gurobipy import *
import matplotlib.pyplot as plt

time = []
Pl = []
Ps = []
Cd = []
with open('DemandayPV24DRWThermal.csv', newline='') as csvfile:
    data = csv.DictReader(csvfile,delimiter=",")
    for row in data:
        time.append(row["dummy"])
        Pl.append(float(row["dem"]))
        Ps.append(float(row["PV"]))
        Cd.append(float(row["Cbolsa"]))

# Create a new model
m = Model('microgrid')

# Create paremeters
delta_t = 1
Cm = 550
Pm_min = 0
Pm_max = 20
Pe_max = 10
Ee_min = 0
Ee_max = 60

# Create variables
Pd = m.addVars(range(24), name='PowerDistribution')
Pm = m.addVars(range(24), lb= Pm_min, ub=Pm_max, name='PowerMicroturbine')
Pe = m.addVars(range(24), lb=-Pe_max, ub=Pe_max, name='PowerBatteries')
Ee = m.addVars(range(24), lb= Ee_min, ub=Ee_max, name='EnergyBatteries')

# Create constraints
for t in range(24):
    if t >= 1:
        m.addConstr(Ee[t] == Ee[t-1] + Pe[t-1]*delta_t, name="Constr2[{}]".format(t))
m.addConstrs((Pd[t] == Pl[t] - Pe[t] - Pm[t] - Ps[t] for t in range(24)), name="Constr3[{}]".format(t))
m.addConstrs((Pd[t] <= Pl[t] for t in range(24)), name="Constr7")
m.addConstr(Ee[0] == Ee[23], name="Constr8")
# m.addConstr(Ee[0] == 10, name="Constr9")
m.update()

# Create objective function
m.setObjective(quicksum(Cd[t]*Pd[t] + Cm*Pm[t] for t in range(24)), GRB.MINIMIZE)

# Run the optimizer
m.optimize()

# Get solution
if m.status == GRB.Status.OPTIMAL:
    Pd_sol = m.getAttr('x', Pd)
    Pm_sol = m.getAttr('x', Pm)
    Pe_sol = m.getAttr('x', Pe)
    Ee_sol = m.getAttr('x', Ee)

    plt.subplot(221)
    plt.plot(range(24),Pd_sol.values())
    plt.plot(range(24),Pl)
    plt.legend(["Power from distribution network", "Power on load"])
    plt.subplot(222)
    plt.plot(range(24),Pm_sol.values())
    plt.plot(range(24),Ps)
    plt.legend(["Power from microturbine", "Power from solar panels"])
    plt.subplot(223)
    plt.plot(range(24),Pe_sol.values())
    plt.legend(["Power in battery"])
    plt.subplot(224)
    plt.plot(range(24),Ee_sol.values())
    plt.legend(["Energy in battery"])
    plt.grid(True)
    plt.show()
