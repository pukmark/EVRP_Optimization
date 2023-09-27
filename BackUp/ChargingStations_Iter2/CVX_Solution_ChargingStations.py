import numpy as np
import SimDataTypes
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import time


def SolveCVX_ChargingStations(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    DecisionVar = cp.Variable((n,n), boolean=True)
    u = cp.Variable(n) # u[i] is the position of node i in the path
    e = cp.Variable(n) # the energy when arriving at each node
    y = cp.Variable(n) # the charging time spent at each node
    
    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = -999.0
                continue
            else:
                CostMatrix[i,j] = NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[j]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    # Defining the objective function
    objective = cp.Maximize(cp.sum(cp.multiply(CostMatrix, DecisionVar)) - NominalPlan.TimeCoefInCost*y@ones)

    # Defining the constraints
    constraints = []
    if NominalPlan.ReturnToBase == True:
        constraints += [DecisionVar[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar[1:,:] @ ones == ones_m1]
        constraints += [DecisionVar.T[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1]
    else:
        ones_m1 = np.ones((n-1,1))
        constraints += [DecisionVar[0,:] @ ones == 1] # Node 0 has exit edge
        constraints += [cp.sum(DecisionVar[:,0]) == 0] # Node 0 has no entry edges
        constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
        constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge

    constraints += [u[1:] >= 2] # Node 0 is the first node
    constraints += [u[1:] <= n]
    constraints += [u[0] == 1] # Node 0 is the first node
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints += [ u[i] + NominalPlan.MaxNumberOfNodesPerCar*DecisionVar[i, j]  <= u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 ]

    # energy constraints:
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            constraints += [y[i] >= 0.0] # Charging time is always positive
        else:
            constraints += [y[i] == 0.0] # Not a charging station, no charging time
    constraints += [e[0] == PowerLeft] # Initial energy
    constraints += [e[1:] >= PltParams.MinimalSOC] # Energy is always positive
    constraints += [e[1:] <= PltParams.BatteryCapacity] # Energy is always less than battery capacity

    # Energy constraints
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ e[j] + (DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            constraints += [PltParams.MinimalSOC*DecisionVar[i, 0] + (DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,0]*DecisionVar[i, 0] + y[i]*NominalPlan.StationRechargePower ]

    # Energy Dynamics:
    # for j in range(1, n):
    #     for i in range(0, n):
    #         constraints += [ e[j] == e[i] + DecisionVar[i, j]*(y[i]*NominalPlan.StationRechargePower-NominalPlan.NodesEnergyTravel[i,j]-DummyVar[i,j]) + DummyVar[i,j] ]


    # Limit the Charging capacity
    for i in range(1, n):
        if np.any(i == NominalPlan.ChargingStations):
            for j in range(1, n):
                if i != j:
                    constraints += [ e[j] <= PltParams.BatteryCapacity - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j]] # Can't entry a node fully charged

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI, MIPGap=0.01, Threads=15) #, MIPFocus=2

    # Transforming the solution to a path
    X_sol = np.argwhere(DecisionVar.value==1)
    NodesTrajectory = np.zeros((n, NominalPlan.NumberOfCars), dtype=int)

    for m in range(NominalPlan.NumberOfCars):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, prob.value, e.value, y.value
    # Construct the problem.




def SolveCVX_ChargingStations_MinMax(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    DecisionVar = cp.Variable((n,n), boolean=True)
    u = cp.Variable(n, integer=True) # u[i] is the position of node i in the path
    e = cp.Variable(n) # the energy when arriving at each node
    y = cp.Variable(n) # the charging time spent at each node
    T = cp.Variable(n) # Time elapsed when arriving at each node
    Tf = cp.Variable(1) # Final time
    
    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = -999.0
                continue
            else:
                CostMatrix[i,j] = NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[j]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    # Defining the objective function
    if NominalPlan.ReturnToBase == True:
        objective = cp.Minimize(Tf)
    else:
        objective = cp.Minimize(cp.norm(T, 'inf'))

    # Defining the constraints
    constraints = []
    constraints += [cp.diag(DecisionVar) == 0] # No self loops
    if NominalPlan.ReturnToBase == True:
        constraints += [DecisionVar[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar[1:,:] @ ones == ones_m1]
        constraints += [DecisionVar.T[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1]
    else:
        ones_m1 = np.ones((n-1,1))
        constraints += [DecisionVar[0,:] @ ones == 1] # Node 0 has exit edge
        constraints += [cp.sum(DecisionVar[:,0]) == 0] # Node 0 has no entry edges
        constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
        constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge

    constraints += [u[1:] >= 2] # Node 0 is the first node
    constraints += [u[1:] <= n]
    constraints += [u[0] == 1] # Node 0 is the first node
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints += [ u[i] + NominalPlan.MaxNumberOfNodesPerCar*DecisionVar[i, j]  <= u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 ]

    # energy constraints:
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            constraints += [y[i] >= 0.0] # Charging time is always positive
        else:
            constraints += [y[i] == 0.0] # Not a charging station, no charging time
    constraints += [e[0] == PowerLeft] # Initial energy
    constraints += [e[1:] >= PltParams.MinimalSOC] # Energy is always positive
    constraints += [e[1:] <= PltParams.BatteryCapacity] # Energy is always less than battery capacity

    # Energy constraints
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ e[j] + (DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            constraints += [PltParams.MinimalSOC*DecisionVar[i, 0] + (DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,0]*DecisionVar[i, 0] + y[i]*NominalPlan.StationRechargePower ]

    # Limit the Charging capacity
    for i in range(1, n):
        if np.any(i == NominalPlan.ChargingStations):
            for j in range(1, n):
                if i != j:
                    constraints += [ e[j] <= PltParams.BatteryCapacity - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j]] # Can't entry a node fully charged

    # Time Dynamics (constraints):
    MaxT = np.max(NominalPlan.NodesTimeOfTravel)*n
    constraints += [T[0] == 0.0]
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ T[j] + (1-DecisionVar[i, j])*MaxT >= T[i] + DecisionVar[i, j]*NominalPlan.NodesTimeOfTravel[i,j] + y[i] ]
    if NominalPlan.ReturnToBase == True:
        constraints += [Tf >= 0.0]
        L = MaxT*n
        for i in range(1, n):
                constraints += [Tf + (1-DecisionVar[i, 0])*L >= T[i] + NominalPlan.NodesTimeOfTravel[i,0] + y[i] ]

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    # prob._cur_obj = float('inf')
    # prob._time = time.time()

    # prob.optimize(callback=cb)
    prob.solve(verbose=True, solver=cp.GUROBI, MIPGap=0.01, Threads=16, TimeLimit=1000) #, MIPFocus=2


    # Transforming the solution to a path
    X_sol = np.argwhere(np.round(DecisionVar.value)==1)
    NodesTrajectory = np.zeros((n, NominalPlan.NumberOfCars), dtype=int)

    for m in range(NominalPlan.NumberOfCars):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, prob.value, e.value, y.value
    # Construct the problem.



def cb(model, where):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 120s
    if time.time() - model._time > 200:
        model.terminate()





