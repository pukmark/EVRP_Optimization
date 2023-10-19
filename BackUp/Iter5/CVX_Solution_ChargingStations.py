import numpy as np
from scipy.stats import norm
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
                CostMatrix[i,j] = -NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
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
    # constraints += [e[1:] <= PltParams.BatteryCapacity] # Energy is always less than battery capacity

    # Energy constraints
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ e[j] + (DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            constraints += [PltParams.MinimalSOC + (DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,0] + y[i]*NominalPlan.StationRechargePower ]

    # Energy Dynamics:
    # for j in range(1, n):
    #     for i in range(0, n):
    #         constraints += [ e[j] == e[i] + DecisionVar[i, j]*(y[i]*NominalPlan.StationRechargePower-NominalPlan.NodesEnergyTravel[i,j]-DummyVar[i,j]) + DummyVar[i,j] ]


    # Limit the Charging capacity
    for i in range(1, n):
        if np.any(i == NominalPlan.ChargingStations):
            for j in range(1, n):
                if i != j:
                    constraints += [ e[j] <= PltParams.BatteryCapacity + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j]] # Can't entry a node fully charged

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    # prob.solve(verbose=True, solver=cp.GUROBI, MIPGap=0.08, Threads=16, WorkLimit=60) #, MIPFocus=2
    prob.solve(verbose=True, solver=cp.GUROBI, Threads=16) #, MIPFocus=2

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
    MeanT = cp.Variable(n) # Time elapsed when arriving at each node
    SigmaT2 = cp.Variable(n) # Time elapsed when arriving at each node
    SigmaTf2 = cp.Variable(n) # Time elapsed when arriving at each node
    Tf = cp.Variable(n) # Final time
    
    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = -999.0
                continue
            else:
                CostMatrix[i,j] = -NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    # Defining the objective function
    alpha = norm.ppf(NominalPlan.SolutionProbabilityReliability)*0
    SigmaFactor = np.mean(NominalPlan.TravelSigma)
    if NominalPlan.ReturnToBase == True:
        x = (SigmaTf2-SigmaFactor**2)/SigmaFactor**2
        objective = cp.Minimize(cp.norm(Tf + alpha*SigmaFactor*(1+0.5*x), 'inf'))
    else:
        objective = cp.Minimize(cp.norm(MeanT, 'inf') + alpha*cp.pnorm(cp.multiply(NominalPlan.TravelSigma,DecisionVar)))

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
                constraints += [ u[i] + NominalPlan.MaxNumberOfNodesPerCar*DecisionVar[i,j]  <= u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 ]

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
                constraints += [ e[j] + (DecisionVar[i,j]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i,j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            constraints += [PltParams.MinimalSOC + (DecisionVar[i,0]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,0] + y[i]*NominalPlan.StationRechargePower ]

    # Limit the Charging capacity
    for i in NominalPlan.ChargingStations[:,0]:
        constraints += [e[i] + y[i]*NominalPlan.StationRechargePower <= PltParams.BatteryCapacity] # Can't entry a node fully charged

    # Time Dynamics (constraints):
    MaxT = np.max(NominalPlan.NodesTimeOfTravel)*n
    MaxSigmaT = n*np.max(NominalPlan.TravelSigma)**2
    constraints += [MeanT[0] == 0.0]
    constraints += [SigmaT2[0] == 0.0]
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ MeanT[j] + (1-DecisionVar[i,j])*MaxT >= MeanT[i] + DecisionVar[i,j]*NominalPlan.NodesTimeOfTravel[i,j] + y[i] ]
                constraints += [ SigmaT2[j] + (1-DecisionVar[i,j])*MaxSigmaT >= SigmaT2[i] + DecisionVar[i,j]*NominalPlan.TravelSigma[i,j]**2 ]
    if NominalPlan.ReturnToBase == True:
        constraints += [Tf[0] == 0.0]
        constraints += [Tf[:1] >= 0.0]
        constraints += [SigmaTf2[0] == 0.0]
        constraints += [SigmaTf2[:1] >= 0.0]
        L = MaxT*n
        Lf = 10*MaxSigmaT*n
        for i in range(1, n):
                constraints += [Tf[i] + (1-DecisionVar[i,0])*L >= MeanT[i] + NominalPlan.NodesTimeOfTravel[i,0] + y[i] ]
                constraints += [SigmaTf2[i] + (1-DecisionVar[i,0])*Lf >= SigmaT2[i] + NominalPlan.TravelSigma[i,0]**2 ]

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, 
               solver=cp., 
               MIPGap=0.01, 
               Threads=14, 
               TimeLimit=70) #, MIPFocus=2


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

