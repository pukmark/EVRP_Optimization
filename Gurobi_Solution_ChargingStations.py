import numpy as np
from scipy.stats import norm
import SimDataTypes
import gurobipy as gp
from gurobipy import GRB
import time


# def SolveCVX_ChargingStations(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
#     n = NominalPlan.N
#     DecisionVar = cp.Variable((n,n), boolean=True)
#     u = cp.Variable(n) # u[i] is the position of node i in the path
#     e = cp.Variable(n) # the energy when arriving at each node
#     y = cp.Variable(n) # the charging time spent at each node
    
#     # Construct Cost Vector for every decision variable
#     CostMatrix = np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             if i==j:
#                 CostMatrix[i,j] = -999.0
#                 continue
#             else:
#                 CostMatrix[i,j] = -NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
#     ones = np.ones((n,1))
#     ones_m1 = np.ones((n-1,1))
#     # Defining the objective function
#     objective = cp.Maximize(cp.sum(cp.multiply(CostMatrix, DecisionVar)) - NominalPlan.TimeCoefInCost*y@ones)

#     # Defining the constraints
#     constraints = []
#     if NominalPlan.ReturnToBase == True:
#         constraints += [DecisionVar[0,:] @ ones == NominalPlan.NumberOfCars]
#         constraints += [DecisionVar[1:,:] @ ones == ones_m1]
#         constraints += [DecisionVar.T[0,:] @ ones == NominalPlan.NumberOfCars]
#         constraints += [DecisionVar.T[1:,:] @ ones == ones_m1]
#     else:
#         ones_m1 = np.ones((n-1,1))
#         constraints += [DecisionVar[0,:] @ ones == 1] # Node 0 has exit edge
#         constraints += [cp.sum(DecisionVar[:,0]) == 0] # Node 0 has no entry edges
#         constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
#         constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
#         constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge

#     constraints += [u[1:] >= 2] # Node 0 is the first node
#     constraints += [u[1:] <= n]
#     constraints += [u[0] == 1] # Node 0 is the first node
#     for i in range(1, n):
#         for j in range(1, n):
#             if i != j:
#                 constraints += [ u[i] + NominalPlan.MaxNumberOfNodesPerCar*DecisionVar[i, j]  <= u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 ]

#     # energy constraints:
#     for i in range(0, n):
#         if np.any(i == NominalPlan.ChargingStations):
#             constraints += [y[i] >= 0.0] # Charging time is always positive
#         else:
#             constraints += [y[i] == 0.0] # Not a charging station, no charging time
#     constraints += [e[0] == PowerLeft] # Initial energy
#     constraints += [e[1:] >= PltParams.MinimalSOC] # Energy is always positive
#     # constraints += [e[1:] <= PltParams.BatteryCapacity] # Energy is always less than battery capacity

#     # Energy constraints
#     for i in range(0, n):
#         for j in range(1, n):
#             if i != j:
#                 constraints += [ e[j] + (DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
#     if NominalPlan.ReturnToBase == True:
#         for i in range(1, n):
#             constraints += [PltParams.MinimalSOC + (DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= e[i] + NominalPlan.NodesEnergyTravel[i,0] + y[i]*NominalPlan.StationRechargePower ]

#     # Energy Dynamics:
#     # for j in range(1, n):
#     #     for i in range(0, n):
#     #         constraints += [ e[j] == e[i] + DecisionVar[i, j]*(y[i]*NominalPlan.StationRechargePower-NominalPlan.NodesEnergyTravel[i,j]-DummyVar[i,j]) + DummyVar[i,j] ]


#     # Limit the Charging capacity
#     for i in range(1, n):
#         if np.any(i == NominalPlan.ChargingStations):
#             for j in range(1, n):
#                 if i != j:
#                     constraints += [ e[j] <= PltParams.BatteryCapacity + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j]] # Can't entry a node fully charged

#     # Solving the problem
#     prob = cp.Problem(objective, constraints)
#     # prob.solve(verbose=True, solver=cp.GUROBI, MIPGap=0.08, Threads=16, WorkLimit=60) #, MIPFocus=2
#     prob.solve(verbose=True, solver=cp.GUROBI, Threads=16) #, MIPFocus=2

#     # Transforming the solution to a path
#     X_sol = np.argwhere(DecisionVar.value==1)
#     NodesTrajectory = np.zeros((n, NominalPlan.NumberOfCars), dtype=int)

#     for m in range(NominalPlan.NumberOfCars):
#         NodesTrajectory[0,m] = 0
#         NodesTrajectory[1,m] = X_sol[m,1]
#         i = 2
#         while True:
#             NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
#             if NodesTrajectory[i,m] == 0:
#                 break
#             i += 1

#     return NodesTrajectory, prob.value, e.value, y.value
#     # Construct the problem.




def SolveGurobi_ChargingStations_MinMax(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N

    model = gp.Model("ChargingStations_MinMax")

    # Create variables
    DecisionVar = model.addMVar(shape=(n,n),vtype=GRB.BINARY, name="DecisionVar")
    u = model.addMVar(n, name="u") # u[i] is the position of node i in the path
    e = model.addMVar(n, name="e") # the energy when arriving at each node
    ef = model.addMVar(n, name="ef") # the energy when arriving at each node
    eSigma2 = model.addMVar(n, name="eSigma2") # the energy when arriving at each node
    eSigmaF2 = model.addMVar(n, name="eSigmaF2") # the energy when arriving at each node
    eSigmaF = model.addMVar(n, name="eSigmaF") # the energy when arriving at each node
    eSigma = model.addMVar(n, name="eSigma") # the energy when arriving at each node
    y = model.addMVar(n, name="y") # the charging time spent at each node
    a2y = model.addMVar(n, lb=-GRB.INFINITY, name="a2y") # the charging time spent at each node
    expy = model.addMVar(n, name="expy") # the charging time spent at each node
    MeanT = model.addMVar(n, name="MeanT") # Time elapsed when arriving at each node
    SigmaT2 = model.addMVar(n, name="SigmaT2") # Time elapsed when arriving at each node
    SigmaTf2 = model.addMVar(n, name="SigmaTf2") # Time elapsed when arriving at each node
    SigmaTf = model.addMVar(n, name="SigmaTf") # Time elapsed when arriving at each node
    Tf = model.addMVar(n, name="Tf") # Final time
    z = model.addVar(name="Z")
    if NominalPlan.NumberOfCars == 0:
        Ncars = model.addMVar(1,vtype=GRB.INTEGER,name="Ncars")
        model.addConstr(Ncars>=1)
        Ncars.Start = np.ceil(n/2)
        KcostCars = 100
    else:
        Ncars = NominalPlan.NumberOfCars
        KcostCars = 0


    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = -999.0
                continue
            else:
                CostMatrix[i,j] = -NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    
    # Defining the objective function
    TimeAlpha = norm.ppf(NominalPlan.SolutionProbabilityTimeReliability)
    EnergyAlpha = norm.ppf(NominalPlan.SolutionProbabilityEnergyReliability)
    if NominalPlan.ReturnToBase == True:
        model.setObjective(z + KcostCars*Ncars, GRB.MINIMIZE)
    # else:
        # objective = cp.Minimize(cp.norm(MeanT, 'inf') + alpha*cp.pnorm(cp.multiply(NominalPlan.TravelSigma,DecisionVar)))

    # Defining the constraints
    model.addConstrs(DecisionVar[i,i]==0 for i in range(0,n)) # No self loops
    model.addConstr(DecisionVar.sum(0, '*')==Ncars) # Node 0 has Ncars exit edges
    model.addConstrs(DecisionVar.sum('*', i)==1 for i in range(1,n)) # Every node has 1 entry edge
    if NominalPlan.ReturnToBase == True:
        model.addConstrs(DecisionVar[i,:].sum()==1 for i in range(1,n)) # Every node has 1 exit edge
        model.addConstr(DecisionVar[:,0].sum()==Ncars) # Node 0 has Ncars entry edges
    else:
        model.addConstr(DecisionVar.sum('*', 0)==0) # Node 0 has no entry edges

    model.addConstr(u[0]==1) # Node 0 is the first node
    model.addConstrs(u[i] >= 2 for i in range(1,n)) # Node 0 is the first node
    model.addConstrs(u[i] <= n for i in range(1,n))
    model.addConstrs(u[i] + NominalPlan.MaxNumberOfNodesPerCar*(DecisionVar[i,j]-1) + 1  <= u[j] for i in range(1,n) for j in range(1,n) if i != j)

    # energy constraints:
    a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
    a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            model.addConstr(y[i]>=0.0) # Charging time is always positive
            model.addConstr(a2y[i]==-a2*y[i])
            model.addGenConstrLog(expy[i], a2y[i])
            model.addConstr(ef[i]<= PltParams.BatteryCapacity)
            if PltParams.RechargeModel == 'ConstantRate':
                model.addConstr(ef[i]==e[i]+y[i]*NominalPlan.StationRechargePower)
            elif PltParams.RechargeModel == 'ExponentialRate':
                model.addConstr(ef[i]==a1+(e[i]-a1)*expy[i])
        else:
            model.addConstr(y[i]==0.0) # Not a charging station, no charging time
            model.addConstr(e[i]==ef[i])

    model.addConstr(e[0] == PowerLeft) # Initial energy
    model.addConstrs(e[i] >= PltParams.MinimalSOC+EnergyAlpha*eSigma[i] for i in range(1,n)) # Energy is always positive

    # # Energy constraints
    model.addConstr(eSigma2[0] == 0.0)
    model.addConstrs(eSigma2[i] == eSigma[i]*eSigma[i] for i in range(0,n))
    model.addConstrs(eSigmaF2[i] == eSigmaF[i]*eSigmaF[i] for i in range(0,n))
    model.addConstrs(eSigmaF2[i] + (1-DecisionVar[i,0])*PltParams.BatteryCapacity*n >= eSigma2[i] + NominalPlan.NodesEnergyTravelSigma[i,0]**2 for i in range(0,n))
    for j in range(1, n):
        model.addConstr( e[j] == gp.quicksum(ef[i]*DecisionVar[i,j] + NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i,j] for i in range(0,n)) )
        model.addConstr( eSigma2[j] == gp.quicksum(eSigma2[i]*DecisionVar[i,j] + NominalPlan.NodesEnergyTravelSigma[i,j]**2*DecisionVar[i,j] for i in range(0,n)) )

    # Add the energy constraints for the return to base case
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            model.addConstr(PltParams.MinimalSOC + (DecisionVar[i,0]-1)*PltParams.BatteryCapacity <= ef[i] - EnergyAlpha*eSigmaF[i] + NominalPlan.NodesEnergyTravel[i,0] )

    # Time Dynamics (constraints):
    MaxT = np.max(NominalPlan.NodesTimeOfTravel)*n
    MaxSigmaT = n*np.max(NominalPlan.TravelSigma)**2
    model.addConstr(MeanT[0] == 0.0)
    model.addConstr(SigmaT2[0] == 0.0)
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                model.addConstr( MeanT[j] + (1-DecisionVar[i,j])*MaxT >= MeanT[i] + DecisionVar[i,j]*NominalPlan.NodesTimeOfTravel[i,j] + y[i] )
                model.addConstr( SigmaT2[j] + (1-DecisionVar[i,j])*MaxSigmaT >= SigmaT2[i] + DecisionVar[i,j]*NominalPlan.TravelSigma[i,j]**2 )
    
    if NominalPlan.ReturnToBase == True:
        L = MaxT*n
        Lf = MaxSigmaT*n
        for i in range(1, n):
                model.addConstr(Tf[i] + (1-DecisionVar[i,0])*L >= MeanT[i] + NominalPlan.NodesTimeOfTravel[i,0] + y[i] )
                model.addConstr(SigmaTf2[i] + (1-DecisionVar[i,0])*Lf >= SigmaT2[i] + NominalPlan.TravelSigma[i,0]**2 )

    for i in range(0, n):
        model.addConstr(SigmaTf2[i] == SigmaTf[i]*SigmaTf[i])
    model.addConstrs((z >= Tf[i]+TimeAlpha*SigmaTf[i] for i in range(n)), name="max_contraint")

    model._cur_obj = float('inf')
    model._time = time.time()

    # Optimize model
    model.setParam("Threads", 10)
    model.setParam("TimeLimit", 60)
    model.setParam("MIPGap", 0.01)
    model.setParam("NonConvex", 2)
    model.optimize(callback=cb)


    # # Transforming the solution to a path
    X_sol = np.argwhere(np.round(DecisionVar.X)==1)
    if NominalPlan.NumberOfCars == 0:
        M = int(Ncars.X[0])
    else:
        M = NominalPlan.NumberOfCars
    NodesTrajectory = np.zeros((n+1, M), dtype=int)

    for m in range(M):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, model.ObjVal, e.X, y.X, ef.X

def cb(model, where):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 300s
    if (time.time() - model._time > 3600) and model._cur_obj< 1.0e8:
        model.terminate()
    elif (time.time() - model._time > 3600):
        model.terminate()
