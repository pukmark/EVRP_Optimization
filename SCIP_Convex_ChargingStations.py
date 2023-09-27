import numpy as np
import scipy as sp
import SimDataTypes
from scipy import Model
import pyscipopt as sp
import time

def SolveSCIP_Convex_MinMax(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N

    model = sp.Model("ChargingStations_MinMax")

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
    
    # Defining the objective function
    TimeAlpha = norm.ppf(NominalPlan.SolutionProbabilityTimeReliability)
    EnergyAlpha = norm.ppf(NominalPlan.SolutionProbabilityEnergyReliability)
    SigmaFactor = np.mean(NominalPlan.TravelSigma)*n/max(1,NominalPlan.NumberOfCars)
    if NominalPlan.ReturnToBase == True:
        model.setObjective(z + KcostCars*Ncars, GRB.MINIMIZE)
    # else:
        # objective = cp.Minimize(cp.norm(MeanT, 'inf') + alpha*cp.pnorm(cp.multiply(NominalPlan.TravelSigma,DecisionVar)))

    # Defining the constraints
    model.addConstrs(DecisionVar[i,i]==0 for i in range(0,n)) # No self loops
    if NominalPlan.ReturnToBase == True:
        model.addConstrs(DecisionVar[i,:].sum()==1 for i in range(1,n)) # Every node has 1 exit edge
        model.addConstrs(DecisionVar[:,i].sum()==1 for i in range(1,n)) # Every node has 1 entry edge
        model.addConstr(DecisionVar[0,:].sum()==Ncars) # Node 0 has Ncars exit edges
        model.addConstr(DecisionVar[:,0].sum()==Ncars) # Node 0 has Ncars entry edges
    else:
        model.addConstr(DecisionVar.sum(0, '*')==Ncars) # Node 0 has Ncars exit edges
        model.addConstr(DecisionVar.sum('*', 0)==0) # Node 0 has no entry edges
        model.addConstrs(DecisionVar.sum('*', i)==1 for i in range(1,n)) # Every node has 1 entry edge

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
    for i in range(0, n):
        model.addConstr(eSigma2[i] == eSigma[i]*eSigma[i])
        model.addConstr(eSigmaF2[i] == eSigmaF[i]*eSigmaF[i])
        model.addConstr(eSigmaF2[i] + (1-DecisionVar[i,0])*PltParams.BatteryCapacity*n >= eSigma2[i] + NominalPlan.NodesEnergyTravelSigma[i,0]**2 )
        for j in range(1, n):
            if i != j:
                # Energy Sumation over edges (with charging), if edges are not connected, then the energy is not affected
                model.addConstr( eSigma2[j] + (1-DecisionVar[i,j])*PltParams.BatteryCapacity >= eSigma2[i] + NominalPlan.NodesEnergyTravelSigma[i,j]**2 )
                model.addConstr( e[j] + (DecisionVar[i,j]-1)*PltParams.BatteryCapacity <= ef[i] + NominalPlan.NodesEnergyTravel[i,j] ) # Energy Sumation over edges
    
    # Add the energy constraints for the return to base case
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            model.addConstr(PltParams.MinimalSOC + (DecisionVar[i,0]-1)*PltParams.BatteryCapacity <= ef[i] - EnergyAlpha*eSigmaF[i] + NominalPlan.NodesEnergyTravel[i,0] )

    # Limit the Charging capacity
    for i in NominalPlan.ChargingStations[:,0]:
        model.addConstr(e[i] + y[i]*NominalPlan.StationRechargePower <= PltParams.BatteryCapacity) # Can't entry a node fully charged

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

    # Calculate the objective function
    model.addConstrs(SigmaTf2[i] == SigmaTf[i]*SigmaTf[i] for i in range(0, n))
    model.addConstrs((z >= Tf[i]+TimeAlpha*SigmaTf[i] for i in range(n)), name="max_contraint")

    model._cur_obj = float('inf')
    model._time = time.time()

    # Optimize model
    model.setParam("Threads", 14)
    model.setParam("WorkLimit", 160)
    model.setParam("MIPGap", 0.01)
    model.setParam("NonConvex", 2)
    # model.setParam("MIPFocus", 1)
    model.setParam("DisplayInterval", 30)
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
