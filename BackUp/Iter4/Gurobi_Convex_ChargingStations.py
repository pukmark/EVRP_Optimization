import numpy as np
from scipy.stats import norm
import SimDataTypes
import gurobipy as gp
from gurobipy import GRB
import time
import itertools
import math

def SolveGurobi_Convex_MinMax(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N

    model = gp.Model("ChargingStations_MinMax")

    # Create variables
    DecisionVar = model.addMVar(shape=(n,n),vtype=GRB.BINARY, name="DecisionVar")
    y = model.addMVar(n, name="y") # the charging time spent at each node
    a2y = model.addMVar(n, lb=-GRB.INFINITY, name="a2y") # the charging time spent at each node
    expy = model.addMVar(n, name="expy") # the charging time spent at each node
    SigmaTf2 = model.addMVar(n, name="SigmaTf2") # Time elapsed when arriving at each node
    SigmaTf = model.addMVar(n, name="SigmaTf") # Time elapsed when arriving at each node
    Tf = model.addMVar(n, name="Tf") # Final time

    if n == 47:
        InitialTraj = np.zeros((n,n))
        InitialTraj[0,1] = 1; InitialTraj[0,9] = 1; InitialTraj[0,19] = 1; InitialTraj[0,27] = 1; InitialTraj[0,36] = 1;
        InitialTraj[1,2] = 1; InitialTraj[9,10] = 1; InitialTraj[19,20] = 1; InitialTraj[27,28] = 1; InitialTraj[36,37] = 1;
        InitialTraj[2,3] = 1; InitialTraj[10,11] = 1; InitialTraj[20,21] = 1; InitialTraj[28,29] = 1; InitialTraj[37,38] = 1;
        InitialTraj[3,4] = 1; InitialTraj[11,12] = 1; InitialTraj[21,22] = 1; InitialTraj[29,30] = 1; InitialTraj[38,39] = 1;
        InitialTraj[4,5] = 1; InitialTraj[12,13] = 1; InitialTraj[22,23] = 1; InitialTraj[30,31] = 1; InitialTraj[39,40] = 1;
        InitialTraj[5,6] = 1; InitialTraj[13,14] = 1; InitialTraj[23,24] = 1; InitialTraj[31,32] = 1; InitialTraj[40,41] = 1;
        InitialTraj[6,7] = 1; InitialTraj[14,15] = 1; InitialTraj[24,25] = 1; InitialTraj[32,33] = 1; InitialTraj[41,42] = 1;
        InitialTraj[7,8] = 1; InitialTraj[15,16] = 1; InitialTraj[25,26] = 1; InitialTraj[33,34] = 1; InitialTraj[42,43] = 1;
        InitialTraj[8,0] = 1; InitialTraj[16,17] = 1; InitialTraj[26,0] = 1; InitialTraj[34,35] = 1; InitialTraj[43,44] = 1;
        InitialTraj[17,18] = 1; InitialTraj[35,0] = 1; InitialTraj[44,45] = 1;
        InitialTraj[18,0] = 1; InitialTraj[45,46] = 1;
        InitialTraj[46,0] = 1;
        DecisionVar.Start = InitialTraj

    # if n == 20:
    #     InitialTraj = np.zeros((n,n),dtype=int)
    #     BestTraj =  np.array([[ 0 , 4 , 5 , 1  ,9 , 6 , 8 ,12, 17 ,10 , 3, 19 , 0],
    #                     [ 0, 11 ,18 ,14 ,16,15 ,13,  2 , 7 , 0]])
    #     for i in range(BestTraj.shape[0]):
    #         for j in range(len(BestTraj[i])-1):
    #             InitialTraj[BestTraj[i][j],BestTraj[i][j+1]] = 1
    #     DecisionVar.Start = InitialTraj

    if n ==15:
        InitialTraj = np.zeros((n,n),dtype=int)
        BestTraj =  np.array([[ 0,  7, 11,  2 ,12 , 5  ,6 , 9 ,10 , 4 , 0  ],
                                [ 0 , 8 ,14, 13 , 3 , 1 , 0 ]])
        for i in range(BestTraj.shape[0]):
            for j in range(len(BestTraj[i])-1):
                InitialTraj[BestTraj[i][j],BestTraj[i][j+1]] = 1
        DecisionVar.Start = InitialTraj
    


    if n == 12:
        BestTraj =  np.array([[ 0,  5,  2 , 1  ,8 , 9 , 0 ],
                                [ 0 , 6  ,4, 10,  0 ],
                                [ 0 , 7 , 3, 11,  0 ]])
        InitialTraj = np.zeros((n,n),dtype=int)
        for i in range(BestTraj.shape[0]):
            for j in range(len(BestTraj[i])-1):
                InitialTraj[BestTraj[i][j],BestTraj[i][j+1]] = 1
        DecisionVar.Start = InitialTraj
    
    # Defining the objective function
    TimeAlpha = norm.ppf(NominalPlan.SolutionProbabilityTimeReliability)
    EnergyAlpha = norm.ppf(NominalPlan.SolutionProbabilityEnergyReliability)
    SigmaFactor = np.mean(NominalPlan.TravelSigma)*n/max(1,NominalPlan.NumberOfCars)
    if NominalPlan.ReturnToBase == True:
        if NominalPlan.CostFunctionType == 1:
            SumSigmaTf = model.addVar(name="SigmaTf")
            model.setObjective(y.sum() + TimeAlpha*SumSigmaTf + gp.quicksum(DecisionVar[i,j]*NominalPlan.NodesTimeOfTravel[i,j]  for i in range(n) for j in range(n)), GRB.MINIMIZE)
            SumSigmaTf2 = model.addVar(name="SigmaTf2")
            model.addConstr(SumSigmaTf2 == SumSigmaTf*SumSigmaTf)
            model.addConstr(SumSigmaTf2 == sum(DecisionVar[i,j]*NominalPlan.TravelSigma[i,j]**2 for i in range(n) for j in range(n)))
        elif NominalPlan.CostFunctionType == 2:
            z = model.addVar(name="Z")
            model.setObjective(z, GRB.MINIMIZE)
            model.addConstrs((z >= Tf[i]+TimeAlpha*SigmaTf[i] for i in range(n)), name="max_contraint")
# else:
        # objective = cp.Minimize(cp.norm(MeanT, 'inf') + alpha*cp.pnorm(cp.multiply(NominalPlan.TravelSigma,DecisionVar)))

    # Defining the constraints
    model.addConstrs(DecisionVar[i,i]==0 for i in range(0,n)) # No self loops
    if NominalPlan.ReturnToBase == True:
        model.addConstrs(DecisionVar[i,:].sum()==1 for i in range(1,n)) # Every node has 1 exit edge
        model.addConstrs(DecisionVar[:,i].sum()==1 for i in range(1,n)) # Every node has 1 entry edge
        model.addConstr(DecisionVar[:,0].sum()==DecisionVar[0,:].sum()) # Node 0 has the same number of entry and exit edges
        if NominalPlan.NumberOfCars == 0:
            model.addConstr(DecisionVar[0,:].sum()>=1) # Node 0 has Ncars exit edges
        else:
            model.addConstr(DecisionVar[0,:].sum()==NominalPlan.NumberOfCars) # Node 0 has Ncars exit edges
    else:
        model.addConstr(DecisionVar.sum(0, '*')==NominalPlan.NumberOfCars) # Node 0 has Ncars exit edges
        model.addConstr(DecisionVar.sum('*', 0)==0) # Node 0 has no entry edges
        model.addConstrs(DecisionVar.sum('*', i)==1 for i in range(1,n)) # Every node has 1 entry edge

    # Miller–Tucker–Zemlin formulation for subtour elimination
    # u = model.addMVar(n, name="u") # u[i] is the position of node i in the path
    # model.addConstr(u[0]==1) # Node 0 is the first node
    # model.addConstrs(u[i] >= 2 for i in range(1,n)) # Node 0 is the first node
    # model.addConstrs(u[i] <= n for i in range(1,n))
    # model.addConstrs(u[i] + NominalPlan.MaxNumberOfNodesPerCar*(DecisionVar[i,j]-1) + 1  <= u[j] for i in range(1,n) for j in range(1,n) if i != j)

    # Dantzig–Fulkerson–Johnson formulation for subtour elimination
    # for m in range(NominalPlan.MaxNumberOfNodesPerCar+1,n):
        # premut = list(itertools.combinations(range(n),m))
        # model.addConstrs(gp.quicksum(DecisionVar[premut[i][j],premut[i][j+1]] for j in range(m-1))+DecisionVar[premut[i][m-1],premut[i][0]] <= m-1 for i in range(min(3000,len(premut))))

    # energy constraints:
    e = model.addMVar(n, name="e") # the energy when arriving at each node
    ef = model.addMVar(n, name="ef") # the energy when arriving at each node
    eSigma2 = model.addMVar(n, name="eSigma2") # the energy when arriving at each node
    eSigmaF2 = model.addMVar(n, name="eSigmaF2") # the energy when arriving at each node
    eSigmaF = model.addMVar(n, name="eSigmaF") # the energy when arriving at each node
    eSigma = model.addMVar(n, name="eSigma") # the energy when arriving at each node
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

    # # Energy uncertainty constraints
    eSigmaMax = np.max(NominalPlan.NodesEnergyTravelSigma)**2
    model.addConstr(eSigma2[0] == 0.0)
    for i in range(0, n):
        model.addConstr(eSigma2[i] == eSigma[i]*eSigma[i])
        model.addConstr(eSigmaF2[i] == eSigmaF[i]*eSigmaF[i])
        model.addConstr(eSigmaF2[i] + (1-DecisionVar[i,0])*eSigmaMax >= eSigma2[i] + NominalPlan.NodesEnergyTravelSigma[i,0]**2 )
        for j in range(1, n):
            if i != j:
                # Energy Sumation over edges (with charging), if edges are not connected, then the energy is not affected
                model.addConstr( eSigma2[j] + (1-DecisionVar[i,j])*eSigmaMax*n >= eSigma2[i] + NominalPlan.NodesEnergyTravelSigma[i,j]**2 )
                model.addConstr( e[j] + (DecisionVar[i,j]-1)*PltParams.BatteryCapacity <= ef[i] + NominalPlan.NodesEnergyTravel[i,j] ) # Energy Sumation over edges
    
    # Add the energy constraints for the return to base case
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            model.addConstr(PltParams.MinimalSOC + (DecisionVar[i,0]-1)*PltParams.BatteryCapacity <= ef[i] - EnergyAlpha*eSigmaF[i] + NominalPlan.NodesEnergyTravel[i,0] )

    # Time Dynamics (constraints):
    MeanT = model.addMVar(n, name="MeanT") # Time elapsed when arriving at each node
    SigmaT2 = model.addMVar(n, name="SigmaT2") # Time elapsed when arriving at each node
    MaxChargeT = (PltParams.BatteryCapacity-PltParams.MinimalSOC)/NominalPlan.StationRechargePower
    LargeT = np.max(NominalPlan.NodesTimeOfTravel)*n+NominalPlan.NumberOfChargeStations*MaxChargeT
    MaxSigmaT = np.max(NominalPlan.TravelSigma)**2
    model.addConstr(MeanT[0] == 0.0)
    model.addConstr(SigmaT2[0] == 0.0)
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                if np.any(i == NominalPlan.ChargingStations):
                    model.addConstr( MeanT[j] + (1-DecisionVar[i,j])*LargeT >= MeanT[i] + DecisionVar[i,j]*NominalPlan.NodesTimeOfTravel[i,j] + y[i] )
                else:
                    model.addConstr( MeanT[j] + (1-DecisionVar[i,j])*LargeT >= MeanT[i] + DecisionVar[i,j]*NominalPlan.NodesTimeOfTravel[i,j] )
                model.addConstr( SigmaT2[j] + (1-DecisionVar[i,j])*MaxSigmaT*n >= SigmaT2[i] + DecisionVar[i,j]*NominalPlan.TravelSigma[i,j]**2 )
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
                if np.any(i == NominalPlan.ChargingStations):
                    model.addConstr(Tf[i] + (1-DecisionVar[i,0])*LargeT >= MeanT[i] + NominalPlan.NodesTimeOfTravel[i,0] + y[i] )
                else:
                    model.addConstr(Tf[i] + (1-DecisionVar[i,0])*LargeT >= MeanT[i] + NominalPlan.NodesTimeOfTravel[i,0] )
                model.addConstr(SigmaTf2[i] + (1-DecisionVar[i,0])*MaxSigmaT >= SigmaT2[i] + NominalPlan.TravelSigma[i,0]**2 )
        model.addConstrs(SigmaTf2[i] == SigmaTf[i]*SigmaTf[i] for i in range(0, n))
        model.addConstrs((Tf[i]+TimeAlpha*SigmaTf[i] <= NominalPlan.MaxTotalTimePerVehicle for i in range(n)))
    else: # No return to base
        SigmaT = model.addMVar(n, name="SigmaT")
        model.addConstrs(SigmaT2[i] == SigmaT[i]*SigmaT[i] for i in range(0, n))
        model.addConstrs((MeanT[i]+TimeAlpha*SigmaT[i] <= NominalPlan.MaxTotalTimePerVehicle for i in range(n)))

    model._cur_obj = float('inf')
    model._time = time.time()

    # Optimize model
    model.setParam("Threads", 14)
    model.setParam("NodefileStart", 0.5)
    # model.setParam("WorkLimit", 30)
    # model.setParam("MIPGap", 0.01)
    model.setParam("NonConvex", 2)
    # model.setParam("MIPFocus", 3)
    model.setParam("DisplayInterval", 30)
    model.optimize(callback=cb)


    # # Transforming the solution to a path
    X_sol = np.argwhere(np.round(DecisionVar.X)==1)
    if NominalPlan.NumberOfCars == 0:
        M = int(np.sum(DecisionVar.X[0,:]))
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
    if (time.time() - model._time > 360) and model._cur_obj< 1.0e8:
        model.terminate()
    elif (time.time() - model._time > 3600):
        model.terminate()
