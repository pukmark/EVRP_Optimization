import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from Gurobi_Solution_ChargingStations import *
from Gurobi_Convex_ChargingStations import *
from DivideToGroups import *
from RecursiveOptimalSolution_ChargingStations import *
import os
import time
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(10)

PltParams = DataTypes.PlatformParams()
# Define Simulation Parameters:
##############################$
# Number of Nodes
N = 40
NumberOfCars = 2
MaxNumberOfNodesPerCar = int(1.2*(N-1)/(NumberOfCars)) if NumberOfCars > 1 else N
MaxNodesToSolver = 13
SolutionProbabilityTimeReliability = 0.9
SolutionProbabilityEnergyReliability = 0.999
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
MaxPriority = 1 # Max Priority of a node Priority = 1,...,MaxPriority
CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
MaxTotalTimePerVehicle  = 200.0
PltParams.RechargeModel = 'ConstantRate' # 'ExponentialRate' or 'ConstantRate'
SolverType = 'Recursive' # 'Gurobi' or 'Recursive' or 'Gurobi_NoClustering'
ClusteringMethod = "Max_Eigenvalue" # "Max_Eigenvalue" or "Frobenius" or "Sum_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue"
MaxCalcTimeFromUpdate = 20.0 # Max time to calculate the solution from the last update [sec]
iplot = 3 # 0: No Plot, 1: Plot Main Cluster, 2: Plot All Clusters, 3: Plot Connected Tours
##############################$
# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5

# Map Size
Xmin, Xmax = -100, 100
Ymin, Ymax = -100, 100

# Platform Parameters:
PltParams.Vmax = 10 # Platform Max Speed
PltParams.MinVelReductionCoef, PltParams.MaxVelReductionCoef = 0.0, 0.75 # min/max speed reduction factor for node2node travel
PltParams.VelEnergyConsumptionCoef = 0.04 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
PltParams.VelConstPowerConsumption = 0.04
## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask = 2, 10
PltParams.MinTimePerTask, PltParams.MaxTimePerTask = 1, 5
# PltParams.RechargePowerPerDay = 5
PltParams.BatteryCapacity = 100.0
PltParams.MinimalSOC = 0.0*PltParams.BatteryCapacity
PltParams.FullRechargeRateFactor = 0.25

## Randomize The Nodes Locations:
NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])
# NodesPosition[0,0] = 0.0; NodesPosition[0,1] = 0.0

# Set the Nomianl Time of Travel between any 2 nodes as the distance between
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)
NominalPlan.NodesPosition = NodesPosition
for i in range(N):
    NominalPlan.NodesVelocity[i,i+1:] = 6 #np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
    NominalPlan.NodesVelocity[i+1:,i] = NominalPlan.NodesVelocity[i,i+1:].T
    for j in range(i,N):
        if i==j: continue
        NominalPlan.NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
        NominalPlan.NodesDistance[j,i] = NominalPlan.NodesDistance[i,j]
        NominalPlan.NodesTimeOfTravel[i,j] = NominalPlan.NodesDistance[i,j] / NominalPlan.NodesVelocity[i,j]
        NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
        NominalPlan.TravelSigma[i,j] = np.random.uniform(0.05*NominalPlan.NodesTimeOfTravel[i,j], 0.3*NominalPlan.NodesTimeOfTravel[i,j],1)
        NominalPlan.TravelSigma[j,i] = NominalPlan.TravelSigma[i,j]
        NominalPlan.NodesEnergyTravel[i,j] = -NominalPlan.NodesTimeOfTravel[i,j] * (PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i,j]**2)
        NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]
        NominalPlan.NodesEnergyTravelSigma[i,j] = np.abs(np.random.uniform(0.05*NominalPlan.NodesEnergyTravel[i,j], 0.1*NominalPlan.NodesEnergyTravel[i,j],1))
        NominalPlan.NodesEnergyTravelSigma[j,i] = NominalPlan.NodesEnergyTravelSigma[i,j]

NominalPlan.NodesEnergyTravelSigma2 = NominalPlan.NodesEnergyTravelSigma**2
NominalPlan.TravelSigma2 = NominalPlan.TravelSigma**2

## Calculate Nominal Time to spend and Energy Consumption for task in Node i
NominalPlan.NodesTaskTime = np.random.uniform(PltParams.MinTimePerTask, PltParams.MaxTimePerTask, size=(N,1))
NominalPlan.NodesTaskPower = np.random.uniform(PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask, size=(N,1))

## Nodes Task Prioroties:
NominalPlan.NodesPriorities = np.ceil(np.random.uniform(0,MaxPriority,size=(N,1)))
NominalPlan.N = N

InitialChargeStage = 1.0 * PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
NominalPlan.NumberOfCars = NumberOfCars
NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
NominalPlan.SolutionProbabilityTimeReliability = SolutionProbabilityTimeReliability
NominalPlan.SolutionProbabilityEnergyReliability = SolutionProbabilityEnergyReliability
NominalPlan.CostFunctionType = CostFunctionType
NominalPlan.MaxTotalTimePerVehicle = MaxTotalTimePerVehicle
NominalPlan.EnergyAlpha = norm.ppf(SolutionProbabilityEnergyReliability)
NominalPlan.TimeAlpha = norm.ppf(SolutionProbabilityTimeReliability)
NominalPlan.InitialChargeStage = InitialChargeStage



# Divide the nodes to groups (Clustering):
NodesGroups = DivideNodesToGroups(NominalPlan, NumberOfCars, ClusteringMethod, MustIncludeNodeZero=True, isplot=iplot>=1)


## Charging Stations:
NominalPlan.ChargingStations = list()
for i in range (NumberOfCars):
    NumGroupChargers = min(2,int(np.ceil(NodesGroups[i].shape[0]/10)))
    GroupCharging = NodesGroups[i][list(np.random.randint(1,NodesGroups[i].shape[0],size=(NumGroupChargers,1)))].reshape((-1,))
    while len(np.unique(GroupCharging)) < NumGroupChargers:
        GroupCharging = NodesGroups[i][list(np.random.randint(1,NodesGroups[i].shape[0],size=(NumGroupChargers,1)))].reshape((-1,))
    for j in range(NumGroupChargers):
        NominalPlan.ChargingStations.append(GroupCharging[j])
NominalPlan.ChargingStations = np.array(NominalPlan.ChargingStations).reshape(-1,)
NominalPlan.StationRechargePower = 3
NominalPlan.NumberOfChargeStations = len(NominalPlan.ChargingStations)


t = time.time()
NodesTrajectory = np.zeros((N+1,NumberOfCars), dtype=int)
EnergyEnteringNodes = np.zeros((N,))
ChargingTime = np.zeros((N,))
EnergyExitingNodes = np.zeros((N,))
Cost = 0.0
if SolverType == 'Gurobi_NoClustering':
    NodesTrajectory, Cost, EnergyEnteringNodes, ChargingTime, EnergyExitingNodes = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                        NominalPlan= NominalPlan,
                                                        MaxCalcTimeFromUpdate= MaxCalcTimeFromUpdate,
                                                        PowerLeft= InitialChargeStage)
else:
    for Ncar in range(NumberOfCars):
        NominalPlanGroup = CreateSubPlanFromPlan(NominalPlan, NodesGroups[Ncar])
        NodesTrajectoryGroup = []

        if NominalPlanGroup.N > MaxNodesToSolver:
            NodesSubGroups = []
            NumSubGroups = int(np.ceil(NominalPlanGroup.N/MaxNodesToSolver))
            # Method = "SqrSum_Eigenvalue" # "Max_Eigenvalue" or "Frobenius" or "SqrSum_Eigenvalue" or "Mean_MaxRow"
            NodesSubGroups_unindexed = DivideNodesToGroups(NominalPlanGroup,
                                                        NumSubGroups,
                                                        ClusteringMethod,
                                                        MaximizeGroupSize=True,
                                                        MustIncludeNodeZero=False,
                                                        ChargingStations=NominalPlanGroup.ChargingStations,
                                                        MaxGroupSize = MaxNodesToSolver,
                                                        isplot=iplot>=2)
            for iSubGroup in range(NumSubGroups):
                NodesSubGroups.append(NodesGroups[Ncar][list(NodesSubGroups_unindexed[iSubGroup])])
        else:
            NodesSubGroups = []
            NodesSubGroups.append(NodesGroups[Ncar])
            NumSubGroups = 1
            NominalSubPlanGroup = NominalPlanGroup
        
        MaxChargingPotential = 0.0
        for iSubGroup in range(NumSubGroups):
            if NumSubGroups > 1:
                NominalSubPlanGroup = CreateSubPlanFromPlan(NominalPlan, NodesSubGroups[iSubGroup])
            else:
                NominalSubPlanGroup = NominalPlanGroup
            
            if SolverType == 'Gurobi':

                NodesTrajectorySubGroup, CostSubGroup, EnergyEnteringNodesGroup, ChargingTimeGroup, EnergyExitingNodesGroup = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                                        NominalPlan= NominalSubPlanGroup,
                                                                        MaxCalcTimeFromUpdate= MaxMissionTime,
                                                                        PowerLeft= InitialChargeStage + MaxChargingPotential)
                
                NodesTrajectorySubGroup = list(NodesTrajectorySubGroup.reshape(-1,))
                ChargingStationsDataSubGroup = DataTypes.ChargingStations(NominalSubPlanGroup.ChargingStations)
                for i in range(len(ChargingStationsDataSubGroup.ChargingStationsNodes)):
                    ChargingStationsDataSubGroup.MaxChargingPotential += PltParams.BatteryCapacity - EnergyExitingNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                    ChargingStationsDataSubGroup.EnergyEntered[i] = EnergyEnteringNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                    ChargingStationsDataSubGroup.EnergyExited[i] = EnergyExitingNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                    ChargingStationsDataSubGroup.ChargingTime[i] = ChargingTimeGroup[NominalSubPlanGroup.ChargingStations[i]]
                MaxChargingPotential = ChargingStationsDataSubGroup.MaxChargingPotential
            else:
                for i in range(NominalSubPlanGroup.N):
                    NominalSubPlanGroup.NodesTimeOfTravel[i,i] = 999.0
                    NominalSubPlanGroup.TravelSigma[i,i] = 999.0
                    NominalSubPlanGroup.TravelSigma2[i,i] = 9999.0
                    NominalSubPlanGroup.NodesEnergyTravel[i,i] = -999.0
                    NominalSubPlanGroup.NodesEnergyTravelSigma[i,i] = -999.0
                    NominalSubPlanGroup.NodesEnergyTravelSigma2[i,i] = 9999.0

                BestPlan, NodesTrajectorySubGroup, CostSubGroup, ChargingStationsDataSubGroup = SolveParallelRecursive_ChargingStations(PltParams=PltParams,
                                                                        NominalPlan= NominalSubPlanGroup,
                                                                        i_CurrentNode = 0, 
                                                                        TourTime = 0.0,
                                                                        TourTimeUncertainty = 0.0,
                                                                        EnergyLeft = InitialChargeStage + MaxChargingPotential,
                                                                        EnergyLeftUncertainty = 0.0,
                                                                        ChargingStationsData = DataTypes.ChargingStations(NominalSubPlanGroup.ChargingStations),
                                                                        NodesTrajectory = [], 
                                                                        BestPlan = DataTypes.BestPlan(NominalSubPlanGroup.N, NominalSubPlanGroup.ChargingStations, time.time()),
                                                                        MaxCalcTimeFromUpdate = MaxCalcTimeFromUpdate)

            MaxChargingPotential = ChargingStationsDataSubGroup.MaxChargingPotential
            if NumSubGroups <= 1 or iSubGroup == 0:
                NodesTrajectoryGroup = []
                for i in range(len(NodesTrajectorySubGroup)):
                    NodesTrajectoryGroup.append(NodesSubGroups[iSubGroup][NodesTrajectorySubGroup[i]])
                CostGroup = CostSubGroup
                ChargingStationsDataGroup = ChargingStationsDataSubGroup
                if NumSubGroups <= 1 and ChargingStationsDataSubGroup.ChargingStationsNodes.shape[0] > 0:
                    ChargingStationsDataGroup.ChargingStationsNodes = np.array([NodesSubGroups[iSubGroup][ChargingStationsDataSubGroup.ChargingStationsNodes]]).reshape(-1,)
            else:
                NodesTrajectoryGroup_toAdd = []
                for i in range(len(NodesTrajectorySubGroup)):
                    NodesTrajectoryGroup_toAdd.append(NodesSubGroups[iSubGroup][NodesTrajectorySubGroup[i]])
                NodesTrajectoryGroup, CostGroup, ChargingStationsDataGroup = ConnectSubGroups(PltParams=PltParams, 
                                                                                                NominalPlan=NominalPlan, 
                                                                                                NodesTrajectoryGroup=NodesTrajectoryGroup.copy(), 
                                                                                                NodesTrajectorySubGroup=NodesTrajectoryGroup_toAdd.copy(),
                                                                                                isplot = iplot>=3)


        NodesTrajectoryGroup = np.array(NodesTrajectoryGroup).reshape(-1,1)
        for i in range(ChargingStationsDataGroup.ChargingStationsNodes.shape[0]):
            ChargingTime[ChargingStationsDataGroup.ChargingStationsNodes[i]] = ChargingStationsDataGroup.ChargingTime[i]


        for i in range(len(NodesTrajectoryGroup)):
            NodesTrajectory[i,Ncar] = NodesTrajectoryGroup[i][0]
        Cost += CostGroup
if type(Cost) == np.ndarray:
    Cost = Cost[0]
    
# Calculate the Trajectory Time and Energy:
NumberOfCars = NodesTrajectory.shape[1]
print(SolverType+' Calculation Time = ', time.time()-t)
TimeVec = np.zeros((N+1,NumberOfCars))
Energy =np.zeros((N+1,NumberOfCars)); Energy[0,:] = InitialChargeStage
EnergySigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
TimeSigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
for m in range(NumberOfCars):
    i = 0
    while NodesTrajectory[i,m] > 0 or i==0:
        TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]
        if PltParams.RechargeModel == 'ConstantRate':
            Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i+1,m]]
            EnergySigma2[i+1,m] = EnergySigma2[i,m] + NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
            TimeSigma2[i+1,m] = TimeSigma2[i,m] + NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
        else:
            Energy[i+1,m] = NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + (a1 + (Energy[i,m]-a1)*np.exp(-a2*ChargingTime[NodesTrajectory[i,m]]))
        i += 1
EnergySigma = np.sqrt(EnergySigma2)
TimeSigma = np.sqrt(TimeSigma2)
UncertainEnergy = Energy - NominalPlan.EnergyAlpha*EnergySigma

while np.sum(NodesTrajectory[-2,:]) == 0:
    NodesTrajectory = NodesTrajectory[:-1,:]
    TimeVec = TimeVec[:-1,:]
    Energy = Energy[:-1,:]
    EnergySigma = EnergySigma[:-1,:]
    TimeSigma = TimeSigma[:-1,:]

print(SolverType+' Trajectory Time = ', np.sum(np.max(TimeVec+NominalPlan.TimeAlpha*TimeSigma, axis=0)))
for i in range(NumberOfCars):
    print('Car ', i, ' Trajectory: ', NodesTrajectory[:,i].T)


EnergyAlpha = norm.ppf(SolutionProbabilityEnergyReliability)
col_vec = ['m','y','b','r','g','c','k']
plt.figure()
plt.subplot(3,1,(1,2))
leg_str = []
legi = np.zeros((NumberOfCars,), dtype=object)
plt.plot(NodesPosition[0,0].T,NodesPosition[0,1].T,'o',linewidth=20, color='k')
for m in range(NumberOfCars):
    plt.plot(NodesPosition[NodesGroups[m][1:],0].T,NodesPosition[NodesGroups[m][1:],1].T,'o',linewidth=10, color=col_vec[m%len(col_vec)])
plt.grid('on')
plt.xlim((Xmin,Xmax))
plt.ylim((Ymin,Ymax))
if N<=10:
    for i in range(N):
        for j in range(i+1,N):
            plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
            plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
if N<=50:
    for i in range(N):
        colr = 'r' if i==0 else 'c'
        colr = 'k' if i in NominalPlan.ChargingStations else colr
        plt.text(NodesPosition[i,0]+1,NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    for i in range(len(NodesTrajectory)-1):
        j1 = NodesTrajectory[i,m]
        j2 = NodesTrajectory[i+1,m]
        if (ReturnToBase==True and j1 > 0) or (ReturnToBase==False and j2>0) or i==0:
            legi[m] = plt.arrow(NodesPosition[j1,0],NodesPosition[j1,1],NodesPosition[j2,0]-NodesPosition[j1,0],NodesPosition[j2,1]-NodesPosition[j1,1], width= 1, color=colr)
            # plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1]+4,"({:2.3},{:2.2})".format(NominalPlan.NodesTimeOfTravel[j1,j2], NominalPlan.NodesEnergyTravelSigma[j1,j2]), color='r', fontsize=10)
            # plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1]-4,"({:2.3},{:2.2})".format(NominalPlan.NodesEnergyTravel[j1,j2], NominalPlan.NodesEnergyTravelSigma[j1,j2]), color='b', fontsize=10)
    if np.max(NodesTrajectory[:,m])>0:
        indx = np.argwhere(NodesTrajectory[:,m] > 0)
        indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    else:
        indx = 2
    leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx-2))

plt.legend(legi,leg_str)
plt.title("Depot - Node 0, Charging Stations: "+str(NominalPlan.ChargingStations))
plt.subplot(3,1,3)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    if np.max(NodesTrajectory[:,m])>0:
        indx = np.argwhere(NodesTrajectory[:,m] > 0)
        indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    else:
        indx = 0
    plt.plot(Energy[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Nominal Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
    for i in range(indx):
        plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.grid('on')
plt.ylim((0,PltParams.BatteryCapacity))
plt.legend(leg_str)
plt.ylabel('Energy')
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    if np.max(NodesTrajectory[:,m])>0:
        indx = np.argwhere(NodesTrajectory[:,m] > 0)
        indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    else:
        indx = 0
    plt.plot(Energy[0:indx,m]-EnergyAlpha*EnergySigma[0:indx,m],'-.',color=colr)
    # plt.plot(EnergyExitingNodes[NodesTrajectory[:,m]],'x:',color='k')
    

for i in range(NominalPlan.NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(NodesTrajectory == j)
    if ChargingTime[j] > 0:
        colr = col_vec[indx[1][0]%len(col_vec)]
        if PltParams.RechargeModel == 'ConstantRate':
            Engery_i = ChargingTime[j]*NominalPlan.StationRechargePower
        else:
            Engery_i = a1 + (Energy[indx[0][0],indx[1][0]]-a1)*np.exp(-a2*ChargingTime[j])
        plt.arrow(indx[0][0],0,0,max(Engery_i,1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5+i*5,"{:.2f}".format(Engery_i), color=colr,fontsize=20)
plt.xlabel('Nodes')
FileName = SolverType+"_N"+str(N)+"_Cars"+str(NumberOfCars)+"_MaxNodesPerCar"+str(MaxNumberOfNodesPerCar)+"_MaxNodesToSolver"+str(MaxNodesToSolver)+"Method"+str(ClusteringMethod)+"_RechargeModel"+str(PltParams.RechargeModel)+'.png'
plt.savefig("Results//Nodes_"+FileName, dpi=300)

############################################################################################################
# Run Monte-Carlo Simulation for solution:
Nmc = 10000
FinalTime = np.zeros((Nmc,NumberOfCars))
MinEnergyLevel = np.zeros((Nmc,NumberOfCars))
for n in range(Nmc):
    TimeVec = np.zeros((NumberOfCars,1))
    for m in range(NumberOfCars):
        EnergyEntered = np.zeros((N+1,1)) + InitialChargeStage
        EnergyExited = np.zeros((N+1,1)) + InitialChargeStage
        for i in range(NodesTrajectory.shape[0]):
            TimeVec[m] += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]] + np.random.normal(0,1)*NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
            EnergyEntered[i+1] = EnergyExited[i] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + np.random.normal(0,1)*NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
            EnergyExited[i+1] = EnergyEntered[i+1] + ChargingTime[NodesTrajectory[i+1,m]]*NominalPlan.StationRechargePower
            EnergyExited[i+1] = min(EnergyExited[i+1],PltParams.BatteryCapacity)
            if i>0 and NodesTrajectory[i+1,m] == 0:
                break
        MinEnergyLevel[n,m] = np.min(EnergyEntered)
    FinalTime[n,:] = TimeVec[:,0]

leg_str1 = []
leg_str2 = []
for m in range(NumberOfCars):
    sorted = np.sort(FinalTime[:,m])
    indx = int(SolutionProbabilityTimeReliability*Nmc)
    print('Car '+str(m+1)+' '+str(SolutionProbabilityTimeReliability*100)+'% Lap Time: '+"{:.2f}".format(sorted[indx])+', '+str(95)+'% Lap Time: '+"{:.2f}".format(sorted[int(0.95*Nmc)]))
    print('Car '+str(m+1)+' Has a '+"{:.2f}".format(100*np.sum(MinEnergyLevel[:,m]<0)/Nmc)+'% chance of running out of energy')
    leg_str1.append('Car '+' '+str(SolutionProbabilityTimeReliability*100)+'% Lap Time: '+"{:.2f}".format(sorted[indx])+', '+str(95)+'% Lap Time: '+"{:.2f}".format(sorted[int(0.95*Nmc)]))
    leg_str2.append('Car '+str(m+1)+' Has a '+"{:.2f}".format(100*np.sum(MinEnergyLevel[:,m]<0)/Nmc)+'% chance of running out of energy')
plt.figure()
plt.subplot(2,1,1)
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    plt.plot(FinalTime[:,m],'s',color=colr)
plt.legend(leg_str1)
plt.grid('on')
plt.ylabel('Time')
plt.title(str(SolutionProbabilityTimeReliability*100)+'% Cost is '+"{:.2f}".format(Cost))
plt.subplot(2,1,2)
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    plt.plot(MinEnergyLevel[:,m],'s',color=colr)
plt.grid('on')
plt.ylabel('Min Energy')
plt.xlabel('Monte-Carlo Simulation')
plt.title(str(SolutionProbabilityEnergyReliability*100)+'% Energy Reliability')
plt.legend(leg_str2)
plt.savefig("Results//MC_"+FileName, dpi=200)

plt.show()

print("Finished")

