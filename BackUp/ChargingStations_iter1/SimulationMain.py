import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from RecursiveOptimalSolution import *
from RecursiveOptimalSolution_ChargingStations import *
from CVX_Solution import *
from CVX_Solution_ChargingStations import *
import os
import time
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(5)

# Define Simulation Parameters:
##############################$
# Number of Nodes
N = 10
NumberOfChargeStations = 2
MaxPriority = 3
MaxMissionTime = 120000
ReturnToBase = False
MustVisitAllNodes = True
MaxPriority = 1 # Max Priority of a node Priority = 1,...,MaxPriority
SimType = 'ChargeStations' # 'ChargingStations' or 'SelfRecharge'
SolverTypes = ['CVX'] # 'Recursive' or 'CVX'

# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5

# Map Size
Xmin, Xmax = -100, 100
Ymin, Ymax = -100, 100

# Platform Parameters:
PltParams = DataTypes.PlatformParams()
PltParams.Vmax = 10 # Platform Max Speed
PltParams.MinVelReductionCoef, PltParams.MaxVelReductionCoef = 0.0, 0.75 # min/max speed reduction factor for node2node travel
PltParams.VelEnergyConsumptionCoef = 0.02 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
PltParams.VelConstPowerConsumption = 0.04
## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask = 2, 10
PltParams.MinTimePerTask, PltParams.MaxTimePerTask = 1, 5
PltParams.RechargePowerPerDay = 5
PltParams.BatteryCapacity = 80

## Randomize The Nodes Locations:
NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])


# Set the Nomianl Time of Travel between any 2 nodes as the distance between 
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)

for i in range(N):
    NominalPlan.NodesVelocity[i,i+1:] = np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
    NominalPlan.NodesVelocity[i+1:,i] = NominalPlan.NodesVelocity[i,i+1:].T
    for j in range(i,N):
        if i==j: continue
        NominalPlan.NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
        NominalPlan.NodesDistance[j,i] = NominalPlan.NodesDistance[i,j]
        NominalPlan.NodesTimeOfTravel[i,j] = NominalPlan.NodesDistance[i,j] / NominalPlan.NodesVelocity[i,j]
        NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
        NominalPlan.NodesEnergyTravel[i,j] = NominalPlan.NodesTimeOfTravel[i,j] * (PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i,j]**2)
        NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]

## Calculate Nominal Time to spend and Energy Consumption for task in Node i
NominalPlan.NodesTaskTime = np.random.uniform(PltParams.MinTimePerTask, PltParams.MaxTimePerTask, size=(N,1))
NominalPlan.NodesTaskPower = np.random.uniform(PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask, size=(N,1))

## Nodes Task Prioroties:
NominalPlan.NodesPriorities = np.ceil(np.random.uniform(0,MaxPriority,size=(N,1)))
NominalPlan.N = N

## Charging Stations:
NominalPlan.ChargingStations = np.floor(np.random.uniform(0,N,size=(NumberOfChargeStations,1)))
while len(np.unique(NominalPlan.ChargingStations)) < NumberOfChargeStations:
    NominalPlan.ChargingStations = np.floor(np.random.uniform(0,N,size=(NumberOfChargeStations,1)))
NominalPlan.NRechargeLevels = 5
NominalPlan.StationRechargePower = 3
## Nominal Solution to the problem:
#
#       max sum(A*Priority_i^2 * Task_i - B*Time_ij)
#
InitialChargeStage = 0.75* PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
for SolverType in SolverTypes:
    BestPlan = DataTypes.BestPlan(N)
    if SolverType == 'CVX':
        t = time.time()
        NodesTrajectory, Cost, PowerLeft, ChargingTime = SolveCVX_ChargingStations(PltParams=PltParams,
                                                                NominalPlan= NominalPlan,
                                                                NodesWorkDone = np.zeros((N,1), dtype=int),
                                                                TimeLeft= MaxMissionTime,
                                                                PowerLeft= InitialChargeStage,
                                                                i_CurrentNode= 0,
                                                                NodesTrajectory= [],
                                                                NodesWorkSequence= [],
                                                                Cost= 0.0)
        print('CVX Calculation Time = ', time.time()-t)
        BestPlan.NodesTrajectory = NodesTrajectory
        BestPlan.PowerLeft = np.append(PowerLeft,BestPlan.PowerLeft[-1] - NominalPlan.NodesEnergyTravel[NodesTrajectory[-2],0] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[-2]])
        Time = 0.0
        Energy =np.zeros(len(NodesTrajectory),); Energy[0] = InitialChargeStage
        for i in range(len(NodesTrajectory)-1):
            Time += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i],NodesTrajectory[i+1]]+ChargingTime[NodesTrajectory[i]]
            Energy[i+1] = Energy[i] - NominalPlan.NodesEnergyTravel[NodesTrajectory[i],NodesTrajectory[i+1]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i]]
        print('CVX Time = ', Time)
    elif SolverType == 'Recursive':
        BestPlan = DataTypes.BestPlan(N)
        t = time.time()
        if SimType == 'SelfCharge':
            BestPlan, NodesTrajectory, NodesWorkDone, Cost = SolveRecursive(PltParams=PltParams,
                                                                    NominalPlan= NominalPlan, 
                                                                    NodesWorkDone = np.zeros((N,1), dtype=int), 
                                                                    TimeLeft= MaxMissionTime, 
                                                                    PowerLeft= InitialChargeStage, 
                                                                    i_CurrentNode= 0, 
                                                                    NodesTrajectory= [], 
                                                                    NodesWorkSequence= [],
                                                                    Cost= 0.0,
                                                                    BestPlan=BestPlan)
        elif SimType == 'ChargeStations':
            BestPlan, NodesTrajectory, NodesWorkDone, Cost = SolveRecursive_ChargingStations(PltParams=PltParams,
                                                                    NominalPlan= NominalPlan, 
                                                                    NodesWorkDone = np.zeros((N,1), dtype=int), 
                                                                    TimeLeft= MaxMissionTime, 
                                                                    PowerLeft= InitialChargeStage, 
                                                                    i_CurrentNode= 0, 
                                                                    NodesTrajectory= [], 
                                                                    NodesWorkSequence= [],
                                                                    Cost= 0.0,
                                                                    BestPlan=BestPlan,
                                                                    PowerLeftVec=[])
            NodesTrajectory = BestPlan.NodesTrajectory
            ChargingTime = np.zeros((N,))
            for i in range(len(BestPlan.PowerLeft)-1):
                if BestPlan.PowerLeft[i+1]-BestPlan.PowerLeft[i]-1e-2 > -NominalPlan.NodesEnergyTravel[NodesTrajectory[i],NodesTrajectory[i+1]]:
                    ChargingTime[NodesTrajectory[i]] = (BestPlan.PowerLeft[i+1]-BestPlan.PowerLeft[i] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i],NodesTrajectory[i+1]])/NominalPlan.StationRechargePower
        print('Recursive Calculation Time = ', time.time()-t)
        Time = 0.0
        Energy =np.zeros(N+1,); Energy[0] = InitialChargeStage
        for i in range(N):
            Time += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i],NodesTrajectory[i+1]]+ChargingTime[NodesTrajectory[i]]
            Energy[i+1] = Energy[i] - NominalPlan.NodesEnergyTravel[NodesTrajectory[i],NodesTrajectory[i+1]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i]]
        print('Recursive Time = ', Time)
    else:
        print('Solver Type is not defined')

    print('Best Traj = ',NodesTrajectory)

    col = ['m','y','b','r']
    plt.figure()
    plt.subplot(3,1,(1,2))
    for i in range(MaxPriority):
        PriorIndx_i = np.where(NominalPlan.NodesPriorities[:,0] == i+1)
        plt.plot(NodesPosition[PriorIndx_i,0].T,NodesPosition[PriorIndx_i,1].T,'o',linewidth=10, color=col[i])
    plt.grid('on')
    plt.xlim((Xmin,Xmax))
    plt.ylim((Ymin,Ymax))
    plt.legend(['Priority 1','Priority 2','Priority 3'])
    if N<=10:
        for i in range(N):
            for j in range(i+1,N):
                plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
                plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
                plt.text(0.5*NodesPosition[i,0]+0.5*NodesPosition[j,0], 1+0.5*NodesPosition[i,1]+0.5*NodesPosition[j,1],"{:.3}".format(NominalPlan.NodesTimeOfTravel[i,j]), color='r', fontsize=10)
    for i in range(N):
        plt.text(NodesPosition[i,0]+1,NodesPosition[i,1]+1,"{:}".format(i), color='c',fontsize=30)
    for i in range(len(NodesTrajectory)-1):
        j1 = NodesTrajectory[i]
        j2 = NodesTrajectory[i+1]
        plt.arrow(NodesPosition[j1,0],NodesPosition[j1,1],NodesPosition[j2,0]-NodesPosition[j1,0],NodesPosition[j2,1]-NodesPosition[j1,1], width= 0.5, color='g')
        plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1],"{:.3}".format(NominalPlan.NodesTimeOfTravel[j1,j2]), color='r', fontsize=10)
    plt.title(SolverType)
    plt.subplot(3,1,3)
    plt.plot(Energy,'o-')
    plt.grid('on')
    a_strings = ["%3.i" % x for x in BestPlan.NodesTrajectory]
    plt.xticks(ticks=np.arange(len(BestPlan.NodesTrajectory)),labels=a_strings)
    for i in range(NumberOfChargeStations):
        j = NominalPlan.ChargingStations[i]
        indx = np.where(BestPlan.NodesTrajectory == j)
        if ChargingTime[int(j)] > 0:
            plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)]*NominalPlan.StationRechargePower,1.0), color='g', width= 0.1)

plt.show()







