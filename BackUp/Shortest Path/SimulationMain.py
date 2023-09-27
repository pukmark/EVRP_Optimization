import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from RecursiveOptimalSolution import *
from CVX_Solution import *
import os
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(5)

# Define Simulation Parameters:
##############################$
# Number of Nodes
N = 20
NumberOfCars = 3
MaxNumberOfNodesPerCar = int(N*0.75) if NumberOfCars > 1 else N
MaxPriority = 3
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
MaxPriority = 1 # Max Priority of a node Priority = 1,...,MaxPriority
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
PltParams.VelEnergyConsumptionCoef = 1 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
PltParams.VelConstPowerConsumption = 0.25
## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask = 2, 10
PltParams.MinTimePerTask, PltParams.MaxTimePerTask = 1, 5
PltParams.RechargePowerPerDay = 5
PltParams.BatteryCapacity = 50000

## Randomize The Nodes Locations:
NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])


# Set the Nomianl Time of Travel between any 2 nodes as the distance between 
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)

# NominalPlan.NodesVelocity = np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(N,N))
for i in range(N):
    NominalPlan.NodesVelocity[i,i+1:] = np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
    NominalPlan.NodesVelocity[i+1:,i] = NominalPlan.NodesVelocity[i,i+1:].T
    for j in range(i,N):
        if i==j: continue
        NominalPlan.NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
        NominalPlan.NodesDistance[j,i] = NominalPlan.NodesDistance[i,j]
        NominalPlan.NodesTimeOfTravel[i,j] = NominalPlan.NodesDistance[i,j] / NominalPlan.NodesVelocity[i,j]
        NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]

## Calculate Nominal Time to spend and Energy Consumption for task in Node i
NominalPlan.NodesTaskTime = np.random.uniform(PltParams.MinTimePerTask, PltParams.MaxTimePerTask, size=(N,1))
NominalPlan.NodesTaskPower = np.random.uniform(PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask, size=(N,1))

## Nodes Task Prioroties:
NominalPlan.NodesPriorities = np.ceil(np.random.uniform(0,MaxPriority,size=(N,1)))
NominalPlan.N = N

## Nominal Solution to the problem:
#
#       max sum(A*Priority_i^2 * Task_i - B*Time_ij)
#
InitialChargeStage = 0.75* PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
NominalPlan.NumberOfCars = NumberOfCars
NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
BestPlan = DataTypes.BestPlan(N)
for SolverType in SolverTypes:
    if SolverType == 'CVX':
        NodesTrajectory, Cost, TimeLeft = SolveCVX(PltParams=PltParams,
                                                    NominalPlan= NominalPlan,
                                                    NodesWorkDone = np.zeros((N,1), dtype=int),
                                                    TimeLeft= MaxMissionTime,
                                                    PowerLeft= InitialChargeStage,
                                                    i_CurrentNode= 0,
                                                    NodesTrajectory= [],
                                                    NodesWorkSequence= [],
                                                    Cost= 0.0)
        TimeVec = np.zeros((N,NumberOfCars))
        for m in range(NumberOfCars):
            
            TimeVec[1,m] = NominalPlan.NodesTimeOfTravel[NodesTrajectory[0,m],NodesTrajectory[1,m]]
            i = 1
            while NodesTrajectory[i,m] > 0:
                TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
                i += 1

        print('CVX Time = ', np.sum(np.max(TimeVec, axis=0)))
    elif SolverType == 'Recursive':
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
        Time = 0.0
        for i in range(len(NodesTrajectory)-1):
            # print('Node ', NodesTrajectory[i], ' -> ', NodesTrajectory[i+1], ' Time = ', NominalPlan.NodesTimeOfTravel[NodesTrajectory[i],NodesTrajectory[i+1]])
            Time += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i],NodesTrajectory[i+1]]
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
                plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.1)
                plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.1)
                plt.text(0.5*NodesPosition[i,0]+0.5*NodesPosition[j,0], 1+0.5*NodesPosition[i,1]+0.5*NodesPosition[j,1],"{:.3}".format(NominalPlan.NodesTimeOfTravel[i,j]), color='r', fontsize=10)
    for i in range(N):
        colr = 'r' if i==0 else 'c'
        plt.text(NodesPosition[i,0]+1,NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
    for m in range(NumberOfCars):
        colr = col[m] if m < len(col) else 'k'
        for i in range(len(NodesTrajectory)-1):
            j1 = NodesTrajectory[i,m]
            j2 = NodesTrajectory[i+1,m]
            if (ReturnToBase==True and j1 > 0) or (ReturnToBase==False and j2>0) or i==0:
                plt.arrow(NodesPosition[j1,0],NodesPosition[j1,1],NodesPosition[j2,0]-NodesPosition[j1,0],NodesPosition[j2,1]-NodesPosition[j1,1], width= 1, color=colr)
    plt.title(SolverType)
    plt.subplot(3,1,3)
    plt.plot(TimeVec,'o-')
    plt.grid('on')
    # a_strings = ["%3.i" % x for x in NodesTrajectory]
    # plt.xticks(ticks=np.arange(len(NodesTrajectory)),labels=a_strings)
    plt.xlabel('Node Number [-]')
    plt.ylabel('Time [s]')

plt.show()







