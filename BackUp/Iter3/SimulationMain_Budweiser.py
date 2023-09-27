import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
import csv
from RecursiveOptimalSolution import *
from RecursiveOptimalSolution_ChargingStations import *
from CVX_Solution import *
from CVX_Solution_ChargingStations import *
from Pyomo_Solution_ChargingStations import *
from Gurobi_Solution_ChargingStations import *
import os
import time
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(10)

# Define Simulation Parameters:
##############################$
# Number of Nodes
NumberOfCars = 5
NumberOfChargeStations = 0
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
SolverTypes = ['CVX'] # 'Recursive' or 'CVX'

# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5


# Load Nodes Database (budweiser)
Dir = './/Bay_Field//'
filename = "Bay_Field_2022-09-24-20-00"
filetypes = ['d','tt','ec','ec_based_on_soc']
Data = dict()
for filetype in filetypes:
    with open(Dir+filename+"_"+filetype+".csv", "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        Data[filetype] = np.zeros((len(csvreader.fieldnames)-1,len(csvreader.fieldnames)-1))
        i=0
        for row in csvreader:
            temp = row
            for j in range(len(csvreader.fieldnames)-1):
                Data[filetype][i,j] = temp[str(j)]
            i += 1
N = len(Data['d'])
MaxNumberOfNodesPerCar = int(N*0.75) if NumberOfCars > 1 else N

# Platform Parameters:
PltParams = DataTypes.PlatformParams()
PltParams.BatteryCapacity = 400.0
PltParams.MinimalSOC = 0
PltParams.RechargeModel = 'ExponentialRate' # 'ExponentialRate' or 'ConstantRate'
PltParams.FullRechargeRateFactor = 0.5

# Set the Nomianl Time of Travel between any 2 nodes as the distance between 
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)
NominalPlan.NodesTimeOfTravel = Data['tt']
NominalPlan.NodesEnergyTravel = Data['ec']
NominalPlan.NodesEnergyTravelBasedOnSOC = Data['ec_based_on_soc']
NominalPlan.NodesDistance = Data['d']
NominalPlan.N = N

NominalPlan.NodesEnergyTravel = NominalPlan.NodesEnergyTravelBasedOnSOC

## Nominal Solution to the problem:
#
#       max sum(Time_ij)
#
InitialChargeStage = 0.75* PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
NominalPlan.NumberOfCars = NumberOfCars
NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
NominalPlan.ChargingStations = np.zeros((1,1), dtype=int)
NominalPlan.StationRechargePower = 0.0
NominalPlan.FullRechargeRateFactor = 0.5


t = time.time()
BestPlan = DataTypes.BestPlan(N)
NodesTrajectory, Cost, PowerLeft, ChargingTime = SolveGurobi_ChargingStations_MinMax(PltParams=PltParams,
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
BestPlan.PowerLeft = np.append(PowerLeft,BestPlan.PowerLeft[-1] + NominalPlan.NodesEnergyTravel[NodesTrajectory[-2],0] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[-2]])
TimeVec = np.zeros((N,NumberOfCars))
Energy =np.zeros((N,NumberOfCars)); Energy[0,:] = InitialChargeStage
for m in range(NumberOfCars):
    i = 0
    while NodesTrajectory[i,m] > 0 or i==0:
        TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]
        Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i,m]]
        i += 1
print('CVX Time = ', np.sum(np.max(TimeVec, axis=0)))
    

print('Best Traj = ',NodesTrajectory.T)
print('Best Cost = ',Cost)

col_vec = ['m','y','b','r','g','c','k']
plt.figure()
plt.subplot(2,1,1)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx))
    for i in range(indx):
        plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]-0.3,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.grid('on')
# plt.ylim((0,PltParams.BatteryCapacity))
plt.legend(leg_str)
plt.ylabel('Energy')
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)]*NominalPlan.StationRechargePower,1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]*NominalPlan.StationRechargePower), color=colr, fontsize=20)

plt.subplot(2,1,2)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(TimeVec[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
    for i in range(indx):
        plt.text(i,TimeVec[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.grid('on')
plt.ylabel('Time [sec]')
plt.legend(leg_str)
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)],1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]), color=colr, fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='k',fontsize=10)

plt.show()