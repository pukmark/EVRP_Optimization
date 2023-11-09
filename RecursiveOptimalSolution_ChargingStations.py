import numpy as np
import SimDataTypes
from copy import deepcopy
import time
from multiprocessing import Pool, Lock, Value
import itertools
import os
import math

SharedBestCost = Value('d', 1.0e6)
TimeLastSolutionFound = Value('d', 0.0)
NumberOfTrajExplored = Value('i', 0)
StartTime = Value('d', 0)
StopProgram = Value('b', 0)
DeltaTimeToStop = Value('d', 0)

def SolveRecursive_ChargingStations(PltParams: SimDataTypes.PlatformParams, 
                                    NominalPlan: SimDataTypes.NominalPlanning, 
                                    i_CurrentNode, 
                                    TourTime,
                                    TourTimeUncertainty,
                                    EnergyLeft,
                                    EnergyLeftUncertainty,
                                    ChargingStationsData: SimDataTypes.ChargingStations,
                                    NodesTrajectory, 
                                    BestPlan: SimDataTypes.BestPlan):

    # append current node to trajectory:
    if BestPlan.TimeStarted <= 1.0e-4 and StopProgram.value == False:
        # print("Starting Trajectory: ", [NodesTrajectory, i_CurrentNode])
        BestPlan.TimeStarted = time.time()
    if i_CurrentNode >= 0:
        NodesTrajectory.append(i_CurrentNode)
    else:
        i_CurrentNode = NodesTrajectory[-1]
    NumActiveNodes = NominalPlan.N - np.sum(ChargingStationsData.Active==False)
    NotActiveCS = []
    for i in range(len(ChargingStationsData.ChargingStationsNodes)):
        if ChargingStationsData.Active[i] == False:
            NotActiveCS.append(ChargingStationsData.ChargingStationsNodes[i])
    
    # Check if current node is has a optimal or feasible potential:
    Nodes2Go = list(set(range(NominalPlan.N)) - set(NodesTrajectory) - set(ChargingStationsData.ChargingStationsNodes))
    if len(Nodes2Go) >= 1 and i_CurrentNode >= 0:
        EstMinTimeToGo = np.min(NominalPlan.NodesTimeOfTravel[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,0])
        EstMinEnergyToGo = np.max(NominalPlan.NodesEnergyTravel[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,0])
        EstMinTimeToGoUncertainty = np.sqrt(np.min(NominalPlan.TravelSigma2[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.TravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.TravelSigma2[Nodes2Go,0]) + TourTimeUncertainty**2)
        EstMinEnergyToGoUncertainty = np.sqrt(np.min(NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,0]) + EnergyLeftUncertainty**2)        
        # Current Trajectory not feasible or not better than best plan:
        EstMaxChargingPotential = PltParams.BatteryCapacity*(len(np.where(ChargingStationsData.EnergyEntered==0.0)[0])-len(NotActiveCS))
        if (EnergyLeft + EstMinEnergyToGo + EstMaxChargingPotential + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EstMinEnergyToGoUncertainty < 0.0) or (EstMinTimeToGo + TourTime + NominalPlan.TimeAlpha*EstMinTimeToGoUncertainty >= SharedBestCost.value):
            Cost = np.inf
            return BestPlan, NodesTrajectory, Cost, ChargingStationsData

    # Update ChargingStationsData:
    if i_CurrentNode in NominalPlan.ChargingStations and ChargingStationsData.Active[np.argwhere(NominalPlan.ChargingStations == i_CurrentNode)] == True:
        ChargingStationsData.MaxChargingPotential += PltParams.BatteryCapacity - (EnergyLeft+ChargingStationsData.MaxChargingPotential)
        arg = np.argwhere(NominalPlan.ChargingStations == i_CurrentNode)
        ChargingStationsData.EnergyEntered[arg] = EnergyLeft
    
    # Chaeck if all nodes are visited:
    if (len(NodesTrajectory) == NumActiveNodes) and (NominalPlan.MustVisitAllNodes==True):
         # Go back to Depot if Energy Allows:
        EnergyLeftUncertaintyToDepot = np.sqrt(EnergyLeftUncertainty**2 + NominalPlan.NodesEnergyTravelSigma[i_CurrentNode,0]**2)
        EnergyLeftToDepot = EnergyLeft + NominalPlan.NodesEnergyTravel[i_CurrentNode,0]
        if EnergyLeftToDepot + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyToDepot < 0.0:
            Cost = np.inf
            return BestPlan, NodesTrajectory, Cost, ChargingStationsData
        else:
            NodesTrajectory.append(0)
            Cost = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,0] + NominalPlan.TimeAlpha*np.sqrt(TourTimeUncertainty**2+NominalPlan.TravelSigma[i_CurrentNode,0]**2)
            if EnergyLeftToDepot - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyToDepot < 0.0:
                EnergyForRecharge = -(EnergyLeftToDepot - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyToDepot)
                Cost += EnergyForRecharge/NominalPlan.StationRechargePower
                if Cost >= SharedBestCost.value:
                    Cost = np.inf
                    return BestPlan, NodesTrajectory, Cost, ChargingStationsData
                ChargingNodeSquence = np.argsort(ChargingStationsData.EnergyEntered)[::-1]
                for i in range(len(ChargingStationsData.ChargingStationsNodes)):
                    iCharge = ChargingNodeSquence[i]
                    if ChargingStationsData.Active[iCharge] == False:
                        continue
                    i_MaxEnergyToRecharge = PltParams.BatteryCapacity - ChargingStationsData.EnergyEntered[iCharge]
                    ChargingStationsData.EnergyExited[iCharge] = ChargingStationsData.EnergyEntered[iCharge] + min(i_MaxEnergyToRecharge, EnergyForRecharge)
                    ChargingStationsData.ChargingTime[iCharge] = (ChargingStationsData.EnergyExited[iCharge]-ChargingStationsData.EnergyEntered[iCharge])/NominalPlan.StationRechargePower
                    for ii in range(i+1,len(ChargingStationsData.ChargingStationsNodes)):
                        if ChargingStationsData.Active[ChargingNodeSquence[ii]] == False: 
                            continue
                        ChargingStationsData.EnergyEntered[ChargingNodeSquence[ii]] += ChargingStationsData.ChargingTime[iCharge]*NominalPlan.StationRechargePower
                    EnergyForRecharge -= min(i_MaxEnergyToRecharge, EnergyForRecharge)
            return BestPlan, NodesTrajectory, Cost, ChargingStationsData
    elif (len(NodesTrajectory) == NumActiveNodes) and (NominalPlan.MustVisitAllNodes==False):
        Cost = TourTime + NominalPlan.TimeAlpha*TourTimeUncertainty
        return BestPlan, NodesTrajectory, Cost, ChargingStationsData
             
    # Move To next node:
    i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
    for i in range(len(NodesTrajectory)):
        i_array = np.delete(i_array,np.where(i_array ==NodesTrajectory[i]))
    for i in range(len(NominalPlan.ChargingStations)):
        if ChargingStationsData.Active[i] == False:
            i_array = np.delete(i_array,np.where(i_array ==NominalPlan.ChargingStations[i]))
    
    # Start loop over all remaining nodes:
    for iNode in i_array:
        if np.any(np.array([iNode]) == NodesTrajectory): # Node already visited
            continue
        if StopProgram.value == False and (time.time()-BestPlan.TimeStarted<=DeltaTimeToStop.value*0.5):
            if TimeLastSolutionFound.value > 0.0:  
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value and SharedBestCost.value<np.inf: # 2 hours without improvement
                    print("No Improvement for "+str(DeltaTimeToStop.value)+" Sec. Stopping... Stopping at:", NodesTrajectory)
                    StopProgram.value = True
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value*10 and SharedBestCost.value==np.inf: # 2 hours without improvement
                    print("No Feasiable Solutions Found for "+str(10*DeltaTimeToStop.value)+" Sec. Stopping...")
                    StopProgram.value = True
        else:
            BestPlan.StopProgram = True
            break

        if i_array[0] in NominalPlan.ChargingStations:
            ChargingStationsData.Active[np.argwhere(NominalPlan.ChargingStations == i_array[0])] = False
            NumActiveNodes = NominalPlan.N - np.sum(ChargingStationsData.Active==False)
            BestPlan, Cur_NodesTrajectory, Cur_Cost, Cur_ChargingStationsData = SolveRecursive_ChargingStations(PltParams=PltParams,
                                                            NominalPlan= NominalPlan, 
                                                            i_CurrentNode= -1, 
                                                            TourTime= TourTime,
                                                            TourTimeUncertainty= TourTimeUncertainty,
                                                            EnergyLeft= EnergyLeft,
                                                            EnergyLeftUncertainty= EnergyLeftUncertainty,
                                                            ChargingStationsData= deepcopy(ChargingStationsData),
                                                            NodesTrajectory= NodesTrajectory.copy(), 
                                                            BestPlan=deepcopy(BestPlan))
            
            if ((len(Cur_NodesTrajectory) == NumActiveNodes+1 and NominalPlan.MustVisitAllNodes==True) or (len(Cur_NodesTrajectory) == NumActiveNodes and NominalPlan.MustVisitAllNodes==False)) and (Cur_Cost < SharedBestCost.value-1e-4):
                BestPlan.Cost = Cur_Cost
                BestPlan.NodesTrajectory = Cur_NodesTrajectory
                BestPlan.ChargingStationsData = Cur_ChargingStationsData
                print('New Best Plan Found: ', [NominalPlan.NodesRealNames[i] for i in BestPlan.NodesTrajectory], BestPlan.Cost)
                BestPlan.TimeStarted = time.time()
                TimeLastSolutionFound.value = time.time()
                SharedBestCost.value = BestPlan.Cost
            ChargingStationsData.Active[np.argwhere(NominalPlan.ChargingStations == i_array[0])] = True



        EnergyLeftNext = EnergyLeft + NominalPlan.NodesEnergyTravel[i_CurrentNode,iNode]
        EnergyLeftUncertaintyNext = np.sqrt(EnergyLeftUncertainty**2 + NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,iNode])
        TourTimeNext = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,iNode]
        TourTimeUncertaintyNext = np.sqrt(TourTimeUncertainty**2 + NominalPlan.TravelSigma2[i_CurrentNode,iNode])
        if (EnergyLeftNext + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyNext < 0.0) or (TourTimeNext + NominalPlan.TimeAlpha*TourTimeUncertaintyNext>=SharedBestCost.value-1e-4):
            continue
        
        BestPlan, Cur_NodesTrajectory, Cur_Cost, Cur_ChargingStationsData = SolveRecursive_ChargingStations(PltParams=PltParams,
                                                        NominalPlan= NominalPlan, 
                                                        i_CurrentNode= iNode, 
                                                        TourTime= TourTimeNext,
                                                        TourTimeUncertainty= TourTimeUncertaintyNext,
                                                        EnergyLeft= EnergyLeftNext,
                                                        EnergyLeftUncertainty= EnergyLeftUncertaintyNext,
                                                        ChargingStationsData= deepcopy(ChargingStationsData),
                                                        NodesTrajectory= NodesTrajectory.copy(), 
                                                        BestPlan=deepcopy(BestPlan))
        if ((len(Cur_NodesTrajectory) == NumActiveNodes+1 and NominalPlan.MustVisitAllNodes==True) or (len(Cur_NodesTrajectory) == NumActiveNodes and NominalPlan.MustVisitAllNodes==False)) and (Cur_Cost < SharedBestCost.value):
            BestPlan.Cost = Cur_Cost
            BestPlan.NodesTrajectory = Cur_NodesTrajectory
            BestPlan.ChargingStationsData = Cur_ChargingStationsData
            print('New Best Plan Found: ', [NominalPlan.NodesRealNames[i] for i in BestPlan.NodesTrajectory], BestPlan.Cost)
            TimeLastSolutionFound.value = time.time()
            SharedBestCost.value = BestPlan.Cost
            BestPlan.TimeStarted = time.time()
    return BestPlan, BestPlan.NodesTrajectory, BestPlan.Cost, BestPlan.ChargingStationsData

def SolveParallelRecursive_ChargingStations(PltParams: SimDataTypes.PlatformParams, 
                                    NominalPlan: SimDataTypes.NominalPlanning, 
                                    i_CurrentNode, 
                                    TourTime,
                                    TourTimeUncertainty,
                                    EnergyLeft,
                                    EnergyLeftUncertainty,
                                    ChargingStationsData: SimDataTypes.ChargingStations,
                                    NodesTrajectory, 
                                    BestPlan: SimDataTypes.BestPlan,
                                    MaxCalcTimeFromUpdate: float = 60.0):
    
    SharedBestCost.value = 1.0e6
    StartTime.value = time.time()
    NumberOfTrajExplored.value = 0
    StopProgram.value = False
    DeltaTimeToStop.value = MaxCalcTimeFromUpdate
    cpus = os.cpu_count()
    SingleCoreN = 9

    # append current node to trajectory:
    if NominalPlan.N <= SingleCoreN:
        TimeLastSolutionFound.value = time.time()
        BestPlan, NodesTrajectorySubGroup, CostSubGroup, ChargingStationsDataSubGroup = SolveRecursive_ChargingStations(PltParams=PltParams,
                                                        NominalPlan= NominalPlan,
                                                        i_CurrentNode = i_CurrentNode, 
                                                        TourTime = TourTime,
                                                        TourTimeUncertainty = TourTimeUncertainty,
                                                        EnergyLeft = EnergyLeft,
                                                        EnergyLeftUncertainty = EnergyLeftUncertainty,
                                                        ChargingStationsData = SimDataTypes.ChargingStations(NominalPlan.ChargingStations),
                                                        NodesTrajectory = [], 
                                                        BestPlan = SimDataTypes.BestPlan(NominalPlan.N, NominalPlan.ChargingStations, time.time()))
        return BestPlan, NodesTrajectorySubGroup, CostSubGroup, ChargingStationsDataSubGroup
    args = []
    # i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
    # i_array = i_array[i_array != i_CurrentNode]
    Nodes = set(range(NominalPlan.N))-set(NodesTrajectory)-set([i_CurrentNode])
    i = 0
    while math.factorial(NominalPlan.N-1)/math.factorial(SingleCoreN+i-1)>1000:
        i+=1
    premuts = list(itertools.permutations(Nodes,NominalPlan.N-SingleCoreN-i))
    NumberOfIters = len(premuts)
    Times = np.zeros((NumberOfIters,))
    indxes = np.zeros((NumberOfIters,NominalPlan.N-SingleCoreN), dtype=int)
    
    k=0
    for premut in premuts:
        TourTimeNext = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,premut[0]]
        TourTimeUncertainty2Next = TourTimeUncertainty**2 + NominalPlan.TravelSigma2[i_CurrentNode,premut[0]]
        for i in range(len(premut)-1):
            TourTimeNext += NominalPlan.NodesTimeOfTravel[premut[i],premut[i+1]]
            TourTimeUncertainty2Next = TourTimeUncertainty2Next+NominalPlan.TravelSigma2[premut[i],premut[i+1]]
        Times[k] = TourTimeNext + np.sqrt(TourTimeUncertainty2Next)*NominalPlan.TimeAlpha
        k+=1
    indxes = np.argsort(Times)

    for indx in indxes:
        premut = premuts[indx]
        NodesTrajectoryNext = NodesTrajectory.copy()
        NodesTrajectoryNext.append(i_CurrentNode)
        TourTimeNext = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,premut[0]]
        TourTimeUncertainty2Next = TourTimeUncertainty**2 + NominalPlan.TravelSigma2[i_CurrentNode,premut[0]]
        EnergyLeftNext = EnergyLeft + NominalPlan.NodesEnergyTravel[i_CurrentNode,premut[0]]
        EnergyLeftUncertainty2Next = EnergyLeftUncertainty**2 + NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,premut[0]]
        for i in range(len(premut)-1):
            NodesTrajectoryNext.append(premuts[indx][i])
            TourTimeNext += NominalPlan.NodesTimeOfTravel[premut[i],premut[i+1]]
            TourTimeUncertainty2Next += NominalPlan.TravelSigma2[premut[i],premut[i+1]]
            EnergyLeftNext += NominalPlan.NodesEnergyTravel[premut[i],premut[i+1]]
            EnergyLeftUncertainty2Next += NominalPlan.NodesEnergyTravelSigma2[premut[i],premut[i+1]]
        args.append((PltParams, NominalPlan, premut[-1], TourTimeNext, np.sqrt(TourTimeUncertainty2Next), EnergyLeftNext, np.sqrt(EnergyLeftUncertainty2Next), deepcopy(ChargingStationsData), NodesTrajectoryNext.copy(), deepcopy(BestPlan)))

    TimeLastSolutionFound.value = time.time()
    with Pool(cpus) as pool:
        results = pool.starmap(SolveRecursive_ChargingStations, args)
        # pool.close()
        # pool.join()

    Cost = np.inf
    for i in range(len(results)):
        result = results[i]
        if result[2] < Cost and result[2] > 0.0:
            NodesTrajectory = result[1]
            BestPlan = result[0]
            Cost = result[2]
            ChargingStationsData = result[3]
    
    return BestPlan, NodesTrajectory, Cost, ChargingStationsData
