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

def CalcRechargeProfile(NominalPlan: SimDataTypes.NominalPlanning, 
                        ChargingStationsData: SimDataTypes.ChargingStations,
                        NodesTrajectory: list,
                        BatteryCapacity: float,
                        EnergyForRecharge: float):
    RechargeTime = 0.0
    iCS_order = []
    ChargingStationsData.EnergyExited = ChargingStationsData.EnergyEntered.copy()
    for iNode in NodesTrajectory:
        if not iNode in NominalPlan.ChargingStations:
            continue
        else:
            iCS_order.append(iNode)
    for iCS in range(1,len(iCS_order)):
        if EnergyForRecharge <= 0.0:
            break
        iCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[iCS])[0][0]
        while ChargingStationsData.EnergyEntered[iCS_index] < 0.0:
            ChargeRate = np.zeros((iCS,))
            for jCS in range(iCS):
                jCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[jCS])[0][0]
                iProfile = np.argwhere(NominalPlan.ChargingProfile[jCS_index][:,0] <= max(0.0,ChargingStationsData.EnergyExited[jCS_index]))[-1][0]
                ChargeRate[jCS]= NominalPlan.ChargingRateProfile[jCS_index][iProfile,1]
            iMaxChargeRate = np.argmax(ChargeRate)
            kCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[iMaxChargeRate])[0][0]
            iProfile = np.argwhere(NominalPlan.ChargingProfile[kCS_index][:,0] <= max(0.0,ChargingStationsData.EnergyExited[jCS_index]))[-1][0]
            dEnergy = np.min([BatteryCapacity-ChargingStationsData.EnergyExited[kCS_index], NominalPlan.ChargingProfile[kCS_index][iProfile+1,0]-NominalPlan.ChargingProfile[kCS_index][iProfile,0], EnergyForRecharge, -ChargingStationsData.EnergyEntered[iCS_index]])
            EnergyForRecharge -= dEnergy
            RechargeTime += dEnergy/NominalPlan.ChargingRateProfile[kCS_index][iProfile,1]
            ChargingStationsData.ChargingTime[kCS_index] += dEnergy/NominalPlan.ChargingRateProfile[kCS_index][iProfile,1]
            ChargingStationsData.EnergyExited[kCS_index] += dEnergy
            kStartIndex = np.argwhere(iCS_order[iMaxChargeRate] == np.array(iCS_order))[0][0]
            for kCS in range(kStartIndex+1, len(iCS_order)):
                kCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[kCS])[0][0]
                ChargingStationsData.EnergyEntered[kCS_index] += dEnergy
                ChargingStationsData.EnergyExited[kCS_index] += dEnergy

    while EnergyForRecharge > 0.0:
        ChargeRate = np.zeros((len(iCS_order),))
        for jCS in range(len(iCS_order)):
            jCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[jCS])[0][0]
            iProfile = np.argwhere(NominalPlan.ChargingProfile[jCS_index][:,0] <= max(0.0,ChargingStationsData.EnergyExited[jCS_index]))[-1][0]
            ChargeRate[jCS]= NominalPlan.ChargingRateProfile[jCS_index][iProfile,1]
        iMaxChargeRate = np.argmax(ChargeRate)
        iCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[iMaxChargeRate])[0][0]
        iProfile = np.argwhere(NominalPlan.ChargingProfile[iCS_index][:,0] <= max(0.0,ChargingStationsData.EnergyExited[jCS_index]))[-1][0]
        dEnergy = np.min([BatteryCapacity-ChargingStationsData.EnergyExited[iCS_index], NominalPlan.ChargingProfile[iCS_index][iProfile+1,0]-NominalPlan.ChargingProfile[iCS_index][iProfile,0], EnergyForRecharge])
        ChargingStationsData.EnergyExited[iCS_index] += dEnergy
        EnergyForRecharge -= dEnergy
        RechargeTime += dEnergy/NominalPlan.ChargingRateProfile[iCS_index][iProfile,1]
        ChargingStationsData.ChargingTime[iCS_index] += dEnergy/NominalPlan.ChargingRateProfile[iCS_index][iProfile,1]
        kStartIndex = np.argwhere(iCS_order[iMaxChargeRate] == np.array(iCS_order))[0][0]
        for kCS in range(kStartIndex+1,len(iCS_order)):
            kCS_index = np.argwhere(NominalPlan.ChargingStations == iCS_order[kCS])[0][0]
            ChargingStationsData.EnergyEntered[kCS_index] += dEnergy
            ChargingStationsData.EnergyExited[kCS_index] += dEnergy

    return RechargeTime, ChargingStationsData

def SolveRecursive_ChargingStations(PltParams: SimDataTypes.PlatformParams, 
                                    NominalPlan: SimDataTypes.NominalPlanning, 
                                    i_CurrentNode, 
                                    TourTime,
                                    TourTimeUncertainty,
                                    EnergyLeft,
                                    EnergyLeftUncertainty,
                                    ChargingStationsData: SimDataTypes.ChargingStations,
                                    NodesTrajectory, 
                                    BestPlan: SimDataTypes.BestPlan,):

    # append current node to trajectory:
    if BestPlan.TimeStarted <= 1.0e-4:
        # print("Starting Trajectory: ",[NodesTrajectory, i_CurrentNode])
        BestPlan.TimeStarted = time.time()
    if i_CurrentNode >= 0:
        NodesTrajectory.append(i_CurrentNode)
    else:
        i_CurrentNode = NodesTrajectory[-1]
    NumActiveNodes = NominalPlan.N - np.sum(ChargingStationsData.Active==False)

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
                RechargeTime, ChargingStationsData = CalcRechargeProfile(NominalPlan, ChargingStationsData, NodesTrajectory, PltParams.BatteryCapacity, EnergyForRecharge)
                Cost += RechargeTime

            return BestPlan, NodesTrajectory, Cost, ChargingStationsData
    elif (len(NodesTrajectory) == NumActiveNodes) and (NominalPlan.MustVisitAllNodes==False):
        Cost = TourTime + NominalPlan.TimeAlpha*TourTimeUncertainty
        return BestPlan, NodesTrajectory, Cost, ChargingStationsData
             
    # Move To next node:
    i_array = []
    # first Do all nodes in the groups of current node:
    for iGroup in range(len(NominalPlan.SubGroups)):
        if i_CurrentNode in NominalPlan.SubGroups[iGroup]:
            i_array = list(set(NominalPlan.SubGroups[iGroup]) - set(NodesTrajectory))
            break

    if i_array == []:
        i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
        for i in range(len(NodesTrajectory)):
            i_array = np.delete(i_array,np.where(i_array ==NodesTrajectory[i]))
        for i in range(len(NominalPlan.ChargingStations)):
            if ChargingStationsData.Active[i] == False:
                i_array = np.delete(i_array,np.where(i_array ==NominalPlan.ChargingStations[i]))
    
    NotActiveCS = ChargingStationsData.ChargingStationsNodes[ChargingStationsData.Active==False]

    if len(i_array) > 5:
        i_array = i_array[0:4]

    # Start loop over all remaining nodes:
    for iNode in i_array:
        if np.any(np.array([iNode]) == NodesTrajectory): # Node already visited
            continue
        if StopProgram.value == False and (time.time()-BestPlan.TimeStarted<=DeltaTimeToStop.value*0.75):
            if TimeLastSolutionFound.value > 0.0:  
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value and SharedBestCost.value<np.inf: # 2 hours without improvement
                    StopProgram.value = True
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value*10 and SharedBestCost.value==np.inf: # 2 hours without improvement
                    print("No Feasiable Solutions Found for "+str(10*DeltaTimeToStop.value)+" Sec. Stopping...")
                    StopProgram.value = True
        else:
            BestPlan.StopProgram = True
            break
        # Check if current node is a charging station and check ignoring it:
        if iNode in NominalPlan.ChargingStations:
            ChargingStationsData.Active[np.argwhere(NominalPlan.ChargingStations == iNode)] = False
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
                # print('New Best Plan Found: ', [NominalPlan.NodesRealNames[i] for i in BestPlan.NodesTrajectory], BestPlan.Cost)
                BestPlan.TimeStarted = time.time()
                TimeLastSolutionFound.value = time.time()
                SharedBestCost.value = BestPlan.Cost
            ChargingStationsData.Active[np.argwhere(NominalPlan.ChargingStations == iNode)] = True



        EnergyLeftNext = EnergyLeft + NominalPlan.NodesEnergyTravel[i_CurrentNode,iNode]
        EnergyLeftUncertaintyNext = np.sqrt(EnergyLeftUncertainty**2 + NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,iNode])
        TourTimeNext = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,iNode]
        TourTimeUncertaintyNext = np.sqrt(TourTimeUncertainty**2 + NominalPlan.TravelSigma2[i_CurrentNode,iNode])
        if (EnergyLeftNext + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyNext < 0.0) or (TourTimeNext + np.min(NominalPlan.NodesTimeOfTravel[i_array,0]) + NominalPlan.TimeAlpha*TourTimeUncertaintyNext>=SharedBestCost.value-1e-8):
            continue

        # # Check if current node is has a optimal or feasible potential:
        Nodes2Go = list(set(range(NominalPlan.N)) - set(NodesTrajectory) - set([iNode]) - set(ChargingStationsData.ChargingStationsNodes))
        if len(Nodes2Go) >= 1:
            EstMinTimeToGo = np.min(NominalPlan.NodesTimeOfTravel[iNode,Nodes2Go]) + np.sum(np.sort(np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,0])
            EstMinEnergyToGo = np.max(NominalPlan.NodesEnergyTravel[iNode,Nodes2Go]) + np.sum(np.sort(np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,0])
            EstMinTimeToGoUncertainty = np.sqrt(np.min(NominalPlan.TravelSigma2[iNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.TravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.TravelSigma2[Nodes2Go,0]) + TourTimeUncertaintyNext**2)
            EstMinEnergyToGoUncertainty = np.sqrt(np.min(NominalPlan.NodesEnergyTravelSigma2[iNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,0]) + EnergyLeftUncertaintyNext**2)        
            # Current Trajectory not feasible or not better than best plan:
            EstMaxChargingPotential = PltParams.BatteryCapacity*(len(np.where(ChargingStationsData.EnergyEntered==0.0)[0])-len(NotActiveCS))
            if (EnergyLeftNext + EstMinEnergyToGo + EstMaxChargingPotential + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EstMinEnergyToGoUncertainty < 0.0) or (EstMinTimeToGo + TourTimeNext + NominalPlan.TimeAlpha*EstMinTimeToGoUncertainty >= SharedBestCost.value):
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
            # print('New Best Plan Found: ', [NominalPlan.NodesRealNames[i] for i in BestPlan.NodesTrajectory], BestPlan.Cost)
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
    SingleCoreN = 10

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
                                                        ChargingStationsData = deepcopy(ChargingStationsData),
                                                        NodesTrajectory = NodesTrajectory.copy(), 
                                                        BestPlan = SimDataTypes.BestPlan(NominalPlan.N, NominalPlan.ChargingStations, time.time()))
        if StopProgram.value == True:
            print("No Improvement for "+str(DeltaTimeToStop.value)+" Sec. Stopping... Stopping with Cost:", SharedBestCost.value)
        return BestPlan, NodesTrajectorySubGroup, CostSubGroup, ChargingStationsDataSubGroup
    args = []
    # i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
    # i_array = i_array[i_array != i_CurrentNode]
    Nodes = set(range(NominalPlan.N))-set(NodesTrajectory)-set([i_CurrentNode])
    i = 0
    while math.factorial(NominalPlan.N-1)/math.factorial(SingleCoreN+i-1)>2000:
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
    indxes = np.argsort(Times)[0:int(len(Times))]
    for indx in indxes[:int(len(indxes)/2)]:
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

    # args = args[0:cpus]
    TimeLastSolutionFound.value = time.time()
    with Pool(cpus) as pool:
        results = pool.starmap(SolveRecursive_ChargingStations, args, chunksize=1)
        # pool.close()
        # pool.join()

    if StopProgram.value == True:
        print("No Improvement for "+str(DeltaTimeToStop.value)+" Sec. Stopping... Stopping with Cost:", SharedBestCost.value)

    Cost = np.inf
    for i in range(len(results)):
        result = results[i]
        if result[2] < Cost and result[2] > 0.0:
            NodesTrajectory = result[1]
            BestPlan = result[0]
            Cost = result[2]
            ChargingStationsData = result[3]
    
    return BestPlan, NodesTrajectory, Cost, ChargingStationsData
