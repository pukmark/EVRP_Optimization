import numpy as np
import SimDataTypes
from copy import deepcopy
import time
from multiprocessing import Pool, Lock, Value

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
    NodesTrajectory.append(i_CurrentNode)
    
    # Check if current node is has a optimal or feasible potential:
    Nodes2Go = list(set(range(NominalPlan.N)) - set(NodesTrajectory))
    if len(Nodes2Go) >= 1:
        EstMinTimeToGo = np.min(NominalPlan.NodesTimeOfTravel[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesTimeOfTravel[Nodes2Go,0])
        EstMinEnergyToGo = np.max(NominalPlan.NodesEnergyTravel[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.max(NominalPlan.NodesEnergyTravel[Nodes2Go,0])
        EstMinTimeToGoUncertainty = np.sqrt(np.min(NominalPlan.TravelSigma2[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.TravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.TravelSigma2[Nodes2Go,0]) + TourTimeUncertainty**2)
        EstMinEnergyToGoUncertainty = np.sqrt(np.min(NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(NominalPlan.NodesEnergyTravelSigma2[Nodes2Go,0]) + EnergyLeftUncertainty**2)        

        # Current Trajectory not feasible or not better than best plan:
        EstMaxChargingPotential = PltParams.BatteryCapacity*len(np.where(ChargingStationsData.EnergyEntered==0.0)[0])
        if (EnergyLeft + EstMinEnergyToGo + EstMaxChargingPotential + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EstMinEnergyToGoUncertainty < 0.0) or (EstMinTimeToGo + TourTime + NominalPlan.TimeAlpha*EstMinTimeToGoUncertainty >= SharedBestCost.value):
            Cost = np.inf
            return BestPlan, NodesTrajectory, Cost, ChargingStationsData

    # Update ChargingStationsData:
    if np.any(i_CurrentNode == NominalPlan.ChargingStations):
        ChargingStationsData.MaxChargingPotential += PltParams.BatteryCapacity - (EnergyLeft+ChargingStationsData.MaxChargingPotential)
        arg = np.argwhere(NominalPlan.ChargingStations == i_CurrentNode)
        ChargingStationsData.EnergyEntered[arg] = EnergyLeft
    
    # Chaeck if all nodes are visited:
    if (len(NodesTrajectory) == NominalPlan.N) and (NominalPlan.MustVisitAllNodes==True):
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
                    i_MaxEnergyToRecharge = PltParams.BatteryCapacity - ChargingStationsData.EnergyEntered[iCharge]
                    ChargingStationsData.EnergyExited[iCharge] = ChargingStationsData.EnergyEntered[iCharge] + min(i_MaxEnergyToRecharge, EnergyForRecharge)
                    ChargingStationsData.ChargingTime[iCharge] = (ChargingStationsData.EnergyExited[iCharge]-ChargingStationsData.EnergyEntered[iCharge])/NominalPlan.StationRechargePower
                    for ii in range(i+1,len(ChargingStationsData.ChargingStationsNodes)):
                            ChargingStationsData.EnergyEntered[ChargingNodeSquence[ii]] += ChargingStationsData.ChargingTime[iCharge]*NominalPlan.StationRechargePower
                    EnergyForRecharge -= min(i_MaxEnergyToRecharge, EnergyForRecharge)
            return BestPlan, NodesTrajectory, Cost, ChargingStationsData
    elif (len(NodesTrajectory) == NominalPlan.N) and (NominalPlan.MustVisitAllNodes==False):
        Cost = TourTime + NominalPlan.TimeAlpha*TourTimeUncertainty
        return BestPlan, NodesTrajectory, Cost, ChargingStationsData
             
    # Move To next node:
    # i_array = (NominalPlan.NodesTimeOfTravel[i_CurrentNode,:] + NominalPlan.TimeAlpha*NominalPlan.TravelSigma[i_CurrentNode,:]).argsort()
    i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
    for i in range(len(NodesTrajectory)):
        i_array = np.delete(i_array,np.where(i_array ==NodesTrajectory[i]))
    
    if i_CurrentNode == 0:
        t = time.time()
    for iNode in i_array:
        if np.any(np.array([iNode]) == NodesTrajectory): # Node already visited
            continue
        if StopProgram.value == False:
            if TimeLastSolutionFound.value > 0.0:  
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value and SharedBestCost.value<np.inf: # 2 hours without improvement
                    print("No Improvement for 5 Min. Stopping... Stopping at:", NodesTrajectory)
                    StopProgram.value = True
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value*10 and SharedBestCost.value==np.inf: # 2 hours without improvement
                    print("No Feasiable Solutions Found for 10 Min. Stopping...")
                    StopProgram.value = True
        else:
            break

        EnergyLeftNext = EnergyLeft + NominalPlan.NodesEnergyTravel[i_CurrentNode,iNode]
        EnergyLeftUncertaintyNext = np.sqrt(EnergyLeftUncertainty**2 + NominalPlan.NodesEnergyTravelSigma2[i_CurrentNode,iNode])
        TourTimeNext = TourTime + NominalPlan.NodesTimeOfTravel[i_CurrentNode,iNode]
        TourTimeUncertaintyNext = np.sqrt(TourTimeUncertainty**2 + NominalPlan.TravelSigma2[i_CurrentNode,iNode])
        if EnergyLeftNext + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*EnergyLeftUncertaintyNext < 0.0 or TourTimeNext + NominalPlan.TimeAlpha*TourTimeUncertaintyNext>=SharedBestCost.value:
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
                                                        BestPlan=BestPlan)
        if ((len(Cur_NodesTrajectory) == NominalPlan.N+1 and NominalPlan.MustVisitAllNodes==True) or (len(Cur_NodesTrajectory) == NominalPlan.N and NominalPlan.MustVisitAllNodes==False)) and (Cur_Cost < SharedBestCost.value):
            lock = Lock()
            BestPlan.Cost = Cur_Cost
            BestPlan.NodesTrajectory = Cur_NodesTrajectory
            BestPlan.ChargingStationsData = Cur_ChargingStationsData
            print('New Best Plan Found: ', BestPlan.NodesTrajectory, BestPlan.Cost)
            TimeLastSolutionFound.value = time.time()
            SharedBestCost.value = BestPlan.Cost

    # if len(NodesTrajectory) == 3 and StopProgram.value == False:
    #     NumberOfTrajExplored.value += 1
    #     Explored = NumberOfTrajExplored.value/((NominalPlan.N-2)*(NominalPlan.N-1))*100
    #     print("Done!",NodesTrajectory," Trajectories Explored[%]:", "{:.3}".format(Explored), "TimeLeft: ", "{:.3}".format((time.time()-StartTime.value)*(100-Explored)/Explored/60.0), "[min]")
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
    TimeLastSolutionFound.value = time.time()
    StartTime.value = time.time()
    NumberOfTrajExplored.value = 0
    StopProgram.value = False
    DeltaTimeToStop.value = MaxCalcTimeFromUpdate

    # append current node to trajectory:
    NodesTrajectory.append(i_CurrentNode)
    if NominalPlan.N <= 11:
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
    i_array = NominalPlan.NodesTimeOfTravel[i_CurrentNode,:].argsort()
    i_array = i_array[i_array != i_CurrentNode]

    Times = np.zeros(((NominalPlan.N-1)*(NominalPlan.N-2),1))
    indxes = np.zeros(((NominalPlan.N-1)*(NominalPlan.N-2),2), dtype=int)
    k=0
    for i in range(1,NominalPlan.N):
        for j in range(1,NominalPlan.N):
            if i==j:
                continue
            TourTime = NominalPlan.NodesTimeOfTravel[i_CurrentNode,i] + NominalPlan.NodesTimeOfTravel[i,j]
            TourTimeUncertainty = np.sqrt(NominalPlan.TravelSigma2[i_CurrentNode,i] + NominalPlan.TravelSigma2[i,j])
            Times[k] = TourTime + TourTimeUncertainty*NominalPlan.TimeAlpha
            indxes[k,:] = [i,j]
            k+=1
    indxes = indxes[np.argsort(Times[:,0]),:]

    for indx in indxes:
        i = indx[0]
        j = indx[1]
        TourTime = NominalPlan.NodesTimeOfTravel[0,i] + NominalPlan.NodesTimeOfTravel[i,j]
        TourTimeUncertainty = np.sqrt(NominalPlan.TravelSigma2[0,i] + NominalPlan.TravelSigma2[i,j])
        EnergyLeft = NominalPlan.InitialChargeStage + NominalPlan.NodesEnergyTravel[0,i] + NominalPlan.NodesEnergyTravel[i,j]
        EnergyLeftUncertainty = np.sqrt(NominalPlan.NodesEnergyTravelSigma2[0,i] + NominalPlan.NodesEnergyTravelSigma2[i,j])
        NodesTrajectory = [0,i]
        args.append((PltParams, NominalPlan, j, TourTime, TourTimeUncertainty, EnergyLeft, EnergyLeftUncertainty, deepcopy(ChargingStationsData), NodesTrajectory.copy(), deepcopy(BestPlan)))


    with Pool(16) as pool:
        results = pool.starmap(SolveRecursive_ChargingStations, args)
        # pool.close()
        # pool.join()

    Cost = np.inf
    for result in results:
        if result[2] < Cost and result[2] > 0.0:
            BestPlan = result[0]
            NodesTrajectory = result[1]
            Cost = result[2]
            ChargingStationsData = result[3]
    
    print('Final Best Plan Found: ', BestPlan.NodesTrajectory, BestPlan.Cost)
    
    return BestPlan, NodesTrajectory, Cost, ChargingStationsData
