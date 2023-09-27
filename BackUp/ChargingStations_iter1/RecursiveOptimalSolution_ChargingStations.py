import numpy as np
import SimDataTypes



def SolveRecursive_ChargingStations(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost, BestPlan, PowerLeftVec):
    # Current Trajectory not feasible:
    if PowerLeft < 0 or TimeLeft < 0:
        Cost = -np.inf
        return BestPlan, NodesTrajectory, NodesWorkDone, Cost
    # append current node to trajectory:
    NodesWorkDone[i_CurrentNode] = 1
    NodesTrajectory.append(i_CurrentNode)
    PowerLeftVec.append(PowerLeft)
    # Check if charging is possible at current node:
    NRechargeLevels = NominalPlan.NRechargeLevels if np.any(i_CurrentNode == NominalPlan.ChargingStations) else 1
    for iRecharge in range(0,NRechargeLevels): #
        WantedChargeLevel = iRecharge/(NominalPlan.NRechargeLevels-1)*PltParams.BatteryCapacity
        if WantedChargeLevel <= PowerLeft and iRecharge>0: # No need to recharge. First Interation is without recharge
            continue
        else:
            # Recharge Time:
            TimeToRecharge = 0.0 if iRecharge==0 else (WantedChargeLevel-PowerLeft)/NominalPlan.StationRechargePower
            if (np.sum(NodesWorkDone) == NominalPlan.NodesPriorities.shape[0] and NominalPlan.MustVisitAllNodes) and NominalPlan.ReturnToBase==True:
                 if (PowerLeft+TimeToRecharge*NominalPlan.StationRechargePower-NominalPlan.NodesEnergyTravel[i_CurrentNode,0]>0.0) and (TimeLeft-TimeToRecharge-NominalPlan.NodesTimeOfTravel[i_CurrentNode,0]>0):
                      Cost -= NominalPlan.TimeCoefInCost*(TimeToRecharge+NominalPlan.NodesTimeOfTravel[i_CurrentNode,0])
                      PowerLeftVec.append(PowerLeft)
                      PowerLeftVec[-1] = PowerLeftVec[-1] + TimeToRecharge*NominalPlan.StationRechargePower-NominalPlan.NodesEnergyTravel[i_CurrentNode,0]
                      NodesTrajectory.append(0)
                      if Cost > BestPlan.Cost:
                        BestPlan.NodesTrajectory = NodesTrajectory.copy()
                        BestPlan.Cost = Cost
                        BestPlan.PowerLeft = PowerLeftVec
                        print('BestPlan.Cost = ', BestPlan.Cost, 'BestPlan.NodesTrajectory = ', BestPlan.NodesTrajectory)
                      return BestPlan, NodesTrajectory, NodesWorkDone, Cost
                 else:
                      if iRecharge<NRechargeLevels-1:
                           continue
                      Cost = -np.inf
                      return BestPlan, NodesTrajectory, NodesWorkDone, Cost
            elif (np.sum(NodesWorkDone) == NominalPlan.NodesPriorities.shape[0] and NominalPlan.MustVisitAllNodes) and NominalPlan.ReturnToBase==False:
                 if Cost > BestPlan.Cost:
                        BestPlan.NodesTrajectory = NodesTrajectory.copy()
                        BestPlan.Cost = Cost
                        BestPlan.PowerLeft = PowerLeftVec
                        # print('BestPlan.Cost = ', BestPlan.Cost, 'BestPlan.NodesTrajectory = ', BestPlan.NodesTrajectory)
                 return BestPlan, NodesTrajectory, NodesWorkDone, Cost
            # Move To next node:
            for iNode in range(1,NominalPlan.NodesPriorities.shape[0]):
                if NodesWorkDone[iNode] == 1 or i_CurrentNode == iNode: # Node already visited
                    continue
                BestPlan, Cur_NodesTrajectory, Cur_NodesWorkDone, Cur_Cost = SolveRecursive_ChargingStations(PltParams=PltParams,
                                                                NominalPlan= NominalPlan, 
                                                                NodesWorkDone= NodesWorkDone.copy(), 
                                                                TimeLeft= TimeLeft - NominalPlan.NodesTimeOfTravel[i_CurrentNode,iNode] - TimeToRecharge, 
                                                                PowerLeft= PowerLeft + TimeToRecharge*NominalPlan.StationRechargePower - NominalPlan.NodesEnergyTravel[i_CurrentNode,iNode],
                                                                i_CurrentNode= iNode, 
                                                                NodesTrajectory= NodesTrajectory.copy(), 
                                                                NodesWorkSequence= NodesWorkSequence.copy(),
                                                                Cost= Cost - NominalPlan.TimeCoefInCost*(TimeToRecharge+NominalPlan.NodesTimeOfTravel[i_CurrentNode,iNode]),
                                                                BestPlan=BestPlan,
                                                                PowerLeftVec=PowerLeftVec.copy())
                if ((np.sum(Cur_NodesWorkDone) == NominalPlan.NodesPriorities.shape[0] and NominalPlan.MustVisitAllNodes==True) or (NominalPlan.MustVisitAllNodes==False)) and (Cur_Cost > BestPlan.Cost):
                        BestPlan.Cost = Cur_Cost
                        BestPlan.NodesTrajectory = Cur_NodesTrajectory
            # finish loop on all nodes
    return BestPlan, BestPlan.NodesTrajectory, NodesWorkDone, BestPlan.Cost
        