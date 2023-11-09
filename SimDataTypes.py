import numpy as np

class PlatformParams():
    def __init__(self):
        self.Vmax = 0
        self.MinVelReductionCoef = 0
        self.MaxVelReductionCoef = 0
        self.VelEnergyConsumptionCoef = 0
        self.VelConstPowerConsumption = 0
        self.MinPowerConsumptionPerTask = 0
        self.MaxPowerConsumptionPerTask = 0
        self.MinTimePerTask = 0
        self.MaxTimePerTask = 0
        self.RechargePowerPerDay = 0
        self.BatteryCapacity = 0
        self.LoadCapacity = 0
        self.TimeCoefInCost = 0


## Nomianl Planning Class
class NominalPlanning():
    def __init__(self,N):
        self.N = 0
        self.NumberOfDepots = 1
        self.NumberOfCars = []
        self.CarsInDepots = []
        self.NodesPosition = np.zeros((N,2))
        self.NodesVelocity = np.zeros((N,N))
        self.NodesDistance = np.zeros((N,N))
        self.NodesTimeOfTravel = np.zeros((N,N))
        self.NodesEnergyTravel = np.zeros((N,N))
        self.NodesEnergyTravelSigma = np.zeros((N,N))
        self.NodesEnergyTravelSigma2 = np.zeros((N,N))
        self.TravelSigma = np.zeros((N,N))
        self.TravelSigma2 = np.zeros((N,N))
        self.NodesTaskTime = np.zeros((N,1))
        self.NodesTaskPower = np.zeros((N,1))
        self.NodesPriorities = np.zeros((N,1))
        self.NodesRealNames = np.zeros((N,1))
        self.LoadDemand = np.zeros((N,), dtype=int)

class ChargingStations():
    def __init__(self, ChargingStationsNodes):
        self.ChargingStationsNodes = ChargingStationsNodes
        self.EnergyEntered = np.zeros((ChargingStationsNodes.shape[0],))
        self.EnergyExited = np.zeros((ChargingStationsNodes.shape[0],))
        self.ChargingTime = np.zeros((ChargingStationsNodes.shape[0],))
        self.Active = np.ones((ChargingStationsNodes.shape[0],), dtype=bool)
        self.MaxChargingPotential = 0.0

class BestPlan():
    def __init__(self,N, ChargingStationsNodes, t = 0.0):
        self.Cost = np.inf
        self.NodesTrajectory = np.zeros((N,1), dtype=int)
        self.ChargingStationsData = ChargingStations(ChargingStationsNodes=ChargingStationsNodes)
        self.TimeStarted = 0
        self.StopProgram = False
