#ifndef EVRP_H
#define EVRP_H

#define CHAR_LEN 100
using namespace std;

struct ScenarioType {
  int N;
  int NumberOfDepots;
  int NumberOfCars;
  int* CarsInDepots;
  double** NodesPosition;
  double** NodesTimeOfTravel;
  double** NodesEnergyTravel;
  double** NodesEnergyTravelSigma;
  double** NodesEnergyTravelSigma2;
  double** TravelSigma;
  double** TravelSigma2;
  int* NodesRealNames;
  double* LoadDemand;
  int* ChargingStations;
  int NumberOfChargingStations;
  double LoadCapacity;
  double BatteryCapacity;
  double OptimalValue;
  double PhiTime;
  double PhiEnergy;
};


ScenarioType Read_EVRP_Instance();


#endif