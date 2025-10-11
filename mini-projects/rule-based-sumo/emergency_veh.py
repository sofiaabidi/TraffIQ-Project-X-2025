import os 
import sys  

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

Sumo_config = [
    'sumo-gui',
    '-c', 'Test1.sumocfg',
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

traci.start(Sumo_config)


desired_phase_mapping = {
    'Node2_EW': 0,  
    'Node2_NS': 2, 
    'Node5_EW': 0,  
    'Node5_NS': 2,  
    
}

adjusted_tls = {} 
step = 0


def get_emergency_vehicle_direction(vehicle_id):
    current_edge = traci.vehicle.getRoadID(vehicle_id).lower()
    print(f"Vehicle {vehicle_id} is on edge {current_edge}")
    if 'nb' in current_edge or 'sb' in current_edge:
        return 'NS'
    elif 'eb' in current_edge or 'wb' in current_edge:
        return 'EW'
    else:
        return None

def process_emergency_vehicles(desired_phase_mapping, adjusted_tls, step):
    # Identify emergency vehicles
    emergency_vehicles = [
        veh for veh in traci.vehicle.getIDList()
        if traci.vehicle.getTypeID(veh)=="emergency"
    ]
    active_tls = set()

    for veh in emergency_vehicles:
        direction = get_emergency_vehicle_direction(veh)
        if direction:
            next_tls = traci.vehicle.getNextTLS(veh)
            print(f"next_tls for {veh}: {next_tls}")

            if next_tls:
                tls_info = next_tls[0]
                tlsID, linkIndex, distance, state = tls_info
                tl_key = f"{tlsID}_{direction}"
                desired_phase = desired_phase_mapping.get(tl_key)

                if desired_phase is not None:
                    current_phase = traci.trafficlight.getPhase(tlsID)
                    print(f"TLS {tlsID}, Current phase: {current_phase}, "
                          f"Desired phase: {desired_phase}")
                    active_tls.add(tlsID)

                    if tlsID not in adjusted_tls or adjusted_tls[tlsID] != desired_phase:
                        adjusted_tls[tlsID] = desired_phase  
                        if current_phase == desired_phase:
                            new_duration = max(20, traci.trafficlight.getPhaseDuration(tlsID) + 10)
                            traci.trafficlight.setPhaseDuration(tlsID, new_duration)
                            print(f"Extended phase {current_phase} of {tlsID} to "
                                  f"{new_duration} seconds")
                        else:
                            traci.trafficlight.setPhaseDuration(tlsID, 0.1)
                            print(f"Shortened phase {current_phase} of {tlsID} to 1 second")

    for tlsID in list(adjusted_tls.keys()):
        if tlsID not in active_tls:
            del adjusted_tls[tlsID]
            print(f"Resetting traffic light {tlsID} to normal operation.")

    step += 1
    return step

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()  
    step=process_emergency_vehicles(desired_phase_mapping, adjusted_tls, step)

traci.close()