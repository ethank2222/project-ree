import os
import sys
import traci

CONFIG_PATH_4_WAY = os.path.join("simulations", "Easy_4_Way", "map.sumocfg")
# 4-Way Intersection light
TL_ID = "TCenter"

# 1. Setup SUMO environment variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Need to update env-vars in your computer, please lmk when you do it so I can help - Erick <3
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run_simulation():
    """Main control loop"""
    # Start SUMO as a subprocess
    if not os.path.exists(CONFIG_PATH_4_WAY):
        print(f"Error: Could not find config file at {CONFIG_PATH_4_WAY}")
        return

    # 'sumo-gui' to see visuals + text
    # 'sumo'     to see text-only
    traci.start(["sumo-gui", "-c", CONFIG_PATH_4_WAY])

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Logic: Every 200 steps, toggle light
        if step % 200 == 0:
            current_phase = traci.trafficlight.getPhase(TL_ID)
            # Switch between phase 0 (North-South Green) and 2 (East-West Green)
            # 1 & 4 are respective Yellows
            new_phase = 2 if current_phase == 0 else 0
            traci.trafficlight.setPhase(TL_ID, new_phase)
            print(f"Step {step}: Switching traffic light to phase {new_phase}")

        step += 1

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    run_simulation()