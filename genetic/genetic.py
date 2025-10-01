import traci
import random
import matplotlib.pyplot as plt
NET_FILE="gen.net.xml"
ADD_FILE="gen.add.xml"
ROUTE_FILE="gen.rou.xml"  
SUMO_BINARY="sumo"         # or "sumo-gui"
TLS_ID="Node2"             # target traffic light

POP_SIZE=8
N_GENERATIONS=10
MUTATION_RATE=0.2
CROSSOVER_RATE=0.7
GREEN_MIN,GREEN_MAX=10,60
YELLOW_MIN,YELLOW_MAX=2,5
SIM_STEPS=2000  # steps per simulation

#fitness function
def run_simulation(phase_durations):
    sumo_cmd = [SUMO_BINARY, "-n", NET_FILE, "-a", ADD_FILE, "-r", ROUTE_FILE,
                "--no-step-log", "true", "--duration-log.disable", "true"]
    traci.start(sumo_cmd)

    program = traci.trafficlight.getCompleteRedYellowGreenDefinition(TLS_ID)[0]
    new_phases = []
    for i, p in enumerate(program.phases):
        new_phases.append(traci.trafficlight.Phase(phase_durations[i], p.state))
    new_logic = traci.trafficlight.Logic(program.programID, program.type, program.currentPhaseIndex, new_phases)
    traci.trafficlight.setCompleteRedYellowGreenDefinition(TLS_ID, new_logic)

    step = 0
    total_wait, vehicle_count, total_queue, throughput = 0, 0, 0, 0
    while step < SIM_STEPS:
        traci.simulationStep()
        detectors = traci.lanearea.getIDList()
        for det in detectors:
            total_wait += traci.lanearea.getLastStepHaltingNumber(det)
            vehicle_count += traci.lanearea.getLastStepVehicleNumber(det)
            total_queue += traci.lanearea.getLastStepHaltingNumber(det)
        throughput += traci.simulation.getArrivedNumber()
        step += 1
    traci.close()

    if vehicle_count == 0:
        return 99999

    avg_wait = total_wait / vehicle_count
    queue_penalty = total_queue / SIM_STEPS
    throughput_bonus = throughput / SIM_STEPS

    fitness = avg_wait + 0.5 * queue_penalty - 0.2 * throughput_bonus
    return fitness


def init_population(num_phases, phase_types):
    population=[]
    for _ in range(POP_SIZE):
        ind=[]
        for i, p_type in enumerate(phase_types):
            if p_type=="green":
                ind.append(random.randint(GREEN_MIN, GREEN_MAX))
            else:
                ind.append(random.randint(YELLOW_MIN, YELLOW_MAX))
        population.append(ind)
    return population


def crossover(p1,p2):
    if random.random()<CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]


def mutate(ind, phase_types):
    for i,p_type in enumerate(phase_types):
        if random.random() < MUTATION_RATE:
            if p_type=="green":
                ind[i]=random.randint(GREEN_MIN, GREEN_MAX)
            else:
                ind[i]=random.randint(YELLOW_MIN, YELLOW_MAX)
    return ind


if __name__ == "__main__":
    traci.start([SUMO_BINARY, "-n", NET_FILE, "-a", ADD_FILE, "-r", ROUTE_FILE,
                 "--no-step-log", "true", "--duration-log.disable", "true"])
    phases=traci.trafficlight.getCompleteRedYellowGreenDefinition(TLS_ID)[0].phases
    phase_types=["yellow" if "y" in p.state else "green" for p in phases]
    traci.close()
    print("Phase types:", phase_types)

    population = init_population(len(phases), phase_types)
    best_solution,best_score=None,float("inf")
    best_scores = []
    for gen in range(N_GENERATIONS):
        print(f"\n--- Generation {gen + 1} ---")
        fitness = []
        for ind in population:
            score = run_simulation(ind)
            fitness.append(score)
            print(f"Phases {ind} -> Fitness {score:.2f}")

            if score < best_score:
                best_score, best_solution = score, ind[:]
        
        best_scores.append(best_score)
        # Select best half
        sorted_pop = [x for _, x in sorted(zip(fitness, population), key=lambda z: z[0])]
        population = sorted_pop[:POP_SIZE // 2]

        # Reproduce
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2)
            new_pop.extend([mutate(c1, phase_types), mutate(c2, phase_types)])
        population = new_pop[:POP_SIZE]

    print("\nOPTIMIZATION FINISHED")
    print(f"Best TLS Plan: {best_solution}")
    print(f"Best Fitness: {best_score:.2f}")

    plt.plot(range(1, N_GENERATIONS+1), best_scores, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Optimization Progress")
    plt.grid(True)
    plt.show()

