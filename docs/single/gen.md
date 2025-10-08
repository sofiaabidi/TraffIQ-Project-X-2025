# 🚦TLS Optimization using Genetic Algorithm

The optimization minimizes the **average vehicle waiting time** and **queue length**, while maximizing **throughput**.


## What is a Genetic Algorithm (GA)?

A **Genetic Algorithm** is an optimization method inspired by **natural evolution**.  
It works by evolving a population of possible solutions over several generations to find the best one.  

### GA Core Steps:
1. **Initialization**–Generate a random population of solutions.
2. **Evaluation**–Measure fitness (performance) of each solution.
3. **Selection**–Choose the best-performing individuals.
4. **Crossover**–Combine parts of two solutions to create offspring.
5. **Mutation**–Randomly alter some parts to maintain diversity.
6. **Iteration**–Repeat until reaching a desired number of generations or convergence.

---

## Project Files

| File | Description |
|------|--------------|
| `gen.net.xml` | Road network for the intersection |
| `gen.add.xml` | Additional network elements (e.g., detectors) |
| `gen.rou.xml` | Vehicle route definitions |
| `gen.py` | Main optimization script |

---

## Algorithm Summary

### Parameters
```python
POP_SIZE = 8              # Number of solutions per generation
N_GENERATIONS = 10        # Number of generations
MUTATION_RATE = 0.2       # Probability of mutation
CROSSOVER_RATE = 0.7      # Probability of crossover
GREEN_MIN, GREEN_MAX = 10, 60
YELLOW_MIN, YELLOW_MAX = 2, 5
SIM_STEPS = 2000          # Number of simulation steps per run
```

### Fitness Function
The **fitness** evaluates how effective a given signal plan is:

```python
fitness = avg_wait + 0.5 * queue_penalty - 0.2 * throughput_bonus
```

- **avg_wait:** average vehicle waiting time (lower is better)  
- **queue_penalty:** total halting vehicles over time  
- **throughput_bonus:** number of vehicles that successfully left the network  

The **goal** is to **minimize** the fitness value.

---

## Code Structure

### Run SUMO Simulation
```python
def run_simulation(phase_durations):
    # Launch SUMO with the specified network and route files
    traci.start([...])
    # Modify traffic light program phases
    # Run the simulation for SIM_STEPS and record stats
    ...
    traci.close()
    return fitness
```

### Initialize Population
```python
def init_population(num_phases, phase_types):
    # Create random green/yellow durations
    return [[random.randint(GREEN_MIN, GREEN_MAX) if t == "green"
             else random.randint(YELLOW_MIN, YELLOW_MAX) for t in phase_types]
            for _ in range(POP_SIZE)]
```

### Crossover & Mutation
```python
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]

def mutate(ind, phase_types):
    for i, t in enumerate(phase_types):
        if random.random() < MUTATION_RATE:
            ind[i] = random.randint(GREEN_MIN, GREEN_MAX) if t == "green" else random.randint(YELLOW_MIN, YELLOW_MAX)
    return ind
```

### Main GA Loop
```python
population = init_population(len(phases), phase_types)
best_solution, best_score = None, float("inf")

for gen in range(N_GENERATIONS):
    fitness = [run_simulation(ind) for ind in population]
    # Select, crossover, and mutate
    ...
    print(f"Generation {gen+1}: Best Fitness = {best_score:.2f}")
```

### Visualization
At the end of optimization, results are visualized using Matplotlib:
```python
plt.plot(range(1, N_GENERATIONS+1), best_scores, marker='o')
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("GA Optimization Progress")
plt.grid(True)
plt.show()
```

## Dependencies
Install required modules:
```bash
pip install traci matplotlib
```
Make sure SUMO is installed and available in your system path.
---

