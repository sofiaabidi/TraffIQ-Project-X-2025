## Fitness Function

```python
avg_wait = total_wait / vehicle_count
    queue_penalty = total_queue / SIM_STEPS
    throughput_bonus = throughput / SIM_STEPS

    fitness = avg_wait + 0.5 * queue_penalty - 0.2 * throughput_bonus
    return fitness
```
## Mutation