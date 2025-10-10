import random

# Objective (fitness) function: De Jong function
def fitness_function(position):
    x, y = position
    return x*2 + y*2  # minimize this function

# PSO parameters
num_particles = 10
num_iterations = 50
W = 0.3       # inertia weight (from PDF)
C1 = 2        # cognitive coefficient
C2 = 2        # social coefficient

# Initialize particles and velocities
particles = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(num_particles)]
velocities = [[0.0, 0.0] for _ in range(num_particles)]

# Initialize personal bests
pbest_positions = [p[:] for p in particles]
pbest_values = [fitness_function(p) for p in particles]

# Initialize global best
gbest_index = pbest_values.index(min(pbest_values))
gbest_position = pbest_positions[gbest_index][:]
gbest_value = pbest_values[gbest_index]

# PSO main loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()

        # Update velocity
        velocities[i][0] = (W * velocities[i][0] +
                            C1 * r1 * (pbest_positions[i][0] - particles[i][0]) +
                            C2 * r2 * (gbest_position[0] - particles[i][0]))
        velocities[i][1] = (W * velocities[i][1] +
                            C1 * r1 * (pbest_positions[i][1] - particles[i][1]) +
                            C2 * r2 * (gbest_position[1] - particles[i][1]))

        # Update position
        particles[i][0] += velocities[i][0]
        particles[i][1] += velocities[i][1]

        # Evaluate fitness
        current_value = fitness_function(particles[i])

        # Update personal best
        if current_value < pbest_values[i]:
            pbest_positions[i] = particles[i][:]
            pbest_values[i] = current_value

            # Update global best
            if current_value < gbest_value:
                gbest_value = current_value
                gbest_position = particles[i][:]

    print(f"Iteration {iteration+1}/{num_iterations} | Best Value: {gbest_value:.6f} at {gbest_position}")

print("\n✅ Optimal Solution Found:")
print(f"Best Position: {gbest_position}")
print(f"Minimum Value: {gbest_value}")
