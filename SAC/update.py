def pso_optimize(env, model, vecnorm, trajectories,
                 n_iterations=30, swarm_size=30,
                 bounds=(0.5, 1.5),
                 w=0.7, c1=1.4, c2=1.4,
                 seed=0,
                 min_l1_std=0.3):
    """
    Simple PSO to optimize scale parameters (3D).
    Returns mean, std (with L1 constraint), best point, best score.
    """
    rng = np.random.RandomState(seed)

    dim = 3
    low, high = bounds

    particles = rng.uniform(low, high, size=(swarm_size, dim))
    velocities = rng.normal(0, 0.1*(high-low), size=(swarm_size, dim))
    pbest_pos = particles.copy()
    pbest_score = np.full(swarm_size, np.inf)

    def objective(scale):
        scale = np.clip(scale, low, high)
        disc = [step_discrepancy(traj, env, model, vecnorm, scale) for traj in trajectories]
        return float(np.mean(disc))

    # inizializza pbest
    for i in range(swarm_size):
        pbest_score[i] = objective(particles[i])

    gbest_idx = np.argmin(pbest_score)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx].copy()

    # PSO iterations
    for it in range(n_iterations):
        r1 = rng.rand(swarm_size, dim)
        r2 = rng.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - particles)
                      + c2 * r2 * (gbest_pos - particles))
        particles = particles + velocities
        particles = np.clip(particles, low, high)

        scores = np.array([objective(p) for p in particles], dtype=np.float64)
        improved = scores < pbest_score
        pbest_pos[improved] = particles[improved]
        pbest_score[improved] = scores[improved]

        cur_gbest_idx = np.argmin(pbest_score)
        cur_gbest_score = pbest_score[cur_gbest_idx]
        cur_gbest_pos = pbest_pos[cur_gbest_idx].copy()
        if cur_gbest_score < gbest_score:
            gbest_score = cur_gbest_score
            gbest_pos = cur_gbest_pos.copy()

        print(f"[Iter {it+1}/{n_iterations}] gbest_score={gbest_score:.6f}, gbest_pos={gbest_pos}")

    # calcola mean e std
    final_mean = particles.mean(axis=0)
    final_std = particles.std(axis=0, ddof=0)

    # impone vincolo sulla norma L1 della std
    l1_norm = np.sum(final_std)
    if l1_norm < min_l1_std:
        scale_factor = min_l1_std / l1_norm
        final_std = final_std * scale_factor

    return final_mean, final_std, gbest_pos, gbest_score


def cma_optimize_entropy(env, model, vecnorm, trajectories,
                         n_generations=20, population_size=10,
                         entropy_threshold=1.0):
    
    def objective(scale):
        scale = np.clip(scale, 0.5, 1.5)
        disc = [step_discrepancy(traj, env, model, vecnorm, scale) for traj in trajectories]
        return np.mean(disc)

    x0 = np.ones(3)
    sigma0 = 0.5
    bounds = [0.5*np.ones(3), 1.5*np.ones(3)]
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': population_size, 'bounds': bounds})

    best_solution = None
    best_fitness = float("inf")

    for gen in range(n_generations):
        solutions = es.ask()
        fitnesses = [objective(sol) for sol in solutions]
        es.tell(solutions, fitnesses)
        
        # Control entropy: reduce sigma if necessary
        cov = es.sigma**2 * np.diag(es.C)
        entropy = 0.5 * np.sum(np.log(2*np.pi*np.e*cov))
        if entropy < entropy_threshold:
            # Increase sigma to push entropy up
            factor = np.sqrt(entropy_threshold / max(entropy, 1e-12))
            es.sigma *= factor
            print(f"[Gen {gen+1}] Entropy too low ({entropy:.3f}), increasing sigma to {es.sigma:.3f}")
        gen_best_idx = np.argmin(fitnesses)
        gen_best_solution = solutions[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution

        print(f"[Gen {gen+1}] Best scale: {gen_best_solution}, discrepancy: {gen_best_fitness:.6f}, entropy: {entropy:.3f}")

    # Return final CMA mean and standard deviation
    final_mean = es.mean
    final_std = es.sigma * np.sqrt(np.diag(es.C))
    return final_mean, final_std, best_solution


