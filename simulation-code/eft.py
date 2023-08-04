
import rl

rl.np.random.seed(123)

n_reps = 100
n_train = 100
max_n_wait = 10
hazard_rate = 0.05

smaller_sooner = 1
larger_later = 3

trials = []
for variant in ['default',
                'eft']:
    print(variant)
    for n_wait in range(max_n_wait + 1):
        print(n_wait/max_n_wait)
        transitions = {}
        waiting_states = [f'Waiting {n + 1}' for n in range(n_wait)]
        initial_and_waiting = ['Waiting 0'] + waiting_states
        # You can always opt for the immediate reward
        for state in initial_and_waiting:
            transitions[(state, 'Immediate')] = [('Smaller sooner', 1)]
        # Or you can wait
        for state_idx in range(len(initial_and_waiting) - 1):
            transitions[(initial_and_waiting[state_idx], 'Delayed')] = [
                (initial_and_waiting[state_idx + 1], 1 - hazard_rate),
                ('No reward', hazard_rate)]
        # Once you've waited, you can get the delayed reward
        transitions[(f'Waiting {n_wait}', 'Delayed')] = [('Larger later', 1)]
    
        env = rl.Environment(transitions=transitions, initial_state='Waiting 0')
        for rep_n in range(n_reps):            
            agent = rl.MB_Agent(env)
            trial = {
                'n_wait': n_wait,
                'rep_n': rep_n,
                'variant': variant
            }
            # Training
            for train_n in range(n_train):
                env.reset()
                agent.reset()
                while env.state not in env.terminal_states:
                    s1 = env.state
                    a1 = agent.decide(s1)
                    env.take_action(a1)
                    s2 = env.state
                    r = larger_later if s2 == 'Larger later' else smaller_sooner if s2 == 'Smaller sooner' else 0
                    agent.update(s1, a1, s2, r)
            # Both are accurate
            agent.R['Larger later']['mu'] = larger_later
            agent.R['Smaller sooner']['mu'] = 0
            if variant == 'eft':
                # Reduce the variance of the larger later reward
                agent.R['Larger later']['s2'] = 1e-3
                for wait_n in range(n_wait + 1):
                    for i, (sp, p) in enumerate(agent.M[(f'Waiting {wait_n}', 'Immediate')]):
                        if sp == 'Smaller sooner':
                            agent.M[(f'Waiting {wait_n}', 'Immediate')][i] = (sp, 1e3)
                            break
                    if wait_n < n_wait:
                        # Increase the estimated probability that waiting will lead to the next waiting state
                        for i, (sp, p) in enumerate(agent.M[(f'Waiting {wait_n}', 'Delayed')]):
                            if sp == f'Waiting {wait_n + 1}':
                                agent.M[(f'Waiting {wait_n}', 'Delayed')][i] = (sp, 1e3)
                                break
                    else:
                        # Increase the estimated probability that waiting will lead to the larger later reward
                        for i, (sp, p) in enumerate(agent.M[(f'Waiting {wait_n}', 'Delayed')]):
                            if sp == 'Larger later':
                                agent.M[(f'Waiting {wait_n}', 'Delayed')][i] = (sp, 1e3)
                                break
                        
            agent.value_iteration()
            # Testing
            p_delayed = []
            for s in initial_and_waiting:
                aps = {a: p for a, p in zip(*agent.get_actions_and_probs(s))}
                p_delayed.append(aps['Delayed'])
            trial['p_delayed'] = rl.np.prod(p_delayed)
            trial['uncertainty'] = agent.uncertainty('Waiting 0')
            trials.append(trial)

rl.write_results('C:/Users/isaac/Projects/bayesian-rl-chapter/dev/eft.csv', trials)