
import rl

default_params = rl.params
rl.np.random.seed(123)

n_reps = 100
n_train = 100
max_n_wait = 10
hazard_rate = 0.05

smaller_sooner = 1
larger_later = 3

for variant in ['default',
                'adjusted-params',
                'revaluation']:
    print(variant)
    if variant == 'adjusted-params':
        rl.params = {
            'forget_rate': 0.95,
            'temperature': 1/3,
            'elig_decay': 1,
            'sensory_noise': 0.1,
            'theta0': 5,
            'normal_prior': {'mu': 0, 's2': 1}
        }
    else:
        rl.params = default_params
    
    trials = []
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
            agents = {
                'MB': rl.MB_Agent(env),
                'MF': rl.MF_Agent(env)
            }
            for agent_type, agent in agents.items():
                trial = {
                    'n_wait': n_wait,
                    'rep_n': rep_n,
                    'agent_type': agent_type
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
                if variant == 'revaluation':
                    if agent_type == 'MB':
                        agent.R['Larger later']['mu'] = larger_later
                        agent.R['Smaller sooner']['mu'] = 0
                        agent.value_iteration()
                # Testing
                agent.reset()
                p_delayed = []
                for s in initial_and_waiting:
                    aps = {a: p for a, p in zip(*agent.get_actions_and_probs(s))}
                    p_delayed.append(aps['Delayed'])
                trial['p_delayed'] = rl.np.prod(p_delayed)
                trials.append(trial)
    
    rl.write_results(rl.data_path + 'impulsivity-' + variant + '.csv', trials)