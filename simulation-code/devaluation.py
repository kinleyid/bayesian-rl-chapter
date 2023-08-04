
# The purpose of this script is to get results for a compulsivity task

import rl
rl.np.random.seed(123)

params = {
    'forget_rate': 0.95,
    'temperature': 1/3,
    'elig_decay': 0,
    'sensory_noise': 0.1,
    'theta0': 1,
    'normal_prior': {'mu': 0, 's2': 1}
}

transitions = {}
level_1 = {
    ('Initial', 'Lever'):   [('Reward delivered', 1), ('No reward', 0)],
    ('Initial', 'Enter'):   [('Reward delivered', 0), ('No reward', 1)]
}
level_2 = {
    ('Reward delivered', 'Lever'):  [('Reward obtained', 0), ('No reward', 1)],
    ('Reward delivered', 'Enter'):  [('Reward obtained', 1), ('No reward', 0)]
}

transitions.update(level_1)
transitions.update(level_2)

env = rl.Environment(transitions=transitions, initial_state='Initial')

# Pre-train, model-based

n_rep = 100
n_train = 200
n_test = 100

trials = []
sub_trials = []
sub_trials_deval = []

for rep_n in range(n_rep):
    print(rep_n/n_rep)
    agents = {
        'MF': rl.MF_Agent(env),
        'MB': rl.MB_Agent(env)
    }
    for agent_type, agent in agents.items():
        for train_n in range(n_train):
            env.reset()
            agent.reset()
            sub_trials.append({
                'agent_type': agent_type,
                'rep_n': rep_n,
                'train_n': train_n,
                'uncertainty': agent.uncertainty(env.state)
            })
            while env.state not in env.terminal_states:
                s1 = env.state
                a1 = agent.decide(s1)
                env.take_action(a1)
                s2 = env.state
                r = 1 if 'obtained' in s2 else 0
                # r = np.random.normal(mu, 0.2)
                agent.update(s1, a1, s2, r)
        
        # Test responses before devaluation
        actions = [agent.decide('Initial') for i in range(n_test)]
        trials.append({
            'rep_n': rep_n,
            'agent_type': agent_type,
            'phase': 'pre',
            'lever_rate': actions.count('Lever')/n_test
        })
        # Devaluation
        if agent_type == 'MB':
            agent.R['Reward obtained']['mu'] = 0
            agent.value_iteration()
        else:
            pass # No way to represent this for model-free agent
        # Test responses after devaluation
        actions = [agent.decide('Initial') for i in range(n_test)]
        trials.append({
            'rep_n': rep_n,
            'agent_type': agent_type,
            'phase': 'post',
            'lever_rate': actions.count('Lever')/n_test
        })

rl.write_results(rl.data_path + 'devaluation-choices.csv', trials)
rl.write_results(rl.data_path + 'devaluation-uncertainty.csv', sub_trials)