
import rl
rl.np.random.seed(123)

transitions = {
    # Level 1:
    ('Initial', 'Lever'): [('Reward delivered', 1), ('No reward', 0)],
    ('Initial', 'Enter'):   [('Reward delivered', 0), ('No reward', 1)],
    # Level 2:
    ('Reward delivered', 'Lever'):  [('Reward obtained', 0), ('No reward', 1)],
    ('Reward delivered', 'Enter'):  [('Reward obtained', 1), ('No reward', 0)]
}
env = rl.Environment(transitions=transitions, initial_state='Initial')

n_rep = 100
n_train = 50

trials = []
for rep_n in range(n_rep):
    print(rep_n/n_rep)
    for addictive in [True, False]:
        agents = {
            'MF': rl.MF_Agent(env),
            'MB': rl.MB_Agent(env)
        }
        for agent_type, agent in agents.items():
            trial = {'rep_n': rep_n, 'agent_type': agent_type}
            # Pre-train
            uncertainties = []
            reward_value = 1
            for train_n in range(n_train):
                env.reset()
                agent.reset()
                punishment = train_n >= 20 # Add punishment after x number of trials
                trials.append({
                    'rep_n': rep_n,
                    'train_n': train_n,
                    'addictive': addictive,
                    'agent_type': agent_type,
                    'Q_lever': agent.Q('Initial', 'Lever'),
                    'reward_value': reward_value
                })
                while env.state not in env.terminal_states:
                    s1 = env.state
                    a1 = agent.decide(s1)
                    env.take_action(a1)
                    s2 = env.state
                    if s2 == 'Reward obtained':
                        r = reward_value
                    else:
                        r = 0
                    if r > 0 and addictive:
                        sensory_noise = rl.params['sensory_noise'] / 10
                    else:
                        sensory_noise = rl.params['sensory_noise']
                    if s2 == 'Reward obtained' and punishment:
                        r = [r, -2]
                        sensory_noise = [sensory_noise, rl.params['sensory_noise']]
                    agent.update(s1, a1, s2, r, s2=sensory_noise)

rl.write_results(rl.data_path + 'compulsivity.csv', trials)