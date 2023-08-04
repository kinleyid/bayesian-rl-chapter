
import rl
import random
random.seed(123)
rl.np.random.seed(123)

# Create the transition structure of the 2-stage task
actions = ['left', 'right']
states = ['2a', '2b']
transitions = {}
reward_probs = {}
for action_n, action in enumerate(actions):
    transitions[('1', action)] = []
    for state_n, state in enumerate(states):
        if state_n == action_n:
            p = 0.7
        else:
            p = 0.3
        transitions[('1', action)].append((state, p))
        terminal_state = state + '-' + action
        transitions[(state, action)] = [(terminal_state, 1)]

# Function to update the reward probabilities
def update_prob(prob):
    prob += random.gauss(0, 0.025)
    if prob < 0.25:
        prob = 0.5 - prob
    elif prob > 0.75:
        prob = 1.5 - prob
    return prob

n_reps = 100
n_trials = 250

trials = []
for rep_n in range(n_reps):
    print(rep_n/n_reps)
    # Create environment
    env = rl.Environment(transitions, '1')
    # Reset reward probabilities
    reward_probs = {}
    for s in env.terminal_states:
        p_reward = 0.25 + 0.5*random.random()
        reward_probs[s] = [(1, p_reward), (0, 1 - p_reward)]
    agents = {
        'MF': rl.MF_Agent(env),
        'MB': rl.MB_Agent(env)
    }
    for agent_type, agent in agents.items():
        if agent_type == 'MB':
            agent.R['2a']['s2'] = 0
            agent.R['2b']['s2'] = 0
        last_a1 = None
        last_trans = None
        last_r = None
        for trial_n in range(n_trials):
            if trial_n > 900:
                x = 10
            env.reset()
            agent.reset()
            # First stage
            s1 = env.state
            a1 = agent.decide(env.state)
            p = env.take_action(a1)
            trans = 'common' if p > 0.5 else 'rare'
            s2 = env.state
            r1 = 0 # No reward when transitioning to any of the second stages
            agent.update(s1, a1, s2, r1)
            # Second stage
            a2 = agent.decide(env.state)
            env.take_action(a2)
            s3 = env.state
            r2 = random.choices(
                population=[r for r, p in reward_probs[s3]],
                weights=[p for r, p in reward_probs[s3]])[0]
            agent.update(s2, a2, s3, r2)
            trials.append({
                'rep_n': rep_n,
                'trial_n': trial_n,
                'agent_type': agent_type,
                'a1': a1,
                'a2': a2,
                'stay': int(a1 == last_a1),
                'r': r2,
                'last_r': last_r,
                'last_trans': last_trans
                })
            last_a1 = a1
            last_trans = trans
            last_r = r2
            # Update reward probabilities
            for s in reward_probs:
                p_reward = {r: p for r, p in reward_probs[s]}[1]
                p_reward = update_prob(p_reward)
                reward_probs[s] = [(1, p_reward), (0, 1 - p_reward)]            

rl.write_results(rl.data_path + 'two-stage-task.csv', trials)
