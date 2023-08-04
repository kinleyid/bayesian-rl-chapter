
import csv
import numpy as np

proj_path = 'C:/Users/isaac/Projects/bayesian-rl-chapter/github/'
data_path = proj_path + 'data/'

params = {
    'forget_rate': 0.95,
    'temperature': 1/3,
    'elig_decay': 0.1,
    'sensory_noise': 0.1,
    'theta0': 1,
    'normal_prior': {'mu': 0, 's2': 1}
}

def softmax(x, temperature):
    numerator = [np.exp(item/temperature) for item in x]
    denominator = sum(numerator)
    return [each/denominator for each in numerator]

class Environment:
    def __init__(self, transitions, initial_state):
        self.transitions = transitions # mapping states and actions to states and probabilities
        self.initial_state = initial_state # remember the initial state
        self.state = self.initial_state
        # get all states
        self.all_states = list(set(s for s, a in transitions))
        for SA in self.transitions:
            self.all_states += [s for s, p in self.transitions[SA] if s not in self.all_states]
        non_terminal_states = [s for s, a in self.transitions]
        self.non_terminal_states = non_terminal_states
        self.terminal_states = [s for s in self.all_states if s not in non_terminal_states]
        self.all_actions = list(set(a for s, a in self.transitions))
    def reset(self):
        self.state = self.initial_state
    def take_action(self, action):
        distribution = self.transitions[(self.state, action)]
        sp = np.random.choice(
            [s for s, p in distribution],
            p=[p for s, p in distribution])
        self.state = sp
        # Return probability of the transition that took place
        return {s: p for s, p in distribution}[sp]

class MF_Agent:
    def __init__(self, env):
        self.temperature = params['temperature']
        self.cache = {sa: params['normal_prior'].copy() for sa in env.transitions}
        self.elig = {sa: 0 for sa in env.transitions}
    def reset(self):
        self.elig = {sa: 0 for sa in self.elig}
    def Q(self, state, action):
        # Access cached values
        return self.cache[(state, action)]['mu']
    def get_actions_and_probs(self, state):
        Qs = [(a, self.Q(s, a)) for s, a in self.cache if s == state]
        actions = [a for a, q in Qs]
        probs = softmax([q for a, q in Qs], self.temperature)
        return (actions, probs)
    def best_aQ(self, state):
        aQs = [{'a': a, 'Q': self.cache[(s, a)]} for s, a in self.cache if s == state]
        return max(aQs, key=lambda aQ: aQ['Q']['mu'])
    def uncertainty(self, state):
        # Find best next action
        Q = self.best_aQ(state)['Q']
        variance = Q['s2'] + params['sensory_noise']
        return variance
    def decide(self, state):
        actions, probs = self.get_actions_and_probs(state)
        a = np.random.choice(actions, p=probs)
        return a
    def update(self, s, a, sp, r, s2=None):
        Q = self.cache[(s, a)]
        self.elig[(s, a)] = 1
        if s2 is None:
            s2 = params['sensory_noise'] # Default value
        # Compute delta
        mu0 = self.cache[(s, a)]['mu']
        r = np.array(r, ndmin=1)
        delta = r - mu0
        s2 = np.array(s2, ndmin=1)
        # Compute variance
        terminal = sp not in [s for s, a in self.cache]
        if not terminal:
            best_Q = self.best_aQ(sp)['Q']
            delta += best_Q['mu']
            s2_sp = best_Q['s2']
        for sa in self.elig:
            Q = self.cache[sa]
            m = self.elig[sa]
            # Exponential forgetting
            w = 1 - self.elig[sa]*(1 - params['forget_rate'])
            Q['s2'] = w*Q['s2'] + (1 - w)*params['normal_prior']['s2']
            Q['mu'] = w*Q['mu'] + (1 - w)*params['normal_prior']['mu']
            for i in range(len(delta)):
                # Priors
                s20 = Q['s2']
                mu0 = Q['mu']
                alpha = m*s20/(s2[i] + m*s20)
                # Posterior
                Q['s2'] = (1/s20 + m/s2[i])**-1
                Q['mu'] = mu0 + alpha*delta[i]
                if not terminal:
                    Q['s2'] += alpha**2*(s2_sp + s2[i])
            self.elig[sa] *= params['elig_decay']

class MB_Agent:
    def __init__(self, env):
        # Model of the environment---number of observed transitions from one state to another
        self.M = {(s, a): [(sp, params['theta0']) for sp in env.all_states] for s, a in env.transitions}
        # Reward values for each state
        self.R = {s: params['normal_prior'].copy() if s in env.terminal_states else {'mu': 0, 's2': 0} for s in env.all_states}
        self.iterated_values = {(s, a): None for s, a in env.transitions}
        self.iterated_uncertainties = {(s, a): None for s, a in env.transitions}
        self.value_iteration()
        self.temperature = params['temperature']
    def reset(self):
        pass
    def transition_probabilities(self, state, action):
        # Returns list of (state, probability) tuples
        t = sum(n for sp, n in self.M[(state, action)])
        return [(sp, n/t) for sp, n in self.M[(state, action)]]
    def value_iteration(self):
        M1 = {s: self.R[s]['mu'] for s in self.R} # First moments/values
        M2 = {s: 0 if s in self.M else (self.R[s]['mu']**2 + self.R[s]['s2'] + params['sensory_noise']) for s in self.R} # Second moments
        non_terminal_states = list(set(s for s, a in self.M))
        while True:
            delta = 0
            for s in non_terminal_states:
                m1 = M1[s] # Store initial value for later comparison
                m2 = M2[s]
                best_M1 = -float('inf')
                available_actions = [action for state, action in self.M if state == s]
                for a in available_actions:
                    tp = self.transition_probabilities(s, a)
                    # Compute expected moments
                    E_M1 = sum(p*M1[sp] for sp, p in tp)
                    E_M2 = sum(p*M2[sp] for sp, p in tp)
                    # Store for later use
                    self.iterated_values[(s, a)] = E_M1
                    self.iterated_uncertainties[(s, a)] = E_M2 - E_M1**2
                    if E_M1 > best_M1:
                        best_M1 = E_M1
                        best_M2 = E_M2
                M1[s] = best_M1
                M2[s] = best_M2
                delta = max(delta, abs(m1 - M1[s]), abs(m2 - M2[s]))
            if delta < 1e-3:
                break
    def Q(self, state, action):
        return self.iterated_values[(state, action)]
    def uncertainty(self, state):
        best_action = self.best_aQ(state)['a']
        return self.iterated_uncertainties[state, best_action]
    def best_aQ(self, state):
        # returns optimal a, Q pair from a given state
        aps = [a for s, a in self.M if s == state]
        if len(aps) > 0:
            aQs = [{'a': a, 'Q': self.Q(state, a)} for a in aps]
            max_aQ = max(aQs, key=lambda aQ: aQ['Q'])
        else:
            # Terminal state
            max_aQ = {'a': None, 'Q': 0}
        return max_aQ
    def get_actions_and_probs(self, state):
        Qs = [(a, self.Q(s, a)) for s, a in self.M if s == state]
        actions = [a for a, q in Qs]
        probs = softmax([q for a, q in Qs], self.temperature)
        return (actions, probs)
    def decide(self, state):
        if state in [s for s, a in self.M]:    
            actions, probs = self.get_actions_and_probs(state)
            a = np.random.choice(actions, p=probs)
        else:
            # Terminal state
            a = None
        return a
    def update(self, s, a, sp, r, s2=None):
        w = params['forget_rate']
        # Update transitions
        # Exponential forgetting
        self.M[(s, a)] = [(ss, w*n + (1 - w)*params['theta0']) for ss, n in self.M[(s, a)]]
        # Updates
        self.M[(s, a)] = [(ss, n + (1 if ss == sp else 0)) for ss, n in self.M[(s, a)]]
        # Update reward estimates
        R = self.R[sp]
        if s2 is None:
            s2 = params['sensory_noise'] # Default value
        # Exponential forgetting
        R['s2'] = w*R['s2'] + (1 - w)*params['normal_prior']['s2']
        R['mu'] = w*R['mu'] + (1 - w)*params['normal_prior']['mu']
        r = np.array(r, ndmin=1)
        s2 = np.array(s2, ndmin=1)
        for i in range(len(r)):
            # Priors
            s20 = R['s2']
            mu0 = R['mu']
            # Posteriors
            s2n = (1/s20 + 1/s2[i])**-1
            mun = (1/s20 + 1/s2[i])**-1 * (mu0/s20 + 1*r[i]/s2[i])
            R['s2'] = s2n
            R['mu'] = mun
        self.value_iteration()

def write_results(filename, data):  
    with open(filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cols = list(data[0].keys())
        writer.writerow(cols)
        for row in data:
            writer.writerow([row[col] for col in cols])