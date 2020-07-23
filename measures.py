import numpy as np

char_a = 97
decimal_places = 10


def get_alphabet(size):
	return [chr(i+char_a) for i in range(size)]


def calculate_entropy(arr):
	return np.nansum(-arr*np.log2(arr))


def compute_jensen_shannon(distributions, weights=None):
	size = distributions[0].shape[0]
	if not weights:
		weights = np.array([1./len(distributions)] * len(distributions))
	weighted = [w*d for w, d in zip(weights, distributions)]
	mixture = ([w[i] for w in weighted] for i in range(size))
	pmf = np.array([np.array(m).sum() for m in mixture])
	H = np.around(np.nansum(-pmf*np.log2(pmf)), decimal_places)
	S = sum(w*calculate_entropy(d) for w, d in zip(weights, distributions))
	S = np.around(S, decimal_places)
	return H-S


def compute_agents_jensen_shannon(agents):
	acc = 0.0
	row_size = agents[0].architecture.language_matrix.shape[0]
	for i in range(row_size):
			dists = [agent.architecture.language_matrix[i, :] for agent in agents]
			acc += compute_jensen_shannon(dists)
	return acc


def compute_agents_jensen_shannon_3d(agents):
	acc = 0.0
	row_size, col_size = agents[0].architecture.language_matrix.shape[0], agents[0].architecture.language_matrix.shape[1]
	for i in range(row_size):
		for j in range(col_size):
			dists = [agent.architecture.language_matrix[i, j, :] for agent in agents]
			acc += compute_jensen_shannon(dists)
	return acc


def compute_hellinger(matrix1, matrix2, alpha=.5):
	acc = np.array([hellinger_divergence(matrix1[i, :], matrix2[i, :], alpha) for i in range(matrix1.shape[0])])
	return acc.sum()


def hellinger_divergence(p, q, alpha=1.):
	s = double_power_sum(p, q, alpha, 1.-alpha)
	return (s-1.)/(alpha-1.)


def double_power_sum(p, q, exp1=1., exp2=1.):
	return np.nansum(np.power(p, exp1) * np.power(q, exp2))


def consistent_pairings(n_signals, agent_combs, potential_pairings):
	sum_ = 0
	for agent1, agent2 in agent_combs:
		for signal in range(n_signals):
			signal_dist_1 = agent1.architecture.language_matrix[signal,:]
			signal_dist_2 = agent2.architecture.language_matrix[signal,:]
			max_1 = np.argwhere(signal_dist_1 == signal_dist_1.max())
			max_2 = np.argwhere(signal_dist_2 == signal_dist_2.max())
			if max_1.shape[0] == 1 and max_2.shape[0]==1 and max_1 == max_2:
				sum_ += 1
	sum_ /= n_signals
	sum_ /= potential_pairings
	return sum_