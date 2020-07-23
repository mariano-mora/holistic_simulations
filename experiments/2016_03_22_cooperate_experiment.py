from agent import AgentWithStrategy, reward_agents
from game import GameResult
from strategies import StrategyTypes
from game_utils import setup_game_with_agents
import numpy as np

num_agents = 10
num_objects = 50
num_games = 100
size = 8
reward = 100


def calculate_energy(coop_agents, non_coop_agents, strat1, strat2):
	energy = {}
	energy[strat1.name] = np.mean([agent.cost for agent in coop_agents])
	energy[strat2.name] = np.mean([agent.cost for agent in non_coop_agents])
	return energy


def create_teams(strategy1, strategy2, max_tries_1=1, max_tries_2=1):
	agents1 = [AgentWithStrategy(strategy1, architecture='multiple', creator='uniform_random',
	                             max_tries=max_tries_1)
	           for i in range(num_agents)]
	agents2 = [AgentWithStrategy(strategy2, architecture='multiple', creator='uniform_random',
	                             max_tries=max_tries_2)
	           for i in range(num_agents)]
	return agents1, agents2


if __name__ == "__main__":

	agents_1, agents_2 = create_teams(StrategyTypes.NON_EXHAUSTIVE,StrategyTypes.MIXED, max_tries_2=2)
	results = []
	costs = []
	for i in range(num_games):
		number_of_time_steps = 0
		strategy1 = agents_1[0].strategy
		strategy2 = agents_2[0].strategy
		game1, game2 = setup_game_with_agents(size, num_objects, agents_1, agents_2)
		is_game_finished = False
		while not is_game_finished:
			if game1.consume_time_step():
				results.append(GameResult(strategy1.name, number_of_time_steps))
				reward_agents(game1.agents, reward)
				costs.append(calculate_energy(agents_1, agents_2, strategy1, strategy2))
				is_game_finished = True
			if game2.consume_time_step():
				results.append(GameResult(strategy2.name, number_of_time_steps))
				reward_agents(game2.agents, reward)
				costs.append(calculate_energy(agents_1, agents_2, strategy1, strategy2))
				is_game_finished = True
			number_of_time_steps += 1
