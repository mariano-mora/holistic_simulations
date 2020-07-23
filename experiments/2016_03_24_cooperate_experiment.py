"""
	Script to test language binding represented as a matrix
"""

from agent import AgentWithStrategy
from game import GameResult
from strategies import StrategyTypes
from game_utils import setup_game_with_agents


def create_teams(strategy1, strategy2, max_tries_1=1, max_tries_2=1, num_agents=10):
	agents1 = [
		AgentWithStrategy(strategy1, architecture='matrix', creator='uniform_random', max_tries=max_tries_1)
                              for i in range(num_agents)]
	agents2 = [
		AgentWithStrategy(strategy2, architecture='matrix', creator='uniform_random', max_tries=max_tries_2)
                          for i in range(num_agents)]
	return agents1, agents2


def play_one_game(agents1, agents2):
	results = []
	game1, game2 = setup_game_with_agents(8, 50, agents1, agents2)
	is_game_finished = False
	number_of_time_steps = 0
	strategy1 = agents1[0].strategy
	strategy2 = agents2[0].strategy
	while not is_game_finished:
		if game1.consume_time_step():
			results.append(GameResult(strategy1.name, number_of_time_steps))
			is_game_finished = True
		if game2.consume_time_step():
			results.append(GameResult(strategy2.name, number_of_time_steps))
			is_game_finished = True
		number_of_time_steps += 1
	return results, game1, game2

if __name__ == "__main__":
	agents_1, agents_2 = create_teams(StrategyTypes.EXHAUSTIVE, StrategyTypes.MIXED, max_tries_2=3)
	results, game_1, game_2 = play_one_game(agents_1, agents_2)
	for agent in agents_1[:2]:
		print(agent.architecture.language_matrix