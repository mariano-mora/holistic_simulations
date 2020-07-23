from agent import AgentWithStrategy
from GridEnvironment import GridEnvironment
from environment import RandomInitializer
from game import CooperationGame
from strategies import strategies, StrategyTypes

def coop_interaction_cb(game):
	return game.perform_interaction(True)

def non_coop_interaction_cb(game):
	return game.perform_interaction(False)

def setup_game():
	grid_env_1 = GridEnvironment(8, 8)
	grid_env_2 = GridEnvironment(8, 8)
	initializer = RandomInitializer(num_objects=100)
	initializer.init_environments(grid_env_1, grid_env_2)
	number_agents = 10
	cooperative_agents = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE) for i in range(number_agents)]
	cooperative_game = CooperationGame(grid_env_1, cooperative_agents, "cooperative_10_agents_1")
	non_cooperative_agents = [AgentWithStrategy(StrategyTypes.NON_EXHAUSTIVE) for i in range(number_agents)]
	non_cooperative_game = CooperationGame(grid_env_2, non_cooperative_agents, "non_cooperative_10_agents_1")
	return cooperative_game, non_cooperative_game

if __name__ == "__main__":
	number_of_games = 50
	victories = {'cooperative':0, 'non-cooperative':0}
	number_of_time_steps = 0
	for i in range(number_of_games):
		coop_game, not_coop_game = setup_game()
		is_game_finished = False
		while not is_game_finished:
			if coop_game.consume_time_step():
				victories['cooperative'] += 1
				is_game_finished = True
			if not_coop_game.consume_time_step():
				victories['non-cooperative'] += 1
				is_game_finished = True
			number_of_time_steps += 1
