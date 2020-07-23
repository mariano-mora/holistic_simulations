from agent import AgentWithStrategy, AgentKnowledge
from game import CooperationGameWithTeachersAndLearners, Role
from strategies import StrategyTypes
from game_utils import get_new_environment, get_stochastic_permutation_matrix
from environment import meanings, Categories




num_agents = 20
num_games = 5
env_size = 8
num_objects = 60



if __name__ == "__main__":
	learners = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='counting_variable_word_order', role=Role.LISTENER) for i in range(num_agents/2)]
	teachers = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='matrix_3D_fixed_word_order', role=Role.SPEAKER) for i in range(num_agents/2)]
	language_matrix = get_stochastic_permutation_matrix(len(AgentKnowledge.symbols), len(meanings), Categories.get_number_of_categories())
	for teacher in teachers:
		teacher.architecture.language_matrix = language_matrix
	env = get_new_environment(env_size, num_objects)
	game = CooperationGameWithTeachersAndLearners(env, teachers, learners, "teachers_learners_game")
	time_steps = []
	num_timesteps = 0
	is_game_finished = False
	for i in range(num_games):
		while not is_game_finished:
			if game.status.number_of_attempts == 6:
				pass
			is_game_finished = game.consume_time_step()
			num_timesteps += 1
		time_steps.append(num_timesteps)
		env = get_new_environment(env_size, num_objects)
		game.env = env
		num_timesteps = 0
		game.umpire.compare_agents_grammar(learners)
		is_game_finished = False
	print(time_steps
