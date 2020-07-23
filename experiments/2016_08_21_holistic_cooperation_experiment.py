## An experiment for determining the effect of cooperation in agents learning single utterances for each action
## Includes an experiment where the system is underdetermined: there are elements that have same color or shape


from agent import AgentWithStrategy, AgentKnowledge, HolisticCognitiveArchitecture
from game import CooperationHolisticGameWithTeachersAndLearners, Role, HolisticGameUmpire
from strategies import StrategyTypes
from game_utils import get_new_reduced_environment, get_permutation_matrix
import numpy as np

num_agents = 20
num_games = 30
env_size = 8
num_objects = 60


if __name__ == "__main__":
	env = get_new_reduced_environment(env_size, num_objects)

	learners = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.symbols), len(env.goals)), role=Role.LISTENER) for i in range(num_agents/2)]
	teachers = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture=HolisticCognitiveArchitecture(len(AgentKnowledge.symbols), len(env.goals)), role=Role.SPEAKER) for i in range(num_agents/2)]
	language_matrix = get_permutation_matrix(len(AgentKnowledge.symbols), len(env.goals), float)
	for teacher in teachers:
		teacher.architecture.language_matrix = language_matrix
	game = CooperationHolisticGameWithTeachersAndLearners(env, teachers, learners, "teachers_learners_game_holistic", umpire=HolisticGameUmpire(env, len(env.goals)))
	is_game_finished = False
	for i in range(num_games):
		while not is_game_finished:
			is_game_finished = game.consume_time_step()
		env = get_new_reduced_environment(env_size, num_objects)
		print("GAME FINISHED ", i
		game.env = env
		is_game_finished = False

	for i in range(language_matrix.shape[0]):
		print(language_matrix[i,:]
		print(learners[0].architecture.language_matrix[i,:]
		print(np.argmax(language_matrix[i,:])
		print(np.argmax(learners[0].architecture.language_matrix[i,:])
	print(game.umpire.goal_track
	for i in range(game.umpire.goal_track.shape[0]):
		if game.umpire.goal_track[i] == 0:
			print(env.goals[i].as_symbols()



