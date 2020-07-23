from agent import AgentWithStrategy, AgentKnowledge, MatrixCognitiveArchitecture, get_symbol_indices, get_goal_indices
from game import CooperationGameWithTeachersAndLearners, Role, StochasticVariableWordOrderUmpire
from strategies import StrategyTypes
from game_utils import get_new_environment, get_permutation_matrix
from environment import meanings, meanings_dict, Categories, GOALS
import numpy as np
from itertools import product
from random import choice


num_agents = 20
num_games = 1
env_size = 8
num_objects = 60


class SamplingArchitecture(MatrixCognitiveArchitecture):
    def __init__(self):
        super(SamplingArchitecture, self).__init__()
        row_size = len(AgentKnowledge.symbols)
        col_size = len(meanings)
        initial_weight = 1. / (row_size * col_size)
        self.language_matrix = np.full((row_size, col_size), initial_weight, dtype=float)
        self.interaction_matrix = None
        self.interaction_memory = None
        self.combinations = None
        self.update_factor = 0.3
        self.threshold = 0.0001
        self.built_combinations = False

    def choose_goal_to_perform(self, status, env):
        goal = None
        if not self.built_combinations:
            self.new_interaction_memory()
            self.combinations = self.build_combinations(status.sequence)
            status.number_of_combinations = len(self.combinations)
            self.built_combinations = True
            print("number of combinations:  ", status.number_of_combinations
        while self.combinations and not goal:
            if len(self.combinations)==0:
                return None
            combination = self.combinations.pop(0)
            goal = self.get_goal_from_combination(combination)
        if not goal:
            print("NO GOAL AFTER COMBINATIONS"
            print(self.language_matrix
            print(self.interaction_matrix
        return goal

    def get_goal_from_combination(self, combination):
        possible_goals = GOALS
        for cat, comb in enumerate(combination):
            meaning = meanings[comb]
            possible_goals = self.filter_goal_by_category(possible_goals, cat, meaning)
            if not possible_goals:
                return None
        if possible_goals:
            return choice(possible_goals)

    def new_interaction_memory(self):
        categories = Categories.get_categories()
        self.interaction_memory = dict(zip([cat[1] for cat in categories], [[] for i in range(len(categories))]))

    def build_combinations(self, sequence):
        symbol_indices = get_symbol_indices(sequence)
        for symbol_index in symbol_indices:
            row = np.copy(self.interaction_matrix[symbol_index, :])
            non_zero = np.nonzero(row)[0]
            args_values = zip(non_zero, row[non_zero])
            for arg in args_values:
                cat = meanings_dict[meanings[arg[0]]]
                if arg not in self.interaction_memory[cat]:
                    self.interaction_memory[cat].append(arg)
        for k, v in self.interaction_memory.iteritems():
            sorted_combs = sorted(v, key=lambda x: x[1])
            self.interaction_memory[k] = [comb[0] for comb in sorted_combs]
        known_categories = [(k, v) for k, v in self.interaction_memory.iteritems() if v]
        return list(product(*(t[1] for t in known_categories)))

    def prepare_for_interaction(self, status, env):
        self.combinations = None
        self.interaction_memory = None
        self.built_combinations = False
        self.interaction_matrix = np.copy(self.language_matrix)
        for i in range(self.interaction_matrix.shape[1]):
            meaning = meanings[i]
            category = meanings_dict[meaning]
            if not self.check_feasibility_of_meaning(category, meaning, env, status):
                self.interaction_matrix[:, i] = 0.

    def react_to_success(self, status):
        print("number of attempts: ", status.number_of_attempts
        goal = status.goal
        symbol_indices = list(get_symbol_indices(status.sequence))
        meaning_indices = list(get_goal_indices(goal))
        for index in symbol_indices:
            row = self.language_matrix[index, :]
            mask = np.zeros(row.shape, dtype=bool)
            mask[meaning_indices] = True
            row[~mask] *= (1 - self.update_factor)
            row[mask] *= (1 + self.update_factor)
            row[row < self.threshold] = 0.
        self.language_matrix /= self.language_matrix.sum()

if __name__ == '__main__':
    learners = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture=SamplingArchitecture, role=Role.LISTENER)
        for i in range(num_agents)]
    teachers = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='matrix_fixed_word_order', role=Role.SPEAKER)]
    language_matrix = get_permutation_matrix(len(AgentKnowledge.symbols), len(meanings), float)
    for teacher in teachers:
        teacher.architecture.language_matrix = language_matrix
    env = get_new_environment(env_size, num_objects)
    game = CooperationGameWithTeachersAndLearners(env, teachers, learners, "teachers_learners_game")
    is_game_finished = False
    for i in range(num_games):
        while not is_game_finished:
            is_game_finished = game.consume_time_step()
        env = get_new_environment(env_size, num_objects)
        game.env = env
        is_game_finished = False
