from agent import AgentWithStrategy
from GridEnvironment import GridEnvironment
from environment import RandomInitializer
from game import CooperationGame, GameResult
from strategies import StrategyTypes
from numpy import mean


def setup_environments(size, num_objects):
    grid_env_1 = GridEnvironment(size, size)
    grid_env_2 = GridEnvironment(size, size)
    initializer = RandomInitializer(num_objects=num_objects)
    initializer.init_environments(grid_env_1, grid_env_2)
    return grid_env_1, grid_env_2


def setup_game_with_agents(size, num_objects, coop_agents, non_coop_agents):
    env_1, env_2 = setup_environments(size, num_objects)
    cooperative_game = CooperationGame(env_1, coop_agents, "cooperative_10_agents_1")
    non_cooperative_game = CooperationGame(env_2, non_coop_agents, "non_cooperative_10_agents_1")
    return cooperative_game, non_cooperative_game


def setup_game_with_new_agents(size, num_agents, num_objects):
    env_1, env_2 = setup_environments(size, num_objects)
    number_agents = num_agents
    cooperative_agents = [AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='multiple', creator='uniform_random')
                          for i in range(number_agents)]
    cooperative_game = CooperationGame(env_1, cooperative_agents, "cooperative_10_agents_1")
    non_cooperative_agents = [
        AgentWithStrategy(StrategyTypes.NON_EXHAUSTIVE, architecture='multiple', creator='uniform_random')
        for i in range(number_agents)]
    non_cooperative_game = CooperationGame(env_2, non_cooperative_agents, "non_cooperative_10_agents_1")
    return cooperative_game, non_cooperative_game


def reward_agents(agents, reward):
    for agent in agents:
        agent.receive_reward(reward)

def calculate_costs(coop_agents, non_coop_agents, strat1, strat2):
    costs = {}
    costs[strat1] = mean([agent.cost for agent in coop_agents])
    costs[strat2] = mean([agent.cost for agent in non_coop_agents])
    return costs

if __name__ == "__main__":
    num_agents = 10
    num_objects = 50
    num_games = 100
    size = 8
    reward = 300
    results = []
    costs = []
    strat1 = StrategyTypes.EXHAUSTIVE
    strat2 = StrategyTypes.MIXED
    cooperative_agents = [
        AgentWithStrategy(StrategyTypes.EXHAUSTIVE, architecture='multiple', creator='uniform_random')
                          for i in range(num_agents)]
    non_cooperative_agents = [
        AgentWithStrategy(StrategyTypes.MIXED, architecture='multiple', creator='uniform_random',
                          max_tries=2)
                              for i in range(num_agents)]

    for i in range(num_games):
        number_of_time_steps = 0
        coop_game, not_coop_game = setup_game_with_agents(size, num_objects, cooperative_agents, non_cooperative_agents)
        is_game_finished = False
        while not is_game_finished:
            if coop_game.consume_time_step():
                results.append(GameResult(strat1, number_of_time_steps))
                reward_agents(coop_game.agents, reward)
                costs.append(calculate_costs(coop_game.agents, not_coop_game.agents, strat1, strat2))
                is_game_finished = True
            if not_coop_game.consume_time_step():
                results.append(GameResult(strat2, number_of_time_steps))
                reward_agents(not_coop_game.agents, reward)
                costs.append(calculate_costs(coop_game.agents, not_coop_game.agents, strat1, strat2))
                is_game_finished = True
            number_of_time_steps += 1
