# from coordinate_game_Qt import select_destination
from numpy import mean, random


class StrategyTypes:
	ALTRUISTIC = 0
	MUTUALISTIC = 1
	EXHAUSTIVE = 2
	NON_EXHAUSTIVE = 3
	MIXED = 4
	STOCHASTIC_MIXED = 5



class ExhaustiveStrategy:
	def __init__(self, max_tries=0):
		self.name = "exhaustive"
		self.max_tries = None

	def should_repeat(self, status):
		return True


class NonExhaustiveStrategy:
	def __init__(self, max_tries=0):
		self.name = "non-exhaustive"
		self.max_tries = None

	def should_repeat(self, status):
		return False

class MixedStrategy:
	def __init__(self, max_tries=0):
		self.name = "mixed_"+str(max_tries)
		self.max_tries = max_tries

	def should_repeat(self, status):
		return status.attempts < self.max_tries


class StochasticMixedStrategy:
	def __init__(self, max_tries=0):
		self.name = "stochastic-mixed_"+str(max_tries)
		self.max_tries = max_tries

	def should_repeat(self, status):
		pass


class MutualisticStrategy:
	def __init__(self):
		self.name = "mutualistic"
		self.type = StrategyTypes.MUTUALISTIC

	def should_repeat(self, reward, coord_cost, status, agent):
		expected_cost = mean(agent.cost_memory[-1]) + (status.number_of_attempts*coord_cost)
		diff = (reward - expected_cost)/reward
		prob = max(diff, 0)
		sample = random.uniform()
		return sample < prob


class AltruisticStrategy:
	def __init__(self):
		self.name = "altruistic"
		self.type = StrategyTypes.ALTRUISTIC

	def should_repeat(self, reward, coord_cost, status, agent):
		return True


strategies = {StrategyTypes.EXHAUSTIVE: ExhaustiveStrategy,
					StrategyTypes.NON_EXHAUSTIVE: NonExhaustiveStrategy,
					StrategyTypes.MIXED: MixedStrategy,
					StrategyTypes.STOCHASTIC_MIXED: StochasticMixedStrategy,
					StrategyTypes.MUTUALISTIC: MutualisticStrategy,
					StrategyTypes.ALTRUISTIC: AltruisticStrategy
			}

