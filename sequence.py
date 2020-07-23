from abc import abstractmethod, ABCMeta
import random

__all__ = ['RandomStrategy', 'Sequence', 'SequenceCreator',
			'UniformSequenceCreator', 'ErdosRanyiSequenceCreator',
			'creators', 'Outcome'
			]


class RandomStrategy:
	""" Enum for different strategies for creating sequences """
	uniform_random = 0
	erdos_ranyi = 1
	literal = 2


class Sequence(object):
	"""
		A sequence of symbols
	"""

	def __init__(self, symbols=None):
		self.symbols = [] if not symbols else symbols

	def add_symbol(self, symbol):
		self.symbols.append(symbol)

	def insert_symbol(self, index, symbol):
		self.symbols.insert(index, symbol)

	def length(self):
		return len(self.symbols)

	def compare(self, other):
		return self.symbols == other.symbols

	def as_string(self):
		return ' '.join(self.symbols)

	def __iter__(self):
		return iter(self.symbols)

	def __len__(self):
		return len(self.symbols)


# METHODS TO CREATE SEQUENCES

class SequenceCreator(object):
	__metaclass__ = ABCMeta

	''' A class to create sequences according to a strategy  '''

	def __init__(self, max_length=5):
		self.max_length = max_length

	@abstractmethod
	def create_sequence(self, symbols, length=None):
		return


class UniformSequenceCreator(SequenceCreator):
	""" Create sequence using a uniform distribution. """

	def __init__(self, *args, **kwargs):
		super(UniformSequenceCreator, self).__init__(*args, **kwargs)

	def create_sequence(self, symbols, length=None):
		length = length if length else random.randint(1, self.max_length)
		return Sequence(random.sample(symbols, length))


class ErdosRanyiSequenceCreator(SequenceCreator):
	""" Create sequence using binomial distribution  """

	def __init__(self):
		super(ErdosRanyiSequenceCreator, self).__init__()

	# TODO: implement this
	def create_sequence(self, symbols, length=None):
		print("creating")


class LiteralSequenceCreator(SequenceCreator):
	def __init__(self):
		super(LiteralSequenceCreator, self).__init__()

	def create_sequence(self, symbols, length=None):
		sym = '_'.join([symbol for symbol in symbols if symbol])
		return Sequence(sym)


''' Dictionary outside of a class, to call the creator constructor from the strategy  '''
creators = {'uniform_random': UniformSequenceCreator,
			'erdos_ranyi': ErdosRanyiSequenceCreator,
			'literal': LiteralSequenceCreator}

'''
	The outcome follows a sequence
	The outcome could be a sequence as well
'''


class Outcome:
	def __init__(self, symbol=None):
		self.symbol = symbol

	def set_symbol(self, symbol):
		self.symbol = symbol

	def get_outcome(self):
		return self.symbol
