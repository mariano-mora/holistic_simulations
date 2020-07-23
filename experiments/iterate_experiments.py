from scripts import experiments
from scripts.alignment import alignment_strategies
import argparse
import nltk

# directory = '/home/mariano/repos/research_sketches/game_results/'
directory = '/Users/mariano/developing/results/'
num_games = 50
num_iterations = 1
exps = {"holistic" :{'teachers':experiments.holistic_teachers, 'negotiated':experiments.holistic_negotiated},
		'word':{'teachers':experiments.word_teachers, 'negotiated':experiments.word_negotiated},
        'position':{'teachers':experiments.position_teachers, 'negotiated':experiments.position_negotiated}
}

if __name__ == "__main__":

	nltk.parse_cfg()
	parser = argparse.ArgumentParser()
	parser.add_argument('--a', action='store_true')
	parser.add_argument('--e', type=str)
	parser.add_argument('--t', type=str)
	parser.add_argument('--s', type=str, default=None)

	args = parser.parse_args()
	negative_alignment = args.a
	exp = args.e
	exp_type = args.t
	alig_strategy = alignment_strategies.get(args.s, None)
	experiment = exps[exp][exp_type]
	experiment.main(num_iterations, negative_alignment, alig_strategy, num_games, directory)
