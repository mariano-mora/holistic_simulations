from collections import deque
import itertools

def product_(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

if __name__ == "__main__":
	test_dict = {1: [1, 2], 2: [4, 5, 6], 4:[7,8,9], 3:[]}
	categories = [(k,v) for k,v in test_dict.iteritems() if v]
	prod = list(product_(*(t[1] for t in categories)))
	print(len(prod), prod
	print(categories
	# for k0, v0 in test_dict.values()[0]:
	# 	for k1,v1 in test_dict.iteritems()[1]:
	# 		for k2,v2 in test_dict.iteritems()[2]:
	# 			total_paths.append([k0, k1, k2])

	# total_paths = []
	# root = 0
	# previous_children = []
	# for key in reversed(keys_sorted):
	# 	values = test_dict[key]
	# 	for value in values:
	#
	# 		value.children = previous_children
	# 	previous_children = values
	# tree = Node(root, previous_children)
