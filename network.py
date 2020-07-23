import networkx as nx 

__all__ = ['Network', 'is_symbol_in_network', 'insert_sequence', 'get_most_likely_sequence']



class Network(nx.DiGraph):
    
    ''' Class to hold sequences of symbols as a graph. '''
    
    def __init__(self):
        super(Network, self).__init__()
    
    def insert_sequence(self, sequence):
        for symbol in sequence.symbols:
            if symbol not in self.nodes() :
                self.add_node(symbol)
    
    
    def add_sequence_edges(self, sequence):
        prev = sequence.symbols[0]
        for symbol in sequence.symbols[1:] :
            if self.has_edge(prev, symbol):
                self[prev][symbol]['weight'] += 1
            else :
                self.add_edge(prev, symbol, weight=1)
            prev = symbol





def is_symbol_in_network(symbol_net, symbol):
    return symbol in symbol_net


def insert_sequence(network, sequence):
    assert sequence.symbols
    network.insert_sequence(sequence)
    network.add_sequence_edges(sequence)

#TODO: implement this
def get_most_likely_sequence(network, symbol):
    '''Return the sequence that most likely follows a given symbol.'''
    pass

