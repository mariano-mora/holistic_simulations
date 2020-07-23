
from os import system
#action_dirs = ['0.65', '0.75', '0.85', '0.95', '1.05']
action_dirs = ['1.05']
rate_dirs = ['0.75', '0.85', '0.95']
#coord_dirs = rate_dirs + ['1.05', '1.15']
coord_dirs = ['0.45','0.55','0.65','0.75', '0.85','0.95', '1.05', '1.15']
frank = 'frank:/homes/mmm31/developing/game_results/infinite_game/population_game/parameters/var_action_coord'
disk = '/Volumes/WD/simulation_results/population_game/var_action_coord_multi/var_action_coord'
action_dir = '0.04'

if __name__=="__main__":
#    for action_dir in action_dirs:
    for coord_dir in coord_dirs:
        for rate_dir in rate_dirs:
            from_ = '{0}/{1}/{2}/{3}/*'.format(frank, action_dir, coord_dir, rate_dir)  
            to_ = '{0}/{1}/{2}/{3}/'.format(disk, action_dir, coord_dir, rate_dir)
            print(from_, to_)
            system('scp -r %s %s' % (from_, to_))
           # from_ =  '{0}/{1}/{2}/{3}/2*'.format(frank, action_dir, coord_dir, rate_dir)
            #system('scp -r %s %s' % (from_, to_))
            
