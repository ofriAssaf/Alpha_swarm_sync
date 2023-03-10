import collections
consts = collections.namedtuple('parameters',
                                ['grid_size',
                                 'time_to_dig',
                                 'time_to_shovel',
                                 'num_mines',
                                 'view',
                                 'num_navigators',
                                 'num_diggers',
                                 'num_shovels',
                                 'mine_value',
                                 'pad_width',
                                 'N',
                                 'mine_val_finished',
                                 'starting_pos',
                                 'mine_val_digger_finished'])

vals = consts(grid_size= 32,
              time_to_dig=0.6,
              time_to_shovel=0.3,
              num_mines=10,
              view=5,
              num_navigators=2,
              num_diggers=2,
              num_shovels=2,
              mine_value=-5,
              pad_width=2,
              N=5,
              mine_val_finished=0,
              starting_pos=[8,9],
              mine_val_digger_finished=-10)