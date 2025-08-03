'''
	Parameters for DeepStack.
'''
import os
import numpy as np

def get_bet_sizing(min_val, max_val, num):
    return list(np.linspace(min_val, max_val, num))

class Parameters():
	def __init__(self):
		# the tensor datatype used for storing DeepStack's internal data
		self.dtype = np.float32
		self.int_dtype = np.int32
		# cached results path (caching only first street)
		self.cache_path = './data/cache/'
		# self.cache_path = r'D:\Datasets\Pystack\cache'
		# GAME INFORMATION
		# list of pot-scaled bet sizes to use in tree
		self.bet_sizing = {
			'preflop': [1,2,4,8],
			'flop': [1,2,4,8],
			'turn': [1,2,4,8],
			'river': [1,2,4,8,16]
		}
		# self.bet_sizing = {
		# 	'preflop': [0.5, 1],  # 模拟 2bb 和 4bb，加注层级足够覆盖 open 和 3bet
		# 	'flop': [0.5, 1],  # 小、中、大下注（覆盖 Cbet, semi-bluff, polar value） [0.33, 0.66, 1]
		# 	'turn': [0.66, 1.25],  # 中/大注，适合构建压力
		# 	'river': [0.5, 1, 2]  # blocker bluff / std value / polar shove
		# }
		# the size of the game's ante, in chips
		self.ante = 2
		self.sb = self.ante // 2
		self.bb = self.ante
		# the size of each player's stack, in chips
		self.stack = 200
		# NEURAL NETWORK
		self.XLA = True
		# path to the neural net model
		self.model_path = './data/Models/'
		# self.model_path = r'D:\Datasets\Pystack\models'
		# self.model_filename = 'weights.{epoch:02d}-{val_loss:.2f}' # show epoch and loss on filename
		self.model_filename ='weights' # without ending
		# the neural net architecture
		self.num_neurons = [500,500,500,500] # must be size of num_layers
		self.learning_rate = 1e-4
		self.batch_size = 1024  # 原为1024
		self.num_epochs = 50 
		# how often to save the model during training
		self.save_epoch = 2
		# how many epochs to train for
		self.epoch_count = 10
		# TF RECORDS
		self.tfrecords_batch_size = 64  # ~200MB   原为1024*10
		# DATA GENERATION
		# path to the solved poker situation data used to train the neural net
		self.data_path = './data/TrainSamples/'
		# self.data_path = r'D:\Datasets\Pystack\NoLimitTexasHoldem'
		# the number of iterations that DeepStack runs CFR for
		self.cfr_iters = 300 # 原为300
		# the number of preliminary CFR iterations which DeepStack doesn't
		# factor into the average strategy (included in cfr_iters)
		self.cfr_skip_iters = 0 # 原为0
		# the number of starting iters used on approximating leaf nodes
		# after these iterations next street's root nodes are approximated and averaged
		# no need for 'river', because you get values from leaf nodes anyway (using terminal equity)
		self.leaf_nodes_iterations = {
			'preflop':300,
			'flop':300,
			'turn':300
		}
		# how many poker situations are solved simultaneously during
		# data generation
		self.gen_batch_size = 100
		# TOTAL SITUATIONS = different_boards x batch_size
		# how many files to create (single element = ~22kB)
		self.gen_num_files = 1 #原为1





arguments = Parameters()
