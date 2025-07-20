'''
	Evaluates all possbile player hands for particular board at terminal nodes
'''
import os
import numpy as np
import torch

from Game.card_tools import card_tools
from TerminalEquity.evaluator import evaluator
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_combinations import card_combinations

class TerminalEquity():
	def __init__(self):
		# load preflop matrix
		# 创建单私牌冲突矩阵
		self._block_matrix = self._create_block_matrix()

	def _create_block_matrix(self):
		''' 创建单私牌冲突矩阵（双方私牌不能相同）'''
		HC = constants.hand_count
		out = np.ones([HC, HC], dtype=bool)
		# 对角线为冲突（同一张牌）
		np.fill_diagonal(out, False)
		return out

	def set_board(self, board):
		''' Sets the board cards for the evaluator and creates internal data structures
		@param: [0-5] :vector of board cards (int)
		'''
		self.board, street, HC = board, card_tools.board_to_street(board), constants.hand_count
		# set equity matrix
		if street == 1:
			# 初始化权益矩阵
			self.equity_matrix = np.zeros([HC, HC], dtype=arguments.dtype)

			# 获取所有可能的3张牌组合（可重复）
			next_three_round_boards = card_tools.get_all_round_boards_combinations()  # 形状：[729, 3]
			boards_count = next_three_round_boards.shape[0]

			# 用于存储每个 board 对应的临时权益矩阵
			next_round_equity_matrix = np.zeros((constants.card_count, constants.card_count), dtype=arguments.dtype)

			for board in range(boards_count):
				board_cards = next_three_round_boards[board]
				self.get_last_round_call_matrix(board_cards, next_round_equity_matrix)  # 应该是 in-place 修改
				self.equity_matrix += next_round_equity_matrix
			# 平均化权益矩阵
			self.equity_matrix /= boards_count
		elif street == constants.streets_count:
			self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
			self._set_last_round_equity_matrix(self.equity_matrix, board)
			self._handle_blocking_cards(self.equity_matrix, board)
		elif street == 2 or street == 3:
			self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
			last_round_boards = card_tools.get_last_round_boards(board)
			self._set_transitioning_equity_matrix(self.equity_matrix, last_round_boards, street)
			self._handle_blocking_cards(self.equity_matrix, board)
		else:
			assert(False) # bad street/board
		# set fold matrix
		self.fold_matrix = np.ones([HC,HC], dtype=arguments.dtype)
		# setting cards that block each other to zero
		self._handle_blocking_cards(self.fold_matrix, board)


	def get_equity_matrix(self):
		''' Returns the matrix which gives rewards for any ranges
		@return [I,I] :for nodes in the last betting round, the matrix `A` such
				that for player ranges `x` and `y`, `x'Ay` is the equity for
				the first player when no player folds. For nodes in the first
				betting round, the weighted average of all such possible matrices
		'''
		return self.equity_matrix


	def get_fold_matrix(self):
		''' Returns the matrix which gives equity for any ranges
		@return [I,I] :matrix `B` such that for player
				ranges `x` and `y`, `x'Ay` is the equity
				for the player who doesn't fold
		'''
		return self.fold_matrix


	def get_hand_strengths(self):
		''' Get strengths of all hand combinations (I). The bigger the number is,
			the stronger the hand is for particular board
			(used in GUI app to evaluate stronger hand)
		@return [I] :strength for all hand combinations
		'''
		HC = constants.hand_count
		return np.sum(self.equity_matrix, axis=1)


	def _set_last_round_equity_matrix(self, equity_matrix, board_cards):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds
		@param: [I,I] :matrix that needs to be modified
		@param: [I,I] :board_cards a non-empty vector of board cards
		'''
		HC = constants.hand_count
		# batch eval with only single batch, because its last round
		strength = evaluator.evaluate_board(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([HC,1])
		strength_view_2 = strength.reshape([1,HC])

		equity_matrix[:,:]  = (strength_view_1 > strength_view_2).astype(int)
		equity_matrix[:,:] -= (strength_view_1 < strength_view_2).astype(int)

	def get_last_round_call_matrix(self, board_cards, call_matrix):
		"""
        构建将玩家 range 转化为对决收益 (showdown equity) 的矩阵。

        Parameters:
            board_cards: 一个非空的公共牌向量，形如 [a, b, c]
            call_matrix: 预分配好的 NumPy 数组，用来写入结果，形状为 (card_count, card_count)
        """
		# 计算所有手牌的强度 (胜率)
		strength: np.ndarray = evaluator.evaluate_board(board_cards)  # 返回 shape: (card_count,)

		# 构造两个视图以对比所有手牌之间的胜负关系
		strength_view_1 = strength.reshape(-1, 1)  # shape: (card_count, 1)
		strength_view_2 = strength.reshape(1, -1)  # shape: (1, card_count)

		# 胜负关系计算
		gt_mask = (strength_view_1 > strength_view_2).astype(call_matrix.dtype)
		lt_mask = (strength_view_1 < strength_view_2).astype(call_matrix.dtype)

		# call_matrix = 1 (胜) - 1 (负) = 1, -1, 0
		np.copyto(call_matrix, gt_mask - lt_mask)

		# 处理 blocker（阻断手牌）
		self._handle_blocking_cards(call_matrix, board_cards)

	def _set_transitioning_equity_matrix(self, equity_matrix, last_round_boards, street):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param: [I,I] :matrix that needs to be modified
		@param: [B,5] :all possible combinations in the last round/street
		@param: int   :current round/street
		'''
		HC, num_boards = constants.hand_count, last_round_boards.shape[0]
		BCC, CC = constants.board_card_count, constants.card_count
		# evaluating all possible last round boards
		strength = evaluator.evaluate_board(last_round_boards) # [b,I]
		# strength from player 1 perspective for all the boards and all the card combinations
		strength_view_1 = strength.reshape([num_boards, HC, 1])
		# strength from player 2 perspective
		strength_view_2 = strength.reshape([num_boards, 1, HC])
		#
		player_possible_mask = (strength > 0).astype(int)

		for i in range(num_boards):
			possible_mask = player_possible_mask[i].reshape([1, HC]) * player_possible_mask[i].reshape([HC, 1])
			# handling hand stregths (winning probs)
			matrix_mem = (strength_view_1[i] > strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask
			equity_matrix[:,:] += matrix_mem

			matrix_mem = (strength_view_1[i] < strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask
			equity_matrix[:,:] -= matrix_mem
		# normalize sum
		num_possible_boards = card_combinations.count_last_boards_possible_boards(street)
		equity_matrix[:,:] *= (1 / num_possible_boards)


	def _handle_blocking_cards(self, matrix, board):
		''' Zeroes entries in an equity matrix that correspond to invalid hands.
			A hand is invalid if it shares any cards with the board
		@param: [I,I] :matrix that needs to be modified
		@param: [0-5] :vector of board cards
		'''
		matrix[:,:] *= self._block_matrix





#
