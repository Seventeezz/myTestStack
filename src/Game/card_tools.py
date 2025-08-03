'''
	A set of tools for basic operations on cards and sets of cards.

	Several of the functions deal with "range vectors", which are probability
	vectors over the set of possible private hands. For Leduc Hold'em,
	each private hand consists of one card.
'''
import itertools

import numpy as np
import torch

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string

class CardTools():
	def __init__(self):
		pass

	def convert_board_to_nn_feature(self, board):
		'''
		@param: [0-5]     :vector of board cards, where card is unique index (int)
		@return [52+4+13] :vector of shape [total cards in deck + suit count + rank count]
		'''
		num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
		max_board_size = constants.board_card_count[-1]
		# init output
		out = np.zeros([num_cards + num_suits + num_ranks], dtype=np.float32)
		if board.ndim == 0 or board.shape[0] == 0: # no cards were placed
			return out
		# 确保卡牌索引在0-8范围内
		assert((board >= 0).all() and (board < num_cards).all()), f"Invalid card indices: {board} must be in range [0,{num_cards})"
		# init vars
		# one_hot_board = np.zeros([num_cards], dtype=np.float32)
		board_hot_board = np.zeros([num_cards], dtype=np.float32) # 9
		suit_counts = np.zeros([num_suits], dtype=np.float32)
		rank_counts = np.zeros([num_ranks], dtype=np.float32)
		# encode cards, so that all ones show what card is placed
		# one_hot_board[ board ] = 1
		# count number of different suits and ranks on board
		for card in board:
			board_hot_board[card] += 1
			suit = card_to_string.card_to_suit(card)
			rank = card_to_string.card_to_rank(card)
			suit_counts[ suit ] += 1
			rank_counts[ rank ] += 1
		# normalize counts
		if len(board) > 0:
			board_hot_board /= max_board_size
			rank_counts /= num_ranks
			suit_counts /= num_suits
		# combine all arrays and return
		out[ :num_cards ] = board_hot_board
		out[ num_cards:num_cards+num_suits ] = suit_counts
		out[ num_cards+num_suits: ] = rank_counts
		return out

	def get_possible_hands_mask(self, board):
		''' 生成合法单私牌掩码（允许与公牌重复）'''
		#暂时只考虑自己
		HC = constants.hand_count
		# 所有单卡私牌均合法（包括与公牌重复的情况）
		out = np.ones([HC], dtype=arguments.int_dtype)
		return out

	# def get_possible_hands_mask(self, board):
	# 	''' Gives the private hands which are valid with a given board.
	# 	@param: [0-5] :vector of board cards, where card is unique index (int)
	# 	@return [I]   :vector with an entry for every possible hand (private card),
	# 			which is `1` if the hand shares no cards with the board and `0` otherwise
	# 	'''
	# 	HC, CC = constants.hand_count, constants.card_count
	# 	out = np.zeros([HC], dtype=arguments.int_dtype)
	# 	if board.ndim == 0 or board.shape[0] == 0:
	# 		out.fill(1)
	# 		return out
	# 	# 生成已用卡牌掩码
	# 	# TODO：需要传入对方私牌信息，排除对方私牌重复
	#
	# 	used_mask = np.zeros(CC, dtype=bool)
	# 	# if opponent_private_card is not None:
	# 	# 	used_mask[opponent_private_card] = True  # 排除对方私牌
	# 	used_mask[board] = True
	# 	# 遍历所有单私牌（直接索引即手牌ID）
	# 	for card in range(CC):
	# 		out[card] = 0 if used_mask[card] else 1
	# 	return out
	# 	# used = np.zeros([CC], dtype=bool)
	# 	# for card in board:
	# 	# 	used[ card ] = 1
	# 	#
	# 	# for card1 in range(CC):
	# 	# 	if not used[card1]:
	# 	# 		for card2 in range(card1+1,CC):
	# 	# 			if not used[card2]:
	# 	# 				hand = [card1, card2]
	# 	# 				hand_index = self.get_hand_index(hand)
	# 	# 				out[ hand_index ] = 1
	# 	#return out


	def same_boards(self, board1, board2):
		''' checks if board1 == board2
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@param: [0-5] :vector of board cards, where card is unique index (int)
		'''
		for card1 in board1:
			found_match = False
			for card2 in board2:
				found_match = True
				break
			if not found_match:
				return False
		return True


	def board_to_street(self, board):
		''' Gives the current betting round based on a board vector
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return int   :current betting round/street
		'''
		BCC, SC = constants.board_card_count, constants.streets_count
		card_count = board.shape[0] if board.ndim != 0 else 0
		for street in range(SC):
			if card_count == BCC[street]:
				return street + 1  # preflop=1, flop=2等
		raise ValueError(f"Invalid board size: {card_count}")
		# if board.ndim == 0 or board.shape[0] == 0:
		# 	return 1
		# else:
		# 	for i in range(SC):
		# 		if board.shape[0] == BCC[i]:
		# 			return i+1


	def _build_boards(self, boards, cur_board, out, card_index, last_index, base_index):
		CC = constants.card_count
		current_length = card_index  # 使用card_index作为当前索引位置

		# 确保new_board数组大小足够
		for new_card in range(CC):
			# 生成新的公牌组合
			new_board = cur_board.copy()
			new_board[current_length] = new_card  # 设置新牌

			# 当达到目标卡牌数量时保存
			if current_length + 1 == last_index:
				# 确保索引在有效范围内
				board_idx = boards[1] - 1
				if board_idx < out.shape[0]:  # 添加范围检查
					out[board_idx] = new_board
					boards[1] += 1
				else:
					# 停止生成，数组已满
					return
			else:
				# 递归生成更长的组合
				self._build_boards(boards, new_board, out,
							   current_length + 1, last_index, base_index)


	def get_next_round_boards(self, board):
		''' Gives all possible sets of board cards for the game.
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return [B,I] :tensor, where B is all possible next round boards
		'''
		BCC, CC = constants.board_card_count, constants.card_count
		street = self.board_to_street(board)

		# 计算可能的牌组合数量
		num_new_cards = BCC[street] - BCC[street-1]
		# 限制生成的牌数量，避免组合爆炸（单牌游戏中）
		MAX_BOARDS = 9999  # 最大生成牌数量
		boards_count = min(MAX_BOARDS, CC ** num_new_cards)

		out = np.zeros([boards_count, BCC[street]], dtype=arguments.int_dtype)
		boards = [out, 1]  # (boards, index)
		cur_board = np.zeros([BCC[street]], dtype=arguments.int_dtype)

		# 复制现有的公牌
		if board.ndim > 0:
			for i in range(board.shape[0]):
				cur_board[i] = board[i]

		# 生成新的公牌组合（传递当前公牌数量作为起始索引）
		self._build_boards(boards, cur_board, out, BCC[street-1], BCC[street], BCC[street-1])
		return out


	def get_last_round_boards(self, board):
		''' Gives all possible sets of board cards for the game.
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return [B,I] :tensor, where B is all possible next round boards
		'''
		BCC, SC = constants.board_card_count, constants.streets_count
		street = self.board_to_street(board)

		# 计算可能的牌组合数量
		num_new_cards = BCC[SC-1] - BCC[street-1]
		# 限制生成的牌数量，避免组合爆炸（单牌游戏中）
		MAX_BOARDS = 100  # 最大生成牌数量
		boards_count = min(MAX_BOARDS, constants.card_count ** num_new_cards)

		out = np.zeros([boards_count, BCC[SC-1]], dtype=arguments.int_dtype)
		boards = [out, 1]  # (boards, index)
		cur_board = np.zeros([BCC[SC-1]], dtype=arguments.int_dtype)

		# 复制现有的公牌
		if board.ndim > 0:
			for i in range(board.shape[0]):
				cur_board[i] = board[i]

		# 生成新的公牌组合（传递当前公牌数量作为起始索引）
		self._build_boards(boards, cur_board, out, BCC[street-1], BCC[SC-1], BCC[street-1])
		return out


	def get_hand_index(self, hand):
		''' Gives a numerical index for a set of hand
		@param: [2] :vector of player private cards, where card is unique index (int)
		@return int :numerical index for the hand (0-1326)
		(first card is always smaller then second!)
		'''
		''' 修改点：单私牌直接返回索引（原计算组合数） '''
		#assert len(hand) == 1  # 新规则下私牌只有1张
		return hand[0]  # 直接返回卡牌索引（0-8）
		# index = 1
		# for i in range(len(hand)):
		# 	index += card_combinations.choose(hand[i], i+1)
		# return index - 1

	def get_all_round_boards_combinations(self):
		# 从 0 到 8 中任取 3 张牌，允许重复（即笛卡尔积）
		all_combinations = list(itertools.product(range(9), repeat=3))  # 共 9^3 = 729 种
		# 转换为 PyTorch 张量
		combinations_tensor = torch.IntTensor(all_combinations)
		return combinations_tensor




card_tools = CardTools()
