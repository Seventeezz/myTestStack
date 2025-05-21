import numpy as np

from Settings.constants import constants


class CardCombinations():
	def count_last_street_boards(self, street):
		'''
		@param: int :current street/round
		@return int :number of possible last round boards
		'''
		BCC, SC = constants.board_card_count, constants.streets_count
		new_cards = BCC[SC - 1] - BCC[street - 1]
		return constants.card_count ** new_cards  # 使用幂运算代替组合数


	def count_next_street_boards(self, street):
		''' counts the number of boards in next street
		@param: int :current street/round
		@return int :number of all next round boards
		'''
		return constants.card_count


	def count_last_boards_possible_boards(self, street):
		''' counts the number of possible boards if 2 cards where already taken (in players hand)
			the answer will be the same for all player's holding cards
		@param: int :current street/round
		@return int :number of possible last round boards
		'''
		max_cards_in_deck = constants.card_count
		num_cards_to_draw = constants.board_card_count[-1] - constants.board_card_count[street - 1]
		return max_cards_in_deck ** num_cards_to_draw  # 9^需要抽取的数量


	def count_next_boards_possible_boards(self, street):
		''' counts the number of possible boards if 2 cards where already taken (in players hand)
			the answer will be the same for all player's holding cards
		@param: int :current street/round
		@return int :number of possible next round boards
		'''
		num_cards_to_draw = constants.board_card_count[street] - constants.board_card_count[street - 1]
		return constants.card_count ** num_cards_to_draw  # 直接使用9的幂次




card_combinations = CardCombinations()
