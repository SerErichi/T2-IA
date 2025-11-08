"""
Implementação do Jogo da Velha
"""
import numpy as np
from copy import deepcopy

class TicTacToe:
    def __init__(self):
        """Inicializa um novo jogo"""
        self.board = [0] * 9  # 0: vazio, 1: X, 2: O
        self.current_player = 1  # Sempre começa com X
        
    def reset(self):
        """Reseta o jogo"""
        self.board = [0] * 9
        self.current_player = 1
        
    def get_available_moves(self):
        """Retorna lista de jogadas disponíveis"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position, player=None):
        """
        Faz uma jogada no tabuleiro
        
        Args:
            position: Posição no tabuleiro (0-8)
            player: Jogador (1 ou 2). Se None, usa current_player
            
        Returns:
            True se a jogada foi válida, False caso contrário
        """
        if player is None:
            player = self.current_player
            
        if position < 0 or position > 8 or self.board[position] != 0:
            return False
        
        self.board[position] = player
        self.current_player = 3 - player  # Alterna entre 1 e 2
        return True
    
    def check_winner(self):
        """
        Verifica se há um vencedor
        
        Returns:
            0: Jogo em andamento
            1: X venceu
            2: O venceu
            -1: Empate
        """
        # Combinações vencedoras
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colunas
            [0, 4, 8], [2, 4, 6]              # Diagonais
        ]
        
        # Verifica cada combinação
        for combo in winning_combinations:
            if (self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0):
                return self.board[combo[0]]
        
        # Verifica empate
        if 0 not in self.board:
            return -1
        
        # Jogo em andamento
        return 0
    
    def is_game_over(self):
        """Verifica se o jogo terminou"""
        return self.check_winner() != 0
    
    def get_board_copy(self):
        """Retorna uma cópia do tabuleiro"""
        return deepcopy(self.board)
    
    def set_board(self, board):
        """Define o estado do tabuleiro"""
        self.board = deepcopy(board)
    
    def print_board(self):
        """Imprime o tabuleiro de forma legível"""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("\n")
        for i in range(3):
            row = []
            for j in range(3):
                idx = i * 3 + j
                row.append(symbols[self.board[idx]])
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def get_board_string(self):
        """Retorna uma representação em string do tabuleiro"""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        lines = []
        for i in range(3):
            row = []
            for j in range(3):
                idx = i * 3 + j
                row.append(symbols[self.board[idx]])
            lines.append(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                lines.append("-----------")
        return "\n".join(lines)
