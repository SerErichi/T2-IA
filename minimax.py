"""
Implementação do algoritmo Minimax com modos de dificuldade
"""
import random
from copy import deepcopy

class Minimax:
    def __init__(self, difficulty='hard'):
        """
        Inicializa o Minimax
        
        Args:
            difficulty: 'medium' (50% minimax, 50% aleatório) ou 'hard' (100% minimax)
        """
        self.difficulty = difficulty
        
    def set_difficulty(self, difficulty):
        """Define a dificuldade"""
        self.difficulty = difficulty
    
    def get_best_move(self, game, player):
        """
        Retorna a melhor jogada para o jogador
        
        Args:
            game: Instância do TicTacToe
            player: Jogador (1 ou 2)
            
        Returns:
            Melhor posição para jogar
        """
        # Modo médio: 50% aleatório
        if self.difficulty == 'medium' and random.random() < 0.5:
            available_moves = game.get_available_moves()
            return random.choice(available_moves) if available_moves else None
        
        # Modo difícil ou 50% do médio: usa minimax
        best_score = float('-inf')
        best_move = None
        
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return None
        
        for move in available_moves:
            # Simula a jogada
            game_copy = deepcopy(game)
            game_copy.make_move(move, player)
            
            # Avalia a jogada
            score = self.minimax(game_copy, 0, False, player)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, game, depth, is_maximizing, player):
        """
        Algoritmo Minimax recursivo
        
        Args:
            game: Estado do jogo
            depth: Profundidade atual
            is_maximizing: Se é o turno do maximizador
            player: Jogador original (que estamos otimizando)
            
        Returns:
            Score da jogada
        """
        winner = game.check_winner()
        
        # Estados terminais
        if winner == player:
            return 10 - depth  # Vitória (prefere vitórias mais rápidas)
        elif winner == (3 - player):
            return depth - 10  # Derrota (prefere derrotas mais lentas)
        elif winner == -1:
            return 0  # Empate
        
        available_moves = game.get_available_moves()
        
        if is_maximizing:
            max_score = float('-inf')
            for move in available_moves:
                game_copy = deepcopy(game)
                game_copy.make_move(move, player)
                score = self.minimax(game_copy, depth + 1, False, player)
                max_score = max(max_score, score)
            return max_score
        else:
            min_score = float('inf')
            opponent = 3 - player
            for move in available_moves:
                game_copy = deepcopy(game)
                game_copy.make_move(move, opponent)
                score = self.minimax(game_copy, depth + 1, True, player)
                min_score = min(min_score, score)
            return min_score
