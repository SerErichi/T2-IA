"""
Sistema de Treinamento da Rede Neural usando Algoritmo Genético
"""
import numpy as np
from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax
import time

class Trainer:
    def __init__(self, population_size=50, mutation_rate=0.1):
        """
        Inicializa o treinador
        
        Args:
            population_size: Tamanho da população do AG
            mutation_rate: Taxa de mutação
        """
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=0.8,
            elite_size=5,
            tournament_size=3
        )
        
        self.training_history = []
        
    def train(self, max_generations=100, convergence_patience=20, 
              difficulty_schedule=None, verbose=True):
        """
        Treina a rede neural usando o Algoritmo Genético
        
        Args:
            max_generations: Número máximo de gerações
            convergence_patience: Número de gerações sem melhoria para parar
            difficulty_schedule: Dicionário com gerações e dificuldades
                Exemplo: {0: 'medium', 50: 'hard'}
            verbose: Se True, imprime progresso
            
        Returns:
            Melhor cromossomo (pesos da rede)
        """
        if difficulty_schedule is None:
            # Agenda padrão: começa médio e passa para difícil
            difficulty_schedule = {
                0: 'medium',
                max_generations // 2: 'hard'
            }
        
        print("=" * 60)
        print("INICIANDO TREINAMENTO DA REDE NEURAL")
        print("=" * 60)
        print(f"População: {self.ga.population_size}")
        print(f"Taxa de Mutação: {self.ga.mutation_rate}")
        print(f"Taxa de Cruzamento: {self.ga.crossover_rate}")
        print(f"Gerações Máximas: {max_generations}")
        print("=" * 60)
        
        start_time = time.time()
        
        for gen in range(max_generations):
            # Determina a dificuldade atual
            current_difficulty = 'medium'
            for threshold_gen, difficulty in sorted(difficulty_schedule.items()):
                if gen >= threshold_gen:
                    current_difficulty = difficulty
            
            # Avalia a população
            self.ga.evaluate_population(
                minimax_difficulty=current_difficulty,
                num_games=5
            )
            
            # Obtém estatísticas
            stats = self.ga.get_statistics()
            stats['difficulty'] = current_difficulty
            self.training_history.append(stats)
            
            if verbose:
                print(f"\nGeração {stats['generation']} | Dificuldade: {current_difficulty}")
                print(f"  Melhor Fitness: {stats['best_fitness']:.2f}")
                print(f"  Fitness Médio:  {stats['avg_fitness']:.2f}")
                print(f"  Pior Fitness:   {stats['worst_fitness']:.2f}")
                print(f"  Desvio Padrão:  {stats['std_fitness']:.2f}")
            
            # Verifica convergência
            if self.ga.has_converged(patience=convergence_patience):
                print(f"\n{'='*60}")
                print(f"CONVERGÊNCIA ATINGIDA NA GERAÇÃO {gen}")
                print(f"{'='*60}")
                break
            
            # Evolui para próxima geração
            if gen < max_generations - 1:
                self.ga.evolve()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print("TREINAMENTO CONCLUÍDO")
        print(f"{'='*60}")
        print(f"Tempo de Treinamento: {training_time:.2f} segundos")
        print(f"Gerações Executadas: {self.ga.generation}")
        print(f"Melhor Fitness Final: {max(self.ga.fitness_scores):.2f}")
        print(f"{'='*60}\n")
        
        return self.ga.get_best_chromosome()
    
    def get_training_history(self):
        """Retorna o histórico de treinamento"""
        return self.training_history
    
    def save_best_weights(self, filename='best_weights.npy'):
        """Salva os melhores pesos em um arquivo"""
        best_weights = self.ga.get_best_chromosome()
        np.save(filename, best_weights)
        print(f"Pesos salvos em: {filename}")
    
    def load_weights(self, filename='best_weights.npy'):
        """Carrega pesos de um arquivo"""
        weights = np.load(filename)
        return weights


def test_network_accuracy(weights, num_games=100, verbose=True):
    """
    Testa a acurácia da rede neural treinada
    
    Args:
        weights: Pesos da rede treinada
        num_games: Número de jogos para testar
        verbose: Se True, imprime resultados detalhados
        
    Returns:
        Dicionário com estatísticas
    """
    nn = NeuralNetwork()
    nn.set_weights(weights)
    
    minimax = Minimax(difficulty='hard')
    
    wins = 0
    losses = 0
    draws = 0
    invalid_moves_total = 0
    
    for game_num in range(num_games):
        game = TicTacToe()
        game.reset()
        
        nn_player = 1
        minimax_player = 2
        invalid_moves = 0
        
        while not game.is_game_over():
            current_player = game.current_player
            
            if current_player == nn_player:
                output = nn.predict(game.board)
                available_moves = game.get_available_moves()
                
                if not available_moves:
                    break
                
                # Seleciona melhor jogada disponível
                masked_output = np.full(9, float('-inf'))
                for move in available_moves:
                    masked_output[move] = output[move]
                
                move = np.argmax(masked_output)
                
                if move not in available_moves:
                    invalid_moves += 1
                    move = available_moves[0]
                
                game.make_move(move, nn_player)
            else:
                move = minimax.get_best_move(game, minimax_player)
                if move is not None:
                    game.make_move(move, minimax_player)
        
        winner = game.check_winner()
        
        if winner == nn_player:
            wins += 1
        elif winner == minimax_player:
            losses += 1
        elif winner == -1:
            draws += 1
        
        invalid_moves_total += invalid_moves
    
    # Calcula estatísticas
    win_rate = (wins / num_games) * 100
    draw_rate = (draws / num_games) * 100
    loss_rate = (losses / num_games) * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print("TESTE DE ACURÁCIA DA REDE NEURAL")
        print(f"{'='*60}")
        print(f"Jogos Testados: {num_games}")
        print(f"Vitórias: {wins} ({win_rate:.2f}%)")
        print(f"Empates:  {draws} ({draw_rate:.2f}%)")
        print(f"Derrotas: {losses} ({loss_rate:.2f}%)")
        print(f"Jogadas Inválidas Total: {invalid_moves_total}")
        print(f"{'='*60}\n")
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'invalid_moves': invalid_moves_total
    }
