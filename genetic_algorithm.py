"""
Algoritmo Genético para evoluir os pesos da Rede Neural
Utiliza operadores para valores reais (cruzamento aritmético e mutação gaussiana)
"""
import numpy as np
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax
import random

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8,
                 elite_size=5, tournament_size=3):
        """
        Inicializa o Algoritmo Genético
        
        Args:
            population_size: Tamanho da população
            mutation_rate: Taxa de mutação
            crossover_rate: Taxa de cruzamento
            elite_size: Número de melhores indivíduos preservados (elitismo)
            tournament_size: Tamanho do torneio para seleção
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Cria uma rede neural para determinar o tamanho dos cromossomos
        nn = NeuralNetwork()
        self.chromosome_size = nn.total_weights
        
        # Inicializa população
        self.population = []
        self.fitness_scores = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation = 0
        
        self.initialize_population()
        
    def initialize_population(self):
        """Inicializa a população com cromossomos aleatórios"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.random.randn(self.chromosome_size) * 0.5
            self.population.append(chromosome)
    
    def evaluate_fitness(self, chromosome, minimax_difficulty='hard', num_games=5):
        """
        Avalia a aptidão de um cromossomo jogando contra o Minimax
        
        Args:
            chromosome: Array de pesos da rede
            minimax_difficulty: Dificuldade do Minimax ('medium' ou 'hard')
            num_games: Número de jogos para avaliar
            
        Returns:
            Score de fitness
        """
        # Cria a rede neural com os pesos do cromossomo
        nn = NeuralNetwork()
        nn.set_weights(chromosome)
        
        # Cria o Minimax
        minimax = Minimax(difficulty=minimax_difficulty)
        
        total_score = 0
        
        for _ in range(num_games):
            game = TicTacToe()
            game.reset()
            
            # A rede sempre joga como X (player 1)
            nn_player = 1
            minimax_player = 2
            
            invalid_moves = 0
            moves_made = 0
            
            while not game.is_game_over():
                current_player = game.current_player
                
                if current_player == nn_player:
                    # Turno da rede neural
                    output = nn.predict(game.board)
                    
                    # Máscara para células disponíveis
                    available_moves = game.get_available_moves()
                    
                    if not available_moves:
                        break
                    
                    # Seleciona a jogada com maior score entre as disponíveis
                    masked_output = np.full(9, float('-inf'))
                    for move in available_moves:
                        masked_output[move] = output[move]
                    
                    move = np.argmax(masked_output)
                    
                    # Tenta fazer a jogada
                    if move in available_moves:
                        game.make_move(move, nn_player)
                        moves_made += 1
                    else:
                        # Jogada inválida (não deveria acontecer com a máscara)
                        invalid_moves += 1
                        # Faz uma jogada aleatória válida
                        move = random.choice(available_moves)
                        game.make_move(move, nn_player)
                else:
                    # Turno do Minimax
                    move = minimax.get_best_move(game, minimax_player)
                    if move is not None:
                        game.make_move(move, minimax_player)
            
            # Calcula o score baseado no resultado
            winner = game.check_winner()
            
            if winner == nn_player:
                # Vitória: +100 pontos
                total_score += 100
            elif winner == minimax_player:
                # Derrota: -50 pontos
                total_score -= 50
            elif winner == -1:
                # Empate: +50 pontos (empate contra Minimax é bom!)
                total_score += 50
            
            # Penalização por jogadas inválidas
            total_score -= invalid_moves * 20
            
            # Bonificação por fazer jogadas (evita redes que não jogam)
            total_score += moves_made * 2
        
        return total_score / num_games
    
    def evaluate_population(self, minimax_difficulty='hard', num_games=5):
        """
        Avalia toda a população
        
        Args:
            minimax_difficulty: Dificuldade do Minimax
            num_games: Número de jogos por indivíduo
        """
        self.fitness_scores = []
        for chromosome in self.population:
            fitness = self.evaluate_fitness(chromosome, minimax_difficulty, num_games)
            self.fitness_scores.append(fitness)
        
        # Registra estatísticas
        best_fitness = max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
    
    def tournament_selection(self):
        """Seleção por torneio"""
        tournament = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament]
        winner_idx = tournament[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def arithmetic_crossover(self, parent1, parent2):
        """
        Cruzamento aritmético para valores reais
        Gera dois filhos: child1 = α*p1 + (1-α)*p2, child2 = α*p2 + (1-α)*p1
        
        Args:
            parent1, parent2: Cromossomos pais
            
        Returns:
            Dois cromossomos filhos
        """
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    
    def blend_crossover(self, parent1, parent2, alpha=0.5):
        """
        BLX-α crossover para valores reais
        
        Args:
            parent1, parent2: Cromossomos pais
            alpha: Parâmetro de extensão
            
        Returns:
            Dois cromossomos filhos
        """
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            child1[i] = np.random.uniform(lower, upper)
            child2[i] = np.random.uniform(lower, upper)
        
        return child1, child2
    
    def gaussian_mutation(self, chromosome):
        """
        Mutação gaussiana para valores reais
        
        Args:
            chromosome: Cromossomo a ser mutado
            
        Returns:
            Cromossomo mutado
        """
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Adiciona ruído gaussiano
                mutated[i] += np.random.normal(0, 0.5)
        return mutated
    
    def evolve(self):
        """Executa uma geração do algoritmo genético"""
        new_population = []
        
        # Elitismo: preserva os melhores indivíduos
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Gera o restante da população
        while len(new_population) < self.population_size:
            # Seleção
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Cruzamento
            if random.random() < self.crossover_rate:
                child1, child2 = self.arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutação
            child1 = self.gaussian_mutation(child1)
            child2 = self.gaussian_mutation(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def get_best_chromosome(self):
        """Retorna o melhor cromossomo da população atual"""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy()
    
    def get_statistics(self):
        """Retorna estatísticas da geração atual"""
        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        }
    
    def has_converged(self, patience=20, threshold=1e-3):
        """
        Verifica se o algoritmo convergiu
        
        Args:
            patience: Número de gerações sem melhoria
            threshold: Threshold de melhoria mínima
            
        Returns:
            True se convergiu, False caso contrário
        """
        if len(self.best_fitness_history) < patience:
            return False
        
        recent_best = self.best_fitness_history[-patience:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < threshold
