"""
Versão MELHORADA do Algoritmo Genético com Mutação Adaptativa
Demonstração de como otimizar o AG para melhor desempenho
"""
import numpy as np
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax
import random

class ImprovedGeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8,
                 elite_size=5, tournament_size=3, adaptive_mutation=True):
        """
        Algoritmo Genético Melhorado com Mutação Adaptativa
        
        Args:
            population_size: Tamanho da população
            mutation_rate: Taxa de mutação inicial (será adaptada)
            crossover_rate: Taxa de cruzamento
            elite_size: Número de melhores indivíduos preservados
            tournament_size: Tamanho do torneio para seleção
            adaptive_mutation: Se True, usa mutação adaptativa
        """
        self.population_size = population_size
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.adaptive_mutation = adaptive_mutation
        
        # Cria uma rede neural para determinar o tamanho dos cromossomos
        nn = NeuralNetwork()
        self.chromosome_size = nn.total_weights
        
        # Inicializa população
        self.population = []
        self.fitness_scores = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.mutation_rate_history = []
        self.generation = 0
        self.max_generations = 100  # Será definido no train
        
        self.initialize_population()
        
    def initialize_population(self):
        """Inicializa a população com cromossomos aleatórios"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.random.randn(self.chromosome_size) * 0.5
            self.population.append(chromosome)
    
    def calculate_population_diversity(self):
        """
        Calcula a diversidade genética da população
        Usa a média do desvio padrão de cada gene
        """
        pop_array = np.array(self.population)
        diversity = np.mean(np.std(pop_array, axis=0))
        return diversity
    
    def update_mutation_rate(self):
        """
        Atualiza a taxa de mutação adaptativamente baseado em:
        1. Progresso da evolução (geração atual)
        2. Diversidade genética da população
        3. Melhoria recente no fitness
        """
        if not self.adaptive_mutation:
            return  # Mantém taxa fixa
        
        # Fator 1: Decaimento baseado na geração (exploração -> refinamento)
        progress = min(1.0, self.generation / self.max_generations)
        decay_factor = 1.0 - (0.7 * progress)  # Reduz até 70%
        
        # Fator 2: Ajuste baseado na diversidade
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        if diversity < 0.1:  # Baixa diversidade - aumenta mutação
            diversity_factor = 1.5
        elif diversity > 0.5:  # Alta diversidade - reduz mutação
            diversity_factor = 0.8
        else:
            diversity_factor = 1.0
        
        # Fator 3: Ajuste baseado na estagnação
        stagnation_factor = 1.0
        if len(self.best_fitness_history) >= 10:
            recent_improvement = (self.best_fitness_history[-1] - 
                                self.best_fitness_history[-10])
            if recent_improvement < 1.0:  # Pouca melhoria - aumenta mutação
                stagnation_factor = 1.3
        
        # Combina os fatores
        self.mutation_rate = (self.initial_mutation_rate * 
                             decay_factor * 
                             diversity_factor * 
                             stagnation_factor)
        
        # Limita entre 0.01 e 0.3
        self.mutation_rate = max(0.01, min(0.3, self.mutation_rate))
        
        self.mutation_rate_history.append(self.mutation_rate)
    
    def adaptive_gaussian_mutation(self, chromosome):
        """
        Mutação gaussiana ADAPTATIVA
        - Taxa de mutação varia conforme a evolução
        - Desvio padrão adaptativo baseado na diversidade
        """
        mutated = chromosome.copy()
        
        # Desvio padrão adaptativo
        diversity = self.calculate_population_diversity()
        if diversity > 0.5:
            std_dev = 0.5  # Alta diversidade -> mutações maiores
        elif diversity > 0.2:
            std_dev = 0.3  # Média diversidade -> mutações médias
        else:
            std_dev = 0.1  # Baixa diversidade -> mutações sutis
        
        # Aplica mutação
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, std_dev)
        
        return mutated
    
    def evaluate_fitness(self, chromosome, minimax_difficulty='hard', num_games=5):
        """
        Avalia a aptidão de um cromossomo jogando contra o Minimax
        (Mesmo código do original, mantido para compatibilidade)
        """
        nn = NeuralNetwork()
        nn.set_weights(chromosome)
        minimax = Minimax(difficulty=minimax_difficulty)
        
        total_score = 0
        
        for _ in range(num_games):
            game = TicTacToe()
            game.reset()
            
            nn_player = 1
            minimax_player = 2
            invalid_moves = 0
            moves_made = 0
            
            while not game.is_game_over():
                current_player = game.current_player
                
                if current_player == nn_player:
                    output = nn.predict(game.board)
                    available_moves = game.get_available_moves()
                    
                    if not available_moves:
                        break
                    
                    masked_output = np.full(9, float('-inf'))
                    for move in available_moves:
                        masked_output[move] = output[move]
                    
                    move = np.argmax(masked_output)
                    
                    if move in available_moves:
                        game.make_move(move, nn_player)
                        moves_made += 1
                    else:
                        invalid_moves += 1
                        move = random.choice(available_moves)
                        game.make_move(move, nn_player)
                else:
                    move = minimax.get_best_move(game, minimax_player)
                    if move is not None:
                        game.make_move(move, minimax_player)
            
            winner = game.check_winner()
            
            if winner == nn_player:
                total_score += 100
            elif winner == minimax_player:
                total_score -= 50
            elif winner == -1:
                total_score += 50
            
            total_score -= invalid_moves * 20
            total_score += moves_made * 2
        
        return total_score / num_games
    
    def evaluate_population(self, minimax_difficulty='hard', num_games=5):
        """Avalia toda a população"""
        self.fitness_scores = []
        for chromosome in self.population:
            fitness = self.evaluate_fitness(chromosome, minimax_difficulty, num_games)
            self.fitness_scores.append(fitness)
        
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
        """Cruzamento aritmético para valores reais"""
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    
    def evolve(self):
        """Executa uma geração do algoritmo genético"""
        # Atualiza taxa de mutação adaptativamente
        self.update_mutation_rate()
        
        new_population = []
        
        # Elitismo
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Gera o restante
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Usa mutação adaptativa
            child1 = self.adaptive_gaussian_mutation(child1)
            child2 = self.adaptive_gaussian_mutation(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def get_best_chromosome(self):
        """Retorna o melhor cromossomo"""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy()
    
    def get_statistics(self):
        """Retorna estatísticas detalhadas"""
        diversity = self.calculate_population_diversity()
        
        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores),
            'diversity': diversity,
            'mutation_rate': self.mutation_rate
        }
    
    def has_converged(self, patience=20, threshold=1e-3):
        """Verifica se o algoritmo convergiu"""
        if len(self.best_fitness_history) < patience:
            return False
        
        recent_best = self.best_fitness_history[-patience:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < threshold


# ============================================================================
# COMPARAÇÃO: Execute para ver a diferença!
# ============================================================================

def compare_algorithms():
    """
    Compara AG original vs AG melhorado
    Execute: python genetic_algorithm_improved.py
    """
    print("="*70)
    print("COMPARAÇÃO: AG ORIGINAL vs AG MELHORADO (Mutação Adaptativa)")
    print("="*70)
    
    from genetic_algorithm import GeneticAlgorithm
    import time
    
    # Configuração
    pop_size = 20
    max_gen = 30
    
    print(f"\nConfigurações:")
    print(f"  População: {pop_size}")
    print(f"  Gerações: {max_gen}")
    print(f"  Jogos por avaliação: 3")
    print("\n" + "="*70)
    
    # AG Original
    print("\n🔵 TREINANDO AG ORIGINAL (mutação fixa)...")
    ga_original = GeneticAlgorithm(population_size=pop_size, mutation_rate=0.1)
    ga_original.max_generations = max_gen
    
    start = time.time()
    for gen in range(max_gen):
        ga_original.evaluate_population(minimax_difficulty='medium', num_games=3)
        if gen < max_gen - 1:
            ga_original.evolve()
    time_original = time.time() - start
    
    # AG Melhorado
    print("\n🟢 TREINANDO AG MELHORADO (mutação adaptativa)...")
    ga_improved = ImprovedGeneticAlgorithm(population_size=pop_size, 
                                          mutation_rate=0.1,
                                          adaptive_mutation=True)
    ga_improved.max_generations = max_gen
    
    start = time.time()
    for gen in range(max_gen):
        ga_improved.evaluate_population(minimax_difficulty='medium', num_games=3)
        if gen < max_gen - 1:
            ga_improved.evolve()
    time_improved = time.time() - start
    
    # Resultados
    print("\n" + "="*70)
    print("RESULTADOS FINAIS:")
    print("="*70)
    
    print(f"\n📊 AG ORIGINAL:")
    print(f"   Melhor Fitness: {max(ga_original.fitness_scores):.2f}")
    print(f"   Fitness Médio:  {np.mean(ga_original.fitness_scores):.2f}")
    print(f"   Tempo:          {time_original:.2f}s")
    
    print(f"\n📊 AG MELHORADO:")
    print(f"   Melhor Fitness: {max(ga_improved.fitness_scores):.2f}")
    print(f"   Fitness Médio:  {np.mean(ga_improved.fitness_scores):.2f}")
    print(f"   Tempo:          {time_improved:.2f}s")
    print(f"   Diversidade:    {ga_improved.calculate_population_diversity():.4f}")
    
    print(f"\n📈 MELHORIA:")
    fitness_improvement = max(ga_improved.fitness_scores) - max(ga_original.fitness_scores)
    print(f"   Fitness: {fitness_improvement:+.2f} pontos")
    
    if fitness_improvement > 0:
        print("   ✅ AG Melhorado teve MELHOR desempenho!")
    else:
        print("   ⚠️  Resultados similares (rode mais gerações)")
    
    print("\n" + "="*70)
    print("💡 DICA: Para ver diferença significativa, rode com 50+ gerações")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_algorithms()
