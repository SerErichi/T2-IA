"""
Script de teste para validar todos os componentes do sistema
"""
import numpy as np
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax
from genetic_algorithm import GeneticAlgorithm

def test_neural_network():
    """Testa a rede neural"""
    print("Testando Rede Neural...")
    
    nn = NeuralNetwork()
    
    # Testa get_weights e set_weights
    weights = nn.get_weights()
    assert len(weights) == nn.total_weights, "Tamanho dos pesos incorreto"
    
    # Testa forward
    board = [0, 1, 2, 0, 1, 0, 0, 0, 2]
    output = nn.predict(board)
    assert len(output) == 9, "Output deve ter 9 valores"
    
    print("✓ Rede Neural OK")

def test_tic_tac_toe():
    """Testa o jogo da velha"""
    print("Testando Jogo da Velha...")
    
    game = TicTacToe()
    
    # Testa reset
    game.reset()
    assert game.board == [0] * 9, "Reset falhou"
    assert game.current_player == 1, "Jogador inicial deve ser 1"
    
    # Testa make_move
    assert game.make_move(0, 1) == True, "Jogada válida falhou"
    assert game.board[0] == 1, "Jogada não foi registrada"
    assert game.make_move(0, 2) == False, "Jogada inválida foi aceita"
    
    # Testa get_available_moves
    available = game.get_available_moves()
    assert 0 not in available, "Posição ocupada está disponível"
    assert len(available) == 8, "Número incorreto de jogadas disponíveis"
    
    # Testa check_winner - vitória horizontal
    game.reset()
    game.board = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert game.check_winner() == 1, "Vitória horizontal não detectada"
    
    # Testa check_winner - vitória vertical
    game.reset()
    game.board = [2, 0, 0, 2, 0, 0, 2, 0, 0]
    assert game.check_winner() == 2, "Vitória vertical não detectada"
    
    # Testa check_winner - vitória diagonal
    game.reset()
    game.board = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert game.check_winner() == 1, "Vitória diagonal não detectada"
    
    # Testa check_winner - empate
    game.reset()
    game.board = [1, 2, 1, 2, 1, 2, 2, 1, 2]
    assert game.check_winner() == -1, "Empate não detectado"
    
    print("✓ Jogo da Velha OK")

def test_minimax():
    """Testa o Minimax"""
    print("Testando Minimax...")
    
    minimax_hard = Minimax(difficulty='hard')
    minimax_medium = Minimax(difficulty='medium')
    
    # Testa situação de vitória iminente
    game = TicTacToe()
    game.board = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    move = minimax_hard.get_best_move(game, 1)
    assert move == 2, "Minimax deveria bloquear/ganhar na posição 2"
    
    # Testa situação de bloqueio
    game.board = [2, 2, 0, 0, 0, 0, 0, 0, 0]
    move = minimax_hard.get_best_move(game, 1)
    assert move == 2, "Minimax deveria bloquear na posição 2"
    
    print("✓ Minimax OK")

def test_genetic_algorithm():
    """Testa o Algoritmo Genético"""
    print("Testando Algoritmo Genético...")
    
    ga = GeneticAlgorithm(population_size=10)
    
    # Testa inicialização da população
    assert len(ga.population) == 10, "População não inicializada corretamente"
    assert len(ga.population[0]) == ga.chromosome_size, "Tamanho do cromossomo incorreto"
    
    # Testa evaluate_fitness
    chromosome = ga.population[0]
    fitness = ga.evaluate_fitness(chromosome, minimax_difficulty='medium', num_games=2)
    assert isinstance(fitness, (int, float)), "Fitness deve ser numérico"
    
    # Testa tournament_selection
    ga.fitness_scores = list(range(10))  # Fitness 0 a 9
    selected = ga.tournament_selection()
    assert len(selected) == ga.chromosome_size, "Cromossomo selecionado com tamanho errado"
    
    # Testa arithmetic_crossover
    parent1 = np.array([1.0, 2.0, 3.0])
    parent2 = np.array([4.0, 5.0, 6.0])
    child1, child2 = ga.arithmetic_crossover(parent1, parent2)
    assert len(child1) == 3 and len(child2) == 3, "Filhos com tamanho errado"
    
    # Testa gaussian_mutation
    chromosome = np.array([1.0, 2.0, 3.0])
    mutated = ga.gaussian_mutation(chromosome)
    assert len(mutated) == 3, "Cromossomo mutado com tamanho errado"
    
    print("✓ Algoritmo Genético OK")

def test_integration():
    """Teste de integração - jogo completo"""
    print("Testando Integração (jogo completo)...")
    
    game = TicTacToe()
    minimax = Minimax(difficulty='hard')
    nn = NeuralNetwork()
    
    # Jogo entre NN e Minimax
    moves_count = 0
    max_moves = 9
    
    while not game.is_game_over() and moves_count < max_moves:
        if game.current_player == 1:
            # NN joga
            output = nn.predict(game.board)
            available = game.get_available_moves()
            
            if available:
                masked = np.full(9, float('-inf'))
                for m in available:
                    masked[m] = output[m]
                move = np.argmax(masked)
                
                if move in available:
                    game.make_move(move, 1)
                    moves_count += 1
        else:
            # Minimax joga
            move = minimax.get_best_move(game, 2)
            if move is not None:
                game.make_move(move, 2)
                moves_count += 1
    
    winner = game.check_winner()
    assert winner in [1, 2, -1], "Resultado do jogo inválido"
    
    print("✓ Integração OK")

def main():
    """Executa todos os testes"""
    print("="*60)
    print("EXECUTANDO TESTES DO SISTEMA")
    print("="*60 + "\n")
    
    try:
        test_neural_network()
        test_tic_tac_toe()
        test_minimax()
        test_genetic_algorithm()
        test_integration()
        
        print("\n" + "="*60)
        print("✓✓✓ TODOS OS TESTES PASSARAM ✓✓✓")
        print("="*60)
        print("\nO sistema está funcionando corretamente!")
        print("Execute 'python main.py --mode gui' para iniciar a interface.\n")
        
    except AssertionError as e:
        print(f"\n✗ TESTE FALHOU: {str(e)}")
    except Exception as e:
        print(f"\n✗ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
