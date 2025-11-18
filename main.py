"""
Script principal para executar o sistema de treinamento e jogo
Pode ser usado via linha de comando ou interface gráfica
"""
import argparse
import numpy as np
from trainer import Trainer, test_network_accuracy
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax

def train_mode(args):
    """Modo de treinamento via linha de comando"""
    print("Iniciando treinamento via linha de comando...")
    
    trainer = Trainer(
        population_size=args.population,
        mutation_rate=args.mutation
    )
    
    difficulty_schedule = {
        0: 'medium',
        args.generations // 2: 'hard'
    }
    
    best_weights = trainer.train(
        max_generations=args.generations,
        convergence_patience=20,
        difficulty_schedule=difficulty_schedule,
        verbose=True
    )
    
    # Salva os pesos
    np.save('best_weights.npy', best_weights)
    print(f"\nPesos salvos em: best_weights.npy")
    
    # Testa acurácia
    if args.test:
        print("\nTestando acurácia...")
        test_network_accuracy(best_weights, num_games=100, verbose=True)

def play_mode(args):
    """Modo de jogo via linha de comando"""
    if args.opponent == 'nn':
        try:
            weights = np.load('best_weights.npy')
            nn = NeuralNetwork()
            nn.set_weights(weights)
            print("Rede neural carregada!")
        except FileNotFoundError:
            print("Arquivo de pesos não encontrado. Treine a rede primeiro!")
            return
    
    game = TicTacToe()
    minimax = Minimax(difficulty='hard')
    
    print("\n" + "="*60)
    print("JOGO DA VELHA")
    print("="*60)
    print("Você é X (jogador 1)")
    print("Posições: 0-8 (linhas da esquerda para direita, cima para baixo)")
    print("="*60 + "\n")
    
    while not game.is_game_over():
        game.print_board()
        
        if game.current_player == 1:
            # Turno do humano
            while True:
                try:
                    move = int(input("Sua jogada (0-8): "))
                    if move >= 0 and move <= 8 and game.board[move] == 0:
                        game.make_move(move, 1)
                        break
                    else:
                        print("Jogada inválida! Tente novamente.")
                except (ValueError, IndexError):
                    print("Entrada inválida! Digite um número entre 0 e 8.")
        else:
            # Turno da IA
            if args.opponent == 'minimax':
                move = minimax.get_best_move(game, 2)
                print(f"\nMinimax jogou na posição: {move}")
            else:  # nn
                output = nn.predict(game.board, player=2)
                available_moves = game.get_available_moves()
                
                masked_output = np.full(9, float('-inf'))
                for m in available_moves:
                    masked_output[m] = output[m]
                
                move = np.argmax(masked_output)
                print(f"\nRede Neural jogou na posição: {move}")
            
            if move is not None:
                game.make_move(move, 2)
    
    game.print_board()
    
    winner = game.check_winner()
    if winner == 1:
        print("Você venceu!")
    elif winner == 2:
        print("A IA venceu!")
    else:
        print("Empate!")

def gui_mode():
    """Inicia a interface gráfica"""
    from gui import main
    main()

def test_mode():
    """Testa a rede neural treinada"""
    try:
        weights = np.load('best_weights.npy')
        print("Pesos carregados com sucesso!")
    except FileNotFoundError:
        print("Arquivo de pesos não encontrado. Treine a rede primeiro!")
        return
    
    test_network_accuracy(weights, num_games=100, verbose=True)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Jogo da Velha com IA usando Algoritmo Genético'
    )
    
    parser.add_argument('--mode', type=str, default='gui',
                       choices=['train', 'play', 'gui', 'test'],
                       help='Modo de execução (train, play, gui, test)')
    
    parser.add_argument('--population', type=int, default=30,
                       help='Tamanho da população (modo train)')
    
    parser.add_argument('--generations', type=int, default=50,
                       help='Número máximo de gerações (modo train)')
    
    parser.add_argument('--mutation', type=float, default=0.1,
                       help='Taxa de mutação (modo train)')
    
    parser.add_argument('--test', action='store_true',
                       help='Testar acurácia após treinamento')
    
    parser.add_argument('--opponent', type=str, default='minimax',
                       choices=['minimax', 'nn'],
                       help='Oponente no modo play (minimax ou nn)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'play':
        play_mode(args)
    elif args.mode == 'gui':
        gui_mode()
    elif args.mode == 'test':
        test_mode()

if __name__ == "__main__":
    main()
