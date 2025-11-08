"""
Interface Gráfica para o Jogo da Velha com Rede Neural e Minimax
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from neural_network import NeuralNetwork
from tic_tac_toe import TicTacToe
from minimax import Minimax
from trainer import Trainer, test_network_accuracy
import threading
import os

class TicTacToeGUI:
    def __init__(self, root):
        """Inicializa a interface gráfica"""
        self.root = root
        self.root.title("Jogo da Velha - IA com Algoritmo Genético")
        self.root.geometry("900x700")
        
        # Estado do jogo
        self.game = TicTacToe()
        self.minimax = Minimax()
        self.nn = NeuralNetwork()
        self.trainer = None
        self.trained_weights = None
        
        # Modo de jogo
        self.game_mode = tk.StringVar(value="human_vs_minimax")
        
        # Botões do tabuleiro
        self.buttons = []
        
        # Criar interface
        self.create_widgets()
        
    def create_widgets(self):
        """Cria os widgets da interface"""
        # Frame principal dividido em duas colunas
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Coluna esquerda - Tabuleiro e controles de jogo
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, sticky=(tk.N))
        
        # Título
        title_label = ttk.Label(left_frame, text="Jogo da Velha - IA", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Modo de jogo
        mode_frame = ttk.LabelFrame(left_frame, text="Modo de Jogo", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        modes = [
            ("Humano vs Minimax", "human_vs_minimax"),
            ("Humano vs Rede Neural", "human_vs_nn"),
            ("Treinar Rede Neural", "train_nn")
        ]
        
        for i, (text, mode) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.game_mode, 
                                value=mode, command=self.on_mode_change)
            rb.grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Tabuleiro
        board_frame = ttk.Frame(left_frame)
        board_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        for i in range(9):
            btn = tk.Button(board_frame, text="", font=('Arial', 24, 'bold'),
                          width=5, height=2, command=lambda pos=i: self.on_cell_click(pos))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
            self.buttons.append(btn)
        
        # Botões de controle
        control_frame = ttk.Frame(left_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.new_game_btn = ttk.Button(control_frame, text="Novo Jogo", 
                                       command=self.new_game)
        self.new_game_btn.grid(row=0, column=0, padx=5)
        
        # Status
        self.status_label = ttk.Label(left_frame, text="Selecione um modo de jogo", 
                                     font=('Arial', 12))
        self.status_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Coluna direita - Treinamento e logs
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, sticky=(tk.N, tk.S))
        
        # Configurações de treinamento
        train_frame = ttk.LabelFrame(right_frame, text="Configurações de Treinamento", 
                                     padding="10")
        train_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(train_frame, text="Tamanho da População:").grid(row=0, column=0, 
                                                                   sticky=tk.W, pady=2)
        self.pop_size_var = tk.StringVar(value="30")
        ttk.Entry(train_frame, textvariable=self.pop_size_var, width=10).grid(row=0, 
                                                                               column=1, pady=2)
        
        ttk.Label(train_frame, text="Gerações Máximas:").grid(row=1, column=0, 
                                                               sticky=tk.W, pady=2)
        self.max_gen_var = tk.StringVar(value="50")
        ttk.Entry(train_frame, textvariable=self.max_gen_var, width=10).grid(row=1, 
                                                                              column=1, pady=2)
        
        ttk.Label(train_frame, text="Taxa de Mutação:").grid(row=2, column=0, 
                                                              sticky=tk.W, pady=2)
        self.mutation_var = tk.StringVar(value="0.1")
        ttk.Entry(train_frame, textvariable=self.mutation_var, width=10).grid(row=2, 
                                                                               column=1, pady=2)
        
        self.train_btn = ttk.Button(train_frame, text="Iniciar Treinamento", 
                                    command=self.start_training)
        self.train_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.test_btn = ttk.Button(train_frame, text="Testar Acurácia", 
                                   command=self.test_accuracy, state=tk.DISABLED)
        self.test_btn.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Log de treinamento
        log_frame = ttk.LabelFrame(right_frame, text="Log de Treinamento", padding="10")
        log_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=50, height=25, 
                                                  font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar pesos de redimensionamento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
    def on_mode_change(self):
        """Callback quando o modo de jogo muda"""
        mode = self.game_mode.get()
        
        if mode == "human_vs_minimax":
            self.status_label.config(text="Você é X - Clique em uma célula para jogar")
            self.new_game()
        elif mode == "human_vs_nn":
            if self.trained_weights is None:
                messagebox.showwarning("Aviso", 
                    "Rede neural não treinada! Treine a rede primeiro.")
                self.game_mode.set("human_vs_minimax")
            else:
                self.status_label.config(text="Você é X - Jogue contra a Rede Neural")
                self.new_game()
        elif mode == "train_nn":
            self.status_label.config(text="Modo de Treinamento - Configure e inicie")
            self.disable_board()
    
    def new_game(self):
        """Inicia um novo jogo"""
        self.game.reset()
        self.update_board()
        self.enable_board()
        
        mode = self.game_mode.get()
        if mode == "human_vs_minimax":
            self.status_label.config(text="Seu turno (X)")
        elif mode == "human_vs_nn":
            self.status_label.config(text="Seu turno (X)")
    
    def on_cell_click(self, position):
        """Callback quando uma célula é clicada"""
        mode = self.game_mode.get()
        
        if mode == "train_nn":
            return
        
        # Jogada do humano
        if self.game.board[position] == 0:
            self.game.make_move(position, 1)  # Humano é sempre X (1)
            self.update_board()
            
            if self.game.is_game_over():
                self.handle_game_over()
                return
            
            # Jogada da IA
            self.root.after(500, self.ai_move)
    
    def ai_move(self):
        """Realiza a jogada da IA"""
        mode = self.game_mode.get()
        
        if mode == "human_vs_minimax":
            move = self.minimax.get_best_move(self.game, 2)
            if move is not None:
                self.game.make_move(move, 2)
        elif mode == "human_vs_nn":
            output = self.nn.predict(self.game.board)
            available_moves = self.game.get_available_moves()
            
            if available_moves:
                masked_output = np.full(9, float('-inf'))
                for move in available_moves:
                    masked_output[move] = output[move]
                move = np.argmax(masked_output)
                
                if move in available_moves:
                    self.game.make_move(move, 2)
        
        self.update_board()
        
        if self.game.is_game_over():
            self.handle_game_over()
        else:
            self.status_label.config(text="Seu turno (X)")
    
    def handle_game_over(self):
        """Trata o fim do jogo"""
        winner = self.game.check_winner()
        
        if winner == 1:
            self.status_label.config(text="Você venceu! (X)")
            messagebox.showinfo("Fim de Jogo", "Você venceu!")
        elif winner == 2:
            self.status_label.config(text="IA venceu! (O)")
            messagebox.showinfo("Fim de Jogo", "A IA venceu!")
        elif winner == -1:
            self.status_label.config(text="Empate!")
            messagebox.showinfo("Fim de Jogo", "Empate!")
        
        self.disable_board()
    
    def update_board(self):
        """Atualiza a exibição do tabuleiro"""
        symbols = {0: "", 1: "X", 2: "O"}
        colors = {0: "white", 1: "blue", 2: "red"}
        
        for i, btn in enumerate(self.buttons):
            btn.config(text=symbols[self.game.board[i]], 
                      fg=colors[self.game.board[i]])
    
    def enable_board(self):
        """Habilita o tabuleiro"""
        for btn in self.buttons:
            btn.config(state=tk.NORMAL)
    
    def disable_board(self):
        """Desabilita o tabuleiro"""
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
    
    def log(self, message):
        """Adiciona mensagem ao log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_training(self):
        """Inicia o treinamento em uma thread separada"""
        try:
            pop_size = int(self.pop_size_var.get())
            max_gen = int(self.max_gen_var.get())
            mutation_rate = float(self.mutation_var.get())
        except ValueError:
            messagebox.showerror("Erro", "Valores de configuração inválidos!")
            return
        
        self.log_text.delete(1.0, tk.END)
        self.log("Iniciando treinamento...\n")
        
        self.train_btn.config(state=tk.DISABLED)
        
        # Executa treinamento em thread separada
        thread = threading.Thread(target=self.run_training, 
                                 args=(pop_size, max_gen, mutation_rate))
        thread.start()
    
    def run_training(self, pop_size, max_gen, mutation_rate):
        """Executa o treinamento"""
        self.trainer = Trainer(population_size=pop_size, mutation_rate=mutation_rate)
        
        # Redireciona saída para o log
        class LogRedirector:
            def __init__(self, log_func):
                self.log_func = log_func
            
            def write(self, message):
                if message.strip():
                    self.log_func(message.strip())
            
            def flush(self):
                pass
        
        import sys
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(self.log)
        
        try:
            # Define agenda de dificuldade
            difficulty_schedule = {
                0: 'medium',
                max_gen // 2: 'hard'
            }
            
            self.trained_weights = self.trainer.train(
                max_generations=max_gen,
                convergence_patience=20,
                difficulty_schedule=difficulty_schedule,
                verbose=True
            )
            
            # Carrega pesos na rede
            self.nn.set_weights(self.trained_weights)
            
            self.log("\n" + "="*60)
            self.log("TREINAMENTO CONCLUÍDO COM SUCESSO!")
            self.log("="*60 + "\n")
            
            # Salva pesos
            np.save('best_weights.npy', self.trained_weights)
            self.log("Pesos salvos em: best_weights.npy\n")
            
        except Exception as e:
            self.log(f"\nERRO durante o treinamento: {str(e)}\n")
        finally:
            sys.stdout = old_stdout
            self.train_btn.config(state=tk.NORMAL)
            self.test_btn.config(state=tk.NORMAL)
    
    def test_accuracy(self):
        """Testa a acurácia da rede treinada"""
        if self.trained_weights is None:
            messagebox.showwarning("Aviso", "Nenhuma rede treinada para testar!")
            return
        
        self.log("\nTestando acurácia da rede...\n")
        self.test_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_accuracy_test)
        thread.start()
    
    def run_accuracy_test(self):
        """Executa o teste de acurácia"""
        import sys
        
        class LogRedirector:
            def __init__(self, log_func):
                self.log_func = log_func
            
            def write(self, message):
                if message.strip():
                    self.log_func(message.strip())
            
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(self.log)
        
        try:
            results = test_network_accuracy(self.trained_weights, num_games=100, 
                                           verbose=True)
        except Exception as e:
            self.log(f"\nERRO durante o teste: {str(e)}\n")
        finally:
            sys.stdout = old_stdout
            self.test_btn.config(state=tk.NORMAL)


def main():
    """Função principal"""
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
