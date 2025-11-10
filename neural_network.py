"""
Rede Neural MLP de 2 camadas para o Jogo da Velha
Apenas propagação forward (sem backpropagation)
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=9, hidden_size=18, output_size=9):
        """
        Inicializa a rede neural MLP com 2 camadas
        
        Args:
            input_size: Tamanho da entrada (9 células do tabuleiro)
            hidden_size: Tamanho da camada oculta
            output_size: Tamanho da saída (9 possíveis jogadas)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Número total de pesos
        self.weights1_size = input_size * hidden_size
        self.bias1_size = hidden_size
        self.weights2_size = hidden_size * output_size
        self.bias2_size = output_size
        self.total_weights = self.weights1_size + self.bias1_size + self.weights2_size + self.bias2_size
        
        # Inicializa pesos aleatoriamente
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.5
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.5
        self.bias2 = np.zeros((1, output_size))
    
    def set_weights(self, weights_array):
        """
        Define os pesos da rede a partir de um array linear (cromossomo)
        
        Args:
            weights_array: Array com todos os pesos da rede
        """
        idx = 0
        
        # Weights1
        self.weights1 = weights_array[idx:idx + self.weights1_size].reshape(self.input_size, self.hidden_size)
        idx += self.weights1_size
        
        # Bias1
        self.bias1 = weights_array[idx:idx + self.bias1_size].reshape(1, self.hidden_size)
        idx += self.bias1_size
        
        # Weights2
        self.weights2 = weights_array[idx:idx + self.weights2_size].reshape(self.hidden_size, self.output_size)
        idx += self.weights2_size
        
        # Bias2
        self.bias2 = weights_array[idx:idx + self.bias2_size].reshape(1, self.output_size)
    
    def get_weights(self):
        """
        Retorna todos os pesos da rede em um array linear
        
        Returns:
            Array numpy com todos os pesos
        """
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten()
        ])
    
    def sigmoid(self, x):
        """Função de ativação sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Função de ativação tangente hiperbólica"""
        return np.tanh(x)
    
    def relu(self, x):
        """Função de ativação ReLU"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """
        Propagação forward pela rede
        
        Args:
            x: Input (tabuleiro codificado)
            
        Returns:
            Output da rede (probabilidades para cada jogada)
        """
        # Garante que x seja um array 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Camada oculta
        self.hidden = self.tanh(np.dot(x, self.weights1) + self.bias1)
        
        # Camada de saída
        self.output = np.dot(self.hidden, self.weights2) + self.bias2
        
        return self.output.flatten()
    
    def predict(self, board, player=1):
        """
        Faz uma predição para o tabuleiro atual a partir da perspectiva do jogador informado
        
        Args:
            board: Tabuleiro do jogo (lista ou array)
            player: Jogador que utilizará a rede (1 ou 2)
            
        Returns:
            Scores para cada jogada possível
        """
        board_array = np.array(board, dtype=int)
        board_normalized = np.zeros_like(board_array, dtype=float)
        board_normalized[board_array == player] = 1.0
        board_normalized[board_array == 0] = 0.0
        board_normalized[(board_array != 0) & (board_array != player)] = -1.0

        output = self.forward(board_normalized)
        return output
