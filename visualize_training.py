"""
Script para visualizar a evolução do treinamento
"""
import numpy as np
import matplotlib.pyplot as plt
from trainer import Trainer

def plot_training_history(history):
    """
    Plota o histórico de treinamento
    
    Args:
        history: Lista de dicionários com estatísticas de cada geração
    """
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    worst_fitness = [h['worst_fitness'] for h in history]
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico de fitness
    plt.subplot(2, 1, 1)
    plt.plot(generations, best_fitness, 'g-', label='Melhor Fitness', linewidth=2)
    plt.plot(generations, avg_fitness, 'b-', label='Fitness Médio', linewidth=2)
    plt.plot(generations, worst_fitness, 'r-', label='Pior Fitness', linewidth=2)
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Fitness ao Longo das Gerações')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de desvio padrão
    plt.subplot(2, 1, 2)
    std_fitness = [h['std_fitness'] for h in history]
    plt.plot(generations, std_fitness, 'purple', linewidth=2)
    plt.xlabel('Geração')
    plt.ylabel('Desvio Padrão do Fitness')
    plt.title('Diversidade da População (Desvio Padrão)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_evolution.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo em: training_evolution.png")
    plt.show()

def main():
    """Treina e plota a evolução"""
    print("Treinando rede neural...")
    print("Este processo pode demorar alguns minutos...\n")
    
    trainer = Trainer(population_size=30, mutation_rate=0.1)
    
    difficulty_schedule = {
        0: 'medium',
        25: 'hard'
    }
    
    best_weights = trainer.train(
        max_generations=50,
        convergence_patience=20,
        difficulty_schedule=difficulty_schedule,
        verbose=True
    )
    
    # Salva os pesos
    np.save('best_weights.npy', best_weights)
    print(f"\nPesos salvos em: best_weights.npy")
    
    # Plota a evolução
    history = trainer.get_training_history()
    plot_training_history(history)
    
    # Testa a rede
    from trainer import test_network_accuracy
    print("\nTestando acurácia da rede...")
    test_network_accuracy(best_weights, num_games=100, verbose=True)

if __name__ == "__main__":
    main()
