# Jogo da Velha - IA com Algoritmo Genético

Sistema de Inteligência Artificial que usa **Aprendizagem por Reforço** com **Algoritmo Genético** para treinar uma **Rede Neural** a jogar o Jogo da Velha.

## 📋 Descrição do Projeto

Este projeto implementa uma solução completa de IA onde:
- Uma **Rede Neural MLP** de 2 camadas aprende a jogar o Jogo da Velha
- O **Algoritmo Genético** evolui os pesos da rede (sem backpropagation)
- O **Minimax** atua como professor, com dois níveis de dificuldade:
  - **Médio**: 50% jogadas minimax, 50% aleatórias
  - **Difícil**: 100% jogadas minimax
- Interface gráfica completa para jogar e treinar

## 🏗️ Arquitetura

### Rede Neural (MLP)
- **Entrada**: 9 neurônios (tabuleiro 3x3)
- **Camada Oculta**: 18 neurônios (ativação tanh)
- **Saída**: 9 neurônios (uma para cada posição)
- Apenas propagação forward (sem backpropagation)

### Algoritmo Genético
- **Cromossomos**: Pesos da rede neural (valores reais)
- **Seleção**: Elitismo + Torneio
- **Cruzamento**: Aritmético (para valores reais)
- **Mutação**: Gaussiana
- **Função de Aptidão**: Baseada em vitórias, empates, derrotas e jogadas inválidas

### Minimax
- Implementação clássica do algoritmo
- Modos: Médio (50% aleatório) e Difícil (100% minimax)
- Usado para treinar a rede neural

## 🚀 Instalação

### Requisitos
```bash
Python 3.7+
numpy
matplotlib
tkinter (geralmente já vem com Python)
```

### Instalando Dependências
```bash
pip install numpy matplotlib
```

## 💻 Como Usar

### 1. Interface Gráfica (Recomendado)
```bash
python main.py --mode gui
```

Na interface você pode:
- **Jogar contra o Minimax**
- **Treinar a Rede Neural**
- **Jogar contra a Rede Neural treinada**
- **Testar a acurácia da rede**

### 2. Treinar via Linha de Comando
```bash
# Treinamento básico
python main.py --mode train

# Treinamento customizado
python main.py --mode train --population 50 --generations 100 --mutation 0.15 --test
```

Parâmetros:
- `--population`: Tamanho da população (padrão: 30)
- `--generations`: Número máximo de gerações (padrão: 50)
- `--mutation`: Taxa de mutação (padrão: 0.1)
- `--test`: Testa a acurácia após o treinamento

### 3. Jogar via Linha de Comando
```bash
# Jogar contra Minimax
python main.py --mode play --opponent minimax

# Jogar contra Rede Neural treinada
python main.py --mode play --opponent nn
```

### 4. Testar Acurácia
```bash
python main.py --mode test
```

### 5. Visualizar Evolução do Treinamento
```bash
python visualize_training.py
```

Gera gráficos mostrando:
- Evolução do fitness (melhor, médio, pior)
- Diversidade da população
- Salva em `training_evolution.png`

## 📊 Estrutura do Projeto

```
T2_IA/
├── neural_network.py      # Implementação da Rede Neural MLP
├── genetic_algorithm.py   # Algoritmo Genético
├── tic_tac_toe.py        # Lógica do Jogo da Velha
├── minimax.py            # Algoritmo Minimax
├── trainer.py            # Sistema de Treinamento
├── gui.py                # Interface Gráfica
├── main.py               # Script Principal
├── visualize_training.py # Visualização da Evolução
├── README.md             # Este arquivo
└── best_weights.npy      # Pesos treinados (gerado após treino)
```

## 🎮 Modos de Jogo

### 1. Humano vs Minimax
- Você joga como X (sempre começa)
- Minimax joga como O
- Perfeito para entender o jogo

### 2. Treinar Rede Neural
- Configure: tamanho da população, gerações, mutação
- Acompanhe a evolução em tempo real
- Agenda de dificuldade:
  - Primeira metade: Minimax Médio
  - Segunda metade: Minimax Difícil

### 3. Humano vs Rede Neural
- Teste a rede treinada
- Você joga como X
- Rede Neural joga como O

## 📈 Função de Aptidão

A função de aptidão avalia cada rede baseando-se em:

```
Fitness = (Σ resultados dos jogos) / número de jogos

Onde cada jogo contribui:
- Vitória: +100 pontos
- Empate: +50 pontos
- Derrota: -50 pontos
- Jogada inválida: -20 pontos
- Jogada válida: +2 pontos
```

## 🔧 Parâmetros Recomendados

### Treinamento Rápido (Teste)
```python
população = 20
gerações = 30
mutação = 0.15
```

### Treinamento Balanceado (Recomendado)
```python
população = 30
gerações = 50
mutação = 0.1
```

### Treinamento Intensivo (Melhor Resultado)
```python
população = 50
gerações = 100
mutação = 0.08
```

## 📝 Estratégia de Treinamento

1. **Fase 1 (0 - 50% das gerações)**: Minimax Médio
   - Permite à rede aprender padrões básicos
   - Maior diversidade de situações

2. **Fase 2 (50% - 100% das gerações)**: Minimax Difícil
   - Refinamento das estratégias
   - Aprende a jogar otimamente

## 🎯 Resultados Esperados

Após treinamento adequado, a rede neural deve:
- **Acurácia contra Minimax Difícil**: 0-10% vitórias, 60-90% empates
- **Jogadas inválidas**: < 5%
- **Convergência**: 30-50 gerações

> **Nota**: Empate contra Minimax perfeito é considerado excelente!

## 🐛 Troubleshooting

### Erro: "numpy not found"
```bash
pip install numpy
```

### Erro: "tkinter not found"
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (geralmente já incluído)
```

### Rede não aprende bem
- Aumente o tamanho da população
- Aumente o número de gerações
- Ajuste a taxa de mutação (0.08 - 0.15)
- Verifique se a agenda de dificuldade está adequada

## 🔬 Experimentos Sugeridos

1. **Variar topologia da rede**: Teste diferentes tamanhos de camada oculta
2. **Operadores genéticos**: Teste diferentes cruzamentos e mutações
3. **Função de aptidão**: Ajuste os pesos das penalizações/bonificações
4. **Agenda de dificuldade**: Teste diferentes progressões

## 📚 Conceitos Implementados

- ✅ Rede Neural MLP (2 camadas)
- ✅ Propagação Forward
- ✅ Algoritmo Genético
- ✅ Operadores para valores reais
- ✅ Elitismo
- ✅ Seleção por Torneio
- ✅ Cruzamento Aritmético
- ✅ Mutação Gaussiana
- ✅ Minimax (Médio e Difícil)
- ✅ Aprendizagem por Reforço
- ✅ Função de Aptidão customizada
- ✅ Critérios de parada (gerações e convergência)
- ✅ Interface Gráfica
- ✅ Visualização da evolução

## 👨‍💻 Autor

Projeto desenvolvido para a disciplina de Inteligência Artificial.

## 📄 Licença

Este projeto é de código aberto para fins educacionais.
