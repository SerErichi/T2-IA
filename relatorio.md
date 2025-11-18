# Relatorio do Projeto T2 - IA com Aprendizagem por Reforco

## Visao Geral

Este trabalho combina uma rede neural MLP, um algoritmo genetico e o algoritmo Minimax para treinar uma IA a jogar Jogo da Velha sem uso de backpropagation. O Minimax atua como tutor durante o treinamento, enquanto o algoritmo genetico busca os melhores pesos para a rede. O sistema conta ainda com uma interface grafica em Tkinter que permite executar os cenarios exigidos: humano vs minimax, treinamento da rede vs minimax e humano vs rede treinada.

## Topologia da Rede Neural

- Entrada: 9 neuronios (tabuleiro 3x3 linearizado).
- Camada oculta: 18 neuronios com ativacao `tanh`, inicializados com pesos aleatorios N(0, 0.5).
- Camada de saida: 9 neuronios lineares (score por posicao).
- A funcao `predict` normaliza o tabuleiro na perspectiva do jogador atual (proria marca = 1, celula vazia = 0, adversario = -1) e realiza apenas a propagacao forward. Nao ha calculo de gradiente.
- Os pesos sao armazenados em vetor unico para facilitar a manipulacao pelo algoritmo genetico (total de 9*18 + 18 + 18*9 + 9 = 459 parametros).

## Estrutura do Algoritmo Genetico

- Representacao: cromossomos de valores reais com 459 genes correspondendo a todos os pesos e vieses da rede.
- Inicializacao: populacao com distribuicao normal N(0, 0.5).
- Avaliacao (funcao de aptidao): media dos resultados em partidas contra o Minimax.
  - Vitoria: +100.
  - Empate: +50.
  - Derrota: -50.
  - Jogada invalida: -20 por ocorrencia.
  - Jogada valida realizada: +2 (evita redes inoperantes).
- Agenda de dificuldade: primeiras geracoes jogam contra Minimax medio (50% das jogadas aleatorias), geracoes posteriores usam Minimax dificil (100% minimax).
- Selecionadores e operadores:
  - Selecao: elitismo (top 5) + torneio de tamanho 3.
  - Cruzamento: aritmetico (filhos = combinacao convexa dos pais) quando sorte <= taxa de cruzamento (padrao 0.8).
  - Mutacao: gaussiana independente por gene (ruido N(0, 0.5) aplicado com probabilidade igual a taxa de mutacao, padrao 0.1).
- Critérios de parada: numero maximo de geracoes ou convergencia (melhor fitness variando menos que 1e-3 ao longo de `patience` geracoes).

## Etapas de Desenvolvimento

1. Implementacao do motor de Jogo da Velha (`tic_tac_toe.py`): controle de jogadas, verificacao de vencedor, funcoes utilitarias.
2. Adaptacao do Minimax (`minimax.py`): modos medio e dificil, com possibilidade de alternar dificuldade em tempo de execucao.
3. Rede neural (`neural_network.py`): montagem da MLP, metodos `set_weights`, `get_weights` e `predict` sem retropropagacao.
4. Algoritmo genetico (`genetic_algorithm.py`): operadores para numeros reais, avaliacao por partidas contra o Minimax.
5. Sistema de treinamento (`trainer.py`): agenda de dificuldade, log de estatisticas, salvamento dos melhores pesos (`best_weights.npy`).
6. Interface grafica (`gui.py`): tabuleiro interativo, configuracoes do AG, log textual e botoes para iniciar treino ou testar acuracia. O modo Humano vs Rede Neural carrega automaticamente os pesos evoluidos.
7. Scripts auxiliares: `main.py` (CLI com modos train/play/gui/test), `visualize_training.py` (graficos da evolucao) e `test_system.py` (suite de smoke tests).

## Analise de Resultados

### Execucao Exemplar

- Configuracao: populacao 30, 10 geracoes, taxa de mutacao 0.1, 5 partidas por individuo.
- Historico de fitness mostra queda ao trocar para Minimax dificil, seguida de estabilizacao em media ligeiramente positiva.
- Convergencia atingida em 9 geracoes, melhor fitness final 60.

### Teste de Acuracia (apos treino)

```text
python main.py --mode test
Jogos testados: 100
Vitorias: 0 (0.00%)
Empates: 100 (100.00%)
Derrotas: 0 (0.00%)
Jogadas invalidas: 0
```

Contra um adversario perfeito (Minimax dificil), empatar consistentemente e o resultado esperado.

### Observacoes

- A normalizacao por perspectiva garante que a rede use os mesmos pesos jogando como X ou O, ainda que o ciclo de treinamento original priorize a rede iniciando as partidas.
- O tempo de treinamento cresce linearmente com `populacao * partidas * geracoes`. Para demonstracao, valores menores garantem execucao em poucos minutos.
- A visualizacao (`visualize_training.py`) gera `training_evolution.png` com curvas do melhor, medio e pior fitness mais o desvio padrao da populacao.

## Consideracoes Finais

### Pontos positivos

- Arquitetura modular permite substituir facilmente componentes (por exemplo, novos operadores geneticos ou funcoes de aptidao).
- Interface grafica integra todos os modos exigidos pelo enunciado e fornece log textual para acompanhamento do AG.
- Suite de testes basicos (`test_system.py`) ajuda a validar integridade do sistema apos mudancas.

### Pontos negativos e melhorias

- Treinamento focado em partidas em que a rede inicia pode deixala vulneravel quando joga como O; treinos alternando o primeiro lance podem equilibrar o desempenho.
- Tempo de execucao do AG ainda alto para configuracoes robustas (muitas partidas por individuo). Podem ser exploradas avaliacoes paralelas ou caches de estados.
- A MLP de duas camadas atende ao requisito, mas pode ficar limitada para representar heuristicas mais sofisticadas. Experimentar tamanhos/ativacoes diferentes ou regularizacao pode melhorar a performance.

### Trabalhos futuros

1. Introduzir agenda de dificuldade adaptativa baseada no progresso do fitness.
2. Estender a funcao de aptidao com premiacao por vitorias rapidas e penalidades por escolhas subotimas mesmo sem derrota.
3. Investigar transferencia de conhecimento treinando a rede em ambos os papeis de jogador com rotacao ao longo das geracoes.

## Como Reproduzir

```text
# Preparar ambiente
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Interface grafica
python main.py --mode gui

# Treinamento CLI com teste de acuracia
python main.py --mode train --population 30 --generations 50 --mutation 0.1 --test

# Suite de testes rapida
python test_system.py
```
