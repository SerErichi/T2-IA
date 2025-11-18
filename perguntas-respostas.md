# Perguntas e Respostas sobre o Projeto

## Perguntas Faceis

**Pergunta:** Qual e a topologia da rede neural implementada aqui?

**Resposta:** A MLP possui 9 neuronios na entrada (tabuleiro 3x3 linearizado), 18 neuronios na camada oculta com ativacao `tanh` e 9 neuronios lineares na saida, cada um indicando o score de uma posicao.

**Pergunta:** Por que o projeto nao usa backpropagation e como os pesos sao atualizados?

**Resposta:** O enunciado proibe backpropagation. Os pesos sao atualizados pelo algoritmo genetico: cada cromossomo representa todos os pesos/vieses, e a evolucao usa selecao, cruzamento e mutacao com base no desempenho contra o Minimax.

**Pergunta:** Quais modos de jogo a interface grafica oferece e o que cada um demonstra?

**Resposta:** Ha tres modos: "Humano vs Minimax" (experiencia com o tutor), "Treinar Rede Neural" (executa o AG e registra o log) e "Humano vs Rede Neural" (jogar contra os pesos evoluidos).

**Pergunta:** Onde ficam salvos os melhores pesos depois do treinamento e como carregalos?

**Resposta:** O melhor cromossomo e salvo em `best_weights.npy`. A GUI e a CLI carregam esse arquivo automaticamente em `main.py --mode gui`, `--mode play --opponent nn` ou `--mode test`.

## Perguntas Medias

**Pergunta:** Como a funcao de aptidao equilibra vitorias, empates, derrotas e jogadas invalidas?

**Resposta:** Cada partida soma +100 pela vitoria, +50 pelo empate e -50 pela derrota. Jogadas invalidas penalizam 20 pontos por ocorrencia e cada jogada valida concede +2, evitando redes que nao jogam.

**Pergunta:** De que forma a agenda de dificuldade do Minimax foi configurada ao longo das geracoes?

**Resposta:** Durante o treinamento, as primeiras geracoes usam Minimax no modo medio (50% aleatorio). A partir da metade das geracoes, muda para modo dificil (100% minimax), permitindo inicialmente diversidade de situacoes e depois refinamento contra o adversario perfeito.

**Pergunta:** Como o metodo `predict` garante funcionamento como X ou O?

**Resposta:** `predict` normaliza o tabuleiro na perspectiva do jogador informado: celulas próprias viram 1, vazias 0 e adversarias -1. Assim, os mesmos pesos servem para ambos os papeis.

**Pergunta:** Quais componentes sao testados por `test_system.py` e o que se valida?

**Resposta:** O script testa a rede neural (dimensao dos pesos e saida), o motor do jogo (reset e vencedores), o Minimax (bloqueios/ganhos), o algoritmo genetico (inicializacao, selecao, operadores) e a integracao da rede contra o Minimax.

## Perguntas Dificeis

**Pergunta:** O que mudar para treinar bem a rede tanto iniciando quanto respondendo?

**Resposta:** Alternar o papel da rede em `evaluate_fitness`, permitindo que o Minimax faca o primeiro lance em parte das partidas, chamando `predict` com o jogador correto e ajustando a agenda de dificuldade por papel.

**Pergunta:** Quais limitacoes o cruzamento aritmetico e a mutacao gaussiana impõem e que alternativas existem?

**Resposta:** Ambos mantem a busca perto da combinacao convexa dos pais; com variancia pequena, a exploracao e local. Podemos experimentar BLX-alpha com `alpha > 0`, cruzamento segmentado ou mutacao adaptativa/caotica para explorar mais o espaco de pesos.

**Pergunta:** Como avaliar a convergencia real do AG alem do criterio atual de variacao do melhor fitness?

**Resposta:** Monitorar tambem fitness medio, desvio padrao (diversidade), taxa de jogadas invalidas e desempenho do melhor individuo contra ambos os modos do Minimax, evitando estagnacao ou overfitting ao modo medio.

**Pergunta:** Quais adaptacoes seriam necessarias para paralelizar a avaliacao dos cromossomos?

**Resposta:** Distribuir partidas por cromossomo em threads/processos (por exemplo `multiprocessing.Pool`), fixar sementes por worker para reprodutibilidade, sincronizar coleta de fitness e evitar condicoes de corrida ao atualizar as listas da populacao.
