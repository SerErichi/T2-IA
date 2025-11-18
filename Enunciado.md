# T2 - Aprendizagem por Reforço: RN + AG + Minimax

Profa. Silvia Moraes

## Objetivo

Construir uma solução de IA usando aprendizagem por reforço para que uma rede neural aprenda a jogar o jogo da velha.

## Arquitetura

Na arquitetura da solução, utilize uma rede neural cujos pesos serão evoluídos por um algoritmo genético (AG). O algoritmo backpropagation não será empregado; caberá ao AG encontrar o melhor conjunto de pesos para a rede.

Para facilitar a aprendizagem, acople o algoritmo Minimax (apresentado em aula) como o agente que treinará a rede. O Minimax deve ser adaptado para jogar em dois modos:

- **Modo Médio:** o Minimax é executado em apenas 50% das jogadas; as demais são aleatórias.
- **Modo Difícil:** o Minimax é utilizado em 100% das jogadas.

Esses modos devem ser combinados ao longo do treinamento, e a IA deve sempre iniciar jogando.

## Componentes Principais

- **Rede Neural:** construa uma MLP de duas camadas. A topologia inicial será discutida em aula. Implemente apenas a fase de propagação, tomando como referência os códigos fornecidos nos exercícios. A rede atuará como um dos jogadores.
- **Algoritmo Genético:** os cromossomos correspondem aos pesos da rede. Adapte os exemplos vistos em aula, substituindo o operador de cruzamento por uma versão adequada a valores em ponto flutuante (consulte o Moodle). Defina uma função de aptidão que meça o desempenho da rede ao final de cada partida e ajuste os parâmetros do AG para este problema.
- **Front End:** desenvolva uma interface mínima para o jogo da velha. O programa deve controlar o estado do jogo, detectando vitórias e empates a cada jogada. O front end deve permitir que:
  - o usuário jogue contra o Minimax;
  - a rede aprenda jogando com o Minimax;
  - o usuário jogue contra a rede treinada.
- **Monitoramento do Desenvolvimento:** apresente a evolução da aprendizagem, mostrando como a população do AG progride. Ao final do treinamento, permita que o usuário jogue contra a rede neural e calcule a acurácia obtida.

## Etapas de Desenvolvimento

1. Garanta que a solução permita a alternância de turnos entre a rede neural e o Minimax em um front end mínimo.
2. Defina como entrada da rede o tabuleiro atual e faça com que a saída indique a jogada (célula) selecionada para receber X ou O.
3. Inicie o processo de aprendizagem por reforço criando uma população de tamanho configurável, na qual cada cromossomo contém todos os pesos da rede neural.
4. Crie uma função de aptidão que avalie o desempenho de cada cromossomo jogar contra o Minimax. Penalize jogadas inválidas (células já ocupadas) e derrotas.
5. Ao final de cada geração, execute o ciclo completo do AG aplicando os operadores estudados: seleção (elitismo e torneio), cruzamento para valores reais e mutação.
6. Rode o AG por diversas gerações e monitore a evolução da rede. Defina critérios de parada por número máximo de gerações e por convergência.
7. Após concluir o treinamento, teste a rede jogando contra ela e calcule sua acurácia.

## Pontuação

- 1,0: adaptação do Minimax (modos) para jogar com a rede neural no front end.
- 2,5: construção da topologia da rede neural e implementação da propagação.
- 2,0: ciclo completo do AG (operadores e função de aptidão).
- 1,5: módulos de teste do front end (0,3 para usuário vs. Minimax; 0,6 para rede aprendendo com Minimax; 0,6 para usuário vs. rede treinada).
- 3,0: relatório de desenvolvimento em formato PPT, descrevendo topologia da rede, estrutura e operadores do AG, etapas e resultados, além de considerações finais.
- 1,5: bônus pela escolha deste trabalho.
- Observação: falta de domínio na apresentação pode acarretar desconto significativo ou perda total da nota. Trabalhos sem apresentação não serão considerados.

## Orientações

- Grupo a ser definido.
- Datas de entrega e apresentação conforme cronograma da disciplina.