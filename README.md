## 📦 Planejamento de Produção com Branch & Bound (Streamlit)

Este projeto implementa um sistema de planejamento de produção otimizado para maximizar o lucro, utilizando o algoritmo **Branch & Bound** para encontrar quantidades de produção inteiras (Problema da Mochila com Múltipla Escolha e Limites). A interface é construída com **Streamlit**, permitindo a visualização interativa dos dados e dos resultados da otimização.

-----

## 🚀 Como Executar

1.  **Pré-requisitos:** Certifique-se de ter o Python (3.7+) instalado.

2.  **Instalação das dependências:**

    ```bash
    pip install streamlit pandas
    ```

3.  **Estrutura de Arquivos:** O projeto requer um arquivo de configuração de linha de produção no formato JSON, nomeado `production-line.json` no mesmo diretório do script.

      * `main.py`: O código principal com a lógica B\&B e a interface Streamlit.
      * `production-line.json`: Contém a configuração das máquinas, tarefas, modos e produtos, além dos pedidos (`product_requests`).

4.  **Execução:** Execute o aplicativo Streamlit:

    ```bash
    streamlit run main.py
    ```

    O aplicativo abrirá automaticamente no seu navegador padrão.

-----

## 🛠️ Algoritmo e Lógica

### 1\. Pré-processamento e Modelagem

O sistema modela o problema como uma variação do **Problema da Mochila (Knapsack Problem)**, onde o recurso limitado é o **tempo total de produção** disponível em todas as máquinas (capacidade).

  * **Capacidade de Tempo:** Calculada como `tempo_janela * número_de_máquinas`. O slider na sidebar permite ajustar essa capacidade.
  * **Complexidade da Máquina:**
    $$\text{Complexidade Bruta} = \text{Nº de Modos} \times \text{Média das Potências Médias dos Modos}$$
    A complexidade é então normalizada entre 0 e 1.
  * **Estimativa de Lucro Unitário:** O lucro unitário é estimado com base no número de *runs* de tarefas por produto, ajustado pela complexidade média das máquinas usadas:
    $$\text{Lucro Unitário} = \text{Lucro Base} \times (1 + \text{Coeficiente Complexidade} \times \text{Complexidade Média})$$
  * **Itens para o B\&B:** Cada produto requisitado (com limite de quantidade igual ao total dos pedidos) torna-se um "item" com seu **Lucro Unitário** e **Tempo Estimado** de produção.

### 2\. Branch & Bound (B\&B)

O algoritmo B\&B é usado para encontrar a combinação de quantidades inteiras de produtos que maximiza o lucro, respeitando a capacidade de tempo.

  * **Ordenação:** Os produtos são ordenados pela razão **Lucro/Tempo** para uma melhor heurística de relaxação.
  * **Nó (`BBNode`):** Representa um estado de decisão: `level` (produto atual), `profit`, `time_used`, `quantities` e `bound`.
  * **Heurística de Busca:** Utiliza uma **max-heap** para priorizar a exploração de nós com o maior `bound` (limite superior) para encontrar a melhor solução mais rapidamente.
  * **Relaxação (`bound_estimate`):** O limite superior é calculado usando a **relaxação fracionária** (como no knapsack 0/1, mas adaptado para quantidades limitadas) para os itens restantes, garantindo que o lucro ótimo está abaixo ou igual a este limite.
  * **Processo:** O algoritmo itera sobre cada produto (`level`), testando todas as quantidades possíveis (de `amount` até 0).
      * Se o `bound` do nó filho for menor ou igual ao `best_profit` atual (solução inteira encontrada), o nó é **podado** (`pruned`).
      * Se uma nova solução inteira for encontrada com lucro maior, `best_profit` é atualizado.

-----

## 📊 Dashboard Streamlit

O painel fornece uma visão completa do processo:

  * **Parâmetros (Sidebar):** Permite ajustar o **Fator de Capacidade** para explorar o impacto da capacidade total no resultado da otimização.
  * **Dados dos Produtos:** Tabela com as estimativas de tempo, complexidade média e lucro unitário de cada produto.
  * **Resultados da Otimização:**
      * Exibe o **Lucro Total** máximo e o **Tempo Usado**.
      * Métricas de desempenho do algoritmo (Nós **Explorados** e **Podados**).
      * Tabela com as **Quantidades Produzidas** na solução ótima.
  * **Visualizações:**
      * **Evolução do B\&B:** Gráfico de linha mostrando o `best_profit` (melhor solução inteira) e o `bound` (limite superior) em função do nível/iteração, ilustrando o estreitamento da busca.
      * **Distribuição de Lucro:** Gráfico de barras mostrando a contribuição de lucro de cada produto na solução ótima.

-----

## 📄 Estrutura de Código

  * **Configurações:** Constantes como o caminho do JSON e o limite de tempo (`TIME_LIMIT_SECONDS`).
  * **Funções de Pré-processamento:**
      * `compute_machine_complexity`: Calcula a complexidade de cada máquina.
      * `estimate_products`: Calcula o tempo e o lucro unitário de cada produto.
      * `prepare_candidates`: Agrega pedidos e calcula a capacidade total.
  * **Lógica B\&B:**
      * `BBNode`: Classe para representar os nós da árvore.
      * `bound_estimate`: Implementa a função de relaxação fracionária (limite superior).
      * `branch_and_bound_integer`: O algoritmo principal de Branch & Bound.
  * **Interface Streamlit (`main`):** Responsável por carregar os dados, configurar a interface, executar o B\&B e exibir os resultados.