# Replicação: An Efficient Intrusion Prevention System for CAN Hindering Cyber-Attacks With a Low-Cost Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: UFPE / CIn](https://img.shields.io/badge/Institution-UFPE%20%2F%20CIn-red.svg)](https://www.cin.ufpe.br/)

Este repositório contém a replicação metodológica rigorosa e a expansão experimental do artigo científico:

> **"An Efficient Intrusion Prevention System for CAN: Hindering Cyber-Attacks With a Low-Cost Platform"**  
> *M. D. Hossain, H. Inoue, H. Ochiai, D. Fall and Y. Kadobayashi*  
> **IEEE Access**, vol. 9, pp. 138356-138371, 2021.  
> DOI: [10.1109/ACCESS.2021.3117464](https://doi.org/10.1109/ACCESS.2021.3117464)

Projeto desenvolvido no âmbito da disciplina **CIN0114** do **Centro de Informática (CIn)** da **Universidade Federal de Pernambuco (UFPE)**.

---

## Autores

* **Bruno Antonio dos Santos Bezerra** ([@brunoantonio](https://github.com/))
* **Filipe Maciel Leicht** ([@filipeleicht](https://github.com/))

---

## 📌 Sumário
- [1. Visão Geral e Objetivos](#1-visão-geral-e-objetivos)
- [2. Estrutura do Repositório](#2-estrutura-do-repositório)
- [3. Instalação e Configuração](#3-instalação-e-configuração)
- [4. Obtenção dos Datasets](#4-obtenção-dos-datasets)
- [5. Análise Exploratória de Dados (EDA)](#5-análise-exploratória-de-dados-eda)
- [6. Pipeline de Treinamento e Otimização](#6-pipeline-de-treinamento-e-otimização)
- [7. Execução no Cluster HPC (SLURM)](#7-execução-no-cluster-hpc-slurm)
- [8. Análise Estatística e Pós-Processamento](#8-análise-estatística-e-pós-processamento)
- [9. Metodologia e Isolamento de Dados](#9-metodologia-e-isolamento-de-dados)
- [10. Contribuições e Melhorias em Relação ao Artigo](#10-contribuições-e-melhorias-em-relação-ao-artigo)

---

## 1. Visão Geral e Objetivos

O barramento **CAN (Controller Area Network)** é o padrão de fato para comunicação entre Unidades de Controle Eletrônico (ECUs) em veículos automotivos. No entanto, sua concepção original carece de mecanismos nativos de segurança cibernética (como autenticação, integridade de dados e criptografia), tornando redes veiculares vulneráveis a ataques de injeção de mensagens maliciosas como **Fuzzing**, **Drive Gear Spoofing** e **RPM Gauge Spoofing**.

O artigo propõe um Sistema de Prevenção de Intrusão (IPS) baseado em aprendizado de máquina não supervisionado:
1. **Modelos One-Class**: Treinados exclusivamente com tráfego normal para detectar desvios no padrão da carga útil (*payload*).
   * **One-Class Support Vector Machine (OCSVM)** com kernel RBF.
   * **Isolation Forest (iForest)**.
2. **Abordagem Byte-a-Byte ($N = 1 \dots 8$)**: Avaliação de 16 modelos distintos por dataset (8 OCSVM e 8 iForest), variando a dimensão dos dados de entrada de 1 a 8 bytes do campo *DATA*. O objetivo é permitir que o IPS atue preventivamente nos primeiros bytes recebidos pelo microcontrolador, abortando o frame antes do final da transmissão.
3. **Validação Rigorosa**: 10-Fold Cross-Validation, otimização por Grid Search e análise estatística confirmatória via **ANOVA One-Way** e teste post-hoc de **Tukey HSD**.

---

## 2. Estrutura do Repositório

```text
projeto_CIN0114/
├── datasets/                                 # Datasets HCRL (ignorados pelo git)
│   └── .gitkeep                              # Mantém a pasta versionada
├── splits_cache/                             # Caches binários (.npz) de índices
│   └── .gitkeep
├── results/                                  # CSVs de validação cruzada e teste
│   └── .gitkeep
├── logs/                                     # Logs detalhados de execução e SLURM
│   └── .gitkeep
├── models/                                   # Modelos treinados (.joblib)
│   └── .gitkeep
├── An_Efficient_Intrusion_Prevention...pdf   # Cópia do artigo científico original
├── EDA_CAN_Intrusion_Datasets.ipynb          # Notebook de EDA (zero test leakage)
├── build_notebook.py                         # Script construtor/validador do Notebook
├── config.py                                 # Grades de parâmetros e configurações
├── notes.txt                                 # Notas de metodologia e requisitos
├── post_analysis.py                          # Análise ANOVA, Tukey HSD, SOTA e Fig. 5
├── requirements.txt                          # Dependências Python mapeadas
├── run_experiments.py                        # Pipeline principal de experimentos
├── submit_job_cluster.sh                     # Script SLURM (Job Array) para Cluster HPC
├── utils.py                                  # Módulo central de I/O, métricas e splits
└── README.md                                 # Documentação do projeto
```

---

## 3. Instalação e Configuração

### 3.1. Requisitos do Sistema
* Sistema Operacional: Linux (testado em Arch Linux e Debian/Ubuntu HPC Cluster)
* Python: `>= 3.10`

### 3.2. Criação do Ambiente Virtual (venv)
Por boas práticas, recomenda-se criar o ambiente virtual fora do diretório do projeto:

```bash
# Criar ambiente virtual
python3 -m venv ~/venvs/cin0114_venv

# Ativar ambiente virtual
source ~/venvs/cin0114_venv/bin/activate

# Atualizar pip e instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Registro do Kernel no Jupyter
Para abrir o notebook com o interpretador correto:

```bash
python -m ipykernel install --user --name=cin0114_venv --display-name="Python (CIN0114 CAN IPS)"
```

---

## 4. Obtenção dos Datasets

Os dados utilizados são provenientes do **Car-Hacking Dataset** disponibilizado pelo *HCRL (Hacking and Countermeasure Research Lab)*:
* [Link do HCRL Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)

Os arquivos descompactados devem ser alocados dentro do diretório `datasets/`:
* `datasets/Fuzzy_dataset.csv`
* `datasets/gear_dataset.csv`
* `datasets/RPM_dataset.csv`
* `datasets/DoS_dataset.csv` *(opcional/referência)*

O script `utils.py` verifica automaticamente a integridade e mapeamento dos datasets através de streaming otimizado por geradores, evitando estouro de memória RAM.

---

## 5. Análise Exploratória de Dados (EDA)

O arquivo [`EDA_CAN_Intrusion_Datasets.ipynb`](file:///home/bruno/Documents/projeto_CIN0114/EDA_CAN_Intrusion_Datasets.ipynb) foi construído sob **estrito isolamento metodológico**:
* **Zero Test Leakage**: A EDA é executada **exclusivamente** sobre a partição de Treinamento + Validação (amostras $0 \dots 299.999$), assegurando que as distribuições do conjunto de Teste nunca sejam observadas na fase exploratória.
* **Filtragem de DLC**: Considera apenas mensagens CAN completas de 8 bytes de dados ($DLC = 8$), conforme premissa do artigo.
* **Normalização por Domínio**: Mapeamento dos valores de bytes $[0, 255]$ para escala $[0.0, 1.0]$.
* **Análise Temporal ($\Delta t$)**: Estudo dos intervalos de chegada entre mensagens consecutivas para identificar injeção de alta frequência.
* **Mapas de Transição e Correlação**: Análise espacial dos bytes $D_0 \dots D_7$ para cada tipo de ataque versus tráfego normal.

---

## 6. Pipeline de Treinamento e Otimização

O script [`run_experiments.py`](file:///home/bruno/Documents/projeto_CIN0114/run_experiments.py) gerencia o ciclo completo de treinamento, seleção de modelos e avaliação final.

### 6.1. Características Principais
* **Validação Cruzada 10-Fold**: 250.000 amostras normais divididas em 10 folds de 25.000. Em cada iteração, 9 folds (225.000 normais) são usados no treino e 1 fold normal (25.000) somado a 25.000 amostras maliciosas são usados para validação balanceada.
* **Otimização por Grid Search**: Avaliação exaustiva dos hiperparâmetros configurados em [`config.py`](file:///home/bruno/Documents/projeto_CIN0114/config.py).
* **Métrica Primária**: Acurácia (com cômputo adicional de Precisão, Recall, F1-Score e AUC-ROC).
* **Salvamento Incremental e Resiliência (`--resume`)**: Permite pausar e retomar experimentos sem perda de progresso em caso de interrupção ou queda de nós no cluster.
* **Cache Inteligente de Partições**: Índices de particionamento são serializados em `splits_cache/` em formato comprimido `.npz`.

### 6.2. Exemplos de Execução

```bash
# 1. Teste Rápido de Fumaça (Smoke Test)
python run_experiments.py --dataset "Fuzzing" --grid smoke

# 2. Execução Padrão com Salvamento de Modelos Serializados (.joblib)
python run_experiments.py --dataset "Fuzzing" --grid default --save_models

# 3. Execução Completa para Todos os Datasets
python run_experiments.py --dataset all --grid default

# 4. Retomada de Execução Interrompida
python run_experiments.py --dataset "Drive Gear" --grid full --resume
```

---

## 7. Execução no Cluster HPC (SLURM)

Para execuções pesadas no cluster do Centro de Informática (CIn/UFPE), o script [`submit_job_cluster.sh`](file:///home/bruno/Documents/projeto_CIN0114/submit_job_cluster.sh) está preparado com suporte a **SLURM Job Arrays**.

### 7.1. Submissão Paralela Multi-Nó (Recomendado)
Dispara 3 tarefas independentes em paralelo (uma para cada dataset: Fuzzing, Drive Gear, RPM Gauge):

```bash
sbatch --array=0-2 submit_job_cluster.sh
```

### 7.2. Submissão para um Dataset Específico

```bash
sbatch submit_job_cluster.sh "Drive Gear"
```

Os logs de saída e erro são gerados automaticamente na pasta `logs/` com carimbos do SLURM Job ID.

---

## 8. Análise Estatística e Pós-Processamento

O módulo [`post_analysis.py`](file:///home/bruno/Documents/projeto_CIN0114/post_analysis.py) reproduz fielmente as análises inferenciais descritas na **Seção VI** do artigo:

```bash
python post_analysis.py
```

### O que o script realiza:
1. **Teste de Normalidade de D'Agostino-Pearson (`scipy.stats.normaltest`)**: Valida o pressuposto de normalidade paramétrica das taxas de acerto nos 10 folds de validação cruzada ($p > 0.05$).
2. **One-Way ANOVA (`scipy.stats.f_oneway`)**: Testa se há diferenças estatisticamente significantes entre as médias dos 16 modelos ($F\text{-statistic}$, $p\text{-value}$).
3. **Teste Post-Hoc Tukey HSD (`scipy.stats.tukey_hsd`)**: Comparações par a par para identificar o **menor número de bytes ($N$)** cuja acurácia não difere estatisticamente do modelo completo de 8 bytes ($p > 0.05$).
4. **Tabelas 7 a 11 do Artigo**: Confronta os resultados obtidos com os modelos do Estado da Arte:
   * **GIDS** (Kang et al.)
   * **DCNN** (Song et al.)
   * **LSTM-IDS** (Taylor et al.)
   * **MTH-IDS** (Yang et al.)
   * **Artigo Original (Proposed IPS)**
5. **Matrizes de Confusão (Figura 5)**: Plota e salva em formato gráfico as matrizes de confusão comparativas (Normal vs. Ataque) para os conjuntos de teste.

---

## 9. Metodologia e Isolamento de Dados

A replicação segue estritamente as especificações das **Tabelas 2, 3, 4 e 5** do artigo:

| Dataset | Total de Mensagens | Treinamento (Normais) | Validação (Normais + Ataque) | Teste (Restante) |
| :--- | :---: | :---: | :---: | :---: |
| **Fuzzing** | 3.838.860 | 225.000 (9 folds) | 25.000 normais + 25.000 ataque | 3.563.860 |
| **Drive Gear** | 4.443.142 | 225.000 (9 folds) | 25.000 normais + 25.000 ataque | 4.168.142 |
| **RPM Gauge** | 4.621.702 | 225.000 (9 folds) | 25.000 normais + 25.000 ataque | 4.346.702 |

* O particionamento é determinístico e reprodutível via semente pseudoaleatória controlada (`seed = 42`).
* O conjunto de **Treinamento contém exclusivamente dados normais**, respeitando o paradigma One-Class.
* Os conjuntos de **Validação e Teste contêm tráfego misto** (normal e malicioso).

---

## 10. Contribuições e Melhorias em Relação ao Artigo

1. **Métricas Adicionais de Avaliação**: O artigo original reporta apenas Acurácia, Precisão e Recall. Nossa implementação adiciona **F1-Score** e **AUC-ROC (Area Under the ROC Curve)**, proporcionando uma avaliação mais completa frente a desbalanceamentos severos no conjunto de teste.
2. **Reprodutibilidade Aberta**: O trabalho original não disponibilizou o código-fonte nem a especificação detalhada da grade de hiperparâmetros. Este repositório fornece toda a infraestrutura com parametrizações completas e modularizadas.
3. **Escalabilidade e Tolerância a Falhas**: Implementação de leitura em blocos, cacheamento binário de partições e suporte a *checkpointing* para execução em clusters de alto desempenho.

---

## Licença

Este projeto é disponibilizado sob a licença [MIT](https://opensource.org/licenses/MIT).
