# Replicação e Extensão: Sistema de Prevenção de Intrusão (IPS) Eficiente para Barramentos CAN Automotivos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: UFPE / CIn](https://img.shields.io/badge/Institution-UFPE%20%2F%20CIn-red.svg)](https://www.cin.ufpe.br/)

Este repositório contém a replicação metodológica (`DLC = 8`), a avaliação de generalização em um segundo conjunto de dados independente e a **prova de conceito de uma arquitetura aprimorada em dois estágios com Dual-Cache** para o artigo científico:

> **"An Efficient Intrusion Prevention System for CAN: Hindering Cyber-Attacks With a Low-Cost Platform"**  
> *Paulo Freitas de Araujo-Filho, Antônio J. Pinheiro, Georges Kaddoum, Divanilson R. Campelo, and Fabio L. Soares*  
> **IEEE Access**, vol. 9, pp. 166855–166869, 2021.  
> DOI: [10.1109/ACCESS.2021.3136147](https://doi.org/10.1109/ACCESS.2021.3136147)

Projeto desenvolvido para a disciplina **CIN0114 — Técnicas de Ataque e Detecção de Intrusão**, no **Centro de Informática (CIn)** da **Universidade Federal de Pernambuco (UFPE)**.

---

## Autores

* **Bruno Antonio dos Santos Bezerra** (`basb@cin.ufpe.br`)
* **Filipe Maciel Leicht** (`fml@cin.ufpe.br`)

---

## Sumário
- [1. Visão Geral e Objetivos](#1-visão-geral-e-objetivos)
- [2. Estrutura do Repositório](#2-estrutura-do-repositório)
- [3. Instalação e Configuração](#3-instalação-e-configuração)
- [4. Datasets e Isolamento Metodológico](#4-datasets-e-isolamento-metodológico)
- [5. Análise Exploratória de Dados (EDA)](#5-análise-exploratória-de-dados-eda)
- [6. Pipeline Base de Experimentos e Análise Estatística](#6-pipeline-base-de-experimentos-e-análise-estatística)
- [7. Melhoria Proposta e Prova de Conceito (`poc_ips/`)](#7-melhoria-proposta-e-prova-de-conceito-poc_ips)
- [8. Relatório Técnico em LaTeX](#8-relatório-técnico-em-latex)

---

## 1. Visão Geral e Objetivos

O barramento **CAN (Controller Area Network)** é o padrão dominante para comunicação entre Unidades de Controle Eletrônico (ECUs) automotivas, mas carece nativamente de autenticação, criptografia e controle de acesso.

Este projeto realiza três frentes principais:
1. **Replicação Metodológica (`DLC = 8`) do Artigo Base**: Avaliação cruzada de 10 *folds* dos 16 modelos não supervisionados (`OCSVM_1F`…`OCSVM_8F` e `iForest_1F`…`iForest_8F`) nos ataques *Fuzzing*, *Drive Gear Spoofing* e *RPM Gauge Spoofing* do *Car-Hacking Dataset*, com análise estatística exploratória via D'Agostino-Pearson, One-Way ANOVA, Tukey HSD e TOST pareado (`Δ = ±1,00%` p.p., executado independentemente do filtro da ANOVA).
2. **Generalização em Segundo Dataset Independente**: Avaliação completa do protocolo (com particionamento proporcional em 10 *folds* sem vazamento de dados) no cenário físico *Prototype Speedometer Spoofing*.
3. **Proposta de Melhoria e Prova de Conceito (`poc_ips/`)**: Arquitetura IPS física ativa em linha (*inline*) assíncrona em **dois estágios** (`iForest_5F → CAN_ID + 8 bytes`, exclusivamente com *Isolation Forest*) executada em **três *threads*** com memória compartilhada (motor em C validado por suítes executáveis contra o `scikit-learn`), sistema **Dual-Cache** com expiração temporal limitada (`TTL = 100 ms`) e avaliação via simulador orientado a eventos cronológicos globais (`heapq`) de confinamento de erros CAN 2.0A por `CAN_ID` (`TEC += 8`, `Error Warning >= 96`, penalidade *Suspend Transmission* de `16 us` em `Error Passive > 127`, descarte de fila em `Bus-Off > 255`), incluindo análise de sensibilidade paramétrica.

---

## 2. Estrutura do Repositório

```text
projeto_CIN0114/
├── datasets/                                 # Datasets locais (ignorados pelo git)
│   └── .gitkeep
├── splits_cache/                             # Índices determinísticos (.npz) de folds/teste (ignorado)
│   └── .gitkeep
├── results/                                  # CSVs dos experimentos base (versionados no repo)
│   ├── cv_grid_search_results.csv
│   ├── final_test_results.csv
│   └── .gitkeep
├── models/                                   # Modelos base serializados (.joblib) (ignorado)
│   └── .gitkeep
├── logs/                                     # Logs de execução (ignorado)
│   └── .gitkeep
├── poc_ips/                                  # Prova de Conceito: IPS 2 Estágios + Dual-Cache (3 threads)
│   ├── fast_iforest.c                        # Motor nativo em C para inferência por quadro único
│   ├── fast_iforest.py                       # Wrapper ctypes (sem APIs privadas) + Liu et al. c(n)
│   ├── test_equivalence.py                   # Testes unitários sintéticos e de integração C vs sklearn
│   ├── test_simulator.py                     # Testes determinísticos (heapq, barreira Cache 2, Cache 1 TTL, worker finally, Error Passive, TOST, SHA-256)
│   ├── models_stage2.py                      # Modelos Estágio 2 (9F Global e Híbridos por CAN ID)
│   ├── train_stage2.py                       # Treinamento/seleção 10-Fold CV e avaliação em cascata
│   ├── poc_simulator.py                      # Motor 3-threads Dual-Cache + fila heapq CAN 2.0A por CAN ID + --sensitivity
│   ├── models/                               # Modelos Estágio 1/2 persistidos (.joblib, ignorado)
│   │   └── .gitkeep
│   └── results/                              # CSVs da PoC (versionados) e figuras (ignoradas)
│       ├── stage2_cv_results.csv
│       ├── stage2_test_comparison.csv
│       ├── poc_simulation_results.csv
│       ├── poc_sensitivity_results.csv
│       └── .gitkeep
├── relatorio/                                # Relatório técnico em formato IEEEtran (LaTeX + PDF)
│   ├── relatorio_can_ips.tex                 # Código-fonte LaTeX principal
│   ├── relatorio_can_ips.pdf                 # Relatório compilado final (15 páginas)
│   ├── IEEEtran.cls                          # Classe IEEEtran Conference
│   ├── fig_eda_signatures.png                # Fig. 1: Assinaturas de bytes e PCA (EDA)
│   ├── fig_bytes_progression.png             # Fig. 2: Evolução da acurácia por N bytes (Fuzzing)
│   ├── figure5_confusion_matrices.png        # Fig. 3: Matrizes de confusão (5F..8F)
│   ├── fig_speedometer_analysis.png          # Fig. 4: Perfil e desempenho no Speedometer Spoofing
│   ├── fig_stage2_cascade.png                # Fig. 5: Resgate de falsos positivos no Estágio 2
│   └── fig_poc_comparison.png                # Fig. 6: Impacto sistêmico no simulador CAN 2.0A
├── An_Efficient_Intrusion_Prevention_System_for_CAN_Hindering_Cyber-Attacks_With_a_Low-Cost_Platform.pdf
├── config.py                                 # Presets de grades (smoke, default, refined, fast_refined, full)
├── utils.py                                  # Parsing DLC=8, conversão Prototype->CSV, fingerprint e splits isolados
├── run_experiments.py                        # Pipeline principal de CV 10-fold, Grid Search e Teste
├── post_analysis.py                          # Testes D'Agostino-Pearson, ANOVA, Tukey HSD, TOST, SOTA e Fig. 5
├── build_notebook.py                         # Gerador programático do notebook de EDA
├── EDA_CAN_Intrusion_Datasets.ipynb          # Notebook de EDA (zero test leakage)
├── requirements.txt                          # Dependências mínimas Python
├── requirements-lock.txt                     # Versões exatas fixadas no ambiente testado (Python 3.14.10)
├── .gitignore                                # Regras de arquivos ignorados
└── README.md                                 # Documentação do repositório
```

---

## 3. Instalação e Configuração

### Requisitos
* **Sistema Operacional**: Linux (testado em ambiente local)
* **Python**: `>= 3.10` (ambiente de referência fixado em `Python 3.14.10` via [`requirements-lock.txt`](requirements-lock.txt))
* **Compilador C (`gcc`)**: necessário apenas para compilar sob demanda o módulo nativo `poc_ips/fast_iforest.c` (`_fast_iforest_c.so`) usado na simulação PoC
* **LaTeX (`latexmk` + `pdflatex`)**: opcional, para recompilar o relatório em `relatorio/`

### Criação do Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt   # ou: pip install -r requirements.txt
```

Para executar o notebook no Jupyter com o mesmo ambiente:

```bash
python -m ipykernel install --user --name=cin0114 --display-name="Python (CIN0114 CAN IPS)"
```

---

## 4. Datasets e Isolamento Metodológico

Os experimentos empregam quatro conjuntos de dados públicos com mensagens filtradas em `DLC = 8`:

1. **Car-Hacking Dataset (HCRL)** — Conjunto original avaliado pelo artigo base:
   * **Link oficial**: <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset>
   * **Uso local**: alocar em `datasets/` os arquivos `Fuzzy_dataset.csv`, `gear_dataset.csv` e `RPM_dataset.csv` (opcionalmente `DoS_dataset.csv`).
2. **Automotive CAN Bus Intrusion Dataset — Prototype (`spoofing_speedometer.log`)** — Segundo conjunto independente para avaliação de generalização:
   * **Link oficial (4TU.ResearchData)**: <https://data.4tu.nl/articles/_/12696950/2>
   * **DOI**: [`10.4121/uuid:b74b4928-c377-4585-9432-2004dfa20a5d`](https://doi.org/10.4121/uuid:b74b4928-c377-4585-9432-2004dfa20a5d)
   * **Descrição**: Captura real em barramento CAN de dois veículos (*Opel Astra* e *Renault Clio*) e de uma bancada física (*Prototype*), contendo dados normais e ataques de *Diagnostic*, *Fuzzing*, *Replay*, *Suspension*, *DoS* e *Speedometer Spoofing*.
   * **Uso local**: extrair para `datasets/data/` (garantindo `datasets/data/Prototype/spoofing_speedometer.log`) e executar `utils.convert_prototype_speedometer_log_to_csv()` para gerar `datasets/Speedometer_dataset.csv`.

> Os CSVs de resultados gerados pela replicação ([`results/`](results/)) e pela Prova de Conceito ([`poc_ips/results/`](poc_ips/results/)) encontram-se versionados no repositório para consulta e reprodutibilidade direta dos resultados numéricos.

### Particionamento (`seed = 42`, zero *test leakage*)

| Dataset | Treino por *Fold* (9 *folds*) | Validação por *Fold* (50/50) | Teste Final (Blindado, DLC=8) |
| :--- | :---: | :---: | :---: |
| **Fuzzing** | 225.000 normais | 25.000 normais + 25.000 ataque | 3.476.024 (3.009.177 normais + 466.847 ataque) |
| **Drive Gear** | 225.000 normais | 25.000 normais + 25.000 ataque | 4.127.977 (3.555.725 normais + 572.252 ataque) |
| **RPM Gauge** | 225.000 normais | 25.000 normais + 25.000 ataque | 4.305.226 (3.675.329 normais + 629.897 ataque) |
| **Speedometer** | 22.500 normais | 250 normais + 250 ataque | 118.046 (114.774 normais + 3.272 ataque) |

---

## 5. Análise Exploratória de Dados (EDA)

O notebook [EDA_CAN_Intrusion_Datasets.ipynb](EDA_CAN_Intrusion_Datasets.ipynb) (gerado via [build_notebook.py](build_notebook.py)) conduz toda a exploração **exclusivamente sobre as partições de Treinamento e Validação** (quadros `0..299.999` no *Car-Hacking* e `27.500` quadros no *Speedometer*), preservando os conjuntos de teste intactos:
* Auditoria de contagens e distribuição por `CAN_ID`;
* Perfis espaciais dos bytes normalizados (`D0..D7`) e identificação de fronteiras de separabilidade (ex.: `D2 = 0x7E` no *Speedometer*);
* Análise de separabilidade via PCA e dinâmica temporal (`Δt`).

---

## 6. Pipeline Base de Experimentos e Análise Estatística

### Execução dos Experimentos ([run_experiments.py](run_experiments.py))

```bash
# Smoke test rápido
python run_experiments.py --datasets Fuzzing --grid smoke

# Replicação metodológica completa nos 3 datasets do artigo (DLC = 8)
python run_experiments.py --datasets Fuzzing "Drive Gear" "RPM Gauge" --models OCSVM iForest --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10

# Avaliação separada no 2º dataset (Speedometer)
python run_experiments.py --datasets Speedometer --models iForest --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10
python run_experiments.py --datasets Speedometer --models OCSVM --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10
```

O pipeline utiliza leitura em *streaming*, fatoração exata de limiares na validação cruzada do *Isolation Forest*, verificação estrita de integridade SHA-256 integral de arquivo nos caches de *splits* (`utils.py`) e retomada incremental estrita via `--resume` (ativa por padrão; flags `--no-strict-checkpoint` e `--allow-legacy-checkpoint` disponíveis caso seja necessário reaproveitar checkpoints antigos).

### Análise Estatística e Matrizes de Confusão ([post_analysis.py](post_analysis.py))

```bash
python post_analysis.py --fig5_dataset Fuzzing
```

Executa os testes D'Agostino-Pearson, One-Way ANOVA, *post-hoc* de Tukey HSD e TOST pareado (`Δ = ±1,00%` p.p. vs `iForest_8F`, executado de forma incondicional independentemente do resultado da ANOVA e interpretado como evidência exploratória devido à sobreposição `9/10` dos *folds* e pós-seleção), compara os resultados com métricas reportadas da literatura (**GIDS** — Seo *et al.*, **DCNN** e **LSTM-IDS** — Song *et al.*, **MTH-IDS** — Yang *et al.*) e gera `results/figure5_confusion_matrices.png`.

---

## 7. Melhoria Proposta e Prova de Conceito (`poc_ips/`)

Para atacar os falsos positivos do modelo `iForest_5F` e evitar o bloqueio em cascata de ECUs legítimas por *Bus-Off* (`TEC += 8` a cada *Active Error Flag* injetado pelo IPS físico em linha até `TEC > 255`), o diretório [`poc_ips/`](poc_ips/) implementa uma arquitetura em dois estágios com três *threads* em memória compartilhada (avaliada sob escalonador global cronológico `heapq`, penalidade de *Suspend Transmission* de `16 us` em `Error Passive` e descarte de retransmissões enfileiradas em `Bus-Off`):

* **Estágio 1 (`Worker_Stage1`)**: modelo `iForest_5F` pré-carregado atuando de forma síncrona no 5º byte (`D0..D4`).
* **Estágio 2 Assíncrono (`Worker_Stage2`)**: modelo exclusivo de *Isolation Forest* com `CAN_ID` normalizado + `8` bytes (`D0..D7`), avaliando variantes globais (`iForest_9F_Global`) e híbridas condicionadas por identificador (`iForest_Hybrid_Fallback`, `iForest_Hybrid_StrictUnknown`).
* **Coordenador com Dual-Cache (`Coordinator`)**: chave compacta de 75 bits `(CAN_ID << 64) | payload_64`, **Cache 1** para rastreamento/deduplicação de alertas suspeitos (`k_trig` configurável, com expiração independente por `TTL = 100 ms` para entradas `OBSERVING`) e **Cache 2** com expiração temporal limitada (`TTL = 100 ms`, invalidação *lazy* + varredura periódica a cada 50 mil eventos) para respostas em `O(1)` antes de nova penalização do `TEC` (não provê proteção *anti-replay*).

### Comandos da PoC

```bash
# Etapa 1: Seleção 10-Fold CV e persistência dos modelos Estágio 1 e Estágio 2 (requer results/final_test_results.csv)
python poc_ips/train_stage2.py --datasets Speedometer Fuzzing "Drive Gear" "RPM Gauge"

# Verificação de equivalência numérica e decisória do motor nativo em C vs scikit-learn
# (defina CAN_IPS_EQUIV_MAX_SCAN=0 para varrer 100% dos traços reais)
python poc_ips/test_equivalence.py

# Testes unitários determinísticos (heapq cronológico, Error Passive/Bus-Off, TOST, SHA-256 e checkpoints)
python poc_ips/test_simulator.py

# Etapas 2–4: Simulação PoC em 3 threads + modelo CAN 2.0A por CAN ID + ablações e figuras
python poc_ips/poc_simulator.py --datasets Speedometer Fuzzing "Drive Gear" "RPM Gauge"

# Análise de sensibilidade paramétrica (latência Estágio 2, recuperação Bus-Off e TTL da Cache 2)
python poc_ips/poc_simulator.py --sensitivity --datasets Speedometer "Drive Gear" Fuzzing
```

---

## 8. Relatório Técnico em LaTeX

O relatório final está disponível em [`relatorio/relatorio_can_ips.pdf`](relatorio/relatorio_can_ips.pdf) e seu código-fonte em [`relatorio/relatorio_can_ips.tex`](relatorio/relatorio_can_ips.tex). Para recompilar:

```bash
cd relatorio
latexmk -pdf -interaction=nonstopmode relatorio_can_ips.tex
```

---

## Licença

Este projeto é disponibilizado sob a licença [MIT](https://opensource.org/licenses/MIT).
