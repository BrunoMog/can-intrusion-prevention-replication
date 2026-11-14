# Replicação e Extensão: Sistema de Prevenção de Intrusão (IPS) Eficiente para Barramentos CAN Automotivos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: UFPE / CIn](https://img.shields.io/badge/Institution-UFPE%20%2F%20CIn-red.svg)](https://www.cin.ufpe.br/)

Este repositório contém a replicação experimental rigorosa, a avaliação de generalização em um segundo conjunto de dados independente e a **prova de conceito de uma arquitetura aprimorada em dois estágios com Dual-Cache** para o artigo científico:

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
1. **Replicação Integral do Artigo Base**: Avaliação cruzada de 10 *folds* dos 16 modelos não supervisionados ($\text{OCSVM\_1F}\dots\text{8F}$ e $\text{iForest\_1F}\dots\text{8F}$) nos ataques *Fuzzing*, *Drive Gear Spoofing* e *RPM Gauge Spoofing* do *Car-Hacking Dataset*, com validação estatística via D'Agostino-Pearson, One-Way ANOVA e Tukey HSD.
2. **Generalização em Segundo Dataset Independente**: Avaliação completa do protocolo (com particionamento proporcional em 10 *folds* sem vazamento de dados) no cenário físico *Prototype Speedometer Spoofing*.
3. **Proposta de Melhoria e Prova de Conceito (`poc_ips/`)**: Arquitetura IPS assíncrona em **dois estágios** ($\text{iForest\_5F} \rightarrow \text{CAN\_ID} + 8\text{ bytes}$, exclusivamente com *Isolation Forest*) executada em **três *threads*** com memória compartilhada, sistema **Dual-Cache** com expiração anti-*replay* ($\text{TTL} = 100\text{ ms}$) e avaliação em simulador cronológico de confinamento de erros CAN 2.0A (`TEC += 8`, `Error Warning >= 96`, `Error Passive > 127`, `Bus-Off > 255`).

---

## 2. Estrutura do Repositório

```text
projeto_CIN0114/
├── datasets/                                 # Datasets locais (ignorados pelo git)
│   └── .gitkeep
├── splits_cache/                             # Índices determinísticos (.npz) de folds/teste (ignorado)
│   └── .gitkeep
├── results/                                  # CSVs e gráficos dos experimentos base (ignorado)
│   └── .gitkeep
├── models/                                   # Modelos base serializados (.joblib) (ignorado)
│   └── .gitkeep
├── logs/                                     # Logs de execução (ignorado)
│   └── .gitkeep
├── poc_ips/                                  # Prova de Conceito: IPS 2 Estágios + Dual-Cache (3 threads)
│   ├── fast_iforest.c                        # Motor nativo em C para inferência por quadro único
│   ├── fast_iforest.py                       # Wrapper ctypes + validação exata contra scikit-learn
│   ├── models_stage2.py                      # Modelos Estágio 2 (9F Global e Híbridos por CAN ID)
│   ├── train_stage2.py                       # Treinamento/seleção 10-Fold CV e avaliação em cascata
│   ├── poc_simulator.py                      # Motor 3-threads Dual-Cache + simulador CAN 2.0A (TEC/Bus-Off)
│   ├── models/                               # Modelos Estágio 1/2 persistidos (.joblib, ignorado)
│   │   └── .gitkeep
│   └── results/                              # CSVs e figuras da PoC (ignorado)
│       └── .gitkeep
├── relatorio/                                # Relatório técnico em formato IEEEtran (LaTeX + PDF)
│   ├── relatorio_can_ips.tex                 # Código-fonte LaTeX principal
│   ├── relatorio_can_ips.pdf                 # Relatório compilado final (13 páginas)
│   ├── IEEEtran.cls                          # Classe IEEEtran Conference
│   ├── fig_eda_signatures.png                # Fig. 1: Assinaturas de bytes e PCA (EDA)
│   ├── fig_bytes_progression.png             # Fig. 2: Evolução da acurácia por N bytes (Fuzzing)
│   ├── figure5_confusion_matrices.png        # Fig. 3: Matrizes de confusão (5F..8F)
│   ├── fig_speedometer_analysis.png          # Fig. 4: Perfil e desempenho no Speedometer Spoofing
│   ├── fig_stage2_cascade.png                # Fig. 5: Resgate de falsos positivos no Estágio 2
│   └── fig_poc_comparison.png                # Fig. 6: Impacto sistêmico no simulador CAN 2.0A
├── An_Efficient_Intrusion_Prevention_System_for_CAN_Hindering_Cyber-Attacks_With_a_Low-Cost_Platform.pdf
├── config.py                                 # Presets de grades (smoke, default, refined, fast_refined, full)
├── utils.py                                  # Parsing DLC=8, conversão Prototype->CSV e splits isolados
├── run_experiments.py                        # Pipeline principal de CV 10-fold, Grid Search e Teste
├── post_analysis.py                          # Testes D'Agostino-Pearson, ANOVA, Tukey HSD, SOTA e Fig. 5
├── build_notebook.py                         # Gerador programático do notebook de EDA
├── EDA_CAN_Intrusion_Datasets.ipynb          # Notebook de EDA (zero test leakage)
├── requirements.txt                          # Dependências Python do projeto
├── .gitignore                                # Regras de arquivos ignorados
└── README.md                                 # Documentação do repositório
```

---

## 3. Instalação e Configuração

### Requisitos
* **Sistema Operacional**: Linux (testado em ambiente local)
* **Python**: `>= 3.10`
* **Compilador C (`gcc`)**: necessário apenas para compilar sob demanda o módulo nativo `poc_ips/fast_iforest.c` (`_fast_iforest_c.so`) usado na simulação PoC
* **LaTeX (`latexmk` + `pdflatex`)**: opcional, para recompilar o relatório em `relatorio/`

### Criação do Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para executar o notebook no Jupyter com o mesmo ambiente:

```bash
python -m ipykernel install --user --name=cin0114 --display-name="Python (CIN0114 CAN IPS)"
```

---

## 4. Datasets e Isolamento Metodológico

Os experimentos empregam quatro conjuntos de dados com mensagens filtradas em `DLC = 8`:
1. **Car-Hacking Dataset (HCRL)**: [Link oficial](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) — coloque em `datasets/` os arquivos `Fuzzy_dataset.csv`, `gear_dataset.csv` e `RPM_dataset.csv` (opcionalmente `DoS_dataset.csv`).
2. **Prototype Speedometer Spoofing (2º dataset)**: arquivo bruto `datasets/data/Prototype/spoofing_speedometer.log` em formato `candump`, convertido para `datasets/Speedometer_dataset.csv` pela função `utils.convert_prototype_speedometer_log_to_csv()`.

### Particionamento (`seed = 42`, zero *test leakage*)

| Dataset | Treino por *Fold* (9 *folds*) | Validação por *Fold* (50/50) | Teste Final (Blindado, DLC=8) |
| :--- | :---: | :---: | :---: |
| **Fuzzing** | 225.000 normais | 25.000 normais + 25.000 ataque | 3.476.024 (3.009.177 normais + 466.847 ataque) |
| **Drive Gear** | 225.000 normais | 25.000 normais + 25.000 ataque | 4.127.977 (3.555.725 normais + 572.252 ataque) |
| **RPM Gauge** | 225.000 normais | 25.000 normais + 25.000 ataque | 4.305.226 (3.675.329 normais + 629.897 ataque) |
| **Speedometer** | 22.500 normais | 250 normais + 250 ataque | 118.046 (114.774 normais + 3.272 ataque) |

---

## 5. Análise Exploratória de Dados (EDA)

O notebook [EDA_CAN_Intrusion_Datasets.ipynb](EDA_CAN_Intrusion_Datasets.ipynb) (gerado via [build_notebook.py](build_notebook.py)) conduz toda a exploração **exclusivamente sobre as partições de Treinamento e Validação** ($0\dots 299.999$ no *Car-Hacking* e $27.500$ quadros no *Speedometer*), preservando os conjuntos de teste intactos:
* Auditoria de contagens e distribuição por `CAN_ID`;
* Perfis espaciais dos bytes normalizados $D_0 \dots D_7$ e identificação de fronteiras de separabilidade (ex.: $D_2 = \texttt{0x7E}$ no *Speedometer*);
* Análise de separabilidade via PCA e dinâmica temporal ($\Delta t$).

---

## 6. Pipeline Base de Experimentos e Análise Estatística

### Execução dos Experimentos ([run_experiments.py](run_experiments.py))

```bash
# Smoke test rápido
python run_experiments.py --datasets Fuzzing --grid smoke

# Replicação completa (grade full nos 3 datasets do artigo)
python run_experiments.py --datasets Fuzzing "Drive Gear" "RPM Gauge" --models OCSVM iForest --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10

# Avaliação separada no 2º dataset (Speedometer)
python run_experiments.py --datasets Speedometer --models iForest --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10
python run_experiments.py --datasets Speedometer --models OCSVM --bytes 1 2 3 4 5 6 7 8 --grid full --folds 10
```

O pipeline utiliza leitura em *streaming*, fatoração exata de limiares na validação cruzada do *Isolation Forest* e retomada incremental via `--resume` (ativa por padrão).

### Análise Estatística e Matrizes de Confusão ([post_analysis.py](post_analysis.py))

```bash
python post_analysis.py --fig5_dataset Fuzzing
```

Executa o teste de normalidade de D'Agostino-Pearson, One-Way ANOVA, teste *post-hoc* de Tukey HSD ($N$ mínimo equivalente ao modelo completo de 8 bytes), compara os resultados com o estado da arte (**GIDS** — Seo *et al.*, **DCNN** e **LSTM-IDS** — Song *et al.*, **MTH-IDS** — Yang *et al.*) e gera `results/figure5_confusion_matrices.png`.

---

## 7. Melhoria Proposta e Prova de Conceito (`poc_ips/`)

Para atacar os falsos positivos do modelo $\text{iForest\_5F}$ e evitar o bloqueio em cascata de ECUs legítimas por *Bus-Off* (`TEC += 8` a cada *Error Frame* consecutivo até `TEC > 255`), o diretório [`poc_ips/`](poc_ips/) implementa uma arquitetura em dois estágios com três *threads* em memória compartilhada:

* **Estágio 1 (`Worker_Stage1`)**: modelo $\text{iForest\_5F}$ pré-carregado atuando de forma síncrona no 5º byte ($D_0 \dots D_4$).
* **Estágio 2 Assíncrono (`Worker_Stage2`)**: modelo exclusivo de *Isolation Forest* com `CAN_ID` normalizado + $8$ bytes ($D_0 \dots D_7$), avaliando variantes globais (`iForest_9F_Global`) e híbridas condicionadas por identificador (`iForest_Hybrid_Fallback`, `iForest_Hybrid_StrictUnknown`).
* **Coordenador com Dual-Cache (`Coordinator`)**: chave compacta de 75 bits `(CAN_ID << 64) | payload_64`, **Cache 1** para rastreamento/deduplicação de alertas suspeitos ($k_{\text{trig}}$ configurável) e **Cache 2** com $\text{TTL} = 100\text{ ms}$ anti-*replay* para respostas em $\mathcal{O}(1)$ antes de nova penalização do `TEC`.

### Comandos da PoC

```bash
# Etapa 1: Seleção 10-Fold CV e persistência dos modelos Estágio 1 e Estágio 2 (requer results/final_test_results.csv)
python poc_ips/train_stage2.py --datasets Speedometer Fuzzing "Drive Gear" "RPM Gauge"

# Etapas 2–4: Simulação PoC em 3 threads + confinamento de erros CAN 2.0A + ablações e figuras
python poc_ips/poc_simulator.py --datasets Speedometer Fuzzing "Drive Gear" "RPM Gauge"
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
