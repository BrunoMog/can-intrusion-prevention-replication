"""
================================================================================
Módulo: config.py
Projeto: CIN0114 - Replicação de Artigo CAN Intrusion Prevention
Artigo: "An Efficient Intrusion Prevention System for CAN:
         Hindering Cyber-Attacks With a Low-Cost Platform" (IEEE Access, 2021)
================================================================================
Configurações de experimentos, grades de hiperparâmetros (ParamGrid),
definição de modelos, diretórios e métricas de otimização.
================================================================================
"""

import os

# Datasets avaliados no artigo + segundo dataset do projeto (Prototype Speedometer Spoofing)
DATASETS = ['Fuzzing', 'Drive Gear', 'RPM Gauge']
ALL_DATASETS = ['Fuzzing', 'Drive Gear', 'RPM Gauge', 'Speedometer']

# Famílias de modelos e número de bytes de entrada (N = 1 a 8)
MODEL_FAMILIES = ['OCSVM', 'iForest']
BYTE_CONFIGS = [1, 2, 3, 4, 5, 6, 7, 8]

# Métrica principal para seleção do melhor modelo na validação cruzada
# 'Accuracy (%)' | 'F1-Score (%)' | 'Precision (%)' | 'Recall (%)' | 'AUC-ROC (%)'
PRIMARY_METRIC = 'Accuracy (%)'

# Diretórios padrão
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SPLITS_CACHE_DIR = os.path.join(BASE_DIR, 'splits_cache')

# Semente pseudoaleatória padrão
DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 10

# ==============================================================================
# GRADES DE HIPERPARÂMETROS (PARAM GRIDS)
# ==============================================================================

# Grade Completa (Inclui também as combinações campeãs do grid refinado do 1º experimento)
PARAM_GRID_FULL = {
    'OCSVM': {
        'kernel': ['rbf'],
        'nu': [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        'gamma': ['scale', 'auto', 0.1, 0.5, 1.0]
    },
    'iForest': {
        'n_estimators': [50, 100, 200],
        'max_samples': [128, 256, 512, 16384, 'auto'],
        'contamination': ['auto', 0.005, 0.01, 0.015, 0.02, 0.05, 0.1]
    }
}

# Grade Padrão (Balanceada para boa exploração e tempo de execução viável)
PARAM_GRID_DEFAULT = {
    'OCSVM': {
        'kernel': ['rbf'],
        'nu': [0.01, 0.05],
        'gamma': ['scale', 0.1, 0.5]
    },
    'iForest': {
        'n_estimators': [50, 100],
        'max_samples': [256, 512],
        'contamination': ['auto', 0.05]
    }
}

# Grade Rápida / Smoke Test (Para testes de integridade, pipelines locais rápidos)
PARAM_GRID_SMOKE = {
    'OCSVM': {
        'kernel': ['rbf'],
        'nu': [0.05],
        'gamma': ['scale']
    },
    'iForest': {
        'n_estimators': [50],
        'max_samples': [256],
        'contamination': ['auto']
    }
}

# Grade Refinada (Baseada na eliminação dos piores parâmetros da execução original)
# - OCSVM: descarta gamma auto/0.1/0.5 e nu 0.05/0.1; foca em scale/1.0 e nu 0.005/0.01/0.02 (6 combinações)
# - iForest: descarta contaminação auto/0.1 e max_samples 128/auto; foca em contaminação 0.005/0.01/0.02 (12 combinações)
PARAM_GRID_REFINED = {
    'OCSVM': {
        'kernel': ['rbf'],
        'nu': [0.005, 0.01, 0.02],
        'gamma': ['scale', 1.0]
    },
    'iForest': {
        'n_estimators': [100, 200],
        'max_samples': [256, 512, 16384],
        'contamination': [0.005, 0.01, 0.015, 0.02]
    }
}

# Grade Refinada Rápida (Ultraleve: 2 combinações OCSVM e 4 combinações iForest com árvores profundas)
# Ideal para rodadas de ajuste fino quase instantâneas
PARAM_GRID_FAST_REFINED = {
    'OCSVM': {
        'kernel': ['rbf'],
        'nu': [0.01],
        'gamma': ['scale', 1.0]
    },
    'iForest': {
        'n_estimators': [100],
        'max_samples': [16384],
        'contamination': [0.005, 0.01, 0.015, 0.02]
    }
}

GRID_PRESETS = {
    'full': PARAM_GRID_FULL,
    'default': PARAM_GRID_DEFAULT,
    'smoke': PARAM_GRID_SMOKE,
    'refined': PARAM_GRID_REFINED,
    'fast_refined': PARAM_GRID_FAST_REFINED
}
