"""
================================================================================
Módulo: utils.py
Projeto: CIN0114 - Replicação de Artigo CAN Intrusion Prevention
Artigo: "An Efficient Intrusion Prevention System for CAN:
         Hindering Cyber-Attacks With a Low-Cost Platform" (IEEE Access, 2021)
Autores do Artigo: Paulo Freitas de Araujo-Filho et al.
================================================================================
Este módulo centraliza todas as rotinas utilitárias para:
1. Carregamento, parsing e decodificação vetorial de mensagens CAN com DLC=8.
2. Indexação reproduzível e particionamento estrito de Treino, Validação e Teste
   conforme a Seção V-A e Tabelas 3, 4 e 5 do artigo original.
3. Cache em disco (.npz) dos índices para garantir reprodutibilidade perfeita
   e carregamento instantâneo em notebooks, scripts locais e jobs no cluster HPC.
4. Funções para validação cruzada de 10 folds e métricas de desempenho.
================================================================================
"""

import os
import csv
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd


# ==============================================================================
# CONSTANTES GLOBAIS E CAMINHOS PADRÃO
# ==============================================================================

DATASET_PATHS = {
    'Fuzzing': 'datasets/Fuzzy_dataset.csv',
    'Drive Gear': 'datasets/gear_dataset.csv',
    'RPM Gauge': 'datasets/RPM_dataset.csv'
}

BYTE_COLS = [f'D{i}' for i in range(8)]
NORM_BYTE_COLS = [f'D{i}_norm' for i in range(8)]

# Números de referência reportados na Tabela 2 do artigo
PAPER_TABLE_2 = {
    'Fuzzing': {'total': 3838860, 'normal': 3347013, 'attack': 491847},
    'Drive Gear': {'total': 4443142, 'normal': 3845890, 'attack': 597252},
    'RPM Gauge': {'total': 4621702, 'normal': 3966805, 'attack': 654897}
}

# Parâmetros de partição descritos na metodologia do artigo (Seção V-A)
N_TRAIN_NORMAL = 225_000         # 9 folds de 25.000 amostras normais
N_VAL_NORMAL = 25_000            # 1 fold de 25.000 amostras normais
N_VAL_ATTACK = 25_000            # 25.000 amostras de ataque selecionadas aleatoriamente
N_TOTAL_NORMAL_SAMPLED = 250_000 # Total de amostras normais selecionadas para Treino + Validação


def _resolve_filepath(dataset_key_or_path: str) -> str:
    """Resolve o caminho absoluto ou relativo a partir da chave do dataset ou caminho direto."""
    if dataset_key_or_path in DATASET_PATHS:
        return DATASET_PATHS[dataset_key_or_path]
    return dataset_key_or_path


# ==============================================================================
# 1. AUDITORIA E VARREDURA GLOBAL DE METADADOS (O(1) MEMÓRIA)
# ==============================================================================

def scan_dataset_metadata(filepath_or_name: str) -> Dict[str, Union[int, float, dict]]:
    """
    Realiza uma varredura completa em streaming sobre o arquivo CSV do dataset,
    computando estatísticas globais exatas sem carregar o arquivo inteiro na memória.
    
    Retorna:
        dict com: total de linhas, contagem de normais ('R'), contagem de ataques ('T'),
        distribuição de DLCs, IDs únicos, duração temporal e taxa média de transmissão (fps).
    """
    filepath = _resolve_filepath(filepath_or_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    total = 0
    flags = Counter()
    dlcs = Counter()
    can_ids_normal = Counter()
    can_ids_attack = Counter()
    first_ts = None
    last_ts = None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            total += 1
            ts = float(row[0])
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            can_id = row[1].lower().zfill(4)
            dlc = row[2]
            flag = row[-1]

            flags[flag] += 1
            dlcs[dlc] += 1
            if flag == 'T':
                can_ids_attack[can_id] += 1
            else:
                can_ids_normal[can_id] += 1

    duration_sec = last_ts - first_ts if (last_ts and first_ts) else 0.0
    normal_count = flags.get('R', 0)
    attack_count = flags.get('T', 0)

    return {
        'filepath': filepath,
        'total': total,
        'normal': normal_count,
        'attack': attack_count,
        'attack_pct': (attack_count / total) * 100.0 if total else 0.0,
        'dlcs': dict(sorted(dlcs.items(), key=lambda x: int(x[0]))),
        'can_ids_normal': can_ids_normal,
        'can_ids_attack': can_ids_attack,
        'unique_ids_normal': len(can_ids_normal),
        'unique_ids_attack': len(can_ids_attack),
        'duration_sec': duration_sec,
        'duration_min': duration_sec / 60.0,
        'fps': total / duration_sec if duration_sec > 0 else 0.0
    }


def audit_against_paper_table_2(datasets_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Gera uma tabela comparativa auditando os datasets locais contra a Tabela 2 do artigo original.
    """
    targets = datasets_dict or DATASET_PATHS
    rows = []
    for name, path in targets.items():
        s = scan_dataset_metadata(path)
        ref = PAPER_TABLE_2.get(name, {'total': 0, 'normal': 0, 'attack': 0})
        exact_match = (
            s['total'] == ref['total'] and
            s['normal'] == ref['normal'] and
            s['attack'] == ref['attack']
        )
        rows.append({
            'Dataset': name,
            'Total Auditado': f"{s['total']:,}",
            'Total Artigo': f"{ref['total']:,}",
            'Normais (R)': f"{s['normal']:,}",
            'Normais Artigo': f"{ref['normal']:,}",
            'Ataques (T)': f"{s['attack']:,}",
            'Ataques Artigo': f"{ref['attack']:,}",
            'Taxa Ataque (%)': f"{s['attack_pct']:.2f}%",
            'Duração (min)': f"{s['duration_min']:.1f}",
            'Status': 'EXATO (100% Match)' if exact_match else 'DIVERGÊNCIA'
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 2. SISTEMA DE INDEXAÇÃO E PARTICIONAMENTO REPRODUZÍVEL COM CACHE (.NPZ)
# ==============================================================================

def get_split_indices(
    filepath_or_name: str,
    random_state: int = 42,
    cache_dir: str = "splits_cache",
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Retorna os índices das linhas correspondentes aos conjuntos de Treino, Validação e Teste
    conforme definido no artigo (Seção V-A e Tabelas 3-5).
    
    Regras metodológicas do artigo:
    - 250.000 amostras normais selecionadas aleatoriamente:
        * 225.000 (9 folds de 25k) -> Treino (apenas tráfego normal)
        * 25.000 (1 fold de 25k)   -> Validação
    - 25.000 amostras maliciosas selecionadas aleatoriamente -> Validação
    - Validação total = 50.000 amostras (25k normal + 25k ataque)
    - Teste = Todas as amostras restantes (normais e de ataque)
    - 10-Fold Cross-Validation: O conjunto normal de 250k é particionado em 10 folds
      de 25k, permitindo rotacionar o fold de validação.
      
    Utiliza cache em arquivo .npz para acelerar execuções futuras de 4s para 0.01s.
    """
    filepath = _resolve_filepath(filepath_or_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    cache_path = os.path.join(cache_dir, f"{base_name}_splits_seed{random_state}.npz")
    
    if os.path.exists(cache_path) and not force_recompute:
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}

    print(f"Indexando frames para partição metodológica em '{os.path.basename(filepath)}'...")
    t0 = time.time()
    
    normal_line_indices = []
    attack_line_indices = []

    # Varredura para coletar números de linha exatos dos frames com DLC=8
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) == 12 and parts[2] == '8':
                flag = parts[11]
                if flag == 'R':
                    normal_line_indices.append(line_idx)
                elif flag == 'T':
                    attack_line_indices.append(line_idx)

    normal_line_indices = np.array(normal_line_indices, dtype=np.int32)
    attack_line_indices = np.array(attack_line_indices, dtype=np.int32)

    rng = np.random.default_rng(random_state)
    rng.shuffle(normal_line_indices)
    rng.shuffle(attack_line_indices)

    # 250k normais para treino + validação (10 folds de 25k)
    train_val_normal_idx = normal_line_indices[:N_TOTAL_NORMAL_SAMPLED]
    test_normal_idx = normal_line_indices[N_TOTAL_NORMAL_SAMPLED:]

    # 25k ataques para validação
    val_attack_idx = attack_line_indices[:N_VAL_ATTACK]
    test_attack_idx = attack_line_indices[N_VAL_ATTACK:]

    # Padrão fold 0: primeiros 225k treino, últimos 25k validação
    train_normal_idx = train_val_normal_idx[:N_TRAIN_NORMAL]
    val_normal_idx = train_val_normal_idx[N_TRAIN_NORMAL:N_TOTAL_NORMAL_SAMPLED]

    # Validação combinada (25k normal + 25k ataque = 50.000)
    val_indices = np.concatenate([val_normal_idx, val_attack_idx])
    rng.shuffle(val_indices)

    # Treino + Validação combinados (para EDA focada sem data leakage com teste)
    train_val_indices = np.concatenate([train_val_normal_idx, val_attack_idx])
    rng.shuffle(train_val_indices)

    # Teste combinado
    test_indices = np.concatenate([test_normal_idx, test_attack_idx])
    rng.shuffle(test_indices)

    split_dict = {
        'train_normal_idx': train_normal_idx,
        'val_normal_idx': val_normal_idx,
        'val_attack_idx': val_attack_idx,
        'val_indices': val_indices,
        'train_val_normal_idx': train_val_normal_idx,
        'train_val_indices': train_val_indices,
        'test_normal_idx': test_normal_idx,
        'test_attack_idx': test_attack_idx,
        'test_indices': test_indices
    }

    # Salva cache comprimido
    np.savez_compressed(cache_path, **split_dict)
    elapsed = time.time() - t0
    print(f" -> Indexação concluída e salva em '{cache_path}' em {elapsed:.2f}s.")
    return split_dict


# ==============================================================================
# 3. CARREGAMENTO VETORIAL DE DADOS CAN POR ÍNDICES
# ==============================================================================

def load_can_records_by_indices(
    filepath_or_name: str,
    target_indices: Union[np.ndarray, List[int], set],
    add_normalized_cols: bool = True
) -> pd.DataFrame:
    """
    Carrega especificamente os registros das linhas indicadas em `target_indices`,
    decodificando bytes hexadecimais para inteiros uint8 (0 a 255) e opcionalmente
    adicionando colunas normalizadas [0.0, 1.0].
    
    Esta função é extremamente rápida e garante que apenas as amostras desejadas
    sejam trazidas para a memória RAM.
    """
    filepath = _resolve_filepath(filepath_or_name)
    target_set = set(target_indices) if not isinstance(target_indices, set) else target_indices

    timestamps = []
    can_ids = []
    data_bytes = []
    flags = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line_idx, line in enumerate(f):
            if line_idx in target_set:
                parts = line.strip().split(',')
                if len(parts) == 12 and parts[2] == '8':
                    timestamps.append(float(parts[0]))
                    can_ids.append(parts[1].lower().zfill(4))
                    data_bytes.append([int(b, 16) for b in parts[3:11]])
                    flags.append(parts[11])

    # Construção otimizada com tipos de dados explícitos
    byte_array = np.array(data_bytes, dtype=np.uint8)
    df = pd.DataFrame(byte_array, columns=BYTE_COLS)
    df.insert(0, 'Timestamp', np.array(timestamps, dtype=np.float64))
    df.insert(1, 'CAN_ID', can_ids)
    df.insert(2, 'DLC', np.uint8(8))
    df['Flag'] = flags
    df['Label'] = (df['Flag'] == 'T').astype(np.uint8)  # 0: Normal, 1: Attack

    if add_normalized_cols:
        norm_array = (byte_array / 255.0).astype(np.float32)
        df_norm = pd.DataFrame(norm_array, columns=NORM_BYTE_COLS)
        df = pd.concat([df, df_norm], axis=1)

    return df


def load_train_data(
    filepath_or_name: str,
    random_state: int = 42,
    cache_dir: str = "splits_cache",
    add_normalized_cols: bool = True
) -> pd.DataFrame:
    """
    Carrega o conjunto oficial de Treinamento (225.000 amostras 100% normais).
    """
    splits = get_split_indices(filepath_or_name, random_state=random_state, cache_dir=cache_dir)
    return load_can_records_by_indices(
        filepath_or_name,
        splits['train_normal_idx'],
        add_normalized_cols=add_normalized_cols
    )


def load_val_data(
    filepath_or_name: str,
    random_state: int = 42,
    cache_dir: str = "splits_cache",
    add_normalized_cols: bool = True
) -> pd.DataFrame:
    """
    Carrega o conjunto oficial de Validação (50.000 amostras: 25k normais + 25k ataques).
    """
    splits = get_split_indices(filepath_or_name, random_state=random_state, cache_dir=cache_dir)
    return load_can_records_by_indices(
        filepath_or_name,
        splits['val_indices'],
        add_normalized_cols=add_normalized_cols
    )


def load_train_val_data(
    filepath_or_name: str,
    random_state: int = 42,
    cache_dir: str = "splits_cache",
    add_normalized_cols: bool = True
) -> pd.DataFrame:
    """
    Carrega o conjunto de Treino + Validação combinado (250.000 normais + 25.000 ataques).
    
    Esta é a base EXCLUSIVA recomendada para a Análise Exploratória de Dados (EDA),
    assegurando que o conjunto de Teste (>3 milhões de frames) permaneça 100% isolado
    contra qualquer forma de vazamento de dados (data leakage).
    """
    splits = get_split_indices(filepath_or_name, random_state=random_state, cache_dir=cache_dir)
    return load_can_records_by_indices(
        filepath_or_name,
        splits['train_val_indices'],
        add_normalized_cols=add_normalized_cols
    )


def load_test_data(
    filepath_or_name: str,
    random_state: int = 42,
    cache_dir: str = "splits_cache",
    sample_size: Optional[int] = None,
    add_normalized_cols: bool = True
) -> pd.DataFrame:
    """
    Carrega o conjunto oficial de Teste (restante de todos os frames normais e ataques).
    Permite especificar `sample_size` para testes preliminares de inferência.
    """
    splits = get_split_indices(filepath_or_name, random_state=random_state, cache_dir=cache_dir)
    test_indices = splits['test_indices']
    if sample_size and sample_size < len(test_indices):
        rng = np.random.default_rng(random_state)
        test_indices = rng.choice(test_indices, size=sample_size, replace=False)
        
    return load_can_records_by_indices(
        filepath_or_name,
        test_indices,
        add_normalized_cols=add_normalized_cols
    )


# ==============================================================================
# 4. VALIDAÇÃO CRUZADA DE 10 FOLDS (K-FOLD CV DO ARTIGO)
# ==============================================================================

def get_kfold_cv_indices(
    train_val_normal_idx: np.ndarray,
    val_attack_idx: np.ndarray,
    fold_idx: int,
    n_splits: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera os índices de treino e validação para o fold `fold_idx` (0 a 9) da validação
    cruzada de 10 folds descrita no artigo:
    - 9 folds de amostras normais (225k) para treino.
    - 1 fold de amostras normais (25k) + 25k amostras de ataque para validação.
    """
    if not (0 <= fold_idx < n_splits):
        raise ValueError(f"fold_idx deve estar entre 0 e {n_splits - 1}, recebido: {fold_idx}")

    fold_size = len(train_val_normal_idx) // n_splits
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size

    val_norm_fold = train_val_normal_idx[val_start:val_end]
    train_norm_fold = np.concatenate([
        train_val_normal_idx[:val_start],
        train_val_normal_idx[val_end:]
    ])

    val_combined = np.concatenate([val_norm_fold, val_attack_idx])
    rng = np.random.default_rng(random_state + fold_idx)
    rng.shuffle(val_combined)

    return train_norm_fold, val_combined


# ==============================================================================
# 5. UTILITÁRIOS TEMPORAIS E DINÂMICA DE ATAQUES
# ==============================================================================

def compute_inter_arrival_times(df: pd.DataFrame, target_id: Optional[str] = None) -> pd.DataFrame:
    """
    Calcula o tempo inter-chegada (Delta T em milissegundos) para as mensagens de um DataFrame.
    Pode filtrar opcionalmente por um CAN ID específico.
    """
    if target_id is not None:
        sub = df[df['CAN_ID'] == target_id.lower().zfill(4)].copy()
    else:
        sub = df.copy()

    sub = sub.sort_values('Timestamp').reset_index(drop=True)
    sub['Delta_T_ms'] = sub['Timestamp'].diff() * 1000.0
    return sub


def detect_attack_bursts(df: pd.DataFrame, gap_threshold_sec: float = 1.0) -> List[Dict[str, Union[float, int]]]:
    """
    Identifica as rajadas contínuas (bursts) de ataque no DataFrame, agrupando mensagens
    consecutivas com Flag == 'T' separadas por um intervalo de repouso (gap).
    """
    sub = df.sort_values('Timestamp').reset_index(drop=True)
    bursts = []
    in_burst = False
    start_ts = None
    last_atk_ts = None
    pkt_count = 0

    for _, row in sub.iterrows():
        flag = row['Flag']
        ts = row['Timestamp']

        if flag == 'T':
            pkt_count += 1
            if not in_burst:
                in_burst = True
                start_ts = ts
            last_atk_ts = ts
        else:
            if in_burst:
                if ts - last_atk_ts > gap_threshold_sec:
                    bursts.append({
                        'start_ts': start_ts,
                        'end_ts': last_atk_ts,
                        'duration_sec': last_atk_ts - start_ts,
                        'packet_count': pkt_count
                    })
                    in_burst = False
                    pkt_count = 0

    if in_burst:
        bursts.append({
            'start_ts': start_ts,
            'end_ts': last_atk_ts,
            'duration_sec': last_atk_ts - start_ts,
            'packet_count': pkt_count
        })

    return bursts


from sklearn.metrics import roc_auc_score

# ==============================================================================
# 6. MÉTRICAS DE AVALIAÇÃO DE DESEMPENHO (SEÇÃO VI DO ARTIGO)
# ==============================================================================

def evaluate_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> Dict[str, Union[float, int]]:
    """
    Calcula as métricas exatas utilizadas na avaliação do artigo com a adição
    solicitada de AUC-ROC e F1-Score:
    - Acurácia (%)  [métrica principal de otimização na validação balanceada]
    - Precisão (%)
    - Recall (%)
    - F1-Score (%)
    - AUC-ROC (%)
    - Matriz de confusão (TP, FP, TN, FN)
    
    Nota: no contexto do artigo, 1 (ou 'T') representa a classe maliciosa (anomalia)
          e 0 (ou 'R') representa o tráfego normal (inlier).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = ((tp + tn) / len(y_true)) * 100.0 if len(y_true) else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    aucroc = np.nan
    if scores is not None:
        try:
            # Garante que existem pelo menos duas classes distintas para computar AUC-ROC
            if len(np.unique(y_true)) > 1:
                aucroc = float(roc_auc_score(y_true, scores) * 100.0)
        except Exception:
            aucroc = np.nan

    return {
        'Accuracy (%)': accuracy,
        'Precision (%)': precision,
        'Recall (%)': recall,
        'F1-Score (%)': f1,
        'AUC-ROC (%)': aucroc,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }
