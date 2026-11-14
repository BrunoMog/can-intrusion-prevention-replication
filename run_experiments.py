#!/usr/bin/env python3
"""
================================================================================
Script de Execução de Experimentos: run_experiments.py
Projeto: CIN0114 - Replicação de Artigo CAN Intrusion Prevention
Artigo: "An Efficient Intrusion Prevention System for CAN:
         Hindering Cyber-Attacks With a Low-Cost Platform" (IEEE Access, 2021)
================================================================================
Executa o protocolo experimental completo:
- Itera sobre os datasets (Fuzzing, Drive Gear, RPM Gauge).
- Treina 16 modelos por dataset (8 OCSVM e 8 Isolation Forest, variando de 1 a 8 bytes).
- Validação Cruzada de 10 Folds com busca em grade (Grid Search).
- Otimização pela métrica primária: Acurácia (%) no conjunto de validação balanceado.
- Computação estendida de F1-Score e AUC-ROC.
- Salvamento parcial imediato (checkpointing) em CSV após cada combinação de parâmetros.
- Suporte a retormar execução (--resume) a partir do último ponto em caso de falhas ou timeout no cluster.
- Registro detalhado de logs em arquivo e console para monitoramento em tempo real.
- Avaliação final do modelo ótimo no conjunto de Teste blindado.
================================================================================
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import config
import utils


# ==============================================================================
# CONFIGURAÇÃO DE LOGGING
# ==============================================================================

def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    """Configura logger para arquivo e saída padrão (console)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"experiment_{run_name}_{timestamp}.log")
    latest_filepath = os.path.join(log_dir, f"latest_{run_name}.log")

    logger = logging.getLogger("CAN_IPS_Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Handler para arquivo
    fh = logging.FileHandler(log_filepath, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Cria ou atualiza link simbólico / cópia de 'latest'
    try:
        if os.path.exists(latest_filepath) or os.path.islink(latest_filepath):
            os.remove(latest_filepath)
        os.symlink(os.path.basename(log_filepath), latest_filepath)
    except Exception:
        pass

    logger.info(f"Logger inicializado. Gravando em: {log_filepath}")
    return logger


# ==============================================================================
# GERENCIAMENTO DE CSV E CHECKPOINTING RESILIENTE
# ==============================================================================

class CheckpointManager:
    """
    Gerencia o salvamento incremental de resultados parciais em CSV, garantindo
    que cada linha seja persistida no disco via flush e fsync, e permitindo
    retomar a execução ignorando combinações já finalizadas.
    """

    def __init__(
        self,
        cv_csv_path: str,
        test_csv_path: str,
        resume: bool = True,
        expected_metadata: Optional[Dict[str, Any]] = None,
        strict_checkpoint: bool = True,
        allow_legacy_checkpoint: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.cv_csv_path = cv_csv_path
        self.test_csv_path = test_csv_path
        self.meta_path = os.path.join(os.path.dirname(self.cv_csv_path), "checkpoint_meta.json")
        self.resume = resume
        self.expected_metadata = expected_metadata
        self.strict_checkpoint = strict_checkpoint
        self.allow_legacy_checkpoint = allow_legacy_checkpoint
        self.logger = logger
        self.completed_cv_keys = set()
        self.completed_test_keys = set()

        os.makedirs(os.path.dirname(self.cv_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_csv_path), exist_ok=True)

        self._validate_checkpoint_metadata()
        self._init_cv_csv()
        self._init_test_csv()

    def _warn_or_fail(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(f"[CHECKPOINT WARNING] {msg}")
        if self.strict_checkpoint:
            raise RuntimeError(msg)

    def _validate_checkpoint_metadata(self) -> None:
        """Valida se seed, cv_folds, grade e SHA-256 integral de datasets batem ao retomar CSV."""
        if not self.expected_metadata:
            return

        if os.path.exists(self.cv_csv_path) and self.resume:
            if os.path.exists(self.meta_path):
                try:
                    with open(self.meta_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                    for key in ('cv_folds', 'random_state', 'grid_preset', 'primary_metric'):
                        if saved.get(key) != self.expected_metadata.get(key):
                            self._warn_or_fail(
                                f"Divergência de '{key}' no checkpoint ({saved.get(key)} != {self.expected_metadata.get(key)})."
                            )
                    saved_fps = saved.get('dataset_fingerprints', {})
                    cur_fps = self.expected_metadata.get('dataset_fingerprints', {})
                    for ds_name, cur_fp in cur_fps.items():
                        if ds_name in saved_fps:
                            for fk in ('file_size_bytes', 'file_sha256', 'split_config'):
                                if saved_fps[ds_name].get(fk) != cur_fp.get(fk):
                                    self._warn_or_fail(
                                        f"Fingerprint de '{ds_name}' diverge em '{fk}' contra checkpoint salvo."
                                    )
                except RuntimeError:
                    raise
                except (OSError, ValueError) as exc:
                    self._warn_or_fail(f"Falha ao ler '{self.meta_path}': {exc}")
            else:
                msg = (
                    f"Checkpoint CSV '{self.cv_csv_path}' não possui 'checkpoint_meta.json'. "
                    "Use --allow-legacy-checkpoint ou --no-resume para prosseguir."
                )
                if self.allow_legacy_checkpoint:
                    if self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print(f"[CHECKPOINT INFO] {msg}")
                else:
                    self._warn_or_fail(msg)
        elif not os.path.exists(self.cv_csv_path) or not self.resume:
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.expected_metadata, f, indent=2, sort_keys=True)

    def _init_cv_csv(self):
        cv_headers = [
            "timestamp", "dataset", "model_family", "n_bytes", "model_name",
            "params_json", "cv_folds",
            "mean_val_accuracy", "std_val_accuracy",
            "mean_val_precision", "std_val_precision",
            "mean_val_recall", "std_val_recall",
            "mean_val_f1", "std_val_f1",
            "mean_val_aucroc", "std_val_aucroc",
            "fold_accuracies_json", "fold_f1_json", "fold_aucroc_json",
            "total_fit_time_sec", "total_eval_time_sec"
        ]
        if not os.path.exists(self.cv_csv_path):
            with open(self.cv_csv_path, 'w', encoding='utf-8') as f:
                f.write(",".join(cv_headers) + "\n")
        elif self.resume:
            df = pd.read_csv(self.cv_csv_path)
            if (
                self.expected_metadata
                and 'cv_folds' in df.columns
                and not df.empty
                and 'cv_folds' in self.expected_metadata
            ):
                existing_folds = {int(v) for v in df['cv_folds'].dropna().unique()}
                if int(self.expected_metadata['cv_folds']) not in existing_folds:
                    self._warn_or_fail(
                        f"cv_folds={self.expected_metadata['cv_folds']} diverge do CSV existente ({sorted(existing_folds)})."
                    )
            for _, row in df.iterrows():
                key = (str(row['dataset']), str(row['model_name']), str(row['params_json']))
                self.completed_cv_keys.add(key)

    def _init_test_csv(self):
        test_headers = [
            "timestamp", "dataset", "model_family", "n_bytes", "model_name",
            "best_params_json", "best_val_accuracy", "best_val_f1", "best_val_aucroc",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_aucroc",
            "test_samples_count", "test_time_sec", "test_inference_us_per_frame",
            "tp", "fp", "tn", "fn"
        ]
        if not os.path.exists(self.test_csv_path):
            with open(self.test_csv_path, 'w', encoding='utf-8') as f:
                f.write(",".join(test_headers) + "\n")
        elif self.resume:
            df = pd.read_csv(self.test_csv_path)
            for _, row in df.iterrows():
                key = (str(row['dataset']), str(row['model_name']))
                self.completed_test_keys.add(key)

    def is_cv_completed(self, dataset: str, model_name: str, params_json: str) -> bool:
        if not self.resume:
            return False
        return (dataset, model_name, params_json) in self.completed_cv_keys

    def is_test_completed(self, dataset: str, model_name: str) -> bool:
        if not self.resume:
            return False
        return (dataset, model_name) in self.completed_test_keys

    def append_cv_result(self, row_dict: Dict[str, Any]):
        """Grava uma linha no CSV de validação cruzada com descarga síncrona no disco."""
        df_row = pd.DataFrame([row_dict])
        with open(self.cv_csv_path, 'a', encoding='utf-8') as f:
            df_row.to_csv(f, header=False, index=False)
            f.flush()
            os.fsync(f.fileno())
        key = (str(row_dict['dataset']), str(row_dict['model_name']), str(row_dict['params_json']))
        self.completed_cv_keys.add(key)

    def append_test_result(self, row_dict: Dict[str, Any]):
        """Grava ou atualiza uma linha no CSV de teste final com descarga síncrona no disco."""
        if os.path.exists(self.test_csv_path) and os.path.getsize(self.test_csv_path) > 0:
            df = pd.read_csv(self.test_csv_path)
            mask = (df['dataset'] == row_dict['dataset']) & (df['model_name'] == row_dict['model_name'])
            if mask.any():
                df = df[~mask]
            df_new = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
            df_new.to_csv(self.test_csv_path, index=False)
        else:
            df_row = pd.DataFrame([row_dict])
            df_row.to_csv(self.test_csv_path, index=False)
        key = (str(row_dict['dataset']), str(row_dict['model_name']))
        self.completed_test_keys.add(key)


# ==============================================================================
# MOTOR DE TREINAMENTO E VALIDAÇÃO CRUZADA (10 FOLDS)
# ==============================================================================

def generate_param_combinations(grid_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Gera o produto cartesiano das listas de parâmetros de uma grade."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def instantiate_model(model_family: str, params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1):
    """Instancia o modelo especificado com os hiperparâmetros informados."""
    if model_family == 'OCSVM':
        # OneClassSVM é baseado em libsvm (núcleo único C++)
        return OneClassSVM(**params)
    elif model_family == 'iForest':
        return IsolationForest(**params, random_state=random_state, n_jobs=n_jobs)
    else:
        raise ValueError(f"Família de modelo desconhecida: {model_family}")


def fit_predict_evaluate(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[Dict[str, float], float, float]:
    """
    Treina o modelo nos dados normais (X_train) e avalia nos dados de validação (X_val).
    Retorna (dicionário_de_métricas, tempo_treino_segundos, tempo_inferencia_segundos).
    """
    t_start_fit = time.time()
    model.fit(X_train)
    fit_time = time.time() - t_start_fit

    t_start_eval = time.time()
    raw_preds = model.predict(X_val)
    # No scikit-learn: +1 é inlier (normal), -1 é outlier (anomalia/ataque)
    y_pred = (raw_preds == -1).astype(int)

    # Anomaly score para AUC-ROC: quanto maior o score, mais provável ser ataque
    scores = -model.decision_function(X_val)
    eval_time = time.time() - t_start_eval

    metrics = utils.evaluate_detection_metrics(y_val, y_pred, scores=scores)
    return metrics, fit_time, eval_time


def run_kfold_cv_for_params(
    model_family: str,
    params: Dict[str, Any],
    X_norm_pool: np.ndarray,
    val_attack_X: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    iforest_fold_cache: Optional[Dict[Tuple[Any, Any], List[Dict[str, Any]]]] = None,
    partition_val_attacks_across_folds: bool = False
) -> Dict[str, float]:
    """
    Executa a validação cruzada de 10 folds para uma dada combinação de hiperparâmetros.
    Em cada fold:
    - Datasets originais: Treino = 9 folds (225k) normais; Validação = 1 fold (25k) normal + 25k ataques (50k balanceadas).
    - Speedometer (Opção B, partition_val_attacks_across_folds=True):
      Treino = 9 folds (22.500) normais; Validação = 250 normais do 10º fold + 250 ataques do fold (500 balanceadas).
    """
    fold_metrics_list = []
    total_fit_time = 0.0
    total_eval_time = 0.0

    fold_size = len(X_norm_pool) // n_splits
    atk_fold_size = (len(val_attack_X) // n_splits) if partition_val_attacks_across_folds else len(val_attack_X)

    # Otimização matemática exata para IsolationForest: as árvores dependem apenas de
    # (n_estimators, max_samples), enquanto 'contamination' apenas define o percentil (offset_)
    if model_family == 'iForest' and iforest_fold_cache is not None:
        tree_key = (params.get('n_estimators', 100), params.get('max_samples', 'auto'))
        if tree_key not in iforest_fold_cache:
            cached_folds = []
            base_params = {**params, 'contamination': 'auto'}
            for fold_idx in range(n_splits):
                val_start = fold_idx * fold_size
                val_end = val_start + fold_size
                X_train_norm = np.concatenate([X_norm_pool[:val_start], X_norm_pool[val_end:]], axis=0)
                if partition_val_attacks_across_folds:
                    X_val_norm = X_norm_pool[val_start:val_start + atk_fold_size]
                    X_val_atk = val_attack_X[fold_idx * atk_fold_size:(fold_idx + 1) * atk_fold_size]
                else:
                    X_val_norm = X_norm_pool[val_start:val_end]
                    X_val_atk = val_attack_X
                y_val_attack_fold = np.ones(len(X_val_atk), dtype=int)
                X_val_combined = np.concatenate([X_val_norm, X_val_atk], axis=0)
                y_val_combined = np.concatenate([np.zeros(len(X_val_norm), dtype=int), y_val_attack_fold], axis=0)

                rng = np.random.default_rng(random_state + fold_idx)
                val_perm = rng.permutation(len(X_val_combined))
                X_val_combined = X_val_combined[val_perm]
                y_val_combined = y_val_combined[val_perm]

                t0_fit = time.time()
                model = instantiate_model(model_family, base_params, random_state=random_state + fold_idx, n_jobs=n_jobs)
                model.fit(X_train_norm)
                s_train = model.score_samples(X_train_norm)
                fit_t = time.time() - t0_fit

                t0_eval = time.time()
                s_val = model.score_samples(X_val_combined)
                eval_t = time.time() - t0_eval

                cached_folds.append({
                    's_train': s_train,
                    's_val': s_val,
                    'y_val': y_val_combined,
                    'fit_t': fit_t,
                    'eval_t': eval_t
                })
            iforest_fold_cache[tree_key] = cached_folds

        cont = params.get('contamination', 'auto')
        for fdata in iforest_fold_cache[tree_key]:
            offset = -0.5 if cont == 'auto' else float(np.percentile(fdata['s_train'], 100.0 * float(cont)))
            dec = fdata['s_val'] - offset
            y_pred = (dec < 0).astype(int)
            scores = -dec
            metrics = utils.evaluate_detection_metrics(fdata['y_val'], y_pred, scores=scores)
            fold_metrics_list.append(metrics)
            total_fit_time += fdata['fit_t']
            total_eval_time += fdata['eval_t']
    else:
        for fold_idx in range(n_splits):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size

            X_train_norm = np.concatenate([
                X_norm_pool[:val_start],
                X_norm_pool[val_end:]
            ], axis=0)

            if partition_val_attacks_across_folds:
                X_val_norm = X_norm_pool[val_start:val_start + atk_fold_size]
                X_val_atk = val_attack_X[fold_idx * atk_fold_size:(fold_idx + 1) * atk_fold_size]
            else:
                X_val_norm = X_norm_pool[val_start:val_end]
                X_val_atk = val_attack_X
            y_val_attack_fold = np.ones(len(X_val_atk), dtype=int)

            # Monta conjunto de validação balanceado
            X_val_combined = np.concatenate([X_val_norm, X_val_atk], axis=0)
            y_val_combined = np.concatenate([np.zeros(len(X_val_norm), dtype=int), y_val_attack_fold], axis=0)

            # Embaralha validação
            rng = np.random.default_rng(random_state + fold_idx)
            val_perm = rng.permutation(len(X_val_combined))
            X_val_combined = X_val_combined[val_perm]
            y_val_combined = y_val_combined[val_perm]

            # Instancia e avalia
            model = instantiate_model(model_family, params, random_state=random_state + fold_idx, n_jobs=n_jobs)
            metrics, fit_t, eval_t = fit_predict_evaluate(model, X_train_norm, X_val_combined, y_val_combined)

            fold_metrics_list.append(metrics)
            total_fit_time += fit_t
            total_eval_time += eval_t

    # Calcula médias e desvios padrão agregados sobre os 10 folds
    summary = {
        'mean_val_accuracy': float(np.mean([m['Accuracy (%)'] for m in fold_metrics_list])),
        'std_val_accuracy': float(np.std([m['Accuracy (%)'] for m in fold_metrics_list])),
        'mean_val_precision': float(np.mean([m['Precision (%)'] for m in fold_metrics_list])),
        'std_val_precision': float(np.std([m['Precision (%)'] for m in fold_metrics_list])),
        'mean_val_recall': float(np.mean([m['Recall (%)'] for m in fold_metrics_list])),
        'std_val_recall': float(np.std([m['Recall (%)'] for m in fold_metrics_list])),
        'mean_val_f1': float(np.mean([m['F1-Score (%)'] for m in fold_metrics_list])),
        'std_val_f1': float(np.std([m['F1-Score (%)'] for m in fold_metrics_list])),
        'mean_val_aucroc': float(np.mean([m['AUC-ROC (%)'] for m in fold_metrics_list])),
        'std_val_aucroc': float(np.std([m['AUC-ROC (%)'] for m in fold_metrics_list])),
        'fold_accuracies_json': json.dumps([float(m['Accuracy (%)']) for m in fold_metrics_list]),
        'fold_f1_json': json.dumps([float(m['F1-Score (%)']) for m in fold_metrics_list]),
        'fold_aucroc_json': json.dumps([float(m['AUC-ROC (%)']) for m in fold_metrics_list]),
        'total_fit_time_sec': float(total_fit_time),
        'total_eval_time_sec': float(total_eval_time)
    }
    return summary


# ==============================================================================
# PIPELINE COMPLETO DE EXPERIMENTOS
# ==============================================================================

def run_experiment_pipeline(
    datasets: List[str],
    model_families: List[str],
    byte_configs: List[int],
    grid_preset: str = 'default',
    cv_folds: int = 10,
    n_jobs: int = -1,
    resume: bool = True,
    evaluate_test: bool = True,
    test_sample_size: Optional[int] = None,
    output_dir: str = config.RESULTS_DIR,
    log_dir: str = config.LOGS_DIR,
    models_dir: str = config.MODELS_DIR,
    save_models: bool = False,
    force_test: bool = False,
    random_state: int = config.DEFAULT_SEED,
    strict_checkpoint: bool = True,
    allow_legacy_checkpoint: bool = False,
):
    """
    Executa o pipeline completo de experimentos com salvamento incremental e logs.
    """
    run_id = f"{len(datasets)}ds_{grid_preset}"
    logger = setup_logger(log_dir, run_id)

    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE EXPERIMENTOS (REPLICAÇÃO IEEE ACCESS 2021)")
    logger.info("=" * 80)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Famílias de Modelo: {model_families}")
    logger.info(f"Configurações de Bytes: {byte_configs}")
    logger.info(f"Grade de Hiperparâmetros: '{grid_preset}'")
    logger.info(f"Folds de Validação Cruzada: {cv_folds}")
    logger.info(f"Métrica Primária de Otimização: {config.PRIMARY_METRIC}")
    logger.info(f"Modo Resume Ativo: {resume}")
    logger.info(f"Avaliar Conjunto de Teste: {evaluate_test}")
    logger.info(f"Diretório de Resultados: {output_dir}")
    logger.info("=" * 80)

    cv_csv_path = os.path.join(output_dir, "cv_grid_search_results.csv")
    test_csv_path = os.path.join(output_dir, "final_test_results.csv")
    dataset_fingerprints = {}
    for dname in datasets:
        ds_file = utils.DATASET_PATHS.get(dname, '')
        if ds_file and os.path.exists(ds_file):
            dataset_fingerprints[dname] = utils.compute_dataset_fingerprint(dname, random_state=random_state)
    expected_metadata = {
        "cv_folds": int(cv_folds),
        "random_state": int(random_state),
        "grid_preset": str(grid_preset),
        "primary_metric": str(config.PRIMARY_METRIC),
        "dataset_fingerprints": dataset_fingerprints,
    }
    cp_manager = CheckpointManager(
        cv_csv_path,
        test_csv_path,
        resume=resume,
        expected_metadata=expected_metadata,
        strict_checkpoint=strict_checkpoint,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
        logger=logger,
    )

    param_grids = config.GRID_PRESETS[grid_preset]

    total_models = len(datasets) * len(model_families) * len(byte_configs)
    current_model_idx = 0

    for dname in datasets:
        logger.info(f"\n{'#'*80}\nPROCESSANDO DATASET: {dname}\n{'#'*80}")
        filepath = utils._resolve_filepath(dname)
        split_cfg = utils._get_split_config(dname)
        part_val_atk = bool(split_cfg['partition_val_attacks_across_folds'])

        # Obtém índices particionados do cache (.npz)
        splits = utils.get_split_indices(filepath, random_state=random_state)
        train_val_normal_idx = splits['train_val_normal_idx']
        val_attack_idx = splits['val_attack_idx']

        logger.info(f"Carregando {len(train_val_normal_idx):,} amostras normais e {len(val_attack_idx):,} de ataque para CV...")
        t_load = time.time()
        df_norm_pool = utils.load_can_records_by_indices(filepath, train_val_normal_idx, add_normalized_cols=True)
        df_val_atk = utils.load_can_records_by_indices(filepath, val_attack_idx, add_normalized_cols=True)
        logger.info(f"Dados de Treino/Validação carregados em {time.time()-t_load:.2f}s.")

        # Carrega dados de teste se a avaliação final estiver ativada
        df_test = None
        if evaluate_test:
            logger.info("Carregando conjunto de Teste blindado para avaliação final dos modelos ótimos...")
            t_test_load = time.time()
            df_test = utils.load_test_data(filepath, random_state=random_state, sample_size=test_sample_size)
            logger.info(f"Conjunto de Teste ({len(df_test):,} frames) carregado em {time.time()-t_test_load:.2f}s.")

        for n_bytes in byte_configs:
            # Colunas de características correspondentes aos primeiros N bytes normalizados
            feature_cols = [f'D{i}_norm' for i in range(n_bytes)]
            X_norm_pool = df_norm_pool[feature_cols].values
            val_attack_X = df_val_atk[feature_cols].values

            X_test = df_test[feature_cols].values if df_test is not None else None
            y_test = df_test['Label'].values if df_test is not None else None

            for family in model_families:
                current_model_idx += 1
                model_name = f"{family}_{n_bytes}F"
                grid = param_grids[family]
                combinations = generate_param_combinations(grid)

                logger.info("-" * 70)
                logger.info(f"[{current_model_idx}/{total_models}] Modelo: {model_name} | Dataset: {dname} | Features: {feature_cols}")
                logger.info(f"Grade com {len(combinations)} combinações a avaliar em {cv_folds}-Fold CV...")

                best_params = None
                best_val_score = -float('inf')
                best_cv_summary = None
                iforest_fold_cache: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}

                for p_idx, params in enumerate(combinations, start=1):
                    params_json = json.dumps(params, sort_keys=True)

                    if cp_manager.is_cv_completed(dname, model_name, params_json):
                        logger.info(f" -> [CHECKPOINT] Combinação {p_idx}/{len(combinations)} já avaliada: {params_json} (Pulando)")
                        continue

                    logger.info(f" -> Executando {cv_folds}-Fold CV {p_idx}/{len(combinations)}: {params_json}")
                    t_cv = time.time()
                    cv_res = run_kfold_cv_for_params(
                        model_family=family,
                        params=params,
                        X_norm_pool=X_norm_pool,
                        val_attack_X=val_attack_X,
                        n_splits=cv_folds,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        iforest_fold_cache=iforest_fold_cache,
                        partition_val_attacks_across_folds=part_val_atk
                    )
                    cv_time = time.time() - t_cv

                    logger.info(
                        f"    Resultado: Acc={cv_res['mean_val_accuracy']:.2f}% (±{cv_res['std_val_accuracy']:.2f}) | "
                        f"F1={cv_res['mean_val_f1']:.2f}% | AUC={cv_res['mean_val_aucroc']:.2f}% | "
                        f"Tempo={cv_time:.2f}s"
                    )

                    # Registro parcial imediato no CSV
                    cv_record = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dname,
                        "model_family": family,
                        "n_bytes": n_bytes,
                        "model_name": model_name,
                        "params_json": params_json,
                        "cv_folds": cv_folds,
                        **cv_res
                    }
                    cp_manager.append_cv_result(cv_record)

                # ==============================================================
                # SELEÇÃO DO MODELO ÓTIMO E AVALIAÇÃO NO TESTE BLINDADO
                # ==============================================================
                if evaluate_test:
                    df_cv_all = pd.read_csv(cv_csv_path)
                    sub_cv = df_cv_all[(df_cv_all['dataset'] == dname) & (df_cv_all['model_name'] == model_name)]
                    if sub_cv.empty:
                        logger.warning(f"Nenhum resultado de CV encontrado para {dname} - {model_name}.")
                        continue

                    # Considera os parâmetros da grade ativa se tiverem recall válido (>10%);
                    # caso contrário, faz fallback para os modelos com recall válido (>10%) já avaliados.
                    current_param_keys = set(json.dumps(c, sort_keys=True) for c in combinations)
                    sub_cv_filtered = sub_cv[sub_cv['params_json'].apply(lambda x: json.dumps(json.loads(x), sort_keys=True) in current_param_keys)]
                    sub_cv_valid = sub_cv_filtered[sub_cv_filtered['mean_val_recall'] >= 10.0]
                    if not sub_cv_valid.empty:
                        sub_cv = sub_cv_valid
                    else:
                        sub_all_valid = sub_cv[sub_cv['mean_val_recall'] >= 10.0]
                        if not sub_all_valid.empty:
                            sub_cv = sub_all_valid

                    best_row = sub_cv.sort_values(['mean_val_accuracy', 'mean_val_f1'], ascending=[False, False]).iloc[0]
                    best_params = json.loads(best_row['params_json'])
                    best_val_acc = float(best_row['mean_val_accuracy'])
                    best_val_f1 = float(best_row['mean_val_f1'])
                    best_val_auc = float(best_row['mean_val_aucroc'])

                    needs_test = force_test or not cp_manager.is_test_completed(dname, model_name)
                    if not needs_test and os.path.exists(test_csv_path):
                        try:
                            df_test_prev = pd.read_csv(test_csv_path)
                            prev_row = df_test_prev[(df_test_prev['dataset'] == dname) & (df_test_prev['model_name'] == model_name)]
                            if not prev_row.empty:
                                prev_best_val = float(prev_row['best_val_accuracy'].iloc[0])
                                prev_test_rec = float(prev_row['test_recall'].iloc[0]) if 'test_recall' in prev_row else 100.0
                                prev_params = json.loads(prev_row['best_params_json'].iloc[0]) if 'best_params_json' in prev_row else None
                                if best_val_acc > prev_best_val + 1e-5 or prev_test_rec < 1.0 or (best_params != prev_params and best_val_acc >= 60.0):
                                    logger.info(f"⚡ Novo modelo de CV superior detectado para {model_name} (Acc: {best_val_acc:.2f}% vs {prev_best_val:.2f}%). Re-avaliando no Teste!")
                                    needs_test = True
                        except Exception:
                            pass

                    if not needs_test:
                        logger.info(f"Teste já concluído para {model_name} ({dname}) e parâmetros ótimos inalterados. Pulando.")
                        continue

                    logger.info(f"Identificando melhores hiperparâmetros para {model_name} no dataset {dname}...")
                    logger.info(f"🏆 Melhor configuração selecionada por Acurácia na Validação: {best_params} (Val Acc: {best_val_acc:.2f}%)")
                    logger.info(f"Treinando modelo final em todo o conjunto de treino (225k) e avaliando no Teste...")

                    # Treina no conjunto completo de 225k amostras normais
                    train_normal_idx = splits['train_normal_idx']
                    df_train_full = utils.load_can_records_by_indices(filepath, train_normal_idx, add_normalized_cols=True)
                    X_train_full = df_train_full[feature_cols].values

                    final_model = instantiate_model(family, best_params, random_state=random_state, n_jobs=n_jobs)
                    t0_fit = time.time()
                    final_model.fit(X_train_full)
                    t_final_fit = time.time() - t0_fit

                    if save_models:
                        os.makedirs(models_dir, exist_ok=True)
                        model_path = os.path.join(models_dir, f"{dname}_{model_name}_best.joblib")
                        import joblib
                        joblib.dump(final_model, model_path)
                        logger.info(f"💾 Modelo persistido em: {model_path}")

                    # Avaliação no Teste
                    t0_test = time.time()
                    raw_test_preds = final_model.predict(X_test)
                    y_test_pred = (raw_test_preds == -1).astype(int)
                    test_scores = -final_model.decision_function(X_test)
                    t_test_eval = time.time() - t0_test

                    test_metrics = utils.evaluate_detection_metrics(y_test, y_test_pred, scores=test_scores)
                    inference_us_per_frame = (t_test_eval / len(y_test)) * 1_000_000.0

                    logger.info(
                        f"📊 [DESEMPENHO NO TESTE - {model_name} ({dname})]\n"
                        f"    - Acurácia:  {test_metrics['Accuracy (%)']:.2f}%\n"
                        f"    - Precisão:  {test_metrics['Precision (%)']:.2f}%\n"
                        f"    - Recall:    {test_metrics['Recall (%)']:.2f}%\n"
                        f"    - F1-Score:  {test_metrics['F1-Score (%)']:.2f}%\n"
                        f"    - AUC-ROC:   {test_metrics['AUC-ROC (%)']:.2f}%\n"
                        f"    - Tempo Inferência: {inference_us_per_frame:.2f} µs/frame (Total: {t_test_eval:.2f}s para {len(y_test):,} frames)\n"
                        f"    - Matriz de Confusão: TP={test_metrics['TP']:,}, FP={test_metrics['FP']:,}, TN={test_metrics['TN']:,}, FN={test_metrics['FN']:,}"
                    )

                    test_record = {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dname,
                        "model_family": family,
                        "n_bytes": n_bytes,
                        "model_name": model_name,
                        "best_params_json": json.dumps(best_params),
                        "best_val_accuracy": best_val_acc,
                        "best_val_f1": best_val_f1,
                        "best_val_aucroc": best_val_auc,
                        "test_accuracy": test_metrics['Accuracy (%)'],
                        "test_precision": test_metrics['Precision (%)'],
                        "test_recall": test_metrics['Recall (%)'],
                        "test_f1": test_metrics['F1-Score (%)'],
                        "test_aucroc": test_metrics['AUC-ROC (%)'],
                        "test_samples_count": len(y_test),
                        "test_time_sec": t_test_eval,
                        "test_inference_us_per_frame": inference_us_per_frame,
                        "tp": test_metrics['TP'],
                        "fp": test_metrics['FP'],
                        "tn": test_metrics['TN'],
                        "fn": test_metrics['FN']
                    }
                    cp_manager.append_test_result(test_record)
                elif evaluate_test:
                    logger.info(f" -> [CHECKPOINT] Avaliação no Teste já concluída para {dname} - {model_name} (Pulando)")

    logger.info("\n" + "=" * 80)
    logger.info("TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO!")
    logger.info(f"Resultados de Validação Cruzada: {cv_csv_path}")
    logger.info(f"Resultados Finais de Teste: {test_csv_path}")
    logger.info("=" * 80)


# ==============================================================================
# CLI (INTERFACE DE LINHA DE COMANDO)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Execução de Experimentos para Replicação do Artigo CAN Intrusion Prevention (IEEE Access 2021)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=config.DATASETS,
        choices=config.ALL_DATASETS, help="Datasets a processar (default: todos os 3 do artigo; suporta também 'Speedometer')"
    )
    parser.add_argument(
        "--models", nargs="+", default=config.MODEL_FAMILIES,
        choices=config.MODEL_FAMILIES, help="Famílias de modelos (default: OCSVM e iForest)"
    )
    parser.add_argument(
        "--bytes", nargs="+", type=int, default=config.BYTE_CONFIGS,
        choices=config.BYTE_CONFIGS, help="Configurações de N bytes (default: 1 a 8)"
    )
    parser.add_argument(
        "--grid", default="default", choices=list(config.GRID_PRESETS.keys()),
        help="Predefinição da grade de parâmetros: 'smoke' (rápido/teste), 'default' (padrão balanceado), 'full' (cluster)"
    )
    parser.add_argument(
        "--folds", type=int, default=config.DEFAULT_CV_FOLDS,
        help=f"Número de folds da validação cruzada (default: {config.DEFAULT_CV_FOLDS})"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Número de workers paralelos para iForest (-1 = todos os núcleos de CPU)"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Desativa o modo resume e re-executa todos os experimentos"
    )
    parser.add_argument(
        "--no-test", dest="evaluate_test", action="store_false",
        help="Pula a avaliação final no conjunto de Teste"
    )
    parser.add_argument(
        "--test_sample_size", type=int, default=None,
        help="Sub-amostra opcional do conjunto de Teste para avaliação rápida (default: None = todo o teste)"
    )
    parser.add_argument(
        "--output_dir", default=config.RESULTS_DIR,
        help="Diretório onde os CSVs de resultados serão persistidos"
    )
    parser.add_argument(
        "--log_dir", default=config.LOGS_DIR,
        help="Diretório onde os arquivos de log serão persistidos"
    )
    parser.add_argument(
        "--models_dir", default=config.MODELS_DIR,
        help="Diretório onde os modelos treinados (.joblib) serão persistidos"
    )
    parser.add_argument(
        "--save_models", action="store_true", default=False,
        help="Salva os modelos ótimos finais treinados em disco (.joblib)"
    )
    parser.add_argument(
        "--force_test", action="store_true", default=False,
        help="Força re-avaliação do melhor modelo no conjunto de teste mesmo se já concluído anteriormente"
    )
    parser.add_argument(
        "--seed", type=int, default=config.DEFAULT_SEED,
        help="Semente aleatória para partições e modelos"
    )
    parser.add_argument(
        "--strict-checkpoint", dest="strict_checkpoint", action="store_true", default=True,
        help="Aborta se metadados do checkpoint divergirem (comportamento padrão ativo)"
    )
    parser.add_argument(
        "--no-strict-checkpoint", dest="strict_checkpoint", action="store_false",
        help="Rebaixa divergências de metadados do checkpoint para avisos"
    )
    parser.add_argument(
        "--allow-legacy-checkpoint", action="store_true", default=False,
        help="Permite retomar CSV legado sem arquivo checkpoint_meta.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment_pipeline(
        datasets=args.datasets,
        model_families=args.models,
        byte_configs=args.bytes,
        grid_preset=args.grid,
        cv_folds=args.folds,
        n_jobs=args.n_jobs,
        resume=args.resume,
        evaluate_test=args.evaluate_test,
        test_sample_size=args.test_sample_size,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        models_dir=args.models_dir,
        save_models=args.save_models,
        force_test=args.force_test,
        random_state=args.seed,
        strict_checkpoint=args.strict_checkpoint,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
