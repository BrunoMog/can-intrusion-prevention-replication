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

    def __init__(self, cv_csv_path: str, test_csv_path: str, resume: bool = True):
        self.cv_csv_path = cv_csv_path
        self.test_csv_path = test_csv_path
        self.resume = resume
        self.completed_cv_keys = set()
        self.completed_test_keys = set()

        os.makedirs(os.path.dirname(self.cv_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_csv_path), exist_ok=True)

        self._init_cv_csv()
        self._init_test_csv()

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
    n_jobs: int = -1
) -> Dict[str, float]:
    """
    Executa a validação cruzada de 10 folds para uma dada combinação de hiperparâmetros.
    Em cada fold:
    - Treino: 9 folds (225k) de amostras normais.
    - Validação: 1 fold (25k) de normais + 25k de ataques (50k amostras balanceadas).
    """
    fold_metrics_list = []
    total_fit_time = 0.0
    total_eval_time = 0.0

    fold_size = len(X_norm_pool) // n_splits
    y_val_attack = np.ones(len(val_attack_X), dtype=int)

    for fold_idx in range(n_splits):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size

        X_val_norm = X_norm_pool[val_start:val_end]
        X_train_norm = np.concatenate([
            X_norm_pool[:val_start],
            X_norm_pool[val_end:]
        ], axis=0)

        # Monta conjunto de validação balanceado (25k normais + 25k ataques)
        X_val_combined = np.concatenate([X_val_norm, val_attack_X], axis=0)
        y_val_combined = np.concatenate([np.zeros(len(X_val_norm), dtype=int), y_val_attack], axis=0)

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
    random_state: int = config.DEFAULT_SEED
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
    cp_manager = CheckpointManager(cv_csv_path, test_csv_path, resume=resume)

    param_grids = config.GRID_PRESETS[grid_preset]

    total_models = len(datasets) * len(model_families) * len(byte_configs)
    current_model_idx = 0

    for dname in datasets:
        logger.info(f"\n{'#'*80}\nPROCESSANDO DATASET: {dname}\n{'#'*80}")
        filepath = utils.DATASET_PATHS[dname]

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
                        n_jobs=n_jobs
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
                                prev_params = json.loads(prev_row['best_params_json'].iloc[0]) if 'best_params_json' in prev_row else None
                                if best_val_acc > prev_best_val + 1e-5 or best_params != prev_params:
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
        choices=config.DATASETS, help="Datasets a processar (default: todos os 3)"
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
        force_test=args.force_test
    )
