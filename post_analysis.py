#!/usr/bin/env python3
"""
================================================================================
Módulo de Análise Estatística Pós-Experimento: post_analysis.py
Projeto: CIN0114 - Replicação de Artigo CAN Intrusion Prevention
Artigo: "An Efficient Intrusion Prevention System for CAN:
         Hindering Cyber-Attacks With a Low-Cost Platform" (IEEE Access, 2021)
================================================================================
Realiza a análise estatística rigorosa reportada na Seção VI do artigo:
1. Teste de Normalidade de D'Agostino-Pearson (scipy.stats.normaltest) nas taxas
   de detecção dos 10 folds de validação cruzada.
2. Análise de Variância One-Way ANOVA (scipy.stats.f_oneway) entre os 16 modelos.
3. Teste Post-Hoc de Tukey HSD (scipy.stats.tukey_hsd) para comparação par a par,
   determinando o número mínimo de bytes (N) estatisticamente equivalente ao modelo completo.
4. Geração das Tabelas 7, 8, 9, 10 e 11 do artigo, confrontando com o estado da arte
   (GIDS, DCNN, LSTM-IDS e MTH-IDS).
5. Renderização das Matrizes de Confusão em grade comparativa (Figura 5 do artigo).
================================================================================
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import config


# ==============================================================================
# TABELAS DE REFERÊNCIA DO ARTIGO (ESTADO DA ARTE)
# ==============================================================================

SOTA_BENCHMARKS = {
    'Fuzzing': {
        'GIDS': {'Accuracy (%)': 98.00, 'Precision (%)': 97.30, 'Recall (%)': 99.50, 'F1-Score (%)': 98.39},
        'DCNN': {'Accuracy (%)': 99.82, 'Precision (%)': 99.95, 'Recall (%)': 99.65, 'F1-Score (%)': 99.80},
        'LSTM-based IDS': {'Accuracy (%)': 99.35, 'Precision (%)': 99.36, 'Recall (%)': 99.16, 'F1-Score (%)': 99.26},
        'MTH-IDS': {'Accuracy (%)': 99.99, 'Precision (%)': 99.99, 'Recall (%)': 99.99, 'F1-Score (%)': 99.99},
        'Proposed IPS (Paper)': {'Accuracy (%)': 99.29, 'Precision (%)': 95.07, 'Recall (%)': 99.93, 'F1-Score (%)': 97.44}
    },
    'Drive Gear': {
        'GIDS': {'Accuracy (%)': 96.20, 'Precision (%)': 98.10, 'Recall (%)': 96.50, 'F1-Score (%)': 97.29},
        'DCNN': {'Accuracy (%)': 99.95, 'Precision (%)': 99.99, 'Recall (%)': 99.89, 'F1-Score (%)': 99.94},
        'LSTM-based IDS': {'Accuracy (%)': 99.76, 'Precision (%)': 99.75, 'Recall (%)': 99.68, 'F1-Score (%)': 99.72},
        'MTH-IDS': {'Accuracy (%)': 99.99, 'Precision (%)': 99.99, 'Recall (%)': 99.99, 'F1-Score (%)': 99.99},
        'Proposed IPS (Paper)': {'Accuracy (%)': 99.24, 'Precision (%)': 94.79, 'Recall (%)': 100.00, 'F1-Score (%)': 97.33}
    },
    'RPM Gauge': {
        'GIDS': {'Accuracy (%)': 98.00, 'Precision (%)': 98.30, 'Recall (%)': 99.00, 'F1-Score (%)': 98.65},
        'DCNN': {'Accuracy (%)': 99.97, 'Precision (%)': 99.99, 'Recall (%)': 99.94, 'F1-Score (%)': 99.96},
        'LSTM-based IDS': {'Accuracy (%)': 99.87, 'Precision (%)': 100.00, 'Recall (%)': 99.71, 'F1-Score (%)': 99.85},
        'MTH-IDS': {'Accuracy (%)': 99.99, 'Precision (%)': 99.99, 'Recall (%)': 99.99, 'F1-Score (%)': 99.99},
        'Proposed IPS (Paper)': {'Accuracy (%)': 99.85, 'Precision (%)': 98.97, 'Recall (%)': 100.00, 'F1-Score (%)': 99.48}
    }
}

PAPER_TABLE_8_TIMES = {
    'Fuzzing': 79.4645,
    'Drive Gear': 78.0415,
    'RPM Gauge': 77.9576
}


# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS RESULTADOS EXPERIMENTAIS
# ==============================================================================

def load_results(results_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carrega os arquivos CSV de validação cruzada e teste final."""
    cv_path = os.path.join(results_dir, "cv_grid_search_results.csv")
    test_path = os.path.join(results_dir, "final_test_results.csv")

    df_cv = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    if df_cv is None and df_test is None:
        raise FileNotFoundError(f"Nenhum arquivo de resultado encontrado em '{results_dir}'.")
    return df_cv, df_test


def get_best_models_per_dataset(df_cv: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Para cada dataset e cada modelo (16 modelos por dataset), extrai a configuração
    ótima (maior mean_val_accuracy) e a lista das acurácias dos 10 folds.
    """
    best_models = {}
    for dataset in df_cv['dataset'].unique():
        best_models[dataset] = {}
        sub_d = df_cv[df_cv['dataset'] == dataset]

        for model_name in sub_d['model_name'].unique():
            sub_m = sub_d[sub_d['model_name'] == model_name]
            best_row = sub_m.sort_values('mean_val_accuracy', ascending=False).iloc[0]

            fold_accs = []
            if 'fold_accuracies_json' in best_row and pd.notna(best_row['fold_accuracies_json']):
                try:
                    fold_accs = json.loads(best_row['fold_accuracies_json'])
                except Exception:
                    pass

            best_models[dataset][model_name] = {
                'model_name': model_name,
                'model_family': best_row['model_family'],
                'n_bytes': int(best_row['n_bytes']),
                'params': json.loads(best_row['params_json']),
                'mean_acc': float(best_row['mean_val_accuracy']),
                'std_acc': float(best_row['std_val_accuracy']),
                'mean_f1': float(best_row['mean_val_f1']),
                'mean_auc': float(best_row['mean_val_aucroc']),
                'fold_accuracies': fold_accs
            }
    return best_models


# ==============================================================================
# TESTES ESTATÍSTICOS (SEÇÃO VI-B DO ARTIGO: ANOVA & TUKEY HSD)
# ==============================================================================

def run_statistical_analysis(best_models_dict: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Executa os testes de hipótese do artigo:
    1. D'Agostino-Pearson normality test (scipy.stats.normaltest).
    2. One-Way ANOVA (scipy.stats.f_oneway).
    3. Tukey HSD Post-Hoc Test (scipy.stats.tukey_hsd).
    """
    print("\n" + "=" * 80)
    print("ANÁLISE ESTATÍSTICA DE HIPÓTESE (SEÇÃO VI-B DO ARTIGO)")
    print("=" * 80)

    for dataset, models in best_models_dict.items():
        print(f"\n[{dataset}]")
        print("-" * 50)

        # 1. Teste de Normalidade de D'Agostino-Pearson
        print("1. Teste de Normalidade (D'Agostino-Pearson):")
        normal_count = 0
        total_tested = 0
        model_fold_data = {}

        for m_name, m_info in sorted(models.items(), key=lambda x: (x[1]['model_family'], x[1]['n_bytes'])):
            accs = m_info['fold_accuracies']
            if len(accs) >= 8:
                total_tested += 1
                try:
                    stat, p_val = stats.normaltest(accs)
                    is_normal = p_val > 0.05
                    if is_normal:
                        normal_count += 1
                    model_fold_data[m_name] = accs
                    print(f"   - {m_name:12s}: Stat={stat:6.3f}, p-valor={p_val:7.4f} -> {'Normal (p > 0.05)' if is_normal else 'Não-Normal (p <= 0.05)'}")
                except Exception as e:
                    model_fold_data[m_name] = accs

        if total_tested > 0:
            print(f"   -> {normal_count}/{total_tested} modelos satisfazem a hipótese de normalidade (p > 0.05).")

        # 2. One-Way ANOVA
        if len(model_fold_data) >= 2:
            model_names_list = list(model_fold_data.keys())
            samples_list = [model_fold_data[m] for m in model_names_list]

            f_stat, anova_p = stats.f_oneway(*samples_list)
            print(f"\n2. One-Way ANOVA across all models:")
            print(f"   F-Statistic: {f_stat:.4f} | p-valor: {anova_p:.4e}")
            if anova_p < 0.05:
                print("   -> Conclusão: Há diferença estatisticamente significativa entre pelo menos dois modelos (p < 0.05).")
            else:
                print("   -> Conclusão: Não há diferença estatisticamente significativa entre os modelos (p >= 0.05).")

            # 3. Teste Post-Hoc de Tukey HSD
            if anova_p < 0.05 and hasattr(stats, 'tukey_hsd'):
                print(f"\n3. Teste Post-Hoc Tukey HSD (Avaliação de N bytes equivalentes):")
                tukey_res = stats.tukey_hsd(*samples_list)

                # Compara cada iForest_NF com o modelo completo iForest_8F
                ref_model = 'iForest_8F' if 'iForest_8F' in model_names_list else model_names_list[-1]
                ref_idx = model_names_list.index(ref_model)

                print(f"   Comparação com o modelo completo de referência ({ref_model}):")
                min_equivalent_n = 8
                for idx, m_name in enumerate(model_names_list):
                    if m_name.startswith('iForest') and m_name != ref_model:
                        p_val = tukey_res.pvalue[ref_idx, idx]
                        diff = abs(np.mean(model_fold_data[ref_model]) - np.mean(model_fold_data[m_name]))
                        is_equiv = p_val > 0.05
                        n_b = int(m_name.split('_')[1].replace('F', ''))
                        status_str = "Estatisticamente Equivalente (p > 0.05)" if is_equiv else "Significativamente Diferente (p <= 0.05)"
                        print(f"     - {m_name} vs {ref_model}: p={p_val:.4f}, diff={diff:.3f}% -> {status_str}")
                        if is_equiv and n_b < min_equivalent_n:
                            min_equivalent_n = n_b

                print(f"   🏆 Recomendação pelo Tukey HSD: iForest_{min_equivalent_n}F utiliza o menor número de bytes com desempenho equivalente a {ref_model}!")


# ==============================================================================
# GERAÇÃO DE TABELAS COMPARATIVAS (TABELAS 7, 8, 9, 10, 11)
# ==============================================================================

def generate_paper_tables(df_test: pd.DataFrame):
    """
    Gera tabelas no formato do artigo a partir do CSV de teste final.
    """
    print("\n" + "=" * 80)
    print("TABELAS DE RESULTADOS NO CONJUNTO DE TESTE (CONFRONTO COM O ARTIGO)")
    print("=" * 80)

    # 1. Tabela 7: Desempenho no dataset Fuzzing
    sub_fuzz = df_test[df_test['dataset'] == 'Fuzzing'].sort_values('test_accuracy', ascending=False)
    if not sub_fuzz.empty:
        print("\n[Tabela 7 - Fuzzing Attack Detection Performance]")
        cols_show = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_aucroc']
        cols_rename = {
            'model_name': 'Model',
            'test_accuracy': 'Accuracy (%)',
            'test_precision': 'Precision (%)',
            'test_recall': 'Recall (%)',
            'test_f1': 'F1-Score (%)',
            'test_aucroc': 'AUC-ROC (%)'
        }
        tab7 = sub_fuzz[[c for c in cols_show if c in sub_fuzz.columns]].rename(columns=cols_rename)
        print(tab7.to_string(index=False))

    # 2. Tabela 8: Tempos de Inferência (µs)
    print("\n[Tabela 8 - Tempos Médios de Inferência por Frame (µs)]")
    sub_prop = df_test[df_test['model_name'] == 'iForest_5F']
    if not sub_prop.empty:
        time_rows = []
        for _, row in sub_prop.iterrows():
            dname = row['dataset']
            us_per_frame = float(row['test_inference_us_per_frame'])
            paper_time = PAPER_TABLE_8_TIMES.get(dname, np.nan)
            meets_deadline = "SIM (< 86 µs)" if us_per_frame < 86.0 else "NÃO (> 86 µs)"
            time_rows.append({
                'Attack Type': dname,
                'Tempo Obtido (µs)': f"{us_per_frame:.2f} µs",
                'Artigo RPi4 (µs)': f"{paper_time:.2f} µs",
                'Limite IPS (86 µs)': meets_deadline
            })
        print(pd.DataFrame(time_rows).to_string(index=False))

    # 3. Tabelas 9, 10, 11: Comparativo com o Estado da Arte (GIDS, DCNN, LSTM, MTH-IDS)
    for dname, tab_num in [('Fuzzing', 9), ('Drive Gear', 10), ('RPM Gauge', 11)]:
        print(f"\n[Tabela {tab_num} - Comparativo de Detecção: {dname}]")
        sub_d = df_test[(df_test['dataset'] == dname) & (df_test['model_name'] == 'iForest_5F')]

        comp_rows = []
        benchs = SOTA_BENCHMARKS.get(dname, {})
        for mech, vals in benchs.items():
            comp_rows.append({
                'Detection Mechanism': mech,
                'Accuracy (%)': f"{vals['Accuracy (%)']:.2f}%",
                'Precision (%)': f"{vals['Precision (%)']:.2f}%",
                'Recall (%)': f"{vals['Recall (%)']:.2f}%",
                'F1-Score (%)': f"{vals['F1-Score (%)']:.2f}%"
            })

        if not sub_d.empty:
            r = sub_d.iloc[0]
            comp_rows.append({
                'Detection Mechanism': 'Nossa Replicação (iForest_5F)',
                'Accuracy (%)': f"{r['test_accuracy']:.2f}%",
                'Precision (%)': f"{r['test_precision']:.2f}%",
                'Recall (%)': f"{r['test_recall']:.2f}%",
                'F1-Score (%)': f"{r['test_f1']:.2f}%"
            })

        # Adiciona também o melhor modelo geral da replicação para comparação completa
        sub_all = df_test[df_test['dataset'] == dname].sort_values('test_accuracy', ascending=False)
        if not sub_all.empty and sub_all.iloc[0]['model_name'] != 'iForest_5F':
            best_m = sub_all.iloc[0]
            comp_rows.append({
                'Detection Mechanism': f"Nossa Replicação (Melhor: {best_m['model_name']})",
                'Accuracy (%)': f"{best_m['test_accuracy']:.2f}%",
                'Precision (%)': f"{best_m['test_precision']:.2f}%",
                'Recall (%)': f"{best_m['test_recall']:.2f}%",
                'F1-Score (%)': f"{best_m['test_f1']:.2f}%"
            })

        print(pd.DataFrame(comp_rows).to_string(index=False))


# ==============================================================================
# VISUALIZAÇÃO DAS MATRIZES DE CONFUSÃO (FIGURA 5 DO ARTIGO)
# ==============================================================================

def plot_figure_5_confusion_matrices(df_test: pd.DataFrame, dataset_name: str = "Fuzzing", output_path: str = "results/figure5_confusion_matrices.png"):
    """
    Gera o grid de matrizes de confusão dos 8 modelos principais (OCSVM 5F..8F e iForest 5F..8F)
    reproduzindo exatamente o layout da Figura 5 do artigo original.
    """
    sub_d = df_test[df_test['dataset'] == dataset_name]
    if sub_d.empty:
        return

    models_to_plot = [
        ('OCSVM_8F', 'iForest_8F'),
        ('OCSVM_7F', 'iForest_7F'),
        ('OCSVM_6F', 'iForest_6F'),
        ('OCSVM_5F', 'iForest_5F')
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    fig.suptitle(f"Matrizes de Confusão para o Ataque {dataset_name} (Figura 5 do Artigo)", fontsize=14, fontweight='bold', y=0.99)

    for row_idx, (m_ocsvm, m_iforest) in enumerate(models_to_plot):
        for col_idx, m_name in enumerate([m_ocsvm, m_iforest]):
            ax = axes[row_idx, col_idx]
            match_row = sub_d[sub_d['model_name'] == m_name]

            if not match_row.empty:
                r = match_row.iloc[0]
                tp = float(r['tp'])
                fp = float(r['fp'])
                tn = float(r['tn'])
                fn = float(r['fn'])
                total_normal = tn + fp
                total_attack = tp + fn

                # Percentuais por classe
                tn_pct = (tn / total_normal) * 100.0 if total_normal else 0.0
                fp_pct = (fp / total_normal) * 100.0 if total_normal else 0.0
                fn_pct = (fn / total_attack) * 100.0 if total_attack else 0.0
                tp_pct = (tp / total_attack) * 100.0 if total_attack else 0.0

                cm_matrix = np.array([[tn_pct, fn_pct], [fp_pct, tp_pct]])
                annot_text = np.array([
                    [f"{tn_pct:.2f}%\n(TN)", f"{fn_pct:.2f}%\n(FN)"],
                    [f"{fp_pct:.2f}%\n(FP)", f"{tp_pct:.2f}%\n(TP)"]
                ])

                sns.heatmap(
                    cm_matrix, annot=annot_text, fmt="", cmap="Blues",
                    cbar=False, ax=ax, linewidths=1, linecolor='black',
                    xticklabels=['Normal', 'Malicious'],
                    yticklabels=['Normal', 'Malicious']
                )
                ax.set_title(m_name, fontweight='bold', fontsize=12)
                ax.set_xlabel('Target Class')
                ax.set_ylabel('Output Class')
            else:
                ax.text(0.5, 0.5, f"{m_name}\n(Não executado)", ha='center', va='center')
                ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n📊 Gráfico da Figura 5 salvo com sucesso em: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Análise Estatística e Relatórios de Replicação (IEEE Access 2021)")
    parser.add_argument("--results_dir", default=config.RESULTS_DIR, help="Diretório dos CSVs de resultados")
    parser.add_argument("--fig5_dataset", default="Fuzzing", choices=config.DATASETS, help="Dataset para plotar a Figura 5")
    args = parser.parse_args()

    try:
        df_cv, df_test = load_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Execute run_experiments.py primeiro para gerar os CSVs de resultado.")
        return

    if df_cv is not None and not df_cv.empty:
        best_models = get_best_models_per_dataset(df_cv)
        run_statistical_analysis(best_models)

    if df_test is not None and not df_test.empty:
        generate_paper_tables(df_test)
        fig5_path = os.path.join(args.results_dir, "figure5_confusion_matrices.png")
        plot_figure_5_confusion_matrices(df_test, dataset_name=args.fig5_dataset, output_path=fig5_path)


if __name__ == "__main__":
    main()
