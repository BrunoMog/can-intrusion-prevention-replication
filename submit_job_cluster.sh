#!/bin/bash
# ==============================================================================
# Script de Submissão de Job para Cluster HPC via SLURM
# Projeto: CIN0114 - Replicação de Artigo CAN Intrusion Prevention
# Artigo: "An Efficient Intrusion Prevention System for CAN:
#          Hindering Cyber-Attacks With a Low-Cost Platform" (IEEE Access, 2021)
# ==============================================================================
# Instruções de Uso:
#   1. Para submeter todos os datasets sequencialmente em 1 nó:
#      sbatch submit_job_cluster.sh
#
#   2. Para submeter como Job Array (1 nó paralelo por dataset - RECOMENDADO):
#      sbatch --array=0-2 submit_job_cluster.sh
#
#   3. Para rodar um dataset específico diretamente (ex: Fuzzing):
#      bash submit_job_cluster.sh "Fuzzing"
# ==============================================================================

#SBATCH --job-name=can_ips_rep
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --partition=short-simple
#SBATCH --nodelist=cluster-node6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # Ajuste conforme a quantidade de núcleos por nó
#SBATCH --mem=32G                  # Memória RAM alocada
#SBATCH --time=24:00:00            # Limite de tempo (Walltime)

# Carrega módulos do cluster se necessário (exemplo comum em clusters):
 module load python3.10

# Caminho para o ambiente virtual criado para o projeto (fora da pasta)
VENV_PATH="$HOME/venvs/cin0114_venv"
PROJECT_DIR="/home/CIN/basb/projeto_CIN0114"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "[HPC] Venv não encontrada. Criando ambiente virtual em $VENV_PATH..."
    mkdir -p "$(dirname "$VENV_PATH")"
    if ! python -m venv "$VENV_PATH"; then
        echo "[ERRO] Não foi possível criar a venv em $VENV_PATH." >&2
        exit 1
    fi
fi

echo "[HPC] Instalando dependências de $PROJECT_DIR/requirements.txt..."
if ! "$VENV_PATH/bin/python" -m pip install --upgrade pip ||
   ! "$VENV_PATH/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"; then
    echo "[ERRO] Não foi possível instalar as dependências do projeto." >&2
    exit 1
fi

echo "[HPC] Ativando ambiente virtual em $VENV_PATH..."
source "$VENV_PATH/bin/activate"

# Lista de datasets
ALL_DATASETS=("Fuzzing" "Drive Gear" "RPM Gauge")

# Se for disparado como SLURM Job Array (--array=0-2), seleciona o dataset da tarefa
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    TARGET_DATASET="${ALL_DATASETS[$SLURM_ARRAY_TASK_ID]}"
    echo "[HPC] Executando Tarefa do Array: ID=$SLURM_ARRAY_TASK_ID -> Dataset: '$TARGET_DATASET'"
    DATASET_ARG="--datasets \"$TARGET_DATASET\""
elif [ -n "$1" ]; then
    TARGET_DATASET="$1"
    echo "[HPC] Dataset especificado via argumento: '$TARGET_DATASET'"
    DATASET_ARG="--datasets \"$TARGET_DATASET\""
else
    echo "[HPC] Executando todos os 3 datasets sequencialmente..."
    DATASET_ARG="--datasets \"Fuzzing\" \"Drive Gear\" \"RPM Gauge\""
fi

echo "[HPC] Iniciando em: $(date)"
echo "[HPC] Nó de execução: $(hostname)"
echo "[HPC] CPU Cores alocadas: ${SLURM_CPUS_PER_TASK:-16}"

# Garante que os diretórios existem
cd "$PROJECT_DIR"
mkdir -p results logs splits_cache

# Executa o script de experimentos
# Parâmetros:
#   --grid full: grade completa de hiperparâmetros
#   --folds 10: validação cruzada oficial de 10 folds
#   --n_jobs: usa todos os núcleos de CPU alocados
#   (O modo --resume está ativo por padrão, garantindo que timeouts no cluster possam ser retomados)
eval python run_experiments.py \
    $DATASET_ARG \
    --models OCSVM iForest \
    --bytes 1 2 3 4 5 6 7 8 \
    --grid full \
    --folds 10 \
    --n_jobs ${SLURM_CPUS_PER_TASK:-16}

EXIT_CODE=$?

echo "[HPC] Finalizado em: $(date) com código de saída: $EXIT_CODE"
exit $EXIT_CODE
