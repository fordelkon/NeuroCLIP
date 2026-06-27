# Usage:
#   .\scripts\windows\train_kg.ps1 <subject_id> <model> <k_fold> <fold_idx> [hydra_overrides...]
#
# Description:
#   Train model WITH knowledge graph smoothing.
#
# Arguments:
#   subject_id          Subject ID (e.g., sub-01, sub-02, ..., sub-10)
#   model               Model name (e.g., flatnet, nice, atms)
#   k_fold              Number of folds
#   fold_idx            Fold index for k-fold cross-validation (e.g., 0, 1, 2, 3, 4)
#   hydra_overrides     Optional Hydra overrides forwarded to src/train.py.
#
# Examples:
#   .\scripts\windows\train_kg.ps1 sub-01 flatnet 5 0
#   .\scripts\windows\train_kg.ps1 sub-02 nice 10 1
#   .\scripts\windows\train_kg.ps1 sub-03 atms 5 0 model.kgnet.alpha=0.5

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$SubjectId,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$Model,

    [Parameter(Mandatory=$true, Position=2)]
    [int]$KFold,

    [Parameter(Mandatory=$true, Position=3)]
    [int]$FoldIdx,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

$RunDir = "logs/train/runs/${Model}/${Model}_kg_k${KFold}_fold${FoldIdx}_${SubjectId}"

uv run --no-sync python src/train.py `
  data=thingseeg2 `
  "data.subjects=[$SubjectId]" `
  data.k_fold=$KFold `
  data.fold_idx=$FoldIdx `
  model=$Model `
  model.loss_type=cliploss `
  model.enable_kg_smooth=true `
  trainer=cpu `
  trainer.max_epochs=200 `
  callbacks.early_stopping.patience=20 `
  callbacks.early_stopping.min_delta=0 `
  logger=csv `
  hydra.run.dir=$RunDir `
  extract=thingseeg2 `
  build_kg=thingseeg2 `
  @HydraOverrides
