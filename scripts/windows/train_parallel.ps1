# Usage:
#   .\scripts\windows\train_parallel.ps1 <subject_id> <model> <num_folds> [hydra_overrides...]
#
# Description:
#   Run parallel training jobs for all folds (0 to num_folds-1) in background.
#
# Arguments:
#   subject_id          Subject ID (e.g., sub-01, sub-02, ..., sub-10)
#   model               Model name (e.g., flatnet, nice, atms)
#   num_folds           Number of folds for k-fold cross-validation
#   hydra_overrides     Optional Hydra overrides forwarded to each training job
#
# Examples:
#   .\scripts\windows\train_parallel.ps1 sub-10 flatnet 5
#   .\scripts\windows\train_parallel.ps1 sub-01 nice 10 trainer=gpu

param(
    [Parameter(Mandatory=$true)]
    [string]$Subject,

    [Parameter(Mandatory=$true)]
    [string]$Model,

    [Parameter(Mandatory=$true)]
    [int]$NumFolds,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraOverrides
)

$scriptPath = Join-Path $PSScriptRoot "train.ps1"

Write-Host "Starting $NumFolds background training jobs for $Subject with $Model..."

$processes = @()
for ($fold = 0; $fold -lt $NumFolds; $fold++) {
    $args = @($scriptPath, $Subject, $Model, $NumFolds, $fold) + $HydraOverrides
    $process = Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "& $($args -join ' ')" -WindowStyle Hidden -PassThru
    $processes += $process
    Write-Host "Started fold $fold in background (PID: $($process.Id))"
}

Write-Host "`nAll jobs started. Waiting for completion..."
Write-Host "Check logs at: logs/train/runs/$Model/"

$processes | Wait-Process

Write-Host "`nAll jobs finished."
