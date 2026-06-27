# Usage:
#   .\train_inter_loop.ps1 [model] [num_folds]
#
# Description:
#   Run intra-subject k-fold training for subjects sub-01 through sub-10.
#   Each subject is dispatched through scripts\windows\train_parallel.ps1,
#   which starts one background job per fold and waits for completion.
#
# Arguments:
#   model       Model name (default: flatnet). Examples: flatnet, nice, atms
#   num_folds   Number of folds for k-fold cross-validation (default: 5)
#
# Examples:
#   .\train_inter_loop.ps1
#   .\train_inter_loop.ps1 nice 5
#   .\train_inter_loop.ps1 atms 10

param(
    [Parameter(Position=0)]
    [string]$Model = "flatnet",

    [Parameter(Position=1)]
    [int]$NumFolds = 5
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "scripts\windows\train_parallel.ps1"

for ($i = 1; $i -le 10; $i++) {
    $subject = "sub-{0:D2}" -f $i

    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "Training intra-subject for $subject with $Model ($NumFolds folds)" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan

    & $scriptPath $subject $Model $NumFolds trainer=gpu

    Write-Host "Completed $subject" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All subjects completed!" -ForegroundColor Green
