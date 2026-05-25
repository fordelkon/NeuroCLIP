# Train cross-subject with KG for subjects 01-10
# Usage: .\train_cross_kg_loop.ps1 [model]
#
# Arguments:
#   model    Model name (default: flatnet). Options: flatnet, nice, atms
#
# Examples:
#   .\train_cross_kg_loop.ps1
#   .\train_cross_kg_loop.ps1 nice
#   .\train_cross_kg_loop.ps1 atms

param(
    [Parameter(Position=0)]
    [string]$Model = "flatnet"
)

$ErrorActionPreference = "Stop"

for ($i = 1; $i -le 10; $i++) {
    $subject = "sub-{0:D2}" -f $i
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "Training cross-subject KG for $subject with $Model" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan

    uv run --no-sync python .\src\train.py `
        data=thingseeg2 `
        data.subjects=$subject `
        data.experiment_setting=cross-subject `
        model=$Model `
        model.loss_type=cliploss `
        model.enable_kg_smooth=true `
        trainer=gpu `
        trainer.max_epochs=200 `
        callbacks.early_stopping.patience=20 `
        callbacks.early_stopping.min_delta=0 `
        logger=csv `
        hydra.run.dir=logs/train/runs/${Model}_cross/${Model}_kg_$subject `
        extract=thingseeg2 `
        build_kg=thingseeg2

    Write-Host "Completed $subject" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All subjects completed!" -ForegroundColor Green
