# Usage:
#   .\scripts\windows\build_kg.ps1 [hydra_overrides...]
#
# Arguments:
#   hydra_overrides     Optional Hydra overrides forwarded to src/build_kg.py.
#
# Examples:
#   .\scripts\windows\build_kg.ps1
#   .\scripts\windows\build_kg.ps1 build_kg.k=20
#   .\scripts\windows\build_kg.ps1 build_kg.feature_mode=no_blur
#   .\scripts\windows\build_kg.ps1 build_kg.k=15 build_kg.partition=training

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

uv run --no-sync python src/build_kg.py @HydraOverrides
