# Usage:
#   .\scripts\windows\extract.ps1 [hydra_overrides...]
#
# Arguments:
#   hydra_overrides     Optional Hydra overrides forwarded to src/extract.py.
#
# Examples:
#   .\scripts\windows\extract.ps1
#   .\scripts\windows\extract.ps1 extract.reference_subject_id=1
#   .\scripts\windows\extract.ps1 extract.model_name=ViT-B-32 extract.feature_mode=pooled
#   .\scripts\windows\extract.ps1 paths.thingseeg2_preprocessed_dir=D:\data\prep paths.thingseeg2_clip_features_dir=D:\data\clip

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

uv run --no-sync python src/extract.py @HydraOverrides
