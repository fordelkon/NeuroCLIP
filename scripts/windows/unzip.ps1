# Expected downloaded zip layout:
#   <eeg_zip_dir>\
#     sub-01.zip
#     sub-02.zip
#     ...
#   <image_zip_dir>\
#     training_images.zip
#     test_images.zip
#
# Extraction target layout is resolved from Hydra paths:
#   paths.thingseeg2_raw_dir\
#     sub-01\
#     sub-02\
#     ...
#   paths.thingseeg2_img_dir\
#     training_images\
#     test_images\
#
# Usage:
#   .\scripts\windows\unzip.ps1 [eeg_zip_dir] [image_zip_dir] [hydra_overrides...]
#
# Examples:
#   .\scripts\windows\unzip.ps1
#   .\scripts\windows\unzip.ps1 D:\zips\eeg D:\zips\img
#   .\scripts\windows\unzip.ps1 D:\zips\eeg D:\zips\img paths.thingseeg2_raw_dir=D:\data\eeg paths.thingseeg2_img_dir=D:\data\img

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

function Test-HydraOverrideArgument {
    param(
        [string]$Value
    )

    return $Value.Contains("=") -or $Value.StartsWith("+") -or $Value.StartsWith("~")
}

$argumentIndex = 0
$EegZipDir = ""
$ImageZipDir = ""

if ($Arguments.Count -gt $argumentIndex -and -not (Test-HydraOverrideArgument -Value $Arguments[$argumentIndex])) {
    $EegZipDir = $Arguments[$argumentIndex]
    $argumentIndex += 1
}

if ($Arguments.Count -gt $argumentIndex -and -not (Test-HydraOverrideArgument -Value $Arguments[$argumentIndex])) {
    $ImageZipDir = $Arguments[$argumentIndex]
    $argumentIndex += 1
}

$HydraOverrides = @()
if ($Arguments.Count -gt $argumentIndex) {
    $HydraOverrides = $Arguments[$argumentIndex..($Arguments.Count - 1)]
}

if (-not $env:PROJECT_ROOT) {
    $env:PROJECT_ROOT = (Get-Location).Path
}

function Resolve-DatasetDirectories {
    param(
        [string[]]$Overrides
    )

    $script = @'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

config_dir = str(Path('configs').resolve())
with initialize_config_dir(version_base='1.3', config_dir=config_dir):
    cfg = compose(config_name='preprocess', overrides=list(sys.argv[1:]))

print(cfg.paths.thingseeg2_raw_dir)
print(cfg.paths.thingseeg2_img_dir)
'@

    $paths = uv run --no-sync python -c $script @Overrides
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve dataset directories from Hydra config."
    }

    return $paths
}

function Expand-DatasetArchives {
    param(
        [string]$SourceDir,
        [string]$TargetDir,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
        Write-Host "Skipping ${Label}: source directory not found: $SourceDir"
        return
    }

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

    $archives = @(Get-ChildItem -LiteralPath $SourceDir -Filter "*.zip" -File)
    if ($archives.Count -eq 0) {
        Write-Host "No ${Label} zip archives found in $SourceDir"
        return
    }

    foreach ($archive in $archives) {
        Write-Host "Extracting $($archive.FullName) -> $TargetDir"
        Expand-Archive -LiteralPath $archive.FullName -DestinationPath $TargetDir -Force
        Remove-Item -LiteralPath $archive.FullName
    }
}

$datasetDirs = Resolve-DatasetDirectories -Overrides $HydraOverrides
if ($datasetDirs.Count -lt 2) {
    throw "Expected Hydra config to resolve EEG and image directories."
}

$EegDir = $datasetDirs[0]
$ImageDir = $datasetDirs[1]

if (-not $EegZipDir) {
    $EegZipDir = $EegDir
}

if (-not $ImageZipDir) {
    $ImageZipDir = $ImageDir
}

Expand-DatasetArchives -SourceDir $EegZipDir -TargetDir $EegDir -Label "EEG"
Expand-DatasetArchives -SourceDir $ImageZipDir -TargetDir $ImageDir -Label "image"
