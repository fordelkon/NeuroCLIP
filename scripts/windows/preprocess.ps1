# Usage:
#   .\scripts\windows\preprocess.ps1 [start_subject] [end_subject] [hydra_overrides...]
#
# Arguments:
#   start_subject       First THINGS-EEG2 subject id to preprocess. Defaults to 1.
#   end_subject         Last THINGS-EEG2 subject id to preprocess. Defaults to 10.
#   hydra_overrides     Optional Hydra overrides forwarded to src/preprocess.py.
#
# Examples:
#   .\scripts\windows\preprocess.ps1
#   .\scripts\windows\preprocess.ps1 1 10
#   .\scripts\windows\preprocess.ps1 1 10 preprocess.sfreq=250
#   .\scripts\windows\preprocess.ps1 1 10 paths.thingseeg2_raw_dir=D:\data\eeg paths.thingseeg2_preprocessed_dir=D:\data\prep

param(
    [int]$StartSubject = 1,
    [int]$EndSubject = 10,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

foreach ($subjectId in $StartSubject..$EndSubject) {
    uv run --no-sync python src/preprocess.py `
        "preprocess.subject_id=$subjectId" `
        @HydraOverrides
}
