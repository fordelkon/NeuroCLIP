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
