param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

uv run --no-sync python src/extract.py @HydraOverrides
