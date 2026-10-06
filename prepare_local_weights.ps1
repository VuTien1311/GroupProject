param([Parameter(Mandatory = $true)][string]$SourceRoot)
$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$weightPaths = @(
    'yolo11x-seg.pt',
    'hybrid2\weights\hybridnets.pth',
    'checkpoints\depth_anything_v2_vitl.pth'
)
foreach ($relativePath in $weightPaths) {
    $sourcePath = Join-Path $SourceRoot $relativePath
    $destinationPath = Join-Path $packageRoot $relativePath
    if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        Write-Host "Kept existing: $relativePath"
        continue
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        Write-Warning "Missing at source: $sourcePath. See README for official checkpoints."
        continue
    }
    $parentPath = Split-Path -Parent $destinationPath
    if (-not (Test-Path -LiteralPath $parentPath -PathType Container)) {
        New-Item -ItemType Directory -Path $parentPath | Out-Null
    }
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
    Write-Host "Copied: $relativePath"
}
Write-Host 'No training or download started. Run run_paper.py --check after all three weights are present.'
