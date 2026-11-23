$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 n'est pas installé ou le lanceur 'py' est absent. Installe Python 3.12 puis relance ce script."
    }
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 est requis. Installe-le puis relance ce script." }
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}

$envPath = Join-Path $PSScriptRoot ".env"
$lines = [System.Collections.Generic.List[string]]::new()
foreach ($line in [System.IO.File]::ReadAllLines($envPath)) { $lines.Add($line) }

function Get-EnvValue([string]$name) {
    foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($name))\s*=(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

function Set-EnvValue([string]$name, [string]$value) {
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($name))\s*=") {
            $lines[$i] = "$name=$value"
            return
        }
    }
    $lines.Add("$name=$value")
}

if ([string]::IsNullOrWhiteSpace((Get-EnvValue "APP_ACCESS_TOKEN"))) {
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $bytes = New-Object byte[] 32
    $rng.GetBytes($bytes)
    $rng.Dispose()
    $localToken = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    Set-EnvValue "APP_ACCESS_TOKEN" $localToken
}

if ([string]::IsNullOrWhiteSpace((Get-EnvValue "GEMINI_API_KEY"))) {
    $secureKey = Read-Host "Colle ta clé Google AI Studio (saisie masquée)" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $geminiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if ([string]::IsNullOrWhiteSpace($geminiKey) -or $geminiKey.Contains("`r") -or $geminiKey.Contains("`n")) {
        throw "Clé Gemini vide ou invalide. Relance le script et saisis la clé dans l'invite masquée."
    }
    Set-EnvValue "GEMINI_API_KEY" $geminiKey
}

[System.IO.File]::WriteAllLines($envPath, $lines, [System.Text.UTF8Encoding]::new($false))
foreach ($line in $lines) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $name = $Matches[1]
        $value = $Matches[2].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Write-Host "Installation des dépendances Python..."
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "L'installation Python a échoué." }

Write-Host "API locale : http://127.0.0.1:8000"
Write-Host "Documentation/test interactif : http://127.0.0.1:8000/docs"
Write-Host "Jeton local enregistré dans .env (fichier ignoré par Git). Ne le publie pas."
& $python -m uvicorn app:app --host 127.0.0.1 --port 8000
