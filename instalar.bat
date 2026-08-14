@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; Get-Content -LiteralPath '%~f0' | Select-Object -Skip 3 | Out-String | Invoke-Expression"
exit /b %errorlevel%
# Duplo clique neste arquivo instala tudo que o projeto Pintura precisa.
# (as 3 linhas batch acima so mandam o PowerShell ler e rodar o resto deste
# mesmo arquivo - dai pra baixo e tudo PowerShell, um arquivo so.)

$ErrorActionPreference = "Stop"

function Escrever($texto, $cor = "White") {
    Write-Host $texto -ForegroundColor $cor
}

try {
    Escrever "==============================================="
    Escrever " Instalando dependencias do projeto Pintura"
    Escrever "==============================================="
    Escrever ""

    # 1) Achar um Python instalado (o launcher "py" e o mais confiavel no Windows)
    $python = $null
    foreach ($candidato in @("py", "python")) {
        if (Get-Command $candidato -ErrorAction SilentlyContinue) {
            $python = $candidato
            break
        }
    }
    if (-not $python) {
        Escrever "ERRO: nao encontrei o Python instalado nesta maquina." "Red"
        Escrever "Baixe e instale em https://www.python.org/downloads/ (marque 'Add python.exe to PATH')" "Red"
        Escrever "e rode este arquivo de novo." "Red"
        Read-Host "Pressione ENTER para fechar"
        exit 1
    }
    Escrever "Python encontrado: $python"

    # 2) Criar o ambiente virtual (.venv), se ainda nao existir
    $pythonVenv = ".venv\Scripts\python.exe"
    if (-not (Test-Path $pythonVenv)) {
        Escrever "Criando ambiente virtual em .venv ..."
        & $python -m venv .venv
        if (-not (Test-Path $pythonVenv)) {
            Escrever "ERRO: nao consegui criar o ambiente virtual .venv." "Red"
            Read-Host "Pressione ENTER para fechar"
            exit 1
        }
    }
    Escrever "Ambiente virtual OK: $pythonVenv"
    Escrever ""

    # 3) Detectar se esta rede exige proxy (Caterpillar) ou nao (casa/outra rede)
    Escrever "Verificando se esta rede precisa de proxy..."
    $destino = New-Object System.Uri("https://pypi.org")
    $proxySistema = [System.Net.WebRequest]::GetSystemWebProxy()
    $proxyUrl = $proxySistema.GetProxy($destino)

    if ($proxyUrl -and $proxyUrl.AbsoluteUri -ne $destino.AbsoluteUri) {
        Escrever "Rede corporativa detectada. Usando proxy: $($proxyUrl.AbsoluteUri)"
        $env:HTTPS_PROXY = $proxyUrl.AbsoluteUri
        $env:HTTP_PROXY = $proxyUrl.AbsoluteUri
    } else {
        Escrever "Nenhum proxy necessario (conexao direta)."
    }
    Escrever ""

    # 4) Instalar as bibliotecas do requirements.txt
    Escrever "Instalando bibliotecas (pode demorar alguns minutos, aguarde)..."
    & $pythonVenv -m pip install --upgrade pip
    & $pythonVenv -m pip install -r requirements.txt

    if ($LASTEXITCODE -ne 0) {
        Escrever ""
        Escrever "ERRO: a instalacao das bibliotecas falhou. Copie a mensagem acima e" "Red"
        Escrever "mande pro responsavel pelo projeto." "Red"
        Read-Host "Pressione ENTER para fechar"
        exit 1
    }

    Escrever ""
    Escrever "==============================================="
    Escrever " Tudo instalado! Pode fechar esta janela." "Green"
    Escrever " Para rodar o sistema de fotos, use:" "Green"
    Escrever " .venv\Scripts\python.exe fotografar_pecas.py" "Green"
    Escrever "==============================================="
}
catch {
    Escrever ""
    Escrever "ERRO INESPERADO: $($_.Exception.Message)" "Red"
    Escrever "Copie esta mensagem e mande pro responsavel pelo projeto." "Red"
    Read-Host "Pressione ENTER para fechar"
    exit 1
}

Read-Host "Pressione ENTER para fechar"
