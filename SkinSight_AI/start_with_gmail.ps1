param([switch]$Reset)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Export-Clixml protects PSCredential passwords using Windows DPAPI.
# Only this Windows user on this computer can decrypt the saved password.
$gmailConfigPath = Join-Path $PSScriptRoot 'data\gmail-credentials.clixml'
if ($Reset -and (Test-Path $gmailConfigPath)) { Remove-Item $gmailConfigPath }
if (Test-Path $gmailConfigPath) {
    try {
        $gmailCredential = Import-Clixml $gmailConfigPath
        if ($gmailCredential -isnot [System.Management.Automation.PSCredential]) { throw 'Invalid credential file' }
    } catch {
        throw 'Cannot read saved Gmail settings. Delete data\gmail-credentials.clixml and run again to configure this Windows account.'
    }
} else {
    $gmailSender = (Read-Host 'One-time setup: sending Gmail address').Trim()
    if ($gmailSender -notmatch '^[^\s@]+@gmail\.com$') { throw 'Enter the sending Gmail address.' }
    $gmailSecret = Read-Host 'Google App Password (not your normal Gmail password)' -AsSecureString
    $gmailCredential = New-Object System.Management.Automation.PSCredential($gmailSender, $gmailSecret)
    New-Item -ItemType Directory -Path (Split-Path $gmailConfigPath) -Force | Out-Null
    $gmailCredential | Export-Clixml -Path $gmailConfigPath
    Write-Host 'Gmail settings saved encrypted for this Windows account. Future starts will not prompt.'
}
$gmailPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($gmailCredential.Password)
try {
    $env:SKINSIGHT_SMTP_HOST = 'smtp.gmail.com'
    $env:SKINSIGHT_SMTP_PORT = '465'
    $env:SKINSIGHT_SMTP_USER = $gmailCredential.UserName
    $env:SKINSIGHT_MAIL_FROM = $gmailCredential.UserName
    $env:SKINSIGHT_SMTP_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($gmailPointer).Replace(' ', '')
    Write-Host 'Open http://127.0.0.1:8000. Reports are emailed directly to the registered address after analysis.'
    & '.\.venv\Scripts\python.exe' run.py
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($gmailPointer)
    Remove-Item Env:SKINSIGHT_SMTP_PASSWORD -ErrorAction SilentlyContinue
    $gmailCredential.Password.Dispose()
}
