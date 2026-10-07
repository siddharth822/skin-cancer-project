$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$gmailSender = Read-Host 'Sending Gmail address (the app sends from this account)'
$gmailSecret = Read-Host 'Google App Password (not your normal Gmail password)' -AsSecureString
$gmailPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($gmailSecret)
try {
    $env:SKINSIGHT_SMTP_HOST = 'smtp.gmail.com'
    $env:SKINSIGHT_SMTP_PORT = '465'
    $env:SKINSIGHT_SMTP_USER = $gmailSender
    $env:SKINSIGHT_MAIL_FROM = $gmailSender
    $env:SKINSIGHT_SMTP_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($gmailPointer).Replace(' ', '')
    Write-Host 'Open http://127.0.0.1:8000. Register, log in, then verify your email in Email settings.'
    & '.\.venv\Scripts\python.exe' run.py
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($gmailPointer)
    Remove-Item Env:SKINSIGHT_SMTP_PASSWORD -ErrorAction SilentlyContinue
    $gmailSecret.Dispose()
}
