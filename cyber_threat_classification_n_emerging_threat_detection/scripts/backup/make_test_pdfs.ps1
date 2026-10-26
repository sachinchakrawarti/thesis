# make_test_pdfs.ps1 — generates 10 valid CTI-looking PDFs in sample_documents/.
# No external tools required. Run from anywhere.

$ErrorActionPreference = "Stop"

# Resolve output folder relative to this script's location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = Join-Path $scriptDir "..\sample_documents"
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}
$outDir = (Resolve-Path -LiteralPath $outDir).Path
Write-Host "Output: $outDir" -ForegroundColor Cyan

# ---------- CTI content ----------
$actors  = @("APT28","APT29","Lazarus Group","FIN7","Sandworm","Wizard Spider","LockBit","BlackCat","Cl0p","Hive")
$malware = @("Emotet","TrickBot","QakBot","IcedID","Cobalt Strike","Mimikatz","PlugX","Remcos","AgentTesla","Formbook")
$sectors = @("healthcare","finance","energy","government","manufacturing","education","retail","telecom","defense","logistics")
$cves    = @("CVE-2023-23397","CVE-2023-4966","CVE-2024-3400","CVE-2024-21762","CVE-2023-38831","CVE-2021-44228","CVE-2023-34362","CVE-2022-30190")
$protos  = @("HTTPS","DNS","TLS","SMTP","SMB","RDP","SSH")

function Pick($arr) { return $arr[(Get-Random -Minimum 0 -Maximum $arr.Count)] }

function New-Line {
    $templates = @(
        "The {0} threat group was observed deploying {1} against {2} organizations in a coordinated campaign.",
        "Phishing emails with malicious attachments delivered initial access for {0}.",
        "The campaign exploited {3} to compromise internet-facing servers and drop {1}.",
        "{1} established persistence by modifying registry run keys on victim hosts.",
        "Operators used {1} to dump LSASS memory and harvest credentials from endpoints.",
        "Command-and-control traffic from {1} was observed over {4} to attacker-controlled infrastructure.",
        "Data exfiltration by {1} occurred over an encrypted channel to cloud storage.",
        "{0} ransomware encrypted production systems at a {2} organization.",
        "Lateral movement was achieved via {4} using compromised administrative credentials.",
        "{1} performed discovery by enumerating local users and network shares.",
        "{1} disabled endpoint detection to evade analysis and remain resident.",
        "Indicators of compromise include IP 192.0.2.10 and domain malicious-example.test."
    )
    $t = $templates[(Get-Random -Minimum 0 -Maximum $templates.Count)]
    return ($t -f (Pick $actors), (Pick $malware), (Pick $sectors), (Pick $cves), (Pick $protos))
}

function New-Body {
    param([int]$N = 12)
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("Cyber Threat Intelligence Report")
    $lines.Add("Generated: " + (Get-Date -Format "yyyy-MM-dd HH:mm"))
    $lines.Add("")
    for ($i = 0; $i -lt $N; $i++) {
        $lines.Add((New-Line))
        $lines.Add("")
    }
    return ,$lines
}

function Escape-Pdf([string]$s) {
    $s = $s -replace '\\','\\'
    $s = $s -replace '\(','\('
    $s = $s -replace '\)','\)'
    return $s
}

function Write-Pdf {
    param([string]$Path, [System.Collections.Generic.List[string]]$Lines)

    # Build content stream
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("BT`n")
    [void]$sb.Append("/F1 11 Tf`n")
    [void]$sb.Append("50 780 Td`n")
    [void]$sb.Append("14 TL`n")
    foreach ($line in $Lines) {
        $esc = Escape-Pdf $line
        [void]$sb.Append("(" + $esc + ") Tj T*`n")
    }
    [void]$sb.Append("ET`n")
    $content = $sb.ToString()
    $contentBytes = [System.Text.Encoding]::ASCII.GetByteCount($content)

    $objs = New-Object System.Collections.Generic.List[string]
    $objs.Add("<< /Type /Catalog /Pages 2 0 R >>")
    $objs.Add("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    $objs.Add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    $objs.Add("<< /Length $contentBytes >>`nstream`n" + $content + "endstream")
    $objs.Add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Assemble
    $pdf = New-Object System.Text.StringBuilder
    [void]$pdf.Append("%PDF-1.4`n")
    [void]$pdf.Append([char]0x25 + [char]0xE2 + [char]0xE3 + [char]0xCF + [char]0xD3 + "`n")

    $offsets = @()
    for ($i = 0; $i -lt $objs.Count; $i++) {
        $offsets += [System.Text.Encoding]::ASCII.GetByteCount($pdf.ToString())
        [void]$pdf.Append(($i + 1).ToString() + " 0 obj`n")
        [void]$pdf.Append($objs[$i])
        [void]$pdf.Append("`nendobj`n")
    }

    $xrefAt = [System.Text.Encoding]::ASCII.GetByteCount($pdf.ToString())
    [void]$pdf.Append("xref`n")
    [void]$pdf.Append("0 " + ($objs.Count + 1) + "`n")
    [void]$pdf.Append("0000000000 65535 f `n")
    foreach ($off in $offsets) {
        [void]$pdf.Append(("{0:D10} 00000 n `n" -f $off))
    }
    [void]$pdf.Append("trailer`n")
    [void]$pdf.Append("<< /Size " + ($objs.Count + 1) + " /Root 1 0 R >>`n")
    [void]$pdf.Append("startxref`n")
    [void]$pdf.Append($xrefAt.ToString() + "`n")
    [void]$pdf.Append("%%EOF`n")

    # Write as ASCII (all bytes we emit are ASCII)
    [System.IO.File]::WriteAllText($Path, $pdf.ToString(), [System.Text.Encoding]::ASCII)
}

# ---------- Generate 10 ----------
for ($i = 1; $i -le 10; $i++) {
    $name = "synthetic_cti_report_{0:D2}.pdf" -f $i
    $path = Join-Path $outDir $name
    $body = New-Body -N (10 + (Get-Random -Minimum 0 -Maximum 8))
    Write-Pdf -Path $path -Lines $body
    $kb = [math]::Round((Get-Item -LiteralPath $path).Length / 1024, 1)
    Write-Host ("  wrote {0}  ({1} KB)" -f $name, $kb) -ForegroundColor Green
}

Write-Host "`nDone. 10 PDFs written to: $outDir" -ForegroundColor Yellow