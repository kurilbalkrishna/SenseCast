<#
Imports docs/01-brief/backlog.csv into GitHub:
  labels -> milestones -> issues (assigned to owners) -> Project board with Status set.
Rows with status "done" are closed as completed and shown in the Done column;
rows with status "todo" stay open in the Todo column.
Safe to re-run: existing labels, milestones, issues and the project are reused, not duplicated.

Needs the GitHub CLI, logged in with the project scope:
  winget install --id GitHub.cli
  gh auth login
  gh auth refresh -s project

Run from the repo root:
  powershell -ExecutionPolicy Bypass -File scripts\create_backlog.ps1
#>
param(
    [string]$Owner = "kurilbalkrishna",
    [string]$RepoName = "sensecast",
    [string]$ProjectTitle = "SenseCast board",
    [string]$CsvPath = "docs/01-brief/backlog.csv"
)

# "Continue" on purpose: in Windows PowerShell 5.1, "Stop" turns any gh message on stderr into a
# crash. Failures are caught by checking gh's exit code in Invoke-Gh instead.
$ErrorActionPreference = "Continue"
$Repo = "$Owner/$RepoName"
$People = @{
    "member-a" = @("kurilbalkrishna")
    "member-b" = @("chaitanya3132-jpg")
    "both"     = @("kurilbalkrishna", "chaitanya3132-jpg")
}
$LabelColors = @{
    data = "1d76db"; features = "0e8a16"; model = "5319e7"; decision = "fbca04"
    api = "0052cc"; security = "b60205"; ui = "c5def5"; test = "bfdadc"
    ops = "006b75"; docs = "d4c5f9"; must = "e11d21"; should = "f9d0c4"
}

function Invoke-Gh {
    # Runs gh, stops on failure, returns stdout only (stderr kept apart so JSON stays clean).
    $errFile = [IO.Path]::GetTempFileName()
    $out = & gh @args 2> $errFile
    $code = $LASTEXITCODE
    $err = Get-Content $errFile -Raw
    Remove-Item $errFile -ErrorAction SilentlyContinue
    if ($code -ne 0) { throw "gh $($args -join ' ') failed:`n$err" }
    return ($out -join "`n")
}

function Invoke-GhRetry {
    # GitHub sometimes rejects a write made a split second after the issue was created. Wait and retry.
    for ($try = 1; $try -le 4; $try++) {
        try { return (Invoke-Gh @args) }
        catch {
            if ($try -eq 4) { throw }
            Write-Host "    GitHub busy, retrying in $($try * 3)s..." -ForegroundColor Yellow
            Start-Sleep -Seconds ($try * 3)
        }
    }
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI not found. Run: winget install --id GitHub.cli  (then open a new PowerShell)"
}
if (-not (Test-Path $CsvPath)) { throw "Run this from the repo root; $CsvPath not found." }
$rows = Import-Csv $CsvPath
Write-Host "Backlog: $($rows.Count) rows ($(@($rows | Where-Object status -eq 'done').Count) done)" -ForegroundColor Cyan

# 1. Labels
Write-Host "`n[1/5] Labels" -ForegroundColor Cyan
$labels = $rows | ForEach-Object { $_.labels -split "," } | ForEach-Object { $_.Trim() } | Sort-Object -Unique
foreach ($l in $labels) {
    $color = if ($LabelColors.ContainsKey($l)) { $LabelColors[$l] } else { "ededed" }
    Invoke-Gh label create $l --repo $Repo --color $color --force | Out-Null
    Write-Host "  label $l"
}

# 2. Milestones
Write-Host "`n[2/5] Milestones" -ForegroundColor Cyan
$existing = Invoke-Gh api "repos/$Repo/milestones?state=all&per_page=100" | ConvertFrom-Json
$msNumbers = @{}
foreach ($m in $existing) { $msNumbers[$m.title] = $m.number }
foreach ($m in ($rows.milestone | Sort-Object -Unique)) {
    if (-not $msNumbers.ContainsKey($m)) {
        $created = Invoke-Gh api "repos/$Repo/milestones" -f "title=$m" | ConvertFrom-Json
        $msNumbers[$m] = $created.number
        Write-Host "  created $m"
    } else { Write-Host "  exists  $m" }
}

# 3. Project board
Write-Host "`n[3/5] Project board" -ForegroundColor Cyan
$projects = (Invoke-Gh project list --owner $Owner --format json | ConvertFrom-Json).projects
$project = $projects | Where-Object { $_.title -eq $ProjectTitle } | Select-Object -First 1
if (-not $project) {
    $project = Invoke-Gh project create --owner $Owner --title $ProjectTitle --format json | ConvertFrom-Json
    Write-Host "  created '$ProjectTitle' (#$($project.number))"
} else { Write-Host "  exists  '$ProjectTitle' (#$($project.number))" }
try { Invoke-Gh project link $project.number --owner $Owner --repo $Repo | Out-Null } catch { }  # already linked is fine
$fields = (Invoke-Gh project field-list $project.number --owner $Owner --format json | ConvertFrom-Json).fields
$status = $fields | Where-Object { $_.name -eq "Status" } | Select-Object -First 1
$optTodo = ($status.options | Where-Object { $_.name -eq "Todo" }).id
$optDone = ($status.options | Where-Object { $_.name -eq "Done" }).id

# 4. Issues
Write-Host "`n[4/5] Issues" -ForegroundColor Cyan
$have = @{}
foreach ($i in (Invoke-Gh issue list --repo $Repo --state all --limit 500 --json "title,url,state" | ConvertFrom-Json)) { $have[$i.title] = $i }
$tmp = [IO.Path]::GetTempFileName()
$n = 0
foreach ($r in $rows) {
    $n++
    $who = $People[$r.owner]
    if ($have.ContainsKey($r.title)) {
        # Already created on an earlier run: just make sure board status and open/closed are right.
        $url = $have[$r.title].url
        $isOpen = $have[$r.title].state -eq "OPEN"
        $verb = "fixed "
        if (-not $isOpen -and $r.status -ne "done") {
            # A todo that was closed on GitHub (e.g. by a merged PR) is finished: GitHub wins, leave it alone.
            Write-Host ("  {0,2}. kept   closed  {1}" -f $n, $r.title)
            continue
        }
    } else {
        $body = "$($r.body)`n`n**Owner:** $(($who | ForEach-Object { "@$_" }) -join ', ')`n`n_Imported from docs/01-brief/backlog.csv_"
        [IO.File]::WriteAllText($tmp, $body)  # UTF-8 without BOM
        $labelArgs = @()
        foreach ($label in ($r.labels -split ",")) { $labelArgs += @("--label", $label.Trim()) }
        $assigneeArgs = @()
        foreach ($person in $who) { $assigneeArgs += @("--assignee", $person) }
        $url = (Invoke-GhRetry issue create --repo $Repo --title $r.title --body-file $tmp `
                --milestone $r.milestone @labelArgs @assigneeArgs).Trim().Split("`n")[-1]
        $isOpen = $true
        $verb = "new   "
        Start-Sleep -Seconds 2  # give GitHub a moment before editing the new issue
    }

    $item = Invoke-GhRetry project item-add $project.number --owner $Owner --url $url --format json | ConvertFrom-Json
    $opt = if ($r.status -eq "done") { $optDone } else { $optTodo }
    if ($status -and $opt) {
        Invoke-GhRetry project item-edit --id $item.id --project-id $project.id `
            --field-id $status.id --single-select-option-id $opt | Out-Null
    }
    if ($r.status -eq "done" -and $isOpen) {
        Invoke-GhRetry issue close $url --repo $Repo --reason completed | Out-Null
    } elseif ($r.status -eq "todo" -and -not $isOpen) {
        Invoke-GhRetry issue reopen $url --repo $Repo | Out-Null
    }
    Write-Host ("  {0,2}. {1} {2,-4}  {3}" -f $n, $verb, $r.status, $r.title)
}
Remove-Item $tmp -ErrorAction SilentlyContinue

# 5. Close milestones whose issues are all done
Write-Host "`n[5/5] Milestones status" -ForegroundColor Cyan
foreach ($m in ($rows.milestone | Sort-Object -Unique)) {
    $open = @($rows | Where-Object { $_.milestone -eq $m -and $_.status -ne "done" }).Count
    if ($open -eq 0) {
        Invoke-Gh api -X PATCH "repos/$Repo/milestones/$($msNumbers[$m])" -f "state=closed" | Out-Null
        Write-Host "  closed $m"
    } else {
        # Reopen in case a new todo row was added to a milestone closed on an earlier run.
        Invoke-Gh api -X PATCH "repos/$Repo/milestones/$($msNumbers[$m])" -f "state=open" | Out-Null
        Write-Host "  open   $m ($open to do)"
    }
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Issues:  https://github.com/$Repo/issues"
Write-Host "Board:   https://github.com/users/$Owner/projects/$($project.number)"
