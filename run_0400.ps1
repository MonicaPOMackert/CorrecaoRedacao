# Dispara o MTS-v2 (prompts dedicados C1/C5) no Flash Lite, 5 fatias de 60 em paralelo,
# junta, roda evaluate + calibrate. Feito para o Agendador de Tarefas rodar de madrugada,
# depois que o RPD do Gemini renova (~04:00 BRT).
$ErrorActionPreference = "Continue"
$proj = $PSScriptRoot  # pasta onde este .ps1 esta, evita acento quebrado no caminho
Set-Location $proj
$stamp = Get-Date -Format "yyyyMMdd_HHmm"

$procs = 0..4 | ForEach-Object {
    $i = $_ + 1
    $off = $_ * 60
    $args = "run_api_scoring.py --provider gemini --model gemini-3.5-flash-lite --modo mts2 " +
            "--key-env GEMINI_API_KEY_$i --offset $off --limit 60 --rpm 15 " +
            "--out results/api/flmts2_$i.csv"
    Start-Process python -PassThru -WindowStyle Hidden -WorkingDirectory $proj `
        -ArgumentList $args -RedirectStandardError "results/api/log_flmts2_${i}_$stamp.txt"
}
$procs | Wait-Process

python -c "import pandas as pd,glob; d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/api/flmts2_*.csv'))]).drop_duplicates('index_redacao').sort_values('index_redacao'); d.to_csv('results/api/flashlite_mts2.csv',index=False); print(len(d),'redacoes,',int(d['ok'].sum()),'com nota')" *> "results/api/log_mts2_$stamp.txt"
python evaluate.py results/api/flashlite_mts2.csv --confusao *>> "results/api/log_mts2_$stamp.txt"
python calibrate.py results/api/flashlite_mts2.csv --prompts-file results/v5_experimentos_v2/index_prompt_map.csv --out results/calibracao_flashlite_mts2.csv *>> "results/api/log_mts2_$stamp.txt"
