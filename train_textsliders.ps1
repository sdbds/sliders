# Sliders train script by @bdsqlsz

#SD version(Lora、Sdxl_lora)
$sd_version = "sdxl_lora"

#train_mode(text,image)
$train_mode = "text"

#output name
$name = "ageslider"

# Train data config | 设置训练配置路径
$config_file = "./trainscripts/textsliders/data/config-xl.yaml" # config path | 配置路径

# main body attributes| 主体属性
$attributes = 'male, female'

#LoRA rank and alpha
$rank = 4
$alpha = 1

#image model
$folder_main = 'datasets/eyesize/'
$folders = "bigsize, smallsize"
$scales = "1, -1"

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($train_mode -ieq "text") {
  $laungh_script = "textsliders/train_lora"
}
else {
  $laungh_script = "imagesliders/train_lora-scale"
  [void]$ext_args.Add("--folder_main=$folder_main")
  [void]$ext_args.Add("--folders=$folders")
  [void]$ext_args.Add("--scales=$scales")
}

if ($sd_version -ilike "sdxl*") {
  $laungh_script = $laungh_script + "_xl"
}

# run train
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 "./trainscripts/$laungh_script.py" `
  --config_file=$config_file `
  --attributes=$attributes `
  --name=$name `
  --rank=$rank `
  --alpha=$alpha $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;