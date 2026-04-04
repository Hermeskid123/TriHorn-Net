folder="nyu_experiment_resnet18"

ex1="python main.py --checkpoints_dir $folder --config_file configs/nyu_resnet18.yaml"
ex2="python eval.py --path $folder/checkpoints"

$ex1
$ex2
