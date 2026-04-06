folder_results="msra_experiment_resnet18"
main_predFileName="main_pred_resnet18.txt"

mkdir -p $folder_results

for f in $(seq 0 8)
do
   python main.py --checkpoints_dir MSRA_resnet18_P${f} --config_file configs/msra_resnet18.yaml --leaveout_subject $f
   python eval.py --path MSRA_resnet18_P${f}/checkpoints --save_preds best --pred_file_name preds_${f}.txt
   mv log.txt MSRA_resnet18_P${f}
   mv results.txt MSRA_resnet18_P${f}

   mv MSRA_resnet18_P${f} $folder_results/MSRA_resnet18_P${f}
   echo "Experiment Subject $f Finished at : $(date)">>progress_resnet18.txt

   cat $folder_results/MSRA_resnet18_P${f}/preds_${f}.txt>>$main_predFileName

done

echo >>progress_resnet18.txt
echo "-------------------------- Aggregate Result --------------" >>progress_resnet18.txt
python utils/compute3Derror_MSRA.py $main_predFileName >>progress_resnet18.txt
mv progress_resnet18.txt $folder_results
mv $main_predFileName $folder_results
