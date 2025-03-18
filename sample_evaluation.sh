datapath=/mnt/data/mvtec_anomaly_detection
# 这个是sample_trainning的保存路径
loadpath=/home/yhw/patchcore-inspection/results/MVTecAD_Results
# 注意这里评估的模型是我上面自己训练的模型，如果需要评估官方预训练好的权重，注意修改loadpath和modelfolder
# 反正看results文件夹下的模型名字选一个
modelfolder="IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_2"
savefolder="evaluated_results/$modelfolder"

# 还可以指定单独测试某一个类别：搜索我来

# Define datasets
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')

# Generate flags
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "-d $dataset"; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo "-p ${loadpath}/${modelfolder}/models/mvtec_${dataset}"; done))

# Run the evaluation
# 注意去掉官方提供的代码中的--faiss_on_gpu参数
python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0  --save_segmentation_images "$savefolder" \
patch_core_loader "${model_flags[@]}" \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec "$datapath"

