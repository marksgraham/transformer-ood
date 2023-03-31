# transformer
ngc batch run --name "transformer_decathlon_task01" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.8.norm \
--commandline "torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
/mount/transformer-ood/train_transformer.py  \
--output_dir=/mount/output/  \
--model_name=transformer_decathlon_task01  \
--vqvae_checkpoint=/mount/output/vqvae_decathlon_task01_3layer_128_2048_crop160x160x128_decay0.9_lr3e-5/checkpoint.pth \
--training_dir=/mount/data/data_splits/Task01_BrainTumour_train.csv \
--validation_dir=/mount/data/data_splits/Task01_BrainTumour_val.csv \
--image_roi=[160,160,128] \
--image_size=128 \
--n_epochs=700 \
--batch_size=4 \
--eval_freq=10 \
--cache_data=1 \
--transformer_type=transformer" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/transformer-ood:v0.1.3" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50
