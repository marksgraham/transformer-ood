# pixel space transformer
ngc batch run --name "transformer_mednist_hand" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.8.norm \
--commandline "torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
/mount/transformer-ood/train_transformer.py  \
--output_dir=/mount/output/  \
--model_name=transformer_mednist_hand_pixelspace  \
--training_dir=/mount/data/data_splits/Hand_train.csv \
--validation_dir=/mount/data/data_splits/Hand_val.csv \
--n_epochs=100 \
--batch_size=48 \
--eval_freq=10 \
--cache_data=1 \
--checkpoint_every=10 \
--transformer_type=transformer \
--spatial_dimension=2 \
--image_size=32" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/transformer-ood:v0.1.2" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

# vqvae + transformer
ngc batch run --name "transformer_mednist_hand" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.8.norm \
--commandline "torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
/mount/transformer-ood/train_transformer.py  \
--output_dir=/mount/output/  \
--model_name=transformer_mednist_hand_vqvae  \
--training_dir=/mount/data/data_splits/Hand_train.csv \
--validation_dir=/mount/data/data_splits/Hand_val.csv \
--n_epochs=100 \
--batch_size=48 \
--eval_freq=10 \
--cache_data=1 \
--checkpoint_every=10 \
--transformer_type=transformer \
--spatial_dimension=2 \
--image_size=32 \
--vqvae_checkpoint=/mount/output/vqvae_mednist_hand/checkpoint.pth" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/transformer-ood:v0.1.2" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50
