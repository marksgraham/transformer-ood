# transformer
ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.8.norm \
--commandline "torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
/mount/transformer-ood/train_transformer.py  \
--output_dir=/mount/output/  \
--model_name=transformer_cromis_4layer_64_2048  \
--vqvae_checkpoint=/mount/output/vqvae_cromis_4layer_64_2048/checkpoint.pth \
--training_dir=/mount/data/cromis/train \
--validation_dir=/mount/data/cromis/val \
--n_epochs=500 \
--batch_size=16 \
--eval_freq=10 \
--cache_data=1" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.1" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50