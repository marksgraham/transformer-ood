# Out-of-distribution detection with Transformers

This repo shows how a VQ-GAN + Transformer can be trained to perform unsupervised OOD detection on 3D medical data. It uses the freely available [Medical Decathlon dataset](http://medicaldecathlon.com/) for its experiments.

The method is fully  described in the MIDL 2022 paper [Transformer-based out-of-distribution detection for clinically safe segmentation](https://proceedings.mlr.press/v172/graham22a).

### Set-up
Create a fresh environment (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image
```bash
cd docker/
bash create_docker_image.sh
```

### Download and prepare the data
Download all classes of the Medical Decathlon dataset. NB: this will take a while.

```bash
export DATA_ROOT=/desired/path/to/data
python src/data/get_decathlon_datasets.py --data_root=${DATA_ROOT}
```


### Train the VQ-GAN
We will treat the BRATs data as the in-distribution class and trained a VQ-GAN on it.

First set up your output dir:
```bash
export OUTPUT_DIR=/desired/path/to/output
```

Then train the VQ-GAN:
```bash
python train_vqvae.py \
--output_dir=${OUTPUT_DIR} \
--model_name=vqgan_decathlon \
--training_dir=${DATA_ROOT}/data_splits/Task01_BrainTumour_train.csv \
--validation_dir=${DATA_ROOT}/data_splits/Task01_BrainTumour_val.csv \
--n_epochs=300 \
--batch_size=8 \
--eval_freq=10 \
--cache_data=1 \
--vqvae_downsample_parameters=[[2,4,1,1],[2,4,1,1],[2,4,1,1],[2,4,1,1]] \
--vqvae_upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0]] \
--vqvae_num_channels=[256,256,256,256] \
--vqvae_num_res_channels=[256,256,256,256] \
--vqvae_embedding_dim=128 \
--vqvae_num_embeddings=2048 \
--spatial_dimension=3 \
--image_roi=[160,160,128] \
--image_size=128 \
--num_workers=4
```

Code is DDP compatible. To train with N GPus, train with:

`torchrun --nproc_per_node=N --nnodes=1 --node_rank=0 train_vqvae.py --args`

### Train the transformer
```bash
python train_transformer.py \
--output_dir=${OUTPUT_DIR}
--model_name=transformer_decathlon
--training_dir=${DATA_ROOT}/data_splits/Task01_BrainTumour_train.csv \
--validation_dir=${DATA_ROOT}/data_splits/Task01_BrainTumour_val.csv \
--n_epochs=100 \
--batch_size=4 \
--eval_freq=10 \
--cache_data=1 \
--image_roi=[160,160,128] \
--image_size=128 \
--num_workers=4
--vqvae_checkpoint=${OUTPUT_DIR}/vqgan_decathlon/checkpoint.pth
--transformer_type=transformer
```
### Evaluate
Get likelihoods on the test set of the BRATs dataset, and on the test sets of the other 9 classes of the Medical Decathlon dataset.
```bash
python perform_ood.py \
--output_dir=${OUTPUT_DIR} \
--model_name=transformer_decathlon \
--training_dir=${DATA_DIR}/data_splits/Task01_BrainTumour_train.csv \
--validation_dir=${DATA_DIR}/data_splits/Task01_BrainTumour_val.csv \
--ood_dir=${DATA_DIR}/data_splits/Task01_BrainTumour_test.csv,${DATA_DIR}/data_splits/Task02_Heart_test.csv,${DATA_DIR}/data_splits/Task04_Hippocampus_test.csv,${DATA_DIR}/data_splits/Task05_Prostate_test.csv,${DATA_DIR}/data_splits/Task06_Lung_test.csv,${DATA_DIR}/data_splits/Task07_Pancreas_test.csv,${DATA_DIR}/data_splits/Task08_HepaticVessel_test.csv,${DATA_DIR}/data_splits/Task09_Spleen_test.csv,${DATA_DIR}/data_splits/Task10_Colon_test.csv
--cache_data=0
--batch_size=2
--num_workers=0
--vqvae_checkpoint=${OUTPUT_DIR}/vqgan_decathlon/checkpoint.pth
--transformer_checkpoint=${OUTPUT_DIR}/transformer_decathlon/checkpoint.pth
--transformer_type=transformer
--image_roi=[160,160,128]
--image_size=128
```

You can print the results with:
```bash
python print_ood_scores.py --results_file=${OUTPUT_DIR}/transformer_decathlon/results.csv
```

### Acknowledgements
Built on top of [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels) and [MONAI](https://github.com/Project-MONAI/MONAI).

### Citation
If you use these codebase, please cite the following paper:
```
@inproceedings{graham2022transformer,
  title={Transformer-based out-of-distribution detection for clinically safe segmentation},
  author={Graham, Mark S and Tudosiu, Petru-Daniel and Wright, Paul and Pinaya, Walter Hugo Lopez and Jean-Marie, U and Mah, Yee H and Teo, James T and Jager, Rolf and Werring, David and Nachev, Parashkev and others},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  pages={457--476},
  year={2022},
  organization={PMLR}
}
```
