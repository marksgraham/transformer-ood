import argparse

from src.trainers import TransformerTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_dir", help="Location of folder with training niis.")
    parser.add_argument("--validation_dir", help="Location of folder with validation niis.")
    parser.add_argument(
        "--ood_dir", help="Location or list of locations of folder(s) to perform OOD on."
    )

    # model params
    parser.add_argument("--vqvae_checkpoint", help="Path to a VQ-VAE model checkpoint.")
    parser.add_argument(
        "--spatial_dimension", default=3, type=int, help="Dimension of images: 2d or 3d."
    )
    parser.add_argument("--image_size", default=None, help="Resize images.")
    parser.add_argument("--transformer_checkpoint", help="Path to a VQ-VAE model checkpoint.")
    parser.add_argument(
        "--transformer_type",
        default="transformer",
        help="transformer or performer or memory-efficient",
    )
    parser.add_argument(
        "--transformer_max_seq_length",
        default=None,
        help="Maximum sequence length for transformer.",
    )

    # training param
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Number of epochs to between evaluations.",
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=1,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_transformer.py --args
if __name__ == "__main__":
    args = parse_args()
    trainer = TransformerTrainer(args)
    trainer.ood(args)
