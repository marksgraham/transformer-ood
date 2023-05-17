import argparse
import ast

from src.trainers import VQVAETrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_dir", help="Location of folder with training niis.")
    parser.add_argument("--validation_dir", help="Location of folder with validation niis.")

    parser.add_argument("--image_size", default=None, help="Resize images.")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )
    # model params
    parser.add_argument("--vqvae_in_channels", default=1, type=int)
    parser.add_argument("--vqvae_out_channels", default=1, type=int)
    parser.add_argument("--vqvae_num_res_layers", default=3, type=int)
    parser.add_argument(
        "--vqvae_downsample_parameters",
        default=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--vqvae_upsample_parameters",
        default=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        type=ast.literal_eval,
    )
    parser.add_argument("--vqvae_num_channels", default=[128, 128, 128, 256], type=ast.literal_eval)
    parser.add_argument(
        "--vqvae_num_res_channels", default=[128, 128, 128, 256], type=ast.literal_eval
    )
    parser.add_argument("--vqvae_num_embeddings", default=256, type=int)
    parser.add_argument("--vqvae_embedding_dim", default=256, type=int)
    parser.add_argument("--vqvae_decay", default=0.99, type=float)

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


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_vqvae.py --args
if __name__ == "__main__":
    args = parse_args()
    trainer = VQVAETrainer(args)
    trainer.train(args)
