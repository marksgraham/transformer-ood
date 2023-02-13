import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.inferers import VQVAETransformerInferer
from generative.networks.nets import VQVAE, DecoderOnlyTransformer
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from monai.utils.misc import first
from torch.nn import CrossEntropyLoss

# from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_data import get_training_data_loader


class TransformerTrainer:
    def __init__(self, args):

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # set up VQ-VAE
        self.vqvae_model = VQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_layers=3,
            num_levels=4,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=(
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
            ),
            num_channels=[128, 128, 128, 256],
            num_res_channels=[128, 128, 128, 256],
            num_embeddings=256,
            embedding_dim=32,
        )
        self.vqvae_model.to(self.device)
        # load checkpoint
        vqvae_checkpoint_path = Path(args.vqvae_checkpoint)
        if vqvae_checkpoint_path.exists():
            vqvae_checkpoint = torch.load(vqvae_checkpoint_path)
            self.vqvae_model.load_state_dict(vqvae_checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError(f"Cannot find VQ-VAE checkpoint {args.vqvae_checkpoint}")
        print(
            f"VQ-VAE with {sum(p.numel() for p in self.vqvae_model.parameters()):,} model parameters"
        )

        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.training_dir,
            validation_ids=args.validation_dir,
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
        )
        data_sample = first(self.train_loader)
        latent_sample = self.vqvae_model.index_quantize(data_sample["image"].to(self.device))
        latent_spatial_shape = tuple(latent_sample.shape[1:])
        # set up transformer
        self.model = DecoderOnlyTransformer(
            num_tokens=256 + 1,
            max_seq_len=math.prod(latent_spatial_shape) + 1,
            attn_layers_dim=4,
            attn_layers_depth=22,
            attn_layers_heads=8,
            with_cross_attention=False,
        )
        self.model.to(self.device)
        self.inferer = VQVAETransformerInferer()
        print(
            f"Transformer with {sum(p.numel() for p in self.model.parameters()):,} model parameters"
        )

        self.ordering = Ordering(
            ordering_type=OrderingType.RASTER_SCAN.value,
            spatial_dims=3,
            dimensions=(1,) + latent_spatial_shape,
        )

        # set up optimizer, loss, checkpoints
        self.run_dir = Path(args.output_dir) / args.model_name
        checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(
                self.vqvae_model,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=110,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_step = 0
        self.model.train()
        ce_loss = CrossEntropyLoss()
        epoch_loss = 0
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.inferer(
                inputs=images,
                vqvae_model=self.vqvae_model,
                transformer_model=self.model,
                ordering=self.ordering,
            )

            # get the ground truth
            with torch.no_grad():
                latent = self.vqvae_model.index_quantize(images)
            latent = latent.reshape(latent.shape[0], -1)
            latent = latent[:, self.ordering.get_sequence_ordering()]
            latent = F.pad(latent, (1, 0), "constant", self.vqvae_model.num_embeddings)
            latent = latent.long()

            loss = ce_loss(prediction.transpose(1, 2), latent)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_step += images.shape[0]
            self.global_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "ce_loss": epoch_loss / epoch_step,
                }
            )
            self.logger_train.add_scalar(
                tag="ce_loss", scalar_value=loss.item(), global_step=self.global_step
            )

            if self.quick_test:
                break
        epoch_loss = epoch_loss / epoch_step
        return epoch_loss

    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=110,
                position=0,
                leave=True,
                desc="Validation",
            )

            global_val_step = self.global_step
            ce_loss = CrossEntropyLoss()

            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                prediction = self.inferer(
                    inputs=images,
                    vqvae_model=self.vqvae_model,
                    transformer_model=self.model,
                    ordering=self.ordering,
                )

                # get the ground truth
                with torch.no_grad():
                    latent = self.vqvae_model.index_quantize(images)
                    latent_spatial_dim = tuple(latent.shape[1:])
                latent = latent.reshape(latent.shape[0], -1)
                latent = latent[:, self.ordering.get_sequence_ordering()]
                latent = F.pad(latent, (1, 0), "constant", self.vqvae_model.num_embeddings)
                latent = latent.long()

                loss = ce_loss(prediction.transpose(1, 2), latent)

                self.logger_val.add_scalar(
                    tag="ce_loss", scalar_value=loss.item(), global_step=global_val_step
                )

                global_val_step += images.shape[0]

                # generate a sample

                if step == 0:
                    sample = self.inferer.sample(
                        starting_tokens=self.vqvae_model.num_embeddings
                        * torch.ones(1, 1).to(self.device),
                        latent_spatial_dim=latent_spatial_dim,
                        vqvae_model=self.vqvae_model,
                        transformer_model=self.model,
                        ordering=self.ordering,
                        verbose=False,
                    )
                    slices = [50, 100, 150]
                    fig = plt.figure()
                    for i in range(len(slices)):
                        plt.subplot(1, len(slices), i + 1)
                        plt.imshow(sample[0, 0, :, :, slices[i]].cpu(), cmap="gray")
                    plt.show()
                    self.logger_val.add_figure(
                        tag="samples", figure=fig, global_step=self.global_step
                    )