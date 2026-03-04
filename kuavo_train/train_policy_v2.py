import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import copy
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.policies.factory import make_pre_post_processors
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext
from lerobot.processor import ProcessorStep, NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
# import ipdb


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler

def build_policy(name, policy_cfg):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": CustomACTPolicyWrapper,
    }[name](policy_cfg)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

class AugmentationProcessorStep(ProcessorStep):
    def __init__(self, transform, cam_keys):
        super().__init__()
        self.transform = transform
        self.cam_keys = [k for k in cam_keys if "depth" not in k]  # list of keys in the transition dict to augment

    def __call__(self, transition):
        # Store the current transition (required by ProcessorStep)
        new_transition = transition.copy()

        # Apply transform to each camera key
        data_dict = new_transition.get(TransitionKey.OBSERVATION)
        if data_dict is not None:
            # new_data_dict = {
            #     k: self.transform(v) if k in self.cam_keys else v
            #     for k, v in data_dict.items()
            # }
            new_data_dict = {}
            for k, v in data_dict.items():
                
                if k in self.cam_keys:
                    # print(k)
                    new_data_dict[k] = self.transform(v)
                else:
                    new_data_dict[k] = v
            # print(new_data_dict['observation.images.head_cam_h'].device)
            new_transition[TransitionKey.OBSERVATION] = new_data_dict
            return new_transition
        else:
            return new_transition
        

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
    

def insert_before_normalizer(pipeline, new_step):
    """
    Insert a processor step before the first NormalizerProcessorStep.
    If no NormalizerProcessorStep is found, append at the end.
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, NormalizerProcessorStep):
            pipeline.steps.insert(i, new_step)
            print(f"Inserted {new_step.__class__.__name__} before NormalizerProcessorStep", {i})
            return new_step
    pipeline.steps.append(new_step)
    print(f"No NormalizerProcessorStep found, appended {new_step.__class__.__name__} at the end")
    return new_step

def remove_aug_step(pipeline, step_to_remove):
    """
    Remove the given step from the pipeline if it exists.
    """
    if step_to_remove in pipeline.steps:
        pipeline.steps.remove(step_to_remove)
        print(f"Removed {step_to_remove.__class__.__name__}")
    else:
        print(f"Step {step_to_remove.__class__.__name__} not found in pipeline")

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    # Setup output directory
    output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # Validate roots
    if "train_root" not in cfg or cfg.train_root is None:
        raise ValueError("cfg.train_root must be provided. Please check your config yaml.")
    
    if "test_root" not in cfg or cfg.test_root is None:
        raise ValueError("cfg.test_root must be provided for testing dataset.")

    # Dataset metadata and features
    # CRITICAL: Initialize metadata using ONLY the Training Dataset (cfg.train_root)
    # This ensures normalization stats are calculated from Training data to prevent leakage.
    train_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.train_root)
    print("Camera_keys:", train_metadata.camera_keys)
    print("Original dataset features:", train_metadata.features)

    features = dataset_to_policy_features(train_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("policy_cfg", policy_cfg)

    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg)
    # create preprocessor using stats from Training Metadata
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=train_metadata.stats)
    
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, train_metadata.info["total_frames"])
    
    # Initialize AMP GradScaler if use_amp is True
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # autocast context (cuda, or no-op when disabled/non-cuda)
    has_torch_autocast = hasattr(torch, "autocast")
    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)  # noqa
            else:
                from torch.cuda.amp import autocast as cuda_autocast  # noqa
                return cuda_autocast()
        # Fallback: disable on non-cuda to avoid dtype surprises
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)
    # print("scaler", device.type, make_autocast(amp_enabled))
    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # ===== Resume logic (perfect resume for AMP & RNG) =====
    
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print("Resuming from:", resume_path)
        try:
            # Load RNG state
            load_rng_state(resume_path / "rng_state.pth")
            
            # Load policy
            policy = policy.from_pretrained(resume_path, strict=True)
            preprocessor = preprocessor.from_pretrained(resume_path,config_filename="policy_preprocessor.json")

            """ Warning: using `from_pretrained` creates a new policy instance, 
            so the optimizer must be reinitialized here! """
            # print("load policy done ! ")
            optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, train_metadata.info["total_frames"])
            
            # Load optimizer, scheduler, scaler and training state
            checkpoint = torch.load(resume_path / "learning_state.pth", map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer"])

            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            if "scaler" in checkpoint and amp_enabled:
                scaler.load_state_dict(checkpoint["scaler"])
            
            if "steps" in checkpoint:
                steps = checkpoint["steps"]
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
            
            # Copy and load log_event
            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)
                
            print(f"Resumed training from epoch {start_epoch}, step {steps}")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            return
    else:
        print("Training from scratch!")

    # Deepcopy the preprocessor for Testing.
    # The 'preprocessor' will be modified with Augmentations for training.
    # The 'test_preprocessor' must remain clean (Normalizer only).
    test_preprocessor = copy.deepcopy(preprocessor)

    policy.train().to(device)
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Using AMP: {amp_enabled}")
    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(train_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    train_dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.train_root,
        image_transforms=None,
    )

    # Testing Dataset
    test_dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.test_root,
        image_transforms=None,
    )

    # Training loop setup
    # Add Augmentation to Training Preprocessor
    aug_step = insert_before_normalizer(preprocessor, AugmentationProcessorStep(image_transforms, train_dataset.meta.camera_keys))  # just for training
    
    # Prepare Samplers
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # Training Sampler (Shuffle=True)
        train_sampler = EpisodeAwareSampler(
            train_dataset.meta.episodes["dataset_from_index"],
            train_dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        # Testing Sampler (Shuffle=False)
        test_sampler = EpisodeAwareSampler(
            test_dataset.meta.episodes["dataset_from_index"],
            test_dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
        train_shuffle = False # Handled by sampler
    else:
        train_shuffle = True
        train_sampler = None
        test_sampler = None

    # Test DataLoader (created once)
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=(device.type != "cpu"),
        drop_last=False,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )

    for epoch in range(start_epoch, cfg.training.max_epoch):
        # Create Train DataLoader (reshuffled if needed via sampler or shuffle arg)
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=2 if cfg.training.num_workers > 0 else None,
        )

        epoch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")

        total_loss = 0.0
        total_l1_loss = 0.0

        policy.train() # Ensure train mode
        
        for batch in epoch_bar:
            batch = preprocessor(batch)  # will normalize and put batch to device
            with make_autocast(amp_enabled):
                loss, loss_dict = policy.forward(batch)
            # Scale loss and backward with AMP if enabled
            scaled_loss = loss / cfg.training.accumulation_steps
            
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    # Optionally unscale and clip gradients here if you use clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss_step", scaled_loss.item(), steps)
                writer.add_scalar("train/lr_step", lr_scheduler.get_last_lr()[0], steps)
                writer.add_scalar("l1_loss_step", loss_dict["l1_loss"], steps)

                epoch_bar.set_postfix(loss=f"{scaled_loss.item():.3f}", step=steps, lr=lr_scheduler.get_last_lr()[0])

            steps += 1
            total_loss += scaled_loss.item()
            total_l1_loss += loss_dict["l1_loss"]
        
        # Log average train loss for the epoch
        num_batches = len(train_dataloader) if len(train_dataloader) > 0 else 1
        avg_train_loss = total_loss / num_batches
        avg_train_l1_loss = total_l1_loss / num_batches

        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("train/lr_epoch", lr_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/l1_loss_epoch", avg_train_l1_loss, epoch)


        # -------------------------------------------------------------------------
        # Testing Loop
        # -------------------------------------------------------------------------
        policy.eval()
        test_loss_sum = 0.0
        test_steps_count = 0
        
        # Disable gradient calculation for testing
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, desc=f"Evaluating Epoch {epoch+1}", leave=False)
            for batch in test_pbar:
                batch = test_preprocessor(batch) # Use test_preprocessor (Normalizer ONLY)
                with make_autocast(amp_enabled):
                    loss, _ = policy.forward(batch)
                test_loss_sum += loss.item()
                test_steps_count += 1
        
        avg_test_loss = test_loss_sum / test_steps_count if test_steps_count > 0 else float('inf')
        writer.add_scalar("test/l1_loss_epoch", avg_test_loss, epoch)
        writer.add_scalar("test/l1_loss_step", avg_test_loss, steps if steps == 0 else steps - 1)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f}")

        # Use Average Test Loss for Best Model comparison
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            # Save best model
            policy.save_pretrained(output_directory / "epochbest")
            print(f"New best model saved with test loss {best_loss:.5f}")

        # Reset to train mode
        policy.train()

        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            policy.save_pretrained(output_directory / f"epoch{epoch+1}")
            # preprocessor.save_pretrained(output_directory)

        # Save last checkpoint (includes AMP scaler & progress for perfect resume)
        # Save last checkpoint
        policy.save_pretrained(output_directory)

        # Save training state including optimizer, scheduler, scaler, and step/epoch info
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch + 1,
            "best_loss": best_loss
        }
        torch.save(checkpoint, output_directory / "learning_state.pth")
        save_rng_state(output_directory / "rng_state.pth")

    writer.close()


if __name__ == "__main__":
    main()
