# 没加入debug代码的V5

# 这个版本在官方的基础上实现了松手后竖手腕以帮助物体脱落，强制抓手快速闭合
# 和V3相比差异在于可以直接指定世界坐标系下抓尖指向和水平面的角度(target_drop_angle_deg)和抬手高度(target_lift_height_m)
# 经过我的测试，duration=5, target_drop_angle_deg=90, target_lift_height_m=0.15)是一个比较合适的参数组合



#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""
import sys,os
import gc
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from lerobot_patches import custom_patches

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
import json
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading
import traceback
from geometry_msgs.msg import PoseStamped
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_service.client import PolicyClient
from lerobot.policies.factory import make_pre_post_processors
log_model = setup_logger("model")
log_robot = setup_logger("robot")

from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.kuavo_env.KuavoRealEnv import KuavoRealEnv
from kuavo_deploy.utils.ros_manager import ROSManager


init_evt = threading.Event()
pause_flag = threading.Event()
stop_flag = threading.Event()
success_evt = threading.Event()

def env_init_service(req):
    log_robot.info(f"env_init_callback! req = {req}")
    init_evt.set()
    return TriggerResponse(success=True, message="Env init successful")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

def env_success_callback(msg):
    # log_model.info("env_success_callback!")
    if msg.data:
        success_evt.set()


pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)

def safe_reset_service(reset_service) -> None:
    """安全重置服务 Safe resetting service"""
    try:
        # 调用重置服务 Reset
        response = reset_service(TriggerRequest())
        if response.success:
            log_robot.info(f"Reset service successful: {response.message}")
        else:
            log_robot.warning(f"Reset service failed: {response.message}")
    except rospy.ServiceException as e:
        log_robot.error(f"Reset service exception: {e}")

def check_control_signals():
    """检查控制信号 Check control signal"""
    # 检查暂停状态 Pause state
    while pause_flag.is_set():
        log_robot.info("🔄 Robot arm motion paused")
        time.sleep(0.1)
        if stop_flag.is_set():
            log_robot.info("🛑 Robot arm motion stopped")
            return False
    
    # 检查是否需要停止 Whether it needs to be stopped
    if stop_flag.is_set():
        log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
        return False
        
    return True  # 正常继续 Continue


    
def setup_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """
    Set up and load the policy model.
    
    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion' or 'act')
        
    Returns:
        Loaded policy model and device
    """
    
    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)  
    
    if policy_type == 'diffusion':
        policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'act':
        policy = CustomACTPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'client':
        policy = PolicyClient()
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    
    policy.eval()
    policy.to(device)
    policy.reset()
    # Log model info
    log_model.info(f"Model loaded from {pretrained_path}")
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"Model device: {device}")
    
    return policy


class DropAssist:
    """
    Implements a 'Drop Assist' or 'Object Release Sequence'.
    When the gripper open signal is detected, overrides the trajectory to shake/force the object off.

    Args:
        duration: Number of steps for the drop assist sequence
        target_drop_angle_deg: Target wrist pitch angle in degrees for vertical drop
        target_lift_height_m: Height to lift the forearm before dropping in meters
    """
    def __init__(self, duration=5, target_drop_angle_deg=90, target_lift_height_m=0.15, lockout_duration=30):
        self.active_arms = set()
        self.counter = 0
        self.duration = duration
        self.lockout_counter = 0
        self.lockout_duration = lockout_duration
        self.forearm_length_m = 0.30 # Length of the forearm segment in meters (目前假设是0.30m, 我已经调整好了)
        self.hardware_pitch_offset_rad = 1.57 #  Calibration offset for wrist pitch in radians, 用来补偿机械臂的安装偏差，不用调整
        
        # Real-World Kinematic Parameters
        self.target_drop_angle_deg = target_drop_angle_deg
        self.target_lift_height_m = target_lift_height_m
        
        self.prev_gripper_cmd = None
        
    def process_action(self, action, which_arm='both'):
        """
        action: numpy array of joint targets.
        Structure assumed: 
         Left: action[:7] joints, action[7] gripper
         Right: action[8:15] joints, action[15] gripper (if both)
        """
        # Identify indices
        left_grp_idx = None
        right_grp_idx = None
        left_wrist_idx = None
        right_wrist_idx = None
        left_shoulder_idx = None
        right_shoulder_idx = None
        left_elbow_idx = None
        right_elbow_idx = None

        if self.lockout_counter > 0:
            self.lockout_counter -= 1

        if which_arm == 'both':
            if len(action) >= 16:
                left_grp_idx = 7
                right_grp_idx = 15
                left_wrist_idx = 6
                right_wrist_idx = 14 # 8+6
                left_shoulder_idx = 0 
                right_shoulder_idx = 8
                left_elbow_idx = 3
                right_elbow_idx = 11 # 8+3
        elif which_arm == 'left':
            if len(action) >= 8:
                left_grp_idx = 7
                left_wrist_idx = 6
                left_shoulder_idx = 0
                left_elbow_idx = 3
        elif which_arm == 'right':
            if len(action) >= 8:
                right_grp_idx = 7
                right_wrist_idx = 6
                right_shoulder_idx = 0
                right_elbow_idx = 3
        
        # check trigger (Open detected)
        # Assume 0.0 is OPEN, 1.0 is CLOSED. Trigger on falling edge < 0.5?
        
        triggered_left = False
        triggered_right = False
        
        curr_left = action[left_grp_idx] if left_grp_idx is not None else None
        curr_right_val = action[right_grp_idx] if right_grp_idx is not None else None
        
        # Helper to detect edge
        def is_opening(curr, prev):
            if curr is None or prev is None: return False
            return (prev > 0.5) and (curr <= 0.5)

        if self.prev_gripper_cmd is not None:
             prev_l, prev_r = self.prev_gripper_cmd
             if is_opening(curr_left, prev_l):
                 triggered_left = True
             if is_opening(curr_right_val, prev_r):
                 triggered_right = True
        
        # Update history
        self.prev_gripper_cmd = (curr_left, curr_right_val)
        
        if triggered_left:
            self.active_arms.add('left')
            self.counter = self.duration
            
        if triggered_right:
            self.active_arms.add('right')
            self.counter = self.duration

        if self.counter > 0:
            t = (self.duration - self.counter) / self.duration # 0.0 to 1.0
            
            # Apply Logic
            # Decoupled timing: Retract Fast, Dump Slow
            elbow_progress = min(t * 5.0, 1.0) # Fast Lift (First 20%)
            wrist_progress = min(t * 5.0, 1.0) # Slow Rotation (First 50%)
            
            blend = wrist_progress
            
            # Step B: Calculate Elbow Angle for Height
            # Arc Length: theta = arc_length / radius
            required_elbow_lift_rad = self.target_lift_height_m / self.forearm_length_m
            # Use elbow_progress for lift
            current_lift_offset = required_elbow_lift_rad * elbow_progress
            
            if 'left' in self.active_arms and left_wrist_idx is not None and len(action) > left_wrist_idx:
                # Step A: Calculate Wrist Angle for Verticality
                current_shoulder_pitch = action[left_shoulder_idx]
                current_elbow_pitch = action[left_elbow_idx]
                
                # "Mini-Kinematics" with Manual Calibration
                target_world_rad = np.deg2rad(self.target_drop_angle_deg)
                current_arm_pitch_sum = current_shoulder_pitch + current_elbow_pitch
                # Formula: Req = Target - Sum - HardwareOffset
                required_wrist_rad = target_world_rad - current_arm_pitch_sum - self.hardware_pitch_offset_rad
                
                # LERP
                pitch_idx = left_wrist_idx - 1
                # Left Arm: Invert target for Mirroring
                target_val = -required_wrist_rad
                # Use wrist_progress (blend) for rotation
                action[pitch_idx] = action[pitch_idx] * (1-blend) + target_val * blend
                action[left_elbow_idx] -= current_lift_offset
            
            if 'right' in self.active_arms and right_wrist_idx is not None and len(action) > right_wrist_idx:
                # Step A: Calculate Wrist Angle for Verticality
                current_shoulder_pitch = action[right_shoulder_idx]
                current_elbow_pitch = action[right_elbow_idx]

                # "Mini-Kinematics" with Manual Calibration
                target_world_rad = np.deg2rad(self.target_drop_angle_deg)
                current_arm_pitch_sum = current_shoulder_pitch + current_elbow_pitch
                # Formula: Req = Target - Sum - HardwareOffset
                required_wrist_rad = target_world_rad - current_arm_pitch_sum - self.hardware_pitch_offset_rad

                pitch_idx = right_wrist_idx - 1
                # Right Arm: Standard Positive Down 
                # Use wrist_progress (blend) for rotation
                target_val = required_wrist_rad
                action[pitch_idx] = action[pitch_idx] * (1-blend) + target_val * blend
                action[right_elbow_idx] -= current_lift_offset

            self.counter -= 1
            if self.counter == 0:
                self.active_arms.clear()
                self.lockout_counter = self.lockout_duration # Start grace period

        # --- Grip Assist: Force Close ---
        # Relaxed Threshold: Only force close if intent is clearly > 0.1 AND not in lockout
        # Check Left Arm
        if left_grp_idx is not None and not self.active_arms and self.lockout_counter == 0 and len(action) > left_grp_idx:
            if action[left_grp_idx] > 0.1:
                action[left_grp_idx] = 0.9
                
        # Check Right Arm
        if right_grp_idx is not None and not self.active_arms and self.lockout_counter == 0 and len(action) > right_grp_idx:
            if action[right_grp_idx] > 0.1:
                action[right_grp_idx] = 0.9

        # --- Lockout: Force Open (Optional but Recommended) ---
        if self.lockout_counter > 0:
            if left_grp_idx is not None and len(action) > left_grp_idx: action[left_grp_idx] = 0.0
            if right_grp_idx is not None and len(action) > right_grp_idx: action[right_grp_idx] = 0.0

        # --- Drop Assist Safety Override ---
        # If Drop Assist is active, FORCE OPEN grippers to ensure release.
        # This overrules Grip Assist to preventing "sticking" during shake.
        if 'left' in self.active_arms and left_grp_idx is not None and len(action) > left_grp_idx:
            action[left_grp_idx] = 0.0
            
        if 'right' in self.active_arms and right_grp_idx is not None and len(action) > right_grp_idx:
            action[right_grp_idx] = 0.0
        
        return action

def run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory):
    """运行单个episode Running a single episode"""
    cfg = config.inference
    seed = cfg.seed
    task = cfg.task
    # Initialize environment
    env = gym.make(
        config.env.env_name,
        max_episode_steps=cfg.max_episode_steps,
        config=config,
    )

    run_single_ros_manager = ROSManager()
    # Setup ROS subscribers and services
    run_single_ros_manager.register_subscriber("/simulator/success", Bool, env_success_callback)

    # max_episode_steps = cfg.max_episode_steps

    start_service = rospy.ServiceProxy('/simulator/start', Trigger)


    if cfg.policy_type != 'client':
        log_model.info(f"policy.config.input_features: {policy.config.input_features}")
        log_robot.info(f"env.observation_space: {env.observation_space}")
        log_model.info(f"policy.config.output_features: {policy.config.output_features}")
        log_robot.info(f"env.action_space: {env.action_space}")

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    observation, info = env.reset(seed=seed)
    # first_img =  (observation["observation.images.head_cam_h"].squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
    
    # import cv2
    # first_img = cv2.cvtColor(first_img,cv2.COLOR_RGB2BGR)
    # cv2.imwrite( "obs.png", first_img)
    # raise ValueError("stop for debug!")
    start_service(TriggerRequest())

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]

    frame_temp_dirs = {}
    for k in cam_keys:
        temp_dir = output_directory / f"temp_frames_{episode}_{k}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        frame_temp_dirs[k] = temp_dir


    average_exec_time = 0
    average_action_infer_time = 0
    average_step_time = 0

    drop_assist = DropAssist()
    step = 0
    done = False
    while not done:
        # --- Pause support: block here if pause_flag is set ---
        if not check_control_signals():
            log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
            return 0
        start_time = time.time()
        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        log_model.info(f"Step {step}: predict action {action}")
        action = postprocessor(action)
        # print(f"action: {action}, action.shape: {action.shape}, action min: {action.min()}, action max: {action.max()}")
        action_infer_time = time.time()
        log_model.info(f"episode {episode}, step {step}, action infer time: {action_infer_time - start_time:.3f}s")
        average_action_infer_time += action_infer_time - start_time

        numpy_action = action.squeeze(0).cpu().numpy()

        # Apply Drop Assist to override trajectory when releasing object
        try:
            which_arm_val = env.unwrapped.which_arm
        except:
            which_arm_val = 'both' # Default fallback
            
        numpy_action = drop_assist.process_action(numpy_action, which_arm=which_arm_val)

        log_model.info(f"Step {step}: Executing action {numpy_action}")
        observation, reward, terminated, truncated, info = env.step(numpy_action)

        exec_time = time.time()
        log_model.debug(f"step {step}: exec time: {exec_time - action_infer_time:.3f}s")
        average_exec_time += exec_time - action_infer_time
        
        rewards.append(reward)

        # for k in cam_keys:
        #     frame_path = frame_temp_dirs[k] / f"frame_{step:04d}.png"
        #     img = (observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #     if img.shape[-1] == 1:
        #         img = img.squeeze(-1)
        #     imageio.imwrite(str(frame_path), img)

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        done = done or success_evt.is_set()
        step += 1

        end_time = time.time()
        log_model.debug(f"Step {step} time: {end_time - start_time:.3f}s")
        average_step_time += end_time - start_time
    
    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.unwrapped.ros_rate

    log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
    log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
    log_model.info(f"average step time: {average_step_time / step:.3f}s")
    log_model.info(f"average sleep time: {env.unwrapped.average_sleep_time / step:.3f}s")
    
    for cam in cam_keys:
        temp_dir = frame_temp_dirs[cam]
        frame_files = sorted(temp_dir.glob("frame_*.png"))
        frames = [imageio.imread(str(f)) for f in frame_files]
        output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
        imageio.mimsave(str(output_path), frames, fps=fps)
        

        for f in frame_files:
            f.unlink()
        temp_dir.rmdir()
        
        del frames

    success = success_evt.is_set()
    
    env.close()
    run_single_ros_manager.close()
    
    del rewards
    del observation
    del env
    del run_single_ros_manager
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return 1 if success else 0


def kuavo_eval_autotest(config: KuavoConfig):
    """执行自动测试 Auto testing script"""
    cfg = config.inference
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    policy_type = cfg.policy_type

    # Setup paths
    pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Log evaluation results
    log_file_path = output_directory / "evaluation_autotest.log"
    
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")
    
    
    # Setup policy and environment (只加载一次)
    set_seed(seed)
    device = torch.device(cfg.device)
    policy = setup_policy(pretrained_path, policy_type, device)
    preprocessor, postprocessor = make_pre_post_processors(None, Path(str(pretrained_path).split("/epoch", 1)[0]))
    
    # first reset
    reset_service = rospy.ServiceProxy('/simulator/reset', Trigger)
    # Ros service
    init_service = rospy.Service("/simulator/init", Trigger, env_init_service)


    wait_times = 8
    while not init_evt.is_set():
        log_robot.info("Waiting for first env init...")
        if not check_control_signals():
            log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
            return
        time.sleep(1)
        wait_times -= 1
        if wait_times <=0:
            break
    safe_reset_service(reset_service)
    init_evt.clear()

    success_count = 0
    for episode in range(eval_episodes):

        while not init_evt.is_set():
            log_robot.info("Waiting for env init...")
            if not check_control_signals():
                log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
                return
            time.sleep(1)
        try:
            result = run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory)
            log_robot.info(f"Episode {episode+1} completed with return code: {result}")
            
            # 重置policy状态，清理缓存 Reset policy, clear cache
            policy.reset()
            
            # 强制垃圾回收和GPU缓存清理 Force garbage collection and GPU cache cleaning
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            log_robot.error(f"Exception during episode {episode+1}: {e}")
            log_robot.error(traceback.format_exc())
            result = 0  # Treat as failure
            safe_reset_service(reset_service)
            init_evt.clear()
            success_evt.clear()
            
            # 异常情况下也要清理内存 Garbage collection even under an exception
            gc.collect()
            torch.cuda.empty_cache()
            break

        # 记录episode结果 Record episode result
        episode_end_time = datetime.datetime.now().isoformat()
        is_success = result == 1
        if is_success:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success!")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed!")



        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Success Count: {success_count} / Already eval episodes: {episode+1}")
    
        safe_reset_service(reset_service)
        init_evt.clear()
        success_evt.clear()
    

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    log_model.info(f"📁 Videos and logs saved to: {output_directory}")
    log_model.info("="*50)
    init_service.shutdown()
    pause_sub.unregister()
    stop_sub.unregister()

