# 这个版本的代码删除了推理过程中的image io 操作

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections import deque
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
class DualGripperNearestPoseReleaseAssist:
    """
    当两只夹爪都闭合时：
    - 如果模型对某一只夹爪产生“松爪意图”（gripper cmd 从闭合 -> 打开），则接管该臂：
      1) 比较当前关节到 poseA / poseB 距离，选最近
      2) 平滑移动到最近 pose
      3) 打开夹爪保持 open_hold_steps
      4) 交还控制权
    """

    def __init__(
        self,
        poseA_left: np.ndarray,
        poseB_left: np.ndarray,
        poseA_right: np.ndarray,
        poseB_right: np.ndarray,
        ros_rate: float = 10.0,
        move_time: float = 1.0,
        open_hold_time: float = 0.4,
        return_time: float = 0.6,
        close_threshold: float = 0.75,
        open_threshold: float = 0.2,
        lockout_time: float = 0.5,
    ):

        self.poseA_left = np.asarray(poseA_left, dtype=np.float32).copy()
        self.poseB_left = np.asarray(poseB_left, dtype=np.float32).copy()
        self.poseA_right = np.asarray(poseA_right, dtype=np.float32).copy()
        self.poseB_right = np.asarray(poseB_right, dtype=np.float32).copy()

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold

        self.move_steps = max(1, int(move_time * ros_rate))
        self.open_hold_steps = max(1, int(open_hold_time * ros_rate))
        self.return_steps = max(1, int(return_time * ros_rate))
        self.lockout_steps = max(0, int(lockout_time * ros_rate))

        self.state = {"left": 0, "right": 0}
        self.step_counter = {"left": 0, "right": 0}

        self.active_arm = None
        self.selected_pose = {"left": None, "right": None}
        self.start_joints = {"left": None, "right": None}
        self.start_gripper = {"left": None, "right": None}
        self.prev_gripper_cmd = None

        self.mid_threshold = 0.75        # 触发时要求当前值 <= 这个（你可以从0.6~0.5试）
        self.drop_delta = 0.02         # 3步累计下降幅度阈值（你可以从0.05~0.15试）
        self.hist_len = 2
        self._hist = deque(maxlen=self.hist_len)
        log_robot.info("🧲 ReleaseAssist initialized.")

    def _indices(self, which_arm='both'):
        if which_arm == "both":
            return {
                "left":  (0, 7, 7),
                "right": (8, 15, 15),
            }
        elif which_arm == "left":
            return {"left": (0, 7, 7), "right": None}
        elif which_arm == "right":
            return {"left": None, "right": (0, 7, 7)}
        else:
            return {"left": None, "right": None}

    @staticmethod
    def _l2(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.sqrt(np.sum(d * d)))

    def is_active(self):
        return self.active_arm is not None

    def process_action(self, action, current_joint_q, which_arm="both"):

        idx = self._indices(which_arm)

        left_grp = None
        right_grp = None

        if idx["left"] is not None:
            _, _, lgi = idx["left"]
            if len(action) > lgi:
                left_grp = float(action[lgi])

        if idx["right"] is not None:
            _, _, rgi = idx["right"]
            if len(action) > rgi:
                right_grp = float(action[rgi])

        log_robot.info(f"[ReleaseAssist] Gripper cmd L={left_grp}, R={right_grp}")

        # ===== 1) 记录历史：只要 L/R 都存在就记录（不要绑在 both_closed 上）=====
        if left_grp is not None and right_grp is not None:
            self._hist.append((left_grp, right_grp))

        # ===== 2) 用“上一帧”判断是否曾经双爪闭合（gate）=====
        prev_l = prev_r = None
        if self.prev_gripper_cmd is not None:
            prev_l, prev_r = self.prev_gripper_cmd

        both_closed_prev = False
        if prev_l is not None and prev_r is not None:
            both_closed_prev = (prev_l >= self.close_threshold) and (prev_r >= self.close_threshold)

        # 当前帧的 both_closed 也可打印（只是观察用）
        both_closed_now = False
        if left_grp is not None and right_grp is not None:
            both_closed_now = (left_grp >= self.close_threshold) and (right_grp >= self.close_threshold)

        log_robot.info(f"[ReleaseAssist] both_closed_prev={both_closed_prev}, both_closed_now={both_closed_now}")

        # ===== 3) 基于最近3步历史判断“松爪意图”=====
        # def triggered_by_hist(side: str) -> bool:
        #     if len(self._hist) < 3:
        #         return False
        #     i = 0 if side == "left" else 1
        #     v0 = self._hist[0][i]
        #     v1 = self._hist[1][i]
        #     v2 = self._hist[2][i]

        #     monotonic_down = (v0 >= v1 >= v2)
        #     drop_ok = (v0 - v2) >= self.drop_delta
        #     low_enough = (v2 <= self.mid_threshold)

        #     return monotonic_down and drop_ok and low_enough
        def triggered_by_hist(side: str) -> bool:
            if len(self._hist) < 2:
                return False
            i = 0 if side == "left" else 1
            v0 = self._hist[0][i]
            v1 = self._hist[1][i]

            monotonic_down = (v0 >= v1)
            drop_ok = (v0 - v1) >= self.drop_delta
            low_enough = (v1 <= self.mid_threshold)

            return monotonic_down and drop_ok and low_enough
        triggered_left = triggered_by_hist("left")
        triggered_right = triggered_by_hist("right")

        log_robot.info(
            f"[ReleaseAssist-HIST] hist={list(self._hist)} "
            f"trigL={triggered_left}, trigR={triggered_right}"
        )
        if triggered_left or triggered_right:
            log_robot.info(f"🧲active_arm={self.active_arm},both_closed_prev={both_closed_prev}")
        # ===== 4) 触发接管：要求上一帧双爪闭合 + 当前未在接管中 =====
        # if both_closed_prev and self.active_arm is None:
        if self.active_arm is None:
            if triggered_left:
                log_robot.info("🧲 Trigger LEFT release assist (hist)")
                self._start_arm("left", action, current_joint_q, idx)
            elif triggered_right:
                log_robot.info("🧲 Trigger RIGHT release assist (hist)")
                self._start_arm("right", action, current_joint_q, idx)

        # ===== 5) 如果已接管，运行 FSM 覆盖 action =====
        if self.active_arm is not None:
            log_robot.info(f"[ReleaseAssist] Active arm = {self.active_arm}")
            self._run_fsm_for_arm(self.active_arm, action, idx)
            other = "right" if self.active_arm == "left" else "left"
            if idx.get(other) is not None:
                _, _, other_gi = idx[other]
                if len(action) > other_gi:
                    action[other_gi] = max(float(action[other_gi]), 0.9)  # 强制另一爪抓紧
        # ===== 6) 最后再更新 prev（确保 prev 表示“上一帧”）=====
        self.prev_gripper_cmd = (left_grp, right_grp)

        return action

    def _start_arm(self, arm, action, current_joint_q, idx_map):

        self.active_arm = arm
        self.state[arm] = 1
        self.step_counter[arm] = 0

        js, je, gi = idx_map[arm]

        self.start_joints[arm] = action[js:je].copy()
        self.start_gripper[arm] = float(action[gi])

        cur = current_joint_q[js:je].copy()

        if arm == "left":
            if cur[1] > -0.15:
                self.selected_pose[arm] = self.poseA_left 
            else:
                self.selected_pose[arm] = self.poseB_left 
            # dA = self._l2(cur[:3], self.poseA_left[:3])
            # dB = self._l2(cur[:3], self.poseB_left[:3])
            # if dA <= dB else self.poseB_left
        else:
            if cur[1] < 0.12:
                self.selected_pose[arm] = self.poseA_right 
            else:
                self.selected_pose[arm] = self.poseB_right 
            # dA = self._l2(cur[:3], self.poseA_right[:3])
            # dB = self._l2(cur[:3], self.poseB_right[:3])
            # self.selected_pose[arm] = self.poseA_right 
            # if dA <= dB else self.poseB_right

        # log_robot.info(
        #     f"🧲 {arm} start. Distance A={dA:.4f}, B={dB:.4f}. "
        #     f"Choose {'A' if dA <= dB else 'B'}"
        # )

    def _run_fsm_for_arm(self, arm, action, idx_map):

        js, je, gi = idx_map[arm]
        st = self.state[arm]
        self.step_counter[arm] += 1
        k = self.step_counter[arm]

        log_robot.info(
            f"[ReleaseAssist FSM] arm={arm}, state={st}, step={k}"
        )

        target_pose = self.selected_pose[arm]

        if target_pose is None:
            log_robot.info("[ReleaseAssist] No target pose. Finish.")
            self._finish_arm(arm)
            return

        if st == 1:
            alpha = min(k / self.move_steps, 1.0)
            action[js:je] = (1 - alpha) * self.start_joints[arm] + alpha * target_pose
            action[gi] = max(float(action[gi]), self.close_threshold)

            log_robot.info(
                f"[ReleaseAssist MOVING] alpha={alpha:.3f}"
            )

            if k >= self.move_steps:
                self.state[arm] = 2
                self.step_counter[arm] = 0
                log_robot.info(f"🧲 {arm} reached pose. Opening...")

        elif st == 2:
            action[js:je] = target_pose
            action[gi] = 0.0

            log_robot.info(
                f"[ReleaseAssist OPEN_HOLD] step={k}/{self.open_hold_steps}"
            )

            if k >= self.open_hold_steps:
                self.state[arm] = 3
                self.step_counter[arm] = 0
                log_robot.info(f"🧲 {arm} open hold done.")

        elif st == 3:
            log_robot.info(f"[ReleaseAssist RETURNING] immediate finish.")
            self.state[arm] = 4
            self.step_counter[arm] = 0

        elif st == 4:
            log_robot.info(
                f"[ReleaseAssist LOCKOUT] step={k}/{self.lockout_steps}"
            )
            if k >= self.lockout_steps:
                self._finish_arm(arm)

    def _finish_arm(self, arm):

        log_robot.info(f"🧲 {arm} release assist FINISHED.")

        self.state[arm] = 0
        self.step_counter[arm] = 0
        self.selected_pose[arm] = None
        self.start_joints[arm] = None
        self.start_gripper[arm] = None
        self.active_arm = None
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
    # ===== 你需要填入这 4 组 7-DoF 关节坐标 =====
    # poseA_left = np.array([-1.07,0.591,-1.23,-1.82,-0.55,-0.0646,-0.00558],  dtype=np.float32)   
    # poseB_left = np.array([-1.23,0.0418,-1.32,-1.39,-0.298,0.000237,-0.00649], dtype=np.float32)  
    # poseA_right = np.array([-0.715,-0.445,0.943,-1.79,0.377,-0.0257,0.0209], dtype=np.float32)
    # poseB_right = np.array([-0.715,0.0437,1.15,-1.18,0.377,-0.0257,0.0209],  dtype=np.float32)
    poseA_left = np.array([-0.338,0.115,-0.644,-1.77,-1.51,-0.155,0.105],  dtype=np.float32)   
    poseB_left = np.array([-0.668,-0.154,-0.833,-1.61,-1.51,-0.185,0.0419], dtype=np.float32)  
    poseA_right = np.array([-0.55,-0.0418,0.613,-1.85,1.46,0.0445,0.0349], dtype=np.float32)
    poseB_right = np.array([-0.456,0.3,0.723,-1.39,1.46,0.0445,0.0349],  dtype=np.float32)
# /home/yearn/icra2026_data/kuavo_data_challenge/configs/deploy/kuavo_env_TASK1_dino_act.yaml
    release_assist = DualGripperNearestPoseReleaseAssist(
        poseA_left=poseA_left,
        poseB_left=poseB_left,
        poseA_right=poseA_right,
        poseB_right=poseB_right,
        ros_rate=env.unwrapped.ros_rate,
        move_time=1.0,          # 可调：移动到目标位姿的时间
        open_hold_time=0.4,     # 可调：到位后张开夹爪保持时间
        return_time=0.0,        # 这里类里简化为直接结束接管，你留着也无所谓
        lockout_time=0.5,       # 可调：避免刚松完又触发
    )
    step = 0
    done = False
    while not done:
        # --- Pause support: block here if pause_flag is set ---
        if not check_control_signals():
            log_robot.info("🛑 Stop signal detected, exiting robot arm motion")
            return 0
        start_time = time.time()
        obs_raw = observation

        # 取当前关节角，用于 release_assist 的“选最近 pose”
        raw_state = obs_raw.get("joint_q", obs_raw.get("observation.state"))
        current_joint_q = None
        if raw_state is not None:
            if isinstance(raw_state, torch.Tensor):
                current_joint_q = raw_state.squeeze().detach().cpu().numpy()
            else:
                current_joint_q = np.asarray(raw_state).squeeze()

        # which_arm（可选）
        try:
            which_arm_val = env.unwrapped.which_arm
        except Exception:
            which_arm_val = "both"


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

        # ========= 关键：插入 release_assist（必须在 env.step 前）=========
        if current_joint_q is not None:
            numpy_action = release_assist.process_action(
                numpy_action,
                current_joint_q=current_joint_q,
                which_arm=which_arm_val
            )
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
    
    # for cam in cam_keys:
    #     temp_dir = frame_temp_dirs[cam]
    #     frame_files = sorted(temp_dir.glob("frame_*.png"))
    #     frames = [imageio.imread(str(f)) for f in frame_files]
    #     output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
    #     imageio.mimsave(str(output_path), frames, fps=fps)
        

    #     for f in frame_files:
    #         f.unlink()
    #     temp_dir.rmdir()
        
    #     del frames

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

