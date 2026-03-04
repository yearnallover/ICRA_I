import sys
import os

print(f"python path: {sys.executable}")

# 1. Check GPU
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA NOT Available")
except ImportError:
    print("❌ PyTorch Import Failed")

# 2. Check LeRobot
try:
    import lerobot
    print(f"✅ LeRobot Imported: {lerobot.__package__}")
except ImportError:
    print("❌ LeRobot Import Failed")

# 3. Check ROS Bridge
try:
    import rospy
    import rospkg
    from cv_bridge import CvBridge
    print("✅ ROS-Python Bridge Working")
except ImportError as e:
    print(f"❌ ROS Import Failed: {e}")
    print("   (Did you set PYTHONPATH?)")
