import torch

if torch.cuda.is_available():
    print("CUDA is available! 🎉")
else:
    print("CUDA is not available. 😔")



# 检查 CUDA 是否可用，如果可用则表示至少有一个 GPU 可用
if torch.cuda.is_available():
    print(f"GPU 个数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("没有检测到 GPU。")



print(f"cuDNN available: {torch.backends.cudnn.enabled}")