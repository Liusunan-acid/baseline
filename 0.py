import torch
import time
import sys

def stress_gpu_lite(target_memory_gb=5, target_utilization=0.1):
    """
    轻负载 GPU 占用脚本
    
    Args:
        target_memory_gb (int): 目标占用显存大小(GB)，默认为 10。
        target_utilization (float): 目标 GPU 使用率 (0.0 - 1.0)，0.1 代表 10%。
    """
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到支持 CUDA 的 GPU 设备。")
        return

    device = torch.device("cuda")
    gpu_props = torch.cuda.get_device_properties(device)
    total_vram = gpu_props.total_memory / (1024**3)
    
    print(f"✅ 检测到 GPU: {gpu_props.name}")
    print(f"📊 总显存: {total_vram:.2f} GB")
    print(f"🎯 目标: 占用 {target_memory_gb} GB 显存, 保持约 {target_utilization*100:.0f}% 使用率")

    # --- 第一步：分配显存 ---
    allocated_tensors = []
    one_gb_elements = 1024 * 1024 * 1024 // 4 # 1GB float32
    
    print("\n🚀 开始分配显存...")
    gb_count = 0
    
    try:
        while gb_count < target_memory_gb:
            try:
                # 申请显存
                tensor = torch.zeros(one_gb_elements, dtype=torch.float32, device=device)
                allocated_tensors.append(tensor)
                gb_count += 1
                sys.stdout.write(f"\r⏳ 已占用显存: {gb_count} / {target_memory_gb} GB")
                sys.stdout.flush()
            except RuntimeError as e:
                print(f"\n⚠️ 显存不足，无法达到 {target_memory_gb}GB，已停止分配。")
                break
                
        print(f"\n🔒 显存占用已稳定在 {gb_count} GB。开始轻负载循环...")
        print("按 Ctrl+C 停止脚本。\n")

        # --- 第二步：轻负载计算循环 ---
        # 减小矩阵尺寸以获得更精细的控制
        size = 2048 
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        step = 0
        while True:
            # 1. 记录开始时间
            start_time = time.time()
            
            # 2. 执行计算 (工作)
            c = torch.matmul(a, b)
            torch.cuda.synchronize() # 等待计算真正完成
            
            # 3. 计算工作耗时
            work_time = time.time() - start_time
            
            # 4. 计算需要的休眠时间
            # 公式: work_time / (work_time + sleep_time) = utilization
            # 变换得: sleep_time = work_time * (1 - utilization) / utilization
            if target_utilization > 0:
                sleep_time = work_time * (1 - target_utilization) / target_utilization
            else:
                sleep_time = 1.0 # 如果设为0%，则只睡不干
            
            # 5. 休眠
            time.sleep(sleep_time)
            
            step += 1
            if step % 5 == 0:
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断。正在释放资源...")
    finally:
        del allocated_tensors
        del a
        del b
        torch.cuda.empty_cache()
        print("✅ 资源已释放。")

if __name__ == "__main__":
    # 这里设置显存为 10GB，使用率为 10% (0.1)
    stress_gpu_lite(target_memory_gb=15, target_utilization=0.5)

#             CUDA_VISIBLE_DEVICES=3 nohup python 0.py > 0.log 2>&1 &