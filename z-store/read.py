import torch
import sys
import os
import pprint

# ================= 配置区域 =================
# 输入文件名 (你的 .pt 文件路径)
INPUT_PATH = "/home/preprocess/_funsearch/baseline/drl_coe_ckpt.pt"
# 输出文件名 (将生成的文本文件)
OUTPUT_PATH = "ckpt_full_content.txt"
# ===========================================

def save_full_content():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到文件 {INPUT_PATH}，请确认路径是否正确。")
        return

    print(f"正在读取 {INPUT_PATH} ... (如果文件很大可能需要几秒)")
    try:
        # 加载 Checkpoint (强制映射到 CPU，防止显存不足)
        checkpoint = torch.load(INPUT_PATH, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"正在将完整内容写入 {OUTPUT_PATH} ...")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 1. 写入文件头信息
        f.write(f"Checkpoint 完整内容导出报告\n")
        f.write(f"源文件: {INPUT_PATH}\n")
        f.write("=" * 60 + "\n\n")

        # 2. 遍历字典中的每一个键
        for key, value in checkpoint.items():
            f.write(f"{'='*20} KEY: {key} {'='*20}\n")
            
            # 针对不同类型的数据做格式化处理
            if key == 'optim_state':
                f.write("(这是 Adam 优化器的内部状态，包含 step, exp_avg, exp_avg_sq 等参数)\n")
                # 使用 str() 可以将 Tensor 转换为可读的字符串（包含数值预览）
                f.write(str(value))
            
            elif key == 'policy_state':
                f.write("(这是 PPO 策略网络的具体权重参数)\n")
                f.write(str(value))
                
            elif key == 'args':
                f.write("(这是训练时的超参数配置)\n")
                # 使用 pprint 模块美化字典打印
                pprint.pprint(value, stream=f)
                
            else:
                # 其他简单数据 (如 episode)
                f.write(str(value))
            
            f.write("\n\n" + "-" * 60 + "\n\n")

    print(f"✅ 成功！完整内容已保存至: {os.path.abspath(OUTPUT_PATH)}")
    print("您可以下载该 txt 文件并使用文本编辑器查看优化器细节。")

if __name__ == "__main__":
    # 设置打印选项，防止 Tensor 显示过多的省略号 (但也不要无限长，防止文件几个G)
    torch.set_printoptions(edgeitems=5, threshold=1000, linewidth=120)
    save_full_content()