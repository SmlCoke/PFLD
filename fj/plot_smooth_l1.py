import numpy as np
import matplotlib.pyplot as plt
import os

def smooth_l1_loss(x, beta):
    """
    计算 Smooth L1 Loss
    公式:
    0.5 * x^2 / beta,    if |x| < beta
    |x| - 0.5 * beta,    otherwise
    """
    x_abs = np.abs(x)
    loss = np.where(x_abs < beta, 
                    0.5 * x_abs ** 2 / beta, 
                    x_abs - 0.5 * beta)
    return loss

def l1_loss(x):
    return np.abs(x)

def l2_loss(x):
    # 使用 0.5 * x^2 以便在原点附近的曲率与 SmoothL1 (beta=1) 匹配
    return 0.5 * x ** 2

def plot_losses():
    # 设置输入差值 x 的范围 (-3 到 3)，展示函数在原点附近的形态
    x = np.linspace(-3, 3, 1000)
    
    # 定义不同的 beta 值进行对比
    betas = [0.5, 1.0, 3.0]
    
    plt.figure(figsize=(12, 8))
    
    # 1. 绘制基准 L1 Loss
    plt.plot(x, l1_loss(x), 'k--', label='L1 Loss (|x|)', linewidth=1.5, alpha=0.5)
    
    # 2. 绘制基准 L2 Loss
    # 注意：为了视觉对比方便，通常对比的是 0.5*x^2
    plt.plot(x, l2_loss(x), 'g--', label='L2 Loss (0.5 * x^2)', linewidth=1.5, alpha=0.5)
    
    # 3. 绘制不同 beta 下的 Smooth L1 Loss
    colors = ['#FF5733', '#3357FF', '#C70039'] # 橙红, 蓝, 深红
    for beta, color in zip(betas, colors):
        y = smooth_l1_loss(x, beta)
        plt.plot(x, y, label=f'Smooth L1 (beta={beta})', color=color, linewidth=2.5)
        
        # 标注 beta 切换点
        # 在正半轴 x = beta 处，这是从 L2 切换到 L1 的转折点
        idx = np.searchsorted(x, beta) # 找到 beta 在 x 轴的近似索引
        if idx < len(x):
            plt.scatter([beta], [y[idx]], color=color, s=50, zorder=5)
            plt.annotate(f'β={beta}', (beta, y[idx]), xytext=(5, -15), textcoords='offset points', color=color)

    plt.title('Smooth L1 Loss Analysis: Combining L1 and L2 Characteristics', fontsize=14)
    plt.xlabel('Error (prediction - target)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # 添加解释性文本
    info_text = (
        "Analysis:\n"
        "• Near 0 (|x| < beta): Quadratic (L2-like), smooth gradient, close to 0.\n"
        "• Far from 0 (|x| > beta): Linear (L1-like), constant gradient, robust to outliers.\n"
        "• Beta controls the width of the quadratic region."
    )
    plt.text(-2.9, 3.5, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # 设置每个坐标轴的比例
    plt.axis('equal')
    plt.ylim(-0.1, 4.0)

    # 保存图片
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'smooth_l1_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to: {output_path}")
    
    # 尝试显示（如果在支持 GUI 的环境中运行）
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_losses()
