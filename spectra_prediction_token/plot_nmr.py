import matplotlib.pyplot as plt
from spectra_prediction_token.NMRTokenizer import NMRSpectrumTokenizer

nmr_tokenizer = NMRSpectrumTokenizer()

def draw_hnmr_spectrum(peaks, ax, label=False, show=True):
    '''
    Draw the 1H NMR spectrum
    Args:
        peaks: list of dicts with keys:
            'centroid': float,
            'category': str,
            'nH': int,
            'j_values': float
    '''
    x = [peak['centroid'] for peak in peaks]
    y = [peak['nH'] for peak in peaks]
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(peaks)):
        ax.stem([x[i]], [y[i]], basefmt=" ", markerfmt=color[i % len(color)] + 'o',
                linefmt='b-')
        if label:
            ax.text(x[i], y[i]+0.1, f"{peaks[i]['category']}"
                                   f"\n{str(peaks[i]['centroid'])[:4]}"
                                   f"\n{peaks[i]['nH']}"
                                   f"\n{str(peaks[i]['j_values'])[:4]}",
                    fontsize=8, ha='center', va='bottom',
                    color=color[i % len(color)])

    # 修正：使用 ax.set_title 和 ax.set_xlabel 等
    ax.set_title('1H NMR Spectrum')
    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel('Intensity')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid()
    ax.set_ylim(0, max(y) * 1.15)  # 设置 y 轴范围
    ax.set_xlim(0, 10)
    if show:
        plt.show()



# 你的 draw_cnmr_spectrum_1 函数
def draw_cnmr_spectrum_1(peaks, ax, label=False, show=True):
    '''
    Draw the 13C NMR spectrum
    Args:
        peaks: list of dicts with keys:
            'delta (ppm)': float,
            'width (ppm)': float,
            'integral': float,
            'intensity': float
    '''

    x = [peak['delta (ppm)'] for peak in peaks]
    y = [peak['intensity'] for peak in peaks]
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(peaks)):
        ax.stem([x[i]], [y[i]], basefmt=" ", markerfmt=color[i % len(color)] + 'o',
                linefmt='b-')
        if label:
            ax.text(x[i]-8, y[i]*0.7,
                    f"\n{str(peaks[i]['delta (ppm)'])[:6]}",
                    fontsize=8, ha='center', va='bottom', rotation=90, color=color[i % len(color)])

    ax.set_title('13C NMR Spectrum')  # 修改为 ax.set_title
    ax.set_xlabel('Chemical Shift (ppm)')  # 修改为 ax.set_xlabel
    ax.set_ylabel('Intensity')  # 修改为 ax.set_ylabel
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid()
    ax.set_ylim(0, max(y) * 1.1)  # 设置 y 轴范围
    ax.set_xlim(0, 250)
    if show: plt.show()


def draw_spectra_comparison(spectra_gt_dict: dict, spectra_pred_dict: dict, show=True):
    '''
    在一张图的两个子图中绘制真实光谱和预测光谱的对比。

    参数:
        spectra_gt_dict: 真实光谱字典
        spectra_pred_dict: 预测光谱字典
        show: 是否显示图像
    '''
    # 解码光谱
    gt_spectra = de_tokenization(spectra_gt_dict)
    pred_spectra = de_tokenization(spectra_pred_dict)

    # 创建子图：一行两列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 处理每个样本
    for i in range(len(gt_spectra)):
        # HNMR 光谱
        if '[hnmr]' in spectra_pred_dict:
            try:
                draw_hnmr_spectrum(gt_spectra[i]['[hnmr]'], show=False,ax=ax1)
                ax1.set_title("Ground Truth HNMR")
            except: pass
            try:
                draw_hnmr_spectrum(pred_spectra[i]['[hnmr]'], show=False,ax=ax2)
                ax2.set_title("Predicted HNMR")
            except:
                pass


        # CNMR 光谱
        elif '[cnmr]' in spectra_pred_dict:
            try:
                draw_cnmr_spectrum_1(gt_spectra[i]['[cnmr]'], ax=ax1, show=False)
                ax1.set_title("Ground Truth CNMR")
            except:
                ax1.text(0.5, 0.5, "CNMR GT Plot Failed", ha='center', va='center')
            try:
                draw_cnmr_spectrum_1(pred_spectra[i]['[cnmr]'], ax=ax2, show=False)
                ax2.set_title("Predicted CNMR")
            except:
                ax2.text(0.5, 0.5, "CNMR Pred Plot Failed", ha='center', va='center')

    # 调整布局并显示
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig  # 返回图像对象以便后续保存或处理

