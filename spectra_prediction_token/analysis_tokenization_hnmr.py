import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from load_models import load_HNMR_GPT
# 可选：使用 UMAP（效果通常比 t-SNE 更好）
# 如果没有安装 umap-learn，先 pip install umap-learn
try:
    import umap

    UMAP_AVAILABLE = True
    print("UMAP available，将优先使用 UMAP 降维")
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP 未安装，将使用 t-SNE 降维")

# ==================== 请在这里修改你的设置 ====================


config_path = 'cnmr_config_RealWorld.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
model,_ = load_HNMR_GPT(config=config,relative_dir='')


centroid_embedding = model.transformer.centroid_embedding.weight.detach().cpu().numpy()
vocab_size, emb_dim = centroid_embedding.shape
print(f"Intensity 词汇表大小: {vocab_size}, 嵌入维度: {emb_dim}")

# L2 归一化，便于余弦相似度计算
embeddings_norm = normalize(centroid_embedding, axis=1)

# 计算余弦相似度矩阵
print("正在计算余弦相似度矩阵...")
sim_matrix = cosine_similarity(embeddings_norm)

# 1. 相似度热力图
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='viridis', center=0, square=True)
plt.title('Intensity Embeddings 余弦相似度热力图')
plt.xlabel('Intensity Index')
plt.ylabel('Intensity Index')
plt.show()

# 2. 相邻强度级别的平均相似度（判断是否有序聚类）
adjacent_sims = [sim_matrix[i, i + 1] for i in range(vocab_size - 1)]
print(f"\n相邻强度级别平均余弦相似度: {np.mean(adjacent_sims):.4f} (±{np.std(adjacent_sims):.4f})")
print(f"整体非对角线平均相似度: {np.mean(sim_matrix[np.triu_indices(vocab_size, k=1)]):.4f}")

# 3. 降维可视化
print("正在进行降维可视化...")
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(embeddings_norm)
    method = "UMAP"
else:
    reducer = TSNE(n_components=2, metric='cosine', perplexity=min(30, vocab_size // 4), random_state=42)
    emb_2d = reducer.fit_transform(embeddings_norm)
    method = "t-SNE"

plt.figure(figsize=(12, 10))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(vocab_size), cmap='Spectral', s=80, alpha=0.9)
plt.colorbar(scatter, label='Intensity Index (0 ~ {})'.format(vocab_size - 1))
plt.title(f'{method} 可视化 - Intensity Embeddings\n'
          f'相邻平均相似度: {np.mean(adjacent_sims):.3f}')

# 标注几个关键点
for i in [0, vocab_size // 4, vocab_size // 2, 3 * vocab_size // 4, vocab_size - 1]:
    plt.annotate(str(i), (emb_2d[i, 0], emb_2d[i, 1]),
                 fontsize=12, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))

plt.grid(True, alpha=0.3)
plt.savefig('random_hnmr_tokenization.svg', dpi=300)
plt.show()

print("\n分析完成！")
print("解读提示：")
print("- 如果散点图颜色从左到右平滑渐变（低强度 → 高强度），且相邻相似度 > 0.6 → 强有序聚类")
print("- 如果形成明显几团（低/中/高各一团）→ 分段聚类")
print("- 如果点分布随机、颜色杂乱、相似度低 → 没有明显聚类，每个强度独立表示")