# Urban Structure Analysis System

都市構造解析システム - MFA/Lacunarity/Percolation統合解析

## 概要

Overture Mapsから都市データを取得し、多重フラクタル解析（MFA）、ラクナリティ解析、パーコレーション解析を統一的に計算・可視化するシステム。

## 機能

### Phase 1: Data Acquisition
- **Overture Maps連携**: DuckDB + httpfsによるS3直接アクセス
- **動的AEQD投影**: 任意地点を中心とした等距離方位図法
- **ラスタライズ**: 建物（バイナリ）、道路（重み付き）
- **ネットワーク構築**: NetworkXによる空間グラフ生成

### Phase 3: Analysis Engine
- **多重フラクタル解析 (MFA)**: 4D reshape + グリッドシフト平均化
- **ラクナリティ解析**: 積分画像によるO(1)ボックスクエリ
- **パーコレーション解析**: 距離閾値に基づく連結成分解析
- **クラスタリング解析**: 3指標の統合・前処理・クラスタリング（Issue 1対応）

### Phase 4: Visualization
- **Dashダッシュボード**: インタラクティブな結果可視化
- **日本語フォント対応**: Noto Sans JP
- **ダークテーマ**: モダンなUI/UX

## プロジェクト構造

```
OS_251127/
├── configs/
│   └── default.yaml              # 全設定を集約
├── data/
│   └── {site_id}/                # 中間データ（前処理済み）
│       ├── metadata.yaml
│       ├── buildings_binary.npy
│       ├── roads_weighted.npy
│       └── network.graphml
├── src/
│   ├── acquisition/              # Phase 1: データ取得
│   │   └── overture_fetcher.py
│   ├── projection/               # Phase 1: 座標変換
│   │   └── aeqd_transformer.py
│   ├── preprocessing/            # Phase 2: 前処理
│   │   ├── rasterizer.py
│   │   └── network_builder.py
│   ├── analysis/                 # Phase 3: 解析
│   │   ├── multifractal.py
│   │   ├── lacunarity.py
│   │   ├── percolation.py
│   │   ├── feature_extraction.py
│   │   ├── clustering_preprocessing.py
│   │   └── clustering.py
│   └── visualization/            # Phase 4: 可視化
│       └── dashboard.py
├── scripts/
│   ├── run_acquisition.py        # データ取得パイプライン
│   ├── run_analysis.py           # 解析パイプライン
│   ├── run_clustering.py         # クラスタリング解析
│   └── run_dashboard.py          # ダッシュボード起動
├── outputs/                      # 解析結果（run_id単位）
│   └── {run_id}/
│       ├── config_snapshot.yaml
│       ├── mfa_spectrum.csv
│       ├── lacunarity.csv
│       └── percolation.csv
├── pyproject.toml
├── Dockerfile
└── README.md
```

## インストール

```bash
# uv を使用（推奨）
uv venv
source .venv/bin/activate
uv pip install -e .

# または pip
pip install -e .
```

## 使用方法

### 1. データ取得

```bash
# 東京駅周辺のデータを取得
python scripts/run_acquisition.py \
    --lat 35.6812 \
    --lon 139.7671 \
    --site-id tokyo-station

# 出力: data/tokyo-station/
```

### 2. 解析実行

```bash
# 全解析を実行
python scripts/run_analysis.py \
    --data-dir data/tokyo-station

# 出力: outputs/{run_id}/
```

### 3. クラスタリング解析（複数地点の統合解析）

```bash
# 全解析結果からクラスタリングを実行
python scripts/run_clustering.py \
    --outputs-dir outputs

# 特定のrun_idのみを使用
python scripts/run_clustering.py \
    --outputs-dir outputs \
    --run-ids run_20241127_120000_abc12345 run_20241127_130000_def67890

# 出力: outputs/clustering_results/
#   - clustering_results.csv: 特徴量とクラスタラベル
#   - processed_features.npy: 前処理済み特徴量
#   - clustering_metadata.yaml: メタデータ
```

### 3-1. クラスタリングパラメータの最適化（推奨）

```bash
# エルボー法で最適なクラスタ数を自動探索
python scripts/optimize_clustering.py \
    --outputs-dir outputs \
    --config configs/default.yaml

# 出力: outputs/clustering_optimization/
#   - elbow_method_results.csv: エルボー法の結果（各クラスタ数での評価指標）
#   - cluster_number_comparison.csv: 複数クラスタ数での比較
#   - optimized_clustering_results.csv: 最適化後のクラスタリング結果
#   - cluster_interpretation.csv: 各クラスタの特徴量統計
#   - optimization_metadata.yaml: 最適化メタデータ
```

**設定ファイル** (`configs/default.yaml`) で最適化パラメータを調整可能:
```yaml
clustering:
  optimization:
    use_elbow_method: true      # エルボー法を使用
    min_clusters: 2              # 最小クラスタ数
    max_clusters: 10             # 最大クラスタ数
    elbow_metric: "silhouette"   # "inertia", "silhouette", "davies_bouldin"
    compare_cluster_numbers: true
    interpret_results: true      # クラスタ解釈を生成
```

### 3-2. クラスタリング手法の比較検証

```bash
# 複数の前処理・クラスタリング手法を自動比較
python scripts/compare_clustering_methods.py \
    --outputs-dir outputs

# 比較結果を可視化
python scripts/visualize_clustering_comparison.py \
    --comparison-dir outputs/clustering_comparison

# 出力: outputs/clustering_comparison/
#   - comparison_results.csv: 全組み合わせの評価結果
#   - comparison_summary.yaml: 最適設定のサマリー
#   - clustering_comparison_all.html: 統合可視化（1つのHTMLファイル）
```

### 4. ダッシュボード起動

```bash
# 最新の解析結果を可視化
python scripts/run_dashboard.py

# 特定のrun_idを指定
python scripts/run_dashboard.py --run-id run_20241127_120000_abc12345

# ブラウザで http://127.0.0.1:8050 にアクセス
```

## 設定ファイル

`configs/default.yaml`:

```yaml
# キャンバス設定
canvas:
  half_size_m: 1000        # 中心から±1000m
  resolution_m: 1.0        # 1px = 1m

# 解析パラメータ
analysis:
  r_min: 2
  r_max: 512
  r_steps: 20
  
  mfa:
    q_min: -10
    q_max: 10
    q_steps: 41
    grid_shift_count: 16
  
  lacunarity:
    method: "integral_image"
    full_scan: true
  
  percolation:
    d_min: 1
    d_max: 100
    d_steps: 50

# 実行設定
execution:
  n_jobs: -1               # 全CPUコアを使用
  cache_integral: true
  verbose: true

# クラスタリング設定
clustering:
  preprocessing:
    normalization_method: "robust"  # "minmax", "standard", "robust"
    dimensionality_reduction: "pca"  # "pca", "umap", "tsne", null
    n_components: null              # null = 自動選択
    random_state: 42
  method: "kmeans"         # "kmeans", "dbscan", "hierarchical"
  n_clusters: null         # null = 自動選択
  eps: 0.5                 # DBSCAN用
  min_samples: 3          # DBSCAN用
  linkage: "ward"         # Hierarchical用
```

## 出力フォーマット

| ファイル | 内容 | 形状 |
|---------|------|------|
| `mfa_spectrum.csv` | q, α(q), f(α), τ(q), R² | (q_steps, 5) |
| `mfa_dimensions.csv` | q, D(q) | (q_steps, 2) |
| `lacunarity.csv` | r, Λ(r), σ, μ, cv | (r_steps, 5) |
| `percolation.csv` | d, max_cluster_size, n_clusters, giant_fraction | (d_steps, 4) |

## Docker

```bash
# ビルド
docker build -t urban-analysis .

# 実行
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs urban-analysis \
    python scripts/run_analysis.py --data-dir data/tokyo-station
```

## 依存関係

- Python >= 3.11
- duckdb >= 1.1.0
- geopandas >= 0.14.0
- rasterio >= 1.3.0
- networkx >= 3.0
- numpy >= 1.26.0
- pandas >= 2.1.0
- dash >= 2.14.0
- plotly >= 5.18.0
- opencv-python >= 4.8.0
- scipy >= 1.11.0
- joblib >= 1.3.0
- tqdm >= 4.66.0
- scikit-learn >= 1.3.0
- umap-learn >= 0.5.5 (オプション: UMAP次元削減用)

## ライセンス

MIT License

## 作者

Kotaro Iwata - Iwata Global Research & Engineering
