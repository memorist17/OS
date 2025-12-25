# Issue 9: ダッシュボードでの指標と構造の対応関係の可視化改善 - 実装仕様書

## 概要

ダッシュボードで指標値と実際の都市構造（建物配置、道路ネットワーク）の対応関係が分かりにくい問題を解決する。以下の2つの主要なビューを実装し、インタラクティブな可視化を提供する：

1. **指標と画像対応ビュー**: チャートと計測地点の画像を対応させ、画像クリックでチャートをハイライトする横スクロール可能なビュー
2. **クラスターと画像対応ビュー**: クラスター分析結果を2次元で表示し、ズームレベルに応じて点と画像を切り替えるインタラクティブなビュー

**優先度**: Medium  
**ラベル**: enhancement, visualization  
**関連ファイル**: `src/visualization/dashboard.py`, `src/visualization/comparison_dashboard.py`, `configs/default.yaml`

## 背景・課題

### 現状の問題
- 指標値と空間位置の対応が不明確
- インタラクティブな対応関係の表示がない
- 指標の空間分布の可視化が不十分
- 建物配置・道路ネットワークと指標値の関係が分かりにくい

## 実装仕様

### 1. 指標と画像対応ビュー

#### 1.1 レイアウト仕様
- **上部**: 横スクロール可能なチャートエリア（1行、正方形画像を下部に固定表示）
- **下部**: 計測地点の画像・ネットワーク画像のギャラリー
- **インタラクション**: 地点画像をタップすると対応するチャートの線や点がハイライト、値のポップアップ表示

#### 1.2 主要関数

```python
def create_indicator_image_correspondence_view(
    results_dirs: list[Path],
    indicator_type: Literal["mfa", "lacunarity", "percolation"],
    indicator_value: str = "D(0)",
) -> dash.Dash:
    """指標と画像対応ビューを作成する。"""
    # 横スクロール可能なチャート + 画像ギャラリー
    # 画像クリック → チャートハイライト + ポップアップ表示

def create_horizontal_scrollable_chart(
    all_results: list[dict[str, Any]],
    indicator_type: str,
    indicator_value: str,
) -> go.Figure:
    """横スクロール可能なチャートを作成（各地点のデータをプロット、下部にサムネイル画像）"""

def create_image_gallery(
    all_results: list[dict[str, Any]],
    results_dirs: list[Path],
) -> list[html.Div]:
    """計測地点の画像ギャラリーを作成（建物、道路、ネットワーク画像）"""

def update_chart_highlight(
    fig: go.Figure,
    point_idx: int,
    all_results: list[dict[str, Any]],
) -> go.Figure:
    """チャートの特定の線/点をハイライト（他は半透明化）"""
```

### 2. クラスターと画像対応ビュー

#### 2.1 レイアウト仕様
- **2次元クラスター表示**: クラスター結果を2次元空間にプロット
- **動的表示切り替え**: ズームレベルに応じて点から画像に切り替え
- **代表画像表示**: 最低一つは代表的な画像が常に描画される

#### 2.2 主要関数

```python
def create_cluster_image_correspondence_view(
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
) -> dash.Dash:
    """クラスターと画像対応ビューを作成する。"""
    # 2次元クラスター図 + ズームレベルに応じた表示切り替え

def create_interactive_cluster_figure(
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
) -> go.Figure:
    """インタラクティブなクラスター図を作成（各クラスターに代表画像を配置）"""

def update_cluster_figure_display(
    fig: go.Figure,
    cluster_results: dict[str, Any],
    results_dirs: list[Path],
    display_mode: Literal["points", "images"],
    zoom_level: float,
) -> go.Figure:
    """ズームレベルに応じて点/画像表示を切り替え"""

def _calculate_zoom_level(relayout_data: dict) -> float:
    """PlotlyのrelayoutDataからズームレベルを計算（0.0-1.0）"""
```

### 3. 既存ダッシュボード拡張

#### 3.1 ヒートマップ重ね合わせ機能
```python
def create_spatial_indicator_figure(
    results: dict[str, Any],
    indicator_type: Literal["mfa", "lacunarity", "percolation"],
    indicator_value: str = "D(0)",
) -> go.Figure:
    """指標値と空間構造を重ね合わせたヒートマップ図を作成"""
```

#### 3.2 インタラクティブな詳細表示機能
```python
def create_interactive_structure_figure(
    results: dict[str, Any],
    indicator_type: Literal["mfa", "lacunarity", "percolation"],
) -> go.Figure:
    """インタラクティブな構造と指標の対応関係図（クリック/ホバー対応）"""
```

### 4. 設定ファイル拡張: `configs/default.yaml`

```yaml
visualization:
  # 空間的可視化設定
  spatial:
    enabled: true
    show_buildings: true
    show_roads: true
    show_network: true
    heatmap_opacity: 0.7
    marker_size_range: [5, 20]
  
  # 指標と画像対応ビュー設定
  indicator_image_view:
    enabled: true
    chart_height: 400
    chart_min_width: 1200
    image_size_per_point: 200
    gallery_columns: "auto"
    thumbnail_size: 200
    highlight_color: "rgba(255, 0, 0, 1.0)"
  
  # クラスターと画像対応ビュー設定
  cluster_image_view:
    enabled: true
    plot_height: 800
    point_size: 8
    representative_image_size: 50
    point_image_size: 20
    zoom_threshold: 0.5
    max_images_per_zoom: 50
    cluster_colors:
      - "#1f77b4"
      - "#ff7f0e"
      - "#2ca02c"
      - "#d62728"
      - "#9467bd"
```

## 実装手順

### Phase 1: 基本実装（MVP）
1. `create_spatial_indicator_figure`関数を実装
2. `create_interactive_structure_figure`関数を実装
3. ダッシュボードに新しいタブを追加

### Phase 2: 指標と画像対応ビュー
1. `create_indicator_image_correspondence_view`関数を実装
2. `create_horizontal_scrollable_chart`関数を実装
3. `create_image_gallery`関数を実装
4. 画像クリック → チャートハイライトのコールバックを実装
5. 値のポップアップ表示機能を実装

### Phase 3: クラスターと画像対応ビュー
1. `create_cluster_image_correspondence_view`関数を実装
2. `create_interactive_cluster_figure`関数を実装
3. ズームレベル計算機能を実装
4. ズームレベルに応じた表示切り替え機能を実装
5. 代表画像の常時表示機能を実装

### Phase 4: テストと最適化
1. ユニットテストを追加
2. パフォーマンス最適化（画像の遅延読み込み、表示範囲外の非表示化など）

## テスト仕様

### 主要テストケース

```python
# 指標と画像対応ビュー
def test_indicator_image_correspondence_view()
def test_horizontal_scrollable_chart()
def test_chart_highlight()

# クラスターと画像対応ビュー
def test_cluster_image_correspondence_view()
def test_interactive_cluster_figure()
def test_zoom_level_calculation()
def test_cluster_display_update()

# 既存機能
def test_spatial_indicator_figure()
def test_interactive_structure_figure()
```

## 後方互換性

- 既存のダッシュボード機能は維持
- 新しいタブは追加のみ（既存タブは変更しない）
- 設定ファイルに`visualization.spatial`が存在しない場合は、デフォルト設定を使用

## 注意事項

- **パフォーマンス**: 大規模なラスター画像や大量の地点がある場合、メモリ使用量とレンダリング時間に注意
- **画像処理**: Base64エンコードはメモリ使用量に注意、必要に応じて画像リサイズを実装
- **ズーム切り替え**: ズームレベルに応じた表示切り替えは、パフォーマンスを考慮して実装
- **画像表示制限**: 大量のポイントがある場合、画像表示モードでは表示数を制限

## 参考実装

- `src/visualization/dashboard.py`: 既存のダッシュボード実装
- `src/visualization/comparison_dashboard.py`: 比較ダッシュボード実装
- `src/analysis/multifractal.py`: MFAメッシュの構造
- `src/analysis/lacunarity.py`: ラクナリティデータの構造
- `src/analysis/percolation.py`: パーコレーションデータの構造
- `src/analysis/clustering.py`: クラスター分析の実装（存在する場合）

## 技術的詳細

### 画像のBase64エンコード
- 画像をBase64エンコードしてDashアプリに埋め込む
- メモリ効率を考慮し、必要に応じて画像リサイズを実装
- キャッシュ機能を検討（同じ画像の再エンコードを避ける）

### 横スクロール実装
- Plotlyの`width`パラメータを動的に設定
- CSSの`overflow-x: auto`を使用してスクロール可能にする
- モバイル対応を考慮（タッチスクロール対応）

### ズームレベル計算
- Plotlyの`relayoutData`から軸の範囲を取得
- デフォルト範囲との比較でズームレベルを計算
- 閾値（`zoom_threshold`）に基づいて表示モードを切り替え

### パフォーマンス最適化
- 大量の画像表示時は、表示範囲外の画像を非表示にする
- ズームレベルに応じた画像数の制限（`max_images_per_zoom`）
- 画像の遅延読み込み（Lazy Loading）を検討
