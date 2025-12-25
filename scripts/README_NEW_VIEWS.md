# 新しいビューの起動方法

実装した2つの新しいビューを実際のデータで確認する方法です。

## 1. 指標と画像対応ビュー

複数の解析結果を横スクロール可能なチャートで表示し、画像ギャラリーと連動させます。

### 起動方法

```bash
# 基本的な起動（MFA指標、D(0)を表示）
python scripts/run_indicator_image_view.py

# ラクナリティ指標を表示
python scripts/run_indicator_image_view.py --indicator-type lacunarity --indicator-value beta

# パーコレーション指標を表示
python scripts/run_indicator_image_view.py --indicator-type percolation --indicator-value d_critical

# ポートを変更
python scripts/run_indicator_image_view.py --port 8051

# デバッグモード
python scripts/run_indicator_image_view.py --debug
```

### 機能

- **横スクロール可能なチャート**: 複数地点の指標データを1行で表示
- **画像ギャラリー**: 各地点の建物配置図、道路ネットワーク図などを表示
- **インタラクティブなハイライト**: 画像をクリックすると対応するチャートの線がハイライト
- **値のポップアップ**: クリックした地点の詳細な指標値を表示

## 2. クラスターと画像対応ビュー

クラスター分析結果を2次元で表示し、ズームレベルに応じて点と画像を切り替えます。

### 起動方法

```bash
# 基本的な起動（クラスタリングを自動実行）
python scripts/run_cluster_image_view.py

# 保存されたクラスタリング結果を使用
python scripts/run_cluster_image_view.py --clustering-config outputs/clustering_results.json

# ポートを変更
python scripts/run_cluster_image_view.py --port 8052
```

### 機能

- **2次元クラスター表示**: クラスター結果を2次元空間にプロット
- **動的表示切り替え**: ズームレベルに応じて点から画像に自動切り替え
- **代表画像表示**: 各クラスターの中心に代表画像を常時表示
- **ポイント詳細表示**: ポイントをクリックすると詳細情報を表示

## 注意事項

### データ要件

- **指標と画像対応ビュー**: 
  - 複数のrunディレクトリが必要（最低2つ以上推奨）
  - 各runディレクトリに解析結果（mfa_spectrum.csv、lacunarity.csv、percolation.csvなど）が必要

- **クラスターと画像対応ビュー**:
  - 複数のrunディレクトリが必要（クラスタリングのため）
  - 各runディレクトリに解析結果が必要
  - クラスタリング設定は`configs/default.yaml`の`analysis.clustering`セクションで設定

### トラブルシューティング

1. **「No run directories found」エラー**:
   - `outputs/`ディレクトリに`run_`で始まるディレクトリが存在するか確認
   - `--outputs-dir`オプションで正しいパスを指定

2. **画像が表示されない**:
   - 各runディレクトリに対応する`data/`ディレクトリに画像ファイルがあるか確認
   - `thumbnail.png`が存在する場合は優先的に使用されます

3. **クラスタリングエラー**:
   - 十分な数のrunディレクトリがあるか確認（最低3つ以上推奨）
   - `configs/default.yaml`のクラスタリング設定を確認

## 設定ファイル

設定は`configs/default.yaml`の`visualization`セクションで変更できます：

```yaml
visualization:
  indicator_image_view:
    chart_height: 400
    chart_min_width: 1200
    highlight_color: "rgba(255, 0, 0, 1.0)"
  
  cluster_image_view:
    plot_height: 800
    point_size: 8
    zoom_threshold: 0.5
    max_images_per_zoom: 50
```

