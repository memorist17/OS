# 新しいビューのクイックスタート

実装した2つの新しいビューを実際のデータで確認する手順です。

## 🎯 統合ダッシュボード（推奨）

**2つのビューを1つのURLで見たい場合は、統合ダッシュボードを使用してください！**

```bash
cd /home/kotaronoubuntu/dev/OS/251223/OS
python scripts/run_unified_dashboard.py
```

ブラウザで `http://127.0.0.1:8050` にアクセスすると、タブで2つのビューを切り替えられます：
- **📊 Indicator & Image**: 指標と画像対応ビュー
- **🔗 Cluster & Image**: クラスターと画像対応ビュー

---

## 個別のビュー（個別に起動する場合）

## 前提条件

- `outputs/`ディレクトリに少なくとも1つの`run_*`ディレクトリが存在すること
- 各runディレクトリに解析結果（`mfa_spectrum.csv`、`lacunarity.csv`、`percolation.csv`など）が存在すること

## 1. 指標と画像対応ビューの起動

### 基本的な起動

```bash
cd /home/kotaronoubuntu/dev/OS/251223/OS
python scripts/run_indicator_image_view.py
```

デフォルトでは：
- ポート: 8051
- 指標タイプ: MFA
- 指標値: D(0)

### オプション指定

```bash
# ラクナリティ指標を表示
python scripts/run_indicator_image_view.py --indicator-type lacunarity --indicator-value beta

# パーコレーション指標を表示
python scripts/run_indicator_image_view.py --indicator-type percolation --indicator-value d_critical

# ポートを変更
python scripts/run_indicator_image_view.py --port 8051

# デバッグモード（コード変更時に自動リロード）
python scripts/run_indicator_image_view.py --debug
```

### ブラウザで確認

起動後、ブラウザで以下のURLにアクセス：
```
http://127.0.0.1:8051
```

### 機能の使い方

1. **チャートの横スクロール**: チャートエリアを横にスクロールして複数地点のデータを確認
2. **画像クリック**: 下部の画像ギャラリーの画像をクリックすると、対応するチャートの線がハイライトされます
3. **値のポップアップ**: 画像をクリックすると、その地点の詳細な指標値がポップアップで表示されます

## 2. クラスターと画像対応ビューの起動

### 注意

クラスターと画像対応ビューは、**複数のrunディレクトリが必要**です（最低3つ以上推奨）。
現在1つしかない場合は、まず複数の解析結果を生成してください。

### 基本的な起動

```bash
cd /home/kotaronoubuntu/dev/OS/251223/OS
python scripts/run_cluster_image_view.py
```

デフォルトでは：
- ポート: 8052
- クラスタリングを自動実行

### ブラウザで確認

起動後、ブラウザで以下のURLにアクセス：
```
http://127.0.0.1:8052
```

### 機能の使い方

1. **ズーム**: マウスホイールでズームイン/アウト
2. **表示切り替え**: ズームレベルが高い（0.5以上）と、点から画像表示に自動切り替え
3. **ポイントクリック**: ポイントをクリックすると、その地点の詳細情報が表示されます
4. **代表画像**: 各クラスターの中心に代表画像が常に表示されます

## トラブルシューティング

### エラー: "No run directories found"

`outputs/`ディレクトリに`run_`で始まるディレクトリが存在するか確認してください。

```bash
ls outputs/run_*
```

### エラー: "Could not load lacunarity_fit.yaml"

これは警告で、NumPy型が含まれているYAMLファイルの読み込みに関するものです。
ダッシュボードは正常に動作しますが、一部の機能が制限される可能性があります。

### 画像が表示されない

各runディレクトリに対応する`data/`ディレクトリに画像ファイルがあるか確認してください。
`thumbnail.png`が存在する場合は優先的に使用されます。

## 既存のダッシュボードとの違い

- **既存のダッシュボード** (`scripts/run_dashboard.py`): 単一地点の詳細な解析結果を表示
- **指標と画像対応ビュー**: 複数地点の指標を比較し、画像と対応付け
- **クラスターと画像対応ビュー**: クラスター分析結果を2次元で可視化

## 次のステップ

1. 複数の地点で解析を実行して、比較機能を試す
2. クラスタリング設定を`configs/default.yaml`で調整
3. 画像の生成と配置を確認（`thumbnail.png`など）

