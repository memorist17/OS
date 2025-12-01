#!/bin/bash
# GitHub Issues 一括作成スクリプト
# 
# 使用方法:
#   1. GitHub CLI (gh) をインストール: https://cli.github.com/
#   2. GitHubにログイン: gh auth login
#   3. このスクリプトを実行: bash scripts/create_issues.sh
#
# 注意: 既存のイシューと重複しないよう、事前に確認してください。

set -e

REPO="memorist17/OS"
BRANCH="feat/full-implementation"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}GitHub Issues 作成スクリプト${NC}"
echo "Repository: ${REPO}"
echo "Branch: ${BRANCH}"
echo ""

# GitHub CLIの確認
if ! command -v gh &> /dev/null; then
    echo -e "${RED}エラー: GitHub CLI (gh) がインストールされていません${NC}"
    echo "インストール方法: https://cli.github.com/"
    exit 1
fi

# 認証確認
if ! gh auth status &> /dev/null; then
    echo -e "${RED}エラー: GitHubにログインしていません${NC}"
    echo "ログイン方法: gh auth login"
    exit 1
fi

echo -e "${YELLOW}以下のイシューを作成します:${NC}"
echo ""

# Issue 1: 3指標のクラスタリングするための値の圧縮方法
echo "Issue 1: 3指標のクラスタリングするための値の圧縮方法"
gh issue create \
  --repo "${REPO}" \
  --title "3指標のクラスタリングするための値の圧縮方法" \
  --body "## 説明
MFA、Lacunarity、Percolationの3指標をクラスタリング分析するために、各指標の値を適切に圧縮・正規化する方法を検討・実装する必要がある。

## 課題
- 3指標のスケールが異なる（MFA: スペクトラム、Lacunarity: スカラー値、Percolation: 連結成分数など）
- 次元数が異なる（MFA: 多次元、Lacunarity: 1次元、Percolation: 複数メトリクス）
- クラスタリング前の前処理方法が未定義

## 検討事項
- PCA/UMAP/t-SNEなどの次元削減手法の適用
- 指標ごとの正規化方法（Min-Max、Z-score、Robust scaling）
- クラスタリングアルゴリズムの選択（K-means、DBSCAN、Hierarchical）

## 優先度
High

## ラベル
enhancement, analysis, critical" \
  --label "enhancement,analysis,critical" || echo "  ⚠️  作成失敗（既に存在する可能性があります）"

# Issue 2: 取得地点サンプルの選び方
echo ""
echo "Issue 2: 取得地点サンプルの選び方"
gh issue create \
  --repo "${REPO}" \
  --title "取得地点サンプルの選び方" \
  --body "## 説明
都市構造解析のための地点サンプリング戦略を確立する必要がある。ランダムサンプリング、階層的サンプリング、代表性のあるサンプリングなど。

## 課題
- 現在の取得地点の選定基準が不明確
- サンプルサイズの妥当性
- 地理的・社会的代表性の確保

## 検討事項
- グリッドベースのサンプリング
- 人口密度に基づく重み付きサンプリング
- 土地利用タイプ別の層化サンプリング
- 統計的有意性の確保

## 優先度
High

## ラベル
enhancement, data-acquisition, critical" \
  --label "enhancement,data-acquisition,critical" || echo "  ⚠️  作成失敗"

# Issue 3: 境界・取得範囲
echo ""
echo "Issue 3: 境界・取得範囲"
gh issue create \
  --repo "${REPO}" \
  --title "境界・取得範囲の最適化" \
  --body "## 説明
データ取得範囲の境界処理を明確にする必要がある。現在の\`half_size_m\`設定が適切か、境界効果の影響をどう扱うか。

## 課題
- 境界でのデータ欠損
- 境界効果による指標のバイアス
- 取得範囲の最適サイズ

## 検討事項
- バッファ領域の追加
- 境界補正アルゴリズム
- 取得範囲の自動最適化

## 優先度
High

## ラベル
enhancement, data-acquisition, critical" \
  --label "enhancement,data-acquisition,critical" || echo "  ⚠️  作成失敗"

# Issue 4: 取得形状
echo ""
echo "Issue 4: 取得形状の最適化"
gh issue create \
  --repo "${REPO}" \
  --title "取得形状の最適化" \
  --body "## 説明
現在は正方形（\`half_size_m\` × \`half_size_m\`）で取得しているが、円形、六角形、行政境界に沿った形状など、取得形状の最適化を検討する。

## 課題
- 正方形以外の形状での投影・ラスタライズ
- 形状による指標への影響
- 形状選択の基準

## 検討事項
- 円形取得（中心からの距離ベース）
- 行政境界に沿った取得
- 形状比較実験

## 優先度
High

## ラベル
enhancement, data-acquisition, critical" \
  --label "enhancement,data-acquisition,critical" || echo "  ⚠️  作成失敗"

# Issue 5: NetworkX → graph-tool
echo ""
echo "Issue 5: NetworkX → graph-toolへの移行"
gh issue create \
  --repo "${REPO}" \
  --title "NetworkX → graph-toolへの移行" \
  --body "## 説明
\`pyproject.toml\`と\`README.md\`の依存関係を\`networkx\`から\`graph-tool\`に変更する。\`graph-tool\`はより高速で、大規模ネットワーク解析に適している。

## 課題
- \`src/preprocessing/network_builder.py\`のNetworkX依存コードの置き換え
- \`src/analysis/percolation.py\`のNetworkX依存コードの置き換え
- \`graph-tool\`のインストール要件（C++依存）

## 作業
- [ ] \`pyproject.toml\`の依存関係更新
- [ ] \`README.md\`の依存関係リスト更新
- [ ] NetworkXコードのgraph-toolへの置き換え
- [ ] テストの更新

## 優先度
Medium

## ラベル
refactor, dependencies" \
  --label "refactor,dependencies" || echo "  ⚠️  作成失敗"

# Issue 6: 道路ネットワークの線形補完
echo ""
echo "Issue 6: 道路ネットワークの線形補完"
gh issue create \
  --repo "${REPO}" \
  --title "道路ネットワークの線形補完" \
  --body "## 説明
道路ネットワークが切れている問題を修正する。道路幅の反映が不十分で、実際には接続されている道路がネットワーク上で切断されている可能性がある。

## 課題
- 道路幅を考慮した接続判定
- 近接ノード間の線形補完
- 接続可能性の判定ロジック

## 検討事項
- 道路幅に基づくバッファ領域での接続判定
- 距離閾値に基づく補完
- 道路タイプ別の接続ルール

## 優先度
Medium

## ラベル
bug, preprocessing" \
  --label "bug,preprocessing" || echo "  ⚠️  作成失敗"

# Issue 7: ラスター画像の品質改善
echo ""
echo "Issue 7: ラスター画像の品質改善"
gh issue create \
  --repo "${REPO}" \
  --title "ラスター画像の品質改善" \
  --body "## 説明
生成されるラスター画像が「ガビガビ」（画質が悪い）している問題。ラスタライズ処理の改善が必要。

## 課題
- アンチエイリアシングの適用
- 解像度とメモリ使用量のバランス
- 補間方法の最適化

## 検討事項
- バイリニア/バイキュービック補間の適用
- スーパーサンプリング
- 出力形式の最適化（PNG vs ベクター）

## 優先度
Medium

## ラベル
bug, preprocessing, visualization" \
  --label "bug,preprocessing,visualization" || echo "  ⚠️  作成失敗"

# Issue 8: 3指標の妥当性検証
echo ""
echo "Issue 8: 3指標のスペクトラム・メッシュの妥当性検証"
gh issue create \
  --repo "${REPO}" \
  --title "3指標のスペクトラム・メッシュの妥当性検証" \
  --body "## 説明
MFA、Lacunarity、Percolationの各指標のスペクトラムやメッシュの妥当性を検証する機能・可視化を追加する。

## 課題
- 指標の理論的妥当性の確認方法
- メッシュサイズの影響評価
- 異常値の検出

## 検討事項
- 既知データセットでの検証
- 統計的検定の実装
- 妥当性スコアの可視化

## 優先度
Medium

## ラベル
enhancement, analysis, documentation" \
  --label "enhancement,analysis,documentation" || echo "  ⚠️  作成失敗"

# Issue 9: ダッシュボードの可視化改善
echo ""
echo "Issue 9: ダッシュボードでの指標と構造の対応関係の可視化改善"
gh issue create \
  --repo "${REPO}" \
  --title "ダッシュボードでの指標と構造の対応関係の可視化改善" \
  --body "## 説明
ダッシュボードで指標値と実際の都市構造（建物配置、道路ネットワーク）の対応関係が分かりにくい。可視化を改善する。

## 課題
- 指標値と空間位置の対応が不明確
- インタラクティブな対応関係の表示
- 指標の空間分布の可視化

## 検討事項
- ヒートマップと指標値の重ね合わせ
- クリック/ホバーでの詳細表示
- アニメーション/スライダーでの時系列表示

## 優先度
Medium

## ラベル
enhancement, visualization" \
  --label "enhancement,visualization" || echo "  ⚠️  作成失敗"

# Issue 10: パーコレーション解析の距離計算ロジック
echo ""
echo "Issue 10: パーコレーション解析の距離計算ロジックの見直し"
gh issue create \
  --repo "${REPO}" \
  --title "パーコレーション解析の距離計算ロジックの見直し" \
  --body "## 説明
ネットワークが全結合していない問題と、パーコレーション解析の距離計算ロジックを再検討する。リンクがつながっていない場合の扱い（例：道沿いでいけない場所）を明確にする。

## 課題
- 現在の距離計算方法の確認
- 非連結ノード間の距離定義
- パーコレーション閾値の妥当性

## 検討事項
- ユークリッド距離 vs ネットワーク距離
- 非連結成分の扱い
- パーコレーション理論との整合性

## 優先度
Medium

## ラベル
bug, analysis, question" \
  --label "bug,analysis,question" || echo "  ⚠️  作成失敗"

# Issue 11: ネットワーク形状と距離計算のスタンス
echo ""
echo "Issue 11: ネットワーク形状と距離計算のスタンス明確化"
gh issue create \
  --repo "${REPO}" \
  --title "ネットワーク形状と距離計算のスタンス明確化" \
  --body "## 説明
現在のネットワーク形状（投影後の形状）が適切か、距離を現実に近づけるべきか、明確なスタンスを取った上で実装を統一する。

## 課題
- 投影による距離の歪み
- 現実距離 vs 投影距離
- 解析目的に応じた距離定義

## 検討事項
- 距離計算方法の選択肢の整理
- 解析目的に応じた設定の切り替え
- 距離補正アルゴリズムの実装

## 優先度
Medium

## ラベル
enhancement, analysis, design" \
  --label "enhancement,analysis,design" || echo "  ⚠️  作成失敗"

# Issue 12: 3次元データの取得と解析
echo ""
echo "Issue 12: 3次元データの取得と解析"
gh issue create \
  --repo "${REPO}" \
  --title "3次元データの取得と解析" \
  --body "## 説明
2階建以上、地下、ビル内など、3次元構造を考慮したデータ取得と解析機能を追加する。

## 課題
- Overture Mapsの3次元データの取得
- 3次元ラスタライズ
- 3次元指標の計算

## 検討事項
- ボクセル表現
- 3次元MFA/Lacunarity/Percolation
- 高さ情報の活用

## 優先度
Low

## ラベル
enhancement, feature, future" \
  --label "enhancement,feature,future" || echo "  ⚠️  作成失敗"

# Issue 13: ミクロスケール解析
echo ""
echo "Issue 13: ミクロスケール解析（建物CAD、ビル構造）"
gh issue create \
  --repo "${REPO}" \
  --title "ミクロスケール解析（建物CAD、ビル構造）" \
  --body "## 説明
建物のCADデータやビル内部構造を考慮したミクロスケール解析機能を追加する。

## 課題
- CADデータの取得・変換
- ビル内部ネットワークの構築
- ミクロスケール指標の定義

## 優先度
Low

## ラベル
enhancement, feature, future" \
  --label "enhancement,feature,future" || echo "  ⚠️  作成失敗"

# Issue 14: マクロスケール解析
echo ""
echo "Issue 14: マクロスケール解析（島、国レベル）"
gh issue create \
  --repo "${REPO}" \
  --title "マクロスケール解析（島、国レベル）" \
  --body "## 説明
島や国レベルでの大規模解析機能を追加する。スケーラビリティと計算効率の最適化が必要。

## 課題
- 大規模データの処理
- 分散処理・並列化
- メモリ効率の最適化

## 優先度
Low

## ラベル
enhancement, feature, future" \
  --label "enhancement,feature,future" || echo "  ⚠️  作成失敗"

echo ""
echo -e "${GREEN}完了しました！${NC}"
echo ""
echo "作成されたイシューを確認:"
echo "  gh issue list --repo ${REPO}"
