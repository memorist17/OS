# 研究コード向け3段階ブランチワークフロー

## 概要

このプロジェクトは研究コードの特性（実験性、再現性、論文投稿）を考慮した3段階ブランチ戦略を採用しています。

## ブランチ構造

### main ブランチ
**目的**: 論文投稿・外部公開用  
**品質基準**: 
- 完全検証済み
- 再現性保証（`config_snapshot.yaml`必須）
- 全テスト通過
- ドキュメント完備

**更新頻度**: 月1回程度  
**保護設定**:
- Force push禁止
- レビュー必須（最低1名）
- CI/CD全チェック必須
- マージ後はタグ付け（v0.1.0, v0.2.0など）

### stage ブランチ
**目的**: 実験結果の検証・比較  
**品質基準**:
- 分析完了
- 結果の解釈中
- 複数パラメータ実験の並行保持

**更新頻度**: 週1-2回  
**マージ条件**:
- 有意な結果が出た時
- 論文執筆用メソッド確定時
- パラメータ最適化完了時

**特徴**:
- 複数の実験結果を保持可能
- 比較用の`outputs/`を保持

### dev ブランチ
**目的**: 日常のコーディング・試行錯誤  
**品質基準**:
- 動作確認レベル
- エラーなく実行可能

**更新頻度**: 毎日複数回  
**マージ条件**:
- 有意な結果が出た時 → stageへ
- バグ修正完了時 → stageへ
- 新機能追加時 → stageへ

**特徴**:
- 高速イテレーション
- 実験的な実装OK
- 複数の`dev/feature-*`ブランチから構成

## ワークフロー

### 1. 日常開発（devブランチ）

```bash
# 新しい機能・実験用ブランチを作成
git checkout dev
git pull origin dev
git checkout -b dev/feature-issue-12

# 実装・実験
python scripts/run_analysis.py --data-dir data/tokyo-station

# 結果を確認
python scripts/run_dashboard.py

# コミット（実験結果も含む）
git add .
git commit -m "feat(analysis): implement new distance calculation (Issue #12)"
git push origin dev/feature-issue-12
```

### 2. 実験結果の検証（stageブランチ）

```bash
# 有意な結果が出たらstageへマージ
git checkout stage
git pull origin stage
git merge dev/feature-issue-12

# 実験結果を登録
python scripts/manage_experiments.py register \
    --branch stage \
    --run-id run_20251203_165429_e51ed958 \
    --config configs/default.yaml

# 他の実験結果と比較
python scripts/manage_experiments.py compare \
    --branch1 stage \
    --branch2 dev/feature-issue-11

# プッシュ
git push origin stage
```

### 3. 論文用確定（mainブランチ）

```bash
# 論文執筆時、メソッド確定時にmainへマージ
git checkout main
git pull origin main

# 再現性チェック
python scripts/check_reproducibility.py \
    --config configs/default.yaml \
    --outputs-dir outputs/

# マージ（レビュー必須）
git merge stage
git tag -a v0.2.0 -m "Paper submission version"
git push origin main --tags
```

## 実験結果管理

### 実験結果の登録

```bash
# 実験結果をレジストリに登録
python scripts/manage_experiments.py register \
    --branch dev/feature-issue-12 \
    --run-id run_20251203_165429_e51ed958 \
    --config configs/default.yaml
```

### 実験結果の比較

```bash
# 2つのブランチの実験結果を比較
python scripts/manage_experiments.py compare \
    --branch1 dev/feature-issue-12 \
    --branch2 dev/feature-issue-11 \
    --output comparison_report.json
```

### 実験レジストリの確認

```bash
# 全実験結果を一覧表示
python scripts/manage_experiments.py list \
    --branch stage
```

## 再現性チェック

### 設定スナップショットの確認

```bash
# 全runディレクトリにconfig_snapshot.yamlが存在するか確認
python scripts/check_reproducibility.py \
    --outputs-dir outputs/
```

### 特定の実験の再現性検証

```bash
# 設定ファイルから実験を再現可能か検証
python scripts/check_reproducibility.py \
    --config configs/default.yaml \
    --run-id run_20251203_165429_e51ed958 \
    --outputs-dir outputs/
```

## 複数エージェント並列開発

### エージェントブランチの作成

```bash
# Issue #12に対して複数エージェントで並列開発
# GitHub Actionsが自動的にブランチを作成します
# または手動で作成:
git checkout dev
git checkout -b agent/implementer-1/issue-12
git checkout -b agent/implementer-2/issue-12
```

### エージェントブランチの評価

```bash
# 複数のエージェントブランチを比較
python scripts/compare_agent_branches.py \
    --current-branch agent/implementer-1/issue-12 \
    --issue-number 12 \
    --output comparison.json
```

## GitHub Actions自動化

### PR品質評価

PRが作成されると、自動的に以下が評価されます：

- **devブランチ**: スコア60以上で自動マージ可能
- **stageブランチ**: スコア75以上、再現性チェック必須
- **mainブランチ**: スコア90以上、レビュー必須

### ブランチ保護

- `main`: Force push禁止、レビュー必須
- `stage`: 再現性チェック必須
- `dev`: 自動マージ可能（低品質基準）

## ベストプラクティス

1. **実験結果は必ず保存**: `outputs/run_*/`に`config_snapshot.yaml`を含める
2. **短期間のブランチ**: 24時間以内にマージを目標
3. **段階的マージ**: dev → stage → main の順で段階的に品質を上げる
4. **再現性の確保**: main/stageブランチでは必ず再現性チェックを実行
5. **タグ付け**: mainブランチへのマージ時は必ずバージョンタグを付ける

## トラブルシューティング

### 実験結果が見つからない

```bash
# 実験レジストリを確認
cat outputs/experiments_registry.json
```

### 再現性チェックが失敗する

```bash
# 設定ファイルの差分を確認
python scripts/check_reproducibility.py \
    --config configs/default.yaml \
    --run-id run_20251203_165429_e51ed958 \
    --outputs-dir outputs/ \
    --verbose
```

### ブランチ間のコンフリクト

```bash
# 最新のdev/stage/mainを取得
git fetch origin
git checkout dev
git pull origin dev

# コンフリクトを解決してからマージ
git merge origin/stage
```

