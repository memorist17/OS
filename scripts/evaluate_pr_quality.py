#!/usr/bin/env python3
"""
PRの品質を評価し、スコアを計算するスクリプト

使用方法:
    python scripts/evaluate_pr_quality.py \
        --target-branch main \
        --min-score 90 \
        --min-coverage 80 \
        --branch feature-branch \
        --coverage-report coverage.json \
        --ruff-report ruff-report.json \
        --pr-number 123
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


def load_coverage_report(path: Path) -> float:
    """テストカバレッジを読み込む"""
    try:
        if not path.exists():
            return 0.0
        with open(path) as f:
            data = json.load(f)
            return data.get('totals', {}).get('percent_covered', 0.0)
    except Exception as e:
        print(f"Warning: Failed to load coverage report: {e}")
        return 0.0


def load_ruff_report(path: Path) -> Dict:
    """Ruffの品質レポートを読み込む"""
    try:
        if not path.exists():
            return {'total_errors': 0, 'score': 100}
        
        with open(path) as f:
            reports = json.load(f)
            if isinstance(reports, list):
                total_errors = sum(len(r.get('violations', [])) for r in reports)
            else:
                total_errors = len(reports.get('violations', []))
            
            # エラー1つで2点減点、最大100点
            score = max(0, 100 - total_errors * 2)
            return {
                'total_errors': total_errors,
                'score': score
            }
    except Exception as e:
        print(f"Warning: Failed to load ruff report: {e}")
        return {'total_errors': 0, 'score': 100}


def check_conflict_risk(branch: str, base: str = 'main') -> float:
    """コンフリクトリスクを評価"""
    try:
        # 変更されたファイルを取得
        result = subprocess.run(
            ['git', 'diff', '--name-only', f'{base}...{branch}'],
            capture_output=True,
            text=True
        )
        changed_files = [f for f in result.stdout.strip().split('\n') if f.strip()]
        
        if not changed_files:
            return 0.0
        
        # configs/の変更は高リスク
        config_changes = sum(1 for f in changed_files if 'configs/' in f)
        # src/のコアモジュール変更もリスク
        core_changes = sum(1 for f in changed_files 
                          if any(m in f for m in ['analysis/', 'preprocessing/']))
        
        # リスク計算（0.0-1.0）
        risk = min(1.0, (config_changes * 0.3 + core_changes * 0.2))
        
        return risk
    except Exception as e:
        print(f"Warning: Failed to check conflict risk: {e}")
        return 0.5  # デフォルトは中リスク


def calculate_complexity(branch: str, base: str = 'main') -> float:
    """コード複雑度を計算（変更行数ベース）"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--stat', f'{base}...{branch}'],
            capture_output=True,
            text=True
        )
        
        # 統計から変更行数を抽出
        total_changes = 0
        for line in result.stdout.split('\n'):
            if '|' in line and '+++' not in line and '---' not in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    change_part = parts[1].strip()
                    # "+123 -45" のような形式から数値を抽出
                    if '+' in change_part and '-' in change_part:
                        add_remove = change_part.split()
                        for part in add_remove:
                            if part.startswith('+'):
                                total_changes += int(part[1:])
                            elif part.startswith('-'):
                                total_changes += int(part[1:])
        
        # 1000行 = 1.0の複雑度
        return total_changes / 1000.0 if total_changes > 0 else 0.0
    except Exception as e:
        print(f"Warning: Failed to calculate complexity: {e}")
        return 1.0


def check_experiments(branch: str, outputs_dir: Path) -> bool:
    """実験結果が存在するか確認"""
    try:
        # このブランチに関連するrunディレクトリを探す
        # （簡易実装: outputs/内にrun_*が存在するか確認）
        run_dirs = [d for d in outputs_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("run_")]
        return len(run_dirs) > 0
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="PRの品質を評価")
    parser.add_argument('--target-branch', required=True, help='ターゲットブランチ')
    parser.add_argument('--min-score', type=float, default=60, help='最小スコア')
    parser.add_argument('--min-coverage', type=float, default=40, help='最小カバレッジ')
    parser.add_argument('--branch', required=True, help='評価するブランチ')
    parser.add_argument('--coverage-report', type=Path, help='カバレッジレポート')
    parser.add_argument('--ruff-report', type=Path, help='Ruffレポート')
    parser.add_argument('--pr-number', type=int, help='PR番号')
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'),
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # メトリクス収集
    coverage = 0.0
    if args.coverage_report:
        coverage = load_coverage_report(args.coverage_report)
    
    quality = {'score': 100, 'total_errors': 0}
    if args.ruff_report:
        quality = load_ruff_report(args.ruff_report)
    
    conflict_risk = check_conflict_risk(args.branch, args.target_branch)
    complexity = calculate_complexity(args.branch, args.target_branch)
    has_experiments = check_experiments(args.branch, args.outputs_dir)
    
    # 総合スコア計算（重み付き）
    total_score = (
        coverage * 0.3 +                    # テストカバレッジ 30%
        quality['score'] * 0.3 +             # コード品質 30%
        (1 - conflict_risk) * 100 * 0.2 +    # コンフリクトリスク 20%
        max(0, 100 - complexity * 10) * 0.2  # 複雑度 20%
    )
    
    # GitHub Actions用の出力（GITHUB_OUTPUT形式）
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"total_score={total_score:.1f}\n")
            f.write(f"test_coverage={coverage:.1f}\n")
            f.write(f"code_quality={quality['score']:.1f}\n")
            f.write(f"conflict_risk={conflict_risk:.2f}\n")
            f.write(f"complexity={complexity:.2f}\n")
            f.write(f"has_experiments={str(has_experiments).lower()}\n")
    
    # コンソール出力
    print(f"\n{'='*60}")
    print(f"PR品質評価結果")
    print(f"{'='*60}")
    print(f"ターゲットブランチ: {args.target_branch}")
    print(f"評価ブランチ: {args.branch}")
    print(f"\nメトリクス:")
    print(f"  総合スコア: {total_score:.1f}/{args.min_score}")
    print(f"  テストカバレッジ: {coverage:.1f}%/{args.min_coverage}%")
    print(f"  コード品質: {quality['score']:.1f}/100")
    print(f"  コンフリクトリスク: {conflict_risk:.2f}")
    print(f"  複雑度: {complexity:.2f}")
    print(f"  実験結果: {'あり' if has_experiments else 'なし'}")
    print(f"{'='*60}\n")
    
    # 判定
    passed = total_score >= args.min_score and coverage >= args.min_coverage
    
    if passed:
        print(f"✅ 品質基準を満たしています")
        return 0
    else:
        print(f"❌ 品質基準を満たしていません")
        if total_score < args.min_score:
            print(f"   スコア不足: {total_score:.1f} < {args.min_score}")
        if coverage < args.min_coverage:
            print(f"   カバレッジ不足: {coverage:.1f}% < {args.min_coverage}%")
        return 1


if __name__ == '__main__':
    exit(main())
