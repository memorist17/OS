#!/usr/bin/env python3
"""
実験結果をブランチごとに管理し、比較を支援するスクリプト

使用方法:
    # 実験結果を登録
    python scripts/manage_experiments.py register \
        --branch dev/feature-issue-12 \
        --run-id run_20251203_165429_e51ed958 \
        --config configs/default.yaml

    # 実験結果を比較
    python scripts/manage_experiments.py compare \
        --branch1 dev/feature-issue-12 \
        --branch2 dev/feature-issue-11

    # 実験結果を一覧表示
    python scripts/manage_experiments.py list --branch stage
"""
import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd


def hash_config(config_path: Path) -> str:
    """設定ファイルのハッシュを計算（再現性確認用）"""
    content = config_path.read_text()
    # コメント行を除去してハッシュ計算
    lines = [l for l in content.split('\n') 
             if not l.strip().startswith('#') and l.strip()]
    clean_content = '\n'.join(lines)
    return hashlib.sha256(clean_content.encode()).hexdigest()[:8]


def extract_metrics(run_dir: Path) -> Dict:
    """実験結果から主要メトリクスを抽出"""
    metrics = {}
    
    # MFA metrics
    mfa_spectrum = run_dir / "mfa_spectrum.csv"
    if mfa_spectrum.exists():
        try:
            df = pd.read_csv(mfa_spectrum)
            if 'alpha' in df.columns and 'f_alpha' in df.columns:
                metrics['mfa_alpha_range'] = float(df['alpha'].max() - df['alpha'].min())
                metrics['mfa_f_alpha_max'] = float(df['f_alpha'].max())
                metrics['mfa_alpha_mean'] = float(df['alpha'].mean())
        except Exception as e:
            print(f"Warning: Failed to read MFA metrics: {e}")
    
    # Lacunarity metrics
    lacunarity = run_dir / "lacunarity.csv"
    if lacunarity.exists():
        try:
            df = pd.read_csv(lacunarity)
            if 'lambda' in df.columns:
                metrics['lacunarity_mean'] = float(df['lambda'].mean())
                metrics['lacunarity_max'] = float(df['lambda'].max())
                metrics['lacunarity_min'] = float(df['lambda'].min())
        except Exception as e:
            print(f"Warning: Failed to read Lacunarity metrics: {e}")
    
    # Percolation metrics
    percolation_stats = run_dir / "percolation_stats.yaml"
    if percolation_stats.exists():
        try:
            stats = yaml.safe_load(percolation_stats.read_text())
            metrics['percolation_d_critical_50'] = stats.get('d_critical_50', 0)
            metrics['percolation_max_giant_fraction'] = stats.get('max_giant_fraction', 0)
        except Exception as e:
            print(f"Warning: Failed to read Percolation metrics: {e}")
    
    return metrics


def load_registry(registry_file: Path) -> Dict:
    """実験レジストリを読み込み"""
    if registry_file.exists():
        return json.loads(registry_file.read_text())
    return {}


def save_registry(registry: Dict, registry_file: Path):
    """実験レジストリを保存"""
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def register_experiment(
    branch: str,
    run_id: str,
    config_path: Path,
    outputs_dir: Path,
    registry_file: Path
):
    """実験結果を登録"""
    run_dir = outputs_dir / run_id
    
    if not run_dir.exists():
        print(f"❌ エラー: runディレクトリが見つかりません: {run_dir}")
        return 1
    
    # config_snapshot.yamlを確認
    config_snapshot = run_dir / "config_snapshot.yaml"
    if not config_snapshot.exists():
        print(f"⚠️  警告: {run_id} に config_snapshot.yaml がありません")
        print("   再現性が保証されない可能性があります")
    
    # 設定ファイルのハッシュを計算
    if config_path.exists():
        config_hash = hash_config(config_path)
    elif config_snapshot.exists():
        config_hash = hash_config(config_snapshot)
        print(f"⚠️  設定ファイルが見つからないため、config_snapshot.yamlを使用します")
    else:
        config_hash = "unknown"
        print(f"⚠️  警告: 設定ファイルが見つかりません")
    
    # メトリクスを抽出
    metrics = extract_metrics(run_dir)
    
    # レジストリに追加
    registry = load_registry(registry_file)
    if branch not in registry:
        registry[branch] = []
    
    experiment = {
        'run_id': run_id,
        'config_hash': config_hash,
        'metrics': metrics,
        'output_dir': str(run_dir.relative_to(outputs_dir.parent))
    }
    
    # 重複チェック
    existing = [e for e in registry[branch] if e['run_id'] == run_id]
    if existing:
        print(f"⚠️  警告: {run_id} は既に登録されています。上書きします。")
        registry[branch] = [e for e in registry[branch] if e['run_id'] != run_id]
    
    registry[branch].append(experiment)
    save_registry(registry, registry_file)
    
    print(f"✅ 実験結果を登録しました:")
    print(f"   ブランチ: {branch}")
    print(f"   Run ID: {run_id}")
    print(f"   設定ハッシュ: {config_hash}")
    print(f"   メトリクス数: {len(metrics)}")
    
    return 0


def compare_experiments(
    branch1: str,
    branch2: str,
    registry_file: Path
):
    """2つのブランチの実験結果を比較"""
    registry = load_registry(registry_file)
    
    exp1 = registry.get(branch1, [])
    exp2 = registry.get(branch2, [])
    
    if not exp1:
        print(f"❌ エラー: ブランチ '{branch1}' に実験結果が登録されていません")
        return 1
    
    if not exp2:
        print(f"❌ エラー: ブランチ '{branch2}' に実験結果が登録されていません")
        return 1
    
    print(f"\n{'='*60}")
    print(f"実験結果比較: {branch1} vs {branch2}")
    print(f"{'='*60}\n")
    
    print(f"ブランチ1 ({branch1}): {len(exp1)}個の実験")
    print(f"ブランチ2 ({branch2}): {len(exp2)}個の実験\n")
    
    # 最新の実験結果を比較
    latest1 = exp1[-1]
    latest2 = exp2[-1]
    
    print(f"最新実験の比較:")
    print(f"  Branch1: {latest1['run_id']}")
    print(f"  Branch2: {latest2['run_id']}\n")
    
    # メトリクスの比較
    metrics1 = latest1.get('metrics', {})
    metrics2 = latest2.get('metrics', {})
    
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    
    if common_metrics:
        print("メトリクス比較:")
        print(f"{'メトリクス':<30} {'Branch1':<15} {'Branch2':<15} {'差分':<15}")
        print("-" * 75)
        
        for metric in sorted(common_metrics):
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            diff = abs(val1 - val2)
            print(f"{metric:<30} {val1:<15.4f} {val2:<15.4f} {diff:<15.4f}")
    else:
        print("⚠️  共通のメトリクスが見つかりません")
    
    # 設定ハッシュの比較
    if latest1['config_hash'] == latest2['config_hash']:
        print("\n✅ 設定ファイルは同一です（再現性保証）")
    else:
        print(f"\n⚠️  設定ファイルが異なります:")
        print(f"  Branch1: {latest1['config_hash']}")
        print(f"  Branch2: {latest2['config_hash']}")
    
    return 0


def list_experiments(
    branch: Optional[str],
    registry_file: Path
):
    """実験結果を一覧表示"""
    registry = load_registry(registry_file)
    
    if branch:
        if branch not in registry:
            print(f"❌ エラー: ブランチ '{branch}' に実験結果が登録されていません")
            return 1
        
        experiments = registry[branch]
        print(f"\nブランチ '{branch}' の実験結果 ({len(experiments)}個):\n")
        
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. Run ID: {exp['run_id']}")
            print(f"   設定ハッシュ: {exp['config_hash']}")
            print(f"   メトリクス: {len(exp.get('metrics', {}))}個")
            if exp.get('metrics'):
                print(f"   主要メトリクス:")
                for key, value in list(exp['metrics'].items())[:3]:
                    print(f"     - {key}: {value:.4f}")
            print()
    else:
        print("\n全ブランチの実験結果:\n")
        for branch_name, experiments in registry.items():
            print(f"  {branch_name}: {len(experiments)}個の実験")
            for exp in experiments:
                print(f"    - {exp['run_id']}")
        print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="実験結果をブランチごとに管理し、比較を支援"
    )
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # register コマンド
    register_parser = subparsers.add_parser('register', help='実験結果を登録')
    register_parser.add_argument('--branch', required=True, help='ブランチ名')
    register_parser.add_argument('--run-id', required=True, help='Run ID')
    register_parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'),
                                help='設定ファイルパス')
    register_parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'),
                                help='出力ディレクトリ')
    register_parser.add_argument('--registry-file', type=Path,
                                default=Path('outputs/experiments_registry.json'),
                                help='レジストリファイルパス')
    
    # compare コマンド
    compare_parser = subparsers.add_parser('compare', help='実験結果を比較')
    compare_parser.add_argument('--branch1', required=True, help='比較するブランチ1')
    compare_parser.add_argument('--branch2', required=True, help='比較するブランチ2')
    compare_parser.add_argument('--registry-file', type=Path,
                               default=Path('outputs/experiments_registry.json'),
                               help='レジストリファイルパス')
    
    # list コマンド
    list_parser = subparsers.add_parser('list', help='実験結果を一覧表示')
    list_parser.add_argument('--branch', help='ブランチ名（省略時は全ブランチ）')
    list_parser.add_argument('--registry-file', type=Path,
                           default=Path('outputs/experiments_registry.json'),
                           help='レジストリファイルパス')
    
    args = parser.parse_args()
    
    if args.command == 'register':
        return register_experiment(
            args.branch,
            args.run_id,
            args.config,
            args.outputs_dir,
            args.registry_file
        )
    elif args.command == 'compare':
        return compare_experiments(
            args.branch1,
            args.branch2,
            args.registry_file
        )
    elif args.command == 'list':
        return list_experiments(
            args.branch,
            args.registry_file
        )
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
