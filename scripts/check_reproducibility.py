#!/usr/bin/env python3
"""
実験結果の再現性を検証するスクリプト

使用方法:
    # 全runディレクトリの再現性を確認
    python scripts/check_reproducibility.py \
        --check-snapshots \
        --outputs-dir outputs/

    # 特定の実験の再現性を検証
    python scripts/check_reproducibility.py \
        --config configs/default.yaml \
        --run-id run_20251203_165429_e51ed958 \
        --outputs-dir outputs/
"""
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional
import yaml


def hash_config(config_path: Path) -> str:
    """設定ファイルのハッシュを計算"""
    content = config_path.read_text()
    # コメント行を除去してハッシュ計算
    lines = [l for l in content.split('\n') 
             if not l.strip().startswith('#') and l.strip()]
    clean_content = '\n'.join(lines)
    return hashlib.sha256(clean_content.encode()).hexdigest()[:8]


def check_config_snapshots(outputs_dir: Path) -> tuple[bool, List[str]]:
    """全run_*ディレクトリにconfig_snapshot.yamlが存在するか確認"""
    run_dirs = [d for d in outputs_dir.iterdir() 
                if d.is_dir() and d.name.startswith("run_")]
    
    missing = []
    for run_dir in run_dirs:
        config_snapshot = run_dir / "config_snapshot.yaml"
        if not config_snapshot.exists():
            missing.append(run_dir.name)
    
    return len(missing) == 0, missing


def verify_reproducibility(
    config_path: Path,
    run_id: str,
    outputs_dir: Path
) -> bool:
    """設定ファイルから実験を再現可能か検証"""
    run_dir = outputs_dir / run_id
    
    if not run_dir.exists():
        print(f"❌ {run_id}: runディレクトリが見つかりません")
        return False
    
    config_snapshot = run_dir / "config_snapshot.yaml"
    
    if not config_snapshot.exists():
        print(f"❌ {run_id}: config_snapshot.yamlが見つかりません")
        return False
    
    if not config_path.exists():
        print(f"⚠️  {run_id}: 現在の設定ファイルが見つかりません")
        print(f"   スナップショットのみ確認します")
        return True
    
    # 設定ファイルのハッシュを比較
    current_hash = hash_config(config_path)
    snapshot_hash = hash_config(config_snapshot)
    
    if current_hash != snapshot_hash:
        print(f"⚠️  {run_id}: 設定ファイルが変更されています")
        print(f"   現在: {current_hash}")
        print(f"   スナップショット: {snapshot_hash}")
        print(f"   → この実験を再現するには、config_snapshot.yamlの設定を使用してください")
        return False
    
    print(f"✅ {run_id}: 再現性が確認されました (ハッシュ: {current_hash})")
    return True


def check_all_reproducibility(
    config_path: Path,
    outputs_dir: Path
) -> tuple[int, int]:
    """全runディレクトリの再現性を確認"""
    run_dirs = [d for d in outputs_dir.iterdir() 
                if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        print("⚠️  runディレクトリが見つかりません")
        return 0, 0
    
    print(f"再現性チェック: {len(run_dirs)}個のrunディレクトリを確認\n")
    
    reproducible = 0
    for run_dir in sorted(run_dirs):
        run_id = run_dir.name
        if verify_reproducibility(config_path, run_id, outputs_dir):
            reproducible += 1
    
    print(f"\n{'='*60}")
    print(f"結果: {reproducible}/{len(run_dirs)}個の実験が再現可能")
    print(f"{'='*60}")
    
    return reproducible, len(run_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="実験結果の再現性を検証"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/default.yaml'),
        help='現在の設定ファイルパス'
    )
    parser.add_argument(
        '--run-id',
        help='特定のrun IDを検証（省略時は全runディレクトリ）'
    )
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path('outputs'),
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--check-snapshots',
        action='store_true',
        help='config_snapshot.yamlの存在のみ確認'
    )
    
    args = parser.parse_args()
    
    if args.check_snapshots:
        # スナップショットの存在確認のみ
        all_exist, missing = check_config_snapshots(args.outputs_dir)
        
        if all_exist:
            run_dirs = [d for d in args.outputs_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("run_")]
            print(f"✅ 全{len(run_dirs)}個のrunディレクトリにconfig_snapshot.yamlが存在します")
            return 0
        else:
            print(f"❌ 以下の{len(missing)}個のrunディレクトリにconfig_snapshot.yamlがありません:")
            for d in missing:
                print(f"  - {d}")
            return 1
    
    if args.run_id:
        # 特定のrun IDを検証
        success = verify_reproducibility(
            args.config,
            args.run_id,
            args.outputs_dir
        )
        return 0 if success else 1
    else:
        # 全runディレクトリを検証
        reproducible, total = check_all_reproducibility(
            args.config,
            args.outputs_dir
        )
        
        if reproducible == total:
            return 0
        elif reproducible > 0:
            print(f"\n⚠️  {total - reproducible}個の実験は設定が変更されています")
            return 0  # 警告のみ
        else:
            return 1


if __name__ == '__main__':
    exit(main())
