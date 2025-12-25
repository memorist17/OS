#!/usr/bin/env python3
"""
分析結果の妥当性を検証するスクリプト

使用方法:
    python scripts/validate_analysis_results.py \
        --outputs-dir outputs/run_20251203_165429_e51ed958
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def validate_mfa_spectrum(mfa_spectrum_path: Path) -> Dict:
    """MFAスペクトラムの妥当性を検証"""
    if not mfa_spectrum_path.exists():
        return {"valid": False, "error": "File not found"}
    
    df = pd.read_csv(mfa_spectrum_path)
    
    issues = []
    warnings = []
    
    # 必須カラムの確認
    required_columns = ["q", "alpha", "f_alpha", "tau"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return {"valid": False, "error": f"Missing columns: {missing}"}
    
    # 理論的制約の確認
    # 1. f(α) <= α (一般化次元の制約)
    if "f_alpha" in df.columns and "alpha" in df.columns:
        invalid = df[df["f_alpha"] > df["alpha"]]
        if len(invalid) > 0:
            issues.append(f"f(α) > α のデータが {len(invalid)} 個あります（理論的に不可能）")
    
    # 2. αの範囲が妥当か（通常 -10 から 10 程度）
    if "alpha" in df.columns:
        alpha_min, alpha_max = df["alpha"].min(), df["alpha"].max()
        if alpha_min < -20 or alpha_max > 20:
            warnings.append(f"αの範囲が異常です: [{alpha_min:.2f}, {alpha_max:.2f}]")
    
    # 3. f(α)が負でないか
    if "f_alpha" in df.columns:
        negative = df[df["f_alpha"] < 0]
        if len(negative) > 0:
            issues.append(f"f(α) < 0 のデータが {len(negative)} 個あります")
    
    # 4. R²値の確認（フィッティング品質）
    if "R2" in df.columns or "r_squared" in df.columns:
        r2_col = "R2" if "R2" in df.columns else "r_squared"
        low_r2 = df[df[r2_col] < 0.9]
        if len(low_r2) > 0:
            warnings.append(f"R² < 0.9 のデータが {len(low_r2)} 個あります（フィッティング品質が低い）")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_points": len(df),
            "alpha_range": [float(df["alpha"].min()), float(df["alpha"].max())],
            "f_alpha_range": [float(df["f_alpha"].min()), float(df["f_alpha"].max())],
        }
    }


def validate_lacunarity(lacunarity_path: Path) -> Dict:
    """Lacunarityの妥当性を検証"""
    if not lacunarity_path.exists():
        return {"valid": False, "error": "File not found"}
    
    df = pd.read_csv(lacunarity_path)
    
    issues = []
    warnings = []
    
    # 必須カラムの確認
    if "lambda" not in df.columns or "r" not in df.columns:
        return {"valid": False, "error": "Missing required columns (lambda, r)"}
    
    # 理論的制約の確認
    # 1. Λ(r) >= 1 (ラクナリティは常に1以上)
    if "lambda" in df.columns:
        invalid = df[df["lambda"] < 1.0]
        if len(invalid) > 0:
            issues.append(f"Λ(r) < 1 のデータが {len(invalid)} 個あります（理論的に不可能）")
    
    # 2. 単調減少の確認（rが大きくなるほどΛ(r)は減少する傾向）
    if len(df) > 1:
        df_sorted = df.sort_values("r")
        lambda_values = df_sorted["lambda"].values
        # 連続する値の増加回数をカウント
        increases = sum(1 for i in range(1, len(lambda_values)) 
                       if lambda_values[i] > lambda_values[i-1] * 1.1)  # 10%以上の増加
        if increases > len(df) * 0.3:  # 30%以上が増加
            warnings.append("Λ(r)が単調減少していません（データに問題がある可能性）")
    
    # 3. 異常値の検出（外れ値）
    if "lambda" in df.columns:
        q1, q3 = df["lambda"].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df["lambda"] < q1 - 3 * iqr) | (df["lambda"] > q3 + 3 * iqr)]
        if len(outliers) > 0:
            warnings.append(f"外れ値が {len(outliers)} 個検出されました")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_points": len(df),
            "lambda_range": [float(df["lambda"].min()), float(df["lambda"].max())],
            "r_range": [float(df["r"].min()), float(df["r"].max())],
        }
    }


def validate_percolation(percolation_path: Path, percolation_stats_path: Path = None) -> Dict:
    """Percolationの妥当性を検証"""
    if not percolation_path.exists():
        return {"valid": False, "error": "File not found"}
    
    df = pd.read_csv(percolation_path)
    
    issues = []
    warnings = []
    
    # 必須カラムの確認
    required_columns = ["d", "max_cluster_size", "n_clusters"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return {"valid": False, "error": f"Missing columns: {missing}"}
    
    # 理論的制約の確認
    # 1. max_cluster_size <= n_clusters * (各クラスタの平均サイズ)
    # 実際には、max_cluster_sizeは全ノード数以下であるべき
    if "max_cluster_size" in df.columns:
        max_cluster = df["max_cluster_size"].max()
        # 妥当性チェック: 極端に大きな値がないか
        if max_cluster > 1e6:
            warnings.append(f"最大クラスタサイズが異常に大きいです: {max_cluster}")
    
    # 2. n_clusters >= 1
    if "n_clusters" in df.columns:
        invalid = df[df["n_clusters"] < 1]
        if len(invalid) > 0:
            issues.append(f"n_clusters < 1 のデータが {len(invalid)} 個あります")
    
    # 3. giant_fractionの範囲チェック [0, 1]
    if "giant_fraction" in df.columns:
        invalid = df[(df["giant_fraction"] < 0) | (df["giant_fraction"] > 1)]
        if len(invalid) > 0:
            issues.append(f"giant_fractionが[0,1]の範囲外のデータが {len(invalid)} 個あります")
    
    # 4. パーコレーション閾値の確認
    if percolation_stats_path and percolation_stats_path.exists():
        stats = yaml.safe_load(percolation_stats_path.read_text())
        d_critical = stats.get("d_critical_50", None)
        if d_critical:
            # 閾値がデータ範囲内にあるか
            d_min, d_max = df["d"].min(), df["d"].max()
            if d_critical < d_min or d_critical > d_max:
                warnings.append(f"パーコレーション閾値({d_critical:.2f})がデータ範囲[{d_min:.2f}, {d_max:.2f}]外です")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_points": len(df),
            "d_range": [float(df["d"].min()), float(df["d"].max())],
            "max_cluster_size_range": [int(df["max_cluster_size"].min()), int(df["max_cluster_size"].max())],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="分析結果の妥当性を検証")
    parser.add_argument("--outputs-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--output", type=Path, help="検証結果の出力ファイル（JSON）")
    
    args = parser.parse_args()
    
    outputs_dir = args.outputs_dir
    
    results = {
        "outputs_dir": str(outputs_dir),
        "validations": {}
    }
    
    print(f"\n{'='*60}")
    print(f"分析結果の妥当性検証: {outputs_dir.name}")
    print(f"{'='*60}\n")
    
    # MFA検証
    mfa_spectrum = outputs_dir / "mfa_spectrum.csv"
    if mfa_spectrum.exists():
        print("MFAスペクトラムの検証...")
        mfa_result = validate_mfa_spectrum(mfa_spectrum)
        results["validations"]["mfa"] = mfa_result
        
        if mfa_result["valid"]:
            print("  ✅ MFA: 妥当")
        else:
            print(f"  ❌ MFA: 問題あり")
            for issue in mfa_result.get("issues", []):
                print(f"    - {issue}")
        
        for warning in mfa_result.get("warnings", []):
            print(f"  ⚠️  {warning}")
        print()
    else:
        print("  ⚠️  MFAスペクトラムファイルが見つかりません\n")
    
    # Lacunarity検証
    lacunarity = outputs_dir / "lacunarity.csv"
    if lacunarity.exists():
        print("Lacunarityの検証...")
        lac_result = validate_lacunarity(lacunarity)
        results["validations"]["lacunarity"] = lac_result
        
        if lac_result["valid"]:
            print("  ✅ Lacunarity: 妥当")
        else:
            print(f"  ❌ Lacunarity: 問題あり")
            for issue in lac_result.get("issues", []):
                print(f"    - {issue}")
        
        for warning in lac_result.get("warnings", []):
            print(f"  ⚠️  {warning}")
        print()
    else:
        print("  ⚠️  Lacunarityファイルが見つかりません\n")
    
    # Percolation検証
    percolation = outputs_dir / "percolation.csv"
    percolation_stats = outputs_dir / "percolation_stats.yaml"
    if percolation.exists():
        print("Percolationの検証...")
        perc_result = validate_percolation(percolation, percolation_stats)
        results["validations"]["percolation"] = perc_result
        
        if perc_result["valid"]:
            print("  ✅ Percolation: 妥当")
        else:
            print(f"  ❌ Percolation: 問題あり")
            for issue in perc_result.get("issues", []):
                print(f"    - {issue}")
        
        for warning in perc_result.get("warnings", []):
            print(f"  ⚠️  {warning}")
        print()
    else:
        print("  ⚠️  Percolationファイルが見つかりません\n")
    
    # 総合評価
    all_valid = all(
        v.get("valid", False) 
        for v in results["validations"].values() 
        if "valid" in v
    )
    
    print(f"{'='*60}")
    if all_valid:
        print("✅ 全指標が妥当です")
        results["overall_valid"] = True
    else:
        print("❌ 一部の指標に問題があります")
        results["overall_valid"] = False
    print(f"{'='*60}\n")
    
    # 結果を保存
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"検証結果を保存: {args.output}")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())

