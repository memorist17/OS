# Issue 2: 取得地点サンプルの選び方 - 実装仕様書

## 概要

都市構造解析のための地点サンプリング戦略を実装する。現在は手動で地点を指定しているが、統計的に代表性のあるサンプルを生成できるようにする。
目的としてはopensparsityの特異性を主張するために指標において有意に異なる多様なデータソースを300個程度でリストアップすることである。
現状の手動で設定しているものはラベルというか代表的なものなので確認に役立つので残す
実行可能な時間で（1時間ほど）サンプリングを行いたいのでそれに適した実行方法にする
他の手法にくらべなぜそれが良いのかあとから主張できるようにしたい


**優先度**: High  
**ラベル**: enhancement, data-acquisition, critical  
**関連ファイル**: `src/acquisition/sampler.py` (新規), `scripts/batch_process_places.py`, `configs/default.yaml`

## 背景・課題

### 現状の問題
- 取得地点の選定基準が不明確（手動指定のみ）
- サンプルサイズの妥当性が検証されていない
- 地理的・社会的代表性が確保されていない
- バイモーダル分布（高密度・低密度領域）を考慮していない

### 参考資料
- `bimodal_all_scales_with_density.pdf`: バイモーダル分布と密度ベースサンプリングの理論的基盤

## 実装仕様

### 1. 新規モジュール: `src/acquisition/sampler.py`

#### 1.1 クラス設計

```python
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import numpy as np
import geopandas as gpd

class SamplingMethod(str, Enum):
    """サンプリング方法の列挙型"""
    GRID = "grid"
    RANDOM = "random"
    DENSITY_WEIGHTED = "density_weighted"
    BIMODAL = "bimodal"
    STRATIFIED = "stratified"

@dataclass
class LocationSampler:
    """
    地点サンプリング戦略を実装するクラス。
    
    Attributes:
        method: サンプリング方法
        n_samples: サンプル数
        bbox_wgs84: サンプリング範囲 (min_lon, min_lat, max_lon, max_lat)
        grid_size_m: グリッドサイズ（メートル、grid method用）
        density_data: 密度データ（GeoDataFrame、density_weighted用）
        seed: 乱数シード（再現性のため）
    """
    method: SamplingMethod = SamplingMethod.GRID
    n_samples: int = 100
    bbox_wgs84: tuple[float, float, float, float] | None = None
    grid_size_m: float = 2000.0  # 2kmグリッド
    density_data: gpd.GeoDataFrame | None = None
    seed: int | None = None
    
    def sample(self) -> list[dict[str, float]]:
        """
        サンプリング地点を生成する。
        
        Returns:
            地点のリスト。各要素は {"latitude": float, "longitude": float} の形式
        """
        ...
```

```

### 2. 設定ファイル拡張: `configs/default.yaml`

```yaml
# ===== Phase 1: Acquisition & Space =====
canvas:
  half_size_m: 1000        # 中心から±1000m = 2km四方
  resolution_m: 1.0        # 1px = 1m

# サンプリング設定（新規追加）
sampling:
  method: "grid"           # "grid" | "random" | "density_weighted"
  n_samples: 100           # サンプル数
  grid_size_m: 2000.0      # グリッドサイズ（grid method用、メートル）
  seed: null               # 乱数シード（null = ランダム）
  use_density: false       # 密度データを使用するか（density_weighted用）
  density_grid_size_m: 500.0  # 密度計算用グリッドサイズ
```

### 3. バッチ処理スクリプト拡張: `scripts/batch_process_places.py`

#### 3.1 サンプリング機能の統合

```python
def generate_sample_locations(
    bbox_wgs84: tuple[float, float, float, float],
    config: dict
) -> list[dict[str, float]]:
    """
    設定に基づいてサンプリング地点を生成する。
    
    Args:
        bbox_wgs84: サンプリング範囲
        config: 設定辞書
    
    Returns:
        地点のリスト
    """
    from src.acquisition.sampler import LocationSampler, SamplingMethod
    
    sampling_config = config.get("sampling", {})
    method = SamplingMethod(sampling_config.get("method", "grid"))
    
    sampler = LocationSampler(
        method=method,
        n_samples=sampling_config.get("n_samples", 100),
        bbox_wgs84=bbox_wgs84,
        grid_size_m=sampling_config.get("grid_size_m", 2000.0),
        seed=sampling_config.get("seed"),
    )
    
    # 密度ベースの場合は密度データを取得
    if method == SamplingMethod.DENSITY_WEIGHTED:
        # Overture Mapsから密度データを取得する処理
        # （実装詳細は後述）
        ...
    
    return sampler.sample()
```

#### 3.2 コマンドライン引数の追加

```python
parser.add_argument(
    "--sampling-method",
    type=str,
    choices=["grid", "random", "density_weighted"],
    default=None,
    help="Sampling method (overrides config)",
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help="Number of samples (overrides config)",
)
parser.add_argument(
    "--bbox",
    type=float,
    nargs=4,
    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    help="Sampling bounding box (WGS84)",
)
```

### 4. 密度データ取得機能

#### 4.1 Overture Mapsからの密度計算

```python
def compute_density_from_overture(
    bbox_wgs84: tuple[float, float, float, float],
    grid_size_m: float = 500.0
) -> gpd.GeoDataFrame:
    """
    Overture Mapsから建物・道路データを取得し、密度を計算する。
    
    Args:
        bbox_wgs84: 取得範囲
        grid_size_m: グリッドサイズ
    
    Returns:
        密度データのGeoDataFrame
    """
    from src.acquisition.overture_fetcher import OvertureFetcher
    
    with OvertureFetcher(bbox_wgs84=bbox_wgs84) as fetcher:
        buildings = fetcher.fetch_buildings()
        roads = fetcher.fetch_roads()
    
    return compute_density(buildings, roads, bbox_wgs84, grid_size_m)
```

## 実装手順

### Phase 1: 基本実装（MVP）
1. `src/acquisition/sampler.py`を作成
2. `LocationSampler`クラスを実装
3. `grid`と`random`メソッドを実装
4. `configs/default.yaml`に`sampling`セクションを追加
5. `scripts/batch_process_places.py`にサンプリング機能を統合
6. テストを追加

### Phase 2: 密度ベースサンプリング
1. `compute_density`関数を実装
2. `compute_density_from_overture`関数を実装
3. `density_weighted`メソッドを実装
4. テストを追加

### Phase 3: 将来拡張
- `bimodal`メソッドの実装
- `stratified`メソッドの実装

## テスト仕様

### ユニットテスト

```python
# tests/test_sampler.py

def test_grid_sampling():
    """グリッドサンプリングのテスト"""
    sampler = LocationSampler(
        method=SamplingMethod.GRID,
        n_samples=10,
        bbox_wgs84=(139.0, 35.0, 140.0, 36.0),
        grid_size_m=10000.0
    )
    samples = sampler.sample()
    assert len(samples) == 10
    assert all("latitude" in s and "longitude" in s for s in samples)

def test_random_sampling():
    """ランダムサンプリングのテスト"""
    sampler = LocationSampler(
        method=SamplingMethod.RANDOM,
        n_samples=10,
        bbox_wgs84=(139.0, 35.0, 140.0, 36.0),
        seed=42
    )
    samples1 = sampler.sample()
    samples2 = sampler.sample()  # 同じシードで再実行
    assert samples1 == samples2  # 再現性の確認

def test_density_weighted_sampling():
    """密度ベース重み付きサンプリングのテスト"""
    # 密度データのモックを作成
    density_data = create_mock_density_data()
    sampler = LocationSampler(
        method=SamplingMethod.DENSITY_WEIGHTED,
        n_samples=10,
        bbox_wgs84=(139.0, 35.0, 140.0, 36.0),
        density_data=density_data
    )
    samples = sampler.sample()
    assert len(samples) == 10
```

## 後方互換性

- 既存の`batch_process_places.py`の動作は維持（手動指定も可能）
- `sampling`設定が存在しない場合は、既存の動作（手動指定）を維持
- デフォルト値は既存の動作に影響しない

## 注意事項

- サンプリング範囲（`bbox_wgs84`）は、実際のデータ取得範囲（`half_size_m`）を考慮する必要がある
- 密度計算は計算コストが高いため、キャッシュ機能を検討
- 大規模なサンプリング（1000以上）では、メモリ使用量に注意

## 参考実装

- `scripts/batch_process_places.py`: 既存のバッチ処理ロジック
- `src/acquisition/overture_fetcher.py`: Overture Mapsからのデータ取得
- `src/projection/aeqd_transformer.py`: 座標変換処理



