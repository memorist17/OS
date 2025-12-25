# Issue 6: 道路ネットワークの線形補完 - 実装仕様書

## 概要

道路ネットワークが切れている問題を修正する。道路幅を考慮した接続判定と、近接ノード間の線形補完を実装し、実際には接続されている道路がネットワーク上で切断されないようにする。

**優先度**: Medium  
**ラベル**: bug, preprocessing  
**関連ファイル**: `src/preprocessing/network_builder.py`, `configs/default.yaml`

## 背景・課題

### 現状の問題
- 道路ネットワークが切れている（実際には接続されているが、ネットワーク上で切断されている）
- 道路幅の反映が不十分
- 近接ノード間の接続判定が不適切
- 道路タイプ別の接続ルールが未定義

### 影響範囲
- パーコレーション解析の精度低下
- ネットワーク距離計算の誤差
- 連結成分数の過大評価

## 実装仕様

### 1. `NetworkBuilder`クラスの拡張

#### 1.1 クラス属性の追加

```python
@dataclass
class NetworkBuilder:
    """Build spatial network graph from roads and buildings using NetworkX."""

    snap_tolerance: float = 1.0  # メートル（既存）
    
    # 新規追加
    interpolation_enabled: bool = True  # 線形補完を有効化
    connection_threshold_m: float = 10.0  # 接続判定の距離閾値（メートル）
    use_road_width: bool = True  # 道路幅を考慮するか
    road_width_buffer_ratio: float = 1.5  # 道路幅に対するバッファ倍率
    min_road_width_m: float = 3.0  # 最小道路幅（メートル）
```

#### 1.2 接続判定メソッドの追加

```python
def _should_connect_nodes(
    self,
    node1: tuple[float, float],
    node2: tuple[float, float],
    road_width1: float | None = None,
    road_width2: float | None = None,
) -> bool:
    """
    2つのノードを接続すべきか判定する。
    
    Args:
        node1: ノード1の座標 (x, y)
        node2: ノード2の座標 (x, y)
        road_width1: ノード1に接続する道路の幅（メートル）
        road_width2: ノード2に接続する道路の幅（メートル）
    
    Returns:
        接続すべき場合はTrue
    """
    # ユークリッド距離を計算
    distance = np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)
    
    # 道路幅を考慮した閾値を計算
    if self.use_road_width and road_width1 is not None and road_width2 is not None:
        # 両方の道路幅の平均を基準にする
        avg_width = (road_width1 + road_width2) / 2
        threshold = avg_width * self.road_width_buffer_ratio
        threshold = max(threshold, self.min_road_width_m)
    else:
        threshold = self.connection_threshold_m
    
    return distance <= threshold
```

#### 1.3 線形補完メソッドの追加

```python
def _interpolate_connection(
    self,
    node1: tuple[float, float],
    node2: tuple[float, float],
    road_width1: float | None = None,
    road_width2: float | None = None,
) -> list[tuple[float, float]]:
    """
    2つのノード間を線形補完する。
    
    Args:
        node1: ノード1の座標 (x, y)
        node2: ノード2の座標 (x, y)
        road_width1: ノード1に接続する道路の幅（メートル）
        road_width2: ノード2に接続する道路の幅（メートル）
    
    Returns:
        補完された中間点のリスト（空の場合は補完不要）
    """
    distance = np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)
    
    # 距離が閾値以下の場合は補完不要
    if distance <= self.connection_threshold_m:
        return []
    
    # 補完点の数を決定（距離に応じて）
    n_intermediate = max(1, int(distance / self.connection_threshold_m))
    
    # 線形補完
    intermediate_points = []
    for i in range(1, n_intermediate + 1):
        t = i / (n_intermediate + 1)
        x = node1[0] + t * (node2[0] - node1[0])
        y = node1[1] + t * (node2[1] - node1[1])
        intermediate_points.append((x, y))
    
    return intermediate_points
```

#### 1.4 `build_network`メソッドの拡張

```python
def build_network(
    self,
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    verbose: bool = True,
) -> nx.Graph:
    """
    Build network graph with road width-aware connection and interpolation.
    
    （既存の実装に以下を追加）
    
    1. 道路幅情報の取得
    2. 近接ノード間の接続判定（道路幅を考慮）
    3. 線形補完の適用
    """
    G = nx.Graph()
    
    # 既存の実装（道路エッジの追加）...
    
    # 新規: 近接ノード間の接続判定と補完
    if self.interpolation_enabled:
        self._apply_interpolation(G, roads, verbose=verbose)
    
    # 既存の実装（建物ノードの追加）...
    
    return G
```

#### 1.5 補完適用メソッド

```python
def _apply_interpolation(
    self,
    graph: nx.Graph,
    roads: gpd.GeoDataFrame,
    verbose: bool = True,
) -> None:
    """
    ネットワークに線形補完を適用する。
    
    Args:
        graph: NetworkXグラフ
        roads: 道路GeoDataFrame（道路幅情報を含む）
    """
    # ノードの座標と道路幅のマッピングを作成
    node_coords = {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}
    node_road_widths = self._get_node_road_widths(graph, roads)
    
    # 近接ノードペアを検出
    node_list = list(graph.nodes())
    n_nodes = len(node_list)
    
    if verbose:
        print(f"Applying interpolation to {n_nodes} nodes...")
    
    added_edges = 0
    for i in tqdm(range(n_nodes), desc="Interpolating connections", disable=not verbose):
        node1 = node_list[i]
        coords1 = node_coords[node1]
        width1 = node_road_widths.get(node1)
        
        for j in range(i + 1, n_nodes):
            node2 = node_list[j]
            coords2 = node_coords[node2]
            width2 = node_road_widths.get(node2)
            
            # 既に接続されている場合はスキップ
            if graph.has_edge(node1, node2):
                continue
            
            # 接続判定
            if self._should_connect_nodes(coords1, coords2, width1, width2):
                # 線形補完
                intermediate_points = self._interpolate_connection(
                    coords1, coords2, width1, width2
                )
                
                # エッジを追加
                prev_node = node1
                for inter_point in intermediate_points:
                    # 中間ノードを作成（既存のノードとスナップ）
                    inter_node = self._get_or_create_node_at(
                        graph, inter_point[0], inter_point[1]
                    )
                    length = self._compute_distance(prev_node, inter_node, graph)
                    graph.add_edge(prev_node, inter_node, length=length, type="interpolated")
                    prev_node = inter_node
                
                # 最後のノードとnode2を接続
                length = self._compute_distance(prev_node, node2, graph)
                graph.add_edge(prev_node, node2, length=length, type="interpolated")
                added_edges += 1
    
    if verbose:
        print(f"Added {added_edges} interpolated edges")
```

#### 1.6 補助メソッド

```python
def _get_node_road_widths(
    self,
    graph: nx.Graph,
    roads: gpd.GeoDataFrame,
) -> dict[int, float]:
    """
    各ノードに接続する道路の幅を取得する。
    
    Args:
        graph: NetworkXグラフ
        roads: 道路GeoDataFrame
    
    Returns:
        ノードID -> 道路幅のマッピング
    """
    node_widths = {}
    width_column = "width"
    
    for _, road_row in roads.iterrows():
        geom = road_row.geometry
        width = road_row.get(width_column, self.min_road_width_m)
        
        if geom is None or geom.is_empty:
            continue
        
        # 道路の端点を取得
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            for coord in [coords[0], coords[-1]]:
                # ノードを検索（スナップ許容範囲内）
                node = self._find_nearby_node(graph, coord[0], coord[1])
                if node is not None:
                    # 既存の幅より大きい場合は更新
                    if node not in node_widths or width > node_widths[node]:
                        node_widths[node] = width
    
    return node_widths

def _find_nearby_node(
    self,
    graph: nx.Graph,
    x: float,
    y: float,
) -> int | None:
    """
    指定座標の近くにあるノードを検索する。
    
    Args:
        graph: NetworkXグラフ
        x: X座標
        y: Y座標
    
    Returns:
        見つかったノードID、見つからない場合はNone
    """
    for node, data in graph.nodes(data=True):
        node_x = data.get("x")
        node_y = data.get("y")
        if node_x is None or node_y is None:
            continue
        
        distance = np.sqrt((node_x - x)**2 + (node_y - y)**2)
        if distance <= self.snap_tolerance:
            return node
    
    return None
```

### 2. 設定ファイル拡張: `configs/default.yaml`

```yaml
# ===== Phase 2: Preprocessing =====
network:
  snap_tolerance: 1.0  # メートル（既存）
  
  # 線形補完設定（新規追加）
  interpolation:
    enabled: true              # 線形補完を有効化
    connection_threshold_m: 10.0  # 接続判定の距離閾値（メートル）
    use_road_width: true      # 道路幅を考慮するか
    road_width_buffer_ratio: 1.5  # 道路幅に対するバッファ倍率
    min_road_width_m: 3.0      # 最小道路幅（メートル）
```

### 3. `run_acquisition.py`の更新

`NetworkBuilder`の初期化時に設定を読み込む：

```python
# scripts/run_acquisition.py の修正箇所

network_config = config.get("network", {})
interpolation_config = network_config.get("interpolation", {})

builder = NetworkBuilder(
    snap_tolerance=network_config.get("snap_tolerance", 1.0),
    interpolation_enabled=interpolation_config.get("enabled", True),
    connection_threshold_m=interpolation_config.get("connection_threshold_m", 10.0),
    use_road_width=interpolation_config.get("use_road_width", True),
    road_width_buffer_ratio=interpolation_config.get("road_width_buffer_ratio", 1.5),
    min_road_width_m=interpolation_config.get("min_road_width_m", 3.0),
)
```

## 実装手順

### Phase 1: 基本実装
1. `NetworkBuilder`クラスに補完関連の属性を追加
2. `_should_connect_nodes`メソッドを実装
3. `_interpolate_connection`メソッドを実装
4. `_apply_interpolation`メソッドを実装
5. `build_network`メソッドに補完処理を統合
6. `configs/default.yaml`に設定を追加
7. テストを追加

### Phase 2: 最適化
1. 近接ノード検出の効率化（空間インデックス使用）
2. 補完エッジの重複チェック
3. パフォーマンステスト

## テスト仕様

### ユニットテスト

```python
# tests/test_network_builder.py

def test_connection_threshold():
    """接続判定の閾値テスト"""
    builder = NetworkBuilder(
        connection_threshold_m=10.0,
        use_road_width=False
    )
    
    node1 = (0.0, 0.0)
    node2 = (5.0, 0.0)  # 距離5m
    assert builder._should_connect_nodes(node1, node2) == True
    
    node3 = (15.0, 0.0)  # 距離15m
    assert builder._should_connect_nodes(node1, node3) == False

def test_road_width_aware_connection():
    """道路幅を考慮した接続判定のテスト"""
    builder = NetworkBuilder(
        connection_threshold_m=10.0,
        use_road_width=True,
        road_width_buffer_ratio=1.5,
        min_road_width_m=3.0
    )
    
    node1 = (0.0, 0.0)
    node2 = (12.0, 0.0)  # 距離12m
    
    # 道路幅10mの場合、閾値は15m（10 * 1.5）
    assert builder._should_connect_nodes(node1, node2, 10.0, 10.0) == True
    
    # 道路幅5mの場合、閾値は7.5m（5 * 1.5）、最小値3.0を下回るので3.0
    assert builder._should_connect_nodes(node1, node2, 5.0, 5.0) == False

def test_interpolation():
    """線形補完のテスト"""
    builder = NetworkBuilder(
        connection_threshold_m=10.0,
        interpolation_enabled=True
    )
    
    node1 = (0.0, 0.0)
    node2 = (30.0, 0.0)  # 距離30m
    
    intermediate = builder._interpolate_connection(node1, node2)
    assert len(intermediate) >= 2  # 少なくとも2つの補完点
    
    # 補完点が線形に配置されていることを確認
    for i, point in enumerate(intermediate):
        expected_x = (i + 1) * 30.0 / (len(intermediate) + 1)
        assert abs(point[0] - expected_x) < 0.1

def test_network_interpolation_integration():
    """ネットワーク構築と補完の統合テスト"""
    # テストデータを作成
    roads = create_test_roads_with_gaps()
    
    builder = NetworkBuilder(
        interpolation_enabled=True,
        connection_threshold_m=10.0
    )
    
    graph = builder.build_network(roads)
    
    # 補完により接続が増えていることを確認
    assert graph.number_of_edges() > 0
    
    # 補完エッジが存在することを確認
    interpolated_edges = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get("type") == "interpolated"
    ]
    assert len(interpolated_edges) > 0
```

## 後方互換性

- `interpolation_enabled=False`がデフォルトの場合、既存の動作を維持
- 既存のネットワーク構築ロジックは変更しない（補完処理を追加するのみ）
- 設定ファイルに`network.interpolation`が存在しない場合は、補完を無効化

## 注意事項

- 補完処理は計算コストが高いため、大規模ネットワークでは注意
- 道路幅データが存在しない場合は、`use_road_width=False`にフォールバック
- 補完エッジが多すぎると、ネットワークが過密になる可能性がある

## 参考実装

- `src/preprocessing/network_builder.py`: 既存のネットワーク構築ロジック
- `src/acquisition/overture_fetcher.py`: 道路幅データの取得



