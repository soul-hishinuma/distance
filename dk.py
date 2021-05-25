from __future__ import annotations
from typing import Tuple, List, Optional, Dict
from heapq import heappush, heappop

VertexStr = str


class Vertex:
    """各頂点を表す文字列の定数を扱うクラス。
    """

    A: VertexStr = 'A'
    B: VertexStr = 'B'
    C: VertexStr = 'C'
    D: VertexStr = 'D'
    E: VertexStr = 'E'
    F: VertexStr = 'F'
    G: VertexStr = 'G'
    H: VertexStr = 'H'
    I: VertexStr = 'I'
    J: VertexStr = 'J'
    K: VertexStr = 'K'


class Edge:

    def __init__(
            self, from_idx: int, to_idx: int,
            from_vertex: VertexStr, to_vertex: VertexStr,
            weight: float) -> None:
        """
        エッジ単体を扱うクラス。

        Parameters
        ----------
        from_idx : int
            接続元の頂点のインデックス。
        to_idx : int
            接続先の頂点のインデックス。
        weight : float
            エッジの重み。
        """
        self.from_idx: int = from_idx
        self.to_idx: int = to_idx
        self.from_vertex: VertexStr = from_vertex
        self.to_vertex: VertexStr = to_vertex
        self.weight: float = weight

    def reversed(self) -> Edge:
        """
        頂点の接続元と接続先を反転させたエッジを取得する。

        Returns
        -------
        reversed_edge : Edge
            頂点の接続元と接続先を反転させたエッジ。
        """
        reversed_edge = Edge(
            from_idx=self.to_idx,
            to_idx=self.from_idx,
            from_vertex=self.from_vertex,
            to_vertex=self.to_vertex,
            weight=self.weight)
        return reversed_edge

    def __str__(self) -> str:
        """
        エッジの情報を文字列に変換する。

        Returns
        -------
        edge_info_str : str
            変換されたエッジ情報の文字列。
        """
        edge_info_str: str = (
            f'from: {self.from_vertex}'
            f'(weight: {self.weight})'
            f' -> to: {self.to_vertex}'
        )
        return edge_info_str


class Graph:

    def __init__(self, vertices: List[VertexStr]) -> None:
        """
        グラフを扱うためのクラス。

        Parameters
        ----------
        vertices : list of str
            頂点の文字列のリスト。
        """
        self._vertices: List[VertexStr] = vertices
        self._edges: List[List[Edge]] = []
        for _ in vertices:
            self._edges.append([])

    def vertex_at(self, index: int) -> VertexStr:
        """
        指定されたインデックスの頂点の文字列を取得する。

        Parameters
        ----------
        index : int
            対象のインデックス。

        Returns
        -------
        vertex_str : str
            対象のインデックス位置の頂点の文字列。
        """
        return self._vertices[index]

    def index_of(self, vertex: VertexStr) -> int:
        """
        対象の頂点のインテックスを取得する。

        Parameters
        ----------
        vertex : str
            対象の頂点識別用の文字列。

        Returns
        -------
        index : int
            対象の頂点のインデックス。
        """
        return self._vertices.index(vertex)

    @property
    def vertex_count(self):
        """
        設定されている頂点数の属性値。

        Returns
        -------
        vertex_count : int
            設定されている頂点数。
        """
        return len(self._vertices)

    def edges_at(self, vertex_index: int) -> List[Edge]:
        """
        指定のインデックス位置の頂点に設定されているエッジの
        リストを取得する。

        Parameters
        ----------
        vertex_index : int
            対象の頂点位置のインデックス。

        Returns
        -------
        edges : list of Edge
            対象の頂点に設定されているエッジのリスト。
        """
        return self._edges[vertex_index]

    def get_neighbor_vertices_and_weights_by_index(
            self, vertex_index: int) -> List[Tuple[VertexStr, float]]:
        """
        指定されたインデックスの頂点にエッジで繋がっている
        頂点とそのエッジの重みの値のタプルのリストを取得する。

        Parameters
        ----------
        vertex_index : int
            対象の頂点のインデックス。

        Returns
        -------
        neighbor_vertices_and_weights : list of tuple
            算出された頂点とそのエッジの重みのタプルを格納したリスト。
        """
        neighbor_vertices_and_weights: List[Tuple[VertexStr, float]] = []
        for edge in self.edges_at(vertex_index=vertex_index):
            tuple_val = (
                self.vertex_at(index=edge.to_idx),
                edge.weight,
            )
            neighbor_vertices_and_weights.append(tuple_val)
        return neighbor_vertices_and_weights

    def add_edge(self, edge: Edge) -> None:
        """
        グラフにエッジの追加を行う。

        Notes
        -----
        反転させたエッジも接続先の頂点に対して追加される（エッジが
        メソッド実行で合計で2件追加される）。

        Parameters
        ----------
        edge : Edge
            追加対象のエッジ。
        """
        self._edges[edge.from_idx].append(edge)
        self._edges[edge.to_idx].append(edge.reversed())

    def add_edge_by_vertices(
            self, from_vertex: VertexStr,
            to_vertex: VertexStr,
            weight: float) -> None:
        """
        指定された2つの頂点間のエッジの追加を行う。

        Parameters
        ----------
        from_vertex : str
            接続元の頂点の指定。
        to_vertex : str
            接続先の頂点の指定。
        weight : float
            エッジの重みの値。
        """
        from_idx = self._vertices.index(from_vertex)
        to_idx = self._vertices.index(to_vertex)
        edge = Edge(
            from_idx=from_idx,
            to_idx=to_idx,
            from_vertex=from_vertex,
            to_vertex=to_vertex,
            weight=weight)
        self.add_edge(edge=edge)

    def __str__(self) -> str:
        """
        グラフ情報の文字列を返却する。

        Returns
        -------
        graph_info : str
            グラフ情報の文字列。
        """
        graph_info: str = ''
        for index in range(self.vertex_count):
            neighbors_data = self.get_neighbor_vertices_and_weights_by_index(
                vertex_index=index)
            graph_info += (
                f'対象の頂点 : {self.vertex_at(index=index)}'
                f' -> 隣接頂点データ : {neighbors_data}\n'
            )
        return graph_info


class DijkstraDistanceVertex:

    def __init__(self, vertex_idx: int, distance: float) -> None:
        """
        ダイクストラ法のための、特定の頂点のスタートの頂点からの距離（重み）
        や他の頂点との比較制御のためのクラス。

        Parameters
        ----------
        vertex : str
            対象の頂点識別用のインデックス。
        distance : float
            スタートの頂点からの距離。
        """
        self.vertex_idx = vertex_idx
        self.distance = distance

    def __lt__(self, other: DijkstraDistanceVertex) -> bool:
        """
        他の頂点の距離（重み）の小なりの比較結果の真偽値を取得する。

        Parameters
        ----------
        other : DijkstraDistanceVertex
            比較対象の頂点。

        Returns
        -------
        result : bool
            Trueで小なりの条件を満たす場合。
        """
        return self.distance < other.distance

    def __eq__(self, other: DijkstraDistanceVertex) -> bool:
        """
        他の頂点との距離（重み）が一致しているかどうかの真偽値を
        取得する。

        Parameters
        ----------
        other : DijkstraDistanceVertex
            比較対象の頂点。

        Returns
        -------
        result : bool
            Trueで一致の条件を満たす場合。
        """
        return self.distance == other.distance


class PriorityQueue:

    def __init__(self) -> None:
        """
        優先度付きキューを扱うクラス。
        """
        self._container: List[DijkstraDistanceVertex] = []

    @property
    def empty(self) -> bool:
        """
        キューが空かどうかの真偽値の属性。

        Returns
        -------
        result : bool
            キューが空であればTrueが設定される。
        """
        return not self._container

    def push(self, item: DijkstraDistanceVertex) -> None:
        """
        キューにダイクストラ法の特定頂点のスタートからの距離を扱う
        インスタンスを追加する。

        Parameters
        ----------
        item : DijkstraDistanceVertex
            追加対象のインスタンス。
        """
        heappush(self._container, item)

    def pop(self) -> DijkstraDistanceVertex:
        """
        キューから優先度に応じたインスタンスを1件取り出す。

        Returns
        -------
        item : DijkstraDistanceVertex
            取り出されたインスタンス。
        """
        return heappop(self._container)


def dijkstra(
        graph: Graph,
        root_vertex: VertexStr
        ) -> tuple[List[Optional[float]], Dict[int, Edge]]:
    """
    ダイクストラ法を実施し、各頂点ごとの距離と各頂点までの最短ルートの
    パスを算出する。

    Parameters
    ----------
    graph : Graph
        対象のグラフ。
    root_vertex : str
        探索を開始する位置の頂点識別用の文字列。

    Returns
    -------
    distances : list of float
        各頂点の探索開始位置の頂点からの距離。
    path_dict : dict
        キーに対象の頂点のインデックス、値にはその頂点に到達する
        のに最短となるルートの、直前のエッジを格納する。
    """
    first_idx: int = graph.index_of(vertex=root_vertex)
    distances: List[Optional[float]] = [
        None for _ in range(graph.vertex_count)]
    distances[first_idx] = 0

    path_dict: Dict[int, Edge] = {}
    priority_queue: PriorityQueue = PriorityQueue()
    priority_queue.push(
        item=DijkstraDistanceVertex(
            vertex_idx=first_idx,
            distance=0))

    while not priority_queue.empty:
        from_idx:int = priority_queue.pop().vertex_idx
        from_vertex_distance: float = distances[from_idx]
        for edge in graph.edges_at(vertex_index=from_idx):

            # 対象のイテレーション前に既に対象の頂点へ設定されている
            # 距離を取得する（別の経路などで設定されている値）。
            to_distance: Optional[float] = distances[edge.to_idx]

            # 今回の経路での距離を算出する。
            current_route_distance: float = edge.weight + from_vertex_distance

            # もし別ルートでの距離が既に設定されており、且つ今回の経路での
            # 距離の方が短くならなければ更新処理をスキップする。
            if (to_distance is not None
                    and to_distance <= current_route_distance):
                continue

            distances[edge.to_idx] = current_route_distance
            path_dict[edge.to_idx] = edge
            priority_queue.push(
                item=DijkstraDistanceVertex(
                    vertex_idx=edge.to_idx,
                    distance=current_route_distance))

    return distances, path_dict


RoutePath = List[Edge]


def to_route_path_from_path_dict(
        start_vertex_idx: int,
        last_vertex_idx: int,
        path_dict: Dict[int, Edge]) -> RoutePath:
    """
    そのパスへ至る際に最短距離となる直前のエッジを格納した辞書を使用して、
    指定された頂点への最短距離となる

    Parameters
    ----------
    start_vertex_idx : int
        求めたい経路の開始地点となる頂点の
    last_vertex_idx : int
        求めたい経路の最終地点となる頂点のインデックス。
    path_dict : dict
        キーに対象の頂点のインデックス、値にその頂点へ至る最短経路の
        直前のエッジを格納した辞書。
    """
    route_path: RoutePath = []
    current_edge: Edge = path_dict[last_vertex_idx]
    route_path.append(current_edge)
    while current_edge.from_idx != start_vertex_idx:
        current_edge = path_dict[current_edge.from_idx]
        route_path.append(current_edge)
    route_path.reverse()
    return route_path


if __name__ == '__main__':
    graph = Graph(
        vertices=[
            Vertex.A,
            Vertex.B,
            Vertex.C,
            Vertex.D,
            Vertex.E,
            Vertex.F,
            Vertex.G,
            Vertex.H,
            Vertex.I,
            Vertex.J,
            Vertex.K,
        ])

    graph.add_edge_by_vertices(
        from_vertex=Vertex.A,
        to_vertex=Vertex.B,
        weight=80)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.A,
        to_vertex=Vertex.E,
        weight=200)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.B,
        to_vertex=Vertex.C,
        weight=92)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.B,
        to_vertex=Vertex.D,
        weight=83)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.B,
        to_vertex=Vertex.E,
        weight=210)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.C,
        to_vertex=Vertex.D,
        weight=43)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.C,
        to_vertex=Vertex.E,
        weight=93)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.C,
        to_vertex=Vertex.F,
        weight=66)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.D,
        to_vertex=Vertex.G,
        weight=95)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.D,
        to_vertex=Vertex.H,
        weight=123)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.E,
        to_vertex=Vertex.F,
        weight=81)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.F,
        to_vertex=Vertex.G,
        weight=46)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.F,
        to_vertex=Vertex.K,
        weight=100)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.G,
        to_vertex=Vertex.H,
        weight=141)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.G,
        to_vertex=Vertex.I,
        weight=53)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.G,
        to_vertex=Vertex.K,
        weight=112)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.H,
        to_vertex=Vertex.I,
        weight=86)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.I,
        to_vertex=Vertex.J,
        weight=95)
    graph.add_edge_by_vertices(
        from_vertex=Vertex.J,
        to_vertex=Vertex.K,
        weight=92)

    print('-' * 20)
    print('グラフ情報:')
    print(graph)
    print('-' * 20)

    distances, path_dict = dijkstra(
        graph=graph,
        root_vertex=Vertex.A)
    print('算出された頂点Aから各頂点への最短距離情報:')
    for index, distance in enumerate(distances):
        vertex: VertexStr = graph.vertex_at(index=index)
        print('頂点 :', vertex, '距離 :', distance)
    print('-' * 20)

    start_vertex_idx = graph.index_of(vertex=Vertex.A)
    last_vertex_idx = graph.index_of(vertex=Vertex.J)
    route_path: RoutePath = to_route_path_from_path_dict(
        start_vertex_idx=start_vertex_idx,
        last_vertex_idx=last_vertex_idx,
        path_dict=path_dict)
    print('頂点Aから頂点Jまでの最短経路:')
    for edge in route_path:
        print(edge)
