import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tkinter.scrolledtext import ScrolledText

import json
from collections import deque, defaultdict

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Circle


class FlowNetworkSolver:
    """
    Motor matematic pentru algoritmul Ford-Fulkerson (varianta Edmonds-Karp / BFS).
    """

    def __init__(self):
        self.nodes: dict[str, str] = {}
        self.node_ids: list[str] = []
        self.original_cap: dict[str, dict[str, int]] = {}
        self.residual: dict[str, dict[str, int]] = {}

    # ===================== Constructie graf =====================
    def add_node(self, node_id: str, name: str = "") -> bool:
        node_id = node_id.strip()
        if not node_id or node_id in self.nodes:
            return False
        self.nodes[node_id] = name.strip() or node_id
        self.node_ids.append(node_id)
        self.original_cap[node_id] = {}
        return True

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.node_ids.remove(node_id)
        del self.original_cap[node_id]
        for nd in self.node_ids:
            self.original_cap[nd].pop(node_id, None)
        return True

    def add_edge(self, u: str, v: str, capacity: int) -> str:
        if u not in self.nodes:
            return f"Nodul '{u}' nu există."
        if v not in self.nodes:
            return f"Nodul '{v}' nu există."
        if u == v:
            return "Buclele nu sunt permise."
        if capacity <= 0:
            return "Capacitatea trebuie să fie un număr întreg pozitiv."
        self.original_cap[u][v] = self.original_cap[u].get(v, 0) + int(capacity)
        return ""

    def remove_edge(self, u: str, v: str) -> bool:
        if u in self.original_cap and v in self.original_cap[u]:
            del self.original_cap[u][v]
            return True
        return False

    def update_capacity(self, u: str, v: str, capacity: int) -> str:
        if capacity <= 0:
            return "Capacitatea trebuie să fie un număr întreg pozitiv."
        if u not in self.original_cap or v not in self.original_cap.get(u, {}):
            return "Arcul nu există."
        self.original_cap[u][v] = int(capacity)
        return ""

    def edge_list(self) -> list[tuple[str, str, int]]:
        return [
            (u, v, c)
            for u in self.node_ids
            for v, c in self.original_cap[u].items()
            if c > 0
        ]

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def has_edges(self) -> bool:
        return len(self.edge_list()) > 0

    # ===================== Algoritmul Ford-Fulkerson =====================
    def _init_residual(self):
        self.residual = {nid: {} for nid in self.node_ids}
        for u in self.node_ids:
            for v, c in self.original_cap[u].items():
                self.residual[u][v] = c
                if u not in self.residual[v]:
                    self.residual[v][u] = 0

    def _bfs(self, source: str, sink: str):
        visited = {source}
        visited_order = [source]
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in self.node_ids:
                cap = self.residual[u].get(v, 0)
                if v not in visited and cap > 0:
                    visited.add(v)
                    visited_order.append(v)
                    parent[v] = u
                    if v == sink:
                        return parent, visited_order
                    queue.append(v)
        return {}, visited_order

    def _capacity_matrix(self):
        n = len(self.node_ids)
        m = np.zeros((n, n), dtype=float)
        for i, u in enumerate(self.node_ids):
            for j, v in enumerate(self.node_ids):
                m[i, j] = self.original_cap.get(u, {}).get(v, 0)
        return m

    def _residual_matrix(self):
        n = len(self.node_ids)
        m = np.zeros((n, n), dtype=float)
        for i, u in enumerate(self.node_ids):
            for j, v in enumerate(self.node_ids):
                m[i, j] = self.residual.get(u, {}).get(v, 0)
        return m

    def _flow_matrix(self):
        n = len(self.node_ids)
        m = np.zeros((n, n), dtype=float)
        for i, u in enumerate(self.node_ids):
            for j, v in enumerate(self.node_ids):
                cap = self.original_cap.get(u, {}).get(v, 0)
                if cap > 0:
                    res = self.residual.get(u, {}).get(v, 0)
                    m[i, j] = cap - res
        return m

    def iteration_generator(self, source: str, sink: str):
        if source not in self.nodes:
            raise ValueError(f"Sursa '{source}' nu există.")
        if sink not in self.nodes:
            raise ValueError(f"Destinația '{sink}' nu există.")
        if source == sink:
            raise ValueError("Sursa și destinația trebuie să fie noduri diferite.")

        self._init_residual()
        capacity_matrix = self._capacity_matrix()
        max_flow = 0
        flow_history = [0]
        iteration = 0

        while True:
            residual_before = self._residual_matrix()
            parent, visited_order = self._bfs(source, sink)

            if not parent or sink not in parent:
                state = self._make_state(
                    iteration=iteration,
                    status="optimal",
                    message="Nu mai există drumuri de augmentare. Fluxul maxim a fost atins.",
                    source=source,
                    sink=sink,
                    capacity_matrix=capacity_matrix,
                    residual_before=residual_before,
                    residual_after=residual_before,
                    path_ids=None,
                    bottleneck=None,
                    bfs_visited=visited_order,
                    parent=parent,
                    max_flow=max_flow,
                    flow_history=flow_history,
                    final_result=self._build_result(
                        source, sink, max_flow, flow_history, iteration, capacity_matrix
                    ),
                )
                yield state
                return

            iteration += 1

            path = []
            cur = sink
            bottleneck = float("inf")
            while cur is not None:
                path.append(cur)
                p = parent.get(cur)
                if p is not None:
                    bottleneck = min(bottleneck, self.residual[p][cur])
                cur = p
            path.reverse()

            cur = sink
            while parent.get(cur) is not None:
                prev = parent[cur]
                self.residual[prev][cur] -= bottleneck
                self.residual[cur][prev] = self.residual[cur].get(prev, 0) + bottleneck
                cur = prev

            max_flow += bottleneck
            flow_history.append(max_flow)
            residual_after = self._residual_matrix()

            state = self._make_state(
                iteration=iteration,
                status="continue",
                message=f"S-a găsit un drum de augmentare cu gâtul de sticlă {bottleneck}.",
                source=source,
                sink=sink,
                capacity_matrix=capacity_matrix,
                residual_before=residual_before,
                residual_after=residual_after,
                path_ids=path,
                bottleneck=float(bottleneck),
                bfs_visited=visited_order,
                parent=parent,
                max_flow=float(max_flow),
                flow_history=flow_history,
                final_result=None,
            )
            yield state

    def _make_state(self, **kwargs):
        path_ids = kwargs.get("path_ids")
        return {
            "it": kwargs["iteration"],
            "status": kwargs["status"],
            "message": kwargs["message"],
            "source": kwargs["source"],
            "sink": kwargs["sink"],
            "node_ids": list(self.node_ids),
            "node_names": [self.nodes[n] for n in self.node_ids],
            "path_ids": list(path_ids) if path_ids else None,
            "path_names": [self.nodes[n] for n in path_ids] if path_ids else None,
            "bottleneck": kwargs["bottleneck"],
            "max_flow": kwargs["max_flow"],
            "capacity_matrix": kwargs["capacity_matrix"].copy(),
            "residual_before": kwargs["residual_before"].copy(),
            "residual_after": kwargs["residual_after"].copy(),
            "flow_after": self._flow_matrix(),
            "bfs_visited": list(kwargs["bfs_visited"]),
            "parent": dict(kwargs["parent"]),
            "flow_history": list(kwargs["flow_history"]),
            "final_result": kwargs["final_result"],
        }

    def _build_result(self, source, sink, max_flow, flow_history, iterations, capacity_matrix):
        flow_arcs = []
        for u in self.node_ids:
            for v, cap in self.original_cap[u].items():
                if cap > 0:
                    res = self.residual[u].get(v, 0)
                    flow = cap - res
                    flow_arcs.append({
                        "u": u,
                        "v": v,
                        "u_name": self.nodes[u],
                        "v_name": self.nodes[v],
                        "flow": int(flow),
                        "capacity": int(cap),
                        "saturated": flow == cap,
                        "utilization": (flow / cap) if cap else 0.0,
                    })

        s_set = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in self.residual[u].items():
                if v not in s_set and cap > 0:
                    s_set.add(v)
                    queue.append(v)
        t_set = set(self.node_ids) - s_set

        cut_edges = []
        cut_capacity = 0
        for u in self.node_ids:
            if u not in s_set:
                continue
            for v, cap in self.original_cap[u].items():
                if cap > 0 and v in t_set:
                    cut_edges.append({
                        "u": u,
                        "v": v,
                        "u_name": self.nodes[u],
                        "v_name": self.nodes[v],
                        "capacity": int(cap),
                    })
                    cut_capacity += cap

        conservation = []
        for nid in self.node_ids:
            if nid == source or nid == sink:
                continue
            inflow = 0
            for u in self.node_ids:
                cap = self.original_cap[u].get(nid, 0)
                if cap > 0:
                    inflow += cap - self.residual[u].get(nid, 0)
            outflow = 0
            for v in self.node_ids:
                cap = self.original_cap[nid].get(v, 0)
                if cap > 0:
                    outflow += cap - self.residual[nid].get(v, 0)
            conservation.append({
                "node": nid,
                "name": self.nodes[nid],
                "inflow": int(inflow),
                "outflow": int(outflow),
                "balanced": abs(inflow - outflow) < 1e-9,
            })

        source_outflow = sum(
            self.original_cap[source].get(v, 0) - self.residual[source].get(v, 0)
            for v in self.node_ids
            if self.original_cap[source].get(v, 0) > 0
        )
        sink_inflow = sum(
            self.original_cap[u].get(sink, 0) - self.residual[u].get(sink, 0)
            for u in self.node_ids
            if self.original_cap[u].get(sink, 0) > 0
        )

        return {
            "status": "optimal",
            "source": source,
            "sink": sink,
            "source_name": self.nodes[source],
            "sink_name": self.nodes[sink],
            "max_flow": float(max_flow),
            "iterations": iterations,
            "flow_arcs": flow_arcs,
            "S_set": [n for n in self.node_ids if n in s_set],
            "T_set": [n for n in self.node_ids if n in t_set],
            "S_names": [self.nodes[n] for n in self.node_ids if n in s_set],
            "T_names": [self.nodes[n] for n in self.node_ids if n in t_set],
            "cut_edges": cut_edges,
            "cut_capacity": int(cut_capacity),
            "max_flow_min_cut_match": abs(max_flow - cut_capacity) < 1e-9,
            "conservation": conservation,
            "source_outflow": int(source_outflow),
            "sink_inflow": int(sink_inflow),
            "flow_history": list(flow_history),
            "node_ids": list(self.node_ids),
            "node_names": [self.nodes[n] for n in self.node_ids],
            "capacity_matrix": capacity_matrix.copy(),
            "residual_matrix": self._residual_matrix(),
            "flow_matrix": self._flow_matrix(),
        }

    def solve_complete(self, source: str, sink: str):
        last_state = None
        for state in self.iteration_generator(source, sink):
            last_state = state
        if last_state is None:
            raise ValueError("Algoritmul nu a produs nicio iterație.")
        return last_state


class MatrixGrid(ttk.Frame):
    """Afișare tabelară pentru matrici și vectori."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.title_label = tk.Label(
            self,
            text=title,
            bg="#1e3a8a",
            fg="#ffffff",
            padx=12,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.title_label.pack(fill="x", pady=(0, 6))
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill="both", expand=True)
        self.widgets = []

    def clear(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.widgets = []

    def set_data(
        self,
        data,
        row_headers=None,
        col_headers=None,
        highlight_rows=None,
        highlight_cols=None,
        highlight_cells=None,
        fmt="{:.0f}",
    ):
        self.clear()
        highlight_rows = set(highlight_rows or [])
        highlight_cols = set(highlight_cols or [])
        highlight_cells = set(tuple(c) for c in (highlight_cells or []))

        if data is None:
            tk.Label(
                self.grid_frame,
                text="—",
                bg="#ffffff",
                fg="#64748b",
                width=12,
                padx=8,
                pady=6,
                relief="solid",
                bd=1,
            ).grid(row=0, column=0, sticky="w")
            return

        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        rows, cols = arr.shape
        if row_headers is None:
            row_headers = [str(i + 1) for i in range(rows)]
        if col_headers is None:
            col_headers = [str(j + 1) for j in range(cols)]

        # Latimea celulelor scade putin pentru matrici mari, dar ramane lizibila.
        cell_width = 11 if cols <= 8 else 10 if cols <= 12 else 9

        def make_cell(r, c, text, bg="#ffffff", fg="#0f172a", bold=False, width=None):
            lbl = tk.Label(
                self.grid_frame,
                text=text,
                bg=bg,
                fg=fg,
                relief="solid",
                bd=1,
                padx=7,
                pady=6,
                width=width if width is not None else cell_width,
                font=("Segoe UI", 9, "bold" if bold else "normal"),
            )
            lbl.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            return lbl

        make_cell(0, 0, "", bg="#dbeafe", fg="#1e3a8a", bold=True)
        for j, header in enumerate(col_headers, start=1):
            bg = "#d9f2d9" if (j - 1) in highlight_cols else "#dbeafe"
            make_cell(0, j, str(header), bg=bg, fg="#1e3a8a", bold=True)

        for i in range(rows):
            row_bg = "#f8d7da" if i in highlight_rows else "#eff6ff"
            make_cell(i + 1, 0, str(row_headers[i]), bg=row_bg, fg="#1e3a8a", bold=True)
            for j in range(cols):
                value = arr[i, j]
                fg = "#94a3b8" if abs(value) < 1e-9 else "#0f172a"
                cell_bg = "#ffffff"
                if (i, j) in highlight_cells:
                    cell_bg = "#fde68a"
                elif i in highlight_rows and j in highlight_cols:
                    cell_bg = "#f6d5b5"
                elif i in highlight_rows:
                    cell_bg = "#fdecef"
                elif j in highlight_cols:
                    cell_bg = "#eaf8ea"
                text = fmt.format(value) if abs(value) > 1e-9 else "0"
                make_cell(i + 1, j + 1, text, bg=cell_bg, fg=fg)

        for i in range(rows + 1):
            self.grid_frame.grid_rowconfigure(i, weight=1)
        for j in range(cols + 1):
            self.grid_frame.grid_columnconfigure(j, weight=1)


class FordFulkersonInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Ford-Fulkerson Pro - Flux maxim / Ediție Azure")
        self.root.geometry("1560x980")
        self.root.minsize(1320, 820)
        self.root.configure(bg="#eff6ff")

        self.solver = FlowNetworkSolver()
        self.problem_name = tk.StringVar(value="Problemă nouă")
        self.source_var = tk.StringVar(value="")
        self.sink_var = tk.StringVar(value="")

        self.generator = None
        self.current_state = None
        self.final_result = None
        self.current_data_signature = None
        self.finished = False
        self.layout_positions = {}

        # Folosit pentru a evita bucle dese de reconfigurare a canvasurilor cu scroll.
        self._process_scroll_update_pending = False
        self._process_last_canvas_size = None
        self._verify_scroll_update_pending = False
        self._verify_last_canvas_size = None
        self._final_check_scroll_update_pending = False
        self._final_check_last_canvas_size = None

        self._build_style()
        self._build_layout()
        self._refresh_node_view()
        self._refresh_edge_view()

    # ===================== Stil si layout =====================
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        azure_dark = "#1e3a8a"
        azure = "#2563eb"
        azure_soft = "#dbeafe"
        azure_bg = "#eff6ff"
        slate_text = "#334155"

        style.configure("TFrame", background=azure_bg)
        style.configure("TLabelframe", background=azure_bg, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=azure_bg, foreground=azure_dark, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=azure_bg, foreground=slate_text)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=azure_dark, background=azure_bg)
        style.configure("PanelTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground=azure_dark, background=azure_bg)
        style.configure("ResultBig.TLabel", font=("Segoe UI", 22, "bold"), foreground=azure, background=azure_bg)
        style.configure("Warn.TLabel", foreground="#b02a37", font=("Segoe UI", 11, "bold"), background=azure_bg)
        style.configure("Info.TLabel", foreground=slate_text, background=azure_bg)
        style.configure("Good.TLabel", foreground="#15803d", font=("Segoe UI", 10, "bold"), background=azure_bg)

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=azure,
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 7),
        )
        style.map(
            "Accent.TButton",
            background=[("active", azure_dark), ("pressed", azure_dark)],
            foreground=[("disabled", "#dbeafe")],
        )
        style.configure(
            "Json.TButton",
            font=("Segoe UI", 10, "bold"),
            background=azure_dark,
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 7),
        )
        style.map(
            "Json.TButton",
            background=[("active", "#1d4ed8"), ("pressed", "#1d4ed8")],
            foreground=[("disabled", "#dbeafe")],
        )
        style.configure(
            "Soft.TButton",
            font=("Segoe UI", 9),
            background=azure_soft,
            foreground=azure_dark,
            borderwidth=1,
            padding=(8, 6),
        )
        style.map("Soft.TButton", background=[("active", "#bfdbfe")])
        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#fee2e2",
            foreground="#b91c1c",
            borderwidth=1,
            padding=(8, 6),
        )
        style.map("Danger.TButton", background=[("active", "#fecaca")])

        style.configure("TNotebook", background=azure_bg, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
            background="#dbeafe",
            foreground=azure_dark,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#bfdbfe")],
            foreground=[("selected", azure_dark)],
        )

        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure("Treeview", rowheight=24, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background=azure_soft, foreground=azure_dark)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self.tab_input = ttk.Frame(notebook, padding=10)
        self.tab_process = ttk.Frame(notebook, padding=10)
        self.tab_result = ttk.Frame(notebook, padding=10)
        self.tab_verify = ttk.Frame(notebook, padding=10)
        self.tab_final_check = ttk.Frame(notebook, padding=10)

        notebook.add(self.tab_input, text="1. Configurare și date")
        notebook.add(self.tab_process, text="2. Proces iterativ")
        notebook.add(self.tab_result, text="3. Rezultate și analiză")
        notebook.add(self.tab_verify, text="4. Verificări și grafice")
        notebook.add(self.tab_final_check, text="5. Verificare finală")

        self._build_tab_input()
        self._build_tab_process()
        self._build_tab_result()
        self._build_tab_verify()
        self._build_tab_final_check()

    # ===================== Tab 1 - Date de intrare =====================
    def _build_tab_input(self):
        self.tab_input.columnconfigure(0, weight=1)
        self.tab_input.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self.tab_input, text="Configurarea problemei", padding=12)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(top, text="Denumirea problemei:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(top, textvariable=self.problem_name, width=24).grid(row=0, column=1, padx=5, pady=4, sticky="w")

        ttk.Label(top, text="Sursă:").grid(row=0, column=2, padx=(14, 5), pady=4, sticky="w")
        self.source_combo = ttk.Combobox(top, textvariable=self.source_var, width=12, state="readonly")
        self.source_combo.grid(row=0, column=3, padx=5, pady=4, sticky="w")

        ttk.Label(top, text="Destinație:").grid(row=0, column=4, padx=(14, 5), pady=4, sticky="w")
        self.sink_combo = ttk.Combobox(top, textvariable=self.sink_var, width=12, state="readonly")
        self.sink_combo.grid(row=0, column=5, padx=5, pady=4, sticky="w")

        ttk.Button(top, text="Încarcă JSON", command=self.load_from_json, style="Json.TButton").grid(row=0, column=6, padx=(14, 6), pady=4, sticky="w")
        ttk.Button(top, text="Salvează JSON", command=self.save_to_json, style="Json.TButton").grid(row=0, column=7, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Exemplu București", command=self.load_example_bucharest, style="Soft.TButton").grid(row=0, column=8, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Exemplu simplu", command=self.load_example_simple, style="Soft.TButton").grid(row=0, column=9, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Pas următor", style="Accent.TButton", command=self.next_step).grid(row=0, column=10, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Rezolvă complet", style="Accent.TButton", command=self.solve_complete).grid(row=0, column=11, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Resetează rularea", command=self.reset_run_state, style="Soft.TButton").grid(row=0, column=12, padx=6, pady=4, sticky="w")
        ttk.Button(top, text="Șterge problema curentă", command=self.delete_current_problem_action, style="Danger.TButton").grid(row=0, column=13, padx=6, pady=4, sticky="w")

        main = ttk.Frame(self.tab_input)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        nodes_box = ttk.LabelFrame(left, text="Noduri (vârfurile rețelei)", padding=8)
        nodes_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        nodes_box.columnconfigure(0, weight=1)
        nodes_box.rowconfigure(0, weight=1)

        self.nodes_tree = ttk.Treeview(nodes_box, columns=("id", "name", "in_deg", "out_deg"), show="headings", height=8)
        for col, text, width in [("id", "ID", 90), ("name", "Denumire afișată", 220), ("in_deg", "Grad intrare", 90), ("out_deg", "Grad ieșire", 90)]:
            self.nodes_tree.heading(col, text=text)
            self.nodes_tree.column(col, width=width, anchor="w" if col == "name" else "center")
        self.nodes_tree.grid(row=0, column=0, columnspan=5, sticky="nsew")
        node_scroll = ttk.Scrollbar(nodes_box, orient="vertical", command=self.nodes_tree.yview)
        node_scroll.grid(row=0, column=5, sticky="ns")
        self.nodes_tree.configure(yscrollcommand=node_scroll.set)

        ttk.Label(nodes_box, text="ID:").grid(row=1, column=0, padx=(0, 4), pady=(8, 0), sticky="e")
        self.new_node_id = tk.StringVar()
        ttk.Entry(nodes_box, textvariable=self.new_node_id, width=12).grid(row=1, column=1, padx=(0, 8), pady=(8, 0), sticky="w")
        ttk.Label(nodes_box, text="Denumire:").grid(row=1, column=2, padx=(0, 4), pady=(8, 0), sticky="e")
        self.new_node_name = tk.StringVar()
        ttk.Entry(nodes_box, textvariable=self.new_node_name, width=24).grid(row=1, column=3, padx=(0, 8), pady=(8, 0), sticky="w")
        ttk.Button(nodes_box, text="Adaugă nod", command=self.add_node_action, style="Accent.TButton").grid(row=1, column=4, padx=4, pady=(8, 0), sticky="w")
        ttk.Button(nodes_box, text="Șterge nodul selectat", command=self.remove_selected_node, style="Danger.TButton").grid(row=1, column=5, padx=4, pady=(8, 0), sticky="w")

        edges_box = ttk.LabelFrame(left, text="Arce (muchii orientate) și capacități", padding=8)
        edges_box.grid(row=1, column=0, sticky="nsew")
        edges_box.columnconfigure(0, weight=1)
        edges_box.rowconfigure(0, weight=1)

        self.edges_tree = ttk.Treeview(edges_box, columns=("u", "v", "cap"), show="headings", height=10)
        for col, text, width in [("u", "De la", 200), ("v", "Către", 200), ("cap", "Capacitate", 130)]:
            self.edges_tree.heading(col, text=text)
            self.edges_tree.column(col, width=width, anchor="w" if col != "cap" else "e")
        self.edges_tree.grid(row=0, column=0, columnspan=7, sticky="nsew")
        edge_scroll = ttk.Scrollbar(edges_box, orient="vertical", command=self.edges_tree.yview)
        edge_scroll.grid(row=0, column=7, sticky="ns")
        self.edges_tree.configure(yscrollcommand=edge_scroll.set)

        ttk.Label(edges_box, text="De la:").grid(row=1, column=0, padx=(0, 4), pady=(8, 0), sticky="e")
        self.new_edge_from = tk.StringVar()
        self.edge_from_combo = ttk.Combobox(edges_box, textvariable=self.new_edge_from, width=12, state="readonly")
        self.edge_from_combo.grid(row=1, column=1, padx=(0, 8), pady=(8, 0), sticky="w")

        ttk.Label(edges_box, text="Către:").grid(row=1, column=2, padx=(0, 4), pady=(8, 0), sticky="e")
        self.new_edge_to = tk.StringVar()
        self.edge_to_combo = ttk.Combobox(edges_box, textvariable=self.new_edge_to, width=12, state="readonly")
        self.edge_to_combo.grid(row=1, column=3, padx=(0, 8), pady=(8, 0), sticky="w")

        ttk.Label(edges_box, text="Capacitate:").grid(row=1, column=4, padx=(0, 4), pady=(8, 0), sticky="e")
        self.new_edge_cap = tk.StringVar()
        ttk.Entry(edges_box, textvariable=self.new_edge_cap, width=10).grid(row=1, column=5, padx=(0, 8), pady=(8, 0), sticky="w")
        ttk.Button(edges_box, text="Adaugă arc", command=self.add_edge_action, style="Accent.TButton").grid(row=1, column=6, padx=4, pady=(8, 0), sticky="w")
        ttk.Button(edges_box, text="Modifică capacitatea", command=self.edit_edge_capacity, style="Soft.TButton").grid(row=2, column=4, columnspan=2, padx=4, pady=(6, 0), sticky="we")
        ttk.Button(edges_box, text="Șterge arcul selectat", command=self.remove_selected_edge, style="Danger.TButton").grid(row=2, column=6, padx=4, pady=(6, 0), sticky="we")

        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        side.rowconfigure(2, weight=1)
        side.columnconfigure(0, weight=1)

        guide = ttk.LabelFrame(side, text="Ajutor", padding=8)
        guide.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        help_text = (
            "• Adăugați mai întâi nodurile (ID + denumire afișată), apoi arcele cu capacități pozitive.\n"
            "• Fiecare arc pleacă dintr-un nod și ajunge în alt nod, având o capacitate întreagă.\n"
            "• Alegeți sursa și destinația din liste; acestea nu pot coincide.\n"
            "• «Pas următor» rulează o iterație BFS pentru drumul de augmentare; «Rezolvă complet» rulează toate iterațiile.\n"
            "• Notație: C = matricea capacităților, R = matricea reziduală, F = matricea fluxului; F* = flux maxim."
        )
        ttk.Label(guide, text=help_text, style="Info.TLabel", justify="left").pack(anchor="w")

        std_box = ttk.LabelFrame(side, text="Previzualizarea stării curente", padding=8)
        std_box.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        std_box.columnconfigure(0, weight=1)
        std_box.rowconfigure(1, weight=1)
        self.preview_label = ttk.Label(std_box, text="Nu a fost rulată încă nicio iterație.", style="Info.TLabel")
        self.preview_label.grid(row=0, column=0, sticky="w")
        self.preview_text = ScrolledText(std_box, height=12, wrap="word", font=("Consolas", 9))
        self.preview_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.preview_text.configure(state="disabled")

        log_box = ttk.LabelFrame(side, text="Jurnal tehnic", padding=8)
        log_box.grid(row=2, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_box, height=14, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    # ===================== Tab 2 - Proces iterativ =====================
    def _build_tab_process(self):
        self.tab_process.rowconfigure(0, weight=1)
        self.tab_process.columnconfigure(0, weight=1)

        self.process_canvas = tk.Canvas(
            self.tab_process,
            background="#eff6ff",
            highlightthickness=0,
            xscrollincrement=25,
            yscrollincrement=25,
        )
        self.process_canvas.grid(row=0, column=0, sticky="nsew")
        process_vbar = ttk.Scrollbar(self.tab_process, orient="vertical", command=self.process_canvas.yview)
        process_vbar.grid(row=0, column=1, sticky="ns")
        process_hbar = ttk.Scrollbar(self.tab_process, orient="horizontal", command=self.process_canvas.xview)
        process_hbar.grid(row=1, column=0, sticky="ew")
        self.process_canvas.configure(yscrollcommand=process_vbar.set, xscrollcommand=process_hbar.set)

        self.process_inner = ttk.Frame(self.process_canvas, padding=10)
        self.process_window = self.process_canvas.create_window((0, 0), window=self.process_inner, anchor="nw")
        self.process_inner.bind("<Configure>", self._on_process_configure)
        self.process_canvas.bind("<Configure>", self._on_process_canvas_configure)
        self.process_canvas.bind("<MouseWheel>", self._on_process_mousewheel)
        self.process_canvas.bind("<Shift-MouseWheel>", self._on_process_shift_mousewheel)
        self.process_canvas.bind("<Button-4>", lambda _e: self.process_canvas.yview_scroll(-3, "units"))
        self.process_canvas.bind("<Button-5>", lambda _e: self.process_canvas.yview_scroll(3, "units"))

        self.process_inner.columnconfigure(0, weight=3, minsize=1500)
        self.process_inner.columnconfigure(1, weight=1, minsize=760)
        self.process_inner.rowconfigure(1, weight=1, minsize=900)

        summary = ttk.LabelFrame(self.process_inner, text="Rezumatul iterației", padding=10)
        summary.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        summary.columnconfigure(0, weight=1)
        self.iteration_summary = ttk.Label(summary, text="Nu există încă date de iterație.", style="Info.TLabel")
        self.iteration_summary.grid(row=0, column=0, sticky="w")

        left = ttk.Frame(self.process_inner)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        upper = ttk.Frame(left)
        upper.grid(row=0, column=0, sticky="nsew")
        upper.columnconfigure(0, weight=1, minsize=700)
        upper.columnconfigure(1, weight=1, minsize=700)
        upper.rowconfigure(0, weight=1)

        lower = ttk.Frame(left)
        lower.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        lower.columnconfigure(0, weight=1, minsize=700)
        lower.columnconfigure(1, weight=1, minsize=700)
        lower.rowconfigure(0, weight=1)

        self.table_capacity = MatrixGrid(upper, "Matricea capacităților C")
        self.table_capacity.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.table_residual_before = MatrixGrid(upper, "Matricea reziduală R (înainte de actualizare)")
        self.table_residual_before.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.table_residual_after = MatrixGrid(lower, "Matricea reziduală R (după actualizare)")
        self.table_residual_after.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.table_flow_after = MatrixGrid(lower, "Matricea fluxului F (curentă)")
        self.table_flow_after.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        right = ttk.Frame(self.process_inner)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2, minsize=540)
        right.rowconfigure(1, weight=1, minsize=260)

        plot_box = ttk.LabelFrame(right, text="Rețeaua cu drumul de augmentare", padding=6)
        plot_box.grid(row=0, column=0, sticky="nsew")
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)

        self.process_figure = Figure(figsize=(7.6, 5.7), dpi=100, constrained_layout=True)
        self.process_ax = self.process_figure.add_subplot(111)
        self.process_ax.set_axis_off()
        self.process_canvas_plot = FigureCanvasTkAgg(self.process_figure, master=plot_box)
        self.process_canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        path_box = ttk.LabelFrame(right, text="Drumul de augmentare și urmărirea BFS", padding=6)
        path_box.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        path_box.columnconfigure(0, weight=1)
        path_box.rowconfigure(0, weight=1)
        self.path_text = ScrolledText(path_box, height=8, wrap="word", font=("Consolas", 9))
        self.path_text.grid(row=0, column=0, sticky="nsew")
        self.path_text.configure(state="disabled")

    # ===================== Tab 3 - Rezultate =====================
    def _build_tab_result(self):
        self.tab_result.columnconfigure(0, weight=2)
        self.tab_result.columnconfigure(1, weight=2)
        self.tab_result.rowconfigure(1, weight=1)

        top = ttk.Frame(self.tab_result)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Flux maxim F*:", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.flow_big_label = ttk.Label(top, text="—", style="ResultBig.TLabel")
        self.flow_big_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.result_status_label = ttk.Label(top, text="Nu există încă o rezolvare completă.", style="Info.TLabel")
        self.result_status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        flows_box = ttk.LabelFrame(self.tab_result, text="Fluxul pe fiecare arc", padding=8)
        flows_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        flows_box.columnconfigure(0, weight=1)
        flows_box.rowconfigure(0, weight=1)
        self.flows_tree = ttk.Treeview(flows_box, columns=("arc", "flow", "cap", "util", "status"), show="headings")
        for col, text, width in [("arc", "Arc", 220), ("flow", "Flux", 90), ("cap", "Capacitate", 100), ("util", "Utilizare", 110), ("status", "Stare", 110)]:
            self.flows_tree.heading(col, text=text)
            anchor = "w" if col == "arc" else "e" if col in {"flow", "cap"} else "center"
            self.flows_tree.column(col, width=width, anchor=anchor)
        self.flows_tree.grid(row=0, column=0, sticky="nsew")
        flow_scroll = ttk.Scrollbar(flows_box, orient="vertical", command=self.flows_tree.yview)
        flow_scroll.grid(row=0, column=1, sticky="ns")
        self.flows_tree.configure(yscrollcommand=flow_scroll.set)

        cut_box = ttk.LabelFrame(self.tab_result, text="Tăietura minimă (teorema Max-Flow Min-Cut)", padding=8)
        cut_box.grid(row=1, column=1, sticky="nsew")
        cut_box.columnconfigure(0, weight=1)
        cut_box.rowconfigure(2, weight=1)

        self.cut_summary = ttk.Label(cut_box, text="Nu există încă o tăietură minimă.", style="Info.TLabel", justify="left")
        self.cut_summary.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.cut_tree = ttk.Treeview(cut_box, columns=("u", "v", "cap"), show="headings")
        for col, text, width in [("u", "Din S", 180), ("v", "Către T", 180), ("cap", "Capacitate", 110)]:
            self.cut_tree.heading(col, text=text)
            anchor = "w" if col != "cap" else "e"
            self.cut_tree.column(col, width=width, anchor=anchor)
        self.cut_tree.grid(row=2, column=0, sticky="nsew")
        cut_scroll = ttk.Scrollbar(cut_box, orient="vertical", command=self.cut_tree.yview)
        cut_scroll.grid(row=2, column=1, sticky="ns")
        self.cut_tree.configure(yscrollcommand=cut_scroll.set)

    # ===================== Tab 4 - Verificare =====================
    def _build_tab_verify(self):
        """Tab 4 este pus într-un canvas cu scroll pe ambele direcții.

        Motivul: tabelul final F* și graful cu tăietura minimă sunt late și înalte
        pentru exemple cu multe noduri. Dacă le forțăm să intre în viewport, Tkinter
        le comprimă, iar graficele Matplotlib devin turtite și greu de citit.
        """
        self.tab_verify.rowconfigure(0, weight=1)
        self.tab_verify.columnconfigure(0, weight=1)

        self.verify_canvas = tk.Canvas(
            self.tab_verify,
            background="#eff6ff",
            highlightthickness=0,
            xscrollincrement=25,
            yscrollincrement=25,
        )
        self.verify_canvas.grid(row=0, column=0, sticky="nsew")

        verify_vbar = ttk.Scrollbar(self.tab_verify, orient="vertical", command=self.verify_canvas.yview)
        verify_vbar.grid(row=0, column=1, sticky="ns")
        verify_hbar = ttk.Scrollbar(self.tab_verify, orient="horizontal", command=self.verify_canvas.xview)
        verify_hbar.grid(row=1, column=0, sticky="ew")
        self.verify_canvas.configure(yscrollcommand=verify_vbar.set, xscrollcommand=verify_hbar.set)

        self.verify_inner = ttk.Frame(self.verify_canvas, padding=10)
        self.verify_window = self.verify_canvas.create_window((0, 0), window=self.verify_inner, anchor="nw")
        self.verify_inner.bind("<Configure>", self._on_verify_configure)
        self.verify_canvas.bind("<Configure>", self._on_verify_canvas_configure)
        self.verify_canvas.bind("<MouseWheel>", self._on_verify_mousewheel)
        self.verify_canvas.bind("<Shift-MouseWheel>", self._on_verify_shift_mousewheel)
        self.verify_canvas.bind("<Button-4>", lambda _e: self.verify_canvas.yview_scroll(-3, "units"))
        self.verify_canvas.bind("<Button-5>", lambda _e: self.verify_canvas.yview_scroll(3, "units"))

        self.verify_inner.columnconfigure(0, weight=0, minsize=720)
        self.verify_inner.columnconfigure(1, weight=0, minsize=1520)
        self.verify_inner.rowconfigure(1, weight=0, minsize=430)
        self.verify_inner.rowconfigure(2, weight=0, minsize=660)

        info = ttk.LabelFrame(self.verify_inner, text="Verificare matematică", padding=8)
        info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.verify_label = ttk.Label(info, text="Nu există încă verificări.", style="Info.TLabel", justify="left")
        self.verify_label.pack(anchor="w")

        cons_box = ttk.LabelFrame(self.verify_inner, text="Conservarea fluxului în nodurile interioare", padding=8)
        cons_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        cons_box.columnconfigure(0, weight=1)
        cons_box.rowconfigure(0, weight=1)
        self.conservation_tree = ttk.Treeview(
            cons_box,
            columns=("node", "in", "out", "diff", "status"),
            show="headings",
            height=13,
        )
        for col, text, width in [
            ("node", "Nod", 230),
            ("in", "Intrare", 100),
            ("out", "Ieșire", 100),
            ("diff", "Dif.", 80),
            ("status", "Stare", 115),
        ]:
            self.conservation_tree.heading(col, text=text)
            anchor = "w" if col == "node" else "e" if col in {"in", "out", "diff"} else "center"
            self.conservation_tree.column(col, width=width, anchor=anchor)
        self.conservation_tree.grid(row=0, column=0, sticky="nsew")
        cons_scroll = ttk.Scrollbar(cons_box, orient="vertical", command=self.conservation_tree.yview)
        cons_scroll.grid(row=0, column=1, sticky="ns")
        self.conservation_tree.configure(yscrollcommand=cons_scroll.set)

        flow_box = ttk.LabelFrame(self.verify_inner, text="Matricea finală a fluxului F*", padding=8)
        flow_box.grid(row=1, column=1, sticky="nsew")
        flow_box.columnconfigure(0, weight=1, minsize=1500)
        flow_box.rowconfigure(0, weight=1, minsize=400)
        self.table_final_flow = MatrixGrid(flow_box, "Fluxul F = C - R*")
        self.table_final_flow.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self.verify_inner)
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        bottom.columnconfigure(0, weight=0, minsize=1050)
        bottom.columnconfigure(1, weight=0, minsize=1150)
        bottom.rowconfigure(0, weight=0, minsize=640)

        chart_box = ttk.LabelFrame(bottom, text="Convergența fluxului maxim", padding=8)
        chart_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        chart_box.rowconfigure(0, weight=1)
        chart_box.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10.2, 5.9), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Convergența monotonă a fluxului total")
        self.ax.set_xlabel("Iterație")
        self.ax.set_ylabel("Flux cumulativ")
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=chart_box)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        cut_plot_box = ttk.LabelFrame(bottom, text="Rețeaua cu tăietura minimă", padding=8)
        cut_plot_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        cut_plot_box.rowconfigure(0, weight=1)
        cut_plot_box.columnconfigure(0, weight=1)

        self.cut_figure = Figure(figsize=(11.2, 5.9), dpi=100, constrained_layout=True)
        self.cut_ax = self.cut_figure.add_subplot(111)
        self.cut_ax.set_axis_off()
        self.cut_canvas_plot = FigureCanvasTkAgg(self.cut_figure, master=cut_plot_box)
        self.cut_canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._schedule_verify_scroll_update()

    # ===================== Tab 5 - Verificarea finală a soluției =====================
    def _build_tab_final_check(self):
        """Tab separat pentru cele 3 verificări finale ale soluției.

        Verificările sunt prezentate explicit, ca la finalul algoritmului:
        1) echilibrul sursă/destinație față de F*;
        2) conservarea fluxului în nodurile interioare;
        3) egalitatea Max-Flow = Min-Cut.
        """
        self.tab_final_check.rowconfigure(0, weight=1)
        self.tab_final_check.columnconfigure(0, weight=1)

        self.final_check_canvas = tk.Canvas(
            self.tab_final_check,
            background="#eff6ff",
            highlightthickness=0,
            xscrollincrement=25,
            yscrollincrement=25,
        )
        self.final_check_canvas.grid(row=0, column=0, sticky="nsew")

        final_vbar = ttk.Scrollbar(
            self.tab_final_check,
            orient="vertical",
            command=self.final_check_canvas.yview,
        )
        final_vbar.grid(row=0, column=1, sticky="ns")
        final_hbar = ttk.Scrollbar(
            self.tab_final_check,
            orient="horizontal",
            command=self.final_check_canvas.xview,
        )
        final_hbar.grid(row=1, column=0, sticky="ew")
        self.final_check_canvas.configure(
            yscrollcommand=final_vbar.set,
            xscrollcommand=final_hbar.set,
        )

        self.final_check_inner = ttk.Frame(self.final_check_canvas, padding=12)
        self.final_check_window = self.final_check_canvas.create_window(
            (0, 0),
            window=self.final_check_inner,
            anchor="nw",
        )
        self.final_check_inner.bind("<Configure>", self._on_final_check_configure)
        self.final_check_canvas.bind("<Configure>", self._on_final_check_canvas_configure)
        self.final_check_canvas.bind("<MouseWheel>", self._on_final_check_mousewheel)
        self.final_check_canvas.bind("<Shift-MouseWheel>", self._on_final_check_shift_mousewheel)
        self.final_check_canvas.bind("<Button-4>", lambda _e: self.final_check_canvas.yview_scroll(-3, "units"))
        self.final_check_canvas.bind("<Button-5>", lambda _e: self.final_check_canvas.yview_scroll(3, "units"))

        self.final_check_inner.columnconfigure(0, weight=0, minsize=620)
        self.final_check_inner.columnconfigure(1, weight=0, minsize=620)
        self.final_check_inner.columnconfigure(2, weight=0, minsize=620)
        self.final_check_inner.rowconfigure(2, weight=0, minsize=430)
        self.final_check_inner.rowconfigure(3, weight=0, minsize=320)

        title_box = ttk.LabelFrame(self.final_check_inner, text="Verificarea finală a soluției", padding=10)
        title_box.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        title_box.columnconfigure(0, weight=1)
        self.final_check_title = ttk.Label(
            title_box,
            text="Apasă 'Rezolvă complet' pentru a obține soluția finală și verificările.",
            style="Info.TLabel",
            justify="left",
        )
        self.final_check_title.grid(row=0, column=0, sticky="w")

        # Check 1: source/sink balance.
        balance_box = ttk.LabelFrame(self.final_check_inner, text="1. Echilibrul sursă / destinație", padding=10)
        balance_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        balance_box.columnconfigure(0, weight=1)
        self.check_balance_label = ttk.Label(
            balance_box,
            text="Nu există încă o soluție finală.",
            style="Info.TLabel",
            justify="left",
        )
        self.check_balance_label.grid(row=0, column=0, sticky="w")

        # Check 2: flow conservation.
        conservation_box = ttk.LabelFrame(self.final_check_inner, text="2. Conservarea fluxului", padding=10)
        conservation_box.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 10))
        conservation_box.columnconfigure(0, weight=1)
        self.check_conservation_label = ttk.Label(
            conservation_box,
            text="Nu există încă o soluție finală.",
            style="Info.TLabel",
            justify="left",
        )
        self.check_conservation_label.grid(row=0, column=0, sticky="w")

        # Check 3: max-flow min-cut.
        mincut_box = ttk.LabelFrame(self.final_check_inner, text="3. Max-Flow = Min-Cut", padding=10)
        mincut_box.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(0, 10))
        mincut_box.columnconfigure(0, weight=1)
        self.check_mincut_label = ttk.Label(
            mincut_box,
            text="Nu există încă o soluție finală.",
            style="Info.TLabel",
            justify="left",
        )
        self.check_mincut_label.grid(row=0, column=0, sticky="w")

        # Detailed table for the interior-node conservation check.
        cons_detail_box = ttk.LabelFrame(self.final_check_inner, text="Tabel detaliat pentru conservare", padding=8)
        cons_detail_box.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        cons_detail_box.columnconfigure(0, weight=1)
        cons_detail_box.rowconfigure(0, weight=1)
        self.final_conservation_tree = ttk.Treeview(
            cons_detail_box,
            columns=("node", "in", "out", "diff", "status"),
            show="headings",
            height=15,
        )
        for col, text_, width in [
            ("node", "Nod", 220),
            ("in", "Intrare", 90),
            ("out", "Ieșire", 90),
            ("diff", "Dif.", 80),
            ("status", "Stare", 105),
        ]:
            self.final_conservation_tree.heading(col, text=text_)
            anchor = "w" if col == "node" else "e" if col in {"in", "out", "diff"} else "center"
            self.final_conservation_tree.column(col, width=width, anchor=anchor)
        self.final_conservation_tree.grid(row=0, column=0, sticky="nsew")
        cons_detail_scroll = ttk.Scrollbar(cons_detail_box, orient="vertical", command=self.final_conservation_tree.yview)
        cons_detail_scroll.grid(row=0, column=1, sticky="ns")
        self.final_conservation_tree.configure(yscrollcommand=cons_detail_scroll.set)

        # Detailed table for the min-cut check.
        cut_detail_box = ttk.LabelFrame(self.final_check_inner, text="Arcele tăieturii minime", padding=8)
        cut_detail_box.grid(row=2, column=1, sticky="nsew", padx=8)
        cut_detail_box.columnconfigure(0, weight=1)
        cut_detail_box.rowconfigure(0, weight=1)
        self.final_cut_tree = ttk.Treeview(
            cut_detail_box,
            columns=("u", "v", "cap"),
            show="headings",
            height=15,
        )
        for col, text_, width in [
            ("u", "Din S", 210),
            ("v", "Către T", 210),
            ("cap", "Capacitate", 110),
        ]:
            self.final_cut_tree.heading(col, text=text_)
            self.final_cut_tree.column(col, width=width, anchor="e" if col == "cap" else "w")
        self.final_cut_tree.grid(row=0, column=0, sticky="nsew")
        final_cut_scroll = ttk.Scrollbar(cut_detail_box, orient="vertical", command=self.final_cut_tree.yview)
        final_cut_scroll.grid(row=0, column=1, sticky="ns")
        self.final_cut_tree.configure(yscrollcommand=final_cut_scroll.set)

        # Tabel suport pentru arce: arată respectarea capacității și arcele saturate.
        arc_detail_box = ttk.LabelFrame(self.final_check_inner, text="Detalii suport: fluxul final pe arce", padding=8)
        arc_detail_box.grid(row=2, column=2, sticky="nsew", padx=(8, 0))
        arc_detail_box.columnconfigure(0, weight=1)
        arc_detail_box.rowconfigure(0, weight=1)
        self.final_arc_tree = ttk.Treeview(
            arc_detail_box,
            columns=("arc", "flow", "cap", "residual", "status"),
            show="headings",
            height=15,
        )
        for col, text_, width in [
            ("arc", "Arc", 260),
            ("flow", "Flux", 80),
            ("cap", "Cap.", 80),
            ("residual", "Rezidual", 90),
            ("status", "Stare", 105),
        ]:
            self.final_arc_tree.heading(col, text=text_)
            anchor = "w" if col == "arc" else "e" if col in {"flow", "cap", "residual"} else "center"
            self.final_arc_tree.column(col, width=width, anchor=anchor)
        self.final_arc_tree.grid(row=0, column=0, sticky="nsew")
        final_arc_scroll = ttk.Scrollbar(arc_detail_box, orient="vertical", command=self.final_arc_tree.yview)
        final_arc_scroll.grid(row=0, column=1, sticky="ns")
        self.final_arc_tree.configure(yscrollcommand=final_arc_scroll.set)

        explanation_box = ttk.LabelFrame(self.final_check_inner, text="Interpretare", padding=10)
        explanation_box.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        explanation_box.columnconfigure(0, weight=1)
        explanation_box.rowconfigure(0, weight=1)
        self.final_check_text = ScrolledText(explanation_box, height=9, wrap="word", font=("Consolas", 9))
        self.final_check_text.grid(row=0, column=0, sticky="nsew")
        self.final_check_text.configure(state="disabled")

        self._schedule_final_check_scroll_update()

    # ===================== Funcții ajutătoare pentru scroll =====================
    def _on_process_configure(self, _event=None):
        self._schedule_process_scroll_update()

    def _on_process_canvas_configure(self, _event=None):
        self._schedule_process_scroll_update()

    def _on_process_mousewheel(self, event):
        self.process_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_process_shift_mousewheel(self, event):
        self.process_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _schedule_process_scroll_update(self):
        if self._process_scroll_update_pending:
            return
        self._process_scroll_update_pending = True
        self.root.after_idle(self._update_process_scroll_region)

    def _matrix_pair_min_width(self):
        n = max(len(self.solver.node_ids), 8)
        one_matrix_width = max(700, (n + 1) * 92)
        return 2 * one_matrix_width + 40

    def _process_required_size(self):
        left_w = self._matrix_pair_min_width()
        right_w = 780
        total_w = left_w + right_w + 50
        total_h = 1010
        return total_w, total_h, left_w, right_w

    def _update_process_scroll_region(self):
        self._process_scroll_update_pending = False
        if not hasattr(self, "process_canvas"):
            return

        self.process_canvas.update_idletasks()
        canvas_w = max(self.process_canvas.winfo_width(), 1)
        canvas_h = max(self.process_canvas.winfo_height(), 1)
        needed_w, needed_h, left_w, right_w = self._process_required_size()

        self.process_inner.columnconfigure(0, weight=3, minsize=left_w)
        self.process_inner.columnconfigure(1, weight=1, minsize=right_w)

        req_w = max(self.process_inner.winfo_reqwidth(), needed_w, canvas_w)
        req_h = max(self.process_inner.winfo_reqheight(), needed_h, canvas_h)
        new_size = (req_w, req_h)

        if self._process_last_canvas_size != new_size:
            self.process_canvas.itemconfigure(self.process_window, width=req_w, height=req_h)
            self.process_canvas.configure(scrollregion=(0, 0, req_w, req_h))
            self._process_last_canvas_size = new_size
        else:
            self.process_canvas.configure(scrollregion=(0, 0, req_w, req_h))


    def _on_verify_configure(self, _event=None):
        self._schedule_verify_scroll_update()

    def _on_verify_canvas_configure(self, _event=None):
        self._schedule_verify_scroll_update()

    def _on_verify_mousewheel(self, event):
        self.verify_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_verify_shift_mousewheel(self, event):
        self.verify_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _schedule_verify_scroll_update(self):
        if self._verify_scroll_update_pending:
            return
        self._verify_scroll_update_pending = True
        self.root.after_idle(self._update_verify_scroll_region)

    def _verify_required_size(self):
        n = max(len(self.solver.node_ids), 8)
        matrix_w = max(1500, (n + 1) * 115)
        left_w = 720
        right_w = matrix_w
        chart_w = 1050
        cut_w = max(1150, n * 115)
        total_w = max(left_w + right_w + 60, chart_w + cut_w + 50)
        total_h = 1160
        return total_w, total_h, left_w, right_w, chart_w, cut_w

    def _update_verify_scroll_region(self):
        self._verify_scroll_update_pending = False
        if not hasattr(self, "verify_canvas"):
            return

        self.verify_canvas.update_idletasks()
        canvas_w = max(self.verify_canvas.winfo_width(), 1)
        canvas_h = max(self.verify_canvas.winfo_height(), 1)
        needed_w, needed_h, left_w, right_w, chart_w, cut_w = self._verify_required_size()

        self.verify_inner.columnconfigure(0, weight=0, minsize=left_w)
        self.verify_inner.columnconfigure(1, weight=0, minsize=right_w)

        # Actualizam si zonele grafice pentru ca Matplotlib sa aiba suficient spatiu.
        for child in self.verify_inner.winfo_children():
            try:
                if str(child).endswith(".!frame"):
                    pass
            except Exception:
                pass

        req_w = max(self.verify_inner.winfo_reqwidth(), needed_w, canvas_w)
        req_h = max(self.verify_inner.winfo_reqheight(), needed_h, canvas_h)
        new_size = (req_w, req_h)

        if self._verify_last_canvas_size != new_size:
            self.verify_canvas.itemconfigure(self.verify_window, width=req_w, height=req_h)
            self.verify_canvas.configure(scrollregion=(0, 0, req_w, req_h))
            self._verify_last_canvas_size = new_size
        else:
            self.verify_canvas.configure(scrollregion=(0, 0, req_w, req_h))

    def _on_final_check_configure(self, _event=None):
        self._schedule_final_check_scroll_update()

    def _on_final_check_canvas_configure(self, _event=None):
        self._schedule_final_check_scroll_update()

    def _on_final_check_mousewheel(self, event):
        self.final_check_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_final_check_shift_mousewheel(self, event):
        self.final_check_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def _schedule_final_check_scroll_update(self):
        if self._final_check_scroll_update_pending:
            return
        self._final_check_scroll_update_pending = True
        self.root.after_idle(self._update_final_check_scroll_region)

    def _final_check_required_size(self):
        n = max(len(self.solver.node_ids), 8)
        col_w = max(620, n * 70)
        total_w = 3 * col_w + 90
        total_h = 880
        return total_w, total_h, col_w

    def _update_final_check_scroll_region(self):
        self._final_check_scroll_update_pending = False
        if not hasattr(self, "final_check_canvas"):
            return

        self.final_check_canvas.update_idletasks()
        canvas_w = max(self.final_check_canvas.winfo_width(), 1)
        canvas_h = max(self.final_check_canvas.winfo_height(), 1)
        needed_w, needed_h, col_w = self._final_check_required_size()

        for col in range(3):
            self.final_check_inner.columnconfigure(col, weight=0, minsize=col_w)

        req_w = max(self.final_check_inner.winfo_reqwidth(), needed_w, canvas_w)
        req_h = max(self.final_check_inner.winfo_reqheight(), needed_h, canvas_h)
        new_size = (req_w, req_h)

        if self._final_check_last_canvas_size != new_size:
            self.final_check_canvas.itemconfigure(self.final_check_window, width=req_w, height=req_h)
            self.final_check_canvas.configure(scrollregion=(0, 0, req_w, req_h))
            self._final_check_last_canvas_size = new_size
        else:
            self.final_check_canvas.configure(scrollregion=(0, 0, req_w, req_h))

    # ===================== Helpers - text widgets =====================
    def clear_text_widget(self, widget):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.configure(state="disabled")

    def set_text_widget(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def _clear_tree(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    # ===================== Actiuni pe noduri =====================
    def add_node_action(self):
        nid = self.new_node_id.get().strip()
        name = self.new_node_name.get().strip()
        if not nid:
            messagebox.showwarning("ID lipsă", "Introduceți un ID de nod nevid.")
            return
        if not self.solver.add_node(nid, name):
            messagebox.showerror("ID duplicat", f"Există deja un nod cu ID-ul '{nid}'.")
            return
        self.append_log(f"S-a adăugat nodul '{nid}' ({self.solver.nodes[nid]}).")
        self.new_node_id.set("")
        self.new_node_name.set("")
        self._on_graph_changed()

    def remove_selected_node(self):
        selected = self.nodes_tree.selection()
        if not selected:
            messagebox.showinfo("Nicio selecție", "Selectați un nod de șters.")
            return
        nid = self.nodes_tree.item(selected[0], "values")[0]
        if self.solver.remove_node(nid):
            self.append_log(f"S-a șters nodul '{nid}' și toate arcele incidente.")
            self._on_graph_changed()

    def add_edge_action(self):
        u = self.new_edge_from.get().strip()
        v = self.new_edge_to.get().strip()
        cap_raw = self.new_edge_cap.get().strip()
        if not u or not v:
            messagebox.showwarning("Noduri lipsă", "Alegeți ambele capete ale arcului.")
            return
        try:
            cap = int(cap_raw)
        except ValueError:
            messagebox.showerror("Capacitate invalidă", "Capacitatea trebuie să fie un număr întreg pozitiv.")
            return
        err = self.solver.add_edge(u, v, cap)
        if err:
            messagebox.showerror("Adăugarea arcului a eșuat", err)
            return
        self.append_log(f"S-a adăugat arcul {u} → {v} cu capacitatea {cap}.")
        self.new_edge_cap.set("")
        self._on_graph_changed()

    def edit_edge_capacity(self):
        selected = self.edges_tree.selection()
        if not selected:
            messagebox.showinfo("Nicio selecție", "Selectați un arc pentru a-i modifica capacitatea.")
            return
        values = self.edges_tree.item(selected[0], "values")
        u_label, v_label, cap = values
        u_id = self._extract_id_from_label(u_label)
        v_id = self._extract_id_from_label(v_label)
        if not u_id or not v_id:
            messagebox.showerror("Modificarea a eșuat", "Nu s-a putut interpreta arcul selectat.")
            return
        new_cap = simpledialog.askinteger(
            "Modifică capacitatea",
            f"Capacitatea nouă pentru {u_id} → {v_id} (curentă {cap}):",
            minvalue=1,
            parent=self.root,
        )
        if new_cap is None:
            return
        err = self.solver.update_capacity(u_id, v_id, new_cap)
        if err:
            messagebox.showerror("Modificarea a eșuat", err)
            return
        self.append_log(f"Capacitatea arcului {u_id} → {v_id} a fost actualizată la {new_cap}.")
        self._on_graph_changed()

    def remove_selected_edge(self):
        selected = self.edges_tree.selection()
        if not selected:
            messagebox.showinfo("Nicio selecție", "Selectați un arc de șters.")
            return
        values = self.edges_tree.item(selected[0], "values")
        u_id = self._extract_id_from_label(values[0])
        v_id = self._extract_id_from_label(values[1])
        if u_id and v_id and self.solver.remove_edge(u_id, v_id):
            self.append_log(f"S-a șters arcul {u_id} → {v_id}.")
            self._on_graph_changed()

    @staticmethod
    def _status_ro(status: str) -> str:
        return {"optimal": "optim", "continue": "în desfășurare"}.get(status, status)

    @staticmethod
    def _da_nu_ro(value: bool) -> str:
        return "DA" if value else "NU"

    @staticmethod
    def _rezultat_ro(value: bool) -> str:
        return "REUȘIT" if value else "EȘUAT"

    def clear_graph(self):
        if not self.solver.nodes:
            return
        if not messagebox.askyesno("Ștergere graf", "Ștergeți toate nodurile și arcele?"):
            return
        self.new_problem_action(log_message="Graful a fost șters.")

    def delete_current_problem_action(self):
        """Șterge imediat problema curentă și pregătește interfața pentru una nouă."""
        self.new_problem_action(log_message="Problema curentă a fost ștearsă. Puteți crea o problemă nouă.")

    def new_problem_action(self, log_message="A fost creată o problemă nouă goală."):
        """Reset direct pentru a începe o altă problemă de la zero."""
        self.solver = FlowNetworkSolver()
        self.problem_name.set("Problemă nouă")
        self.source_var.set("")
        self.sink_var.set("")
        self.new_node_id.set("")
        self.new_node_name.set("")
        self.new_edge_from.set("")
        self.new_edge_to.set("")
        self.new_edge_cap.set("")
        self.layout_positions = {}

        self.reset_run_state(preserve_log=False)
        self._refresh_node_view()
        self._refresh_edge_view()
        self.append_log(log_message)
        self.notebook.select(self.tab_input)
        self._schedule_process_scroll_update()
        self._schedule_verify_scroll_update()
        self._schedule_final_check_scroll_update()

    @staticmethod
    def _extract_id_from_label(label: str) -> str:
        if "(" in label and label.endswith(")"):
            return label[label.rfind("(") + 1: -1].strip()
        return label.strip()

    # ===================== Actualizare vizuala dupa modificari graf =====================
    def _on_graph_changed(self):
        self.reset_run_state(preserve_log=True)
        self._refresh_node_view()
        self._refresh_edge_view()
        self._compute_layout()
        self._schedule_process_scroll_update()
        self._schedule_final_check_scroll_update()

    def _refresh_node_view(self):
        self._clear_tree(self.nodes_tree)
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for u, v, _ in self.solver.edge_list():
            out_deg[u] += 1
            in_deg[v] += 1
        for nid in self.solver.node_ids:
            self.nodes_tree.insert("", "end", values=(nid, self.solver.nodes[nid], in_deg[nid], out_deg[nid]))

        ids = list(self.solver.node_ids)
        self.source_combo["values"] = ids
        self.sink_combo["values"] = ids
        if self.source_var.get() not in ids:
            self.source_var.set(ids[0] if ids else "")
        if self.sink_var.get() not in ids:
            self.sink_var.set(ids[-1] if len(ids) > 1 else "")

        self.edge_from_combo["values"] = ids
        self.edge_to_combo["values"] = ids
        if self.new_edge_from.get() not in ids:
            self.new_edge_from.set("")
        if self.new_edge_to.get() not in ids:
            self.new_edge_to.set("")

    def _refresh_edge_view(self):
        self._clear_tree(self.edges_tree)
        for u, v, cap in self.solver.edge_list():
            u_label = f"{self.solver.nodes[u]} ({u})"
            v_label = f"{self.solver.nodes[v]} ({v})"
            self.edges_tree.insert("", "end", values=(u_label, v_label, f"{cap:,}"))

    # ===================== Layout pentru graf =====================
    def _compute_layout(self):
        ids = list(self.solver.node_ids)
        if not ids:
            self.layout_positions = {}
            return

        source = self.source_var.get() if self.source_var.get() in ids else ids[0]

        levels = {source: 0}
        queue = deque([source])
        adjacency = {nid: [v for v, c in self.solver.original_cap[nid].items() if c > 0] for nid in ids}
        while queue:
            u = queue.popleft()
            for v in adjacency.get(u, []):
                if v not in levels:
                    levels[v] = levels[u] + 1
                    queue.append(v)

        max_lvl = max(levels.values()) if levels else 0
        for nid in ids:
            if nid not in levels:
                max_lvl += 1
                levels[nid] = max_lvl

        groups = defaultdict(list)
        for nid in ids:
            groups[levels[nid]].append(nid)

        max_layer = max(groups.keys()) if groups else 0
        positions = {}
        for lvl in sorted(groups.keys()):
            nodes_in_lvl = groups[lvl]
            count = len(nodes_in_lvl)
            for i, nid in enumerate(nodes_in_lvl):
                x = lvl / max(max_layer, 1) if max_layer else 0.5
                y = (i + 1) / (count + 1)
                positions[nid] = (x, y)

        self.layout_positions = positions

    def _draw_network(self, ax, *, highlight_path=None, residuals=None, S_set=None, T_set=None, cut_edges=None, title="Rețea"):
        ax.clear()
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1e3a8a")

        if not self.layout_positions:
            ax.text(0.5, 0.5, "Nu există încă un graf definit.", ha="center", va="center", fontsize=11, color="#64748b")
            return

        # Aspect egal: impiedica turtirea grafului cand cadrul Tkinter este ingust sau scund.
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.14, 1.14)
        ax.set_aspect("equal", adjustable="box")

        path_pairs = set()
        if highlight_path:
            for a, b in zip(highlight_path[:-1], highlight_path[1:]):
                path_pairs.add((a, b))
        path_nodes = set(highlight_path or [])

        cut_pairs = set()
        if cut_edges:
            for ed in cut_edges:
                cut_pairs.add((ed["u"], ed["v"]))

        S_set = set(S_set or [])
        T_set = set(T_set or [])

        for u, v, cap in self.solver.edge_list():
            if u not in self.layout_positions or v not in self.layout_positions:
                continue
            x1, y1 = self.layout_positions[u]
            x2, y2 = self.layout_positions[v]

            color = "#94a3b8"
            lw = 1.2
            zorder = 1
            if (u, v) in path_pairs:
                color = "#dc2626"
                lw = 2.6
                zorder = 4
            elif (u, v) in cut_pairs:
                color = "#b91c1c"
                lw = 2.4
                zorder = 3

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                lw=lw,
                zorder=zorder,
                shrinkA=14,
                shrinkB=14,
                connectionstyle="arc3,rad=0.07",
            )
            ax.add_patch(arrow)

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            if residuals is not None:
                ids = self.solver.node_ids
                if u in ids and v in ids:
                    iu, iv = ids.index(u), ids.index(v)
                    val = residuals[iu, iv]
                    text = f"{int(val)}/{int(cap)}"
                else:
                    text = f"{int(cap)}"
            else:
                text = f"{int(cap)}"
            ax.text(
                mx,
                my,
                text,
                fontsize=8,
                color="#1e3a8a",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="#dbeafe", lw=0.8),
            )

        source_id = self.source_var.get()
        sink_id = self.sink_var.get()
        for nid, (x, y) in self.layout_positions.items():
            edge_color = "#1e3a8a"
            face_color = "#ffffff"

            if S_set and nid in S_set:
                face_color = "#dbeafe"
            if T_set and nid in T_set:
                face_color = "#fee2e2"
            if nid in path_nodes:
                face_color = "#fef3c7"
                edge_color = "#dc2626"
            if nid == source_id:
                edge_color = "#15803d"
            if nid == sink_id:
                edge_color = "#b91c1c"

            circ = Circle((x, y), radius=0.05, facecolor=face_color, edgecolor=edge_color, lw=2.2, zorder=5)
            ax.add_patch(circ)
            display = self.solver.nodes.get(nid, nid)
            label = nid if display == nid else f"{nid}\n{display}"
            ax.text(x, y - 0.085, label, fontsize=8, color="#1e3a8a", ha="center", va="top", fontweight="bold", zorder=6)

    # ===================== Rulare =====================
    def _problem_signature(self):
        edges = tuple(sorted((u, v, c) for u, v, c in self.solver.edge_list()))
        nodes = tuple(self.solver.node_ids)
        return (self.source_var.get(), self.sink_var.get(), nodes, edges)

    def _ensure_generator(self):
        source = self.source_var.get().strip()
        sink = self.sink_var.get().strip()
        if not source or not sink:
            raise ValueError("Alegeți o sursă și o destinație înainte de rularea algoritmului.")
        if source == sink:
            raise ValueError("Sursa și destinația trebuie să fie noduri diferite.")
        if not self.solver.has_edges():
            raise ValueError("Graful nu are arce. Adăugați arce înainte de rulare.")
        signature = self._problem_signature()
        if self.generator is None or self.current_data_signature != signature:
            self.reset_run_state(preserve_log=True)
            self.current_data_signature = signature
            self.generator = self.solver.iteration_generator(source, sink)
            self.append_log("Generatorul Ford-Fulkerson a fost inițializat (Edmonds-Karp / BFS).")
            self.append_log(f"Sursă = {source} ({self.solver.nodes[source]}), destinație = {sink} ({self.solver.nodes[sink]}).")

    def next_step(self):
        try:
            self._ensure_generator()
            if self.finished:
                messagebox.showinfo("Algoritm finalizat", "Rezolvarea este deja completă.")
                return
            state = next(self.generator)
            self.current_state = state
            self._consume_state(state)
        except StopIteration:
            self.finished = True
        except Exception as exc:
            messagebox.showerror("Eroare", str(exc))
            self.append_log(f"EROARE: {exc}")

    def solve_complete(self):
        try:
            self._ensure_generator()
            if self.finished:
                return
            while True:
                state = next(self.generator)
                self.current_state = state
                self._consume_state(state)
                if state["status"] == "optimal":
                    break
        except StopIteration:
            self.finished = True
        except Exception as exc:
            messagebox.showerror("Eroare", str(exc))
            self.append_log(f"EROARE: {exc}")

    def _consume_state(self, state):
        self.finished = state["status"] == "optimal"
        self._update_iteration_views(state)
        self._write_state_log(state)
        if state.get("final_result") is not None:
            self.final_result = state["final_result"]
            self._display_final_result(self.final_result)
        self._schedule_process_scroll_update()

    # ===================== Resetare =====================
    def reset_run_state(self, preserve_log=True):
        self.generator = None
        self.current_state = None
        self.final_result = None
        self.current_data_signature = None
        self.finished = False

        if not preserve_log:
            self.clear_text_widget(self.log_text)

        self.preview_label.configure(text="Nu a fost rulată încă nicio iterație.")
        self.set_text_widget(self.preview_text, "")
        self.iteration_summary.configure(text="Nu există încă date de iterație.")
        self.flow_big_label.configure(text="—")
        self.result_status_label.configure(text="Nu există încă o rezolvare completă.", style="Info.TLabel")
        self.verify_label.configure(text="Nu există încă verificări.")
        self.cut_summary.configure(text="Nu există încă o tăietură minimă.")
        self._reset_final_check_tab()

        self._clear_tree(self.flows_tree)
        self._clear_tree(self.cut_tree)
        self._clear_tree(self.conservation_tree)

        self.table_capacity.set_data(None)
        self.table_residual_before.set_data(None)
        self.table_residual_after.set_data(None)
        self.table_flow_after.set_data(None)
        self.table_final_flow.set_data(None)

        self.set_text_widget(self.path_text, "")

        self._update_plot([])
        self.process_ax.clear()
        self.process_ax.set_axis_off()
        self.process_canvas_plot.draw_idle()
        self.cut_ax.clear()
        self.cut_ax.set_axis_off()
        self.cut_canvas_plot.draw_idle()

        if self.solver.node_ids:
            self._draw_network(self.process_ax, title="Rețea inițială")
            self.process_canvas_plot.draw_idle()
            self._draw_network(self.cut_ax, title="Rețea inițială")
            self.cut_canvas_plot.draw_idle()

        self._schedule_process_scroll_update()
        self._schedule_verify_scroll_update()
        self._schedule_final_check_scroll_update()

    # ===================== Actualizarea vizualizărilor =====================
    def _update_iteration_views(self, state):
        ids = state["node_ids"]
        names = state["node_names"]
        labels = [f"{names[i]}\n({ids[i]})" if names[i] != ids[i] else ids[i] for i in range(len(ids))]

        if state["status"] == "optimal":
            summary = f"Iterația {state['it']} | Stare: {self._status_ro(state['status'])} | Nu există drum de augmentare. Flux maxim = {state['max_flow']:.0f}."
        else:
            path_str = " → ".join(state["path_names"])
            summary = (
                f"Iterația {state['it']} | Stare: {self._status_ro(state['status'])} | "
                f"Drum: {path_str} | Gât de sticlă = {state['bottleneck']:.0f} | Flux total = {state['max_flow']:.0f}"
            )
        self.iteration_summary.configure(text=summary)

        preview_lines = [
            f"Iterația: {state['it']}",
            f"Stare: {self._status_ro(state['status'])}",
            f"Sursă: {self.solver.nodes.get(state['source'], state['source'])}",
            f"Destinație: {self.solver.nodes.get(state['sink'], state['sink'])}",
        ]
        if state.get("path_names"):
            preview_lines.append(f"Drum de augmentare: {' → '.join(state['path_names'])}")
            preview_lines.append(f"Gât de sticlă: {state['bottleneck']:.0f}")
        preview_lines.append(f"Flux cumulativ: {state['max_flow']:.0f}")
        preview_lines.append(f"Ordine de vizitare BFS: {', '.join(self.solver.nodes.get(n, n) for n in state['bfs_visited'])}")
        self.preview_label.configure(text=f"Ultima stare: iterația {state['it']}")
        self.set_text_widget(self.preview_text, "\n".join(preview_lines))

        path_cells = []
        if state.get("path_ids"):
            for a, b in zip(state["path_ids"][:-1], state["path_ids"][1:]):
                if a in ids and b in ids:
                    path_cells.append((ids.index(a), ids.index(b)))

        self.table_capacity.set_data(state["capacity_matrix"], row_headers=labels, col_headers=labels, highlight_cells=path_cells)
        self.table_residual_before.set_data(state["residual_before"], row_headers=labels, col_headers=labels, highlight_cells=path_cells)
        self.table_residual_after.set_data(state["residual_after"], row_headers=labels, col_headers=labels, highlight_cells=path_cells)
        self.table_flow_after.set_data(state["flow_after"], row_headers=labels, col_headers=labels, highlight_cells=path_cells)

        if state["status"] == "optimal":
            title = f"Iterația {state['it']} - fără drum de augmentare (terminal)"
            self._draw_network(self.process_ax, residuals=state["residual_after"], title=title)
        else:
            title = f"Iterația {state['it']} - gât de sticlă {int(state['bottleneck'])}"
            self._draw_network(self.process_ax, highlight_path=state["path_ids"], residuals=state["residual_after"], title=title)
        self.process_canvas_plot.draw_idle()

        path_lines = []
        if state.get("path_ids"):
            ids_path = state["path_ids"]
            names_path = state["path_names"]
            path_lines.append(f"Drum de augmentare ({len(ids_path)} noduri):")
            for k, (i_, n_) in enumerate(zip(ids_path, names_path)):
                if k == 0:
                    path_lines.append(f"  pornire →  {n_} ({i_})")
                else:
                    prev_id = ids_path[k - 1]
                    cap_here = state["residual_before"][ids.index(prev_id), ids.index(i_)]
                    path_lines.append(f"   pasul {k:>2}  →  {n_} ({i_})    rezidual la intrare = {int(cap_here)}")
            path_lines.append("")
            path_lines.append(f"Gât de sticlă (rezidual minim pe drum) = {int(state['bottleneck'])}")
        else:
            path_lines.append("Nu s-a găsit niciun drum de augmentare în această iterație.")

        path_lines.append("")
        path_lines.append("Ordinea explorării BFS:")
        for k, nid in enumerate(state["bfs_visited"], start=1):
            path_lines.append(f"  {k:>2}. {self.solver.nodes.get(nid, nid)} ({nid})")
        self.set_text_widget(self.path_text, "\n".join(path_lines))

        self._update_plot(state.get("flow_history", []))

    def _reset_final_check_tab(self):
        if not hasattr(self, "final_check_title"):
            return
        self.final_check_title.configure(
            text="Apasă 'Rezolvă complet' pentru a obține soluția finală și verificările.",
            style="Info.TLabel",
        )
        self.check_balance_label.configure(text="Nu există încă o soluție finală.", style="Info.TLabel")
        self.check_conservation_label.configure(text="Nu există încă o soluție finală.", style="Info.TLabel")
        self.check_mincut_label.configure(text="Nu există încă o soluție finală.", style="Info.TLabel")
        for tree in (self.final_conservation_tree, self.final_cut_tree, self.final_arc_tree):
            self._clear_tree(tree)
        self.set_text_widget(self.final_check_text, "")

    def _display_final_checks(self, result):
        ok_balance = (
            abs(result["source_outflow"] - result["max_flow"]) < 1e-9
            and abs(result["sink_inflow"] - result["max_flow"]) < 1e-9
        )
        ok_conservation = all(c["balanced"] for c in result["conservation"])
        ok_mincut = result["max_flow_min_cut_match"]
        all_ok = ok_balance and ok_conservation and ok_mincut

        title_style = "Good.TLabel" if all_ok else "Warn.TLabel"
        self.final_check_title.configure(
            text=(
                f"Soluția finală pentru {result['source_name']} ({result['source']}) -> "
                f"{result['sink_name']} ({result['sink']}): F* = {int(result['max_flow'])}. "
                f"Verificare generală: {self._rezultat_ro(all_ok)}."
            ),
            style=title_style,
        )

        balance_style = "Good.TLabel" if ok_balance else "Warn.TLabel"
        self.check_balance_label.configure(
            text=(
                f"Stare: {self._rezultat_ro(ok_balance)}\n"
                f"Ieșire din sursă = {result['source_outflow']:,}\n"
                f"Intrare în destinație = {result['sink_inflow']:,}\n"
                f"Flux maxim = {int(result['max_flow']):,}\n"
                "Condiție: ieșirea din sursă = intrarea în destinație = F*."
            ),
            style=balance_style,
        )

        conservation_style = "Good.TLabel" if ok_conservation else "Warn.TLabel"
        self.check_conservation_label.configure(
            text=(
                f"Stare: {self._rezultat_ro(ok_conservation)}\n"
                f"Noduri interioare verificate: {len(result['conservation'])}\n"
                "Condiție: pentru fiecare nod interior, intrare = ieșire."
            ),
            style=conservation_style,
        )

        mincut_style = "Good.TLabel" if ok_mincut else "Warn.TLabel"
        self.check_mincut_label.configure(
            text=(
                f"Stare: {self._rezultat_ro(ok_mincut)}\n"
                f"F* = {int(result['max_flow']):,}\n"
                f"Capacitatea tăieturii minime = {result['cut_capacity']:,}\n"
                f"Arce în tăietură: {len(result['cut_edges'])}\n"
                "Condiție: F* = capacitatea tăieturii minime."
            ),
            style=mincut_style,
        )

        self._clear_tree(self.final_conservation_tree)
        for c in result["conservation"]:
            diff = c["inflow"] - c["outflow"]
            self.final_conservation_tree.insert(
                "",
                "end",
                values=(
                    f"{c['name']} ({c['node']})",
                    f"{c['inflow']:,}",
                    f"{c['outflow']:,}",
                    f"{diff:,}",
                    "echilibrat" if c["balanced"] else "încălcat",
                ),
            )

        self._clear_tree(self.final_cut_tree)
        for ed in result["cut_edges"]:
            self.final_cut_tree.insert(
                "",
                "end",
                values=(
                    f"{ed['u_name']} ({ed['u']})",
                    f"{ed['v_name']} ({ed['v']})",
                    f"{ed['capacity']:,}",
                ),
            )

        self._clear_tree(self.final_arc_tree)
        for arc in result["flow_arcs"]:
            residual = arc["capacity"] - arc["flow"]
            capacity_ok = 0 <= arc["flow"] <= arc["capacity"]
            status = "OK"
            if not capacity_ok:
                status = "eroare capacitate"
            elif arc["saturated"]:
                status = "saturat"
            elif arc["flow"] > 0:
                status = "activ"
            else:
                status = "neutilizat"
            self.final_arc_tree.insert(
                "",
                "end",
                values=(
                    f"{arc['u_name']} ({arc['u']}) -> {arc['v_name']} ({arc['v']})",
                    f"{arc['flow']:,}",
                    f"{arc['capacity']:,}",
                    f"{residual:,}",
                    status,
                ),
            )

        S_label = ", ".join(f"{name} ({node})" for node, name in zip(result["S_set"], result["S_names"])) or "-"
        T_label = ", ".join(f"{name} ({node})" for node, name in zip(result["T_set"], result["T_names"])) or "-"
        explanation = [
            "RAPORT FINAL DE VERIFICARE",
            "=" * 78,
            f"Flux maxim F* = {int(result['max_flow']):,}",
            f"Iterații        = {result['iterations']}",
            "",
            "1) Echilibrul sursă / destinație",
            f"   Ieșire din sursă = {result['source_outflow']:,}",
            f"   Intrare în destinație = {result['sink_inflow']:,}",
            f"   Rezultat       = {self._rezultat_ro(ok_balance)}",
            "",
            "2) Conservarea fluxului în nodurile interioare",
            f"   Rezultat       = {self._rezultat_ro(ok_conservation)}",
            "   Consultați tabelul detaliat de conservare pentru fiecare nod interior.",
            "",
            "3) Teorema Max-Flow Min-Cut",
            f"   S = {{ {S_label} }}",
            f"   T = {{ {T_label} }}",
            f"   Capacitatea tăieturii = {result['cut_capacity']:,}",
            f"   F*           = {int(result['max_flow']):,}",
            f"   Rezultat       = {self._rezultat_ro(ok_mincut)}",
            "",
            "Detalii suport",
            "   Tabelul arcelor confirmă și condiția de capacitate 0 <= flux <= capacitate pentru fiecare arc.",
        ]
        self.set_text_widget(self.final_check_text, "\n".join(explanation))
        self._schedule_final_check_scroll_update()

    def _display_final_result(self, result):
        text_status = (
            f"Fluxul maxim a fost atins după {result['iterations']} iterații. "
            f"Ieșire din sursă = {result['source_outflow']}, intrare în destinație = {result['sink_inflow']}. "
            f"Capacitatea tăieturii minime = {result['cut_capacity']}."
        )
        style_name = "Good.TLabel" if result["max_flow_min_cut_match"] else "Warn.TLabel"

        self.flow_big_label.configure(text=f"F* = {result['max_flow']:.0f}")
        self.result_status_label.configure(text=text_status, style=style_name)

        self._clear_tree(self.flows_tree)
        for arc in result["flow_arcs"]:
            arc_label = f"{arc['u_name']} ({arc['u']}) → {arc['v_name']} ({arc['v']})"
            util_pct = 100.0 * arc["utilization"]
            status = "saturat" if arc["saturated"] else ("activ" if arc["flow"] > 0 else "neutilizat")
            self.flows_tree.insert("", "end", values=(arc_label, f"{arc['flow']:,}", f"{arc['capacity']:,}", f"{util_pct:.0f}%", status))

        S_label = ", ".join(f"{n} ({i})" for i, n in zip(result["S_set"], result["S_names"])) or "—"
        T_label = ", ".join(f"{n} ({i})" for i, n in zip(result["T_set"], result["T_names"])) or "—"
        cut_text = (
            f"S = {{ {S_label} }}\n"
            f"T = {{ {T_label} }}\n"
            f"Σ capacităților arcelor (S→T) = {result['cut_capacity']} | F* = {int(result['max_flow'])} | "
            f"Egalitatea este respectată: {self._da_nu_ro(result['max_flow_min_cut_match'])}"
        )
        self.cut_summary.configure(text=cut_text)

        self._clear_tree(self.cut_tree)
        for ed in result["cut_edges"]:
            self.cut_tree.insert("", "end", values=(f"{ed['u_name']} ({ed['u']})", f"{ed['v_name']} ({ed['v']})", f"{ed['capacity']:,}"))

        ok_match = result["max_flow_min_cut_match"]
        ok_conservation = all(c["balanced"] for c in result["conservation"])
        ok_source_sink = abs(result["source_outflow"] - result["max_flow"]) < 1e-9 and abs(result["sink_inflow"] - result["max_flow"]) < 1e-9
        verify_text = (
            f"Teorema Max-Flow Min-Cut: {'respectată' if ok_match else 'NERESPECTATĂ'} "
            f"(F* = {int(result['max_flow'])}, Σ tăietură = {result['cut_capacity']})\n"
            f"Ieșire din sursă = {result['source_outflow']} | Intrare în destinație = {result['sink_inflow']} | "
            f"Flux total = {int(result['max_flow'])} | Echilibru sursă/destinație: {self._da_nu_ro(ok_source_sink)}\n"
            f"Conservarea fluxului în toate nodurile interioare: {self._da_nu_ro(ok_conservation)} | Iterații: {result['iterations']}"
        )
        self.verify_label.configure(text=verify_text)

        self._clear_tree(self.conservation_tree)
        for c in result["conservation"]:
            diff = c["inflow"] - c["outflow"]
            self.conservation_tree.insert("", "end", values=(f"{c['name']} ({c['node']})", f"{c['inflow']:,}", f"{c['outflow']:,}", f"{diff:,}", "echilibrat" if c["balanced"] else "încălcat"))

        ids = result["node_ids"]
        names = result["node_names"]
        labels = [f"{names[i]}\n({ids[i]})" if names[i] != ids[i] else ids[i] for i in range(len(ids))]
        self.table_final_flow.set_data(result["flow_matrix"], row_headers=labels, col_headers=labels)

        self._update_plot(result["flow_history"])
        self._draw_network(
            self.cut_ax,
            S_set=result["S_set"],
            T_set=result["T_set"],
            cut_edges=result["cut_edges"],
            residuals=result["flow_matrix"],
            title=f"Tăietură minimă: |S|={len(result['S_set'])}, |T|={len(result['T_set'])}, Σ={result['cut_capacity']}",
        )
        self.cut_canvas_plot.draw_idle()
        self._display_final_checks(result)
        self._schedule_verify_scroll_update()
        self._schedule_final_check_scroll_update()

    def _update_plot(self, flow_history):
        self.ax.clear()
        self.ax.set_title("Convergența monotonă a fluxului total")
        self.ax.set_xlabel("Iterație")
        self.ax.set_ylabel("Flux cumulativ")
        if flow_history:
            x = list(range(len(flow_history)))
            self.ax.plot(x, flow_history, marker="o", color="#2563eb", linewidth=2)
            self.ax.fill_between(x, flow_history, alpha=0.15, color="#2563eb")
            self.ax.grid(True, alpha=0.3)
        self.canvas_plot.draw_idle()

    def _write_state_log(self, state):
        self.append_log(f"\n--- Iterația {state['it']} ---")
        if state.get("path_names"):
            self.append_log(f"Drum de augmentare: {' → '.join(state['path_names'])} | gât de sticlă = {int(state['bottleneck'])}")
            self.append_log(f"Flux cumulativ = {int(state['max_flow'])}")
        else:
            self.append_log("Nu s-a găsit niciun drum de augmentare - algoritmul se oprește.")
            self.append_log(f"Flux maxim final = {int(state['max_flow'])}")
        if state.get("final_result") is not None:
            fr = state["final_result"]
            self.append_log(f"Capacitatea tăieturii minime = {fr['cut_capacity']} | Max-flow == min-cut: {self._da_nu_ro(fr['max_flow_min_cut_match'])}")

    # ===================== Import/export JSON =====================
    def _collect_form_snapshot(self):
        return {
            "problem_name": self.problem_name.get().strip() or "Problemă nouă",
            "source": self.source_var.get(),
            "sink": self.sink_var.get(),
            "nodes": [{"id": nid, "name": self.solver.nodes[nid]} for nid in self.solver.node_ids],
            "edges": [{"from": u, "to": v, "capacity": int(c)} for u, v, c in self.solver.edge_list()],
        }

    def save_to_json(self):
        try:
            data = self._collect_form_snapshot()
        except Exception as exc:
            messagebox.showerror("Salvarea a eșuat", str(exc))
            return

        path = filedialog.asksaveasfilename(title="Salvează problema", defaultextension=".json", filetypes=[("Fișiere JSON", "*.json"), ("Toate fișierele", "*.*")])
        if not path:
            return

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        self.append_log(f"Problema a fost salvată în: {path}")
        messagebox.showinfo("Salvare reușită", "Problema a fost salvată ca JSON.")

    def load_from_json(self):
        path = filedialog.askopenfilename(title="Încarcă problema", filetypes=[("Fișiere JSON", "*.json"), ("Toate fișierele", "*.*")])
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            problem_name = str(data.get("problem_name", "Problemă încărcată"))
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            source = str(data.get("source", "")).strip()
            sink = str(data.get("sink", "")).strip()

            new_solver = FlowNetworkSolver()
            for nd in nodes:
                nid = str(nd.get("id", "")).strip()
                name = str(nd.get("name", "")).strip()
                if not nid:
                    raise ValueError("Fiecare nod trebuie să aibă un 'id' nevid.")
                if not new_solver.add_node(nid, name):
                    raise ValueError(f"ID de nod duplicat '{nid}' în fișier.")
            for ed in edges:
                u = str(ed.get("from", "")).strip()
                v = str(ed.get("to", "")).strip()
                cap = int(ed.get("capacity", 0))
                err = new_solver.add_edge(u, v, cap)
                if err:
                    raise ValueError(f"Arcul {u}→{v}: {err}")

            self.solver = new_solver
            self.problem_name.set(problem_name)
            if source and source in new_solver.nodes:
                self.source_var.set(source)
            elif new_solver.node_ids:
                self.source_var.set(new_solver.node_ids[0])
            if sink and sink in new_solver.nodes:
                self.sink_var.set(sink)
            elif len(new_solver.node_ids) > 1:
                self.sink_var.set(new_solver.node_ids[-1])

            self.clear_text_widget(self.log_text)
            self.append_log(f"Problema a fost încărcată din: {path}")
            self._on_graph_changed()
            messagebox.showinfo("Încărcare reușită", "Problema a fost încărcată din fișierul JSON.")
        except Exception as exc:
            messagebox.showerror("Încărcarea a eșuat", f"Nu s-a putut încărca JSON-ul.\n\nDetalii: {exc}")

    # ===================== Exemple =====================
    def load_example_bucharest(self):
        self.solver = FlowNetworkSolver()
        self.problem_name.set("Trafic București: Militari → Pipera")

        nodes = [
            ("x1", "Militari"),
            ("x2", "Lujerului"),
            ("x3", "Crângași"),
            ("x4", "Răzoare"),
            ("x5", "Basarab"),
            ("x6", "Eroilor"),
            ("x7", "Arcul de Triumf"),
            ("x8", "Victoriei"),
            ("x9", "B. Văcărescu"),
            ("x10", "Charles de Gaulle"),
            ("x11", "Pipera"),
        ]
        for nid, name in nodes:
            self.solver.add_node(nid, name)

        arcs = [
            ("x1", "x2", 6000),
            ("x2", "x3", 3000),
            ("x2", "x4", 2500),
            ("x3", "x5", 2000),
            ("x3", "x7", 2000),
            ("x4", "x6", 2000),
            ("x5", "x8", 2500),
            ("x6", "x8", 2500),
            ("x7", "x9", 1500),
            ("x7", "x10", 2500),
            ("x8", "x10", 4000),
            ("x9", "x11", 2500),
            ("x10", "x11", 3000),
        ]
        for u, v, c in arcs:
            self.solver.add_edge(u, v, c)

        self.source_var.set("x1")
        self.sink_var.set("x11")
        self.clear_text_widget(self.log_text)
        self.append_log("Exemplul București a fost încărcat: Militari → Pipera (11 noduri, 13 arce).")
        self._on_graph_changed()

    def load_example_simple(self):
        self.solver = FlowNetworkSolver()
        self.problem_name.set("Exemplu simplu S → A,B → T")

        for nid, name in [("S", "Sursă"), ("A", "A"), ("B", "B"), ("T", "Destinație")]:
            self.solver.add_node(nid, name)

        edges = [("S", "A", 10), ("S", "B", 10), ("A", "B", 2), ("A", "T", 10), ("B", "T", 10)]
        for u, v, c in edges:
            self.solver.add_edge(u, v, c)

        self.source_var.set("S")
        self.sink_var.set("T")
        self.clear_text_widget(self.log_text)
        self.append_log("Exemplul simplu a fost încărcat: 4 noduri, 5 arce. Flux maxim așteptat = 20.")
        self._on_graph_changed()


def main():
    root = tk.Tk()
    FordFulkersonInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
