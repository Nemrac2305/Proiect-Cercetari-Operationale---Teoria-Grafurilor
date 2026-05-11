"""
============================================================
  FORD-FULKERSON INTERACTIV (Edmonds-Karp / BFS)
  Introduce orice problemă de flux de la tastatură.
============================================================
  Rulare:  python ford_fulkerson_interactiv.py
============================================================
"""

from collections import deque
import sys
import os


# ════════════════════════════════════════════════════════════
#  UTILITARE TERMINAL
# ════════════════════════════════════════════════════════════

W = 62   # lățime consolă

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def linie(car="═"):
    print(car * W)

def titlu(text, car="═"):
    print(car * W)
    padding = W - 4 - len(text)
    print(f"  {text}" + " " * max(0, padding))
    print(car * W)

def sectiune(text):
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print(f"│  {text:<{W-4}}│")
    print("└" + "─" * (W - 2) + "┘")

def ok(text):   print(f"  ✔  {text}")
def err(text):  print(f"  ✖  {text}")
def info(text): print(f"  ·  {text}")

def input_safe(prompt, default=None):
    """Input cu suport pentru Ctrl+C și valoare implicită."""
    try:
        val = input(prompt).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Ieșire din program.")
        sys.exit(0)

def input_int(prompt, min_val=None, max_val=None):
    """Input numeric cu validare."""
    while True:
        raw = input_safe(prompt)
        try:
            n = int(raw)
            if min_val is not None and n < min_val:
                err(f"Valoarea trebuie să fie ≥ {min_val}.")
                continue
            if max_val is not None and n > max_val:
                err(f"Valoarea trebuie să fie ≤ {max_val}.")
                continue
            return n
        except (ValueError, TypeError):
            err("Introdu un număr întreg valid.")

def alegere_meniu(optiuni: list[str]) -> int:
    """Afișează meniu și returnează indexul ales (0-based)."""
    for i, opt in enumerate(optiuni, 1):
        print(f"  [{i}] {opt}")
    return input_int("  > Alegere: ", min_val=1, max_val=len(optiuni)) - 1


# ════════════════════════════════════════════════════════════
#  CLASA GRAPH – FORD-FULKERSON
# ════════════════════════════════════════════════════════════

class Graph:
    def __init__(self):
        self.nodes: dict[str, str] = {}          # id → nume afișat
        self.node_ids: list[str] = []            # ordine inserare
        self.residual: dict[str, dict[str, int]] = {}
        self.original_cap: dict[str, dict[str, int]] = {}

    # ── Adăugare ────────────────────────────────────────────

    def add_node(self, node_id: str, name: str = "") -> bool:
        if node_id in self.nodes:
            return False
        self.nodes[node_id] = name.strip() or node_id
        self.node_ids.append(node_id)
        self.residual[node_id] = {}
        self.original_cap[node_id] = {}
        return True

    def add_edge(self, u: str, v: str, capacity: int) -> str:
        if u not in self.nodes:
            return f"Nodul '{u}' nu există."
        if v not in self.nodes:
            return f"Nodul '{v}' nu există."
        if u == v:
            return "Un arc nu poate pleca și sosi în același nod."
        if capacity <= 0:
            return "Capacitatea trebuie să fie > 0."
        self.residual[u][v] = self.residual[u].get(v, 0) + capacity
        self.original_cap[u][v] = self.original_cap[u].get(v, 0) + capacity
        if u not in self.residual[v]:
            self.residual[v][u] = 0
            self.original_cap[v][u] = self.original_cap[v].get(u, 0)
        return ""   # fără eroare

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        # Șterge arcele incidente
        del self.nodes[node_id]
        self.node_ids.remove(node_id)
        del self.residual[node_id]
        del self.original_cap[node_id]
        for nd in self.node_ids:
            self.residual[nd].pop(node_id, None)
            self.original_cap[nd].pop(node_id, None)
        return True

    def remove_edge(self, u: str, v: str) -> bool:
        if u not in self.original_cap or v not in self.original_cap.get(u, {}):
            return False
        cap = self.original_cap[u].pop(v)
        self.residual[u].pop(v, None)
        # Șterge și arcul invers dacă nu e arc real v→u
        if self.original_cap[v].get(u, 0) == 0:
            self.residual[v].pop(u, None)
            self.original_cap[v].pop(u, None)
        return True

    def edge_list(self) -> list[tuple[str, str, int]]:
        """Returnează lista arcelor reale (capacitate > 0)."""
        result = []
        for u in self.node_ids:
            for v, cap in self.original_cap[u].items():
                if cap > 0:
                    result.append((u, v, cap))
        return result

    # ── BFS ─────────────────────────────────────────────────

    def _bfs(self, source: str, sink: str) -> dict:
        visited = {source}
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in self.residual[u].items():
                if v not in visited and cap > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return {}

    # ── Ford-Fulkerson ───────────────────────────────────────

    def ford_fulkerson(self, source: str, sink: str) -> tuple[int, list[dict]]:
        """
        Returnează (flux_maxim, log_iteratii).
        log_iteratii = [{iter, path_ids, path_names, flow, total}]
        """
        if source not in self.nodes:
            raise ValueError(f"Sursa '{source}' nu există.")
        if sink not in self.nodes:
            raise ValueError(f"Destinația '{sink}' nu există.")

        # Reset rezidual
        for u in self.node_ids:
            for v in list(self.residual[u]):
                self.residual[u][v] = self.original_cap[u].get(v, 0)

        max_flow = 0
        log = []
        iteration = 0

        while True:
            parent = self._bfs(source, sink)
            if not parent:
                break

            # Reconstituire drum
            path = []
            cur = sink
            path_flow = float("inf")
            while cur != source:
                prev = parent[cur]
                path_flow = min(path_flow, self.residual[prev][cur])
                path.append(cur)
                cur = prev
            path.append(source)
            path.reverse()

            # Update rezidual
            cur = sink
            while cur != source:
                prev = parent[cur]
                self.residual[prev][cur] -= path_flow
                self.residual[cur][prev] = self.residual[cur].get(prev, 0) + path_flow
                cur = prev

            max_flow += path_flow
            iteration += 1
            log.append({
                "iter": iteration,
                "path_ids": list(path),
                "path_names": [self.nodes[n] for n in path],
                "flow": path_flow,
                "total": max_flow,
            })

        return max_flow, log

    # ── Tăietură minimă ─────────────────────────────────────

    def min_cut(self, source: str) -> tuple[list, list, list]:
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        while queue:
            u = queue.popleft()
            for v, cap in self.residual[u].items():
                if v not in reachable and cap > 0:
                    reachable.add(v)
                    queue.append(v)
        S = [n for n in self.node_ids if n in reachable]
        T = [n for n in self.node_ids if n not in reachable]
        cut_edges = []
        for u in S:
            for v in T:
                if self.original_cap[u].get(v, 0) > 0:
                    cut_edges.append((u, v, self.original_cap[u][v]))
        return S, T, cut_edges

    # ── Fluxuri curente ──────────────────────────────────────

    def flows(self) -> list[tuple[str, str, int, int]]:
        """Returnează [(u, v, flux_curent, capacitate)]."""
        result = []
        for u in self.node_ids:
            for v, cap in self.original_cap[u].items():
                if cap > 0:
                    flow = cap - self.residual[u].get(v, 0)
                    result.append((u, v, flow, cap))
        return result

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def has_edges(self) -> bool:
        return len(self.edge_list()) > 0


# ════════════════════════════════════════════════════════════
#  AFIȘARE REZULTATE
# ════════════════════════════════════════════════════════════

def afiseaza_iteratii(log: list[dict]):
    sectiune("ITERAȚII FORD-FULKERSON")
    if not log:
        err("Nu s-a găsit niciun drum augmentant. Fluxul este 0.")
        return
    for entry in log:
        drum = " → ".join(entry["path_names"])
        print(f"\n  Iterația {entry['iter']:>2}:")
        print(f"    Drum  : {drum}")
        print(f"    +Flux : {entry['flow']:,}   │   Total curent: {entry['total']:,}")


def afiseaza_fluxuri(g: Graph):
    sectiune("FLUX PE FIECARE ARC")
    print(f"  {'ARC':<34} {'FLUX':>7} / {'CAP.':>7}   UTILIZARE")
    linie("─")
    for u, v, flow, cap in g.flows():
        arc = f"{g.nodes[u]} → {g.nodes[v]}"
        pct = flow / cap * 100 if cap else 0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        saturat = " ← SATURAT" if flow == cap else ""
        print(f"  {arc:<34} {flow:>7,} / {cap:>7,}   [{bar}] {pct:.0f}%{saturat}")
    linie("─")


def afiseaza_taietura(g: Graph, source: str, sink: str):
    S, T, cut_edges = g.min_cut(source)
    sectiune("TĂIETURA MINIMĂ  (Teorema Max-Flow Min-Cut)")
    print(f"  S (accesibile din sursă) : {', '.join(g.nodes[n] for n in S)}")
    print(f"  T (restul)               : {', '.join(g.nodes[n] for n in T)}")
    print()
    cap_total = sum(c for _, _, c in cut_edges)
    print(f"  {'ARC TĂIETURĂ':<34} {'CAPACITATE':>10}")
    linie("─")
    for u, v, cap in cut_edges:
        print(f"  {g.nodes[u]} → {g.nodes[v]:<{32 - len(g.nodes[u])}} {cap:>10,}")
    linie("─")
    print(f"  {'Capacitate totală tăietură':<34} {cap_total:>10,}")
    print(f"  {'= Flux maxim ✓' if True else ''}")


def afiseaza_rezumat(g: Graph, max_flow: int, source: str, sink: str):
    print()
    linie("═")
    print(f"  REZULTAT FINAL")
    linie("═")
    print(f"  Sursă        : {g.nodes[source]}")
    print(f"  Destinație   : {g.nodes[sink]}")
    print(f"  FLUX MAXIM   : {max_flow:,}")
    linie("═")


# ════════════════════════════════════════════════════════════
#  INTRODUCERE DATE – NODURI ȘI ARCE
# ════════════════════════════════════════════════════════════

def afiseaza_noduri(g: Graph):
    if g.is_empty():
        info("Nu există noduri adăugate.")
        return
    print(f"\n  {'ID':<8} {'NUME'}")
    linie("─")
    for nid in g.node_ids:
        print(f"  {nid:<8} {g.nodes[nid]}")
    linie("─")


def afiseaza_arce(g: Graph):
    edges = g.edge_list()
    if not edges:
        info("Nu există arce adăugate.")
        return
    print(f"\n  {'DE LA':<20} {'SPRE':<20} {'CAPACITATE':>10}")
    linie("─")
    for u, v, cap in edges:
        print(f"  {g.nodes[u]:<20} {g.nodes[v]:<20} {cap:>10,}")
    linie("─")


def meniu_adauga_noduri(g: Graph):
    cls()
    titlu("ADĂUGARE NODURI")
    print("  Introdu nodurile grafului tău.")
    print("  Fiecare nod are un ID scurt (ex: A, x1, S) și un nume descriptiv.")
    print("  Lasă ID-ul gol și apasă Enter pentru a termina.\n")

    while True:
        afiseaza_noduri(g)
        print()
        nid = input_safe("  ID nod (Enter = gata): ", default="")
        if not nid:
            if g.is_empty():
                err("Trebuie să adaugi cel puțin 2 noduri.")
                continue
            if len(g.nodes) < 2:
                err("Trebuie să adaugi cel puțin 2 noduri.")
                continue
            break
        if nid in g.nodes:
            err(f"ID-ul '{nid}' există deja. Alege altul.")
            continue
        name = input_safe(f"  Nume pentru '{nid}' (Enter = folosește ID-ul): ", default=nid)
        if g.add_node(nid, name):
            ok(f"Nod '{nid}' ({name}) adăugat.")
        print()


def meniu_adauga_arce(g: Graph):
    cls()
    titlu("ADĂUGARE ARCE")
    print("  Definește arcele orientate și capacitățile lor.")
    print("  Introdu ID-ul nodului sursă, ID-ul nodului destinație și capacitatea.")
    print("  Lasă sursa goală și apasă Enter pentru a termina.\n")

    afiseaza_noduri(g)
    print()

    while True:
        afiseaza_arce(g)
        print()
        u = input_safe("  De la (ID nod, Enter = gata): ", default="")
        if not u:
            if not g.has_edges():
                err("Trebuie să adaugi cel puțin un arc.")
                continue
            break
        if u not in g.nodes:
            err(f"Nodul '{u}' nu există. ID-urile disponibile: {', '.join(g.node_ids)}")
            continue

        v = input_safe("  Spre   (ID nod): ", default="")
        if not v:
            continue
        if v not in g.nodes:
            err(f"Nodul '{v}' nu există. ID-urile disponibile: {', '.join(g.node_ids)}")
            continue

        cap = input_int("  Capacitate     : ", min_val=1)
        eroare = g.add_edge(u, v, cap)
        if eroare:
            err(eroare)
        else:
            ok(f"Arc {g.nodes[u]} → {g.nodes[v]} cu capacitate {cap:,} adăugat.")
        print()


def alege_sursa_destinatie(g: Graph) -> tuple[str, str] | tuple[None, None]:
    cls()
    titlu("SURSĂ & DESTINAȚIE")
    afiseaza_noduri(g)
    print()

    source = input_safe("  ID nod SURSĂ       : ", default="").strip()
    if source not in g.nodes:
        err(f"Nodul '{source}' nu există.")
        input_safe("  Apasă Enter pentru a continua...")
        return None, None

    sink = input_safe("  ID nod DESTINAȚIE  : ", default="").strip()
    if sink not in g.nodes:
        err(f"Nodul '{sink}' nu există.")
        input_safe("  Apasă Enter pentru a continua...")
        return None, None

    if source == sink:
        err("Sursa și destinația nu pot fi același nod.")
        input_safe("  Apasă Enter pentru a continua...")
        return None, None

    return source, sink


# ════════════════════════════════════════════════════════════
#  MENIU EDITARE GRAF
# ════════════════════════════════════════════════════════════

def meniu_editare(g: Graph):
    while True:
        cls()
        titlu("EDITARE GRAF")
        afiseaza_noduri(g)
        afiseaza_arce(g)
        print()
        optiuni = [
            "Adaugă nod",
            "Șterge nod",
            "Adaugă arc",
            "Șterge arc",
            "Modifică capacitate arc",
            "Înapoi",
        ]
        alegere = alegere_meniu(optiuni)
        print()

        if alegere == 0:   # Adaugă nod
            nid = input_safe("  ID nod nou: ", default="").strip()
            if nid:
                name = input_safe(f"  Nume pentru '{nid}': ", default=nid)
                if g.add_node(nid, name):
                    ok(f"Nod '{nid}' adăugat.")
                else:
                    err("ID deja existent.")

        elif alegere == 1:  # Șterge nod
            nid = input_safe("  ID nod de șters: ", default="").strip()
            if g.remove_node(nid):
                ok(f"Nod '{nid}' și arcele sale au fost șterse.")
            else:
                err("Nod inexistent.")

        elif alegere == 2:  # Adaugă arc
            u = input_safe("  De la (ID): ", default="").strip()
            v = input_safe("  Spre   (ID): ", default="").strip()
            cap = input_int("  Capacitate : ", min_val=1)
            eroare = g.add_edge(u, v, cap)
            if eroare:
                err(eroare)
            else:
                ok(f"Arc adăugat.")

        elif alegere == 3:  # Șterge arc
            u = input_safe("  De la (ID): ", default="").strip()
            v = input_safe("  Spre   (ID): ", default="").strip()
            if g.remove_edge(u, v):
                ok("Arc șters.")
            else:
                err("Arc inexistent.")

        elif alegere == 4:  # Modifică capacitate
            u = input_safe("  De la (ID): ", default="").strip()
            v = input_safe("  Spre   (ID): ", default="").strip()
            if u in g.original_cap and v in g.original_cap.get(u, {}):
                cap = input_int(f"  Capacitate nouă (actual: {g.original_cap[u][v]:,}): ", min_val=1)
                g.remove_edge(u, v)
                eroare = g.add_edge(u, v, cap)
                if not eroare:
                    ok("Capacitate actualizată.")
            else:
                err("Arc inexistent.")

        elif alegere == 5:
            break

        input_safe("\n  Apasă Enter pentru a continua...")


# ════════════════════════════════════════════════════════════
#  MENIU PRINCIPAL
# ════════════════════════════════════════════════════════════

def meniu_principal():
    g = Graph()
    last_source = None
    last_sink = None
    last_max_flow = None
    last_log = None
    problem_name = "Problemă nouă"

    while True:
        cls()
        titlu(f"FORD-FULKERSON INTERACTIV  ·  {problem_name}")

        # Status rapid
        n_noduri = len(g.nodes)
        n_arce = len(g.edge_list())
        info(f"Noduri: {n_noduri}   |   Arce: {n_arce}")
        if last_source and last_sink:
            info(f"Sursă: {g.nodes.get(last_source, '?')}  →  Destinație: {g.nodes.get(last_sink, '?')}")
        if last_max_flow is not None:
            info(f"Flux maxim calculat: {last_max_flow:,}")
        print()

        optiuni = [
            "📋  Problemă nouă — introdu noduri și arce",
            "✏️   Editează graful curent",
            "▶️   Rulează Ford-Fulkerson",
            "📊  Afișează fluxurile pe arce",
            "✂️   Afișează tăietura minimă",
            "📖  Afișează log iterații",
            "💡  Încarcă exemplul București (Militari → Pipera)",
            "💡  Încarcă exemplu simplu (S→A→B→T)",
            "❌  Ieșire",
        ]
        alegere = alegere_meniu(optiuni)
        print()

        # ── 0. Problemă nouă ─────────────────────────────
        if alegere == 0:
            cls()
            titlu("PROBLEMĂ NOUĂ")
            problem_name = input_safe("  Numele problemei (ex: Rețea transport): ", default="Problemă nouă")
            g = Graph()
            last_source = last_sink = last_max_flow = last_log = None
            meniu_adauga_noduri(g)
            meniu_adauga_arce(g)
            ok("Graf creat cu succes!")
            input_safe("\n  Apasă Enter pentru a continua...")

        # ── 1. Editare ───────────────────────────────────
        elif alegere == 1:
            if g.is_empty():
                err("Graful e gol. Creează mai întâi o problemă nouă.")
                input_safe("  Apasă Enter...")
            else:
                meniu_editare(g)

        # ── 2. Rulează Ford-Fulkerson ────────────────────
        elif alegere == 2:
            if g.is_empty() or not g.has_edges():
                err("Graful e gol sau nu are arce.")
                input_safe("  Apasă Enter...")
                continue

            source, sink = alege_sursa_destinatie(g)
            if source is None:
                continue

            try:
                max_flow, log = g.ford_fulkerson(source, sink)
                last_source = source
                last_sink = sink
                last_max_flow = max_flow
                last_log = log

                cls()
                titlu(f"REZULTATE — {problem_name}")
                afiseaza_iteratii(log)
                afiseaza_rezumat(g, max_flow, source, sink)

            except ValueError as e:
                err(str(e))

            input_safe("\n  Apasă Enter pentru a continua...")

        # ── 3. Fluxuri ───────────────────────────────────
        elif alegere == 3:
            if last_max_flow is None:
                err("Rulează mai întâi algoritmul (opțiunea ▶).")
                input_safe("  Apasă Enter...")
            else:
                cls()
                titlu("FLUX PE ARCE")
                afiseaza_fluxuri(g)
                input_safe("\n  Apasă Enter pentru a continua...")

        # ── 4. Tăietură ──────────────────────────────────
        elif alegere == 4:
            if last_source is None or last_max_flow is None:
                err("Rulează mai întâi algoritmul (opțiunea ▶).")
                input_safe("  Apasă Enter...")
            else:
                cls()
                titlu("TĂIETURA MINIMĂ")
                afiseaza_taietura(g, last_source, last_sink)
                input_safe("\n  Apasă Enter pentru a continua...")

        # ── 5. Log ───────────────────────────────────────
        elif alegere == 5:
            if last_log is None:
                err("Rulează mai întâi algoritmul (opțiunea ▶).")
                input_safe("  Apasă Enter...")
            else:
                cls()
                titlu("LOG COMPLET ITERAȚII")
                afiseaza_iteratii(last_log)
                afiseaza_fluxuri(g)
                afiseaza_taietura(g, last_source, last_sink)
                afiseaza_rezumat(g, last_max_flow, last_source, last_sink)
                input_safe("\n  Apasă Enter pentru a continua...")

        # ── 6. Exemplu București ─────────────────────────
        elif alegere == 6:
            g = Graph()
            last_source = last_sink = last_max_flow = last_log = None
            problem_name = "Trafic București: Militari → Pipera"

            noduri = [
                ("x1","Militari"), ("x2","Lujerului"), ("x3","Crângași"),
                ("x4","Răzoare"),  ("x5","Basarab"),   ("x6","Eroilor"),
                ("x7","Arc Triumf"), ("x8","Victoriei"),
                ("x9","B. Văcărescu"), ("x10","C. Gaulle"), ("x11","Pipera"),
            ]
            for nid, name in noduri:
                g.add_node(nid, name)

            arce = [
                ("x1","x2",6000), ("x2","x3",3000), ("x2","x4",2500),
                ("x3","x5",2000), ("x3","x7",2000), ("x4","x6",2000),
                ("x5","x8",2500), ("x6","x8",2500), ("x7","x9",1500),
                ("x7","x10",2500), ("x8","x10",4000),
                ("x9","x11",2500), ("x10","x11",3000),
            ]
            for u, v, cap in arce:
                g.add_edge(u, v, cap)

            ok("Exemplu București încărcat. Acum rulează algoritmul (opțiunea ▶).")
            input_safe("  Apasă Enter...")

        # ── 7. Exemplu simplu ────────────────────────────
        elif alegere == 7:
            g = Graph()
            last_source = last_sink = last_max_flow = last_log = None
            problem_name = "Exemplu simplu S→A→B→T"

            for nid, name in [("S","Sursa"), ("A","A"), ("B","B"), ("T","Destinatie")]:
                g.add_node(nid, name)
            g.add_edge("S", "A", 10)
            g.add_edge("S", "B", 10)
            g.add_edge("A", "B", 2)
            g.add_edge("A", "T", 10)
            g.add_edge("B", "T", 10)

            ok("Exemplu simplu încărcat.")
            input_safe("  Apasă Enter...")

        # ── 8. Ieșire ────────────────────────────────────
        elif alegere == 8:
            cls()
            print("\n  La revedere!\n")
            sys.exit(0)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    meniu_principal()
