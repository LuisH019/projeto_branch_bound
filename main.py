"""
production_bb_project.py
Estrutura inicial para projeto Branch & Bound sobre o dataset de produção.
- Carrega JSON (Input_JSON_Schedule_Optimization.json)
- Constrói tabelas
- Calcula complexidade da máquina
- Calcula lucro unitário por produto baseado em complexidade
- Estima tempo de processamento por produto
- Implementa um Branch & Bound simples (0/1 selection / knapsack style)
- Produz logs (nós expandidos, nós podados, best_bound, best_solution)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import math
import heapq
import time

# ---------- CONFIGURÁVEIS ----------
JSON_PATH = Path("production-line.json")
# coeficiente que determina quanto a complexidade da máquina aumenta o lucro unitário
COMPLEXITY_PROFIT_COEF = 0.20
# capacidade total em "tempo" (minutos) usada para a seleção (ex.: sum máquinas * time_window)
USE_TOTAL_TIME_CAPACITY = True

# ---------- Leitura & EDA básica ----------
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_dataframes(data: dict):
    machines = pd.json_normalize(data["machines"]).explode("task_modes").rename(columns={"id": "machine_id", "task_modes": "task_mode"})
    task_modes = pd.json_normalize(data["task_modes"])
    tasks = pd.json_normalize(data["tasks"]).explode("task_modes").rename(columns={"id": "task_id", "task_modes": "task_mode"})
    products = pd.json_normalize(data["products"]).explode("tasks")
    # expand product tasks structure
    product_rows = []
    for p in data["products"]:
        pid = p["id"]
        for t in p["tasks"]:
            product_rows.append({"product_id": pid, "task": t["task"], "runs": t["runs"]})
    products_df = pd.DataFrame(product_rows)

    product_requests = pd.json_normalize(data.get("product_requests", []))
    return {
        "machines": machines,
        "task_modes": task_modes,
        "tasks": tasks,
        "products": products_df,
        "product_requests": product_requests,
        "raw": data
    }

# ---------- Métrica de complexidade da máquina ----------
def compute_machine_complexity(dataframes: Dict) -> pd.DataFrame:
    """
    Complexidade por máquina := normalizado( num_modes * mean_power_amplitude )
    - Para cada task_mode temos um vetor power. Podemos mapear modes->machines via bloco 'machines' do JSON.
    - Aqui: para cada máquina, encontramos todos os modos que ela suporta,
      calculamos média das médias de potência e multiplicamos por count modos.
    - Normalizamos entre [0,1].
    """
    raw = dataframes["raw"]
    # map task_mode id -> mean power and duration (len)
    mode_stats = {}
    for m in raw["task_modes"]:
        pid = m["id"]
        powers = m.get("power", [])
        mean_power = sum(powers) / len(powers) if len(powers) > 0 else 0.0
        duration = len(powers)  # assumimos cada entrada = 1 unidade de tempo
        mode_stats[pid] = {"mean_power": mean_power, "duration": duration}

    machine_stats = []
    for m in raw["machines"]:
        mid = m["id"]
        modes = m.get("task_modes", [])
        mode_count = len(modes)
        if mode_count == 0:
            mean_of_means = 0.0
            total_duration = 0
        else:
            mean_of_means = sum(mode_stats[mo]["mean_power"] for mo in modes) / mode_count
            total_duration = sum(mode_stats[mo]["duration"] for mo in modes)
        raw_complex = mode_count * mean_of_means
        machine_stats.append({
            "machine_id": mid,
            "mode_count": mode_count,
            "mean_power": mean_of_means,
            "total_duration_modes": total_duration,
            "raw_complexity": raw_complex
        })
    df = pd.DataFrame(machine_stats)
    # normalizar raw_complexity para [0,1]
    if df["raw_complexity"].max() > 0:
        df["complexity_norm"] = (df["raw_complexity"] - df["raw_complexity"].min()) / (df["raw_complexity"].max() - df["raw_complexity"].min())
    else:
        df["complexity_norm"] = 0.0
    return df

# ---------- Lucro unitário por produto (incorporando complexidade) ----------
def estimate_profit_and_time_per_product(dataframes: Dict, machine_complexity_df: pd.DataFrame,
                                         base_profit_per_run: float = 10.0,
                                         complexity_coef: float = COMPLEXITY_PROFIT_COEF) -> pd.DataFrame:
    """
    Para cada produto:
    - estimamos tempo: soma over tasks (runs * best_mode_duration)  -> assume minimal duration among modes available
    - estimamos lucro unitário base * (1 + complexity_factor), onde complexity_factor é média da complexidade
      das máquinas que podem executar as tarefas do produto.
    """
    raw = dataframes["raw"]
    # map mode -> duration
    mode_duration = {m["id"]: len(m.get("power", [])) for m in raw["task_modes"]}
    # map mode -> machines that can run it
    mode_to_machines = {}
    for m in raw["machines"]:
        mid = m["id"]
        for mo in m.get("task_modes", []):
            mode_to_machines.setdefault(mo, []).append(mid)

    # product -> tasks list already in dataframes["products"]
    product_rows = []
    products_df = dataframes["products"]
    grouped = products_df.groupby("product_id")
    for pid, group in grouped:
        tasks = group.to_dict("records")
        # estimate time:
        total_time = 0
        # collect set of machines that can perform product's tasks
        machines_for_product = set()
        for t in tasks:
            task_name = t["task"]
            runs = t["runs"]
            # Which modes correspond to this task?
            # from raw['tasks'] locate task entry
            task_modes = next((task["task_modes"] for task in raw["tasks"] if task["id"] == task_name), [])
            # for each mode, get duration; we choose the minimal duration available (heurística)
            durations = [mode_duration.get(mo, 0) for mo in task_modes]
            min_dur = min(durations) if durations else 0
            total_time += runs * min_dur
            # gather machines for these modes
            for mo in task_modes:
                machines = mode_to_machines.get(mo, [])
                for mm in machines:
                    machines_for_product.add(mm)
        # compute avg complexity among machines_for_product
        if machines_for_product:
            complexities = machine_complexity_df.set_index("machine_id").loc[list(machines_for_product), "complexity_norm"]
            avg_complexity = float(complexities.mean())
        else:
            avg_complexity = 0.0
        # base profit heuristic: base_profit_per_run * total_runs (or any other)
        total_runs = sum(r["runs"] for r in tasks)
        base_profit = base_profit_per_run * total_runs
        profit_unit = base_profit * (1.0 + complexity_coef * avg_complexity)
        product_rows.append({
            "product_id": pid,
            "estimated_time": total_time,
            "total_runs": total_runs,
            "avg_machine_complexity": avg_complexity,
            "base_profit": base_profit,
            "profit_unit": profit_unit,
            "machines_for_product": list(machines_for_product)
        })
    return pd.DataFrame(product_rows)

# ---------- Capacidade (proxy) ----------
def compute_total_time_capacity(raw: dict) -> float:
    """
    Compute capacity as: time_window * number of machines (simple proxy)
    Alternatively you could compute per-machine time windows due to maintenance or other limits.
    """
    time_window = raw["configuration"].get("time_window", 0)
    n_machines = len(raw.get("machines", []))
    return time_window * n_machines

# ---------- Branch & Bound (0/1 knapsack style) ----------
class BBNode:
    def __init__(self, level:int, profit:float, time_used:float, taken:List[int], bound:float):
        self.level = level            # index of the last considered item
        self.profit = profit
        self.time_used = time_used
        self.taken = taken            # list of 0/1 taken flags
        self.bound = bound

    def __lt__(self, other):
        # for max-heap by bound (heapq is min-heap so we invert)
        return self.bound > other.bound

def bound_estimate(items: List[dict], capacity: float, level: int, profit: float, time_used: float) -> float:
    """
    Bound: fractional knapsack relaxation from next item onward.
    items must be sorted by profit/time ratio desc.
    """
    if time_used >= capacity:
        return 0
    b = profit
    tot_time = time_used
    n = len(items)
    i = level + 1
    while i < n and tot_time + items[i]["estimated_time"] <= capacity:
        tot_time += items[i]["estimated_time"]
        b += items[i]["profit_unit"]
        i += 1
    # fractional part
    if i < n:
        remain = capacity - tot_time
        if items[i]["estimated_time"] > 0:
            b += items[i]["profit_unit"] * (remain / items[i]["estimated_time"])
    return b

def branch_and_bound_select(products_df: pd.DataFrame, capacity: float, time_limit_seconds: float = 60) -> Dict:
    start_time = time.time()
    # prepare items list sorted by profit/time ratio
    items = products_df.to_dict("records")
    for it in items:
        it["ratio"] = it["profit_unit"] / (it["estimated_time"] if it["estimated_time"] > 0 else 1e-6)
    items.sort(key=lambda x: x["ratio"], reverse=True)

    n = len(items)
    # root node
    root = BBNode(level=-1, profit=0.0, time_used=0.0, taken=[0]*n, bound=0.0)
    root.bound = bound_estimate(items, capacity, root.level, root.profit, root.time_used)

    # priority queue (max-heap by bound)
    heap = []
    heapq.heappush(heap, (-root.bound, root))
    best_profit = 0.0
    best_taken = None
    nodes_explored = 0
    nodes_pruned = 0

    while heap:
        if time.time() - start_time > time_limit_seconds:
            break
        _, node = heapq.heappop(heap)
        nodes_explored += 1
        if node.bound <= best_profit:
            nodes_pruned += 1
            continue
        # consider next level
        lvl = node.level + 1
        if lvl >= n:
            continue
        # child 1: take item lvl (if fits)
        take_time = node.time_used + items[lvl]["estimated_time"]
        if take_time <= capacity:
            taken1 = node.taken.copy()
            taken1[lvl] = 1
            profit1 = node.profit + items[lvl]["profit_unit"]
            child1 = BBNode(level=lvl, profit=profit1, time_used=take_time, taken=taken1, bound=0.0)
            child1.bound = bound_estimate(items, capacity, child1.level, child1.profit, child1.time_used)
            if profit1 > best_profit:
                best_profit = profit1
                best_taken = taken1
            if child1.bound > best_profit:
                heapq.heappush(heap, (-child1.bound, child1))
            else:
                nodes_pruned += 1
        else:
            nodes_pruned += 1

        # child 2: do not take lvl
        taken2 = node.taken.copy()
        taken2[lvl] = 0
        child2 = BBNode(level=lvl, profit=node.profit, time_used=node.time_used, taken=taken2, bound=0.0)
        child2.bound = bound_estimate(items, capacity, child2.level, child2.profit, child2.time_used)
        if child2.bound > best_profit:
            heapq.heappush(heap, (-child2.bound, child2))
        else:
            nodes_pruned += 1

    elapsed = time.time() - start_time
    # Map best_taken back to original product ids
    selected_products = []
    if best_taken is not None:
        for i, flag in enumerate(best_taken):
            if flag:
                selected_products.append(items[i]["product_id"])
    return {
        "best_profit": best_profit,
        "selected_products": selected_products,
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
        "elapsed_seconds": elapsed
    }

# ---------- MAIN flow ----------
def main():
    data = load_json(JSON_PATH)
    dfs = build_dataframes(data)
    mc_df = compute_machine_complexity(dfs)
    print("Machine complexity (sample):")
    print(mc_df.head())

    prod_info = estimate_profit_and_time_per_product(dfs, mc_df, base_profit_per_run=10.0, complexity_coef=COMPLEXITY_PROFIT_COEF)
    print("\nEstimated product info (sample):")
    print(prod_info.head())

    # Merge with product_requests to decide candidate list and amounts
    reqs = dfs["product_requests"]
    # Normalize product_requests: there can be duplicates (the example has ELASTIC W/ INSCR twice with different deadlines)
    reqs_agg = reqs.groupby("product").agg({"amount": "sum"}).reset_index().rename(columns={"product": "product_id"})
    candidates = prod_info.merge(reqs_agg, on="product_id", how="inner")
    # If estimated_time is zero, set small epsilon to avoid div by zero
    candidates["estimated_time"] = candidates["estimated_time"].apply(lambda x: x if x > 0 else 0.1)

    # capacity: simple proxy
    if USE_TOTAL_TIME_CAPACITY:
        capacity = compute_total_time_capacity(data)
    else:
        capacity = data["configuration"].get("time_window", 1152)  # fallback single-machine window

    print(f"\nUsing capacity (time units) = {capacity:.2f}")
    # run branch and bound
    bb_result = branch_and_bound_select(candidates, capacity=capacity, time_limit_seconds=30)
    print("\nBranch & Bound result:")
    print(bb_result)

if __name__ == "__main__":
    main()
