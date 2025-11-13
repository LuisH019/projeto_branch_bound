"""
main.py
Sistema de planejamento de produção com Branch & Bound (quantidades inteiras)
e dashboard interativo em Streamlit.
"""

import json
import heapq
import time
import pandas as pd
import streamlit as st
from pathlib import Path

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
JSON_PATH = Path("production-line.json")
COMPLEXITY_PROFIT_COEF = 0.2
TIME_LIMIT_SECONDS = 30

# ==========================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==========================
@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_machine_complexity(raw):
    """Complexidade = nº de modos * média das potências médias, normalizado"""
    mode_stats = {
        m["id"]: {
            "mean_power": sum(m.get("power", [])) / len(m.get("power", []))
            if m.get("power") else 0,
            "duration": len(m.get("power", [])),
        }
        for m in raw["task_modes"]
    }
    stats = []
    for mach in raw["machines"]:
        modes = mach.get("task_modes", [])
        mode_count = len(modes)
        mean_power = (
            sum(mode_stats[mo]["mean_power"] for mo in modes) / mode_count
            if mode_count else 0
        )
        raw_complex = mode_count * mean_power
        stats.append(
            dict(machine_id=mach["id"], mode_count=mode_count,
                 mean_power=mean_power, raw_complexity=raw_complex)
        )
    df = pd.DataFrame(stats)
    if df["raw_complexity"].max() > 0:
        df["complexity_norm"] = (
            (df["raw_complexity"] - df["raw_complexity"].min())
            / (df["raw_complexity"].max() - df["raw_complexity"].min())
        )
    else:
        df["complexity_norm"] = 0.0
    return df

def estimate_products(raw, mc_df, base_profit=10.0):
    """Calcula tempo e lucro unitário considerando complexidade média das máquinas."""
    mode_dur = {m["id"]: len(m.get("power", [])) for m in raw["task_modes"]}
    mode_to_machines = {}
    for m in raw["machines"]:
        for mo in m["task_modes"]:
            mode_to_machines.setdefault(mo, []).append(m["id"])

    rows = []
    for prod in raw["products"]:
        pid = prod["id"]
        total_time = 0
        machines = set()
        for t in prod["tasks"]:
            task = t["task"]
            runs = t["runs"]
            task_entry = next(x for x in raw["tasks"] if x["id"] == task)
            modes = task_entry["task_modes"]
            dur = min(mode_dur.get(m, 0) for m in modes) if modes else 0
            total_time += dur * runs
            for mo in modes:
                for mm in mode_to_machines.get(mo, []):
                    machines.add(mm)
        avg_complex = (
            mc_df.set_index("machine_id").loc[list(machines), "complexity_norm"].mean()
            if machines else 0
        )
        total_runs = sum(t["runs"] for t in prod["tasks"])
        base = base_profit * total_runs
        profit_unit = base * (1 + COMPLEXITY_PROFIT_COEF * avg_complex)
        rows.append(
            dict(product_id=pid, estimated_time=total_time, total_runs=total_runs,
                 avg_complex=avg_complex, profit_unit=profit_unit)
        )
    return pd.DataFrame(rows)

def prepare_candidates(raw, prod_df):
    reqs = pd.DataFrame(raw["product_requests"])
    reqs_agg = reqs.groupby("product").agg({"amount": "sum"}).reset_index()
    reqs_agg.rename(columns={"product": "product_id"}, inplace=True)
    merged = prod_df.merge(reqs_agg, on="product_id", how="inner")
    merged["estimated_time"] = merged["estimated_time"].replace(0, 0.1)
    capacity = raw["configuration"].get("time_window", 1152) * len(raw["machines"])
    return merged, capacity

# ==========================
# BRANCH & BOUND INTEIRO
# ==========================
class BBNode:
    def __init__(self, level, profit, time_used, quantities, bound):
        self.level = level
        self.profit = profit
        self.time_used = time_used
        self.quantities = quantities
        self.bound = bound
    def __lt__(self, other):
        return self.bound > other.bound  # max-heap behavior

def bound_estimate(items, capacity, level, profit, time_used):
    """Relaxação fracionária."""
    if time_used >= capacity:
        return 0
    b = profit
    tot_time = time_used
    for i in range(level + 1, len(items)):
        item = items[i]
        max_q = item["amount"]
        for q in range(1, max_q + 1):
            if tot_time + item["estimated_time"] > capacity:
                remain = capacity - tot_time
                if item["estimated_time"] > 0:
                    b += item["profit_unit"] * (remain / item["estimated_time"])
                return b
            tot_time += item["estimated_time"]
            b += item["profit_unit"]
    return b

def branch_and_bound_integer(items, capacity, time_limit=TIME_LIMIT_SECONDS):
    """Branch & Bound para quantidades inteiras (limitadas por amount)."""
    start = time.time()
    items = sorted(items, key=lambda x: x["profit_unit"]/x["estimated_time"], reverse=True)
    n = len(items)
    root = BBNode(-1, 0, 0, [0]*n, 0)
    root.bound = bound_estimate(items, capacity, -1, 0, 0)
    heap = [(-root.bound, root)]
    best_profit = 0
    best_quant = None
    explored, pruned = 0, 0
    logs = []

    while heap and time.time() - start < time_limit:
        _, node = heapq.heappop(heap)
        explored += 1
        if node.bound <= best_profit:
            pruned += 1
            continue
        lvl = node.level + 1
        if lvl >= n:
            continue
        item = items[lvl]
        for q in range(item["amount"], -1, -1):  # testa todas as quantidades possíveis
            t_new = node.time_used + q * item["estimated_time"]
            p_new = node.profit + q * item["profit_unit"]
            if t_new <= capacity:
                new_quant = node.quantities.copy()
                new_quant[lvl] = q
                child = BBNode(lvl, p_new, t_new, new_quant, 0)
                child.bound = bound_estimate(items, capacity, lvl, p_new, t_new)
                if p_new > best_profit:
                    best_profit = p_new
                    best_quant = new_quant
                if child.bound > best_profit:
                    heapq.heappush(heap, (-child.bound, child))
                else:
                    pruned += 1
            else:
                pruned += 1
        logs.append(dict(level=lvl, best_profit=best_profit,
                         bound=node.bound, explored=explored, pruned=pruned))
    return best_profit, best_quant, explored, pruned, pd.DataFrame(logs)

# ==========================
# STREAMLIT DASHBOARD
# ==========================
def main():
    st.set_page_config(page_title="Planejamento de Produção B&B", layout="wide")
    st.title("📦 Planejamento de Produção com Branch & Bound (Quantidades Inteiras)")
    raw = load_json(JSON_PATH)
    mc_df = compute_machine_complexity(raw)
    prod_df = estimate_products(raw, mc_df)
    candidates, capacity = prepare_candidates(raw, prod_df)

    st.sidebar.header("⚙️ Parâmetros")
    capacity_mult = st.sidebar.slider("Fator de Capacidade", 0.1, 2.0, 1.0, 0.1)
    capacity_adj = capacity * capacity_mult
    st.sidebar.write(f"Capacidade total ajustada: {capacity_adj:.0f} unidades de tempo")

    st.subheader("Dados dos Produtos")
    st.dataframe(candidates)

    if st.button("🚀 Executar Branch & Bound"):
        items = candidates.to_dict("records")
        with st.spinner("Executando algoritmo..."):
            best_profit, best_quant, explored, pruned, log_df = branch_and_bound_integer(items, capacity_adj)
        st.success("Execução concluída!")

        # Resultados
        res_df = candidates.copy()
        res_df["quantidade_produzida"] = best_quant
        res_df["lucro_total"] = res_df["profit_unit"] * res_df["quantidade_produzida"]
        total_time = (res_df["estimated_time"] * res_df["quantidade_produzida"]).sum()
        total_profit = res_df["lucro_total"].sum()

        st.metric("Lucro Total", f"${total_profit:,.2f}")
        st.metric("Tempo Usado", f"{total_time:.2f} / {capacity_adj:.2f}")
        st.metric("Nós explorados", explored)
        st.metric("Nós podados", pruned)

        st.subheader("📊 Produtos Selecionados")
        st.dataframe(res_df[res_df["quantidade_produzida"] > 0])

        # Gráficos
        st.subheader("Evolução do B&B")
        st.line_chart(log_df.set_index("level")[["best_profit", "bound"]])

        st.subheader("Distribuição de lucro por produto")
        st.bar_chart(res_df.set_index("product_id")["lucro_total"])

    else:
        st.info("Clique em **Executar Branch & Bound** para iniciar a otimização.")

if __name__ == "__main__":
    main()
