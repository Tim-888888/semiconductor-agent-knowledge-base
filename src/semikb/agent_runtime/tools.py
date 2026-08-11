"""Read-only deterministic manufacturing tools for the Phase 1 demo."""

from __future__ import annotations


def query_demo_manufacturing_data(query: str) -> list[dict[str, str]]:
    """Return simulated facts, clearly labeled as demo data rather than Fab records."""

    normal = query.lower()
    results: list[dict[str, str]] = []
    if any(term in normal for term in ("良率", "yield", "下降")):
        results.append(
            {
                "source_type": "simulated_live_data",
                "tool": "query_lot_yield",
                "fact": "模拟数据：P-ALPHA 在最近 24 小时 ETCH 工序良率较基线下降 1.8%。",
            }
        )
    if any(term in normal for term in ("pressure", "报警", "fdc", "首片")):
        results.append(
            {
                "source_type": "simulated_live_data",
                "tool": "get_fdc_alarms",
                "fact": "模拟数据：ETCH-03 Chamber B 出现短时 chamber pressure 波动和 RF match 偏离。",
            }
        )
    if any(term in normal for term in ("recipe", "变更", "版本")):
        results.append(
            {
                "source_type": "simulated_live_data",
                "tool": "get_recipe_change_history",
                "fact": "模拟数据：当前 ETCH-ALPHA V2.3 已批准，最近 24 小时无未批准 Recipe 变更。",
            }
        )
    return results
