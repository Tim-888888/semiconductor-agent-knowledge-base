"""Strict, read-only manufacturing tools for the synthetic Phase 1 environment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str
    tool_id: str
    chamber: str | None = None
    time_range: str


class RecipeHistoryArguments(ToolArguments):
    recipe_id: str | None = None


class ManufacturingToolbox:
    """Expose bounded demo facts without arbitrary SQL, files, or equipment control."""

    def query_lot_yield(self, arguments: ToolArguments) -> dict[str, Any]:
        return self._fact(
            "query_lot_yield",
            arguments,
            f"模拟数据：{arguments.product} 在 {arguments.time_range} 的 ETCH 工序良率较基线下降 1.8%。",
        )

    def get_fdc_alarms(self, arguments: ToolArguments) -> dict[str, Any]:
        chamber = arguments.chamber or "未指定 Chamber"
        return self._fact(
            "get_fdc_alarms",
            arguments,
            f"模拟数据：{arguments.tool_id} {chamber} 出现短时 chamber pressure 波动和 RF match 偏离。",
        )

    def get_recipe_change_history(self, arguments: RecipeHistoryArguments) -> dict[str, Any]:
        return self._fact(
            "get_recipe_change_history",
            arguments,
            "模拟数据：当前 ETCH-ALPHA V2.3 已批准，最近 24 小时无未批准 Recipe 变更。",
        )

    @staticmethod
    def _fact(name: str, arguments: BaseModel, fact: str) -> dict[str, Any]:
        return {
            "evidence_id": f"tool:{name}",
            "source_type": "simulated_live_data",
            "tool": name,
            "read_only": True,
            "parameters": arguments.model_dump(exclude_none=True),
            "fact": fact,
            "observed_at": datetime.now(UTC).isoformat(),
        }

    def query_for_case(self, query: str, constraints: dict[str, Any]) -> list[dict[str, Any]]:
        required = ("product", "tool_id", "time_range")
        if any(not constraints.get(field) for field in required):
            return []
        base = ToolArguments(
            product=str(constraints["product"]),
            tool_id=str(constraints["tool_id"]),
            chamber=constraints.get("chamber"),
            time_range=str(constraints["time_range"]),
        )
        normal = query.lower()
        results: list[dict[str, Any]] = []
        if any(term in normal for term in ("良率", "yield", "下降", "异常", "根因")):
            results.append(self.query_lot_yield(base))
        if any(term in normal for term in ("pressure", "报警", "fdc", "首片", "异常", "根因")):
            results.append(self.get_fdc_alarms(base))
        if any(term in normal for term in ("recipe", "变更", "版本", "异常", "根因")):
            results.append(
                self.get_recipe_change_history(
                    RecipeHistoryArguments(**base.model_dump(), recipe_id=constraints.get("recipe_id"))
                )
            )
        return results


def query_demo_manufacturing_data(
    query: str,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper; incomplete inputs intentionally execute no tool."""

    return ManufacturingToolbox().query_for_case(query, constraints or {})
