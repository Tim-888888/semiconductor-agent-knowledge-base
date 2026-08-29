"""Strict, read-only manufacturing tools for the synthetic Phase 1 environment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semikb.contracts.models import (
    ActorScope,
    GroupingDimension,
    IntentTarget,
    IntentTaskItem,
    TaskShape,
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str
    tool_id: str
    chamber: str | None = None
    time_range: str


class RecipeHistoryArguments(ToolArguments):
    recipe_id: str | None = None


class AggregateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_range: str
    group_by: list[GroupingDimension]
    authorized_products: list[str] = Field(default_factory=list)
    authorized_tools: list[str] = Field(default_factory=list)


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

    def query_manufacturing_aggregate(
        self,
        arguments: AggregateArguments,
        task: IntentTaskItem,
    ) -> dict[str, Any]:
        dimensions = "、".join(item.value for item in arguments.group_by) or "授权对象"
        if task.target_type is IntentTarget.FDC:
            subject = "FDC 报警"
        elif task.target_type is IntentTarget.SPC:
            subject = "SPC 越界"
        else:
            subject = "良率"
        groups = arguments.authorized_products or arguments.authorized_tools
        scope_text = "、".join(groups[:5]) or "当前授权范围"
        shape_text = {
            TaskShape.AGGREGATE_RANKING: "聚合排行",
            TaskShape.EVENT_LIST: "事件列表",
            TaskShape.TREND_ANALYSIS: "趋势汇总",
        }.get(task.task_shape, "聚合查询")
        return self._fact(
            "query_manufacturing_aggregate",
            arguments,
            (
                f"模拟数据：{arguments.time_range} 内按 {dimensions} 执行{shape_text}，"
                f"授权分组为 {scope_text}，查询对象为 {subject}。"
            ),
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

    def query_for_plan(
        self,
        query: str,
        constraints: dict[str, Any],
        task_items: list[IntentTaskItem],
        actor_scope: ActorScope,
    ) -> list[dict[str, Any]]:
        """Execute only the structured, read-only task plan within actor scope."""

        aggregate_task = next(
            (
                item
                for item in task_items
                if item.task_shape
                in {
                    TaskShape.AGGREGATE_RANKING,
                    TaskShape.EVENT_LIST,
                    TaskShape.TREND_ANALYSIS,
                }
            ),
            None,
        )
        if aggregate_task is None:
            return self.query_for_case(query, constraints)
        time_range = str(constraints.get("time_range", "")).strip()
        if not time_range:
            return []
        group_by = aggregate_task.group_by or [GroupingDimension.PRODUCT]
        arguments = AggregateArguments(
            time_range=time_range,
            group_by=group_by,
            authorized_products=list(actor_scope.products),
            authorized_tools=list(actor_scope.tool_ids),
        )
        return [self.query_manufacturing_aggregate(arguments, aggregate_task)]


def query_demo_manufacturing_data(
    query: str,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper; incomplete inputs intentionally execute no tool."""

    return ManufacturingToolbox().query_for_case(query, constraints or {})
