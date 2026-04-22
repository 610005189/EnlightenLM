"""
Offline Review Service - 离线复盘服务
读取日志，生成自然语言报告
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReviewReport:
    """复盘报告"""
    session_id: str
    generated_at: str
    statistics: Dict[str, Any]
    cutoff_events: List[Dict[str, Any]]
    attention_patterns: List[str]
    safety_events: List[Dict[str, Any]]
    summary: str


class OfflineReviewService:
    """
    离线复盘服务

    功能:
    1. 读取审计日志
    2. 分析截断事件
    3. 分析注意力模式
    4. 生成自然语言报告
    """

    def __init__(self, audit_chain=None, l2_model=None):
        self.audit_chain = audit_chain
        self.l2_model = l2_model

    def generate_report(self, session_id: str) -> ReviewReport:
        """
        生成会话复盘报告

        Args:
            session_id: 会话ID

        Returns:
            ReviewReport: 复盘报告
        """
        logs = self._get_session_logs(session_id)
        snapshots = self._get_snapshots(session_id)

        cutoff_events = self._analyze_cutoffs(logs)
        attention_patterns = self._analyze_attention(snapshots)
        safety_events = self._analyze_safety(logs)
        statistics = self._compute_statistics(logs)

        summary = self._generate_summary(
            session_id, statistics, cutoff_events, attention_patterns
        )

        return ReviewReport(
            session_id=session_id,
            generated_at=datetime.now().isoformat(),
            statistics=statistics,
            cutoff_events=cutoff_events,
            attention_patterns=attention_patterns,
            safety_events=safety_events,
            summary=summary
        )

    def _get_session_logs(self, session_id: str) -> List[Dict]:
        """
        获取会话日志
        """
        if not self.audit_chain:
            return []

        logs = []
        for entry in self.audit_chain.chain:
            if entry.data.get("session_id") == session_id:
                logs.append(entry.to_dict())

        return logs

    def _get_snapshots(self, session_id: str) -> List[Dict]:
        """
        获取L2快照
        """
        if not self.l2_model:
            return []

        return []

    def _analyze_cutoffs(self, logs: List[Dict]) -> List[Dict[str, Any]]:
        """
        分析截断事件
        """
        cutoffs = []

        for log in logs:
            if log.get("data", {}).get("cutoff"):
                cutoffs.append({
                    "timestamp": log.get("timestamp"),
                    "reason": log.get("data", {}).get("cutoff_reason", "Unknown"),
                    "entropy": log.get("data", {}).get("attention_stats", {}).get("mean", 0)
                })

        return cutoffs

    def _analyze_attention(self, snapshots: List[Dict]) -> List[str]:
        """
        分析注意力模式
        """
        patterns = []

        if not snapshots:
            return ["无注意力数据"]

        avg_entropy = sum(s.get("entropy", 0) for s in snapshots) / len(snapshots) if snapshots else 0

        if avg_entropy < 0.3:
            patterns.append("注意力高度聚焦于特定token")
        elif avg_entropy > 0.8:
            patterns.append("注意力分布均匀，分散")
        else:
            patterns.append("注意力分布正常")

        return patterns

    def _analyze_safety(self, logs: List[Dict]) -> List[Dict[str, Any]]:
        """
        分析安全事件
        """
        safety_events = []

        for log in logs:
            if log.get("data", {}).get("van_event"):
                safety_events.append({
                    "timestamp": log.get("timestamp"),
                    "type": "VAN Event",
                    "severity": "High"
                })

        return safety_events

    def _compute_statistics(self, logs: List[Dict]) -> Dict[str, Any]:
        """
        计算统计信息
        """
        if not logs:
            return {
                "total_tokens": 0,
                "cutoff_count": 0,
                "avg_entropy": 0,
                "duration": 0
            }

        total_tokens = sum(log.get("data", {}).get("token_count", 0) for log in logs)
        cutoff_count = sum(1 for log in logs if log.get("data", {}).get("cutoff"))

        entropies = [
            log.get("data", {}).get("attention_stats", {}).get("mean", 0)
            for log in logs
        ]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        timestamps = [log.get("timestamp", 0) for log in logs]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        return {
            "total_tokens": total_tokens,
            "cutoff_count": cutoff_count,
            "avg_entropy": avg_entropy,
            "duration": duration,
            "log_count": len(logs)
        }

    def _generate_summary(
        self,
        session_id: str,
        statistics: Dict,
        cutoff_events: List[Dict],
        attention_patterns: List[str]
    ) -> str:
        """
        生成摘要
        """
        lines = [
            f"=== EnlightenLM 会话复盘报告 ===",
            f"",
            f"会话ID: {session_id}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"一、统计摘要",
            f"  - 总Token数: {statistics.get('total_tokens', 0)}",
            f"  - 截断次数: {statistics.get('cutoff_count', 0)}",
            f"  - 平均注意力熵: {statistics.get('avg_entropy', 0):.4f}",
            f"  - 会话时长: {statistics.get('duration', 0):.1f}秒",
            f"",
            f"二、截断事件分析",
        ]

        if cutoff_events:
            for i, event in enumerate(cutoff_events[:5]):
                lines.append(
                    f"  {i+1}. [{event.get('timestamp', 0):.1f}] {event.get('reason', 'Unknown')}"
                )
        else:
            lines.append("  无截断事件")

        lines.append("")
        lines.append("三、注意力模式分析")
        for pattern in attention_patterns:
            lines.append(f"  - {pattern}")

        lines.append("")
        lines.append("四、建议")
        if statistics.get("cutoff_count", 0) > 10:
            lines.append("  - 截断次数较多，建议检查任务类型是否适合")
        if statistics.get("avg_entropy", 1) < 0.3:
            lines.append("  - 注意力过于集中，可能影响生成多样性")

        return "\n".join(lines)


class SimpleReviewReportGenerator:
    """
    简单的复盘报告生成器
    用于快速生成报告
    """

    @staticmethod
    def generate(
        session_id: str,
        total_tokens: int,
        cutoff_count: int,
        avg_entropy: float
    ) -> str:
        """
        生成简单报告
        """
        return f"""
=== EnlightenLM 会话复盘报告 ===

会话ID: {session_id}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

统计摘要:
- 总Token数: {total_tokens}
- 截断次数: {cutoff_count}
- 平均注意力熵: {avg_entropy:.4f}

评估:
- {'生成过程正常' if cutoff_count == 0 else f'截断了{cutoff_count}次，需要关注'}
- {'注意力分布健康' if avg_entropy > 0.3 else '注意力过于聚焦'}
"""


class InteractiveReviewService:
    """
    交互式复盘服务
    支持查询和问答
    """

    def __init__(self, audit_chain=None):
        self.audit_chain = audit_chain
        self.report_cache = {}

    def query(self, session_id: str, question: str) -> str:
        """
        查询报告内容

        Args:
            session_id: 会话ID
            question: 问题

        Returns:
            answer: 回答
        """
        if session_id not in self.report_cache:
            service = OfflineReviewService(self.audit_chain)
            self.report_cache[session_id] = service.generate_report(session_id)

        report = self.report_cache[session_id]

        question_lower = question.lower()

        if "截断" in question_lower:
            return f"本次会话共截断{len(report.cutoff_events)}次"
        elif "熵" in question_lower or "注意力" in question_lower:
            return f"平均注意力熵为{report.statistics.get('avg_entropy', 0):.4f}"
        elif "token" in question_lower:
            return f"共生成{report.statistics.get('total_tokens', 0)}个token"
        else:
            return "抱歉，我无法理解您的问题"
