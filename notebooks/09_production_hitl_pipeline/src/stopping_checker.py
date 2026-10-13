"""
stopping_checker.py - Bộ Rào Chắn Dừng 5 Tầng Tự Động Cho Production (Stopping Checker)
========================================================================================
Kiểm soát chi phí và chất lượng:
1. Target Ceiling Gate: Đạt hoặc vượt trần mAP của 100% dữ liệu ngẫu nhiên.
2. Plateau Saturation Gate: Bão hòa tiệm cận trần và delta < 0.3% qua 2 vòng liên tiếp.
3. Knowledge Saturation: Delta mAP < 0.3% qua 2 vòng -> dừng để không tốn tiền gán nhãn.
4. Uncertainty Depleted: Mẫu trong pool đã hết ca khó (Uncertainty < 0.15).
5. Budget Exhausted: Chạm hạn mức ngân sách tối đa (ví dụ 80% pool).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StoppingVerdict:
    """Bản án dừng của hệ thống."""
    should_stop: bool
    status: str  # "CONTINUE", "VICTORY_CONFIRMED", "TARGET_PLATEAU", "KNOWLEDGE_SATURATED", "UNCERTAINTY_DEPLETED", "BUDGET_EXHAUSTED", "USER_FORCED_STOP"
    reason: str
    current_map50: float
    target_map50: float
    delta_map50: float
    current_budget_percent: float


class ProductionStoppingChecker:
    """Bộ kiểm tra rào chắn dừng tự động cho Production."""

    def __init__(
        self,
        target_map50: float = 26.85,
        plateau_delta_threshold: float = 0.30,
        min_uncertainty_threshold: float = 0.15,
        max_budget_percent: float = 80.0,
        confirmation_rounds_needed: int = 1,
    ):
        self.target_map50 = target_map50
        self.plateau_delta_threshold = plateau_delta_threshold
        self.min_uncertainty_threshold = min_uncertainty_threshold
        self.max_budget_percent = max_budget_percent
        self.confirmation_rounds_needed = confirmation_rounds_needed
        self.consecutive_plateau_count = 0
        self.target_reached_count = 0

    def check(
        self,
        current_map50: float,
        previous_map50: Optional[float] = None,
        top_k_uncertainty: Optional[float] = None,
        current_budget_percent: float = 20.0,
        user_requested_stop: bool = False,
    ) -> StoppingVerdict:
        """Kiểm tra xem vòng lặp có nên dừng lại hay tiếp tục."""
        delta = round(current_map50 - (previous_map50 or current_map50), 2)

        # 0. Người dùng yêu cầu dừng chủ động
        if user_requested_stop:
            return StoppingVerdict(
                should_stop=True,
                status="USER_FORCED_STOP",
                reason="Người dùng chủ động yêu cầu kết thúc vòng lặp để xuất xưởng mô hình.",
                current_map50=current_map50,
                target_map50=self.target_map50,
                delta_map50=delta,
                current_budget_percent=current_budget_percent,
            )

        # 1. Tầng 1: Target Ceiling Gate (Chiến thắng)
        if current_map50 >= self.target_map50:
            self.target_reached_count += 1
            if self.target_reached_count >= self.confirmation_rounds_needed:
                return StoppingVerdict(
                    should_stop=True,
                    status="VICTORY_CONFIRMED",
                    reason=f"🏆 CHIẾN THẮNG TUYỆT ĐỐI: mAP50 đạt {current_map50:.2f}% đã vượt mốc trần Target={self.target_map50:.2f}%!",
                    current_map50=current_map50,
                    target_map50=self.target_map50,
                    delta_map50=delta,
                    current_budget_percent=current_budget_percent,
                )

        # 2. Tầng 2: Target Plateau Gate (Tiệm cận 95% trần và delta < plateau_delta_threshold qua 2 vòng)
        if current_map50 >= (0.95 * self.target_map50) and delta < self.plateau_delta_threshold:
            self.consecutive_plateau_count += 1
            if self.consecutive_plateau_count >= 2:
                return StoppingVerdict(
                    should_stop=True,
                    status="TARGET_PLATEAU",
                    reason=f"🛑 BÃO HÒA TIỆM CẬN TRẦN: Hiệu năng đạt {current_map50:.2f}% (>=95% Target) và delta={delta:.2f}% < {self.plateau_delta_threshold}%. Dừng để tiết kiệm chi phí.",
                    current_map50=current_map50,
                    target_map50=self.target_map50,
                    delta_map50=delta,
                    current_budget_percent=current_budget_percent,
                )
        else:
            self.consecutive_plateau_count = 0

        # 3. Tầng 3: Knowledge Saturation Gate (Delta < plateau_delta_threshold qua 2 vòng liên tiếp)
        if delta < self.plateau_delta_threshold:
            self.consecutive_flat_count = getattr(self, "consecutive_flat_count", 0) + 1
            if self.consecutive_flat_count >= 2:
                return StoppingVerdict(
                    should_stop=True,
                    status="KNOWLEDGE_SATURATED",
                    reason=f"🛑 BÃO HÒA TRI THỨC: Mô hình liên tiếp 2 vòng cải thiện không đáng kể (delta={delta:.2f}% < {self.plateau_delta_threshold}%). Dừng để tránh lãng phí ngân sách gán nhãn.",
                    current_map50=current_map50,
                    target_map50=self.target_map50,
                    delta_map50=delta,
                    current_budget_percent=current_budget_percent,
                )
        else:
            self.consecutive_flat_count = 0

        # 4. Tầng 4: Uncertainty Depleted Gate
        if top_k_uncertainty is not None and top_k_uncertainty < self.min_uncertainty_threshold:
            return StoppingVerdict(
                should_stop=True,
                status="UNCERTAINTY_DEPLETED",
                reason=f"📉 CẠN KIỆT MẪU KHÓ: Độ bất định Top-K={top_k_uncertainty:.3f} < {self.min_uncertainty_threshold}. Kho dữ liệu không còn ca thách thức.",
                current_map50=current_map50,
                target_map50=self.target_map50,
                delta_map50=delta,
                current_budget_percent=current_budget_percent,
            )

        # 4. Tầng 4: Survival Budget Limit
        if current_budget_percent >= self.max_budget_percent:
            return StoppingVerdict(
                should_stop=True,
                status="BUDGET_EXHAUSTED",
                reason=f"💰 TRẦN NGÂN SÁCH: Đã sử dụng {current_budget_percent:.1f}% pool, chạm hạn mức tối đa {self.max_budget_percent}%.",
                current_map50=current_map50,
                target_map50=self.target_map50,
                delta_map50=delta,
                current_budget_percent=current_budget_percent,
            )

        # Tiếp tục vòng lặp
        return StoppingVerdict(
            should_stop=False,
            status="CONTINUE",
            reason=f"Tiếp tục khai thác vòng mới (mAP50={current_map50:.2f}%, delta=+{delta:.2f}%).",
            current_map50=current_map50,
            target_map50=self.target_map50,
            delta_map50=delta,
            current_budget_percent=current_budget_percent,
        )
