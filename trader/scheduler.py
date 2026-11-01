"""Harmonogram w trakcie sesji: czy bieżący slot (1h / 30min / 15min) jest należny.

Cron odpala `main.py --run` często (np. co 15 min); ta funkcja decyduje, czy
faktycznie uruchomić przebieg:
  - rynek otwarty (zegar Alpaca),
  - teraz >= otwarcie + first_run_after_open_min (zwykle 10:00 NY),
  - slot <= zamknięcie - last_run_before_close_min (zwykle 15:30 NY; skrócone sesje też),
  - dany slot nie był jeszcze przetworzony w tym trybie (live / dry_run).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from .broker import NY, ClockState, SessionHours
from .config import ExperimentCfg


@dataclass
class ScheduleDecision:
    due: bool
    reason: str
    slot_start: datetime | None = None


def evaluate_schedule(now: datetime, clock: ClockState, session: SessionHours | None, cfg: ExperimentCfg,
                      slot_processed: Callable[[datetime], bool]) -> ScheduleDecision:
    now = now.astimezone(NY)
    if not clock.is_open:
        return ScheduleDecision(False, f"rynek zamknięty (następne otwarcie {clock.next_open:%Y-%m-%d %H:%M} NY)")
    if session is None:
        return ScheduleDecision(False, "brak sesji w kalendarzu Alpaca na dziś")

    first_slot = session.open + timedelta(minutes=cfg.first_run_after_open_min)
    last_allowed = session.close - timedelta(minutes=cfg.last_run_before_close_min)
    if now < first_slot:
        return ScheduleDecision(False, f"za wcześnie — pierwszy slot o {first_slot:%H:%M} NY")
    if now > last_allowed:
        return ScheduleDecision(False, f"za blisko zamknięcia — ostatni slot najpóźniej {last_allowed:%H:%M} NY")

    step = timedelta(minutes=cfg.frequency_minutes)
    idx = int((now - first_slot) // step)
    slot = first_slot + idx * step
    if slot > last_allowed:
        return ScheduleDecision(False, f"brak slotu ({cfg.frequency}) przed {last_allowed:%H:%M} NY")
    if slot_processed(slot):
        return ScheduleDecision(False, f"slot {slot:%H:%M} NY już przetworzony", slot)
    return ScheduleDecision(True, f"slot {slot:%H:%M} NY ({cfg.frequency})", slot)
