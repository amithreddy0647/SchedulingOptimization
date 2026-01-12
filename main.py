import os
import json
import re
import logging
import datetime as dt
from typing import Any, Dict, Optional, List

import requests
from flask import Flask, jsonify
from google.cloud import firestore

from openai import OpenAI

# -------------------------
# Config (ENV VARS)
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")  # change if you want

WEB_APP_URL = os.environ.get("WEB_APP_URL", "")
SHEETS_SHARED_SECRET = os.environ.get("SHEETS_SHARED_SECRET", "")

WRITE_BACK_TO_FIRESTORE = os.environ.get("WRITE_BACK_TO_FIRESTORE", "1") == "1"
NOTIFY_SHEET = os.environ.get("NOTIFY_SHEET", "1") == "1"


# -------------------------
# validator
# -------------------------
from typing import Any, Dict, List, Tuple, Optional

# ----------------------------
# Date helpers
# ----------------------------
def _parse_any_dt(v) -> dt.datetime:
    if v is None:
        raise ValueError("datetime is None")

    # Firestore Timestamp objects often come as datetime already
    if isinstance(v, dt.datetime):
        return v

    # Firestore TimestampWithNanoseconds sometimes has .to_datetime()
    if hasattr(v, "to_datetime"):
        return v.to_datetime()

    # Strings
    if isinstance(v, str):
        s = v.strip()
        # handle trailing Z
        s = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s)

    raise ValueError(f"Unsupported datetime type: {type(v)}")

def _parse_date(s: str) -> dt.date:
    # expects "YYYY-MM-DD"
    return dt.date.fromisoformat(s)

def _date_range(start: dt.date, end: dt.date) -> List[dt.date]:
    # [start, end) by day
    out = []
    cur = start
    while cur < end:
        out.append(cur)
        cur += dt.timedelta(days=1)
    return out

def _overlaps(a0: dt.datetime, a1: dt.datetime, b0: dt.datetime, b1: dt.datetime) -> bool:
    return a0 < b1 and a1 > b0

# ----------------------------
# Dogs / capacity rules
# ----------------------------
def _safe_date(s: str) -> dt.date:
    # expects YYYY-MM-DD
    return dt.date.fromisoformat(s)

def _dogs_norm(dogs: Dict[str, Any]) -> Dict[str, int]:
    d = dogs or {}
    small = int(d.get("small", 0) or 0)
    medium = int(d.get("medium", 0) or 0)
    large = int(d.get("large", 0) or 0)
    xl = int(d.get("xl", 0) or 0)
    total = int(d.get("total", small + medium + large + xl) or (small + medium + large + xl))
    return {"small": small, "medium": medium, "large": large, "xl": xl, "total": total}

def _nights_from_segment(seg: Dict[str, Any], default_nights: int) -> int:
    """
    segment is {"from_night":"YYYY-MM-DD","to_night":"YYYY-MM-DD"} (end-exclusive).
    """
    if not isinstance(seg, dict):
        return int(default_nights)

    fn = seg.get("from_night")
    tn = seg.get("to_night")
    if not fn or not tn:
        return int(default_nights)

    try:
        d0 = _safe_date(fn)
        d1 = _safe_date(tn)
        n = (d1 - d0).days
        return max(1, int(n))
    except Exception:
        return int(default_nights)

def _get_prices(rag: Dict[str, Any]) -> Dict[str, Any]:
    # tolerant: you might store prices in rag later
    prices = rag.get("prices") or {}
    base = prices.get("base") or {}
    return {
        "base_standard": float(base.get("standard", 69) or 69),
        "base_king": float(base.get("king", 99) or 99),
        "add_dog": float(prices.get("add_dog", 39) or 39),
    }

def _get_flip_cost_per_move(rag: Dict[str, Any]) -> float:
    flip = rag.get("flip_costs") or {}
    return float(flip.get("Standard", 11.13) or 11.13)


def _std_ok(agg: Dict[str, int]) -> bool:
    # allowed: empty OR (1 small) OR (2 small) OR (1 medium) OR (1 large)
    s, m, l, x, t = agg["small"], agg["medium"], agg["large"], agg["xl"], agg["total"]
    if x != 0: return False
    if t == 0: return True
    if t == 1:
        return (s, m, l) in [(1,0,0),(0,1,0),(0,0,1)]
    if t == 2:
        return (s, m, l) == (2,0,0)
    return False

def _king_ok(agg: Dict[str, int]) -> bool:
    # total<=4, units<=4 (s=1,m=2,l=3,x=4), xl exclusivity
    s, m, l, x, t = agg["small"], agg["medium"], agg["large"], agg["xl"], agg["total"]
    if t < 0 or t > 4: return False
    if x > 0:
        return (x == 1 and t == 1)  # exactly one XL alone
    units = s + 2*m + 3*l
    return units <= 4

def _room_ok(room_type: str, agg: Dict[str, int]) -> bool:
    rt = (room_type or "").lower()
    if rt == "standard": return _std_ok(agg)
    if rt == "king": return _king_ok(agg)
    return False

def _units(dogs: Dict[str, int]) -> int:
    return dogs["small"] + 2*dogs["medium"] + 3*dogs["large"] + 4*dogs["xl"]

# ----------------------------
# Validator + scorer
# ----------------------------
def validate_and_score_plan(plan: Dict[str, Any], rag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic:
    - builds per-room-night aggregates for request window
    - applies moves (for overlap slice) then confirms new
    - checks capacity each night and basic integrity
    - returns score+confidence based on rule compliance + "risk" metrics
    """

    violations: List[Dict[str, Any]] = []

    # --- basic schema
    status = plan.get("status")
    steps = plan.get("steps") or []
    if status not in ("Success", "Failure"):
        violations.append({"severity":"HIGH","code":"BAD_STATUS","message":"status must be Success or Failure"})
    if not isinstance(steps, list):
        violations.append({"severity":"HIGH","code":"BAD_STEPS","message":"steps must be a list"})
        steps = []

    # If plan says Success but has no steps -> reject
    if status == "Success" and len(steps) == 0:
        violations.append({"severity":"HIGH","code":"EMPTY_SUCCESS","message":"Success plan must include steps"})

    # --- room map
    rooms_all = rag.get("rooms", []) or []
    room_type = {}
    room_active = set()
    for r in rooms_all:
        rid = r.get("room_id") or r.get("id")
        rt = (r.get("type") or "").lower()
        if not rid: 
            continue
        room_type[rid] = rt
        if (r.get("status") or "active") == "active" and bool(r.get("allow_overnight", True)):
            room_active.add(rid)

    if not room_type:
        violations.append({"severity":"HIGH","code":"NO_ROOMS","message":"No rooms found in rag.rooms"})

    # --- request window
    req = rag.get("new_request") or {}
    req_id = req.get("request_id") or req.get("id") or "REQ-UNKNOWN"

    try:
        ci = _parse_any_dt(req["requested_range"]["check_in"])
        co = _parse_any_dt(req["requested_range"]["check_out"])
    except Exception as e:
        violations.append({
            "severity": "HIGH",
            "code": "BAD_REQUEST_DATES",
            "message": f"new_request.requested_range dates missing/invalid: {e}"
        })
        ci = dt.datetime(2000,1,1, tzinfo=dt.timezone.utc)
        co = dt.datetime(2000,1,2, tzinfo=dt.timezone.utc)


    # normalize policy times (match your code)
    ci = ci.replace(hour=15, minute=0, second=0, microsecond=0)
    co = co.replace(hour=12, minute=0, second=0, microsecond=0)

    req_from = ci.date()
    req_to = co.date()  # end-exclusive
    nights = _date_range(req_from, req_to)
    if len(nights) == 0:
        violations.append({"severity":"HIGH","code":"ZERO_NIGHTS","message":"Request window yields 0 nights"})

    # --- existing bookings state (only for overlap with request window)
    existing_all = rag.get("existing_bookings", []) or []
    existing = {}  # bid -> {room_id,start,end,dogs}
    for b in existing_all:
        bid = b.get("booking_id") or b.get("id") or b.get("doc_id")
        rid = b.get("room_id")
        if not bid or not rid: 
            continue
        try:
            bs = dt.datetime.fromisoformat((b.get("start_dt") or b.get("start") or b.get("start_date")).replace("Z","+00:00"))
            be = dt.datetime.fromisoformat((b.get("end_dt") or b.get("end") or b.get("end_date")).replace("Z","+00:00"))
        except Exception:
            continue

        # keep only if overlaps request window (at all)
        if not _overlaps(bs, be, ci, co):
            continue

        existing[bid] = {
            "booking_id": bid,
            "room_id": rid,
            "start": bs,
            "end": be,
            "dogs": _dogs(b.get("dogs", {})),
        }

    # --- build per-room-night aggregates for the request window
    # agg[room_id][night] = {"small":..,"medium":..,"large":..,"xl":..,"total":..}
    agg: Dict[str, Dict[dt.date, Dict[str,int]]] = {}
    def _ensure_room_night(rid: str, night: dt.date):
        agg.setdefault(rid, {})
        agg[rid].setdefault(night, {"small":0,"medium":0,"large":0,"xl":0,"total":0})

    # seed with existing bookings
    for b in existing.values():
        rid = b["room_id"]
        for n in nights:
            # night interval: [n 15:00, n+1 12:00) is approximate; use date overlap by checking datetime overlap
            # simpler: treat each night as [n 00:00, n+1 00:00)
            n0 = dt.datetime.combine(n, dt.time(0,0,0, tzinfo=ci.tzinfo))
            n1 = n0 + dt.timedelta(days=1)
            if _overlaps(b["start"], b["end"], n0, n1):
                _ensure_room_night(rid, n)
                d = b["dogs"]
                agg[rid][n]["small"] += d["small"]
                agg[rid][n]["medium"] += d["medium"]
                agg[rid][n]["large"] += d["large"]
                agg[rid][n]["xl"] += d["xl"]
                agg[rid][n]["total"] += d["total"]

    # --- apply steps (moves then confirms) into aggregates
    moves = 0
    confirms = 0
    parts = 0
    king_nights_used = 0
    std_nights_used = 0

    # Track new bookings for "split correctness"
    new_assignments: Dict[str, Dict[str, Any]] = {}  # booking_id -> {room, dogs, seg}

    def _seg_from_step(step: Dict[str, Any]) -> Tuple[dt.date, dt.date]:
        seg = step.get("segment") or {}
        try:
            a = _parse_date(seg.get("from_night", req_from.isoformat()))
            b = _parse_date(seg.get("to_night", req_to.isoformat()))
        except Exception:
            a, b = req_from, req_to
        return a, b  # [a,b)

    def _apply_delta(room_id: str, from_n: dt.date, to_n: dt.date, dogs: Dict[str,int], sign: int):
        for n in _date_range(from_n, to_n):
            if n not in nights:
                # ignore nights outside request horizon
                continue
            _ensure_room_night(room_id, n)
            agg[room_id][n]["small"] += sign * dogs["small"]
            agg[room_id][n]["medium"] += sign * dogs["medium"]
            agg[room_id][n]["large"] += sign * dogs["large"]
            agg[room_id][n]["xl"] += sign * dogs["xl"]
            agg[room_id][n]["total"] += sign * dogs["total"]

    # First pass: moves
    for idx, step in enumerate(steps):
        act = step.get("action")
        if act not in ("MoveExisting", "MoveSegmentExisting"):
            continue

        bid = step.get("booking_id")
        fr = step.get("from_room")
        to = step.get("to_room")

        if not bid or bid not in existing:
            violations.append({"severity":"HIGH","code":"MOVE_UNKNOWN_BOOKING","message":f"Step {idx}: Move booking_id not found in existing_bookings: {bid}"})
            continue

        orig = existing[bid]["room_id"]
        if fr and fr != orig:
            violations.append({"severity":"HIGH","code":"MOVE_FROM_MISMATCH","message":f"Step {idx}: from_room={fr} does not match current room_id={orig} for booking {bid}"})

        if (to not in room_type) or (to not in room_active):
            violations.append({"severity":"HIGH","code":"MOVE_BAD_ROOM","message":f"Step {idx}: to_room {to} unknown or inactive"})
            continue

        seg_from, seg_to = _seg_from_step(step) if act == "MoveSegmentExisting" else (req_from, req_to)

        # remove from current room, add to destination (for overlap slice)
        _apply_delta(existing[bid]["room_id"], seg_from, seg_to, existing[bid]["dogs"], sign=-1)
        _apply_delta(to, seg_from, seg_to, existing[bid]["dogs"], sign=+1)

        # update room in state (so subsequent moves use the updated location)
        existing[bid]["room_id"] = to
        moves += 1

    # Second pass: confirms
    for idx, step in enumerate(steps):
        act = step.get("action")
        if act != "ConfirmNew":
            continue

        bid = step.get("booking_id")
        to = step.get("to_room")
        dogs = _dogs(step.get("dogs", {}))
        seg_from, seg_to = _seg_from_step(step)

        if not bid:
            violations.append({"severity":"HIGH","code":"NEW_NO_ID","message":f"Step {idx}: ConfirmNew missing booking_id"})
            continue
        if (to not in room_type) or (to not in room_active):
            violations.append({"severity":"HIGH","code":"NEW_BAD_ROOM","message":f"Step {idx}: ConfirmNew to_room {to} unknown or inactive"})
            continue

        # Feasibility precheck for this sub-booking by itself (hard rule)
        rt = room_type.get(to, "")
        if rt == "king":
            if _units(dogs) > 4 or dogs["total"] > 4:
                violations.append({"severity":"HIGH","code":"KING_UNITS","message":f"Step {idx}: ConfirmNew {bid} violates king units/total (units={_units(dogs)}, total={dogs['total']})"})
        elif rt == "standard":
            if not _std_ok(dogs):
                violations.append({"severity":"HIGH","code":"STD_ILLEGAL_BUNDLE","message":f"Step {idx}: ConfirmNew {bid} bundle illegal for standard: {dogs}"})
        else:
            violations.append({"severity":"HIGH","code":"BAD_ROOM_TYPE","message":f"Step {idx}: room {to} has unsupported type {rt}"})

        _apply_delta(to, seg_from, seg_to, dogs, sign=+1)
        confirms += 1

        new_assignments[bid] = {"to_room": to, "dogs": dogs, "segment": (seg_from, seg_to)}
        parts += 1

        # usage metrics
        for n in _date_range(seg_from, seg_to):
            if n in nights:
                if rt == "king": king_nights_used += 1
                if rt == "standard": std_nights_used += 1

    # --- capacity check per room-night (after all steps)
    for rid, by_night in agg.items():
        rt = room_type.get(rid, "")
        for n, a in by_night.items():
            if any(v < 0 for v in a.values()):
                violations.append({"severity":"HIGH","code":"NEGATIVE_AGG","message":f"Negative aggregate in room={rid} night={n.isoformat()} agg={a}"})
                continue
            if not _room_ok(rt, a):
                violations.append({"severity":"HIGH","code":"CAPACITY_VIOLATION",
                                   "message":f"Capacity violation room={rid} night={n.isoformat()} agg={a} type={rt}"})

    # --- split sanity (soft): if user intended to split (P1,P2,...) ensure no duplicate IDs + segments sane
    seen = set()
    for st in steps:
        if st.get("action") == "ConfirmNew":
            bid = st.get("booking_id")
            if bid in seen:
                violations.append({"severity":"HIGH","code":"DUP_NEW_ID","message":f"Duplicate ConfirmNew booking_id: {bid}"})
            seen.add(bid)

    # --- Decide valid
    high = [v for v in violations if v["severity"] == "HIGH"]
    valid = (len(high) == 0)


    # ----------------------------
    # Scoring (0..10) + confidence (0..1)
    # ----------------------------
    if not valid:
        # If invalid, score is 0..3 depending on how close it was
        score = 1
        confidence = 0.05
    else:
        # Base
        score = 10.0
        confidence = 0.95

        # Penalize moves and fragmentation
        score -= 1.0 * max(0, moves)            # -1 per move
        score -= 0.5 * max(0, parts - 1)        # -0.5 per extra part beyond 1

        confidence -= 0.06 * max(0, moves)      # risk grows with moves
        confidence -= 0.03 * max(0, parts - 1)

        # Penalize king usage (preference only)
        # (scale by total request nights * parts to keep stable)
        denom = max(1, len(nights) * max(1, parts))
        king_ratio = king_nights_used / denom
        score -= 1.0 * king_ratio               # up to -1 if fully king
        confidence -= 0.03 * king_ratio

        # Tight capacity margin penalty: if any room-night in king has units==4 or std is at max exact
        tight = 0
        for rid, by_night in agg.items():
            rt = room_type.get(rid, "")
            for n, a in by_night.items():
                if rt == "king":
                    if _units(a) == 4:
                        tight += 1
                if rt == "standard":
                    # tight if 2 small or 1 medium or 1 large (any occupied is tight-ish)
                    if a["total"] > 0:
                        tight += 1

        if tight > 0:
            score -= min(1.0, 0.02 * tight)       # small deduction
            confidence -= min(0.08, 0.002 * tight)

        # Clamp
        score = max(0.0, min(10.0, score))
        confidence = max(0.0, min(1.0, confidence))
    

    # Approval gate using your thresholds (edit as you like)
    MIN_SCORE = 8.0
    MIN_CONF = 0.80
    approved = bool(valid and score >= MIN_SCORE and confidence >= MIN_CONF)

    # -----------------------------
    # ECONOMICS (define before return)
    # -----------------------------
    default_nights = max(1, len(nights))  # nights list already computed from request window
    pricing = _get_prices(rag)
    flip_per_move = _get_flip_cost_per_move(rag)

    def revenue_for_confirm(step: Dict[str, Any]) -> float:
        # Determine room type
        rid = step.get("to_room")
        rt = (room_type.get(rid, "standard") or "standard").lower()

        # Determine nights
        seg = step.get("segment") or {"from_night": req_from.isoformat(), "to_night": req_to.isoformat()}
        n = _nights_from_segment(seg, default_nights)

        # Dogs
        dogs = _dogs_norm(step.get("dogs") or {})
        total = dogs["total"]

        # Rate
        base = pricing["base_king"] if rt == "king" else pricing["base_standard"]
        add = pricing["add_dog"] * max(0, total - 1)
        return float(n * base + add)

    def flip_cost_for_move(step: Dict[str, Any]) -> float:
        # If model provides a cost, trust it; else use default per-move
        c = step.get("cost", None)
        if c is not None:
            try:
                return float(c)
            except Exception:
                return float(flip_per_move)

        # For both MoveExisting and MoveSegmentExisting
        if step.get("action") in ("MoveExisting", "MoveSegmentExisting"):
            return float(flip_per_move)

        return 0.0

    new_revenue = 0.0
    total_flip_cost = 0.0

    for st in steps:
        act = st.get("action")
        if act == "ConfirmNew":
            new_revenue += revenue_for_confirm(st)
        elif act in ("MoveExisting", "MoveSegmentExisting"):
            total_flip_cost += flip_cost_for_move(st)

    net_profit = float(new_revenue - total_flip_cost)

    # "Violation loss" for your tables:
    # If plan is not approved, assume you "lose" the net profit it *would* have produced.
    # If approved, loss is 0.
    violation_loss_est = float(max(0.0, net_profit) if not approved else 0.0)

    return {
        "valid": valid,
        "approved": approved,
        "score": round(float(score), 2),
        "confidence": round(float(confidence), 3),
        "metrics": {
            "request_id": req_id,
            "nights": len(nights),
            "moves": moves,
            "parts": parts,
            "king_nights_used": king_nights_used,
            "std_nights_used": std_nights_used,
        },
        "violations": violations[:200],  # keep logs bounded
        "net_profit": net_profit,
        "new_revenue": new_revenue,
        "total_flip_cost": total_flip_cost,
        "violation_loss_est": violation_loss_est,
    }

# -------------------------
# End Validator
# -------------------------
# -------------------------
# CP-SAT
# -------------------------

from typing import Any, Dict, List, Tuple

from typing import Any, Dict, Tuple

from ortools.sat.python import cp_model


# -------------------------
# Helpers (dates + dogs)
# -------------------------
def _parse_dt(v):
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, str):
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    raise ValueError(f"Bad datetime: {v!r}")

def _normalize_window(ci: dt.datetime, co: dt.datetime) -> Tuple[dt.datetime, dt.datetime]:
    ci = ci.replace(hour=15, minute=0, second=0, microsecond=0)
    co = co.replace(hour=12, minute=0, second=0, microsecond=0)
    return ci, co

def _segment_to_datetimes(seg: Dict[str, Any], tz=dt.timezone.utc) -> Tuple[dt.datetime, dt.datetime]:
    """
    segment: {"from_night":"YYYY-MM-DD","to_night":"YYYY-MM-DD"} where to_night is end-exclusive by date.
    We store:
      start_dt = from_night @ 15:00
      end_dt   = to_night   @ 12:00
    """
    from_date = dt.date.fromisoformat(seg["from_night"])
    to_date   = dt.date.fromisoformat(seg["to_night"])
    start_dt = dt.datetime(from_date.year, from_date.month, from_date.day, 15, 0, 0, tzinfo=tz)
    end_dt   = dt.datetime(to_date.year,   to_date.month,   to_date.day,   12, 0, 0, tzinfo=tz)
    return start_dt, end_dt

from google.cloud import firestore

def apply_plan_to_firestore(
    db: firestore.Client,
    request_id: str,
    rag: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    mark_request_status: str = "proposed",  # or "confirmed"
    booking_status: str = "Confirmed",            # or "Proposed"
    solver: str,
) -> None:
    """
    Writes CP-SAT results to Firestore.

    - MoveExisting: updates existing booking docs room_id -> to_room
    - ConfirmNew: creates booking docs with doc_id = booking_id (e.g., REQ-...-P1)
    - Patches requests/<request_id> with status + assigned rooms/booking ids
    """

    if plan.get("status") != "Success":
        raise ValueError("apply_plan_to_firestore called with non-success plan")

    req = rag.get("new_request", {}) or {}

    # Fallback window from request if step.segment is missing
    ci = _parse_dt(req["requested_range"]["check_in"])
    co = _parse_dt(req["requested_range"]["check_out"])
    ci, co = _normalize_window(ci, co)

    steps = plan.get("steps", []) or []

    # Collect confirms for request patch
    assigned_rooms = []
    booking_ids = []

    batch = db.batch()

    # 1) Apply moves (existing bookings)
    for st in steps:
        if st.get("action") == "MoveExisting":
            bid = st["booking_id"]
            to_room = st["to_room"]
            # Update the existing booking doc (assuming doc id == booking_id; if not, adjust lookup)
            ref = db.collection("bookings").document(bid)
            batch.set(ref, {"room_id": to_room, "moved_by": solver, "moved_at": dt.datetime.now(dt.timezone.utc)}, merge=True)

        elif st.get("action") == "MoveSegmentExisting":
            # If you're using segmented moves, you need a richer model (split existing bookings).
            # For now, store the intent + update room_id (simple).
            bid = st["booking_id"]
            to_room = st["to_room"]
            ref = db.collection("bookings").document(bid)
            batch.set(ref, {"room_id": to_room, "moved_by": solver, "moved_at": dt.datetime.now(dt.timezone.utc)}, merge=True)

    # 2) Create new bookings for ConfirmNew steps
    for st in steps:
        if st.get("action") != "ConfirmNew":
            continue

        new_bid = st.get("booking_id")
        to_room = st.get("to_room")
        dogs = st.get("dogs", {}) or {}

        if not new_bid or not to_room:
            continue

        # Determine segment dates
        seg = st.get("segment")
        if seg and seg.get("from_night") and seg.get("to_night"):
            start_dt, end_dt = _segment_to_datetimes(seg, tz=dt.timezone.utc)
        else:
            start_dt, end_dt = ci, co

        booking_doc = {
            "booking_id": new_bid,
            "request_id": request_id,
            "room_id": to_room,
            "status": booking_status,
            "source": solver,
            "dogs": dogs,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "created_at": dt.datetime.now(dt.timezone.utc),
            # Optional passthrough (if present in request)
            "contact": req.get("contact", {}),
        }

        bref = db.collection("bookings").document(new_bid)
        batch.set(bref, booking_doc, merge=True)

        assigned_rooms.append(to_room)
        booking_ids.append(new_bid)

    # 3) Patch request doc so you can SEE the outcome on requests/<request_id>
    req_ref = db.collection("requests").document(request_id)
    batch.set(
        req_ref,
        {
            "status": mark_request_status,  # "cpsat_proposed" or "confirmed"
            "solver": solver,
            "plan": plan,
            "assigned_rooms": assigned_rooms,
            "booking_ids": booking_ids,
            "planned_at": dt.datetime.now(dt.timezone.utc),
        },
        merge=True,
    )

    batch.commit()


def _night_list(ci: dt.datetime, co: dt.datetime) -> List[dt.date]:
    # nights are dates: [ci.date(), ..., co.date()-1]
    d0 = ci.date()
    d1 = co.date()
    out = []
    cur = d0
    while cur < d1:
        out.append(cur)
        cur += dt.timedelta(days=1)
    return out

def _overlaps_night(b_start: dt.datetime, b_end: dt.datetime, night: dt.date, tzinfo=None) -> bool:
    """
    Policy night window: [night 15:00, next day 12:00)
    A booking overlaps the night if it intersects that window.
    """
    tz = tzinfo or b_start.tzinfo  # keep consistent timezone

    ns = dt.datetime.combine(night, dt.time(15, 0, 0), tzinfo=tz)
    ne = dt.datetime.combine(night + dt.timedelta(days=1), dt.time(12, 0, 0), tzinfo=tz)

    return (b_start < ne) and (b_end > ns)


def _dogs(d: Dict[str, Any]) -> Dict[str, int]:
    dd = d or {}
    s = int(dd.get("small", 0) or 0)
    m = int(dd.get("medium", 0) or 0)
    l = int(dd.get("large", 0) or 0)
    x = int(dd.get("xl", 0) or 0)
    t = int(dd.get("total", s + m + l + x) or (s + m + l + x))
    # trust total but keep consistent
    return {"small": s, "medium": m, "large": l, "xl": x, "total": t}

def _fits_single_room(room_type: str, dogs: Dict[str, int]) -> bool:
    s, m, l, x, t = dogs["small"], dogs["medium"], dogs["large"], dogs["xl"], dogs["total"]

    if room_type == "standard":
        # Allowed only: empty OR 1 small OR 2 small OR 1 medium OR 1 large
        if t < 1 or t > 2:
            return False
        if x > 0:
            return False
        # no mixing
        if m > 0 and (s > 0 or l > 0):
            return False
        if l > 0 and (s > 0 or m > 0):
            return False
        if t == 1:
            return (s == 1 and m == 0 and l == 0) or (m == 1 and s == 0 and l == 0) or (l == 1 and s == 0 and m == 0)
        # t == 2
        return (s == 2 and m == 0 and l == 0)

    if room_type == "king":
        # total <= 4, units <= 4, XL alone rule
        if t < 1 or t > 4:
            return False
        if x > 0:
            return (x == 1 and t == 1)  # exactly 1 XL only
        units = 1*s + 2*m + 3*l
        return units <= 4

    return False


# -------------------------
# CP-SAT plan
# -------------------------
# def cpsat_plan(rag: Dict[str, Any], max_moves: int = 4, time_limit_s: float = 3.0) -> Dict[str, Any]:
#     req = rag["new_request"]
#     request_id = req.get("request_id") or req.get("id") or "REQ-UNKNOWN"

#     ci = _parse_dt(req["requested_range"]["check_in"])
#     co = _parse_dt(req["requested_range"]["check_out"])
#     ci, co = _normalize_window(ci, co)
#     nights = _night_list(ci, co)

#     # rooms we allow
#     rooms_all = rag.get("rooms", []) or []
#     rooms = []
#     for r in rooms_all:
#         if (r.get("status") or "active") != "active":
#             continue
#         if not bool(r.get("allow_overnight", True)):
#             continue
#         rt = (r.get("type") or "").lower()
#         if rt not in ("standard", "king"):
#             continue
#         rid = r.get("room_id") or r.get("id")
#         if not rid:
#             continue
#         rooms.append({"room_id": rid, "type": rt})

#     if not rooms:
#         return {"status": "Failure", "message": "No active overnight rooms.", "steps": []}

#     room_ids = [r["room_id"] for r in rooms]
#     room_type = {r["room_id"]: r["type"] for r in rooms}
#     king_rooms = [r["room_id"] for r in rooms if r["type"] == "king"]
#     std_rooms  = [r["room_id"] for r in rooms if r["type"] == "standard"]

#     # existing bookings (only those overlapping the request horizon)
#     existing_all = rag.get("existing_bookings", []) or []
#     existing = []
#     for b in existing_all:
#         bid = b.get("booking_id") or b.get("id") or b.get("doc_id")
#         rid = b.get("room_id")
#         if not bid or not rid:
#             continue
#         try:
#             bs = _parse_dt(b.get("start_dt") or b.get("start") or b.get("start_date"))
#             be = _parse_dt(b.get("end_dt") or b.get("end") or b.get("end_date"))
#         except Exception:
#             continue

#         # keep only if overlaps at least one night in the request window
#         if not any(_overlaps_night(bs, be, n, tzinfo=ci.tzinfo) for n in nights):
#             continue


#         dogs = _dogs(b.get("dogs", {}))
#         existing.append({"booking_id": bid, "orig_room": rid, "start": bs, "end": be, "dogs": dogs})

#     # split the NEW request into single-dog parts (general + avoids illegal bundling)
#     req_dogs = _dogs(req.get("dogs", {}))
#     parts = []
#     k = 1
#     for size in ("xl", "large", "medium", "small"):
#         for _ in range(req_dogs[size]):
#             d = {"small": 0, "medium": 0, "large": 0, "xl": 0, "total": 1}
#             d[size] = 1
#             parts.append({"part_id": f"{request_id}-P{k}", "dogs": d})
#             k += 1

#     if not parts:
#         return {"status": "Failure", "message": "Request has 0 dogs.", "steps": []}

#     # model
#     model = cp_model.CpModel()

#     # decision vars: assign each existing booking to 1 room (possibly moved)
#     x_exist = {}  # (i, rid) -> Bool
#     for i, b in enumerate(existing):
#         feasible_rooms = []
#         for rid in room_ids:
#             if _fits_single_room(room_type[rid], b["dogs"]):
#                 feasible_rooms.append(rid)
#         if not feasible_rooms:
#             return {
#                 "status": "Failure",
#                 "message": f"Existing booking {b['booking_id']} has no feasible room type under current rules.",
#                 "steps": []
#             }

#         for rid in feasible_rooms:
#             x_exist[(i, rid)] = model.NewBoolVar(f"xE_{i}_{rid}")
#         model.Add(sum(x_exist[(i, rid)] for rid in feasible_rooms) == 1)

#     # decision vars: assign each new part to 1 room
#     x_new = {}  # (p, rid) -> Bool
#     for p, part in enumerate(parts):
#         feasible_rooms = []
#         for rid in room_ids:
#             if _fits_single_room(room_type[rid], part["dogs"]):
#                 feasible_rooms.append(rid)
#         if not feasible_rooms:
#             return {
#                 "status": "Failure",
#                 "message": f"New part {part['part_id']} cannot fit in any room type (rules too strict?).",
#                 "steps": []
#             }
#         for rid in feasible_rooms:
#             x_new[(p, rid)] = model.NewBoolVar(f"xN_{p}_{rid}")
#         model.Add(sum(x_new[(p, rid)] for rid in feasible_rooms) == 1)

#     # moves count for existing bookings
#     moved_vars = []
#     for i, b in enumerate(existing):
#         orig = b["orig_room"]
#         # if orig not in model vars, then it MUST move
#         orig_var = x_exist.get((i, orig), None)
#         moved = model.NewBoolVar(f"moved_{i}")
#         if orig_var is None:
#             model.Add(moved == 1)
#         else:
#             # moved == 1 - assigned_to_orig
#             model.Add(moved + orig_var == 1)
#         moved_vars.append(moved)

#     if moved_vars:
#         model.Add(sum(moved_vars) <= max_moves)

#     # per room-night capacity constraints (this is the key part)
#     # We build only LinearExpr terms with nonzero coefficients (avoids your TypeError)
#     for rid in room_ids:
#         rt = room_type[rid]

#         for n_idx, night in enumerate(nights):
#             # sums for this room-night
#             small_terms = []
#             med_terms   = []
#             large_terms = []
#             xl_terms    = []

#             # existing contributions
#             for i, b in enumerate(existing):
#                 if not _overlaps_night(b["start"], b["end"], night):
#                     continue
#                 xi = x_exist.get((i, rid))
#                 if xi is None:
#                     continue

#                 ds = b["dogs"]["small"]
#                 dm = b["dogs"]["medium"]
#                 dl = b["dogs"]["large"]
#                 dx = b["dogs"]["xl"]

#                 if ds: small_terms.append(ds * xi)
#                 if dm: med_terms.append(dm * xi)
#                 if dl: large_terms.append(dl * xi)
#                 if dx: xl_terms.append(dx * xi)

#             # new parts contributions (they overlap every night in request window)
#             for p, part in enumerate(parts):
#                 xp = x_new.get((p, rid))
#                 if xp is None:
#                     continue
#                 ds = part["dogs"]["small"]
#                 dm = part["dogs"]["medium"]
#                 dl = part["dogs"]["large"]
#                 dx = part["dogs"]["xl"]
#                 if ds: small_terms.append(ds * xp)
#                 if dm: med_terms.append(dm * xp)
#                 if dl: large_terms.append(dl * xp)
#                 if dx: xl_terms.append(dx * xp)

#             S = cp_model.LinearExpr.Sum(small_terms) if small_terms else 0
#             M = cp_model.LinearExpr.Sum(med_terms)   if med_terms   else 0
#             L = cp_model.LinearExpr.Sum(large_terms) if large_terms else 0
#             X = cp_model.LinearExpr.Sum(xl_terms)    if xl_terms    else 0

#             if rt == "standard":
#                 # XL forbidden
#                 model.Add(X == 0)
#                 # at most one non-small dog total
#                 model.Add(M <= 1)
#                 model.Add(L <= 1)
#                 model.Add(M + L <= 1)

#                 # enforce "no mixing": if M+L == 1 then S must be 0
#                 has_non_small = model.NewBoolVar(f"std_has_non_small_{rid}_{n_idx}")
#                 # has_non_small == (M+L)
#                 model.Add(M + L == has_non_small)
#                 # if has_non_small=1 => S=0; if 0 => S<=2
#                 model.Add(S <= 2 * (1 - has_non_small))

#             else:
#                 # king
#                 total = S + M + L + X
#                 units = S + 2*M + 3*L + 4*X

#                 model.Add(total <= 4)
#                 model.Add(units <= 4)
#                 model.Add(X <= 1)
#                 # XL exclusivity: if X==1 then total must be 1 (so no other dogs)
#                 model.Add(total <= 4 - 3*X)

#     # objective: prefer standard + fewer moves
#     # (minimize moves heavily; also minimize number of new parts placed in king)
#     king_use_terms = []
#     for p in range(len(parts)):
#         for rid in king_rooms:
#             v = x_new.get((p, rid))
#             if v is not None:
#                 king_use_terms.append(v)

#     move_cost = cp_model.LinearExpr.Sum(moved_vars) if moved_vars else 0
#     king_use  = cp_model.LinearExpr.Sum(king_use_terms) if king_use_terms else 0

#     model.Minimize(1000 * move_cost + 10 * king_use)

#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = float(time_limit_s)
#     solver.parameters.num_search_workers = 8

#     res = solver.Solve(model)
#     # --- diagnostics ---
#     try:
#         status_name = solver.StatusName(res)
#     except Exception:
#         status_name = str(res)

#     diag = {
#         "solver_status": status_name,
#         "wall_time_s": float(solver.WallTime()),
#         "branches": int(solver.NumBranches()),
#         "conflicts": int(solver.NumConflicts()),
#     }

#     # Log it so it shows in run.googleapis.com/stderr
#     app.logger.info(
#         "CPSAT diagnostics status=%s wall=%.3f branches=%d conflicts=%d",
#         diag["solver_status"], diag["wall_time_s"], diag["branches"], diag["conflicts"]
#     )

#     if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         return {"status": "Failure", "message": "No feasible plan found within time limit.", "steps": [],
#         "diag": diag,}

#     steps = []

#     # emit moves
#     for i, b in enumerate(existing):
#         assigned = None
#         for rid in room_ids:
#             v = x_exist.get((i, rid))
#             if v is not None and solver.Value(v) == 1:
#                 assigned = rid
#                 break
#         if assigned and assigned != b["orig_room"]:
#             steps.append({
#                 "action": "MoveExisting",
#                 "booking_id": b["booking_id"],
#                 "from_room": b["orig_room"],
#                 "to_room": assigned,
#                 "cost": 0
#             })

#     # emit confirms
#     for p, part in enumerate(parts):
#         assigned = None
#         for rid in room_ids:
#             v = x_new.get((p, rid))
#             if v is not None and solver.Value(v) == 1:
#                 assigned = rid
#                 break
#         steps.append({
#             "action": "ConfirmNew",
#             "booking_id": part["part_id"],
#             "to_room": assigned,
#             "segment": {"from_night": nights[0].isoformat(), "to_night": (nights[-1] + dt.timedelta(days=1)).isoformat()},
#             "dogs": part["dogs"],
#             "cost": 0
#         })

#     return {"status": "Success", "message": "CP-SAT plan found.", "steps": steps}

# -------------------------
# CP-SAT Ends
# -------------------------


# -------------------------
# Flask / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

db = firestore.Client()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not openai_client:
    app.logger.warning("OpenAI client NOT initialized (missing OPENAI_API_KEY).")

# -------------------------
# Prompt (NO VALIDATION)
# -------------------------
PLANNER_SYSTEM = """
You are a reservation optimizer for a multi-tenant pet boarding facility.

You MUST output ONLY JSON.
Do NOT include markdown, explanations, or extra text.

GOAL:
- Produce a feasible placement plan if possible.
- Prefer using STANDARD rooms when possible. Use KING rooms only when necessary.
- Prefer fewer moves. Keep total moves <= 4.

ROOM RULES (IMPORTANT) — aggregate per room-night:
STANDARD:
- Allowed aggregate states per room-night:
  empty OR (1 small) OR (2 small) OR (1 medium) OR (1 large)
- No mixed sizes, no XL.

KING:
- total dogs <= 4
- units <= 4 where units = small*1 + medium*2 + large*3 + xl*4
- if xl > 0 then must be exactly 1 xl AND total dogs must be exactly 1.

HARD RULE (DO NOT VIOLATE):
- Every ConfirmNew step represents ONE sub-booking, and that sub-booking must be feasible by itself in the chosen room type for every night.
- In KING: units = small*1 + medium*2 + large*3 + xl*4 must be <= 4 for each ConfirmNew step.
- If a dog bundle is not feasible in STANDARD and not feasible in KING, you MUST SPLIT into multiple ConfirmNew parts until each part is feasible.
  Example: (1 medium + 1 large) together is NOT feasible anywhere (STANDARD forbids mixing; KING units=2+3=5>4) so it MUST be split.

SPLITTING:
- You may split the new request into multiple ConfirmNew parts.
- Each part must be feasible for its chosen room type across all nights.
- Do NOT place multiple parts into the same room-night if that would violate that room’s aggregate capacity rules.

MOVES (USE WHEN NEEDED):
- If no feasible plan exists using only ConfirmNew (with splitting allowed),
  you may move existing bookings to other rooms to free capacity.
- Use MoveSegmentExisting steps (moves can be partial over the request night window).
- A move is allowed only if, after the move, ALL room-night aggregate capacity rules remain satisfied.
- General strategy:
  1) Try direct placement (no moves, no split).
  2) Try split-only (no moves).
  3) Try moves to consolidate occupants into fewer rooms (into KING only when legal),
     freeing an entire room for the new request parts.
  4) Use <= 4 total move steps.

OUTPUT FORMAT (ONLY JSON):

SUCCESS:
{
  "status": "Success",
  "message": "Plan found.",
  "steps": [
    {
      "action": "MoveSegmentExisting",
      "booking_id": "<existing_booking_id>",
      "from_room": "<room_id>",
      "to_room": "<room_id>",
      "segment": {"from_night":"YYYY-MM-DD","to_night":"YYYY-MM-DD"},
      "cost": 0
    },
    {
      "action": "ConfirmNew",
      "booking_id": "<request_id>-P1",
      "to_room": "<room_id>",
      "segment": {"from_night":"YYYY-MM-DD","to_night":"YYYY-MM-DD"},
      "dogs": {"small":0,"medium":1,"large":0,"xl":0,"total":1},
      "cost": 0
    }
  ]
}

FAILURE:
{
  "status": "Failure",
  "message": "No feasible plan found (even with splitting and up to 4 moves).",
  "steps": []
}
""".strip()

# -------------------------
# Helpers
# -------------------------
def log_eval_run(db, *, scenario_id: str, method: str, plan: Dict[str, Any],
                 valid: bool, approved: bool, net_profit: float,
                 moves: int, king_used: int, runtime_ms: int,
                 violations: List[Dict[str, Any]]):
    doc = {
        "scenario_id": scenario_id,
        "method": method,
        "status": plan.get("status", "Failure"),
        "valid": bool(valid),
        "approved": bool(approved),
        "net_profit": float(net_profit or 0.0),
        "moves": int(moves or 0),
        "king_used": int(king_used or 0),
        "runtime_ms": int(runtime_ms or 0),
        "violations": violations or [],
        "ts": dt.datetime.now(dt.timezone.utc),
    }
    db.collection("eval_runs").add(doc)

def extract_json(text: str) -> Dict[str, Any]:
    """
    Model might wrap JSON in ```json ... ```
    Strip wrappers and parse JSON.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    return json.loads(t)

def parse_dt(v):
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, str):
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    if hasattr(v, "isoformat"):
        return dt.datetime.fromisoformat(v.isoformat())
    raise ValueError(f"Unsupported datetime: {v}")

def normalize_window(ci: dt.datetime, co: dt.datetime):
    ci = ci.replace(hour=15, minute=0, second=0, microsecond=0)
    co = co.replace(hour=12, minute=0, second=0, microsecond=0)
    return ci, co

def nights_segment(req: Dict[str, Any]) -> Dict[str, str]:
    ci = parse_dt(req["requested_range"]["check_in"])
    co = parse_dt(req["requested_range"]["check_out"])
    ci, co = normalize_window(ci, co)
    return {"from_night": ci.date().isoformat(), "to_night": co.date().isoformat()}  # end-exclusive

def overlaps(a_start: dt.datetime, a_end: dt.datetime, b_start: dt.datetime, b_end: dt.datetime) -> bool:
    return a_start < b_end and a_end > b_start

def filter_bookings_for_window(existing_bookings: List[Dict[str, Any]], ci: dt.datetime, co: dt.datetime) -> List[Dict[str, Any]]:
    """
    Not validation — just reduces prompt size by sending only bookings that overlap the request window.
    """
    out = []
    for b in existing_bookings:
        try:
            s = parse_dt(b.get("start_dt") or b.get("start") or b.get("start_date") or b.get("check_in"))
            e = parse_dt(b.get("end_dt") or b.get("end") or b.get("end_date") or b.get("check_out"))
            if s and e and overlaps(ci, co, s, e):
                out.append(b)
        except Exception:
            continue
    return out

def notify_sheet(request_id: str, rag: Dict[str, Any], plan: Optional[Dict[str, Any]], status: str, error_msg: str = ""):
    if not (WEB_APP_URL and NOTIFY_SHEET):
        return
    sheet_row = (rag.get("new_request", {}) or {}).get("sheet_row")
    if not sheet_row:
        return

    assigned_rooms: List[str] = []
    booking_ids: List[str] = []
    if plan:
        for st in plan.get("steps", []):
            if st.get("action") == "ConfirmNew":
                assigned_rooms.append(st.get("to_room"))
                booking_ids.append(st.get("booking_id"))

    payload = {
        "request_id": request_id,
        "sheet_row": int(sheet_row),
        "status": status,
        "error_msg": error_msg,
        "assigned_rooms": assigned_rooms,
        "booking_ids": booking_ids,
    }

    headers = {}
    if SHEETS_SHARED_SECRET:
        headers["X-Shared-Secret"] = SHEETS_SHARED_SECRET

    try:
        requests.post(WEB_APP_URL, json=payload, headers=headers, timeout=30, allow_redirects=True)
    except Exception as e:
        app.logger.warning(f"notify_sheet failed: {e}")

# -------------------------
# Firestore fetch
# -------------------------
def fetch_rag(request_id: str) -> Dict[str, Any]:
    req_snap = db.collection("requests").document(request_id).get()
    if not req_snap.exists:
        raise FileNotFoundError(f"Request {request_id} not found")

    new_request = req_snap.to_dict() or {}

    rooms = [d.to_dict() for d in db.collection("rooms").stream()]
    existing_bookings = [d.to_dict() for d in db.collection("bookings").stream()]

    return {"new_request": new_request, "rooms": rooms, "existing_bookings": existing_bookings}

# -------------------------
# OpenAI call
# -------------------------
def llm_plan(rag: Dict[str, Any]) -> Dict[str, Any]:
    if not openai_client:
        return {"status": "Failure", "message": "OpenAI not configured.", "steps": []}

    req = rag["new_request"]
    request_id = req.get("request_id") or req.get("id") or "REQ-UNKNOWN"

    seg = nights_segment(req)

    ci = parse_dt(req["requested_range"]["check_in"])
    co = parse_dt(req["requested_range"]["check_out"])
    ci, co = normalize_window(ci, co)

    # Reduce prompt size a bit (optional but helps stability/cost)
    bookings_relevant = filter_bookings_for_window(rag.get("existing_bookings", []), ci, co)

    user_payload = {
        "request_id": request_id,
        "segment": seg,
        "new_request": req,
        "rooms": rag.get("rooms", []),
        "existing_bookings": bookings_relevant,
        "instruction": "Return ONLY the plan JSON object. Enforce capacity strictly. Prefer STANDARD rooms when possible.",
    }

    prompt_json = json.dumps(user_payload, default=str)

    # Responses API (recommended). The model returns text; we parse JSON from it.
    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        input=[
            {"role": "developer", "content": PLANNER_SYSTEM},
            {"role": "user", "content": "INPUT:\n" + prompt_json},
        ],
    )

    raw = (resp.output_text or "").strip()
    plan = extract_json(raw)

    # minimal normalization (not validation)
    plan.setdefault("status", "Failure")
    plan.setdefault("steps", [])
    plan.setdefault("message", "Plan generated.")
    return plan

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running"})

# -------------------------------------------------------------------
# OPTIMIZE ENDPOINT (LLM or CP-SAT) + validate_and_score_plan() only
# -------------------------------------------------------------------
@app.route("/optimize/<request_id>", methods=["POST"])
def optimize(request_id: str):
    """
    One clean pipeline:

    1) Fetch RAG
    2) Produce plan (LLM or CP-SAT)
    3) Validate + Score + Confidence (validate_and_score_plan)
    4) Persist:
       - proposed_plan
       - plan_eval
       - status (proposed / failed / rejected / confirmed)
    5) If approved => apply plan to bookings + mark request confirmed
       Else => do NOT apply bookings
    """

    try:
        rag = fetch_rag(request_id)
        req_id = (rag.get("new_request") or {}).get("request_id") or request_id

        
        # -------------------------
        # Choose solver (toggle)
        # -------------------------
        solver = (os.environ.get("SOLVER", "llm") or "llm").lower()
        if solver not in ("cpsat", "llm"):
            solver = "llm"

        # -------------------------
        # 1) Produce plan
        # -------------------------
        if solver == "llm":
            plan = llm_plan(rag)
        else:
            plan = cpsat_plan(rag, max_moves=4, time_limit_s=3.0)

        # Safety normalize (avoid None surprises)
        if not isinstance(plan, dict):
            plan = {"status": "Failure", "message": "Planner returned non-dict plan.", "steps": []}
        plan.setdefault("status", "Failure")
        plan.setdefault("steps", [])
        plan.setdefault("message", "")

        # Helpful logging: what planner returned
        app.logger.info(
            "PLANNER solver=%s request_id=%s plan_status=%s steps=%d msg=%s",
            solver, req_id, plan.get("status"), len(plan.get("steps") or []), plan.get("message", "")
        )

        # Optional: log each step briefly (so you can see what it is doing)
        for idx, st in enumerate(plan.get("steps") or []):
            app.logger.info(
                "PLAN_STEP %s #%d action=%s booking_id=%s from=%s to=%s seg=%s dogs=%s",
                req_id,
                idx + 1,
                st.get("action"),
                st.get("booking_id"),
                st.get("from_room"),
                st.get("to_room"),
                st.get("segment"),
                st.get("dogs"),
            )

        # -------------------------
        # 2) Validate + Score
        # -------------------------
        eval_result = validate_and_score_plan(plan, rag)

        # Always log eval headline + first violations
        v_first = [v.get("message", "") for v in (eval_result.get("violations") or [])[:3]]
        app.logger.info(
            "EVAL solver=%s request_id=%s valid=%s approved=%s score=%.2f conf=%.3f moves=%s parts=%s v_first=%s",
            solver,
            req_id,
            bool(eval_result.get("valid")),
            bool(eval_result.get("approved")),
            float(eval_result.get("score", 0)),
            float(eval_result.get("confidence", 0)),
            (eval_result.get("metrics") or {}).get("moves"),
            (eval_result.get("metrics") or {}).get("parts"),
            v_first,
        )

        # -------------------------
        # 3) Persist plan + eval (ALWAYS)
        # -------------------------
        # Status semantics (very important):
        # - *_failed     => planner couldn't produce Success plan
        # - *_rejected   => planner Success but validator gate failed
        # - *_proposed   => planner Success AND gate approved (not yet confirmed)
        # - confirmed    => bookings actually written to Firestore
        plan_status = plan.get("status")
        approved = bool(eval_result.get("approved", False))

        if plan_status != "Success":
            request_status = f"{solver}_failed"
        elif not approved:
            request_status = f"{solver}_rejected"
        else:
            request_status = f"{solver}_proposed"

        db.collection("requests").document(request_id).set(
            {
                "solver": solver,
                "proposed_plan": plan,           # store plan JSON
                "plan_eval": eval_result,        # store validator+score+confidence
                "planned_at": dt.datetime.now(dt.timezone.utc),
                "status": request_status,

                # convenience (paper + sheet)
                "approved": approved,
                "valid": bool(eval_result.get("valid", False)),
                "score": float(eval_result.get("score", 0)),
                "confidence": float(eval_result.get("confidence", 0)),
                "violation_count": len(eval_result.get("violations") or []),
            },
            merge=True,
        )

        # Optional: notify your sheet for visibility (if you use it)
        # Keep consistent with your Apps Script status labels
        try:
            if plan_status != "Success":
                notify_sheet(request_id, rag, None, status="solver_failed", error_msg=plan.get("message", "Failure"))
            elif not approved:
                notify_sheet(request_id, rag, plan, status="solver_rejected", error_msg="; ".join(v_first))
            else:
                notify_sheet(request_id, rag, plan, status="solver_proposed", error_msg="")
        except Exception as e:
            app.logger.warning("notify_sheet failed: %s", e)

        # -------------------------
        # 4) Return early on planner failure
        # -------------------------
        if plan_status != "Success":
            return jsonify({"request_id": request_id, "plan": plan, "eval": eval_result}), 422

        # -------------------------
        # 5) Return early if rejected by validator gate
        # -------------------------
        if not approved:
            return jsonify({"request_id": request_id, "plan": plan, "eval": eval_result}), 409

        # -------------------------
        # 6) Apply plan to Firestore bookings ONLY if approved
        # -------------------------
        # IMPORTANT:
        # - This writes docs into bookings/<booking_id> for ConfirmNew
        # - This updates existing bookings for MoveExisting / MoveSegmentExisting if you support it
        apply_plan_to_firestore(
            db,
            request_id,
            rag,
            plan,
            mark_request_status="confirmed",
            booking_status="Confirmed",
            solver = solver,
        )

        # Mark request confirmed (separate from proposed)
        db.collection("requests").document(request_id).set(
            {"status": "confirmed", "confirmed_at": dt.datetime.now(dt.timezone.utc)},
            merge=True,
        )

        # Sheet final update
        try:
            notify_sheet(request_id, rag, plan, status="confirmed", error_msg="")
        except Exception as e:
            app.logger.warning("notify_sheet final failed: %s", e)

        return jsonify({"request_id": request_id, "plan": plan, "eval": eval_result}), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error("ERROR optimize %s: %s", request_id, e, exc_info=True)
        # Also store failure in request doc (helps debugging)
        try:
            db.collection("requests").document(request_id).set(
                {
                    "status": "optimizer_error",
                    "optimizer_error": str(e),
                    "optimizer_error_at": dt.datetime.now(dt.timezone.utc),
                },
                merge=True,
            )
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)




