"""
pbp_shares.py — Per-player usage/shot/point shares + lineup box.

Roster shares (two lenses):
  per_game (active): DNP-corrected shares vs team totals.
  on_court: stints-based exact possession denominator.

Lineup box:
  build_lineup_box(team, season) → per-lineup four factors (both sides),
  possession counts, stint count, W/L/T record.

Usage:
    from pbp_shares import build_roster_shares, build_lineup_box
    df = build_roster_shares("IND", season=2026)
    lb = build_lineup_box("IND", season=2026)
"""
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

from paths import RAPM_DIR, DATA, PLAYER_NAMES, stints

# ── Shared PBP constants (mirror build_stints.py) ────────────────────────────
_ACT_SHOT_MADE  = "made shot"
_ACT_SHOT_MISS  = "missed shot"
_ACT_FREETHROW  = "free throw"
_ACT_REBOUND    = "rebound"
_ACT_TURNOVER   = "turnover"
_ACT_PERIOD     = "period"
_FT_OF_RE       = re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE)

_PBP_DIR   = RAPM_DIR / "raw_pbp"


# ── Team ID map ───────────────────────────────────────────────────────────────

_TEAM_ID_TO_ABBR: dict[int, str] = {}

def _get_team_id_map() -> dict[int, str]:
    global _TEAM_ID_TO_ABBR
    if _TEAM_ID_TO_ABBR:
        return _TEAM_ID_TO_ABBR
    try:
        log = pd.read_csv(DATA / "game_log.csv")
        for _, r in log.iterrows():
            parts = str(r.get("matchup", "")).replace(" vs. ", " ").split()
            if len(parts) >= 2:
                for col, abbr in [(r.get("home_team_id"), parts[0]),
                                  (r.get("away_team_id"), parts[1])]:
                    if col and not pd.isna(col):
                        _TEAM_ID_TO_ABBR[int(col)] = abbr
    except Exception:
        pass
    return _TEAM_ID_TO_ABBR


def _abbr_to_team_ids(team_abbr: str) -> set[int]:
    t2a = _get_team_id_map()
    return {tid for tid, abbr in t2a.items() if abbr == team_abbr}


def _game_ids_for_team(team_abbr: str, season: int) -> list[str]:
    try:
        log = pd.read_csv(DATA / "game_log.csv")
        log = log[log["season"] == season]
        mask = log["matchup"].str.contains(team_abbr, na=False)
        return log[mask]["game_id"].astype(str).tolist()
    except Exception:
        return []


# ── Player names ──────────────────────────────────────────────────────────────

_PLAYER_NAMES_CACHE: dict[int, str] = {}

def _get_player_names() -> dict[int, str]:
    global _PLAYER_NAMES_CACHE
    if _PLAYER_NAMES_CACHE:
        return _PLAYER_NAMES_CACHE
    try:
        df = pd.read_csv(PLAYER_NAMES)
        _PLAYER_NAMES_CACHE = dict(zip(df["player_id"].astype(int), df["player_name"]))
    except Exception:
        pass
    return _PLAYER_NAMES_CACHE


# ── PBP box-score parser ──────────────────────────────────────────────────────

def _parse_box(game_id: str, team_ids: set[int]) -> pd.DataFrame:
    """
    Parse one PBP JSON → per-player box stats for team_ids.
    Returns DataFrame: pid, name, pts, fga, fta, tov, team_pts, team_fga, team_tov, team_fta
    """
    path = _PBP_DIR / f"{game_id}_pbp.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        pbp = json.load(open(path, encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    # Per-player stats
    player: dict[int, dict] = {}

    def _get(pid, tid, name):
        if pid not in player:
            player[pid] = dict(pid=pid, tid=tid, name=name,
                               pts=0, fga=0, fta=0, tov=0)
        return player[pid]

    for ev in pbp:
        pid  = ev.get("person_id", 0)
        tid  = ev.get("team_id",   0)
        if not pid or not tid:
            continue
        name = ev.get("playerNameI", ev.get("playerName", f"ID{pid}"))
        s    = _get(pid, tid, name)
        at   = ev.get("action_type", "")
        sv   = ev.get("shotValue", 0) or 0

        if at == "Made Shot":
            s["pts"] += sv if sv else 2
            s["fga"] += 1
        elif at == "Missed Shot":
            s["fga"] += 1
        elif at == "Free Throw":
            s["fta"] += 1
            if ev.get("shotResult") == "Made":
                s["pts"] += 1
        elif at == "Turnover":
            s["tov"] += 1

    if not player:
        return pd.DataFrame()

    df = pd.DataFrame(player.values())

    # Filter to team
    team_df = df[df["tid"].isin(team_ids)].copy()
    if team_df.empty:
        return pd.DataFrame()

    # Team totals this game
    t_pts = team_df["pts"].sum()
    t_fga = team_df["fga"].sum()
    t_fta = team_df["fta"].sum()
    t_tov = team_df["tov"].sum()

    team_df["t_pts"] = t_pts
    team_df["t_fga"] = t_fga
    team_df["t_fta"] = t_fta
    team_df["t_tov"] = t_tov
    team_df["t_poss"] = t_fga + 0.44 * t_fta + t_tov
    team_df["active"] = (team_df["pts"] + team_df["fga"] + team_df["fta"] + team_df["tov"]) > 0
    return team_df


# ── Stints on-court possession counts ────────────────────────────────────────

def _load_on_court_poss(team_ids: set[int], season: int) -> pd.DataFrame:
    """
    From stints CSV, return on-court offensive possession count per (game_id, pid).
    One row per player per game.
    """
    # Try stints (all games) first, fall back to stints_rich
    path = stints(season)                   # stints/stints_{season}_RS.csv
    if not path.exists():
        from paths import stints_rich
        path = stints_rich(season)
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "pid", "on_court_poss"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["game_id", "pid", "on_court_poss"])

    # Offensive possessions for this team
    off = df[df["off_team"].isin(team_ids)].copy()
    if off.empty:
        return pd.DataFrame(columns=["game_id", "pid", "on_court_poss"])

    off_cols = ["off_p1", "off_p2", "off_p3", "off_p4", "off_p5"]
    # Melt to one row per (game_id, possession, player)
    melted = off[["game_id"] + off_cols].melt(
        id_vars="game_id", value_vars=off_cols, value_name="pid"
    ).dropna(subset=["pid"])
    melted["pid"] = melted["pid"].astype(int)

    # Count possessions per player per game
    counts = (
        melted.groupby(["game_id", "pid"])
        .size()
        .reset_index(name="on_court_poss")
    )
    counts["game_id"] = counts["game_id"].astype(str)
    return counts


# ── Main builder ──────────────────────────────────────────────────────────────

def build_roster_shares(
    team_abbr: str,
    season: int = 2026,
    min_games: int = 1,
) -> pd.DataFrame:
    """
    Returns per-player share stats with two lenses:

      Per-game (active) columns  — DNP-corrected:
        pg_pts_share, pg_shot_share, pg_usage_share

      On-court columns  — stints-based exact possession denominator:
        oc_usage  (poss_used / on_court_off_poss × 100)

      Raw averages: avg_pts, avg_fga, avg_fta, avg_tov
    """
    team_ids = _abbr_to_team_ids(team_abbr)
    game_ids = _game_ids_for_team(team_abbr, season)
    names    = _get_player_names()

    if not game_ids:
        return pd.DataFrame()

    # ── 1. PBP box scores across all games ────────────────────────────────
    pbp_rows = []
    for gid in game_ids:
        box = _parse_box(gid, team_ids)
        if box.empty:
            continue
        box["game_id"] = gid
        pbp_rows.append(box)

    if not pbp_rows:
        return pd.DataFrame()

    pbp = pd.concat(pbp_rows, ignore_index=True)
    pbp["poss"] = pbp["fga"] + 0.44 * pbp["fta"] + pbp["tov"]

    # ── 2. On-court possession counts from stints ─────────────────────────
    oc = _load_on_court_poss(team_ids, season)

    # ── 3. Per-game active shares ─────────────────────────────────────────
    active_pbp = pbp[pbp["active"]].copy()

    pg = (
        active_pbp.groupby("pid")
        .agg(
            games      =("game_id", "nunique"),
            total_pts  =("pts",     "sum"),
            total_fga  =("fga",     "sum"),
            total_fta  =("fta",     "sum"),
            total_tov  =("tov",     "sum"),
            total_poss =("poss",    "sum"),
            sum_t_pts  =("t_pts",   "sum"),
            sum_t_fga  =("t_fga",   "sum"),
            sum_t_poss =("t_poss",  "sum"),
        )
        .reset_index()
    )
    pg = pg[pg["games"] >= min_games].copy()

    pg["avg_pts"]  = (pg["total_pts"] / pg["games"]).round(1)
    pg["avg_fga"]  = (pg["total_fga"] / pg["games"]).round(1)
    pg["avg_fta"]  = (pg["total_fta"] / pg["games"]).round(1)
    pg["avg_tov"]  = (pg["total_tov"] / pg["games"]).round(1)

    pg["pg_pts_share"]   = (pg["total_pts"]  / pg["sum_t_pts"]  * 100).round(1)
    pg["pg_shot_share"]  = (pg["total_fga"]  / pg["sum_t_fga"]  * 100).round(1)
    pg["pg_usage_share"] = (pg["total_poss"] / pg["sum_t_poss"] * 100).round(1)

    # ── 4. On-court usage from stints ─────────────────────────────────────
    if not oc.empty:
        # Total on-court poss per player across all games
        oc_total = oc.groupby("pid")["on_court_poss"].sum().reset_index()

        pg = pg.merge(oc_total, on="pid", how="left")
        pg["oc_usage"] = (
            pg["total_poss"] / pg["on_court_poss"] * 100
        ).round(1)
    else:
        pg["oc_usage"] = None

    # ── 5. Player names ───────────────────────────────────────────────────
    # Prefer player_names CSV (full name), fall back to PBP playerNameI
    pbp_names = (
        pbp[["pid", "name"]].drop_duplicates("pid").set_index("pid")["name"]
    )
    pg["player_name"] = pg["pid"].apply(
        lambda p: names.get(p) or pbp_names.get(p, f"ID{p}")
    )

    return (
        pg[[
            "player_name", "games",
            "avg_pts", "avg_fga", "avg_fta", "avg_tov",
            "pg_pts_share", "pg_shot_share", "pg_usage_share",
            "oc_usage",
        ]]
        .sort_values("oc_usage", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ── Lineup box ───────────────────────────────────────────────────────────────

def _game_poss_stats(game_id: str) -> dict[int, dict]:
    """
    Walk one game's PBP with the same possession-boundary triggers as
    build_stints.py, accumulating four-factor raw counts per poss_id.
    poss_id is incremented at identical events so the join with the stints
    CSV is exact.

    Returns {poss_id: {off_team, fga, fgm, tpa, tpm, fta, oreb, tov}}
    """
    path = _PBP_DIR / f"{game_id}_pbp.json"
    if not path.exists():
        return {}
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

    # Discover the two teams in this game
    teams: set[int] = set()
    for ev in rows:
        t = int(ev.get("team_id", 0) or 0)
        if t:
            teams.add(t)
    if len(teams) < 2:
        return {}

    def _other(t: int) -> int | None:
        others = teams - {t}
        return next(iter(others), None)

    result: dict[int, dict] = {}
    poss_id    = 0
    off_team   = None
    last_shot_tid = None

    def _empty() -> dict:
        return dict(fga=0, fgm=0, tpa=0, tpm=0, fta=0, oreb=0, tov=0, pts=0)

    cur = _empty()

    def _end_poss(next_off: int | None) -> None:
        nonlocal poss_id, off_team, last_shot_tid, cur
        result[poss_id] = dict(off_team=off_team, **cur)
        poss_id   += 1
        off_team   = next_off
        last_shot_tid = None
        cur = _empty()

    for ev in rows:
        act = str(ev.get("action_type", "") or "").lower().strip()
        sub = str(ev.get("sub_type",    "") or "").lower().strip()
        tid = int(ev.get("team_id",     0)  or 0)
        is_fg  = bool(ev.get("isFieldGoal", 0))
        res    = str(ev.get("shotResult",   "") or "")
        sv     = int(ev.get("shotValue",    0)  or 0)
        desc   = str(ev.get("description", "") or "")

        # ── Period boundary ────────────────────────────────────────────────
        if act == _ACT_PERIOD:
            if "end" in sub:
                _end_poss(None)
                last_shot_tid = None
            continue

        if act in ("substitution", "timeout", "instant replay", "violation"):
            continue

        # ── Jump ball — mirrors build_stints.ACT_JUMPBALL handler ─────────
        if act == "jump ball":
            if off_team is None and tid and tid in teams:
                off_team = tid
            continue

        if not tid or tid not in teams:
            continue

        # ── Field goals ────────────────────────────────────────────────────
        if is_fg:
            if off_team is None:
                off_team = tid
            last_shot_tid = tid
            if tid == off_team:
                cur["fga"] += 1
                if sv == 3:
                    cur["tpa"] += 1
                if res == "Made":
                    cur["fgm"] += 1
                    if sv == 3:
                        cur["tpm"] += 1
                    cur["pts"] += sv
            if res == "Made":
                _end_poss(_other(tid))
            continue

        # ── Rebounds ───────────────────────────────────────────────────────
        if act == _ACT_REBOUND:
            # Match build_stints: offensive if sub_type=="offensive" OR same team as last shot
            is_oreb = sub == "offensive" or \
                      (last_shot_tid is not None and tid == last_shot_tid)
            if is_oreb:
                if off_team is None:
                    off_team = tid
                cur["oreb"] += 1
            else:
                _end_poss(tid)
            continue

        # ── Free throws ────────────────────────────────────────────────────
        if act == _ACT_FREETHROW:
            # Mirror build_stints._is_technical: skip technical AND flagrant FTs.
            # Flagrant FTs don't end possession — the fouled team retains ball.
            if "technical" in sub or "flagrant" in sub:
                continue
            if off_team is None:
                off_team = tid
            if tid == off_team:
                cur["fta"] += 1
            m_seq = _FT_OF_RE.search(sub)
            is_last = (m_seq and int(m_seq.group(1)) == int(m_seq.group(2))) \
                      or "1 of 1" in sub
            is_miss = "miss" in desc.lower()
            if not is_miss and tid == off_team:
                cur["pts"] += 1          # count every made FT
            if is_last:
                last_shot_tid = tid
                if not is_miss:
                    _end_poss(_other(tid))
            continue

        # ── Turnovers ──────────────────────────────────────────────────────
        # End possession for ANY team's turnover — matches build_stints.py exactly.
        # Defensive turnovers (e.g. kicked-ball violations) increment poss_id
        # just as build_stints does, keeping the two poss_id counters in sync.
        if act == _ACT_TURNOVER:
            if off_team is None:
                off_team = tid
            if tid == off_team:
                cur["tov"] += 1
            _end_poss(_other(tid))   # always — not just for offensive team
            continue

    return result


def _game_player_stats(
    game_id: str,
    target_poss_ids: set[int],
    off_team_id: int,
) -> dict[int, dict]:
    """
    Walk one game's PBP with the same possession-boundary logic as
    _game_poss_stats, collecting per-player offensive stats ONLY for
    possessions in target_poss_ids where off_team == off_team_id.

    Returns {person_id: {pts, fga, fgm, tpa, tpm, fta, tov}}.
    """
    path = _PBP_DIR / f"{game_id}_pbp.json"
    if not path.exists():
        return {}
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

    teams: set[int] = set()
    for ev in rows:
        t = int(ev.get("team_id", 0) or 0)
        if t:
            teams.add(t)
    if len(teams) < 2:
        return {}

    def _other(t: int) -> int | None:
        others = teams - {t}
        return next(iter(others), None)

    player_stats: dict[int, dict] = {}

    def _ps(pid: int) -> dict:
        if pid not in player_stats:
            player_stats[pid] = dict(pts=0, fga=0, fgm=0, tpa=0, tpm=0, fta=0, tov=0)
        return player_stats[pid]

    poss_id      = 0
    off_team     = None
    last_shot_tid = None

    def _collecting() -> bool:
        return poss_id in target_poss_ids and off_team == off_team_id

    def _end_poss(next_off: int | None) -> None:
        nonlocal poss_id, off_team, last_shot_tid
        poss_id      += 1
        off_team      = next_off
        last_shot_tid = None

    for ev in rows:
        act  = str(ev.get("action_type", "") or "").lower().strip()
        sub  = str(ev.get("sub_type",    "") or "").lower().strip()
        tid  = int(ev.get("team_id",   0)   or 0)
        pid  = int(ev.get("person_id", 0)   or 0)
        is_fg = bool(ev.get("isFieldGoal", 0))
        res  = str(ev.get("shotResult",  "") or "")
        sv   = int(ev.get("shotValue",   0)  or 0)
        desc = str(ev.get("description", "") or "")

        if act == _ACT_PERIOD:
            if "end" in sub:
                _end_poss(None)
                last_shot_tid = None
            continue

        if act in ("substitution", "timeout", "instant replay", "violation"):
            continue

        # ── Jump ball — mirrors build_stints.ACT_JUMPBALL handler ─────
        if act == "jump ball":
            if off_team is None and tid and tid in teams:
                off_team = tid
            continue

        if not tid or tid not in teams:
            continue

        # ── Field goals ────────────────────────────────────────────────
        if is_fg:
            if off_team is None:
                off_team = tid
            last_shot_tid = tid
            if _collecting() and tid == off_team and pid:
                _ps(pid)["fga"] += 1
                if sv == 3:
                    _ps(pid)["tpa"] += 1
                if res == "Made":
                    _ps(pid)["fgm"] += 1
                    if sv == 3:
                        _ps(pid)["tpm"] += 1
                    _ps(pid)["pts"] += sv
            if res == "Made":
                _end_poss(_other(tid))
            continue

        # ── Rebounds ───────────────────────────────────────────────────
        if act == _ACT_REBOUND:
            is_oreb = sub == "offensive" or \
                      (last_shot_tid is not None and tid == last_shot_tid)
            if is_oreb:
                if off_team is None:
                    off_team = tid
            else:
                _end_poss(tid)
            continue

        # ── Free throws ────────────────────────────────────────────────
        if act == _ACT_FREETHROW:
            # Mirror build_stints._is_technical: skip technical AND flagrant FTs.
            if "technical" in sub or "flagrant" in sub:
                continue
            if off_team is None:
                off_team = tid
            m_seq  = _FT_OF_RE.search(sub)
            is_last = (m_seq and int(m_seq.group(1)) == int(m_seq.group(2))) \
                      or "1 of 1" in sub
            is_miss = "miss" in desc.lower()
            if _collecting() and tid == off_team and pid:
                _ps(pid)["fta"] += 1
                if not is_miss:
                    _ps(pid)["pts"] += 1
            if is_last:
                last_shot_tid = tid
                if not is_miss:
                    _end_poss(_other(tid))
            continue

        # ── Turnovers ──────────────────────────────────────────────────
        # End possession for ANY team's turnover — matches build_stints.py.
        # Defensive turnovers (kicked ball) must also increment poss_id.
        if act == _ACT_TURNOVER:
            if off_team is None:
                off_team = tid
            if _collecting() and tid == off_team and pid:
                _ps(pid)["tov"] += 1
            _end_poss(_other(tid))   # always — not just for offensive team
            continue

    return player_stats


def player_stats_for_lineup(
    team_abbr: str,
    season: int,
    lu_str: str,
    game_ids_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Per-player offensive box stats for all possessions where `team_abbr`
    ran the lineup identified by `lu_str` (pipe-delimited sorted player IDs).

    Returns one row per player with columns:
      player, poss, pts, fga, fgm, fg_pct, tpa, tpm, tp_pct,
      fta, tov, usage_pct, pts_share
    """
    from paths import stints as stints_path

    # ── Resolve team IDs ─────────────────────────────────────────────
    team_ids: set[int] = _abbr_to_team_ids(team_abbr)
    if not team_ids:
        return pd.DataFrame()

    # ── Load stints and filter to game scope ─────────────────────────
    try:
        all_stints = pd.read_csv(stints_path(season))
    except Exception:
        return pd.DataFrame()
    all_stints["game_id"] = all_stints["game_id"].astype(str)

    game_ids = sorted(
        all_stints[
            all_stints["off_team"].isin(team_ids) | all_stints["def_team"].isin(team_ids)
        ]["game_id"].unique()
    )
    if game_ids_filter is not None:
        filt = set(str(g) for g in game_ids_filter)
        game_ids = [g for g in game_ids if g in filt]
    if not game_ids:
        return pd.DataFrame()

    off_cols = ["off_p1", "off_p2", "off_p3", "off_p4", "off_p5"]

    # Parse target player IDs from lu_str
    target_pids: frozenset[int] = frozenset(int(x) for x in lu_str.split("|") if x)

    def _lu_str_from_row(row) -> str:
        ids = frozenset(int(row[c]) for c in off_cols if pd.notna(row[c]) and row[c])
        return "|".join(str(p) for p in sorted(ids))

    # ── For each game collect (game_id, poss_id, off_team) matching lineup
    game_stints = all_stints[all_stints["game_id"].isin(game_ids)]
    off_stints  = game_stints[game_stints["off_team"].isin(team_ids)].copy()
    off_stints["lu_str"] = off_stints.apply(_lu_str_from_row, axis=1)
    matched = off_stints[off_stints["lu_str"] == lu_str]

    if matched.empty:
        return pd.DataFrame()

    # ── Walk PBP per game ─────────────────────────────────────────────
    agg: dict[int, dict] = {}
    total_poss = 0

    for gid, g_rows in matched.groupby("game_id"):
        target_poss_ids = set(g_rows["poss_id"].astype(int))
        total_poss     += len(target_poss_ids)

        # Find the offensive team_id for this game
        off_tid = int(g_rows["off_team"].iloc[0])

        per_player = _game_player_stats(str(gid), target_poss_ids, off_tid)
        for pid, stats in per_player.items():
            if pid not in agg:
                agg[pid] = dict(pts=0, fga=0, fgm=0, tpa=0, tpm=0, fta=0, tov=0)
            for k, v in stats.items():
                agg[pid][k] += v

    if not agg:
        return pd.DataFrame()

    # ── Build names lookup (full names preferred, then PBP playerNameI) ──
    names: dict[int, str] = _get_player_names()   # pid → full name
    # Supplement missing IDs from PBP playerNameI ("A. Morrow" format)
    for gid in list(matched["game_id"].unique())[:3]:
        pbp_path = _PBP_DIR / f"{gid}_pbp.json"
        if pbp_path.exists():
            try:
                pbp_rows = json.load(open(pbp_path, encoding="utf-8"))
                for ev in pbp_rows:
                    pid   = int(ev.get("person_id", 0) or 0)
                    nameI = str(ev.get("playerNameI", "") or "")
                    if pid and nameI and pid not in names:
                        names[pid] = nameI
            except Exception:
                pass

    # ── Safety filter: restrict to the 5 players in lu_str ───────────
    # Even if poss_id drift causes a few wrong-lineup possessions to sneak in,
    # we never want to show a player outside the target lineup.
    agg = {pid: s for pid, s in agg.items() if pid in target_pids}

    if not agg:
        return pd.DataFrame()

    # ── Compute totals for usage / share denominators ─────────────────
    team_fga = sum(v["fga"] for v in agg.values())
    team_fta = sum(v["fta"] for v in agg.values())
    team_tov = sum(v["tov"] for v in agg.values())
    team_pts = sum(v["pts"] for v in agg.values())
    denom_usage = (team_fga + 0.44 * team_fta + team_tov) or 1

    rows_out = []
    for pid, s in agg.items():
        raw_name = names.get(pid, f"ID{pid}")
        p_denom  = (s["fga"] + 0.44 * s["fta"] + s["tov"]) or 0
        rows_out.append({
            "player":    _player_short(raw_name) if " " not in raw_name else _player_short(raw_name),
            "poss":      total_poss,
            "pts":       s["pts"],
            "fga":       s["fga"],
            "fgm":       s["fgm"],
            "fg_pct":    round(s["fgm"] / s["fga"], 3) if s["fga"] else 0.0,
            "tpa":       s["tpa"],
            "tpm":       s["tpm"],
            "tp_pct":    round(s["tpm"] / s["tpa"], 3) if s["tpa"] else 0.0,
            "fta":       s["fta"],
            "tov":       s["tov"],
            "usage_pct": round(p_denom / denom_usage * 100, 1) if denom_usage else 0.0,
            "pts_share": round(s["pts"] / team_pts * 100, 1) if team_pts else 0.0,
        })

    out = pd.DataFrame(rows_out).sort_values("usage_pct", ascending=False).reset_index(drop=True)
    return out


def _player_short(name: str) -> str:
    """'Breanna Stewart' → 'B. Stewart'"""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name


def build_lineup_box(
    team_abbr: str,
    season: int = 2026,
    game_ids_filter: list[str] | None = None,
    min_poss: int = 10,
) -> pd.DataFrame:
    """
    Build a lineup-level box score for team_abbr.

    Columns per lineup:
      lineup          – '5 player names' (short form, sorted)
      poss            – offensive possessions
      pts_for         – total points scored
      oeff            – pts_for / poss * 100
      o_efg           – (FGM + 0.5*3PM) / FGA
      o_tov           – TOV / (FGA + 0.44*FTA + TOV)
      o_oreb          – OREB / (OREB + opp_dreb)  ← approximated from off OREB
      o_ftr           – FTA / FGA
      def_poss        – defensive possessions
      pts_against     – total points allowed
      deff            – pts_against / def_poss * 100
      d_efg, d_tov, d_ftr – opponent four factors while this lineup is on D
      net             – oeff - deff
      stints          – number of contiguous runs this lineup appeared
      W / L / T       – stint record (net > 0 / < 0 / == 0 per run)
    """
    team_ids  = _abbr_to_team_ids(team_abbr)
    game_ids  = _game_ids_for_team(team_abbr, season)
    names     = _get_player_names()

    if game_ids_filter:
        game_ids = [g for g in game_ids if g in game_ids_filter]
    if not game_ids:
        return pd.DataFrame()

    # ── Load stints for the season ────────────────────────────────────────
    stint_path = stints(season)
    if not stint_path.exists():
        return pd.DataFrame()
    try:
        all_stints = pd.read_csv(stint_path)
    except Exception:
        return pd.DataFrame()

    all_stints["game_id"] = all_stints["game_id"].astype(str)
    game_stints = all_stints[all_stints["game_id"].isin(game_ids)].copy()
    if game_stints.empty:
        return pd.DataFrame()

    off_cols = ["off_p1", "off_p2", "off_p3", "off_p4", "off_p5"]
    def_cols = ["def_p1", "def_p2", "def_p3", "def_p4", "def_p5"]

    # ── Join four-factor stats from PBP ──────────────────────────────────
    poss_records: list[dict] = []
    for gid in game_ids:
        g_stints = game_stints[game_stints["game_id"] == gid]
        if g_stints.empty:
            continue
        poss_stats = _game_poss_stats(gid)
        if not poss_stats:
            # Fall back: use points from stints (may be inaccurate), zero four factors
            for _, r in g_stints.iterrows():
                pts = int(r["points"]) if int(r["points"]) <= 4 else 0
                poss_records.append({
                    "game_id": gid, "poss_id": r["poss_id"],
                    "off_team": r["off_team"], "def_team": r["def_team"],
                    "s_points": pts,
                    "fga": 0, "fgm": 0, "tpa": 0, "tpm": 0,
                    "fta": 0, "oreb": 0, "tov": 0,
                    **{c: r[c] for c in off_cols + def_cols},
                })
            continue

        for _, r in g_stints.iterrows():
            pid_key = int(r["poss_id"])
            ps = poss_stats.get(pid_key, {})
            poss_records.append({
                "game_id": gid, "poss_id": pid_key,
                "off_team": r["off_team"], "def_team": r["def_team"],
                # Use PBP-derived pts (accurate); stints "points" col has data bugs
                "s_points": ps.get("pts", 0),
                "fga": ps.get("fga", 0), "fgm": ps.get("fgm", 0),
                "tpa": ps.get("tpa", 0), "tpm": ps.get("tpm", 0),
                "fta": ps.get("fta", 0), "oreb": ps.get("oreb", 0),
                "tov": ps.get("tov", 0),
                **{c: r[c] for c in off_cols + def_cols},
            })

    if not poss_records:
        return pd.DataFrame()

    poss_df = pd.DataFrame(poss_records)

    # ── Split into offensive and defensive rows for our team ──────────────
    is_our_off = poss_df["off_team"].isin(team_ids)
    off_df = poss_df[is_our_off].copy()
    def_df = poss_df[~is_our_off].copy()   # we are on defense

    def _lineup_key(row, cols) -> frozenset:
        return frozenset(int(row[c]) for c in cols if pd.notna(row[c]) and row[c])

    # Convert frozenset → stable string key for groupby (pandas 2.x safe)
    off_df["lu_key"] = off_df.apply(_lineup_key, axis=1, args=(off_cols,))
    def_df["lu_key"] = def_df.apply(_lineup_key, axis=1, args=(def_cols,))
    off_df["lu_str"] = off_df["lu_key"].apply(lambda k: "|".join(str(p) for p in sorted(k)))
    def_df["lu_str"] = def_df["lu_key"].apply(lambda k: "|".join(str(p) for p in sorted(k)))

    # ── Aggregate offensive stats per lineup ──────────────────────────────
    off_agg = (
        off_df.groupby("lu_str")
        .agg(
            poss    =("s_points", "count"),
            pts_for =("s_points", "sum"),
            fga_sum =("fga",      "sum"),
            fgm_sum =("fgm",      "sum"),
            tpa_sum =("tpa",      "sum"),
            tpm_sum =("tpm",      "sum"),
            fta_sum =("fta",      "sum"),
            oreb_sum=("oreb",     "sum"),
            tov_sum =("tov",      "sum"),
        )
        .reset_index()
    )
    _f = lambda col: off_agg[col]
    fga1 = off_agg["fga_sum"].clip(lower=1)
    denom_tov = off_agg["fga_sum"] + 0.44 * off_agg["fta_sum"] + off_agg["tov_sum"]
    off_agg["oeff"]   = (off_agg["pts_for"] / off_agg["poss"] * 100).round(1)
    off_agg["o_efg"]  = ((off_agg["fgm_sum"] + 0.5 * off_agg["tpm_sum"]) / fga1).round(3)
    off_agg["o_tov"]  = (off_agg["tov_sum"] / denom_tov.clip(lower=1e-9)).round(3)
    off_agg["o_oreb"] = (off_agg["oreb_sum"] / (off_agg["fga_sum"] - off_agg["fgm_sum"] + off_agg["oreb_sum"]).clip(lower=1)).round(3)
    off_agg["o_ftr"]  = (off_agg["fta_sum"] / fga1).round(3)
    off_agg = off_agg.drop(columns=["fga_sum","fgm_sum","tpa_sum","tpm_sum","fta_sum","oreb_sum","tov_sum"])

    # ── Aggregate defensive stats per lineup ──────────────────────────────
    def_agg = (
        def_df.groupby("lu_str")
        .agg(
            def_poss    =("s_points", "count"),
            pts_against =("s_points", "sum"),
            fga_sum =("fga",  "sum"),
            fgm_sum =("fgm",  "sum"),
            tpa_sum =("tpa",  "sum"),
            tpm_sum =("tpm",  "sum"),
            fta_sum =("fta",  "sum"),
            tov_sum =("tov",  "sum"),
        )
        .reset_index()
    )
    fga1d = def_agg["fga_sum"].clip(lower=1)
    denom_tov_d = def_agg["fga_sum"] + 0.44 * def_agg["fta_sum"] + def_agg["tov_sum"]
    def_agg["deff"]  = (def_agg["pts_against"] / def_agg["def_poss"] * 100).round(1)
    def_agg["d_efg"] = ((def_agg["fgm_sum"] + 0.5 * def_agg["tpm_sum"]) / fga1d).round(3)
    def_agg["d_tov"] = (def_agg["tov_sum"] / denom_tov_d.clip(lower=1e-9)).round(3)
    def_agg["d_ftr"] = (def_agg["fta_sum"] / fga1d).round(3)
    def_agg = def_agg.drop(columns=["fga_sum","fgm_sum","tpa_sum","tpm_sum","fta_sum","tov_sum"])

    # ── Stint count and W/L/T ─────────────────────────────────────────────
    # For each game, sort by poss_id and identify contiguous runs for each
    # team lineup (on either side).  Net = pts scored - pts allowed per run.
    # Uses lu_str (pipe-delimited sorted IDs) as the stable dict key.
    stint_stats: dict[str, dict] = {}

    for gid in game_ids:
        g_off = off_df[off_df["game_id"] == gid].sort_values("poss_id")
        g_def = def_df[def_df["game_id"] == gid].sort_values("poss_id")

        # Merge to one row per possession with our lineup string and net pts
        g_off_mini = g_off[["poss_id", "lu_str", "s_points"]].rename(
            columns={"s_points": "pts_scored", "lu_str": "lu_off"})
        g_def_mini = g_def[["poss_id", "lu_str", "s_points"]].rename(
            columns={"s_points": "pts_allowed", "lu_str": "lu_def"})

        combined = pd.merge(g_off_mini, g_def_mini, on="poss_id", how="outer")
        # Use whichever side has our team's lineup
        combined["lu_str"] = combined["lu_off"].where(
            combined["lu_off"].notna(), combined["lu_def"]
        )
        combined = combined.sort_values("poss_id").dropna(subset=["lu_str"])
        if combined.empty:
            continue

        # Identify contiguous runs (stint = consecutive same lu_str)
        combined["run_id"] = (
            combined["lu_str"] != combined["lu_str"].shift()
        ).cumsum()

        for (lu, run), grp in combined.groupby(["lu_str", "run_id"], sort=False):
            net = grp["pts_scored"].fillna(0).sum() - grp["pts_allowed"].fillna(0).sum()
            if lu not in stint_stats:
                stint_stats[lu] = dict(stints=0, W=0, L=0, T=0)
            ss = stint_stats[lu]
            ss["stints"] += 1
            if net > 0:
                ss["W"] += 1
            elif net < 0:
                ss["L"] += 1
            else:
                ss["T"] += 1

    stint_df = pd.DataFrame([
        dict(lu_str=k, **v) for k, v in stint_stats.items()
    ]) if stint_stats else pd.DataFrame(columns=["lu_str", "stints", "W", "L", "T"])

    # ── Merge everything on lu_str ────────────────────────────────────────
    box = off_agg.merge(def_agg,  on="lu_str", how="outer") \
                 .merge(stint_df, on="lu_str", how="left")

    box = box[box["poss"].fillna(0) >= min_poss].copy()
    if box.empty:
        return pd.DataFrame()

    box["net"] = (box["oeff"] - box["deff"]).round(1)

    # ── Resolve player IDs → display names ───────────────────────────────
    # Build id→short_name from names CSV + PBP fallback
    pid_to_short: dict[int, str] = {
        pid: _player_short(n) for pid, n in names.items()
    }
    for col in off_cols + def_cols:
        for pid in poss_df[col].dropna().astype(int).unique():
            if pid not in pid_to_short:
                pid_to_short[pid] = f"ID{pid}"

    def _lu_display(lu_str_val: str) -> str:
        """Convert '123|456|789' → 'B. Stewart / S. Ionescu / ...' """
        pids = [int(x) for x in lu_str_val.split("|") if x]
        return " / ".join(sorted(pid_to_short.get(p, f"ID{p}") for p in pids))

    box["lineup"] = box["lu_str"].apply(_lu_display)
    box["stints"] = box["stints"].fillna(0).astype(int)
    box["W"]  = box["W"].fillna(0).astype(int)
    box["L"]  = box["L"].fillna(0).astype(int)
    box["T"]  = box["T"].fillna(0).astype(int)

    return box[[
        "lineup", "lu_str", "poss", "pts_for", "oeff",
        "o_efg", "o_tov", "o_oreb", "o_ftr",
        "def_poss", "pts_against", "deff",
        "d_efg", "d_tov", "d_ftr",
        "net", "stints", "W", "L", "T",
    ]].sort_values("poss", ascending=False).reset_index(drop=True)


def available_seasons() -> list[int]:
    try:
        log = pd.read_csv(DATA / "game_log.csv")
        return sorted(log["season"].dropna().astype(int).unique().tolist(), reverse=True)
    except Exception:
        return [2026]
