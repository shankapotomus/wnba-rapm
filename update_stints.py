"""
update_stints.py — fetch new WNBA games and append to stints CSVs.

Reads existing stints to find which games are already processed, fetches
any new games for the season via nba_api (free, no key needed), parses
possessions, and appends to:
  wnba_data/stints/stints_{season}_RS.csv
  wnba_data/stints_rich/stints_rich_{season}_RS.csv

Usage:
    python update_stints.py                   # update current season (auto-detect)
    python update_stints.py --season 2025     # explicit season
    python update_stints.py --dry-run         # show new games, don't write

Requirements:
    pip install nba_api pandas numpy
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("wnba_data")
WNBA_ID    = "10"
SLEEP_SEC  = 0.6   # be polite to stats.nba.com between requests

# ── Clock helpers ─────────────────────────────────────────────────────────────

def _clock_to_secs(clock: str) -> float:
    """'PT05M30.00S' → 330.0 seconds remaining in period."""
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return 0.0


# ── nba_api fetchers ──────────────────────────────────────────────────────────

def fetch_game_ids(season: int) -> list[str]:
    from nba_api.stats.endpoints import leaguegamelog
    gl = leaguegamelog.LeagueGameLog(
        league_id=WNBA_ID,
        season=str(season),
        season_type_all_star="Regular Season",
    )
    df = gl.get_data_frames()[0]
    return sorted(df["GAME_ID"].astype(str).unique().tolist())


def fetch_pbp(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv3
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    return pbp.get_data_frames()[0]


def fetch_starters(game_id: str) -> dict[int, list[int]]:
    """Return {team_id: [starter_player_id, ...]} for both teams."""
    from nba_api.stats.endpoints import boxscoretraditionalv3
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    players = box.get_data_frames()[0]

    # Normalise column names (v3 uses camelCase)
    col_map = {c: c.lower() for c in players.columns}
    players = players.rename(columns=col_map)

    start_col = next((c for c in players.columns if "startposition" in c or "start_position" in c), None)
    team_col  = next((c for c in players.columns if c in ("teamid", "team_id")), None)
    pid_col   = next((c for c in players.columns if c in ("playerid", "person_id", "personid")), None)

    if not all([start_col, team_col, pid_col]):
        return {}

    starters = players[players[start_col].fillna("").str.strip() != ""]
    result: dict[int, list[int]] = {}
    for _, row in starters.iterrows():
        tid = int(row[team_col])
        pid = int(row[pid_col])
        result.setdefault(tid, []).append(pid)
    return result


def fetch_game_date(game_id: str, pbp: pd.DataFrame) -> str:
    """Extract game date (YYYY-MM-DD) from PBP if available, else empty string."""
    for col in pbp.columns:
        if "date" in col.lower() or "gametime" in col.lower():
            val = pbp[col].dropna().iloc[0] if not pbp[col].dropna().empty else ""
            if val:
                return str(val)[:10]
    return ""


# ── Possession parser ─────────────────────────────────────────────────────────

def _norm(pbp: pd.DataFrame) -> pd.DataFrame:
    """Normalise PlayByPlayV3 column names to a stable lowercase schema."""
    rename = {}
    for c in pbp.columns:
        cl = c.lower()
        if cl in ("actiontype", "action_type"):
            rename[c] = "action_type"
        elif cl in ("subtype", "sub_type"):
            rename[c] = "sub_type"
        elif cl in ("teamid", "team_id"):
            rename[c] = "team_id"
        elif cl in ("personid", "person_id"):
            rename[c] = "person_id"
        elif cl in ("isfieldgoal", "is_field_goal"):
            rename[c] = "is_fg"
        elif cl in ("shotresult", "shot_result"):
            rename[c] = "shot_result"
        elif cl in ("pointstotal", "points_total"):
            rename[c] = "points_total"
        elif cl == "clock":
            rename[c] = "clock"
        elif cl == "period":
            rename[c] = "period"
        elif cl in ("scorehome", "score_home"):
            rename[c] = "score_home"
        elif cl in ("scoreaway", "score_away"):
            rename[c] = "score_away"
        elif cl in ("description",):
            rename[c] = "description"
    return pbp.rename(columns=rename)


def parse_game(
    game_id: str,
    pbp_raw: pd.DataFrame,
    starters: dict[int, list[int]],
    game_date: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one game's PBP into possession-level stints + stints_rich rows.

    Returns (stints_df, stints_rich_df) — empty DataFrames on failure.
    """
    pbp = _norm(pbp_raw).copy()

    # Discover the two team IDs
    teams: list[int] = [
        int(t) for t in pbp["team_id"].dropna().unique()
        if int(t) != 0
    ]
    if len(teams) < 2:
        return pd.DataFrame(), pd.DataFrame()

    def _other(tid: int) -> int:
        return next(t for t in teams if t != tid)

    # ── Lineup state ─────────────────────────────────────────────────────────
    # Start with starters; update on substitutions
    lineups: dict[int, set[int]] = {t: set(starters.get(t, [])) for t in teams}

    # ── Per-possession accumulators ───────────────────────────────────────────
    poss_records: list[dict] = []

    poss_id    = 0
    off_team   = None      # which team currently has the ball
    last_secs: float | None = None   # clock at end of last possession (for trans)

    # Current possession stats
    def _empty_poss() -> dict:
        return dict(fga=0, fgm=0, fg3a=0, fg3m=0, fta=0, ftm=0,
                    tov_flag=0, oreb=0, oreb_chance=0, pts=0)

    cur = _empty_poss()
    last_shot_tid: int | None = None
    ft_seq_n: int | None = None   # total FTs in current sequence (from 'M of N')
    ft_seq_i: int = 0             # current FT index

    def _end_poss(next_off: int | None, clock_secs: float) -> None:
        nonlocal poss_id, off_team, last_shot_tid, cur, ft_seq_n, ft_seq_i, last_secs

        if off_team is None:
            poss_id += 1
            off_team = next_off
            last_shot_tid = None
            cur = _empty_poss()
            ft_seq_n = ft_seq_i = 0
            return

        def_team = _other(off_team)
        off_lu = sorted(lineups[off_team])
        def_lu = sorted(lineups[def_team])

        # Pad lineups to exactly 5 (use 0 for missing)
        off_lu = (off_lu + [0, 0, 0, 0, 0])[:5]
        def_lu = (def_lu + [0, 0, 0, 0, 0])[:5]

        # Transition: previous poss ended < 6 seconds ago
        poss_secs = last_secs if last_secs is not None else clock_secs
        elapsed = poss_secs - clock_secs
        trans_flag = 1 if (last_secs is not None and elapsed < 6) else 0

        poss_records.append({
            "game_id":   game_id,
            "poss_id":   poss_id,
            "off_team":  off_team,
            "def_team":  def_team,
            "points":    cur["pts"],
            "off_p1": off_lu[0], "off_p2": off_lu[1], "off_p3": off_lu[2],
            "off_p4": off_lu[3], "off_p5": off_lu[4],
            "def_p1": def_lu[0], "def_p2": def_lu[1], "def_p3": def_lu[2],
            "def_p4": def_lu[3], "def_p5": def_lu[4],
            # rich fields
            "fga": cur["fga"], "fgm": cur["fgm"],
            "fg3a": cur["fg3a"], "fg3m": cur["fg3m"],
            "fta": cur["fta"], "ftm": cur["ftm"],
            "tov_flag": cur["tov_flag"],
            "oreb": cur["oreb"], "oreb_chance": cur["oreb_chance"],
            "trans_flag": trans_flag,
            "game_date": game_date,
        })

        last_secs = clock_secs
        poss_id  += 1
        off_team  = next_off
        last_shot_tid = None
        cur = _empty_poss()
        ft_seq_n = ft_seq_i = 0

    # ── Walk PBP ─────────────────────────────────────────────────────────────
    for _, ev in pbp.iterrows():
        act  = str(ev.get("action_type", "") or "").lower().strip()
        sub  = str(ev.get("sub_type",    "") or "").lower().strip()
        tid  = int(ev.get("team_id",  0) or 0)
        pid  = int(ev.get("person_id",0) or 0)
        is_fg  = bool(int(ev.get("is_fg",         0) or 0))
        res    = str(ev.get("shot_result",         "") or "").lower()
        desc   = str(ev.get("description",         "") or "").lower()
        clock  = _clock_to_secs(ev.get("clock", "PT00M00.00S"))

        # ── Period boundary ───────────────────────────────────────────────
        if act == "period":
            if sub in ("end", "endperiod"):
                _end_poss(None, clock)
            elif sub in ("start", "startperiod"):
                pass  # lineup state already correct from prior period
            continue

        # ── Skip non-play events ─────────────────────────────────────────
        if act in ("substitution",):
            if tid and pid:
                # nba_api v3: separate rows for player OUT (sub_type='out')
                # and player IN (sub_type='in')
                if "out" in sub and pid in lineups.get(tid, set()):
                    lineups[tid].discard(pid)
                elif "in" in sub:
                    lineups.setdefault(tid, set()).add(pid)
                elif sub == "" or sub not in ("in", "out"):
                    # Some versions emit a single sub event — treat personId as the
                    # incoming player; we can't reliably track the outgoing player
                    lineups.setdefault(tid, set()).add(pid)
            continue

        if act in ("timeout", "replay", "violation", "foul", "challenge",
                   "stoppage", "officialchallenge"):
            continue

        # ── Jump ball ─────────────────────────────────────────────────────
        if act == "jumpball":
            if off_team is None and tid and tid in teams:
                off_team = tid
            continue

        if not tid or tid not in teams:
            continue

        # ── Field goals ──────────────────────────────────────────────────
        if is_fg:
            if off_team is None:
                off_team = tid
            last_shot_tid = tid
            is3 = "3pt" in act or "threepoint" in act or "3" in sub
            if tid == off_team:
                cur["fga"]        += 1
                cur["oreb_chance"] = 1  # there's a miss opportunity if not made
                if is3:
                    cur["fg3a"] += 1
                if res == "made":
                    cur["fgm"]       += 1
                    cur["oreb_chance"] = 0   # no rebound chance on a make
                    cur["pts"]       += 3 if is3 else 2
                    if is3:
                        cur["fg3m"] += 1
            if res == "made":
                _end_poss(_other(tid), clock)
            continue

        # ── Rebounds ─────────────────────────────────────────────────────
        if act == "rebound":
            if "offensive" in sub or (last_shot_tid is not None and tid == last_shot_tid):
                if off_team is None:
                    off_team = tid
                cur["oreb"] += 1
                last_shot_tid = None
            elif "deadball" in sub or "dead" in sub:
                pass   # dead-ball reb after made FT — poss already ended
            else:
                # Defensive rebound → end possession, new offense = rebounding team
                _end_poss(tid, clock)
            continue

        # ── Free throws ──────────────────────────────────────────────────
        if act == "freethrow":
            if "technical" in sub or "flagrant" in sub:
                continue
            if off_team is None and tid:
                off_team = tid
            is_made = res == "made" or ("made" in desc and "missed" not in desc)

            # Parse 'M of N' to detect last FT
            m_seq = re.search(r"(\d+)\s*of\s*(\d+)", sub + " " + desc)
            if m_seq:
                ft_i, ft_n = int(m_seq.group(1)), int(m_seq.group(2))
            else:
                ft_i, ft_n = 1, 1   # assume single FT if unparseable
            is_last = (ft_i == ft_n)

            if tid == off_team:
                cur["fta"] += 1
                if is_made:
                    cur["ftm"] += 1
                    cur["pts"] += 1

            if is_last:
                last_shot_tid = tid
                if is_made:
                    _end_poss(_other(tid), clock)
                # missed last FT → defensive (or dead-ball) rebound event follows
            continue

        # ── Turnovers ────────────────────────────────────────────────────
        if act == "turnover":
            if off_team is None and tid:
                off_team = tid
            if tid == off_team:
                cur["tov_flag"] = 1
            _end_poss(_other(tid), clock)
            continue

    # ── Split into base stints and stints_rich ────────────────────────────────
    if not poss_records:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(poss_records)

    STINTS_COLS = [
        "game_id", "poss_id", "off_team", "def_team", "points",
        "off_p1", "off_p2", "off_p3", "off_p4", "off_p5",
        "def_p1", "def_p2", "def_p3", "def_p4", "def_p5",
    ]
    RICH_COLS = STINTS_COLS + [
        "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
        "tov_flag", "oreb", "oreb_chance", "trans_flag", "game_date",
    ]

    return df[STINTS_COLS], df[RICH_COLS]


# ── CSV append helpers ────────────────────────────────────────────────────────

def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def _existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path, usecols=["game_id"])
    return set(existing["game_id"].astype(str).unique())


# ── Main ──────────────────────────────────────────────────────────────────────

def run(season: int, dry_run: bool = False) -> None:
    stints_path      = DATA_DIR / "stints"      / f"stints_{season}_RS.csv"
    stints_rich_path = DATA_DIR / "stints_rich" / f"stints_rich_{season}_RS.csv"

    print(f"Season {season} | stints → {stints_path}")

    # Find which games are already processed
    already_done = _existing_game_ids(stints_path)
    print(f"  {len(already_done)} games already in stints file")

    # Fetch full game list for the season
    print("  Fetching game list from stats.nba.com...")
    all_game_ids = fetch_game_ids(season)
    new_game_ids = [g for g in all_game_ids if g not in already_done]
    print(f"  {len(all_game_ids)} total games | {len(new_game_ids)} new")

    if dry_run:
        print(f"\nDry run — would process: {new_game_ids}")
        return

    if not new_game_ids:
        print("  Nothing to do.")
        return

    n_poss_added = 0
    for i, gid in enumerate(new_game_ids, 1):
        print(f"  [{i}/{len(new_game_ids)}] game {gid}", end=" ... ", flush=True)
        try:
            time.sleep(SLEEP_SEC)
            pbp      = fetch_pbp(gid)
            time.sleep(SLEEP_SEC)
            starters = fetch_starters(gid)
            game_date = fetch_game_date(gid, pbp)

            stints_df, rich_df = parse_game(gid, pbp, starters, game_date)

            if stints_df.empty:
                print("no possessions parsed — skipped")
                continue

            _append_csv(stints_df, stints_path)
            _append_csv(rich_df,   stints_rich_path)

            n_poss_added += len(stints_df)
            print(f"{len(stints_df)} possessions")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone — added {n_poss_added:,} possessions across {len(new_game_ids)} games.")
    print("Re-run rapm_reproducible.ipynb to update RAPM.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch new WNBA games and update stints CSVs.")
    parser.add_argument("--season",  type=int, default=None,
                        help="Season year (default: auto-detect latest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without writing anything")
    args = parser.parse_args()

    season = args.season
    if season is None:
        # Auto-detect: prefer the highest year that has an existing stints file,
        # or fall back to the current season
        import datetime
        current_year = datetime.date.today().year
        for yr in range(current_year, current_year - 2, -1):
            if (DATA_DIR / "stints" / f"stints_{yr}_RS.csv").exists():
                season = yr
                break
        if season is None:
            season = current_year
        print(f"Auto-detected season: {season}")

    run(season=season, dry_run=args.dry_run)
