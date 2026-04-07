import requests
import math
from datetime import datetime

BASE_URL = "https://www.fotmob.com/api"
LEAGUE_AVG_SHOTS_CONCEDED = 11.0
TOUCHES_THRESHOLD = 55.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

POSITIONS_DEF = {"defender", "centre-back", "right back", "left back", "rb", "lb", "cb", "df"}
POSITIONS_MID = {"midfielder", "central midfield", "defensive midfield", "attacking midfield",
                 "right midfield", "left midfield", "mf", "cm", "dm", "am"}
POSITIONS_ATT = {"forward", "striker", "centre-forward", "right winger", "left winger",
                 "second striker", "fw", "st", "lw", "rw", "cf"}


# ─────────────────────────────────────────────
#  FOTMOB API
# ─────────────────────────────────────────────

def get(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def search_player(name: str) -> dict | None:
    data = get(f"{BASE_URL}/search?term={requests.utils.quote(name)}")
    players = data.get("squadMember", {}).get("items", [])
    if not players:
        print(f"  [!] No player found for '{name}'")
        return None
    p = players[0]
    print(f"  [✓] Found player : {p.get('name')} (id={p.get('id')}, team={p.get('teamName')})")
    return p


def search_team(name: str) -> dict | None:
    data = get(f"{BASE_URL}/search?term={requests.utils.quote(name)}")
    teams = data.get("team", {}).get("items", [])
    if not teams:
        print(f"  [!] No team found for '{name}'")
        return None
    t = teams[0]
    print(f"  [✓] Found team   : {t.get('name')} (id={t.get('id')})")
    return t


def fetch_player_stats(player_id: int, team_id: int, season: int = 2024) -> dict:
    return get(f"{BASE_URL}/playerStats?playerId={player_id}&seasonId={season}&teamId={team_id}")


def fetch_team_data(team_id: int) -> dict:
    return get(f"{BASE_URL}/teams?id={team_id}")


def fetch_match_details(match_id: int) -> dict:
    return get(f"{BASE_URL}/matchDetails?matchId={match_id}")


def find_match_between(team1_id: int, team2_id: int) -> dict | None:
    """Search today's matches and team fixtures for a match between two teams."""
    today = datetime.now().strftime("%Y%m%d")
    data = get(f"{BASE_URL}/matches?date={today}")

    for league in data.get("leagues", []):
        for match in league.get("matches", []):
            home = match.get("home", {})
            away = match.get("away", {})
            ids = {home.get("id"), away.get("id")}
            if team1_id in ids and team2_id in ids:
                return match

    # fallback: check team fixtures
    team_data = fetch_team_data(team1_id)
    fixtures = team_data.get("fixtures", {}).get("allFixtures", {}).get("fixtures", [])
    for fix in fixtures:
        home_id = fix.get("home", {}).get("id")
        away_id = fix.get("away", {}).get("id")
        if team2_id in (home_id, away_id):
            return fix
    return None


def fetch_lineup(match_id: int) -> dict | None:
    details = fetch_match_details(match_id)
    return details.get("content", {}).get("lineup")


def extract_stat(data: dict, *keywords) -> float:
    def walk(node):
        if isinstance(node, dict):
            title = node.get("title", "").lower()
            key   = node.get("key",   "").lower()
            if any(kw.lower() in title or kw.lower() in key for kw in keywords):
                raw = node.get("value") or node.get("stat", {}).get("value")
                if raw is not None:
                    try:
                        return float(str(raw).replace(",", ".").replace("%", ""))
                    except ValueError:
                        pass
            for v in node.values():
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for item in node:
                r = walk(item)
                if r is not None:
                    return r
        return None
    return walk(data) or 0.0


def get_shots_conceded(team_data: dict) -> float:
    for kw in ("shots conceded", "shotsConceded", "shots against", "conceded", "against"):
        v = extract_stat(team_data, kw)
        if v:
            return v
    return 0.0


def is_position(pos_str: str, pos_set: set) -> bool:
    p = (pos_str or "").lower()
    return any(pos in p for pos in pos_set)


# ─────────────────────────────────────────────
#  ALGORITHMS
# ─────────────────────────────────────────────

def calc_foul_probability(fouls_per_match_pct: float, touches_per_90: float) -> float:
    """
    P(foul) = (fouls per match %) x (touches / 100)
    fouls_per_match_pct : % of matches the defender commits a foul (0-100)
    touches_per_90      : target player touches per 90 min
    """
    base         = fouls_per_match_pct / 100.0
    touch_factor = min(touches_per_90 / 100.0, 1.0)
    return round(base * touch_factor * 100.0, 2)


def calc_shot_probability(shots_per_90: float, opp_shots_conceded: float,
                          league_avg: float = LEAGUE_AVG_SHOTS_CONCEDED) -> float:
    """
    Poisson model: P(>=1 shot) = 1 - e^(-lambda)
    lambda = shots_per_90 x (opp_shots_conceded / league_avg)
    """
    defensive_factor = opp_shots_conceded / league_avg
    lam = shots_per_90 * defensive_factor
    return round((1 - math.exp(-lam)) * 100.0, 2)


# ─────────────────────────────────────────────
#  MATCH ANALYSIS
# ─────────────────────────────────────────────

def get_lineup_players(lineup: dict, side: str) -> list:
    team     = lineup.get(f"{side}Team", {})
    players  = []
    for line in team.get("players", []):
        if isinstance(line, list):
            for p in line:
                players.append(p)
        elif isinstance(line, dict):
            players.append(line)
    starters = [p for p in players if not p.get("isSub", False)]
    return starters[:11]


def enrich_player(p: dict, team_id: int) -> dict:
    pid = p.get("id") or p.get("playerId")
    if not pid:
        return p
    try:
        stats = fetch_player_stats(int(pid), team_id)
        p["_fouls_per_90"]   = extract_stat(stats, "fouls committed", "foulCommitted", "fouls")
        p["_touches_per_90"] = extract_stat(stats, "touches", "touchesPerGame", "touches per")
        p["_shots_per_90"]   = extract_stat(stats, "shots per game", "shotsPerGame", "shots per 90", "shots")
    except Exception:
        p["_fouls_per_90"]   = 0.0
        p["_touches_per_90"] = 0.0
        p["_shots_per_90"]   = 0.0
    return p


def run_match_analysis():
    print("\n[MATCH ANALYSIS — confirmed/predicted lineup]")
    t1_name = input("  Home team (e.g. Real Madrid): ").strip()
    t2_name = input("  Away team (e.g. Bayern Munich): ").strip()

    print("\n  Searching teams...")
    t1 = search_team(t1_name)
    t2 = search_team(t2_name)
    if not t1 or not t2:
        return

    print("\n  Looking for match...")
    match = find_match_between(t1["id"], t2["id"])
    if not match:
        print("  [!] No upcoming match found between these teams on fotmob.")
        return

    match_id   = match.get("id")
    match_name = (f"{match.get('home', {}).get('name', t1_name)} vs "
                  f"{match.get('away', {}).get('name', t2_name)}")
    print(f"  [✓] Match found  : {match_name} (id={match_id})")

    print("\n  Fetching lineup...")
    lineup = fetch_lineup(match_id)
    if not lineup:
        print("  [!] No lineup available yet.")
        return

    confirmed = lineup.get("confirmed", False)
    status    = "CONFIRMED" if confirmed else "PREDICTED"
    print(f"  [✓] Lineup status: {status}")

    home_players = get_lineup_players(lineup, "home")
    away_players = get_lineup_players(lineup, "away")

    # ensure home/away sides match t1/t2
    lineup_home_id = lineup.get("homeTeam", {}).get("teamId") or lineup.get("homeTeamId")
    if lineup_home_id and int(lineup_home_id) == int(t2["id"]):
        home_players, away_players = away_players, home_players

    total = len(home_players) + len(away_players)
    print(f"\n  Fetching stats for {total} players (this may take a moment)...")
    home_players = [enrich_player(p, t1["id"]) for p in home_players]
    away_players = [enrich_player(p, t2["id"]) for p in away_players]

    print("\n  Fetching team defensive stats...")
    t1_data       = fetch_team_data(t1["id"])
    t2_data       = fetch_team_data(t2["id"])
    t1_shots_conc = get_shots_conceded(t1_data) or float(
        input(f"  Enter {t1_name} shots conceded/match manually: "))
    t2_shots_conc = get_shots_conceded(t2_data) or float(
        input(f"  Enter {t2_name} shots conceded/match manually: "))

    # ── FOUL ANALYSIS ──────────────────────────────────────
    foul_results = []

    def check_fouls(defenders, attackers, def_team, att_team):
        for d in defenders:
            pos_d = str(d.get("position", d.get("positionId", "")))
            if not is_position(pos_d, POSITIONS_DEF):
                continue
            fouls_per_90 = d.get("_fouls_per_90", 0)
            if fouls_per_90 == 0:
                continue
            fouls_pct = min(fouls_per_90 / 3.0, 1.0) * 100.0
            for a in attackers:
                pos_a = str(a.get("position", a.get("positionId", "")))
                if not is_position(pos_a, POSITIONS_MID | POSITIONS_ATT):
                    continue
                touches = a.get("_touches_per_90", 0)
                if touches < TOUCHES_THRESHOLD:
                    continue
                prob = calc_foul_probability(fouls_pct, touches)
                foul_results.append({
                    "committer":       d.get("name", "Unknown"),
                    "committer_team":  def_team,
                    "fouls_pct":       round(fouls_pct, 1),
                    "target":          a.get("name", "Unknown"),
                    "target_team":     att_team,
                    "touches":         touches,
                    "probability":     prob,
                })

    check_fouls(away_players, home_players, t2_name, t1_name)
    check_fouls(home_players, away_players, t1_name, t2_name)
    foul_results.sort(key=lambda x: x["probability"], reverse=True)

    # ── SHOT ANALYSIS ──────────────────────────────────────
    shot_results = []

    def check_shots(attackers, att_team, opp_team, opp_shots_conc):
        for a in attackers:
            pos_a = str(a.get("position", a.get("positionId", "")))
            if not is_position(pos_a, POSITIONS_ATT | POSITIONS_MID):
                continue
            shots = a.get("_shots_per_90", 0)
            if shots == 0:
                continue
            prob = calc_shot_probability(shots, opp_shots_conc)
            shot_results.append({
                "player":      a.get("name", "Unknown"),
                "team":        att_team,
                "shots_per90": shots,
                "opp_team":    opp_team,
                "opp_conc":    opp_shots_conc,
                "probability": prob,
            })

    check_shots(home_players, t1_name, t2_name, t2_shots_conc)
    check_shots(away_players, t2_name, t1_name, t1_shots_conc)
    shot_results.sort(key=lambda x: x["probability"], reverse=True)

    # ── PRINT RESULTS ──────────────────────────────────────
    w = 72
    print(f"\n{'='*w}")
    print(f"  MATCH PREDICTION — {match_name}  [{status} LINEUP]")
    print(f"{'='*w}")

    print(f"\n  FOUL PROBABILITY  (MF/ATT with >{TOUCHES_THRESHOLD} touches/90)")
    print(f"  {'Committer':<22} {'Team':<16} {'Fouls%':>7}  {'Target':<22} {'T/90':>6}  {'Prob':>6}")
    print(f"  {'-'*22} {'-'*16} {'-'*7}  {'-'*22} {'-'*6}  {'-'*6}")
    if foul_results:
        for r in foul_results:
            print(f"  {r['committer']:<22} {r['committer_team']:<16} {r['fouls_pct']:>6}%  "
                  f"{r['target']:<22} {r['touches']:>6.1f}  {r['probability']:>5.1f}%")
    else:
        print("  No qualifying matchups found.")

    print(f"\n  SHOT PROBABILITY")
    print(f"  {'Player':<22} {'Team':<16} {'Shots/90':>8}  {'vs':<16} {'Conc/M':>6}  {'Prob':>6}")
    print(f"  {'-'*22} {'-'*16} {'-'*8}  {'-'*16} {'-'*6}  {'-'*6}")
    if shot_results:
        for r in shot_results:
            print(f"  {r['player']:<22} {r['team']:<16} {r['shots_per90']:>8.2f}  "
                  f"{r['opp_team']:<16} {r['opp_conc']:>6.1f}  {r['probability']:>5.1f}%")
    else:
        print("  No qualifying players found.")

    print(f"\n{'='*w}")


# ─────────────────────────────────────────────
#  SINGLE PLAYER MODES
# ─────────────────────────────────────────────

def run_foul_live():
    print("\n[FOUL PROBABILITY — live fotmob data]")
    committer_name = input("  Foul committer (e.g. Jonathan Tah): ").strip()
    target_name    = input("  Target player  (e.g. Vinicius Jr): ").strip()

    print("\n  Searching fotmob...")
    committer = search_player(committer_name)
    target    = search_player(target_name)
    if not committer or not target:
        return

    print("\n  Fetching stats...")
    c_stats = fetch_player_stats(committer["id"], committer.get("teamId", 0))
    t_stats = fetch_player_stats(target["id"],    target.get("teamId", 0))

    fouls_per_90 = extract_stat(c_stats, "fouls committed", "foulCommitted", "fouls")
    touches      = extract_stat(t_stats, "touches", "touchesPerGame", "touches per")

    if fouls_per_90 == 0:
        fouls_per_90 = float(input(f"\n  Could not find fouls/90 for {committer_name}. Enter manually: "))
    if touches == 0:
        touches = float(input(f"  Could not find touches/90 for {target_name}. Enter manually: "))

    fouls_pct = min(fouls_per_90 / 3.0, 1.0) * 100.0
    prob      = calc_foul_probability(fouls_pct, touches)

    print(f"""
{'='*52}
  FOUL PROBABILITY
{'='*52}
  Foul Committer   : {committer_name}
  Fouls/Match %    : {round(fouls_pct, 1)}%
  Target Player    : {target_name}
  Touches/90       : {touches}
{'-'*52}
  Probability      : {prob}%
{'='*52}""")


def run_shot_live():
    print("\n[SHOT PROBABILITY — live fotmob data]")
    player_name = input("  Player name      (e.g. Vinicius Jr): ").strip()
    opp_name    = input("  Opposition team  (e.g. Bayern Munich): ").strip()

    print("\n  Searching fotmob...")
    player   = search_player(player_name)
    opp_team = search_team(opp_name)
    if not player or not opp_team:
        return

    print("\n  Fetching stats...")
    p_stats = fetch_player_stats(player["id"], player.get("teamId", 0))
    t_stats = fetch_team_data(opp_team["id"])

    shots    = extract_stat(p_stats, "shots per game", "shotsPerGame", "shots per 90", "shots")
    conceded = get_shots_conceded(t_stats)

    if shots == 0:
        shots    = float(input(f"\n  Could not find shots/90 for {player_name}. Enter manually: "))
    if conceded == 0:
        conceded = float(input(f"  Could not find shots conceded for {opp_name}. Enter manually: "))

    prob = calc_shot_probability(shots, conceded)

    print(f"""
{'='*52}
  SHOT PROBABILITY
{'='*52}
  Player           : {player_name}
  Shots/90         : {shots}
  Opposition       : {opp_name}
  Shots Conceded   : {conceded}/match
  League Avg       : {LEAGUE_AVG_SHOTS_CONCEDED}/match
{'-'*52}
  Probability      : {prob}%
{'='*52}""")


def run_foul_manual():
    print("\n[FOUL PROBABILITY — manual input]")
    committer = input("  Foul committer name: ").strip()
    fouls_pct = float(input("  Fouls committed per match % (e.g. 68): "))
    target    = input("  Target player name: ").strip()
    touches   = float(input("  Target touches per 90 (e.g. 70): "))
    prob      = calc_foul_probability(fouls_pct, touches)
    print(f"""
{'='*52}
  FOUL PROBABILITY
{'='*52}
  Foul Committer   : {committer}
  Fouls/Match %    : {fouls_pct}%
  Target Player    : {target}
  Touches/90       : {touches}
{'-'*52}
  Probability      : {prob}%
{'='*52}""")


def run_shot_manual():
    print("\n[SHOT PROBABILITY — manual input]")
    player   = input("  Player name: ").strip()
    shots    = float(input("  Shots per 90 (e.g. 3.2): "))
    opp      = input("  Opposition team: ").strip()
    conceded = float(input("  Opposition shots conceded per match (e.g. 14): "))
    prob     = calc_shot_probability(shots, conceded)
    print(f"""
{'='*52}
  SHOT PROBABILITY
{'='*52}
  Player           : {player}
  Shots/90         : {shots}
  Opposition       : {opp}
  Shots Conceded   : {conceded}/match
  League Avg       : {LEAGUE_AVG_SHOTS_CONCEDED}/match
{'-'*52}
  Probability      : {prob}%
{'='*52}""")


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────

def main():
    print("╔════════════════════════════════════════════╗")
    print("║  FOOTBALL MATCH PROBABILITY PREDICTOR v1   ║")
    print("║  Data source: FotMob                       ║")
    print("╚════════════════════════════════════════════╝")

    while True:
        print("""
  ── MATCH ANALYSIS ───────────────────────────
  7. Full Match Prediction  (lineup + both algos)

  ── SINGLE PLAYER ────────────────────────────
  1. Foul Probability    (live fotmob)
  2. Shot Probability    (live fotmob)
  3. Both                (live fotmob)
  4. Foul Probability    (manual input)
  5. Shot Probability    (manual input)
  6. Both                (manual input)

  0. Exit
""")
        choice = input("  Choice: ").strip()

        if   choice == "0": break
        elif choice == "1": run_foul_live()
        elif choice == "2": run_shot_live()
        elif choice == "3": run_foul_live(); run_shot_live()
        elif choice == "4": run_foul_manual()
        elif choice == "5": run_shot_manual()
        elif choice == "6": run_foul_manual(); run_shot_manual()
        elif choice == "7": run_match_analysis()
        else: print("  Invalid choice.")

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
