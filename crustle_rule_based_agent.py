import os
import random

from cg.api import (
    Observation,
    to_observation_class,
    all_card_data,
    all_attack,
    AreaType,
    CardType,
    OptionType,
    SelectType,
    SelectContext,
    EnergyType,
)

# ---------------------------------------------------------------------------
# Deck card IDs
# ---------------------------------------------------------------------------
KANGA = 756       # Mega Kangaskhan ex (300HP, Run Errand: draw 2, Rapid-Fire Combo CCC 200+)
DWEBBLE = 344     # Dwebble (70HP, Ascension: search evolution and evolve)
CRUSTLE = 345     # Crustle (150HP, immune to damage from Pokemon ex, Superb Scissors GCC 120)
LATIAS = 184      # Latias ex (Skyliner: Basics have free retreat) - BENCH ONLY
CORNERSTONE = 117 # Cornerstone Mask Ogerpon ex (210HP, wall vs Ability Pokemon,
                  #   takes no attack damage while benched, Demolish FCC 140 pierce)
GROW_G = 18       # Grow Grass Energy (special {G}; +20 HP on a {G} Pokemon: Crustle)
ROCK_F = 20       # Rock Fighting Energy (special {F}; blocks attack effects on {F} Pokemon)
SPIKY_E = 14      # Spiky Energy (colorless, punishes attackers)
MIST_E = 11       # Mist Energy (colorless, blocks attack effects)
LILLIE = 1227     # Lillie's Determination (shuffle hand, draw 6; 8 at exactly 6 prizes)
BOSS = 1182       # Boss's Orders (gust opponent's bench)
PETREL = 1219     # Team Rocket's Petrel (search any Trainer)
HILDA = 1225      # Hilda (search Evolution Pokemon + Energy)
XEROSIC = 1197    # Xerosic's Machinations (opp discards to 3)
ERI = 1186        # Eri (discard 2 Items from opp hand)
ICECREAM = 1147   # Jumbo Ice Cream (heal 80 if 3+ energy on active)
POKEGEAR = 1122   # Pokegear 3.0 (top 7, take a Supporter)
POKEPAD = 1152    # Poke Pad (search non-rule-box Pokemon: Dwebble/Crustle)
ULTRABALL = 1121  # Ultra Ball (discard 2, search any Pokemon)
SWITCHI = 1123    # Switch
POFFIN = 1086     # Buddy-Buddy Poffin (bench up to 2 basics with HP<=70: Dwebble)
CAPE = 1159       # Hero's Cape (ACE SPEC, +100HP)
TRIMMER = 1087    # Hand Trimmer (both players discard down to 5, opponent first)
FESTIVAL = 1245   # Festival Grounds (energy-attached Pokemon immune to Special Conditions)
COMMUNITY = 1242  # Community Center (played a Supporter -> heal 10 from each of yours)

DECK_COUNTS = {
    KANGA: 2, DWEBBLE: 4, CRUSTLE: 4, LATIAS: 2, CORNERSTONE: 1,
    LILLIE: 4, BOSS: 4, PETREL: 4, HILDA: 2, ERI: 2, XEROSIC: 1,
    ICECREAM: 4, POKEGEAR: 2, POKEPAD: 4, ULTRABALL: 2,
    SWITCHI: 1, CAPE: 1, TRIMMER: 1, POFFIN: 1,
    FESTIVAL: 1, COMMUNITY: 1,
    GROW_G: 4, MIST_E: 4, SPIKY_E: 3, ROCK_F: 1,
}

STADIUMS = (FESTIVAL, COMMUNITY)
HAND_ENERGIES = (GROW_G, MIST_E, SPIKY_E, ROCK_F)

ATK_RAPID_FIRE = 1092  # Kangaskhan 200 + 50/heads
ATK_ASCENSION = 478    # Dwebble: evolve from deck
ATK_SCISSORS = 479     # Crustle 120, ignores effects on opp active
ATK_DEMOLISH = 148     # Cornerstone 140, ignores weakness/resistance and effects

CARD_DB = {c.cardId: c for c in all_card_data()}
ATTACK_DB = {a.attackId: a for a in all_attack()}

STRICT = bool(os.environ.get("PTCG_STRICT"))


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


# ---------------------------------------------------------------------------
# Card classification helpers
# ---------------------------------------------------------------------------
def db(cid):
    return CARD_DB.get(cid)


def is_ex(cid) -> bool:
    c = db(cid)
    return bool(c and (c.ex or c.megaEx))


def prizes_for(cid) -> int:
    c = db(cid)
    if c is None:
        return 1
    if c.megaEx:
        return 3
    if c.ex:
        return 2
    return 1


_EX_WALL_CACHE = {}


def is_ex_wall(cid) -> bool:
    """Pokemon whose ability prevents all damage from Pokemon ex (like Crustle)."""
    if cid in _EX_WALL_CACHE:
        return _EX_WALL_CACHE[cid]
    c = db(cid)
    v = False
    if c is not None:
        for s in c.skills:
            txt = s.text or ""
            if "Prevent all damage" in txt and "ex}" in txt:
                v = True
                break
    _EX_WALL_CACHE[cid] = v
    return v


_ABILITY_WALL_CACHE = {}


def is_ability_wall(cid) -> bool:
    """Pokemon whose ability prevents all damage from Pokemon that have an
    Ability (like Cornerstone Mask Ogerpon ex)."""
    if cid in _ABILITY_WALL_CACHE:
        return _ABILITY_WALL_CACHE[cid]
    c = db(cid)
    v = False
    if c is not None:
        for s in c.skills:
            txt = s.text or ""
            if "Prevent all damage" in txt and "have an Ability" in txt:
                v = True
                break
    _ABILITY_WALL_CACHE[cid] = v
    return v


def has_ability(cid) -> bool:
    """skills in the card DB are Abilities (attacks live in c.attacks)."""
    c = db(cid)
    return bool(c and c.skills)


def walls_against(defender_cid, attacker_cid) -> bool:
    """True if defender's ability zeroes non-piercing attacks from attacker."""
    if is_ex(attacker_cid) and is_ex_wall(defender_cid):
        return True
    if has_ability(attacker_cid) and is_ability_wall(defender_cid):
        return True
    return False


_GUST_ABILITY_CACHE = {}


def has_gust_ability(cid) -> bool:
    """Pokemon that can drag our bench into the Active Spot (Hariyama's
    Heave-Ho Catcher and friends)."""
    if cid in _GUST_ABILITY_CACHE:
        return _GUST_ABILITY_CACHE[cid]
    c = db(cid)
    v = False
    if c is not None:
        for s in c.skills:
            txt = s.text or ""
            if "Switch in 1 of your opponent" in txt:
                v = True
                break
    _GUST_ABILITY_CACHE[cid] = v
    return v


_BENCH_HITTER_CACHE = {}


def hits_bench(cid) -> bool:
    """Pokemon with an attack that damages or places counters on the
    opponent's Bench (Dragapult's Phantom Dive, Starmie's Jetting Blow).
    Benched walls/Tera do not stop placed counters."""
    if cid in _BENCH_HITTER_CACHE:
        return _BENCH_HITTER_CACHE[cid]
    c = db(cid)
    v = False
    if c is not None:
        for aid in c.attacks:
            atk = ATTACK_DB.get(aid)
            txt = (atk.text or "") if atk is not None else ""
            if "opponent’s Benched" in txt or "opponent's Benched" in txt:
                v = True
                break
    _BENCH_HITTER_CACHE[cid] = v
    return v


_THREAT_NAME_SETS = {}


def threat_names(key, pred) -> set:
    """Names of Pokemon matching pred, plus every pre-evolution that leads
    to one. Seeing a Dreepy on turn 1 means Phantom Dive is coming - react
    before the threat actually evolves into play."""
    s = _THREAT_NAME_SETS.get(key)
    if s is not None:
        return s
    by_name = {}
    for c in CARD_DB.values():
        by_name.setdefault(c.name, c)
    s = set()
    for c in CARD_DB.values():
        if not pred(c.cardId):
            continue
        s.add(c.name)
        cur = c
        for _ in range(3):
            prev = getattr(cur, "evolvesFrom", None)
            if not prev:
                break
            s.add(prev)
            cur = by_name.get(prev)
            if cur is None:
                break
    _THREAT_NAME_SETS[key] = s
    return s


def name_of(cid) -> str:
    c = db(cid)
    return c.name if c is not None else ""


_COUNTER_PLACER_CACHE = {}


def is_counter_placer(cid) -> bool:
    """Pokemon with an attack that PLACES damage counters (Alakazam's
    Powerful Hand etc). Placement is an attack effect, not damage: it goes
    through Crustle/Cornerstone walls and is blocked only by Mist Energy."""
    if cid in _COUNTER_PLACER_CACHE:
        return _COUNTER_PLACER_CACHE[cid]
    c = db(cid)
    v = False
    if c is not None:
        for aid in c.attacks:
            atk = ATTACK_DB.get(aid)
            txt = (atk.text or "") if atk is not None else ""
            if "damage counters on your opponent" in txt and (
                    "Place" in txt or "place" in txt):
                v = True
                break
    _COUNTER_PLACER_CACHE[cid] = v
    return v


# ---------------------------------------------------------------------------
# Prize knowledge: our prizes are face-down, but every deck search reveals
# the ENTIRE deck. Anything missing from deck + visible zones is prized.
# ---------------------------------------------------------------------------
PRIZED = {}  # player index -> {card id: copies stuck in our prizes}

# Only Pokemon / energies / tools: a Trainer in mid-resolution (the Petrel
# whose own search is open) sits in no visible zone and would be flagged
# as prized by mistake. These cards are never in flight during a search.
PRIZE_TRACKED = (CORNERSTONE, ROCK_F, LATIAS, KANGA, CRUSTLE, DWEBBLE,
                 CAPE, MIST_E, GROW_G, SPIKY_E)


_PRIZE_OBS = {}  # player index -> (prizes remaining, last raw observation)


def update_prize_knowledge(ctx):
    sel = ctx.sel
    if sel is None or not sel.deck:
        return
    # Only trust a full-deck view (search selects show the whole deck).
    if len(sel.deck) != ctx.my.deckCount:
        return
    deck_ids = [c.id for c in sel.deck]
    known = {}
    for cid in PRIZE_TRACKED:
        missing = (DECK_COUNTS.get(cid, 0) - deck_ids.count(cid)
                   - ctx.visible_count(cid))
        if missing > 0:
            known[cid] = missing
    # Mid-resolution picks (Hilda's second choice etc.) hide a card for one
    # view. Confirm across two consecutive views before believing a flag;
    # a prize taken in between resets confirmation.
    prizes_left = len(ctx.my.prize)
    prev = _PRIZE_OBS.get(ctx.me)
    if prev is not None and prev[0] == prizes_left:
        PRIZED[ctx.me] = {
            cid: min(cnt, prev[1][cid])
            for cid, cnt in known.items() if prev[1].get(cid, 0) > 0
        }
    else:
        PRIZED[ctx.me] = {}
    _PRIZE_OBS[ctx.me] = (prizes_left, known)


# ---------------------------------------------------------------------------
# Observation context wrapper
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.sel = obs.select
        self.st = obs.current
        self.me = self.st.yourIndex
        self.opp = 1 - self.me
        self.my = self.st.players[self.me]
        self.op = self.st.players[self.opp]

    # ---- board access
    def my_active(self):
        a = self.my.active
        return a[0] if a else None

    def opp_active(self):
        a = self.op.active
        return a[0] if a else None

    def my_in_play(self):
        out = []
        a = self.my_active()
        if a is not None:
            out.append(a)
        out.extend(p for p in self.my.bench if p is not None)
        return out

    def hand_ids(self):
        h = self.my.hand
        return [c.id for c in h] if h else []

    # ---- counting for "still in deck?" estimates
    def visible_count(self, cid) -> int:
        n = 0
        n += self.hand_ids().count(cid)
        n += sum(1 for c in self.my.discard if c.id == cid)
        for p in self.my_in_play():
            if p.id == cid:
                n += 1
            n += sum(1 for e in p.energyCards if e.id == cid)
            n += sum(1 for t in p.tools if t.id == cid)
            n += sum(1 for pe in p.preEvolution if pe.id == cid)
        for c in self.st.stadium:
            if c is not None and c.playerIndex == self.me and c.id == cid:
                n += 1
        for c in self.my.prize:
            if c is not None and c.id == cid:
                n += 1
        return n

    def maybe_in_deck(self, cid) -> bool:
        prized = PRIZED.get(self.me, {}).get(cid, 0)
        return DECK_COUNTS.get(cid, 0) - self.visible_count(cid) - prized > 0

    # ---- deck clock: do not deck ourselves out
    def draw_ok(self) -> bool:
        return self.my.deckCount > 4

    def search_ok(self) -> bool:
        return self.my.deckCount > 3

    # ---- role helpers
    def in_play_ids(self):
        return [p.id for p in self.my_in_play()]

    def kanga_in_play(self):
        return [p for p in self.my_in_play() if p.id == KANGA]

    def line_in_play(self):
        return [p for p in self.my_in_play() if p.id in (DWEBBLE, CRUSTLE)]

    def latias_in_play(self) -> bool:
        return LATIAS in self.in_play_ids()

    def main_kanga(self):
        ks = self.kanga_in_play()
        if not ks:
            return None
        act = self.my_active()
        return max(ks, key=lambda p: (len(p.energies),
                                      1 if (act and p.serial == act.serial) else 0,
                                      p.hp))

    def main_crustle(self):
        cs = [p for p in self.my_in_play() if p.id == CRUSTLE]
        pool = cs if cs else [p for p in self.my_in_play() if p.id == DWEBBLE]
        if not pool:
            return None
        act = self.my_active()
        return max(pool, key=lambda p: (len(p.energies),
                                        1 if (act and p.serial == act.serial) else 0,
                                        p.hp))

    def opp_is_ex(self) -> bool:
        a = self.opp_active()
        return a is not None and is_ex(a.id)

    def opp_in_play(self):
        out = []
        a = self.opp_active()
        if a is not None:
            out.append(a)
        out.extend(p for p in self.op.bench if p is not None)
        return out

    def opp_active_has_ability(self) -> bool:
        a = self.opp_active()
        return a is not None and has_ability(a.id)

    def mist_priority(self) -> bool:
        """Opponent has a counter-placing attacker (e.g. Alakazam) in play
        or brewing (its pre-evolutions seen): placed damage counters bypass
        our walls and only Mist Energy stops them, so the Active Spot must
        carry a Mist Energy."""
        names = threat_names("counter", is_counter_placer)
        for p in self.opp_in_play():
            if name_of(p.id) in names:
                return True
        for c in self.op.discard:
            if name_of(c.id) in names:
                return True
        return False

    def active_needs_mist(self) -> bool:
        a = self.my_active()
        if a is None or not self.mist_priority():
            return False
        return not any(e.id == MIST_E for e in a.energyCards)

    def body_need(self) -> bool:
        """2 or fewer Pokemon in play: one KO from losing on bench-out.
        Getting more bodies down outranks everything else."""
        return len(self.my_in_play()) <= 2

    def opp_spread_threat(self) -> bool:
        """Opponent has (or is evolving toward) a bench-damaging attacker:
        every extra card we bench is a free prize for them."""
        names = threat_names("spread", hits_bench)
        for p in self.opp_in_play():
            if name_of(p.id) in names:
                return True
        for c in self.op.discard:
            if name_of(c.id) in names:
                return True
        return False

    def opp_gust_threat(self) -> bool:
        """Opponent can pull our bench into the Active Spot: Boss's Orders
        in their discard, or a Heave-Ho style line seen in play/discard."""
        names = threat_names("gust", has_gust_ability)
        for c in self.op.discard:
            if c.id == BOSS or name_of(c.id) in names:
                return True
        for p in self.opp_in_play():
            if name_of(p.id) in names:
                return True
        return False

    def opp_threat_damage(self, p) -> int:
        """Best damage the opponent's active can put on p within a turn,
        honoring our wall abilities and p's weakness."""
        oa = self.opp_active()
        if oa is None or p is None:
            return 0
        if walls_against(p.id, oa.id):
            return 0
        co = db(oa.id)
        if co is None:
            return 0
        best = 0
        for aid in co.attacks:
            atk = ATTACK_DB.get(aid)
            if atk is None:
                continue
            if len(atk.energies) > len(oa.energies) + 1:
                continue  # not affordable even after their next attachment
            best = max(best, atk.damage or 0)
        dp = db(p.id)
        if (dp is not None and dp.weakness is not None
                and dp.weakness == co.energyType):
            best *= 2
        return best

    def free_basic_retreat(self) -> bool:
        # Latias ex Skyliner: your Basic Pokemon in play have no Retreat Cost.
        return self.latias_in_play()

    # ---- attack readiness / damage
    def attack_ready(self, p) -> bool:
        if p is None:
            return False
        if p.id == KANGA:
            return len(p.energies) >= 3
        if p.id == CRUSTLE:
            return len(p.energies) >= 3 and EnergyType.GRASS in p.energies
        if p.id == CORNERSTONE:
            return len(p.energies) >= 3 and EnergyType.FIGHTING in p.energies
        if p.id == DWEBBLE:
            return len(p.energies) >= 1
        return False

    def damage_vs(self, attacker, defender, assume_ready=False) -> int:
        """Floor damage attacker deals to defender, honoring wall abilities."""
        if attacker is None:
            return 0
        if not assume_ready and not self.attack_ready(attacker):
            return 0
        if attacker.id == KANGA:
            base, my_type, pierce = 200, EnergyType.COLORLESS, False
        elif attacker.id == CRUSTLE:
            base, my_type, pierce = 120, EnergyType.GRASS, True
        elif attacker.id == CORNERSTONE:
            # Demolish also ignores Weakness/Resistance.
            base, my_type, pierce = 140, None, True
        else:
            return 0
        if defender is None:
            return base
        # Wall abilities (Crustle vs ex, Cornerstone vs Ability Pokemon) stop
        # everything except attacks that ignore effects on the defender.
        if not pierce and walls_against(defender.id, attacker.id):
            return 0
        dd = db(defender.id)
        if dd is not None and my_type is not None:
            if dd.weakness is not None and dd.weakness == my_type:
                base *= 2
            if dd.resistance is not None and dd.resistance == my_type:
                base = max(0, base - 30)
        return base

    def my_active_immune(self) -> bool:
        """Our active walls the opponent's active attacker outright."""
        a = self.my_active()
        oa = self.opp_active()
        if a is None or oa is None:
            return False
        return walls_against(a.id, oa.id)

    def avg_damage_vs(self, attacker, defender, assume_ready=False) -> float:
        d = self.damage_vs(attacker, defender, assume_ready)
        if d > 0 and attacker is not None and attacker.id == KANGA:
            d += 50  # Rapid-Fire Combo averages one heads
        return d

    # ---- who do we WANT in the Active Spot right now?
    def active_score(self, p) -> float:
        if p is None:
            return -1e9
        oa = self.opp_active()
        s = 3.0 * len(p.energies) + p.hp / 50.0
        if p.id == LATIAS:
            return -500  # rule: Latias never stays active
        d = self.damage_vs(p, oa)
        s += d
        if d == 0 and self.attack_ready(p):
            s -= 10
        if oa is not None and walls_against(p.id, oa.id):
            s += 130  # wall: takes zero damage from their attacker
            if self.mist_priority() and not any(
                    e.id == MIST_E for e in p.energyCards):
                s -= 60  # wall leaks: counters still land without Mist
        if p.id == CORNERSTONE and oa is not None and not has_ability(oa.id):
            s -= 20  # no wall value here, and it gives up 2 prizes
        if p.id == DWEBBLE and self.maybe_in_deck(CRUSTLE) and len(p.energies) >= 1:
            s += 30  # Ascension turn is fine
        # Threat model: do not park something their active one-shots
        # (e.g. Kangaskhan into Hariyama's 420 through Fighting weakness).
        threat = self.opp_threat_damage(p)
        if threat >= p.hp:
            s -= 90
        elif threat >= p.hp * 0.7:
            s -= 30
        act = self.my_active()
        if act is not None and p.serial == act.serial:
            s += 25  # stickiness: do not churn
        return s

    def want_active(self):
        pool = self.my_in_play()
        if not pool:
            return None
        return max(pool, key=self.active_score)


# ---------------------------------------------------------------------------
# Option resolution
# ---------------------------------------------------------------------------
def resolve_card_id(ctx: Ctx, opt) -> int:
    """Best-effort card id for a CARD-type option. Returns -1 if unknown."""
    try:
        area = opt.area
        idx = opt.index
        pi = opt.playerIndex if opt.playerIndex is not None else ctx.me
        pl = ctx.st.players[pi]
        if area == AreaType.DECK:
            if ctx.sel.deck:
                return ctx.sel.deck[idx].id
            return -1
        if area == AreaType.HAND:
            if pl.hand:
                return pl.hand[idx].id
            return -1
        if area == AreaType.ACTIVE:
            p = pl.active[idx]
            return p.id if p is not None else -1
        if area == AreaType.BENCH:
            return pl.bench[idx].id
        if area == AreaType.DISCARD:
            return pl.discard[idx].id
        if area == AreaType.PRIZE:
            c = pl.prize[idx]
            return c.id if c is not None else -1
        if area == AreaType.STADIUM:
            c = ctx.st.stadium[idx]
            return c.id if c is not None else -1
        if area == AreaType.LOOKING:
            if ctx.st.looking:
                c = ctx.st.looking[idx]
                return c.id if c is not None else -1
            return -1
    except Exception:
        pass
    return -1


def resolve_pokemon(ctx: Ctx, area, idx, pi=None):
    """Pokemon object at area/index, or None."""
    try:
        if pi is None:
            pi = ctx.me
        pl = ctx.st.players[pi]
        if area == AreaType.ACTIVE:
            return pl.active[idx]
        if area == AreaType.BENCH:
            return pl.bench[idx]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Need flags (what should we be searching for?)
# ---------------------------------------------------------------------------
def need_flags(ctx: Ctx):
    hand = ctx.hand_ids()
    in_play = ctx.in_play_ids()
    line = ctx.line_in_play()
    kangas = ctx.kanga_in_play()
    f = {}
    # Vs gust decks (Boss's Orders seen, Hariyama on board) a benched Latias
    # is 2 free prizes waiting to be dragged up - stop wanting it.
    f["latias_need"] = (LATIAS not in in_play and LATIAS not in hand
                        and ctx.maybe_in_deck(LATIAS)
                        and not ctx.opp_gust_threat())
    f["crustle_need"] = (any(p.id == DWEBBLE for p in line)
                         and CRUSTLE not in hand and ctx.maybe_in_deck(CRUSTLE))
    f["kanga_need"] = (not kangas and KANGA not in hand
                       and ctx.maybe_in_deck(KANGA))
    f["dwebble_need"] = (len(line) < 3 and DWEBBLE not in hand
                         and ctx.maybe_in_deck(DWEBBLE))
    mk = ctx.main_kanga()
    mc = ctx.main_crustle()
    f["energy_need"] = (not any(c in HAND_ENERGIES for c in hand)
                        and ((mk is not None and len(mk.energies) < 3)
                             or (mc is not None and len(mc.energies) < 3)))
    f["grass_need"] = any(EnergyType.GRASS not in p.energies for p in line)
    f["mist_need"] = (ctx.mist_priority()
                      and MIST_E not in hand
                      and ctx.active_needs_mist()
                      and ctx.maybe_in_deck(MIST_E))
    f["cornerstone_need"] = (ctx.opp_active_has_ability()
                             and CORNERSTONE not in ctx.in_play_ids()
                             and CORNERSTONE not in hand
                             and ctx.maybe_in_deck(CORNERSTONE))
    f["supporter_in_hand"] = any(
        db(c) and db(c).cardType == CardType.SUPPORTER for c in hand)
    f["body_need"] = ctx.body_need()
    f["basic_in_hand"] = any(
        db(c) and db(c).cardType == CardType.POKEMON and db(c).basic
        for c in hand)
    return f


# ---------------------------------------------------------------------------
# Scoring: cards fetched to hand (deck searches, Pokegear looks)
# ---------------------------------------------------------------------------
def fetch_score(ctx: Ctx, cid: int) -> float:
    f = need_flags(ctx)
    hand = ctx.hand_ids()
    # Bench-out emergency: with <=2 Pokemon in play, any body beats any
    # evolution, energy or disruption card.
    if f["body_need"]:
        if cid == DWEBBLE:
            return 95
        if cid == KANGA:
            return 92
        if cid == CORNERSTONE:
            return 88
        if cid == LATIAS and LATIAS not in ctx.in_play_ids():
            return 90  # 2-prize risk beats losing to an empty bench
        if cid == POFFIN:
            return 89 if ctx.maybe_in_deck(DWEBBLE) else 10
        if cid == POKEPAD:
            return 86 if ctx.maybe_in_deck(DWEBBLE) else 12
        if cid == ULTRABALL:
            return 84
        if cid == BOSS:
            return 12  # dead card while the board is dying
    if cid == LATIAS:
        return 95 if f["latias_need"] else -5
    if cid == CRUSTLE:
        if f["crustle_need"]:
            return 90
        return 40 if any(p.id == DWEBBLE for p in ctx.line_in_play()) else 8
    if cid == KANGA:
        if f["kanga_need"]:
            return 85
        return 30 if len(ctx.kanga_in_play()) < 2 and KANGA not in hand else 5
    if cid == DWEBBLE:
        if ctx.opp_spread_threat() and len(ctx.line_in_play()) >= 2:
            return 6  # extra 70HP bodies are prizes vs bench spread
        return 60 if len(ctx.line_in_play()) < 3 else 4
    if cid == CORNERSTONE:
        return 82 if f["cornerstone_need"] else 12
    if cid == MIST_E:
        return 88 if f["mist_need"] else 46
    if cid == GROW_G:
        return 70 if f["grass_need"] else 45
    if cid == SPIKY_E:
        return 44
    if cid == ROCK_F:
        cs = [p for p in ctx.my_in_play() if p.id == CORNERSTONE]
        if cs and not any(EnergyType.FIGHTING in p.energies for p in cs):
            return 50
        return 22
    if cid == ULTRABALL:
        # Bridge to rule-box Pokemon: Latias or a needed Cornerstone.
        return 65 if (f["latias_need"] or f["cornerstone_need"]) else 24
    if cid == ICECREAM:
        a = ctx.my_active()
        if (a is not None and a.id in (KANGA, CRUSTLE, CORNERSTONE)
                and len(a.energies) >= 3 and a.maxHp - a.hp >= 70):
            return 75
        return 28
    if cid == BOSS:
        return 55
    if cid == SWITCHI:
        return 33
    if cid == POKEPAD:
        return 58 if f["crustle_need"] else 20
    if cid == POFFIN:
        if ctx.opp_spread_threat() and len(ctx.line_in_play()) >= 2:
            return 10
        return 52 if len(ctx.line_in_play()) < 3 else 10
    if cid == CAPE:
        return 48
    if cid == HILDA:
        return 42 if (f["crustle_need"] or f["energy_need"]) else 26
    if cid == PETREL:
        return 30
    if cid == TRIMMER:
        return 34 if ctx.op.handCount >= 7 else 14
    if cid == LILLIE:
        return 28
    if cid == POKEGEAR:
        return 16
    if cid in STADIUMS:
        opp_stadium = any(c is not None and c.playerIndex != ctx.me
                          for c in ctx.st.stadium)
        return 26 if opp_stadium else 12
    if cid == XEROSIC:
        # Opponent stacking cards: Xerosic is THE search target.
        if ctx.op.handCount >= 8 and XEROSIC not in hand:
            return 92
        return 9
    if cid == ERI:
        return 9
    return 6


# ---------------------------------------------------------------------------
# Scoring: how happy are we to DISCARD a card from our hand
# ---------------------------------------------------------------------------
def discard_score(ctx: Ctx, cid: int) -> float:
    f = need_flags(ctx)
    hand = ctx.hand_ids()
    in_play = ctx.in_play_ids()
    # Bench-out emergency: bodies and body-fetchers are the last to go.
    if f["body_need"]:
        c = db(cid)
        if c is not None and c.cardType == CardType.POKEMON and c.basic:
            return 1
        if cid in (POFFIN, POKEPAD, ULTRABALL):
            return 5
    # Per user strategy: sacrifice non-healing support, especially Lillie's.
    if cid == LILLIE:
        return 90
    if cid == POKEGEAR:
        return 80
    if cid == XEROSIC:
        return 75
    if cid == ERI:
        return 74 if hand.count(ERI) > 1 else 64
    if cid in STADIUMS:
        return 70
    if cid == TRIMMER:
        return 62 if ctx.op.handCount <= 5 else 34
    if cid == LATIAS:
        # A second Latias is dead weight (only one is ever benched).
        return 66 if (LATIAS in in_play or hand.count(LATIAS) > 1) else 0.5
    if cid == BOSS:
        return 65 if hand.count(BOSS) > 1 else 30
    if cid == HILDA:
        return 60 if hand.count(HILDA) > 1 else 22
    if cid == PETREL:
        return 55
    if cid == MIST_E:
        if ctx.mist_priority():
            return 4  # vs counter-placers every Mist is precious
        return 58 if not f["energy_need"] else 26
    if cid == SPIKY_E:
        return 52 if not f["energy_need"] else 24
    if cid == ROCK_F:
        cs = [p for p in ctx.my_in_play() if p.id == CORNERSTONE]
        needs = cs and not any(EnergyType.FIGHTING in p.energies for p in cs)
        return 8 if needs else 48
    if cid == GROW_G:
        return 34 if not f["grass_need"] else 10
    if cid == POKEPAD:
        return 12 if f["crustle_need"] else 46
    if cid == POFFIN:
        return 44 if len(ctx.line_in_play()) >= 3 else 18
    if cid == ULTRABALL:
        return 36
    if cid == SWITCHI:
        return 28
    if cid == ICECREAM:
        return 15  # healing items are kept per strategy
    if cid == DWEBBLE:
        return 40 if len(ctx.line_in_play()) >= 3 else 6
    if cid == CORNERSTONE:
        return 5 if f["cornerstone_need"] or ctx.opp_active_has_ability() else 38
    if cid == CRUSTLE:
        return 10 if not f["crustle_need"] else 2
    if cid == KANGA:
        return 8 if ctx.kanga_in_play() else 2
    if cid == CAPE:
        return 3
    return 25


# ---------------------------------------------------------------------------
# Scoring: which Pokemon to promote to the Active Spot (ours)
# ---------------------------------------------------------------------------
def promote_score(ctx: Ctx, p) -> float:
    if p is None:
        return 0.1
    if p.id == LATIAS:
        return 1  # never, unless it is the only legal choice
    oa = ctx.opp_active()
    s = 20 + 6 * len(p.energies) + p.hp / 25.0
    d = ctx.damage_vs(p, oa)
    s += d if d > 0 else ctx.damage_vs(p, oa, assume_ready=True) / 4.0
    if oa is not None and walls_against(p.id, oa.id):
        s += 130
    if p.id == CORNERSTONE and (oa is None or not has_ability(oa.id)):
        s -= 25  # not a wall here, just a 2-prize gift
    if p.id == DWEBBLE:
        s -= 10
    return s


# ---------------------------------------------------------------------------
# Scoring: opponent Pokemon as a target (Boss's Orders, damage effects)
# ---------------------------------------------------------------------------
def opp_target_score(ctx: Ctx, p) -> float:
    if p is None:
        return 0.1
    a = ctx.my_active()
    if not ctx.attack_ready(a):
        # Stall gust: we cannot attack this turn, so drag up the most
        # helpless body - no energy, small, cheap - to waste their turn.
        return 50 - 15 * len(p.energies) - p.hp / 20.0
    dmg = ctx.damage_vs(a, p)
    avg = ctx.avg_damage_vs(a, p)
    s = prizes_for(p.id) * 30 + 8 * len(p.energies) - p.hp / 10.0
    if dmg > 0 and p.hp <= dmg:
        s += 500  # guaranteed KO this turn
    elif avg > 0 and p.hp <= avg:
        s += 250  # KO with one heads
    if dmg == 0:
        s -= 100
    return s


# ---------------------------------------------------------------------------
# Scoring: benching from hand or deck
# ---------------------------------------------------------------------------
def bench_score(ctx: Ctx, cid: int) -> float:
    in_play = ctx.in_play_ids()
    board = len(ctx.my_in_play())
    spread = ctx.opp_spread_threat()
    # Bench-out emergency: any body goes down, no questions asked.
    if board <= 2:
        c = db(cid)
        if c is not None and c.basic:
            if cid == LATIAS and LATIAS in in_play:
                return -50
            return 80
    # Vs bench-spread attackers (Phantom Dive, Jetting Blow) a healthy
    # board of 3 is enough - every extra body is a free prize for them.
    if spread and board >= 3:
        if cid == DWEBBLE and len(ctx.line_in_play()) < 2:
            return 30  # still need the Crustle line
        if cid == CORNERSTONE and CORNERSTONE not in in_play \
                and ctx.opp_active_has_ability():
            return 25
        return -10
    if cid == LATIAS:
        if LATIAS in in_play:
            return -50
        if ctx.opp_gust_threat() or spread:
            return -10  # do not hand them a 2-prize bench target
        return 85
    if cid == KANGA:
        return 80 if len(ctx.kanga_in_play()) < 2 else -5
    if cid == DWEBBLE:
        return 75 if len(ctx.line_in_play()) < 3 else -5
    if cid == CORNERSTONE:
        if CORNERSTONE in in_play:
            return -5
        # Bench is safe (Tera: no attack damage while benched); urgent when
        # the opponent leans on Ability attackers.
        return 72 if any(has_ability(p.id) for p in ctx.opp_in_play()) else 30
    if cid == CRUSTLE:
        return 40
    c = db(cid)
    if c is not None and c.basic:
        return 15
    return 1


# ---------------------------------------------------------------------------
# Scoring: energy attachment (energy card id -> target Pokemon)
# ---------------------------------------------------------------------------
def energy_attach_score(ctx: Ctx, eid: int, t) -> float:
    if t is None:
        return -900
    if t.id == LATIAS:
        return -900  # never power up Latias (deck has no Psychic energy)
    n = len(t.energies)
    f = need_flags(ctx)
    mk = ctx.main_kanga()
    mc = ctx.main_crustle()
    opp_ex = ctx.opp_is_ex()
    act = ctx.my_active()
    is_active = act is not None and t.serial == act.serial

    # RULE: opponent has a counter-placer (Alakazam) in play -> the Active
    # Spot must hold a Mist Energy before anything else gets fed.
    if ctx.mist_priority():
        has_mist = any(e.id == MIST_E for e in t.energyCards)
        if eid == MIST_E and is_active and not has_mist:
            return 200
        if eid == MIST_E and not is_active and ctx.active_needs_mist():
            return -60  # keep Mist in hand for the Active Spot

    # RULE: never feed the bench while the active attacker is still short.
    # (Fixes: benched Kangaskhan soaking energy while active Crustle dies
    # one energy from Superb Scissors.)
    active_building = (act is not None
                       and act.id in (KANGA, CRUSTLE, CORNERSTONE, DWEBBLE)
                       and not ctx.attack_ready(act))

    if t.id == KANGA:
        is_main = mk is not None and t.serial == mk.serial
        base = 60 if is_main else 25
        if is_active:
            base += 20
        elif active_building:
            base -= 70  # active first, always
        if opp_ex:
            base -= 8  # vs ex decks, Crustle is the carry
        if n >= 3:
            base = 12
        if eid == GROW_G:
            # Grow Grass is the only card that pays Crustle's {G}; hoard it.
            if f["grass_need"] or ctx.maybe_in_deck(CRUSTLE) or CRUSTLE in ctx.hand_ids():
                base -= 100
        elif eid == ROCK_F:
            # The single Rock Fighting is Cornerstone's attack enabler.
            if CORNERSTONE in ctx.in_play_ids() or CORNERSTONE in ctx.hand_ids():
                base -= 60
        elif eid == SPIKY_E:
            base += 2
        return base
    if t.id in (CRUSTLE, DWEBBLE):
        is_main = mc is not None and t.serial == mc.serial
        base = 55 if is_main else 22
        if is_active:
            base += 20
        elif active_building:
            base -= 70
        if opp_ex and t.id == CRUSTLE:
            base += 12
        if n >= 3:
            base = 10
        if eid == GROW_G:
            # Pays the {G} cost and +20 HP on a {G} Pokemon: best on Crustle.
            base += 25 if EnergyType.GRASS not in t.energies else -8
        elif eid == ROCK_F:
            if CORNERSTONE in ctx.in_play_ids() or CORNERSTONE in ctx.hand_ids():
                base -= 60
        return base
    if t.id == CORNERSTONE:
        # Only worth powering when it is actually walling their attacker.
        walling = is_active and ctx.my_active_immune()
        base = 42 if walling else 8
        if n >= 3:
            base = 6
        if eid == ROCK_F:
            base += 30 if EnergyType.FIGHTING not in t.energies else -10
        elif eid == GROW_G:
            base -= 40  # Grow Grass belongs to the Crustle line
        return base
    return 4


# ---------------------------------------------------------------------------
# MAIN selection
# ---------------------------------------------------------------------------
def attack_option_score(ctx: Ctx, attack_id) -> float:
    oa = ctx.opp_active()
    a = ctx.my_active()
    if attack_id == ATK_RAPID_FIRE:
        return max(ctx.avg_damage_vs(a, oa, assume_ready=True), 10)
    if attack_id == ATK_SCISSORS:
        return max(ctx.damage_vs(a, oa, assume_ready=True), 10)
    if attack_id == ATK_DEMOLISH:
        return max(ctx.damage_vs(a, oa, assume_ready=True), 10)
    if attack_id == ATK_ASCENSION:
        # Evolving the active Dwebble into Crustle from the deck.
        return 300 if ctx.maybe_in_deck(CRUSTLE) else 5
    atk = ATTACK_DB.get(attack_id)
    return atk.damage if atk is not None else 50


def supporter_score(ctx: Ctx, cid: int) -> float:
    f = need_flags(ctx)
    hand = ctx.hand_ids()
    if cid == HILDA:
        if not ctx.search_ok():
            return 0
        both = f["crustle_need"] and f["energy_need"]
        if both:
            return 565
        if f["crustle_need"] or f["energy_need"]:
            return 545
        return 0
    if cid == PETREL:
        if not ctx.search_ok():
            return 0
        s = 520
        # Bench-out emergency with no body or fetcher in hand: Petrel is
        # the bridge to Poffin/Poke Pad/Ultra Ball (fetch_score picks them).
        if (f["body_need"] and not f["basic_in_hand"]
                and not any(c in (POFFIN, POKEPAD, ULTRABALL) for c in hand)):
            s = 570
        # Opponent is stacking cards: Petrel's best fetch is Xerosic.
        elif (ctx.op.handCount >= 8 and XEROSIC not in hand
                and ctx.maybe_in_deck(XEROSIC)):
            s += 40
        return s
    if cid == BOSS:
        a = ctx.my_active()
        oa = ctx.opp_active()
        bench = [p for p in ctx.op.bench if p is not None]
        if not bench:
            return 0
        ready = ctx.attack_ready(a)
        d_active = ctx.damage_vs(a, oa)
        if ready:
            # Case 1: their active is a wall we cannot damage - gust around it.
            if d_active == 0 and any(ctx.damage_vs(a, p) > 0 for p in bench):
                return 585
            # Case 2: a bench target dies this turn and is worth it.
            active_koable = oa is not None and d_active > 0 and oa.hp <= d_active
            best = None
            for p in bench:
                dp = ctx.damage_vs(a, p)
                if dp > 0 and p.hp <= dp:
                    if best is None or prizes_for(p.id) > prizes_for(best.id):
                        best = p
            if best is not None:
                if not active_koable:
                    return 585
                if prizes_for(best.id) > prizes_for(oa.id):
                    return 585
            return 0
        # Case 3: emergency stall - we cannot attack, our active is exposed
        # to their powered attacker, and their bench holds a helpless body.
        # Dragging it up buys a full setup turn for one Boss.
        threat = (oa is not None and len(oa.energies) >= 2
                  and not ctx.my_active_immune())
        helpless = [p for p in bench if len(p.energies) == 0]
        if threat and helpless:
            return 495
        return 0
    if cid == LILLIE:
        n = len(hand)
        if not ctx.draw_ok():
            # Only refuels the deck if the hand is larger than the 6 drawn.
            return 590 if n >= 7 else 0
        holding_crustle = CRUSTLE in hand and any(
            p.id == DWEBBLE for p in ctx.line_in_play())
        s = 0
        if n <= 3:
            s = 575
        elif n <= 5:
            s = 460
        elif n <= 7:
            s = 300
        if holding_crustle:
            s -= 120
        return s
    if cid == XEROSIC:
        # Single copy: save it to punish a stacked hand (>10 cards).
        # Kept below Boss-KO (585) so a guaranteed KO still comes first.
        if ctx.op.handCount > 10:
            return 560
        return 0
    if cid == ERI:
        return 440 if ctx.op.handCount >= 3 else 60
    return 0


def item_score(ctx: Ctx, cid: int) -> float:
    f = need_flags(ctx)
    hand = ctx.hand_ids()
    a = ctx.my_active()
    if cid in STADIUMS:  # stadiums, played like items
        cur = None
        for c in ctx.st.stadium:
            if c is not None:
                cur = c
        if cur is not None and cur.playerIndex == ctx.me:
            return 0  # ours is up; hold the other as a counter-stadium
        if cur is not None:
            return 880  # bounce their stadium
        # Community Center first: we play a Supporter nearly every turn.
        return 640 if cid == COMMUNITY else 600
    if cid == TRIMMER:
        # Opponent discards to 5 first; fire it when they lose more than us.
        my_after = len(hand) - 1
        opp_loss = max(0, ctx.op.handCount - 5)
        my_loss = max(0, my_after - 5)
        if opp_loss >= 2 and opp_loss > my_loss:
            return 645
        return 0
    if cid == POFFIN:
        if (not ctx.search_ok() or not ctx.maybe_in_deck(DWEBBLE)
                or len(ctx.my.bench) >= ctx.my.benchMax):
            return 0
        if f["body_need"]:
            return 760  # two bodies at once: best anti-bench-out card
        if ctx.opp_spread_threat() and len(ctx.line_in_play()) >= 2:
            return 0
        if len(ctx.line_in_play()) < 3:
            return 700
        return 0
    if cid == ULTRABALL:
        if len(hand) < 3 or not ctx.search_ok():
            return 0
        if f["body_need"] and not f["basic_in_hand"]:
            return 740
        if f["latias_need"] or f["kanga_need"]:
            return 690
        # Only 1 Rock Fighting, only 1 Cornerstone: Ultra Ball is the only
        # Item that reaches it (Poke Pad excludes rule-box Pokemon).
        # cornerstone_need is prize-aware: never burned on a prized copy.
        if f["cornerstone_need"]:
            return 685
        if f["crustle_need"] and POKEPAD not in hand:
            return 660
        return 0
    if cid == POKEPAD:
        if not ctx.search_ok():
            return 0
        if f["body_need"] and ctx.maybe_in_deck(DWEBBLE):
            return 750  # fetch_score ranks the Dwebble body over Crustle
        if f["crustle_need"]:
            return 680
        if f["dwebble_need"] and len(ctx.line_in_play()) < 2:
            return 640
        return 0
    if cid == ICECREAM:
        if (a is not None and a.id in (KANGA, CRUSTLE, CORNERSTONE)
                and len(a.energies) >= 3 and a.maxHp - a.hp >= 70):
            return 675
        return 0
    if cid == POKEGEAR:
        if ctx.my.deckCount <= 2:
            return 0
        if not f["supporter_in_hand"] and not ctx.st.supporterPlayed:
            return 650
        # Opponent stacking cards: dig for Xerosic even with a supporter
        # already in hand (fetch_score makes Xerosic the pick).
        if (ctx.op.handCount >= 8 and XEROSIC not in hand
                and ctx.maybe_in_deck(XEROSIC)):
            return 630
        return 0
    if cid == SWITCHI:
        return switch_item_score(ctx)
    return 0


def switch_item_score(ctx: Ctx) -> float:
    a = ctx.my_active()
    if a is None or ctx.st.retreated:
        return 0
    want = ctx.want_active()
    if want is None or a.serial == want.serial:
        return 0
    # Meaningfully better target on the bench.
    if ctx.active_score(want) < ctx.active_score(a) + 40:
        return 0
    # If retreat is free, prefer RETREAT and keep the Switch.
    c = db(a.id)
    free = (c is not None and c.basic and ctx.free_basic_retreat()) or a.id == LATIAS
    if free:
        return 0
    return 400


def retreat_score(ctx: Ctx) -> float:
    a = ctx.my_active()
    if a is None:
        return 0
    want = ctx.want_active()
    if want is None or a.serial == want.serial:
        return 0
    if ctx.active_score(want) < ctx.active_score(a) + 40:
        return 0
    c = db(a.id)
    free = (c is not None and c.basic and ctx.free_basic_retreat()) or a.id == LATIAS
    if free:
        return 350
    # Paying energy to retreat: only from a Pokemon whose energy barely matters.
    if a.id == DWEBBLE and len(a.energies) >= 2 and not ctx.maybe_in_deck(CRUSTLE):
        return 330
    # Latias gone / no free retreat: paying is still right when the bench
    # holds a wall that blanks their attacker, or the active dies for
    # nothing next turn anyway (the energy is lost either way).
    oa = ctx.opp_active()
    want_walls = oa is not None and walls_against(want.id, oa.id)
    doomed = (ctx.opp_threat_damage(a) >= a.hp and not ctx.attack_ready(a))
    if want_walls or doomed:
        return 300
    return 0


def ability_score(ctx: Ctx, o) -> float:
    """Run Errand draws cards: skip it when the deck is thin. Community
    Center's heal is always free value."""
    cid = -1
    try:
        if o.area == AreaType.STADIUM:
            c = ctx.st.stadium[o.index]
            cid = c.id if c is not None else -1
        else:
            pk = resolve_pokemon(ctx, o.area, o.index)
            cid = pk.id if pk is not None else -1
    except Exception:
        pass
    if cid == KANGA and not ctx.draw_ok():
        return 0.4  # below END: never take this
    return 1000


def choose_main(ctx: Ctx) -> int:
    hand = ctx.my.hand or []
    best_idx, best_score = 0, -1e18
    for i, o in enumerate(ctx.sel.option):
        s = 0.0
        t = o.type
        if t == OptionType.ABILITY:
            s = ability_score(ctx, o)
        elif t == OptionType.EVOLVE:
            s = 950
            if o.inPlayArea == AreaType.ACTIVE:
                s += 10
        elif t == OptionType.PLAY:
            cid = hand[o.index].id if o.index is not None and o.index < len(hand) else -1
            c = db(cid)
            if c is None:
                s = 0
            elif c.cardType == CardType.POKEMON and c.basic:
                b = bench_score(ctx, cid)
                s = 800 + b if b > 0 else 0
            elif c.cardType == CardType.SUPPORTER:
                s = supporter_score(ctx, cid)
            else:
                s = item_score(ctx, cid)
        elif t == OptionType.ATTACH:
            cid = -1
            try:
                if o.area == AreaType.HAND and ctx.my.hand:
                    cid = ctx.my.hand[o.index].id
            except Exception:
                pass
            target = resolve_pokemon(ctx, o.inPlayArea, o.inPlayIndex)
            if cid == CAPE:
                mk = ctx.main_kanga()
                mc = ctx.main_crustle()
                # Fallback plan for a prized/lost Cornerstone vs one-shot
                # threats (Hariyama does 420 into Kangaskhan): even a caped
                # Kanga dies, so the Cape belongs on Crustle - 270 HP that
                # survives the hit and heals 80 back with Jumbo Ice Cream.
                kanga_doomed = (mk is None or
                                ctx.opp_threat_damage(mk) >= mk.maxHp + 100)
                s = 850
                if target is None:
                    pass
                elif target.id == LATIAS:
                    s = 2
                elif (kanga_doomed and mc is not None and mc.id == CRUSTLE
                        and target.serial == mc.serial):
                    s += 25
                elif (target.id == CORNERSTONE
                        and ctx.opp_active_has_ability()):
                    s += 22  # 310 HP wall that their attacker cannot dent
                elif mk is not None and target.serial == mk.serial:
                    s += 20
                elif target.id == KANGA:
                    s += 10
                elif target.id == CRUSTLE:
                    s += 5
            elif cid in HAND_ENERGIES:
                s = 780 + energy_attach_score(ctx, cid, target)
            else:
                s = 3
        elif t == OptionType.RETREAT:
            s = retreat_score(ctx)
        elif t == OptionType.ATTACK:
            s = 200 + attack_option_score(ctx, o.attackId) / 1000.0
        elif t == OptionType.DISCARD:
            s = 0.2  # discarding our own cards in play: avoid
        elif t == OptionType.END:
            s = 1
        else:
            s = 0.5
        if s > best_score:
            best_idx, best_score = i, s
    return best_idx


# ---------------------------------------------------------------------------
# CARD-type selections
# ---------------------------------------------------------------------------
COST_CONTEXTS = {
    SelectContext.DISCARD,
    SelectContext.TO_DECK,
    SelectContext.TO_DECK_BOTTOM,
    SelectContext.TO_PRIZE,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
}


def card_option_score(ctx: Ctx, o) -> float:
    c = ctx.sel.context
    pi = o.playerIndex if o.playerIndex is not None else ctx.me
    mine = pi == ctx.me
    cid = resolve_card_id(ctx, o)
    pk = None
    if o.area in (AreaType.ACTIVE, AreaType.BENCH):
        pk = resolve_pokemon(ctx, o.area, o.index, pi)

    if c in (SelectContext.SETUP_ACTIVE_POKEMON,):
        # Setup: prefer Kangaskhan tank; keep Latias off the Active Spot.
        return {KANGA: 100, DWEBBLE: 70, CRUSTLE: 60, CORNERSTONE: 40,
                LATIAS: 5}.get(cid, 20)
    if c in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if mine:
            return promote_score(ctx, pk)
        return opp_target_score(ctx, pk)
    if c in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH,
             SelectContext.TO_FIELD):
        return bench_score(ctx, cid)
    if c in (SelectContext.TO_HAND, SelectContext.LOOK):
        return fetch_score(ctx, cid)
    if c in COST_CONTEXTS:
        if mine:
            return discard_score(ctx, cid)
        return 100 + fetch_score(ctx, cid)  # discarding opponent cards: yes
    if c in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
        if pk is None:
            return 1
        dmg = pk.maxHp - pk.hp
        bonus = 30 if pk.id in (KANGA, CRUSTLE, CORNERSTONE) else 0
        return dmg + bonus
    if c in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
             SelectContext.DAMAGE_COUNTER_ANY):
        if mine:
            return -50 - (pk.hp if pk else 0)  # avoid damaging our own
        return opp_target_score(ctx, pk)
    if c in (SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO,
             SelectContext.EFFECT_TARGET):
        if pk is not None:
            if pk.id == LATIAS:
                return 1
            if not mine:
                return opp_target_score(ctx, pk)
            mk, mc = ctx.main_kanga(), ctx.main_crustle()
            if mk is not None and pk.serial == mk.serial:
                return 80
            if mc is not None and pk.serial == mc.serial:
                return 70
            return 20
        return fetch_score(ctx, cid)
    if c in (SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO,
             SelectContext.DEVOLVE):
        return 50
    if c == SelectContext.NOT_MOVE:
        # Keep the valuable ones where they are.
        return fetch_score(ctx, cid)
    # Unknown context: mild preference for useful cards.
    return fetch_score(ctx, cid) if mine else 10


def select_cards(ctx: Ctx) -> list[int]:
    opts = ctx.sel.option
    scores = [card_option_score(ctx, o) for o in opts]
    order = sorted(range(len(opts)), key=lambda i: -scores[i])
    c = ctx.sel.context
    own_cost = c in COST_CONTEXTS and all(
        (o.playerIndex is None or o.playerIndex == ctx.me) for o in opts)
    if own_cost:
        n = ctx.sel.minCount
    else:
        pos = sum(1 for s in scores if s > 0)
        n = max(ctx.sel.minCount, min(pos, ctx.sel.maxCount))
    return order[:n]


# ---------------------------------------------------------------------------
# Other selection types
# ---------------------------------------------------------------------------
def select_energy(ctx: Ctx) -> list[int]:
    """Choosing attached energy (retreat cost, discard effects)."""
    opts = ctx.sel.option
    scores = []
    for o in opts:
        pk = resolve_pokemon(ctx, o.area, o.index,
                             o.playerIndex if o.playerIndex is not None else ctx.me)
        eid = -1
        try:
            if pk is not None and o.energyIndex is not None:
                eid = pk.energyCards[o.energyIndex].id
        except Exception:
            pass
        # Higher score = more willing to discard.
        mist_keep = 4 if (pk is not None and ctx.mist_priority()) else 60
        if pk is not None and pk.id in (CRUSTLE, DWEBBLE):
            s = {MIST_E: mist_keep, SPIKY_E: 50, ROCK_F: 45, GROW_G: 10}.get(eid, 40)
        elif pk is not None and pk.id == CORNERSTONE:
            s = {MIST_E: mist_keep, GROW_G: 55, SPIKY_E: 50, ROCK_F: 10}.get(eid, 40)
        else:
            s = {MIST_E: mist_keep, GROW_G: 55, SPIKY_E: 45, ROCK_F: 35}.get(eid, 40)
        scores.append(s)
    order = sorted(range(len(opts)), key=lambda i: -scores[i])
    return order[:ctx.sel.minCount]  # pay exactly the required cost


def select_yes_no(ctx: Ctx) -> list[int]:
    want_yes = True
    if ctx.sel.context == SelectContext.MORE_DEVOLVE:
        want_yes = False
    if ctx.sel.context == SelectContext.ACTIVATE:
        cc = ctx.sel.contextCard
        if cc is not None and cc.id == KANGA and not ctx.draw_ok():
            want_yes = False  # skip draw-2 when the deck is nearly empty
    for i, o in enumerate(ctx.sel.option):
        if o.type == OptionType.YES and want_yes:
            return [i]
        if o.type == OptionType.NO and not want_yes:
            return [i]
    return [0]


def select_count(ctx: Ctx) -> list[int]:
    best_i, best_n = 0, -1
    for i, o in enumerate(ctx.sel.option):
        n = o.number if o.number is not None else 0
        if n > best_n:
            best_i, best_n = i, n
    return [best_i]


def select_attack(ctx: Ctx) -> list[int]:
    best_i, best_s = 0, -1e18
    for i, o in enumerate(ctx.sel.option):
        s = attack_option_score(ctx, o.attackId)
        if s > best_s:
            best_i, best_s = i, s
    return [best_i]


def select_special_condition(ctx: Ctx) -> list[int]:
    from cg.api import SpecialConditionType as SC
    if ctx.sel.context == SelectContext.RECOVER_SPECIAL_CONDITION:
        order = [SC.PARALYZE, SC.SLEEP, SC.CONFUSE, SC.POISON, SC.BURN]
    else:
        order = [SC.PARALYZE, SC.POISON, SC.CONFUSE, SC.BURN, SC.SLEEP]
    for want in order:
        for i, o in enumerate(ctx.sel.option):
            if o.specialConditionType == want:
                return [i]
    return [0]


def select_evolve(ctx: Ctx) -> list[int]:
    best_i, best_s = 0, -1e18
    for i, o in enumerate(ctx.sel.option):
        s = 10
        if o.inPlayArea == AreaType.ACTIVE:
            s += 5
        if s > best_s:
            best_i, best_s = i, s
    return [best_i]


# ---------------------------------------------------------------------------
# Master dispatch
# ---------------------------------------------------------------------------
class PolicyRuleBased:
    def act(self, obs: Observation) -> list[int]:
        obs = to_observation_class(obs)
        ctx = Ctx(obs)
        update_prize_knowledge(ctx)
        sel = obs.select
        t = sel.type
        if t == SelectType.MAIN:
            return [choose_main(ctx)]
        if t == SelectType.YES_NO:
            return select_yes_no(ctx)
        if t == SelectType.COUNT:
            return select_count(ctx)
        if t == SelectType.ATTACK:
            return select_attack(ctx)
        if t == SelectType.EVOLVE:
            return select_evolve(ctx)
        if t == SelectType.ENERGY:
            return select_energy(ctx)
        if t == SelectType.SPECIAL_CONDITION:
            return select_special_condition(ctx)
        if t == SelectType.SKILL:
            return list(range(sel.maxCount))
        if t in (SelectType.CARD, SelectType.ATTACHED_CARD,
                 SelectType.CARD_OR_ATTACHED_CARD):
            return select_cards(ctx)
        # Unknown selection type: pick the first legal amount.
        return list(range(max(sel.minCount, min(1, sel.maxCount))))


def rule_based_select(obs: Observation) -> list[int]:
    ctx = Ctx(obs)
    update_prize_knowledge(ctx)
    sel = obs.select
    t = sel.type
    if t == SelectType.MAIN:
        return [choose_main(ctx)]
    if t == SelectType.YES_NO:
        return select_yes_no(ctx)
    if t == SelectType.COUNT:
        return select_count(ctx)
    if t == SelectType.ATTACK:
        return select_attack(ctx)
    if t == SelectType.EVOLVE:
        return select_evolve(ctx)
    if t == SelectType.ENERGY:
        return select_energy(ctx)
    if t == SelectType.SPECIAL_CONDITION:
        return select_special_condition(ctx)
    if t == SelectType.SKILL:
        return list(range(sel.maxCount))
    if t in (SelectType.CARD, SelectType.ATTACHED_CARD,
             SelectType.CARD_OR_ATTACHED_CARD):
        return select_cards(ctx)
    # Unknown selection type: pick the first legal amount.
    return list(range(max(sel.minCount, min(1, sel.maxCount))))


def validate(sel, choice: list[int]) -> list[int]:
    """Clamp any bug to a legal selection instead of an invalid move."""
    n_opt = len(sel.option)
    seen = []
    for v in choice:
        if isinstance(v, int) and 0 <= v < n_opt and v not in seen:
            seen.append(v)
    while len(seen) < sel.minCount:
        for v in range(n_opt):
            if v not in seen:
                seen.append(v)
                break
        else:
            break
    return seen[:max(sel.minCount, min(len(seen), sel.maxCount))]


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point: returns option indices for the current selection."""
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        choice = rule_based_select(obs)
        return validate(obs.select, choice)
    except Exception:
        if STRICT:
            raise
        return validate(obs.select, random.sample(
            list(range(len(obs.select.option))), obs.select.maxCount))
