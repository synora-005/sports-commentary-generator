"""Pure-Python logic (no torch): dataset generation, prompt format, cleaning, scoring.

Milestone 1 -> generate_dataset / format_example / save_jsonl / load_jsonl / split_dataset
Milestone 3 -> clean_commentary (turns raw model output into a displayable line)
Milestone 4 -> relevance_score / is_well_formed (simple, repeatable quality checks)
(Milestones 2 and 5, the LoRA training and merging, live in app/lora.py.)
"""
import json
import random
from pathlib import Path

EOS = "<|endoftext|>"

# ---------------------------------------------------------------- Milestone 1
SPORTS = {
    "Football": {
        "teams": ["Red Lions", "Blue Hawks", "City Rovers", "United Stars", "Northside FC", "Harbor Town"],
        "players": ["Rivera", "Okafor", "Silva", "Tanaka", "Muller", "Haddad", "Costa", "Novak"],
        "clock": lambda r: f"{r.randint(1, 90)}'",
        "events": {
            "Goal": [
                "GOAL! {player} finds the net for {team} at {minute}, and it's {score}!",
                "{player} smashes it home! {team} erupt as the score becomes {score}.",
                "What a strike from {player}! {team} make it count, {score} at {minute}.",
                "It's in! {player} buries the chance for {team}, {score} on the board.",
            ],
            "Yellow Card": [
                "{player} goes into the book for {team} at {minute}. He'll need to be careful now.",
                "Yellow card for {player}! The referee has seen enough of that challenge.",
                "A cynical foul from {player} and the card is shown. {team} must stay disciplined.",
            ],
            "Red Card": [
                "RED CARD! {player} is sent off and {team} are down to ten at {minute}!",
                "Disaster for {team}! {player} sees red and walks, with the score at {score}.",
                "The referee points to the tunnel. {player} is gone, and {team} face a long second half.",
            ],
            "Penalty Saved": [
                "SAVED! The keeper stops the penalty from {player} and {team} are denied at {minute}!",
                "{player} steps up, strikes it, and the keeper guesses right! {team} stay at {score}.",
                "What a stop from the goalkeeper! {player} cannot believe it.",
            ],
            "Save": [
                "What a stop! The keeper denies {player} and {team} are left frustrated at {score}.",
                "{player} shoots, the keeper dives, and it's palmed away at {minute}!",
                "Brilliant reflexes to keep out {player}'s effort. {team} are still at {score}.",
            ],
        },
    },
    "Basketball": {
        "teams": ["Thunder Cats", "Metro Kings", "Bay Wolves", "Sky Riders", "Iron Bulls", "Desert Suns"],
        "players": ["Johnson", "Carter", "Reyes", "Williams", "Dubois", "Park", "Mensah", "Ivanov"],
        "clock": lambda r: f"Q{r.randint(1, 4)} {r.randint(0, 11)}:{r.randint(0, 59):02d}",
        "events": {
            "Three Pointer": [
                "{player} from downtown... BANG! {team} hit the three, and it's {score} at {minute}!",
                "Nothing but net! {player} drains the triple for {team}, {score} now.",
                "{player} pulls up and sinks it from deep! {team} fans are on their feet.",
            ],
            "Dunk": [
                "{player} throws it down! A monster dunk for {team} and it's {score}!",
                "Rim-rattling slam from {player}! The crowd goes wild at {minute}.",
                "{player} soars in and posterizes the defender. {team} lead the highlight reel.",
            ],
            "Foul": [
                "Whistle blows! {player} is called for the foul at {minute} and {team} will be in trouble.",
                "A hard foul by {player}, and the referee stops play with {team} at {score}.",
                "{player} reaches in and gets whistled. {team} need to tighten up.",
            ],
            "Block": [
                "REJECTED! {player} swats it away for {team} at {minute}!",
                "What a block from {player}! {team} turn defense into offense, {score}.",
                "{player} sends that shot into the stands. Huge defensive play for {team}.",
            ],
        },
    },
    "Cricket": {
        "teams": ["Mumbai Strikers", "Sydney Sharks", "London Lords", "Cape Titans", "Lahore Falcons", "Chennai Kings"],
        "players": ["Sharma", "Smith", "Root", "De Villiers", "Babar", "Kohli", "Stokes", "Warner"],
        "clock": lambda r: f"Over {r.randint(1, 49)}.{r.randint(1, 6)}",
        "events": {
            "Six": [
                "SIX! {player} launches it into the stands for {team} at {minute}!",
                "{player} dances down the track and sends it sailing over the ropes! {score}.",
                "Massive hit from {player}! That one is gone, and {team} are flying at {score}.",
            ],
            "Four": [
                "FOUR! {player} finds the gap and the ball races away to the boundary for {team}.",
                "Beautiful timing from {player}, and it's four more at {minute}.",
                "{player} caresses it through the covers. {team} move to {score}.",
            ],
            "Wicket": [
                "OUT! {player} is gone for {team} at {minute}, and the bowlers celebrate!",
                "Bowled him! {player} has to walk back, and {team} are in trouble at {score}.",
                "A huge breakthrough! {player} falls and the match swings at {minute}.",
            ],
            "Fifty": [
                "FIFTY for {player}! The bat is raised and {team} fans applaud at {minute}.",
                "A well-made half century from {player}. {team} are building nicely at {score}.",
                "{player} reaches 50 with a calm single. What an innings for {team}!",
            ],
        },
    },
    "Tennis": {
        "teams": ["Alvarez", "Novak", "Petrov", "Kim", "Moreau", "Santos", "Becker", "Ito"],
        "players": None,  # in tennis the player is the "team"
        "clock": lambda r: f"Set {r.randint(1, 3)}",
        "events": {
            "Ace": [
                "ACE! {player} blasts one down the middle and the returner never moves at {minute}.",
                "Untouchable serve from {player}! The score moves to {score}.",
                "{player} fires an ace right on the line. Pure power.",
            ],
            "Break Point": [
                "Break point for {player}! The crowd holds its breath at {minute}.",
                "{player} has a chance to break here, with the score at {score}. Huge moment.",
                "Pressure is building, and {player} sniffs a break point.",
            ],
            "Double Fault": [
                "Double fault from {player}! A costly mistake at {minute}.",
                "{player} sends the second serve long, and the score slips to {score}.",
                "Nerves show as {player} double faults. The opponent smiles.",
            ],
            "Set Won": [
                "{player} takes the set! A superb performance at {minute}, score {score}.",
                "That's the set to {player}, who closes it out with confidence.",
                "{player} wins it, and the crowd rises. What a battle that was!",
            ],
        },
    },
    "Hockey": {
        "teams": ["Ice Wolves", "Polar Bears", "Frost Giants", "Steel Blades", "North Stars", "Glacier City"],
        "players": ["Lindqvist", "Brown", "Petrov", "Nakamura", "Tremblay", "Koskinen", "Murphy", "Dahl"],
        "clock": lambda r: f"P{r.randint(1, 3)} {r.randint(0, 19)}:{r.randint(0, 59):02d}",
        "events": {
            "Goal": [
                "HE SCORES! {player} lights the lamp for {team} at {minute}, and it's {score}!",
                "{player} snaps it past the goalie! {team} take it to {score}.",
                "Top shelf from {player}! The arena erupts for {team}.",
            ],
            "Penalty": [
                "Two minutes in the box for {player}. {team} will have to kill this off at {minute}.",
                "{player} heads to the sin bin and {team} go shorthanded with the score at {score}.",
                "The referee sends {player} off for a trip. {team} are in trouble.",
            ],
            "Save": [
                "What a save! The goalie robs {player} at {minute} and {team} stay at {score}.",
                "{player} rips a shot, and the goalie gloves it away! Incredible.",
                "Point-blank stop on {player}! {team} can't catch a break.",
            ],
        },
    },
}


def _make_row(rng: random.Random) -> dict:
    sport = rng.choice(list(SPORTS))
    spec = SPORTS[sport]
    etype = rng.choice(list(spec["events"]))
    team, opp = rng.sample(spec["teams"], 2)
    player = team if spec["players"] is None else rng.choice(spec["players"])
    a, b = rng.randint(0, 6), rng.randint(0, 6)
    score = f"{team} {a}-{b} {opp}"
    minute = spec["clock"](rng)
    who = player if player == team else f"{player} ({team})"
    event = f"{sport} | {minute} | {etype} | {who} | {score}"
    commentary = rng.choice(spec["events"][etype]).format(player=player, team=team, minute=minute, score=score)
    keywords = [player] if player == team else [player, team]
    return {"event": event, "commentary": commentary, "keywords": keywords}


def generate_dataset(n: int = 300, seed: int = 42) -> list[dict]:
    """Make n unique {event, commentary, keywords} rows. Same seed -> same rows."""
    rng = random.Random(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        row = _make_row(rng)
        key = (row["event"], row["commentary"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def format_prompt(event: str) -> str:
    """What the model sees at inference time. Training text starts with exactly this."""
    return f"Event: {event.strip()}\nCommentary:"


def format_example(event: str, commentary: str) -> str:
    """One full training string: prompt + commentary + end-of-text, so the model learns to stop."""
    return f"{format_prompt(event)} {commentary.strip()}{EOS}"


def save_jsonl(rows: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_dataset(rows: list[dict], test_frac: float = 0.15, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Shuffle (seeded) and split into (train, held_out). Never overlap."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac))
    return rows[n_test:], rows[:n_test]


# ---------------------------------------------------------------- Milestone 3
def clean_commentary(text: str) -> str:
    """Raw model continuation -> one tidy line (cut at EOS, newline, or a new 'Event:' block)."""
    text = text.split(EOS)[0]
    text = text.split("Event:")[0]
    return text.strip().split("\n")[0].strip()


# ---------------------------------------------------------------- Milestone 4
def relevance_score(keywords: list[str], text: str) -> float:
    """Fraction of key facts (player, team...) the commentary actually mentions. 0.0 to 1.0."""
    if not keywords:
        return 0.0
    low = text.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def is_well_formed(text: str, max_words: int = 40) -> bool:
    """Non-empty, a single line, reasonably short, and ends like a sentence."""
    words = text.split()
    return bool(words) and "\n" not in text and len(words) <= max_words and text.rstrip()[-1] in ".!?"
