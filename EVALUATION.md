# Evaluation on 30 held-out events

| Model | Mean relevance (player/team mentioned) | Well-formed |
|---|---|---|
| Base GPT-2 | 15% | 20% |
| LoRA (merged) | 88% | 100% |

## Examples

**Event:** Tennis | Set 1 | Double Fault | Ito | Ito 0-4 Kim

- Base GPT-2: Kim's first match was against a player who had been eliminated from the tournament. Kim was eliminated by a player who had been eliminated from the tournament.
- LoRA (merged): It's set for Ito.

**Event:** Cricket | Over 48.2 | Fifty | De Villiers (Cape Titans) | Cape Titans 4-3 Chennai Kings

- Base GPT-2: Cape Titans:
- LoRA (merged): De Villiers is a menace in the final 30 seconds.

**Event:** Cricket | Over 25.1 | Wicket | Kohli (Sydney Sharks) | Sydney Sharks 0-0 Chennai Kings

- Base GPT-2: The match was a bit of a surprise to me as I had never seen a match like this before. The first half was a bit of a mess as the match was played in a very
- LoRA (merged): Kohli is right on the mark and Sydney Sharks are rewarded.

**Event:** Tennis | Set 2 | Double Fault | Moreau | Moreau 5-5 Novak

- Base GPT-2: The game is a bit of a mess, but it's still a fun game. The game is a bit of a mess, but it's still a fun game.
- LoRA (merged): Moreau is outstretched and the ball is outstretched. Moreau is outstretched again.

**Event:** Cricket | Over 32.5 | Wicket | Sharma (Sydney Sharks) | Sydney Sharks 5-1 Mumbai Strikers

- Base GPT-2: The Sydney Sharks have been a very good team in the past few years. They have been a very good team in the past few years. They have been a very good team in the past
- LoRA (merged): Sharma is a menace in the middle and Sydney Sharks captain Shane Watson is ready to take it to Sydney Sharks 5-1 Mumbai Strikers.
