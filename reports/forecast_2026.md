# World Cup 2026 — My Pre-Tournament Forecast

**Date frozen: June 12, 2026.** Training data runs through June 8, 2026. The
tournament opened on June 11 with Mexico vs South Africa; that result is not
in the training data and I haven't updated anything based on it. 103 of the
104 matches are still to be played, so this stands as a genuine
before-the-fact forecast. The commit timestamp on this file is the proof.

## How these numbers were made

I fit a Dixon-Coles bivariate Poisson model on 15,790 international matches
from 2010 onwards (data: Mart Jurisoo's international results dataset). Each
team gets an attack and a defense rating, there's a home-advantage term, and
a time-decay weight so recent matches count more than old ones. Then I
simulated the full 48-team tournament 5,000 times: every group with the real
tiebreaker rules, the eight best third-placed teams, the official Round of 32
bracket, extra time and penalties in the knockouts, and home advantage for
the three host nations.

## Title odds (top 15)

| Team        | R32  | R16  | QF   | SF   | Final | Champion |
|-------------|------|------|------|------|-------|----------|
| Argentina   | 97.6 | 69.1 | 53.0 | 37.9 | 26.2  | **17.3** |
| Spain       | 99.4 | 63.2 | 43.9 | 31.0 | 19.9  | **11.9** |
| England     | 97.7 | 63.8 | 39.8 | 22.2 | 12.2  | 6.8      |
| Brazil      | 96.9 | 58.3 | 36.1 | 22.0 | 11.4  | 5.9      |
| Portugal    | 89.9 | 62.9 | 37.7 | 20.6 | 11.6  | 5.6      |
| Morocco     | 94.3 | 54.4 | 33.3 | 19.4 | 10.2  | 5.3      |
| France      | 90.6 | 61.2 | 34.9 | 20.4 | 10.2  | 5.2      |
| Germany     | 98.1 | 64.4 | 36.1 | 20.0 | 9.5   | 4.6      |
| Colombia    | 88.6 | 60.2 | 31.9 | 16.9 | 8.8   | 4.4      |
| Netherlands | 91.8 | 48.8 | 30.6 | 16.6 | 8.2   | 4.2      |
| Japan       | 91.0 | 48.5 | 29.6 | 16.4 | 8.2   | 4.1      |
| Belgium     | 95.0 | 63.7 | 36.6 | 17.0 | 8.6   | 3.7      |
| Ecuador     | 94.2 | 54.4 | 27.8 | 14.4 | 6.6   | 3.2      |
| Norway      | 88.9 | 55.6 | 29.3 | 14.6 | 6.4   | 3.0      |
| Switzerland | 98.0 | 63.4 | 29.9 | 13.6 | 6.3   | 2.9      |

All values are percentages: the share of 5,000 simulated tournaments in
which the team reached that stage. Full table for all 48 teams:
`simulation_2026.csv` in this folder.

## What stands out to me

**Argentina are the model's clear favorite at 17.3%.** Two things drive
this. Their fitted defense rating is the best in the dataset by a wide
margin, and a strong defense compounds across seven knockout-style rounds —
you just keep not losing. On top of that they drew one of the softest
groups (Algeria, Austria, Jordan), which is why their Round of 32
probability is near certain.

**The model disagrees with the betting market on France.** Bookmakers have
France as the favorite; my model puts them 7th at 5.2%. Part of this is
real: France landed in the hardest group (Senegal and Norway both rate
strongly here). Part of it is a known blind spot: the model only sees match
results, so it has no idea about squad depth or individual talent. I'm
deliberately not tuning the model to match the market — recording the
disagreement now and checking who was closer afterwards is more useful than
agreeing by construction.

**Hosts get a measurable bump.** Mexico, the USA and Canada all advance
from their groups more often than their raw ratings alone would suggest,
because the model applies its fitted home-advantage term (about a 24% boost
to scoring rate) to every host match.

## What this forecast can't do

Football is high-variance and a single tournament is a tiny sample. A 17%
favorite loses the title 83% of the time. The model also knows nothing
about injuries, lineups, or current squad quality beyond what shows up in
results. The right way to judge this forecast is calibration over many
predictions — when I say 30%, it should happen about 30% of the time — and
I'll evaluate exactly that against the real results after the final on
July 19.
