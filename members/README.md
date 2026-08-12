# Member workspaces

Each participant owns `members/<github-name>/` and may choose their own package,
notebooks, dependencies, and model stack. This prevents routine research edits
from colliding with another participant's files.

Member code should:

1. read immutable inputs from the repository-level `data/raw/`;
2. write features, models, and submissions only inside its own workspace;
3. emit the four standard prediction CSVs described in `../validation/README.md`;
4. commit small experiment configs and shared leaderboard result JSON, not large
   model or prediction artifacts.

Cross-member reusable code should move into a deliberately shared package only
after the interface is stable. Until then, isolation is cheaper than premature
coordination.
