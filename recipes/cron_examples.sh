# 1. Run immediate showdown with local MLflow tracking via uv
uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml

# 2. Run dry-run (offline, no MLflow database overhead)
# uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml --dry-run

# 3. Launch the MLflow UI (using uv run so mlflow is available in your shell)
# uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

# -----------------------------------------------------------------------------
# AUTOMATION EXAMPLES (CRON & AT):
#
# A. One-shot schedule for "Tomorrow at 3:00 AM" (using standard Linux `at`):
# echo "cd /home/federico/Documents/Projects/truco-bot && uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml >> /tmp/truco_showdown.log 2>&1" | at 3:00 AM tomorrow
#
# B. One-shot schedule for a specific date (e.g. 03:00 AM on October 25):
# echo "cd /home/federico/Documents/Projects/truco-bot && uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml >> /tmp/truco_showdown.log 2>&1" | at 03:00 2026-10-25
#
# C. Crontab for recurring nightly runs at 03:00 AM:
# 0 3 * * * cd /home/federico/Documents/Projects/truco-bot && uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml >> /tmp/truco_showdown.log 2>&1
#
# D. Crontab for a fixed specific date (e.g. Oct 25 at 3:00 AM):
# 0 3 25 10 * cd /home/federico/Documents/Projects/truco-bot && uv run python -m truco_bot.eval.runner --config experiments/showdown_example.yaml >> /tmp/truco_showdown.log 2>&1
# -----------------------------------------------------------------------------
