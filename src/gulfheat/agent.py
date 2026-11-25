"""A small LLM agent that audits a finished run and writes reports/REPORT.md.

The agent gets read-only tools over this project's outputs (data checks,
result tables, metrics) and one tool to write the report. It cannot run
training or change data, so the worst a bad answer can do is a bad report,
which a person reviews anyway. Every number in the report has to come from a
tool result; the system prompt says so and the tools return exact values.

It uses the Anthropic SDK's tool runner, which handles the call-tool,
return-result loop. The loop runs locally because the tools read local files;
only tool outputs (small tables and JSON) are sent to the model.

Needs an API key: ``ANTHROPIC_API_KEY`` or ``ant auth login``.
"""

from __future__ import annotations

import json

from .config import PATHS

SYSTEM = """You review a finished machine learning experiment and write its results report.

The project corrects ERA5 hourly 2 m air temperature at weather stations across the Arabian
Peninsula, southern Iran and southern Iraq, with results also reported for a Gulf coast box.
Models predict ERA5's error at a site from terrain, land cover, sea and weather features.
Training uses 2023 and 2024, every score is on 2025 (metrics.json lists the years). An earlier
test, trained on 2023 and scored on 2024, is kept in reports/test_2024 and was studied to find
problems, so 2024 is training data now. "seen" means the test station was in training; "unseen"
means it was not. "lightgbm_hourly" uses only features that change hour by hour;
"lightgbm_cell", "lightgbm_sea" and "lightgbm" add the ERA5 cell features (height above the ERA5
cell, held to "higher can only be colder", and the share of the ERA5 value that comes from sea grid
points), then static sea, then land cover features. "lightgbm_first_design" is the first design's
map model (absolute height, no constraint), kept only for comparison. "mlp" uses all features.
metrics.json names the map's model, chosen by error on held-out stations within the training years
(table cv_train), never by test scores. Table map_by_height checks the map against height contours.

How to work:
1. Run the data checks first, then read the metrics and the tables you need.
2. Use only numbers that appear in tool results. Never estimate or round in a way that changes meaning.
   If something you want to say is not supported by a tool result, leave it out.
3. Look for problems as hard as for wins: stations where the correction made things worse,
   hours or countries where gains vanish, data warnings, anything that could leak the test year into training.
4. Write the report with write_report, in plain English for an engineer who has not seen the project:
   headings Summary, Data, Results, Where it does not work, What to try next. At most 700 words.
   Tables are fine. Say "about" only when the tool result itself is approximate.
"""

TABLES = ["summary", "cv_train", "by_hour", "by_hour_gulf", "by_country", "by_season", "by_coast",
          "by_region", "by_station", "ablation", "importance", "tuning", "map_by_height"]


def _tools():
    from anthropic import beta_tool

    @beta_tool
    def run_data_checks() -> str:
        """Run the data checks on the modelling table and return the results as JSON.

        Includes row and station counts, residual statistics, the share of whole-degree
        readings, missing feature values, per-station summaries and warnings.
        """
        from .checks import run_checks
        res = run_checks()
        return json.dumps(res, default=str)

    @beta_tool
    def read_metrics() -> str:
        """Return reports/metrics.json: tuned LightGBM settings, headline scores,
        ablation results, top features and stations that got worse after correction."""
        path = PATHS.reports / "metrics.json"
        return path.read_text() if path.exists() else "metrics.json not found: run `gulfheat evaluate` first"

    @beta_tool
    def read_table(name: str) -> str:
        """Return one result table as CSV.

        Args:
            name: one of summary, cv_train, by_hour, by_hour_gulf, by_country, by_season,
                by_coast, by_region, by_station, ablation, importance, tuning, map_by_height.
        """
        if name not in TABLES:
            return f"unknown table {name!r}; choose from {', '.join(TABLES)}"
        path = PATHS.reports / "tables" / f"{name}.csv"
        return path.read_text() if path.exists() else f"{name}.csv not found"

    @beta_tool
    def write_report(markdown: str) -> str:
        """Write the finished report to reports/REPORT.md, replacing any earlier version.

        Args:
            markdown: the full report in Markdown.
        """
        path = PATHS.reports / "REPORT.md"
        path.write_text(markdown.strip() + "\n")
        return f"written {len(markdown)} characters to {path.relative_to(PATHS.root)}"

    return [run_data_checks, read_metrics, read_table, write_report]


def run_agent(model: str = "claude-opus-5-5", max_turns: int = 12) -> None:
    import anthropic

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        max_iterations=max_turns,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        # If a request is declined, the API retries it on a fallback model it picks.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        tools=_tools(),
        messages=[{"role": "user", "content": "Audit the latest run and write the results report."}],
    )
    last = None
    for message in runner:
        last = message
        for block in message.content:
            if block.type == "tool_use":
                print(f"  agent called {block.name}")
    if last is not None and last.stop_reason == "refusal":
        print("The model declined to write the report.")
        return
    report = PATHS.reports / "REPORT.md"
    print(f"report: {report}" if report.exists() else "the agent finished without writing a report")
