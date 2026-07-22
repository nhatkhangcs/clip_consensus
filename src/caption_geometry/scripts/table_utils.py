from __future__ import annotations


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]
    header_line = "| " + " | ".join(_pad(value, width) for value, width in zip(headers, widths)) + " |"
    rule_line = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = [
        "| " + " | ".join(_pad(value, width) for value, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, rule_line, *row_lines])


def metric_value(metrics: dict[str, dict[str, float]], direction: str, key: str = "R@1") -> float:
    return float(metrics[direction][key])


def format_metric(value: float) -> str:
    return f"{value:.2f}"


def format_ci(ci: dict[str, float]) -> str:
    return f"{ci['estimate']:.2f} [{ci['low']:.2f}, {ci['high']:.2f}]"


def _pad(value: str, width: int) -> str:
    return value + " " * (width - len(value))

