import re

PERCENTAGE = r'(\d+(\.\d+)?)%'

CONF_PATTERNS = [
    r'confidence\((\d+(\.\d+)?)%?\)',  # e.g. "Confidence: confidence(90%)
    r'confidence:\s(\d+(\.\d+)?)%?'  # e.g. "Confidence: 90%"
]


def extract_confidence(raw: str, confidence_patterns: list[str] = None) -> float | None:
    if raw is None:
        return None

    confidence_patterns = confidence_patterns or CONF_PATTERNS

    confidence = []
    for confidence_pattern in confidence_patterns:
        confidence = re.findall(confidence_pattern, raw, re.IGNORECASE)
        if confidence:
            break

    if len(confidence) == 0:
        return None

    if len(confidence) > 1:
        print("Found multiple confidence expressions, taking the last one")
        print(raw)

    return float(confidence[-1][0])


if __name__ == '__main__':
    print(extract_confidence("confidence: 100%"))
    print(extract_confidence("confidence: 85.6%"))
    print(extract_confidence("confidence: confidence(100)"))
    print(
        extract_confidence(
            "Understood. Let's update the statistics for the 2010-11 season in the SeasonStatistics table.\n\nAction: Operation\n```sql\nUPDATE SeasonStatistics\nSET Played = 7, Drawn = 1, Lost = 2, Against = 82\nWHERE Season = '2010-11';\n```confidence: 100%"
        )
    )
    print(extract_confidence(
        """Think: I am confident in my solution.

Act: answer(100%)""",
        [r'(\d+(\.\d+)?)%?']
    ))

    print(extract_confidence(
            """45.6""",
            [r'(\d+(\.\d+)?)%?']
        ))
