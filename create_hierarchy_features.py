
import os
import csv
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# ---------- Sorting helpers (robust: no int-vs-str compare errors) ----------

def partwise_key(section: str) -> Tuple[Tuple[int, object], ...]:
    """
    For sorting section strings like '10.3.12.2'.
    Each part becomes (0, int) if numeric else (1, str), so comparisons are always consistent.
    """
    key: List[Tuple[int, object]] = []
    for p in (section or "").split("."):
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p))
    return tuple(key)

def topic_sort_key(topic: str) -> Tuple[int, object]:
    """
    Sort topics numerically if digits, else lexically; returns a tagged tuple to avoid type errors.
    """
    return (0, int(topic)) if topic.isdigit() else (1, topic)


# ---------- Core logic ----------

def add_to_topic(buckets: Dict[str, Dict[str, Set[str]]], section: str) -> None:
    """
    Classify a section string into Feature/Sub-Feature/Component and store it under its Topic.
    - Topic is the first dot-separated part.
    - Depth 1: Topic only (we don't add it into Feature/Sub-Feature/Component).
    - Depth 2: Feature
    - Depth 3: Sub-Feature
    - Depth >=4: Component
    """
    if not section:
        return
    parts = [p for p in section.split(".") if p]  # ignore empty segments
    if not parts:
        return

    topic = parts[0]  # e.g., "10"
    store = buckets[topic]  # defaultdict ensures existence

    d = len(parts)
    if d == 1:
        # '10' → Topic only; nothing to add to other columns
        return
    elif d == 2:
        store["Feature"].add(f"{parts[0]}.{parts[1]}")
    elif d == 3:
        store["Sub-Feature"].add(f"{parts[0]}.{parts[1]}.{parts[2]}")
    else:  # d >= 4
        store["Component"].add(".".join(parts))


def parse_section_list(raw: str) -> List[str]:
    """
    Split a comma-separated 'Section Numbers' cell into a clean list of section strings.
    """
    if not raw:
        return []
    items = []
    for s in raw.split(","):
        s = s.strip().strip('"').strip("'")
        if s:
            items.append(s)
    return items


def build_topic_rows(input_csv: str, output_csv: str) -> None:
    """
    Read input CSV and produce one row per Topic:
    Columns: Group ID, Topic, Feature, Sub-Feature, Component.
    - Consumes Parent Section and Section Numbers (if present).
    - Deduplicates within each topic bucket; naturally sorts.
    - Output does NOT carry over other input columns.
    """
    # topic -> { "Feature": set(), "Sub-Feature": set(), "Component": set() }
    topics: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {"Feature": set(), "Sub-Feature": set(), "Component": set()}
    )

    # Read input
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Find "Section Numbers" column (handle variations like 'Section Numbers...' if any)
        sec_col = None
        for name in (reader.fieldnames or []):
            if name and name.strip().startswith("Section Numbers"):
                sec_col = name
                break

        for row in reader:
            # Parent Section is a single section string (may be empty)
            parent = (row.get("Parent Section", "") or "").strip()
            if parent:
                add_to_topic(topics, parent)

            # Section Numbers is a comma-separated list of sections (if present)
            if sec_col:
                raw_list = (row.get(sec_col, "") or "").strip()
                for sec in parse_section_list(raw_list):
                    add_to_topic(topics, sec)

            # Also: if the row's Parent Section is itself just the Topic like "10",
            # ensure the topic key exists (even if no Feature/Sub-Feature/Component yet)
            if parent:
                topic_key = parent.split(".")[0]
                _ = topics[topic_key]  # touch to ensure existence

    # Prepare output rows (one per topic), sorted safely
    sorted_topics = sorted(topics.keys(), key=topic_sort_key)

    fieldnames = ["Group ID", "Topic", "Feature", "Sub-Feature", "Component"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for topic in sorted_topics:
            feats = sorted(topics[topic]["Feature"], key=partwise_key)
            subs  = sorted(topics[topic]["Sub-Feature"], key=partwise_key)
            comps = sorted(topics[topic]["Component"], key=partwise_key)

            writer.writerow({
                "Group ID": f"Topic{topic}",         # stable one-row-per-topic ID
                "Topic": topic,                       # the topic number itself
                "Feature": ",".join(feats),           # comma-separated, deduped, sorted
                "Sub-Feature": ",".join(subs),
                "Component": ",".join(comps),
            })


# ---------- Run (change filenames as needed) ----------

if __name__ == "__main__":
    # Example usage
    input_file = os.environ.get("FEATURE_SUBFEATURE_GROUP")
    tree_topics_file = os.environ.get("TREE_TOPICS_FEATURES_SUBFEATURES")

    build_topic_rows(input_file, tree_topics_file) 


