#!/usr/bin/env python3

import argparse
import json
import sys
import xml.parsers.expat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TextSample:
    path: str
    text: str


@dataclass
class ElementStat:
    name: str
    count: int
    max_depth: int
    text_nodes: int
    attribute_keys: List[str]
    attribute_value_samples: Dict[str, List[str]]


@dataclass
class PathStat:
    path: str
    count: int


@dataclass
class DocumentInfo:
    source: str
    bytes_read: int
    root_element: Optional[str]
    encoding: Optional[str]
    standalone: Optional[int]


@dataclass
class SummaryStats:
    total_elements: int
    unique_elements: int
    total_attributes: int
    unique_paths: int
    max_depth: int


@dataclass
class DerivedStats:
    avg_attributes_per_element: float
    avg_paths_per_element: float
    captured_text_samples: int
    note_count: int


@dataclass
class ParseReport:
    document: DocumentInfo
    summary: SummaryStats
    derived: DerivedStats
    element_stats: List[ElementStat]
    path_stats: List[PathStat]
    text_samples: List[TextSample]
    notes: List[str] = field(default_factory=list)


class XMLStreamAnalyzer:
    def __init__(
        self,
        max_text_samples: int = 50,
        max_attribute_samples: int = 5,
        max_top_paths: int = 100,
        text_min_length: int = 1,
        strip_text: bool = True,
    ) -> None:
        self.max_text_samples = max_text_samples
        self.max_attribute_samples = max_attribute_samples
        self.max_top_paths = max_top_paths
        self.text_min_length = text_min_length
        self.strip_text = strip_text
        self._reset()

    def _reset(self) -> None:
        self.source = ""
        self.bytes_read = 0
        self.total_elements = 0
        self.total_attributes = 0
        self.max_depth = 0
        self.root_element: Optional[str] = None
        self.encoding: Optional[str] = None
        self.standalone: Optional[int] = None
        self.path_stack: List[str] = []
        self.text_buffer_stack: List[List[str]] = []
        self.element_counts: Counter[str] = Counter()
        self.path_counts: Counter[str] = Counter()
        self.element_max_depth: Dict[str, int] = defaultdict(int)
        self.element_text_nodes: Counter[str] = Counter()
        self.attribute_keys: Dict[str, set[str]] = defaultdict(set)
        self.attribute_value_samples: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.text_samples: List[TextSample] = []
        self.notes: List[str] = []

    def _current_path(self) -> str:
        if not self.path_stack:
            return "/"
        return "/" + "/".join(self.path_stack)

    def _record_text_sample(self, path: str, text: str) -> None:
        if len(self.text_samples) >= self.max_text_samples:
            return
        self.text_samples.append(TextSample(path=path, text=text))

    def _on_xml_decl(self, version: str, encoding: str, standalone: int) -> None:
        if encoding:
            self.encoding = encoding
        self.standalone = standalone

    def _on_start(self, name: str, attrs: Dict[str, str]) -> None:
        self.path_stack.append(name)
        self.text_buffer_stack.append([])
        depth = len(self.path_stack)
        path = self._current_path()

        if self.root_element is None:
            self.root_element = name

        self.total_elements += 1
        self.element_counts[name] += 1
        self.path_counts[path] += 1
        self.total_attributes += len(attrs)

        if depth > self.max_depth:
            self.max_depth = depth

        if depth > self.element_max_depth[name]:
            self.element_max_depth[name] = depth

        for key, value in attrs.items():
            self.attribute_keys[name].add(key)
            samples = self.attribute_value_samples[name][key]
            if value not in samples and len(samples) < self.max_attribute_samples:
                samples.append(value)

    def _on_end(self, name: str) -> None:
        if not self.path_stack or not self.text_buffer_stack:
            self.notes.append(f"parser stack mismatch near closing element {name}")
            return

        path = self._current_path()
        pieces = self.text_buffer_stack.pop()
        self.path_stack.pop()

        text = "".join(pieces)
        if self.strip_text:
            text = text.strip()

        if text and len(text) >= self.text_min_length:
            self.element_text_nodes[name] += 1
            self._record_text_sample(path, text)

    def _on_char_data(self, data: str) -> None:
        if not self.text_buffer_stack:
            return
        self.text_buffer_stack[-1].append(data)

    def _build_parser(self) -> xml.parsers.expat.XMLParserType:
        parser = xml.parsers.expat.ParserCreate()
        parser.XmlDeclHandler = self._on_xml_decl
        parser.StartElementHandler = self._on_start
        parser.EndElementHandler = self._on_end
        parser.CharacterDataHandler = self._on_char_data
        return parser

    def analyze_file(self, path: str, chunk_size: int = 65536) -> ParseReport:
        self._reset()
        self.source = path
        parser = self._build_parser()

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(path)

        with open(target, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.bytes_read += len(chunk)
                parser.Parse(chunk, False)
            parser.Parse(b"", True)

        return self._build_report()

    def analyze_string(self, xml_text: str, source: str = "<memory>") -> ParseReport:
        self._reset()
        self.source = source
        parser = self._build_parser()
        data = xml_text.encode("utf-8")
        self.bytes_read = len(data)
        parser.Parse(data, True)
        return self._build_report()

    def _build_report(self) -> ParseReport:
        element_stats: List[ElementStat] = []
        for name, count in self.element_counts.most_common():
            attr_keys = sorted(self.attribute_keys.get(name, set()))
            samples: Dict[str, List[str]] = {}
            for attr_name in sorted(self.attribute_value_samples.get(name, {})):
                samples[attr_name] = list(self.attribute_value_samples[name][attr_name])

            element_stats.append(
                ElementStat(
                    name=name,
                    count=count,
                    max_depth=self.element_max_depth.get(name, 0),
                    text_nodes=self.element_text_nodes.get(name, 0),
                    attribute_keys=attr_keys,
                    attribute_value_samples=samples,
                )
            )

        path_stats = [
            PathStat(path=path, count=count)
            for path, count in self.path_counts.most_common(self.max_top_paths)
        ]

        if self.total_elements == 0:
            self.notes.append("document contains no elements")

        if self.encoding is None:
            self.notes.append("xml declaration not present or encoding unspecified")

        avg_attributes_per_element = self.total_attributes / self.total_elements if self.total_elements else 0.0
        avg_paths_per_element = len(self.path_counts) / self.total_elements if self.total_elements else 0.0

        return ParseReport(
            document=DocumentInfo(
                source=self.source,
                bytes_read=self.bytes_read,
                root_element=self.root_element,
                encoding=self.encoding,
                standalone=self.standalone,
            ),
            summary=SummaryStats(
                total_elements=self.total_elements,
                unique_elements=len(self.element_counts),
                total_attributes=self.total_attributes,
                unique_paths=len(self.path_counts),
                max_depth=self.max_depth,
            ),
            derived=DerivedStats(
                avg_attributes_per_element=avg_attributes_per_element,
                avg_paths_per_element=avg_paths_per_element,
                captured_text_samples=len(self.text_samples),
                note_count=len(self.notes),
            ),
            element_stats=element_stats,
            path_stats=path_stats,
            text_samples=self.text_samples,
            notes=self.notes,
        )


def format_bytes(num: int) -> str:
    value = float(num)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def frame_top(width: int) -> str:
    return "┌" + "─" * (width - 2) + "┐"


def frame_mid(width: int) -> str:
    return "├" + "─" * (width - 2) + "┤"


def frame_bottom(width: int) -> str:
    return "└" + "─" * (width - 2) + "┘"


def frame_line(text: str, width: int) -> str:
    inner = width - 4
    if len(text) > inner:
        text = text[:inner]
    return f"│ {text.ljust(inner)} │"


def print_summary(report: ParseReport, width: int = 120) -> None:
    print(frame_top(width))
    print(frame_line(f"XML STREAM ANALYSIS  source={report.document.source}", width))
    print(frame_mid(width))
    print(frame_line(f"bytes_read={report.document.bytes_read} ({format_bytes(report.document.bytes_read)})", width))
    print(frame_line(f"root_element={report.document.root_element or 'N/A'}", width))
    print(
        frame_line(
            f"encoding={report.document.encoding or 'N/A'}  standalone={report.document.standalone if report.document.standalone is not None else 'N/A'}",
            width,
        )
    )
    print(
        frame_line(
            f"total_elements={report.summary.total_elements}  unique_elements={report.summary.unique_elements}",
            width,
        )
    )
    print(
        frame_line(
            f"total_attributes={report.summary.total_attributes}  unique_paths={report.summary.unique_paths}  max_depth={report.summary.max_depth}",
            width,
        )
    )
    print(
        frame_line(
            f"avg_attributes_per_element={report.derived.avg_attributes_per_element:.4f}  avg_paths_per_element={report.derived.avg_paths_per_element:.4f}",
            width,
        )
    )
    if report.notes:
        print(frame_mid(width))
        for note in report.notes:
            print(frame_line(f"note={note}", width))
    print(frame_bottom(width))


def print_element_stats(report: ParseReport, limit: int = 20, width: int = 120) -> None:
    print(frame_top(width))
    print(frame_line("ELEMENT STATS", width))
    print(frame_mid(width))
    header = f"{'ELEMENT':<24} {'COUNT':>10} {'MAX_DEPTH':>10} {'TEXT_NODES':>12} {'ATTR_KEYS':<50}"
    print(frame_line(header, width))
    print(frame_mid(width))
    for stat in report.element_stats[:limit]:
        attr_text = ",".join(stat.attribute_keys) if stat.attribute_keys else "-"
        row = f"{stat.name:<24} {stat.count:>10} {stat.max_depth:>10} {stat.text_nodes:>12} {attr_text:<50}"
        print(frame_line(row, width))
    print(frame_bottom(width))


def print_path_stats(report: ParseReport, limit: int = 20, width: int = 120) -> None:
    print(frame_top(width))
    print(frame_line("PATH STATS", width))
    print(frame_mid(width))
    header = f"{'COUNT':>10} {'PATH':<100}"
    print(frame_line(header, width))
    print(frame_mid(width))
    for stat in report.path_stats[:limit]:
        row = f"{stat.count:>10} {stat.path:<100}"
        print(frame_line(row, width))
    print(frame_bottom(width))


def print_text_samples(report: ParseReport, limit: int = 10, width: int = 120) -> None:
    print(frame_top(width))
    print(frame_line("TEXT SAMPLES", width))
    print(frame_mid(width))
    if not report.text_samples:
        print(frame_line("no text samples captured", width))
        print(frame_bottom(width))
        return

    for idx, sample in enumerate(report.text_samples[:limit]):
        print(frame_line(f"path={sample.path}", width))
        text = sample.text.replace("\n", "\\n").replace("\r", "\\r")
        print(frame_line(f"text={text}", width))
        if idx != min(limit, len(report.text_samples)) - 1:
            print(frame_mid(width))
    print(frame_bottom(width))


def report_to_json(report: ParseReport) -> str:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False)


def report_to_compact_json(report: ParseReport) -> str:
    return json.dumps(asdict(report), separators=(",", ":"), ensure_ascii=False)


def report_to_structured_text(report: ParseReport, element_limit: int, path_limit: int, sample_limit: int) -> str:
    lines: List[str] = []

    lines.append("[document]")
    lines.append(f"source={report.document.source}")
    lines.append(f"bytes_read={report.document.bytes_read}")
    lines.append(f"bytes_human={format_bytes(report.document.bytes_read)}")
    lines.append(f"root_element={report.document.root_element or 'N/A'}")
    lines.append(f"encoding={report.document.encoding or 'N/A'}")
    lines.append(f"standalone={report.document.standalone if report.document.standalone is not None else 'N/A'}")
    lines.append("")

    lines.append("[summary]")
    lines.append(f"total_elements={report.summary.total_elements}")
    lines.append(f"unique_elements={report.summary.unique_elements}")
    lines.append(f"total_attributes={report.summary.total_attributes}")
    lines.append(f"unique_paths={report.summary.unique_paths}")
    lines.append(f"max_depth={report.summary.max_depth}")
    lines.append("")

    lines.append("[derived]")
    lines.append(f"avg_attributes_per_element={report.derived.avg_attributes_per_element:.6f}")
    lines.append(f"avg_paths_per_element={report.derived.avg_paths_per_element:.6f}")
    lines.append(f"captured_text_samples={report.derived.captured_text_samples}")
    lines.append(f"note_count={report.derived.note_count}")
    lines.append("")

    lines.append("[elements]")
    if report.element_stats[:element_limit]:
        for stat in report.element_stats[:element_limit]:
            attr_keys = ",".join(stat.attribute_keys) if stat.attribute_keys else "-"
            lines.append(
                f"name={stat.name} count={stat.count} max_depth={stat.max_depth} text_nodes={stat.text_nodes} attr_keys={attr_keys}"
            )
    else:
        lines.append("none")
    lines.append("")

    lines.append("[paths]")
    if report.path_stats[:path_limit]:
        for stat in report.path_stats[:path_limit]:
            lines.append(f"count={stat.count} path={stat.path}")
    else:
        lines.append("none")
    lines.append("")

    lines.append("[text_samples]")
    if report.text_samples[:sample_limit]:
        for sample in report.text_samples[:sample_limit]:
            text = sample.text.replace("\n", "\\n").replace("\r", "\\r")
            lines.append(f"path={sample.path} text={text}")
    else:
        lines.append("none")
    lines.append("")

    lines.append("[notes]")
    if report.notes:
        for note in report.notes:
            lines.append(note)
    else:
        lines.append("none")

    return "\n".join(lines) + "\n"


def report_to_ndjson(report: ParseReport, element_limit: Optional[int] = None, path_limit: Optional[int] = None, sample_limit: Optional[int] = None) -> str:
    lines: List[str] = []

    lines.append(json.dumps({"type": "document", **asdict(report.document)}, ensure_ascii=False))
    lines.append(json.dumps({"type": "summary", **asdict(report.summary)}, ensure_ascii=False))
    lines.append(json.dumps({"type": "derived", **asdict(report.derived)}, ensure_ascii=False))

    for stat in report.element_stats[:element_limit] if element_limit is not None else report.element_stats:
        lines.append(json.dumps({"type": "element_stat", **asdict(stat)}, ensure_ascii=False))

    for stat in report.path_stats[:path_limit] if path_limit is not None else report.path_stats:
        lines.append(json.dumps({"type": "path_stat", **asdict(stat)}, ensure_ascii=False))

    for sample in report.text_samples[:sample_limit] if sample_limit is not None else report.text_samples:
        lines.append(json.dumps({"type": "text_sample", **asdict(sample)}, ensure_ascii=False))

    for note in report.notes:
        lines.append(json.dumps({"type": "note", "text": note}, ensure_ascii=False))

    return "\n".join(lines) + ("\n" if lines else "")


def report_to_text(report: ParseReport, element_limit: int, path_limit: int, sample_limit: int, width: int) -> str:
    parts: List[str] = []

    def capture(fn: Any) -> str:
        from io import StringIO

        old_stdout = sys.stdout
        buf = StringIO()
        try:
            sys.stdout = buf
            fn()
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    parts.append(capture(lambda: print_summary(report, width)))
    parts.append(capture(lambda: print_element_stats(report, element_limit, width)))
    parts.append(capture(lambda: print_path_stats(report, path_limit, width)))
    parts.append(capture(lambda: print_text_samples(report, sample_limit, width)))
    return "".join(parts)


def write_output(path: str, data: str) -> None:
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(data, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream-analyze XML structure using Python Expat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file.xml
  %(prog)s file.xml --json
  %(prog)s file.xml --compact-json
  %(prog)s file.xml --structured-text
  %(prog)s file.xml --ndjson
  %(prog)s file.xml --json --output report.json
  %(prog)s file.xml --elements 50 --paths 50 --samples 20
  %(prog)s --string "<root><item id='1'>alpha</item></root>"
        """.strip(),
    )
    parser.add_argument("source", nargs="?", help="Path to XML file")
    parser.add_argument("--string", dest="xml_string", help="Analyze XML string input directly")
    parser.add_argument("--json", action="store_true", help="Print formatted JSON")
    parser.add_argument("--compact-json", action="store_true", help="Print compact JSON")
    parser.add_argument("--structured-text", action="store_true", help="Print sectioned structured text")
    parser.add_argument("--ndjson", action="store_true", help="Print line-delimited JSON")
    parser.add_argument("--output", help="Write output to file")
    parser.add_argument("--elements", type=int, default=20, help="Max element stats to show in text mode")
    parser.add_argument("--paths", type=int, default=20, help="Max path stats to show in text mode")
    parser.add_argument("--samples", type=int, default=10, help="Max text samples to show in text mode")
    parser.add_argument("--max-text-samples", type=int, default=50, help="Max text samples to retain internally")
    parser.add_argument("--max-attribute-samples", type=int, default=5, help="Max attribute value samples per attribute")
    parser.add_argument("--max-top-paths", type=int, default=100, help="Max path stats retained internally")
    parser.add_argument("--text-min-length", type=int, default=1, help="Minimum text length to keep")
    parser.add_argument("--no-strip-text", action="store_true", help="Do not strip text nodes")
    parser.add_argument("--width", type=int, default=120, help="Frame width for text output")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Chunk size for file parsing")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.source and not args.xml_string:
        raise ValueError("either source path or --string must be provided")
    if args.source and args.xml_string:
        raise ValueError("use either source path or --string, not both")
    if args.elements < 1:
        raise ValueError("--elements must be positive")
    if args.paths < 1:
        raise ValueError("--paths must be positive")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.max_text_samples < 1:
        raise ValueError("--max-text-samples must be positive")
    if args.max_attribute_samples < 1:
        raise ValueError("--max-attribute-samples must be positive")
    if args.max_top_paths < 1:
        raise ValueError("--max-top-paths must be positive")
    if args.text_min_length < 0:
        raise ValueError("--text-min-length must be non-negative")
    if args.width < 40:
        raise ValueError("--width must be at least 40")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    selected_formats = sum(
        [
            1 if args.json else 0,
            1 if args.compact_json else 0,
            1 if args.structured_text else 0,
            1 if args.ndjson else 0,
        ]
    )
    if selected_formats > 1:
        raise ValueError("choose only one of --json, --compact-json, --structured-text, --ndjson")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        analyzer = XMLStreamAnalyzer(
            max_text_samples=args.max_text_samples,
            max_attribute_samples=args.max_attribute_samples,
            max_top_paths=args.max_top_paths,
            text_min_length=args.text_min_length,
            strip_text=not args.no_strip_text,
        )

        if args.xml_string is not None:
            report = analyzer.analyze_string(args.xml_string)
        else:
            report = analyzer.analyze_file(args.source, chunk_size=args.chunk_size)

        if args.compact_json:
            output = report_to_compact_json(report)
        elif args.json:
            output = report_to_json(report)
        elif args.structured_text:
            output = report_to_structured_text(
                report,
                element_limit=args.elements,
                path_limit=args.paths,
                sample_limit=args.samples,
            )
        elif args.ndjson:
            output = report_to_ndjson(
                report,
                element_limit=args.elements,
                path_limit=args.paths,
                sample_limit=args.samples,
            )
        else:
            output = report_to_text(
                report,
                element_limit=args.elements,
                path_limit=args.paths,
                sample_limit=args.samples,
                width=args.width,
            )

        if args.output:
            write_output(args.output, output)
        else:
            print(output, end="" if output.endswith("\n") else "\n")

        return 0

    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except xml.parsers.expat.ExpatError as exc:
        print(f"XML parse error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
