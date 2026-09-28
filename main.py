#!/usr/bin/env python3

import argparse
import json
import sys
import time
import xml.parsers.expat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO


class XMLScanError(Exception):
    pass


class ResourceLimitError(XMLScanError):
    pass


class ExternalEntityError(XMLScanError):
    pass


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
    namespace_aware: bool


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
    elapsed_seconds: float
    throughput_mib_per_second: float


@dataclass
class ParseReport:
    document: DocumentInfo
    summary: SummaryStats
    derived: DerivedStats
    element_stats: List[ElementStat]
    path_stats: List[PathStat]
    text_samples: List[TextSample]
    notes: List[str] = field(default_factory=list)


@dataclass
class AnalyzerLimits:
    max_bytes: Optional[int] = None
    max_elements: Optional[int] = None
    max_depth: Optional[int] = None
    max_text_bytes: Optional[int] = None
    max_unique_elements: Optional[int] = None
    max_unique_paths: Optional[int] = None


class XMLStreamAnalyzer:
    def __init__(
        self,
        max_text_samples: int = 50,
        max_attribute_samples: int = 5,
        max_top_paths: int = 100,
        text_min_length: int = 1,
        strip_text: bool = True,
        namespace_aware: bool = False,
        namespace_separator: str = "|",
        limits: Optional[AnalyzerLimits] = None,
    ) -> None:
        self.max_text_samples = max_text_samples
        self.max_attribute_samples = max_attribute_samples
        self.max_top_paths = max_top_paths
        self.text_min_length = text_min_length
        self.strip_text = strip_text
        self.namespace_aware = namespace_aware
        self.namespace_separator = namespace_separator
        self.limits = limits or AnalyzerLimits()
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
        self.text_buffer_sizes: List[int] = []
        self.element_counts: Counter[str] = Counter()
        self.path_counts: Counter[str] = Counter()
        self.element_max_depth: Dict[str, int] = defaultdict(int)
        self.element_text_nodes: Counter[str] = Counter()
        self.attribute_keys: Dict[str, set[str]] = defaultdict(set)
        self.attribute_value_samples: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.text_samples: List[TextSample] = []
        self.notes: List[str] = []
        self.started_at = 0.0
        self.finished_at = 0.0

    def _current_path(self) -> str:
        if not self.path_stack:
            return "/"
        return "/" + "/".join(self.path_stack)

    def _check_byte_limit(self) -> None:
        limit = self.limits.max_bytes
        if limit is not None and self.bytes_read > limit:
            raise ResourceLimitError(
                f"maximum input size exceeded: {self.bytes_read} > {limit} bytes"
            )

    def _check_element_limit(self) -> None:
        limit = self.limits.max_elements
        if limit is not None and self.total_elements > limit:
            raise ResourceLimitError(
                f"maximum element count exceeded: {self.total_elements} > {limit}"
            )

    def _check_depth_limit(self, depth: int) -> None:
        limit = self.limits.max_depth
        if limit is not None and depth > limit:
            raise ResourceLimitError(
                f"maximum XML depth exceeded: {depth} > {limit}"
            )

    def _check_unique_element_limit(self) -> None:
        limit = self.limits.max_unique_elements
        if limit is not None and len(self.element_counts) > limit:
            raise ResourceLimitError(
                f"maximum unique element count exceeded: "
                f"{len(self.element_counts)} > {limit}"
            )

    def _check_unique_path_limit(self) -> None:
        limit = self.limits.max_unique_paths
        if limit is not None and len(self.path_counts) > limit:
            raise ResourceLimitError(
                f"maximum unique path count exceeded: "
                f"{len(self.path_counts)} > {limit}"
            )

    def _record_text_sample(self, path: str, text: str) -> None:
        if len(self.text_samples) >= self.max_text_samples:
            return
        self.text_samples.append(TextSample(path=path, text=text))

    def _on_xml_decl(
        self,
        version: str,
        encoding: Optional[str],
        standalone: int,
    ) -> None:
        if encoding:
            self.encoding = encoding
        self.standalone = standalone

    def _on_start(self, name: str, attrs: Dict[str, str]) -> None:
        self.path_stack.append(name)
        self.text_buffer_stack.append([])
        self.text_buffer_sizes.append(0)

        depth = len(self.path_stack)
        path = self._current_path()

        self._check_depth_limit(depth)

        if self.root_element is None:
            self.root_element = name

        self.total_elements += 1
        self._check_element_limit()

        is_new_element = name not in self.element_counts
        self.element_counts[name] += 1

        if is_new_element:
            self._check_unique_element_limit()

        is_new_path = path not in self.path_counts
        self.path_counts[path] += 1

        if is_new_path:
            self._check_unique_path_limit()

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
            self.notes.append(
                f"parser stack mismatch near closing element {name}"
            )
            return

        path = self._current_path()
        pieces = self.text_buffer_stack.pop()
        self.text_buffer_sizes.pop()
        current_name = self.path_stack.pop()

        if current_name != name:
            self.notes.append(
                f"parser stack mismatch: expected {current_name}, received {name}"
            )

        text = "".join(pieces)

        if self.strip_text:
            text = text.strip()

        if text and len(text) >= self.text_min_length:
            self.element_text_nodes[name] += 1
            self._record_text_sample(path, text)

    def _on_char_data(self, data: str) -> None:
        if not self.text_buffer_stack or not data:
            return

        encoded_size = len(data.encode("utf-8"))

        for index in range(len(self.text_buffer_stack)):
            limit = self.limits.max_text_bytes

            if (
                limit is not None
                and self.text_buffer_sizes[index] + encoded_size > limit
            ):
                remaining = limit - self.text_buffer_sizes[index]

                if remaining <= 0:
                    continue

                encoded = data.encode("utf-8")[:remaining]

                while encoded:
                    try:
                        partial = encoded.decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        encoded = encoded[:-1]
                else:
                    partial = ""

                if partial:
                    self.text_buffer_stack[index].append(partial)
                    self.text_buffer_sizes[index] += len(
                        partial.encode("utf-8")
                    )

                continue

            self.text_buffer_stack[index].append(data)
            self.text_buffer_sizes[index] += encoded_size

    def _on_external_entity(
        self,
        context: Optional[str],
        base: Optional[str],
        system_id: Optional[str],
        public_id: Optional[str],
    ) -> int:
        identifier = system_id or public_id or "unknown"
        raise ExternalEntityError(
            f"external entity rejected: {identifier}"
        )

    def _on_skipped_entity(
        self,
        entity_name: str,
        is_parameter_entity: int,
    ) -> None:
        raise ExternalEntityError(
            f"entity expansion rejected: {entity_name}"
        )

    def _build_parser(self) -> xml.parsers.expat.XMLParserType:
        separator = self.namespace_separator if self.namespace_aware else None

        parser = xml.parsers.expat.ParserCreate(
            namespace_separator=separator
        )

        parser.XmlDeclHandler = self._on_xml_decl
        parser.StartElementHandler = self._on_start
        parser.EndElementHandler = self._on_end
        parser.CharacterDataHandler = self._on_char_data
        parser.ExternalEntityRefHandler = self._on_external_entity
        parser.SkippedEntityHandler = self._on_skipped_entity

        parser.SetParamEntityParsing(
            xml.parsers.expat.XML_PARAM_ENTITY_PARSING_NEVER
        )

        return parser

    def _begin(self, source: str) -> xml.parsers.expat.XMLParserType:
        self._reset()
        self.source = source
        self.started_at = time.perf_counter()
        return self._build_parser()

    def _finish(self) -> ParseReport:
        self.finished_at = time.perf_counter()
        return self._build_report()

    def _parse_stream(
        self,
        stream,
        source: str,
        chunk_size: int,
    ) -> ParseReport:
        parser = self._begin(source)

        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")

            self.bytes_read += len(chunk)
            self._check_byte_limit()
            parser.Parse(chunk, False)

        parser.Parse(b"", True)

        return self._finish()

    def analyze_file(
        self,
        path: str,
        chunk_size: int = 65536,
    ) -> ParseReport:
        target = Path(path)

        if not target.exists():
            raise FileNotFoundError(path)

        if not target.is_file():
            raise ValueError(f"not a regular file: {path}")

        with target.open("rb") as stream:
            return self._parse_stream(
                stream,
                source=str(target),
                chunk_size=chunk_size,
            )

    def analyze_stdin(
        self,
        chunk_size: int = 65536,
    ) -> ParseReport:
        stream = getattr(sys.stdin, "buffer", sys.stdin)

        return self._parse_stream(
            stream,
            source="<stdin>",
            chunk_size=chunk_size,
        )

    def analyze_string(
        self,
        xml_text: str,
        source: str = "<memory>",
    ) -> ParseReport:
        parser = self._begin(source)
        data = xml_text.encode("utf-8")

        self.bytes_read = len(data)
        self._check_byte_limit()

        parser.Parse(data, True)

        return self._finish()

    def _build_report(self) -> ParseReport:
        element_stats: List[ElementStat] = []

        for name, count in self.element_counts.most_common():
            attr_keys = sorted(
                self.attribute_keys.get(name, set())
            )

            samples: Dict[str, List[str]] = {}

            for attr_name in sorted(
                self.attribute_value_samples.get(name, {})
            ):
                samples[attr_name] = list(
                    self.attribute_value_samples[name][attr_name]
                )

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
            for path, count in self.path_counts.most_common(
                self.max_top_paths
            )
        ]

        if self.total_elements == 0:
            self.notes.append("document contains no elements")

        if self.encoding is None:
            self.notes.append(
                "XML declaration absent or no explicit encoding declared"
            )

        if self.limits.max_text_bytes is not None:
            truncated = any(
                size >= self.limits.max_text_bytes
                for size in self.text_buffer_sizes
            )

            if truncated:
                self.notes.append(
                    "one or more text buffers reached the configured limit"
                )

        avg_attributes_per_element = (
            self.total_attributes / self.total_elements
            if self.total_elements
            else 0.0
        )

        avg_paths_per_element = (
            len(self.path_counts) / self.total_elements
            if self.total_elements
            else 0.0
        )

        elapsed = max(
            0.0,
            self.finished_at - self.started_at,
        )

        mib = self.bytes_read / (1024.0 * 1024.0)

        throughput = (
            mib / elapsed
            if elapsed > 0.0
            else 0.0
        )

        return ParseReport(
            document=DocumentInfo(
                source=self.source,
                bytes_read=self.bytes_read,
                root_element=self.root_element,
                encoding=self.encoding,
                standalone=self.standalone,
                namespace_aware=self.namespace_aware,
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
                elapsed_seconds=elapsed,
                throughput_mib_per_second=throughput,
            ),
            element_stats=element_stats,
            path_stats=path_stats,
            text_samples=list(self.text_samples),
            notes=list(self.notes),
        )


def format_bytes(num: int) -> str:
    value = float(num)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = 0

    while value >= 1024.0 and index < len(units) - 1:
        value /= 1024.0
        index += 1

    return f"{value:.2f} {units[index]}"


def sanitize_text(value: str) -> str:
    return (
        value
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def frame_top(width: int) -> str:
    return "┌" + "─" * (width - 2) + "┐"


def frame_mid(width: int) -> str:
    return "├" + "─" * (width - 2) + "┤"


def frame_bottom(width: int) -> str:
    return "└" + "─" * (width - 2) + "┘"


def frame_line(text: str, width: int) -> str:
    inner = width - 4

    if len(text) > inner:
        if inner >= 3:
            text = text[: inner - 3] + "..."
        else:
            text = text[:inner]

    return f"│ {text.ljust(inner)} │"


def render_section(
    title: str,
    lines: List[str],
    width: int,
) -> str:
    output = [
        frame_top(width),
        frame_line(title, width),
        frame_mid(width),
    ]

    if lines:
        output.extend(
            frame_line(line, width)
            for line in lines
        )
    else:
        output.append(
            frame_line("none", width)
        )

    output.append(frame_bottom(width))

    return "\n".join(output) + "\n"


def render_summary(
    report: ParseReport,
    width: int,
) -> str:
    standalone = (
        report.document.standalone
        if report.document.standalone is not None
        else "N/A"
    )

    lines = [
        f"source={report.document.source}",
        (
            f"bytes_read={report.document.bytes_read} "
            f"({format_bytes(report.document.bytes_read)})"
        ),
        f"root_element={report.document.root_element or 'N/A'}",
        (
            f"encoding={report.document.encoding or 'N/A'} "
            f"standalone={standalone} "
            f"namespace_aware={report.document.namespace_aware}"
        ),
        (
            f"total_elements={report.summary.total_elements} "
            f"unique_elements={report.summary.unique_elements}"
        ),
        (
            f"total_attributes={report.summary.total_attributes} "
            f"unique_paths={report.summary.unique_paths} "
            f"max_depth={report.summary.max_depth}"
        ),
        (
            f"avg_attributes_per_element="
            f"{report.derived.avg_attributes_per_element:.4f} "
            f"avg_paths_per_element="
            f"{report.derived.avg_paths_per_element:.4f}"
        ),
        (
            f"elapsed_seconds="
            f"{report.derived.elapsed_seconds:.6f} "
            f"throughput_mib_per_second="
            f"{report.derived.throughput_mib_per_second:.3f}"
        ),
        (
            f"captured_text_samples="
            f"{report.derived.captured_text_samples} "
            f"note_count={report.derived.note_count}"
        ),
    ]

    if report.notes:
        lines.append("")
        lines.extend(
            f"note={note}"
            for note in report.notes
        )

    return render_section(
        "XML STREAM ANALYSIS",
        lines,
        width,
    )


def render_element_stats(
    report: ParseReport,
    limit: int,
    width: int,
) -> str:
    lines = [
        (
            f"{'ELEMENT':<24} "
            f"{'COUNT':>10} "
            f"{'MAX_DEPTH':>10} "
            f"{'TEXT_NODES':>12} "
            f"{'ATTR_KEYS'}"
        )
    ]

    for stat in report.element_stats[:limit]:
        attr_text = (
            ",".join(stat.attribute_keys)
            if stat.attribute_keys
            else "-"
        )

        lines.append(
            f"{stat.name:<24} "
            f"{stat.count:>10} "
            f"{stat.max_depth:>10} "
            f"{stat.text_nodes:>12} "
            f"{attr_text}"
        )

    return render_section(
        "ELEMENT STATS",
        lines,
        width,
    )


def render_path_stats(
    report: ParseReport,
    limit: int,
    width: int,
) -> str:
    lines = [
        f"{'COUNT':>10} PATH"
    ]

    for stat in report.path_stats[:limit]:
        lines.append(
            f"{stat.count:>10} {stat.path}"
        )

    return render_section(
        "PATH STATS",
        lines,
        width,
    )


def render_text_samples(
    report: ParseReport,
    limit: int,
    width: int,
) -> str:
    lines: List[str] = []

    for index, sample in enumerate(
        report.text_samples[:limit],
        start=1,
    ):
        lines.append(
            f"[{index}] path={sample.path}"
        )
        lines.append(
            f"    text={sanitize_text(sample.text)}"
        )

    return render_section(
        "TEXT SAMPLES",
        lines,
        width,
    )


def report_to_text(
    report: ParseReport,
    element_limit: int,
    path_limit: int,
    sample_limit: int,
    width: int,
) -> str:
    return "".join(
        [
            render_summary(report, width),
            render_element_stats(
                report,
                element_limit,
                width,
            ),
            render_path_stats(
                report,
                path_limit,
                width,
            ),
            render_text_samples(
                report,
                sample_limit,
                width,
            ),
        ]
    )


def report_to_json(
    report: ParseReport,
    compact: bool = False,
) -> str:
    data = asdict(report)

    if compact:
        return json.dumps(
            data,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n"

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def report_to_structured_text(
    report: ParseReport,
    element_limit: int,
    path_limit: int,
    sample_limit: int,
) -> str:
    lines: List[str] = []

    lines.append("[document]")
    lines.append(
        f"source={report.document.source}"
    )
    lines.append(
        f"bytes_read={report.document.bytes_read}"
    )
    lines.append(
        f"bytes_human={format_bytes(report.document.bytes_read)}"
    )
    lines.append(
        f"root_element={report.document.root_element or 'N/A'}"
    )
    lines.append(
        f"encoding={report.document.encoding or 'N/A'}"
    )
    lines.append(
        "standalone="
        + (
            str(report.document.standalone)
            if report.document.standalone is not None
            else "N/A"
        )
    )
    lines.append(
        f"namespace_aware={report.document.namespace_aware}"
    )
    lines.append("")

    lines.append("[summary]")
    lines.append(
        f"total_elements={report.summary.total_elements}"
    )
    lines.append(
        f"unique_elements={report.summary.unique_elements}"
    )
    lines.append(
        f"total_attributes={report.summary.total_attributes}"
    )
    lines.append(
        f"unique_paths={report.summary.unique_paths}"
    )
    lines.append(
        f"max_depth={report.summary.max_depth}"
    )
    lines.append("")

    lines.append("[derived]")
    lines.append(
        "avg_attributes_per_element="
        f"{report.derived.avg_attributes_per_element:.6f}"
    )
    lines.append(
        "avg_paths_per_element="
        f"{report.derived.avg_paths_per_element:.6f}"
    )
    lines.append(
        "captured_text_samples="
        f"{report.derived.captured_text_samples}"
    )
    lines.append(
        f"note_count={report.derived.note_count}"
    )
    lines.append(
        "elapsed_seconds="
        f"{report.derived.elapsed_seconds:.6f}"
    )
    lines.append(
        "throughput_mib_per_second="
        f"{report.derived.throughput_mib_per_second:.6f}"
    )
    lines.append("")

    lines.append("[elements]")

    elements = report.element_stats[:element_limit]

    if elements:
        for stat in elements:
            attr_keys = (
                ",".join(stat.attribute_keys)
                if stat.attribute_keys
                else "-"
            )

            lines.append(
                f"name={stat.name} "
                f"count={stat.count} "
                f"max_depth={stat.max_depth} "
                f"text_nodes={stat.text_nodes} "
                f"attr_keys={attr_keys}"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("[paths]")

    paths = report.path_stats[:path_limit]

    if paths:
        for stat in paths:
            lines.append(
                f"count={stat.count} "
                f"path={stat.path}"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("[text_samples]")

    samples = report.text_samples[:sample_limit]

    if samples:
        for sample in samples:
            lines.append(
                f"path={sample.path} "
                f"text={sanitize_text(sample.text)}"
            )
    else:
        lines.append("none")

    lines.append("")
    lines.append("[notes]")

    if report.notes:
        lines.extend(report.notes)
    else:
        lines.append("none")

    return "\n".join(lines) + "\n"


def report_to_ndjson(
    report: ParseReport,
    element_limit: Optional[int] = None,
    path_limit: Optional[int] = None,
    sample_limit: Optional[int] = None,
) -> str:
    lines: List[str] = []

    lines.append(
        json.dumps(
            {
                "type": "document",
                **asdict(report.document),
            },
            ensure_ascii=False,
        )
    )

    lines.append(
        json.dumps(
            {
                "type": "summary",
                **asdict(report.summary),
            },
            ensure_ascii=False,
        )
    )

    lines.append(
        json.dumps(
            {
                "type": "derived",
                **asdict(report.derived),
            },
            ensure_ascii=False,
        )
    )

    elements = (
        report.element_stats[:element_limit]
        if element_limit is not None
        else report.element_stats
    )

    for stat in elements:
        lines.append(
            json.dumps(
                {
                    "type": "element_stat",
                    **asdict(stat),
                },
                ensure_ascii=False,
            )
        )

    paths = (
        report.path_stats[:path_limit]
        if path_limit is not None
        else report.path_stats
    )

    for stat in paths:
        lines.append(
            json.dumps(
                {
                    "type": "path_stat",
                    **asdict(stat),
                },
                ensure_ascii=False,
            )
        )

    samples = (
        report.text_samples[:sample_limit]
        if sample_limit is not None
        else report.text_samples
    )

    for sample in samples:
        lines.append(
            json.dumps(
                {
                    "type": "text_sample",
                    **asdict(sample),
                },
                ensure_ascii=False,
            )
        )

    for note in report.notes:
        lines.append(
            json.dumps(
                {
                    "type": "note",
                    "text": note,
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines) + "\n"


def write_output(
    path: str,
    data: str,
) -> None:
    target = Path(path)

    if target.parent != Path("."):
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    temporary = target.with_name(
        target.name + ".tmp"
    )

    temporary.write_text(
        data,
        encoding="utf-8",
    )

    temporary.replace(target)


def positive_int(value: str) -> int:
    parsed = int(value)

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be positive"
        )

    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be non-negative"
        )

    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream-analyze XML structure using Python Expat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file.xml
  %(prog)s file.xml --format json
  %(prog)s file.xml --format compact-json
  %(prog)s file.xml --format structured
  %(prog)s file.xml --format ndjson
  %(prog)s file.xml --namespace-aware
  %(prog)s file.xml --max-depth 100 --max-elements 1000000
  %(prog)s file.xml --format json --output report.json
  %(prog)s - --format json
  cat file.xml | %(prog)s -
  %(prog)s --string "<root><item id='1'>alpha</item></root>"
        """.strip(),
    )

    parser.add_argument(
        "source",
        nargs="?",
        help="XML file path or - for stdin",
    )

    parser.add_argument(
        "--string",
        dest="xml_string",
        help="Analyze an XML string directly",
    )

    parser.add_argument(
        "--format",
        choices=[
            "text",
            "json",
            "compact-json",
            "structured",
            "ndjson",
        ],
        default="text",
        help="Output format",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Alias for --format json",
    )

    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Alias for --format compact-json",
    )

    parser.add_argument(
        "--structured-text",
        action="store_true",
        help="Alias for --format structured",
    )

    parser.add_argument(
        "--ndjson",
        action="store_true",
        help="Alias for --format ndjson",
    )

    parser.add_argument(
        "--output",
        help="Write output to a file",
    )

    parser.add_argument(
        "--elements",
        type=positive_int,
        default=20,
        help="Maximum element statistics to output",
    )

    parser.add_argument(
        "--paths",
        type=positive_int,
        default=20,
        help="Maximum path statistics to output",
    )

    parser.add_argument(
        "--samples",
        type=positive_int,
        default=10,
        help="Maximum text samples to output",
    )

    parser.add_argument(
        "--max-text-samples",
        type=positive_int,
        default=50,
        help="Maximum text samples retained internally",
    )

    parser.add_argument(
        "--max-attribute-samples",
        type=positive_int,
        default=5,
        help="Maximum sample values retained per attribute",
    )

    parser.add_argument(
        "--max-top-paths",
        type=positive_int,
        default=100,
        help="Maximum path statistics retained for reporting",
    )

    parser.add_argument(
        "--text-min-length",
        type=non_negative_int,
        default=1,
        help="Minimum text length retained as a sample",
    )

    parser.add_argument(
        "--no-strip-text",
        action="store_true",
        help="Preserve leading and trailing text whitespace",
    )

    parser.add_argument(
        "--namespace-aware",
        action="store_true",
        help="Enable namespace-aware element and attribute names",
    )

    parser.add_argument(
        "--namespace-separator",
        default="|",
        help="Namespace separator used in namespace-aware mode",
    )

    parser.add_argument(
        "--width",
        type=positive_int,
        default=120,
        help="Text output frame width",
    )

    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=65536,
        help="Streaming input chunk size in bytes",
    )

    parser.add_argument(
        "--max-bytes",
        type=positive_int,
        help="Maximum input size in bytes",
    )

    parser.add_argument(
        "--max-elements",
        type=positive_int,
        help="Maximum number of XML elements",
    )

    parser.add_argument(
        "--max-depth",
        type=positive_int,
        help="Maximum XML nesting depth",
    )

    parser.add_argument(
        "--max-text-bytes",
        type=positive_int,
        help="Maximum retained text bytes per active element",
    )

    parser.add_argument(
        "--max-unique-elements",
        type=positive_int,
        help="Maximum number of unique element names",
    )

    parser.add_argument(
        "--max-unique-paths",
        type=positive_int,
        help="Maximum number of unique XML paths",
    )

    return parser


def resolve_format(
    args: argparse.Namespace,
) -> str:
    aliases = []

    if args.json:
        aliases.append("json")

    if args.compact_json:
        aliases.append("compact-json")

    if args.structured_text:
        aliases.append("structured")

    if args.ndjson:
        aliases.append("ndjson")

    if len(aliases) > 1:
        raise ValueError(
            "choose only one legacy output format flag"
        )

    if aliases:
        if args.format != "text":
            raise ValueError(
                "do not combine --format with a legacy format flag"
            )

        return aliases[0]

    return args.format


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.source is None and args.xml_string is None:
        raise ValueError(
            "provide an XML file, - for stdin, or --string"
        )

    if args.source is not None and args.xml_string is not None:
        raise ValueError(
            "use either a source or --string, not both"
        )

    if args.width < 40:
        raise ValueError(
            "--width must be at least 40"
        )

    if len(args.namespace_separator) != 1:
        raise ValueError(
            "--namespace-separator must be exactly one character"
        )


def build_analyzer(
    args: argparse.Namespace,
) -> XMLStreamAnalyzer:
    limits = AnalyzerLimits(
        max_bytes=args.max_bytes,
        max_elements=args.max_elements,
        max_depth=args.max_depth,
        max_text_bytes=args.max_text_bytes,
        max_unique_elements=args.max_unique_elements,
        max_unique_paths=args.max_unique_paths,
    )

    return XMLStreamAnalyzer(
        max_text_samples=args.max_text_samples,
        max_attribute_samples=args.max_attribute_samples,
        max_top_paths=args.max_top_paths,
        text_min_length=args.text_min_length,
        strip_text=not args.no_strip_text,
        namespace_aware=args.namespace_aware,
        namespace_separator=args.namespace_separator,
        limits=limits,
    )


def analyze_input(
    analyzer: XMLStreamAnalyzer,
    args: argparse.Namespace,
) -> ParseReport:
    if args.xml_string is not None:
        return analyzer.analyze_string(
            args.xml_string
        )

    if args.source == "-":
        return analyzer.analyze_stdin(
            chunk_size=args.chunk_size
        )

    return analyzer.analyze_file(
        args.source,
        chunk_size=args.chunk_size,
    )


def render_report(
    report: ParseReport,
    output_format: str,
    args: argparse.Namespace,
) -> str:
    if output_format == "json":
        return report_to_json(
            report,
            compact=False,
        )

    if output_format == "compact-json":
        return report_to_json(
            report,
            compact=True,
        )

    if output_format == "structured":
        return report_to_structured_text(
            report,
            element_limit=args.elements,
            path_limit=args.paths,
            sample_limit=args.samples,
        )

    if output_format == "ndjson":
        return report_to_ndjson(
            report,
            element_limit=args.elements,
            path_limit=args.paths,
            sample_limit=args.samples,
        )

    return report_to_text(
        report,
        element_limit=args.elements,
        path_limit=args.paths,
        sample_limit=args.samples,
        width=args.width,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        output_format = resolve_format(args)
        analyzer = build_analyzer(args)
        report = analyze_input(
            analyzer,
            args,
        )
        output = render_report(
            report,
            output_format,
            args,
        )

        if args.output:
            write_output(
                args.output,
                output,
            )
        else:
            sys.stdout.write(output)

            if output and not output.endswith("\n"):
                sys.stdout.write("\n")

            sys.stdout.flush()

        return 0

    except FileNotFoundError as exc:
        print(
            f"File not found: {exc}",
            file=sys.stderr,
        )
        return 2

    except ExternalEntityError as exc:
        print(
            f"XML security error: {exc}",
            file=sys.stderr,
        )
        return 4

    except ResourceLimitError as exc:
        print(
            f"Resource limit exceeded: {exc}",
            file=sys.stderr,
        )
        return 5

    except xml.parsers.expat.ExpatError as exc:
        print(
            f"XML parse error: {exc}",
            file=sys.stderr,
        )
        return 3

    except ValueError as exc:
        print(
            f"Argument error: {exc}",
            file=sys.stderr,
        )
        return 1

    except PermissionError as exc:
        print(
            f"Permission error: {exc}",
            file=sys.stderr,
        )
        return 6

    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0

    except KeyboardInterrupt:
        print(
            "Interrupted",
            file=sys.stderr,
        )
        return 130

    except OSError as exc:
        print(
            f"I/O error: {exc}",
            file=sys.stderr,
        )
        return 7

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
