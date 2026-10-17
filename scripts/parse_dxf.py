"""Parse DXF files into structured JSON (stage 1 of the pipeline)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from _common import JSON_DIR, RAW_DIR, dump_json, load_dxf, run_batch

# Entities whose geometry we serialise in full. Anything else is still recorded
# with its type and layer so the statistics stay accurate.
TEXT_TYPES = ("TEXT", "MTEXT")

# $INSUNITS codes. Without this the drawing's numbers have no meaning: the AISC
# steel sheets are in inches while the other samples are in metres, so a
# question about millimetres cannot be answered from the coordinates alone.
UNIT_NAMES = {
    0: "无单位",
    1: "英寸",
    2: "英尺",
    3: "英里",
    4: "毫米",
    5: "厘米",
    6: "米",
    7: "千米",
    8: "微英寸",
    9: "密尔",
    10: "码",
    11: "埃",
    12: "纳米",
    13: "微米",
    14: "分米",
    15: "十米",
    16: "百米",
    17: "吉米",
    18: "天文单位",
    19: "光年",
    20: "秒差距",
}

# Cell text positions are rounded to this many drawing units before being
# grouped into rows, so glyphs sitting on one baseline stay together.
TABLE_ROW_TOLERANCE = 1.0


def point(value) -> list[float]:
    return [float(v) for v in value]


def text_content(entity) -> str:
    if entity.dxftype() == "MTEXT":
        return entity.plain_text()
    return entity.dxf.text


def geometry_of(entity) -> dict | None:
    dxftype = entity.dxftype()

    if dxftype == "LINE":
        return {"start": point(entity.dxf.start), "end": point(entity.dxf.end)}

    if dxftype == "CIRCLE":
        return {"center": point(entity.dxf.center), "radius": float(entity.dxf.radius)}

    if dxftype == "ARC":
        return {
            "center": point(entity.dxf.center),
            "radius": float(entity.dxf.radius),
            "start_angle": float(entity.dxf.start_angle),
            "end_angle": float(entity.dxf.end_angle),
        }

    if dxftype == "LWPOLYLINE":
        # Keep bulge so arc segments survive; get_points("xy") alone drops them.
        vertices = []
        has_bulge = False
        for p in entity.get_points("xyb"):
            x, y, bulge = float(p[0]), float(p[1]), float(p[2])
            vertex = {"x": x, "y": y}
            if bulge:
                has_bulge = True
                vertex["bulge"] = bulge
            vertices.append(vertex)
        geometry = {
            "closed": bool(entity.closed),
            "vertices": [[v["x"], v["y"]] for v in vertices],
            "vertex_details": vertices,
        }
        if has_bulge:
            geometry["has_bulge"] = True
        return geometry

    if dxftype == "POLYLINE":
        return {
            "closed": bool(entity.is_closed),
            "vertices": [point(v.dxf.location) for v in entity.vertices],
        }

    if dxftype == "ELLIPSE":
        return {
            "center": point(entity.dxf.center),
            "major_axis": point(entity.dxf.major_axis),
            "ratio": float(entity.dxf.ratio),
        }

    if dxftype == "POINT":
        return {"location": point(entity.dxf.location)}

    if dxftype == "INSERT":
        return {"insert": point(entity.dxf.insert)}

    if dxftype in TEXT_TYPES:
        return {"position": point(entity.dxf.insert)}

    if dxftype == "DIMENSION":
        geometry = {"definition_point": point(entity.dxf.defpoint)}
        for name in ("text_midpoint", "defpoint2", "defpoint3"):
            if entity.dxf.hasattr(name):
                geometry[name] = point(entity.dxf.get(name))
        return geometry

    return None


def attributes_of(entity) -> dict | None:
    dxftype = entity.dxftype()

    if dxftype == "INSERT":
        attributes = {
            "block_name": entity.dxf.name,
            "xscale": float(entity.dxf.xscale),
            "yscale": float(entity.dxf.yscale),
            "rotation": float(entity.dxf.rotation),
        }
        if entity.has_attrib:
            attributes["attribs"] = {
                attrib.dxf.tag: attrib.dxf.text for attrib in entity.attribs
            }
        return attributes

    if dxftype in TEXT_TYPES:
        return {
            "content": text_content(entity),
            "height": float(getattr(entity.dxf, "height", 0) or 0),
            "rotation": float(getattr(entity.dxf, "rotation", 0) or 0),
        }

    if dxftype == "DIMENSION":
        # measurement is the value CAD computed; text may override it.
        attributes = {"dimension_type": int(entity.dxf.dimtype)}
        if entity.dxf.hasattr("text"):
            attributes["text"] = entity.dxf.text
        measurement = dimension_measurement(entity)
        if measurement is not None:
            attributes["measurement"] = measurement
        return attributes

    return None


def dimension_measurement(entity) -> float | None:
    """Read the numeric measurement, including older DXF that omit the attrib.

    R14 samples often have no DXF `measurement` group; ezdxf still exposes the
    value through `get_measurement()`. Ordinate-style dims may return a point —
    those are skipped here so the index only keeps scalar lengths.
    """
    if entity.dxf.hasattr("measurement"):
        try:
            return float(entity.dxf.measurement)
        except (TypeError, ValueError):
            pass
    try:
        raw = entity.get_measurement()
    except Exception:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def entity_to_dict(entity) -> dict:
    try:
        data = {
            "type": entity.dxftype(),
            "layer": entity.dxf.layer,
            "handle": entity.dxf.handle,
        }
        geometry = geometry_of(entity)
        if geometry is not None:
            data["geometry"] = geometry
        attributes = attributes_of(entity)
        if attributes is not None:
            data["attributes"] = attributes
        return data
    except Exception as exc:  # noqa: BLE001 — keep one bad entity from aborting the drawing
        try:
            dxftype = entity.dxftype()
        except Exception:
            dxftype = "UNKNOWN"
        try:
            layer = entity.dxf.layer
        except Exception:
            layer = "?"
        try:
            handle = entity.dxf.handle
        except Exception:
            handle = "?"
        return {
            "type": dxftype,
            "layer": layer,
            "handle": handle,
            "parse_error": str(exc)[:200],
        }


def expand_insert(entity) -> dict | None:
    """Describe what a block reference actually contains.

    Answering "how many doors are in this drawing" needs the block contents,
    not just its insertion point.
    """
    if entity.dxftype() != "INSERT":
        return None
    try:
        parts = Counter(sub.dxftype() for sub in entity.virtual_entities())
    except Exception:
        return None
    if not parts:
        return None
    return {
        "block_name": entity.dxf.name,
        "insert": point(entity.dxf.insert),
        "layer": entity.dxf.layer,
        "content_types": dict(parts),
        "content_count": sum(parts.values()),
    }


def collect_texts(entities) -> list[dict]:
    texts = []
    for entity in entities:
        if entity.dxftype() not in TEXT_TYPES:
            continue
        texts.append(
            {
                "content": text_content(entity),
                "layer": entity.dxf.layer,
                "handle": entity.dxf.handle,
                "position": point(entity.dxf.insert),
            }
        )
    return texts


def header_info(doc) -> dict:
    """Pull the header variables that give the coordinates meaning."""
    header = doc.header
    code = header.get("$INSUNITS", None)
    measurement = header.get("$MEASUREMENT", None)

    extents = None
    ext_min, ext_max = header.get("$EXTMIN", None), header.get("$EXTMAX", None)
    if ext_min is not None and ext_max is not None:
        extents = {"min": point(ext_min), "max": point(ext_max)}

    return {
        "insunits_code": int(code) if code is not None else None,
        "units": UNIT_NAMES.get(code, "未指定" if code is None else f"未知({code})"),
        "measurement": (
            None if measurement is None else ("公制" if int(measurement) == 1 else "英制")
        ),
        "ltscale": float(header.get("$LTSCALE", 1.0)),
        "extents": extents,
    }


def layer_info(layer) -> dict:
    """A layer's state is part of the drawing's meaning, not just its colour.

    A frozen layer is invisible, so "which drawing has a frozen layer" is a
    question about this flag -- previously it could only be answered by reading
    a caption off the rendered image.
    """
    return {
        "name": layer.dxf.name,
        "color": int(layer.dxf.color),
        "linetype": layer.dxf.linetype,
        "frozen": bool(layer.is_frozen()),
        "off": bool(layer.is_off()),
        "locked": bool(layer.is_locked()),
        "plot": bool(layer.dxf.plot),
    }


def viewport_info(entity) -> dict:
    """Describe a paper-space viewport, including its per-viewport frozen layers.

    Layer visibility can be overridden inside a single viewport, which is
    exactly what the viewport sample drawing demonstrates.
    """
    record = {
        "id": entity.dxf.get("id", None),
        "layer": entity.dxf.layer,
        "center": point(entity.dxf.center),
        "width": float(entity.dxf.width),
        "height": float(entity.dxf.height),
        "status": entity.dxf.get("status", None),
    }
    if entity.dxf.hasattr("view_center_point"):
        record["view_center"] = point(entity.dxf.view_center_point)
    if entity.dxf.hasattr("view_height"):
        record["view_height"] = float(entity.dxf.view_height)
    try:
        record["frozen_layers"] = list(entity.frozen_layers)
    except Exception:
        record["frozen_layers"] = []
    return record


def collect_layouts(doc) -> list[dict]:
    """Parse the paper-space layouts.

    Only the model space was read before, which silently dropped every viewport
    and every annotation living on a layout -- and rendering ignores them too,
    so nothing downstream could see them at all.
    """
    layouts = []
    for layout in doc.layouts:
        if layout.name.lower() == "model":
            continue

        entities = list(layout)
        if not entities:
            continue

        layouts.append(
            {
                "name": layout.name,
                "entity_count": len(entities),
                "entity_type_counts": dict(
                    Counter(entity.dxftype() for entity in entities)
                ),
                "entities": [entity_to_dict(entity) for entity in entities],
                "texts": collect_texts(entities),
                "viewports": [
                    viewport_info(entity)
                    for entity in entities
                    if entity.dxftype() == "VIEWPORT"
                ],
            }
        )
    return layouts


def table_rows(entity) -> list[list[str]]:
    """Recover a table's visible content, row by row.

    The cell text is not stored as entities: it only appears when the table is
    expanded into virtual ones. Cells are grouped by their y coordinate because
    that is what a reader sees as a row.
    """
    cells: list[tuple[float, float, str]] = []
    try:
        virtual = list(entity.virtual_entities())
    except Exception:
        return []

    for sub in virtual:
        if sub.dxftype() not in TEXT_TYPES:
            continue
        content = text_content(sub).strip()
        if not content:
            continue
        position = sub.dxf.insert
        cells.append((float(position[1]), float(position[0]), content))

    grouped: dict[float, list[tuple[float, str]]] = {}
    for y, x, content in cells:
        key = round(y / TABLE_ROW_TOLERANCE) * TABLE_ROW_TOLERANCE
        grouped.setdefault(key, []).append((x, content))

    # Top row first, matching how the table reads on the sheet.
    return [
        [content for _, content in sorted(row)]
        for _, row in sorted(grouped.items(), reverse=True)
    ]


def collect_tables(entities) -> list[dict]:
    """Extract ACAD_TABLE structure and content."""
    tables = []
    for entity in entities:
        if entity.dxftype() != "ACAD_TABLE":
            continue

        rows = table_rows(entity)
        tables.append(
            {
                "handle": entity.dxf.handle,
                "layer": entity.dxf.layer,
                "insert": point(entity.dxf.insert),
                # These are what the file declares; both are 0 in the sample, so
                # the row count below is derived from cell positions instead and
                # must not be presented as if the file stated it.
                "declared_rows": int(entity.dxf.get("n_rows", 0) or 0),
                "declared_cols": int(entity.dxf.get("n_cols", 0) or 0),
                "rows_detected": len(rows),
                "cell_count": sum(len(row) for row in rows),
                "rows": rows,
            }
        )
    return tables


def collect_dimensions(entities) -> list[dict]:
    dimensions = []
    for entity in entities:
        if entity.dxftype() != "DIMENSION":
            continue
        record = {
            "handle": entity.dxf.handle,
            "layer": entity.dxf.layer,
            "definition_point": point(entity.dxf.defpoint),
        }
        if entity.dxf.hasattr("text"):
            record["text"] = entity.dxf.text
        measurement = dimension_measurement(entity)
        if measurement is not None:
            record["measurement"] = measurement
        dimensions.append(record)
    return dimensions


def parse_dxf(path: Path) -> dict:
    doc, auditor, recovered = load_dxf(path)
    warnings: list[str] = []
    if recovered:
        warnings.append("opened via ezdxf.recover.readfile")
    if auditor is not None:
        try:
            error_count = auditor.has_errors
            if error_count:
                warnings.append(f"auditor reported errors (has_errors={error_count})")
        except Exception:
            pass

    entities = list(doc.modelspace())

    layers = [layer_info(layer) for layer in doc.layers]

    blocks = [
        {"name": block.name, "entity_count": len(list(block))}
        for block in doc.blocks
        if not block.name.startswith("*")
    ]

    parsed_entities: list[dict] = []
    for entity in entities:
        record = entity_to_dict(entity)
        if record.get("parse_error"):
            warnings.append(
                f"entity {record.get('handle')}/{record.get('type')}: "
                f"{record['parse_error']}"
            )
        parsed_entities.append(record)
    texts = collect_texts(entities)
    dimensions = collect_dimensions(entities)
    tables = collect_tables(entities)
    layouts = collect_layouts(doc)
    block_refs = [
        expanded
        for expanded in (expand_insert(entity) for entity in entities)
        if expanded is not None
    ]
    type_counts = Counter(entity["type"] for entity in parsed_entities)

    viewports = [
        viewport for layout in layouts for viewport in layout["viewports"]
    ]
    viewport_frozen_layers = sorted(
        {name for viewport in viewports for name in viewport["frozen_layers"]}
    )

    result = {
        "file_name": path.name,
        "dxf_version": doc.dxfversion,
        "header": header_info(doc),
        "layers": layers,
        "blocks": blocks,
        "block_references": block_refs,
        "entities": parsed_entities,
        "texts": texts,
        "dimensions": dimensions,
        "tables": tables,
        "layouts": layouts,
        "stats": {
            "entity_count": len(parsed_entities),
            "layer_count": len(layers),
            "block_count": len(blocks),
            "block_reference_count": len(block_refs),
            "text_count": len(texts),
            "dimension_count": len(dimensions),
            "table_count": len(tables),
            "table_cell_count": sum(table["cell_count"] for table in tables),
            "layout_count": len(layouts),
            "paperspace_entity_count": sum(
                layout["entity_count"] for layout in layouts
            ),
            "paperspace_text_count": sum(len(layout["texts"]) for layout in layouts),
            "viewport_count": len(viewports),
            "frozen_layers": [
                layer["name"] for layer in layers if layer["frozen"]
            ],
            "viewport_frozen_layers": viewport_frozen_layers,
            "entity_type_counts": dict(type_counts),
        },
    }
    if recovered:
        result["recovered"] = True
    if warnings:
        result["parse_warnings"] = warnings[:50]
        result["stats"]["parse_warning_count"] = len(warnings)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate each written JSON against schemas/drawing.schema.json",
    )
    args = parser.parse_args()

    validator = None
    if args.validate:
        try:
            from jsonschema import Draft202012Validator
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "jsonschema is not installed. Run: py -m pip install jsonschema"
            ) from exc
        from _common import load_json as load_schema_json

        schema = load_schema_json(
            Path(__file__).resolve().parents[1] / "schemas" / "drawing.schema.json"
        )
        validator = Draft202012Validator(schema)

    files = sorted(RAW_DIR.glob("*.dxf"))

    def handle(path: Path) -> str:
        result = parse_dxf(path)
        dump_json(JSON_DIR / f"{path.stem}.json", result)
        if validator is not None:
            errors = sorted(validator.iter_errors(result), key=lambda e: e.path)
            if errors:
                first = errors[0]
                location = "/".join(str(part) for part in first.absolute_path) or "(root)"
                raise ValueError(f"schema: {location}: {first.message[:120]}")
        stats = result["stats"]
        return (
            f"{result['header']['units']} | "
            f"layers {stats['layer_count']}"
            + (f"(冻结{len(stats['frozen_layers'])})" if stats["frozen_layers"] else "")
            + f" | entities {stats['entity_count']} | "
            f"blocks {stats['block_count']} | "
            f"texts {stats['text_count']} | "
            f"dims {stats['dimension_count']} | "
            f"tables {stats['table_count']}"
            + (f"({stats['table_cell_count']}格)" if stats["table_count"] else "")
            + (
                f" | paperspace {stats['paperspace_entity_count']}"
                f"(视口{stats['viewport_count']})"
                if stats["layout_count"]
                else ""
            )
        )

    run_batch(
        files,
        handle,
        out_dir=JSON_DIR,
        empty_hint=f"Put DXF files in {RAW_DIR}",
    )


if __name__ == "__main__":
    main()
