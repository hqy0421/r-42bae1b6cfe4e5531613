"""Resumable batch loading of immutable CSV files into SQLite.

The loader treats the input CSV as an *immutable* logical stream of records.
Progress is tracked by the number of **logical CSV records** (records as
parsed by :mod:`csv`, so a quoted field containing newlines still counts as a
single record) that have been committed, never by the number of physical text
lines or the number of output rows. A cleaning rule may drop a record (zero
output rows) or expand it into multiple output rows; both are handled
correctly because the logical record offset advances by exactly one for every
input record regardless of how many rows it produces.

Each batch is loaded inside a single SQLite transaction that also persists the
new logical-record prefix. If a row transformation, the cleaning generator or
a SQLite statement raises, only the current batch is rolled back; all earlier
committed batches survive and the next run resumes at the committed prefix.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import pyarrow as pa

from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import schema_to_expression

#: Name of the table storing resumable loading progress.
STATE_TABLE = "_triad_csv_load_state"

#: A cleaning rule takes a logical CSV record (list of strings) and yields
#: zero or more raw output rows. Each raw row is a dict (keyed by target
#: column name) or a list/tuple (aligned with the declared schema).
CleaningRule = Callable[[List[str]], Iterable[Union[Dict[str, Any], List[Any]]]]

_CREATE_STATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
    load_key        TEXT PRIMARY KEY,
    source_fingerprint TEXT NOT NULL,
    rule_id         TEXT NOT NULL,
    target_table    TEXT NOT NULL,
    schema_expr     TEXT NOT NULL,
    csv_dialect_fingerprint TEXT NOT NULL,
    logical_offset  INTEGER NOT NULL,
    finished        INTEGER NOT NULL,
    logical_total   INTEGER NOT NULL
)
"""


@dataclass
class CsvLoadResult:
    """Summary of a (possibly resumed) CSV load run.

    :param logical_total: total number of logical CSV records in the input
    :param logical_loaded: number of logical records that were processed and
        committed during this run (excluding records skipped because they
        were already committed by a previous run)
    :param rows_inserted: number of target rows inserted during this run
    :param resumed: whether an existing progress entry was found
    :param already_finished: whether the load had already completed in an
        earlier run and this run was an idempotent no-op
    """

    logical_total: int
    logical_loaded: int
    rows_inserted: int
    resumed: bool
    already_finished: bool


def load_csv_to_sqlite(
    source: Union[str, io.BufferedIOBase, io.RawIOBase],
    conn: sqlite3.Connection,
    table: str,
    schema: Any,
    rule: CleaningRule,
    *,
    rule_id: Optional[str] = None,
    load_key: Optional[str] = None,
    batch_size: int = 1000,
    has_header: bool = True,
    source_name: Optional[str] = None,
    csv_dialect: Optional[csv.Dialect] = None,
    **csv_kwargs: Any,
) -> CsvLoadResult:
    """Clean an immutable CSV in batches and load the rows into SQLite.

    The load is resumable: progress is stored next to the target data in the
    same database and keyed by the input content and the cleaning rule
    identity. Reopening with the same input and rule skips the already
    committed logical-record prefix without missing or duplicating rows.

    :param source: path of the CSV file, or a binary file-like object holding
        its bytes. File-like inputs must be seekable so the content can be
        fingerprinted before parsing and re-parsed on resume.
    :param conn: an open SQLite connection. The connection itself is never
        closed by this function; only transactions opened for batches are
        committed or rolled back.
    :param table: name of the target table. It is created if it does not
        exist; an existing table must have the same declared schema.
    :param schema: schema-like object (see
        :class:`~triad.collections.schema.Schema`) declaring the target
        columns. Every emitted row is converted according to this schema.
    :param rule: cleaning rule taking a logical CSV record (a list of string
        fields) and returning/yielding zero or more rows. A row is a dict
        keyed by column name or a list/tuple aligned with ``schema``.
    :param rule_id: explicit, stable identity of the cleaning rule. Required
        for lambdas/local functions; when omitted the rule's
        ``__uuid__``/qualified name is used.
    :param load_key: explicit key under which progress is stored. When
        omitted, a deterministic key derived from the source fingerprint,
        target table, rule and schema is used.
    :param batch_size: maximum number of logical CSV records processed in a
        single transaction, must be > 0.
    :param has_header: whether the first logical CSV record is a header that
        is never passed to the cleaning rule.
    :param source_name: optional logical name of the source, included in the
        progress identity to distinguish file-like inputs.
    :param csv_dialect: optional :class:`csv.Dialect` used for parsing.
    :param csv_kwargs: extra keyword arguments forwarded to
        :class:`csv.reader` (e.g. ``delimiter``, ``quotechar``).
    :raises ValueError: if the input/rule/schema identity of an existing
        progress entry does not match, if ``batch_size`` is invalid, or if
        the rule identity cannot be derived.
    :raises sqlite3.Error: if a database operation fails (the failing batch
        is rolled back first).
    :return: a :class:`CsvLoadResult` describing the run.
    """
    assert_or_throw(batch_size > 0, ValueError("batch_size must be positive"))
    target_schema = Schema(schema).assert_not_empty()
    rid = _resolve_rule_id(rule, rule_id)
    dialect_fingerprint = _fingerprint_dialect(csv_dialect, csv_kwargs)

    with _SourceReader(source, source_name) as reader:
        fingerprint = reader.fingerprint()
        key = load_key or to_uuid(
            "triad.csv.sqlite.load",
            fingerprint,
            table,
            rid,
            str(target_schema),
        )
        state = _read_state(conn, key)
        schema_expr = schema_to_expression(target_schema.pa_schema)
        if state is not None:
            _validate_state(
                state, fingerprint, rid, table, schema_expr, dialect_fingerprint
            )
            if bool(state["finished"]):
                return CsvLoadResult(
                    logical_total=state["logical_total"],
                    logical_loaded=0,
                    rows_inserted=0,
                    resumed=True,
                    already_finished=True,
                )
            logical_offset = state["logical_offset"]
            resumed = True
        else:
            logical_offset = 0
            resumed = False

        _ensure_target_table(conn, table, target_schema)

        # First pass: total logical records, so the finished row can carry
        # the count and so resume can never silently truncate a changed file
        # (the fingerprint check above already guarantees byte-identity).
        logical_total = 0
        reader.seek(0)
        for _ in _iter_logical_records(reader, csv_dialect, csv_kwargs):
            logical_total += 1
        data_records = logical_total - 1 if has_header else logical_total
        assert_or_throw(
            logical_offset <= data_records,
            ValueError(
                f"committed offset {logical_offset} exceeds the number of "
                f"data records {data_records}"
            ),
        )

        reader.seek(0)
        records = _iter_logical_records(reader, csv_dialect, csv_kwargs)
        if has_header:
            next(records, None)

        logical_loaded = 0
        rows_inserted = 0

        # Empty file, or a file containing only a header: finalize in a
        # state-only transaction so reruns are idempotent no-ops.
        if data_records == 0:
            _commit_batch(
                conn,
                key,
                table,
                target_schema,
                [],
                _StateValues(
                    fingerprint=fingerprint,
                    rule_id=rid,
                    target_table=table,
                    schema_expr=schema_expr,
                    dialect_fingerprint=dialect_fingerprint,
                    logical_offset=0,
                    finished=True,
                    logical_total=logical_total,
                ),
            )
            return CsvLoadResult(
                logical_total=logical_total,
                logical_loaded=0,
                rows_inserted=0,
                resumed=resumed,
                already_finished=False,
            )

        record = next(records, None)
        record_idx = 0
        # Records with index < logical_offset are already committed.
        while record is not None:
            if record_idx < logical_offset:
                record = next(records, None)
                record_idx += 1
                continue
            batch_rows: List[Tuple[Any, ...]] = []
            batch_count = 0
            while batch_count < batch_size and record is not None:
                try:
                    for raw in rule(record):
                        batch_rows.append(
                            _convert_row(raw, target_schema, record_idx + 1)
                        )
                except Exception as e:
                    # The current batch is not committed, so rolling back
                    # keeps the previously committed prefix intact.
                    _rollback_safely(conn)
                    raise RuntimeError(
                        f"cleaning rule or row transformation failed on "
                        f"logical data record {record_idx + 1}: {e}"
                    ) from e
                batch_count += 1
                logical_loaded += 1
                record = next(records, None)
                record_idx += 1

            new_offset = logical_offset + batch_count
            finished = new_offset >= data_records
            try:
                _commit_batch(
                    conn,
                    key,
                    table,
                    target_schema,
                    batch_rows,
                    _StateValues(
                        fingerprint=fingerprint,
                        rule_id=rid,
                        target_table=table,
                        schema_expr=schema_expr,
                        dialect_fingerprint=dialect_fingerprint,
                        logical_offset=new_offset,
                        finished=finished,
                        logical_total=logical_total,
                    ),
                )
            except Exception:
                _rollback_safely(conn)
                raise
            rows_inserted += len(batch_rows)
            logical_offset = new_offset
            if finished:
                break

        return CsvLoadResult(
            logical_total=logical_total,
            logical_loaded=logical_loaded,
            rows_inserted=rows_inserted,
            resumed=resumed,
            already_finished=False,
        )


class _SourceReader:
    """Owns the binary input stream and its sha256 content fingerprint.

    Path inputs are opened lazily and closed on exit; caller-provided binary
    streams are rewound but never closed. Every parsing pass gets a fresh
    :class:`io.TextIOWrapper` so multi-pass reads do not share decoder state.
    """

    def __init__(
        self,
        source: Union[str, io.BufferedIOBase, io.RawIOBase],
        source_name: Optional[str],
    ) -> None:
        self._path: Optional[str] = None
        self._stream: Optional[io.BufferedIOBase] = None
        self._owns_stream = False
        self.name = source_name
        if isinstance(source, str):
            self._path = source
            if self.name is None:
                self.name = source
        else:
            assert_or_throw(
                hasattr(source, "read") and hasattr(source, "seek"),
                ValueError("file-like source must be a seekable binary stream"),
            )
            self._stream = source  # type: ignore
            if self.name is None:
                self.name = getattr(source, "name", f"<stream id={id(source)}>")

    def __enter__(self) -> "_SourceReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _binary_stream(self) -> io.BufferedIOBase:
        if self._stream is None:
            self._stream = open(self._path, "rb")  # noqa: SIM115
            self._owns_stream = True
        return self._stream

    def fingerprint(self) -> str:
        """Return ``"<name>:<size>:<sha256>"`` over the raw bytes."""
        stream = self._binary_stream()
        stream.seek(0)
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        return f"{self.name}:{size}:{digest.hexdigest()}"

    def seek(self, pos: int) -> None:
        self._binary_stream().seek(pos)

    def records(
        self,
        dialect: Optional[csv.Dialect],
        csv_kwargs: Dict[str, Any],
    ) -> Iterator[List[str]]:
        stream = self._binary_stream()
        stream.seek(0)
        # newline="" is required by the csv module for correct quoting.
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        try:
            if dialect is not None:
                reader = csv.reader(text, dialect, **csv_kwargs)
            else:
                reader = csv.reader(text, **csv_kwargs)
            yield from reader
        finally:
            # Detach so the wrapper never closes a caller-owned stream; the
            # wrapper (with any buffered, undecoded bytes) is then dropped.
            # On an exception path the underlying stream may already be
            # closed by ``close()``, in which case detaching is impossible.
            try:
                text.detach()
            except (ValueError, io.UnsupportedOperation):
                pass

    def close(self) -> None:
        if self._owns_stream and self._stream is not None:
            self._stream.close()
        self._stream = None


def _iter_logical_records(
    reader: _SourceReader,
    dialect: Optional[csv.Dialect],
    csv_kwargs: Dict[str, Any],
) -> Iterator[List[str]]:
    """Yield logical CSV records exactly as parsed by :class:`csv.reader`.

    A quoted field containing newlines never splits a record, so the yielded
    count is the logical record count rather than a physical line count.
    A fully blank physical line is parsed as an empty record (``[]``) under
    the default non-strict dialect; it still counts as one logical record,
    which keeps progress deterministic regardless of output-row counts.
    """
    return reader.records(dialect, csv_kwargs)


@dataclass
class _StateValues:
    fingerprint: str
    rule_id: str
    target_table: str
    schema_expr: str
    dialect_fingerprint: str
    logical_offset: int
    finished: bool
    logical_total: int


def _read_state(conn: sqlite3.Connection, key: str) -> Optional[Dict[str, Any]]:
    conn.execute(_CREATE_STATE_SQL)
    cur = conn.execute(
        f"SELECT load_key, source_fingerprint, rule_id, target_table, "
        f"schema_expr, csv_dialect_fingerprint, logical_offset, finished, "
        f"logical_total FROM {STATE_TABLE} WHERE load_key = ?",
        (key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "load_key": row[0],
        "source_fingerprint": row[1],
        "rule_id": row[2],
        "target_table": row[3],
        "schema_expr": row[4],
        "csv_dialect_fingerprint": row[5],
        "logical_offset": row[6],
        "finished": row[7],
        "logical_total": row[8],
    }


def _validate_state(
    state: Dict[str, Any],
    fingerprint: str,
    rule_id: str,
    table: str,
    schema_expr: str,
    dialect_fingerprint: str,
) -> None:
    checks = [
        ("source content", state["source_fingerprint"], fingerprint),
        ("cleaning rule identity", state["rule_id"], rule_id),
        ("target table", state["target_table"], table),
        ("declared schema", state["schema_expr"], schema_expr),
        ("csv dialect", state["csv_dialect_fingerprint"], dialect_fingerprint),
    ]
    for label, old, new in checks:
        assert_or_throw(
            old == new,
            ValueError(
                f"resume aborted: {label} changed since the previous run "
                f"(old={old!r}, new={new!r})"
            ),
        )


def _resolve_rule_id(rule: CleaningRule, explicit: Optional[str]) -> str:
    if explicit is not None:
        return f"explicit:{explicit}"
    uuid_method = getattr(rule, "__uuid__", None)
    if callable(uuid_method):
        return f"uuid:{uuid_method()}"
    module = getattr(rule, "__module__", None)
    qualname = getattr(rule, "__qualname__", None)
    assert_or_throw(
        module is not None
        and qualname is not None
        and "<locals>" not in qualname
        and qualname != "<lambda>",
        ValueError(
            "rule identity cannot be derived for a lambda or local function; "
            "pass an explicit stable rule_id"
        ),
    )
    return f"qualname:{module}.{qualname}"


def _fingerprint_dialect(
    dialect: Optional[csv.Dialect], csv_kwargs: Dict[str, Any]
) -> str:
    parts: Dict[str, Any] = {}
    if dialect is not None:
        # User-defined Dialect subclasses only override a subset of attrs;
        # fall back to the csv module's documented defaults for the rest.
        defaults = {
            "delimiter": ",",
            "doublequote": True,
            "escapechar": None,
            "lineterminator": "\r\n",
            "quotechar": '"',
            "quoting": csv.QUOTE_MINIMAL,
            "skipinitialspace": False,
            "strict": False,
        }
        for attr, default in defaults.items():
            parts[attr] = getattr(dialect, attr, default)
    for k, v in sorted(csv_kwargs.items()):
        parts[k] = repr(v)
    return to_uuid("csv.dialect", parts)


def _sqlite_type(pa_type: pa.DataType) -> str:
    for checker, affinity in (
        (pa.types.is_null, "BLOB"),
        (pa.types.is_boolean, "INTEGER"),
        (pa.types.is_integer, "INTEGER"),
        (pa.types.is_floating, "REAL"),
        (pa.types.is_string, "TEXT"),
        (pa.types.is_large_string, "TEXT"),
        (pa.types.is_binary, "BLOB"),
        (pa.types.is_large_binary, "BLOB"),
        (pa.types.is_date, "TEXT"),
        (pa.types.is_timestamp, "TEXT"),
        (pa.types.is_decimal, "TEXT"),
    ):
        if checker(pa_type):
            return affinity
    raise NotImplementedError(f"sqlite loader does not support type {pa_type}")


def _ensure_target_table(conn: sqlite3.Connection, table: str, schema: Schema) -> None:
    columns = ", ".join(
        f'"{name}" {_sqlite_type(field.type)}' for name, field in schema.items()
    )
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({columns})')
    # Verify an existing table matches the declared schema.
    existing = conn.execute('PRAGMA table_info("' + table + '")').fetchall()
    assert_or_throw(
        [col[1] for col in existing] == schema.names,
        ValueError(
            f"existing table {table!r} columns {[col[1] for col in existing]} "
            f"do not match declared schema {schema.names}"
        ),
    )


def _convert_value(value: Any, field: pa.Field) -> Any:
    """Convert one raw value to the storable form declared by ``field``."""
    if value is None:
        return None
    t = field.type
    try:
        if pa.types.is_boolean(t):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "y", "t"):
                    return True
                if low in ("false", "0", "no", "n", "f", ""):
                    return False
                raise ValueError(f"can't convert {value!r} to bool")
            return bool(value)
        if pa.types.is_integer(t):
            if isinstance(value, bool):
                return int(value)
            return int(value)
        if pa.types.is_floating(t):
            return float(value)
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return str(value)
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            if isinstance(value, bytes):
                return value
            return str(value).encode("utf-8")
        if pa.types.is_date(t):
            if isinstance(value, date) and not isinstance(value, datetime):
                return value.isoformat()
            return str(value)
        if pa.types.is_timestamp(t):
            if isinstance(value, datetime):
                return value.isoformat(sep=" ")
            return str(value)
        if pa.types.is_decimal(t):
            return str(value)
        if pa.types.is_null(t):
            return None
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"can't convert value {value!r} for column {field.name!r} ({t}): {e}"
        ) from e
    raise NotImplementedError(
        f"sqlite loader does not support column {field.name!r} of type {t}"
    )


def _convert_row(
    raw: Union[Dict[str, Any], List[Any], Tuple[Any, ...]],
    schema: Schema,
    logical_index: int,
) -> Tuple[Any, ...]:
    """Validate and convert one cleaning output row against the schema."""
    if isinstance(raw, dict):
        unknown = set(raw.keys()) - set(schema.names)
        assert_or_throw(
            not unknown,
            ValueError(
                f"row for logical record {logical_index} contains columns "
                f"{sorted(unknown)} not in schema {schema.names}"
            ),
        )
        values = (raw.get(name) for name in schema.names)
        return tuple(
            _convert_value(v, schema[name]) for name, v in zip(schema.names, values)
        )
    assert_or_throw(
        isinstance(raw, (list, tuple)),
        TypeError(
            f"cleaning rule must emit dict or list/tuple rows, got "
            f"{type(raw).__name__} for logical record {logical_index}"
        ),
    )
    assert_or_throw(
        len(raw) == len(schema),
        ValueError(
            f"row for logical record {logical_index} has {len(raw)} values "
            f"but schema has {len(schema)} columns"
        ),
    )
    return tuple(_convert_value(v, field) for v, field in zip(raw, schema.fields))


def _commit_batch(
    conn: sqlite3.Connection,
    key: str,
    table: str,
    schema: Schema,
    rows: List[Tuple[Any, ...]],
    state: _StateValues,
) -> None:
    """Insert target rows and advance progress in one transaction."""
    conn.execute("BEGIN IMMEDIATE")
    placeholders = ", ".join("?" for _ in schema.names)
    quoted_cols = ", ".join(f'"{name}"' for name in schema.names)
    insert_sql = f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({placeholders})'
    conn.executemany(insert_sql, rows)
    conn.execute(
        f"INSERT INTO {STATE_TABLE} (load_key, source_fingerprint, rule_id, "
        f"target_table, schema_expr, csv_dialect_fingerprint, logical_offset, "
        f"finished, logical_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        f"ON CONFLICT(load_key) DO UPDATE SET "
        f"source_fingerprint=excluded.source_fingerprint, "
        f"rule_id=excluded.rule_id, "
        f"target_table=excluded.target_table, "
        f"schema_expr=excluded.schema_expr, "
        f"csv_dialect_fingerprint=excluded.csv_dialect_fingerprint, "
        f"logical_offset=excluded.logical_offset, "
        f"finished=excluded.finished, "
        f"logical_total=excluded.logical_total",
        (
            key,
            state.fingerprint,
            state.rule_id,
            state.target_table,
            state.schema_expr,
            state.dialect_fingerprint,
            state.logical_offset,
            1 if state.finished else 0,
            state.logical_total,
        ),
    )
    conn.commit()


def _rollback_safely(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
