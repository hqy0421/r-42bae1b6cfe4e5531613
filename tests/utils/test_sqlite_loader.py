import csv
import io
import os
import sqlite3

import pytest

from triad.utils.sqlite_loader import (
    STATE_TABLE,
    CsvLoadResult,
    load_csv_to_sqlite,
)

SCHEMA = "id:long,name:str,active:bool,score:double"


def _write(path, content, newline=""):
    with open(path, "w", newline=newline) as f:
        f.write(content)


def _rows(conn, table="out"):
    return conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()


def _state(conn, key=None):
    if key is None:
        return conn.execute(
            f"SELECT logical_offset, finished, logical_total FROM {STATE_TABLE}"
        ).fetchall()
    return conn.execute(
        f"SELECT logical_offset, finished, logical_total FROM {STATE_TABLE} "
        f"WHERE load_key=?",
        (key,),
    ).fetchone()


def passthrough_rule(rec):
    """One output row per record, values aligned with SCHEMA."""
    return [(int(rec[0]), rec[1], rec[2] == "1", float(rec[3]))]


def test_basic_load_with_header_and_types(tmpdir):
    src = os.path.join(str(tmpdir), "a.csv")
    db = os.path.join(str(tmpdir), "a.db")
    _write(
        src,
        "id,name,active,score\n1,alice,1,1.5\n2,bob,0,2.25\n",
    )
    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(src, conn, "out", SCHEMA, passthrough_rule, rule_id="r1")
    assert isinstance(res, CsvLoadResult)
    assert res.logical_total == 3  # header counts as a logical record
    assert res.logical_loaded == 2
    assert res.rows_inserted == 2
    assert res.resumed is False
    assert res.already_finished is False
    assert _rows(conn) == [
        (1, "alice", True, 1.5),
        (2, "bob", False, 2.25),
    ]
    state = _state(conn)[0]
    assert state == (2, 1, 3)
    conn.close()


def test_quoted_newline_is_one_logical_record(tmpdir):
    src = os.path.join(str(tmpdir), "q.csv")
    db = os.path.join(str(tmpdir), "q.db")
    # record 2 spans 3 physical lines but is one logical record
    _write(
        src,
        'id,name,active,score\n1,"line1\nline2\nline3",1,1.0\n2,z,0,2.0\n',
    )
    physical_lines = open(src).read().splitlines()
    assert len(physical_lines) == 5  # prove progress is not physical lines

    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, batch_size=1, rule_id="r"
    )
    assert res.logical_total == 3
    assert res.logical_loaded == 2
    assert _rows(conn) == [
        (1, "line1\nline2\nline3", True, 1.0),
        (2, "z", False, 2.0),
    ]
    conn.close()


def test_zero_output_records_still_advance_progress(tmpdir):
    src = os.path.join(str(tmpdir), "z.csv")
    db = os.path.join(str(tmpdir), "z.db")
    _write(src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(6)))

    def drop_even(rec):
        if int(rec[0]) % 2 == 0:
            return []
        return [(int(rec[0]), rec[1], True, 1.0)]

    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, drop_even, batch_size=2, rule_id="drop"
    )
    # 6 data records processed, only 3 rows emitted
    assert res.logical_loaded == 6
    assert res.rows_inserted == 3
    assert _state(conn)[0][0] == 6  # offset by logical records, not outputs
    assert [r[0] for r in _rows(conn)] == [1, 3, 5]
    conn.close()


def test_generator_expands_one_record_to_many_rows(tmpdir):
    src = os.path.join(str(tmpdir), "e.csv")
    db = os.path.join(str(tmpdir), "e.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n2,b,1,2.0\n")

    def expand(rec):
        base = (int(rec[0]), rec[1], True, float(rec[3]))
        yield base
        yield (int(rec[0]) + 100, rec[1] + "_copy", False, float(rec[3]) + 0.5)

    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, expand, batch_size=1, rule_id="exp"
    )
    assert res.logical_loaded == 2
    assert res.rows_inserted == 4
    assert _state(conn)[0][0] == 2
    assert len(_rows(conn)) == 4
    conn.close()


def test_resume_after_failure_keeps_prior_batches_and_skips_prefix(tmpdir):
    src = os.path.join(str(tmpdir), "f.csv")
    db = os.path.join(str(tmpdir), "f.db")
    _write(
        src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(1, 7))
    )
    seen = []

    def flaky(rec):
        seen.append(int(rec[0]))
        if rec[0] == "5":
            raise ValueError("boom")
        return [(int(rec[0]), rec[1], True, 1.0)]

    conn = sqlite3.connect(db)
    with pytest.raises(RuntimeError, match="logical data record 5"):
        load_csv_to_sqlite(
            src, conn, "out", SCHEMA, flaky, batch_size=4, rule_id="stable"
        )
    # batch 1 (records 1-4) committed; failing batch (5-...) rolled back
    assert [r[0] for r in _rows(conn)] == [1, 2, 3, 4]
    assert _state(conn)[0] == (4, 0, 7)
    assert conn.in_transaction is False

    # resume in a fresh connection with a healthy rule sharing rule_id
    conn.close()
    conn = sqlite3.connect(db)
    seen.clear()

    def good(rec):
        seen.append(int(rec[0]))
        return [(int(rec[0]), rec[1], True, 1.0)]

    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, good, batch_size=4, rule_id="stable"
    )
    # records 1-4 skipped: rule must only see 5 and 6
    assert seen == [5, 6]
    assert res.resumed is True
    assert res.logical_loaded == 2
    assert res.rows_inserted == 2
    assert [r[0] for r in _rows(conn)] == [1, 2, 3, 4, 5, 6]
    assert _state(conn)[0] == (6, 1, 7)
    conn.close()


def test_generator_error_rolls_back_only_current_batch(tmpdir):
    src = os.path.join(str(tmpdir), "g.csv")
    db = os.path.join(str(tmpdir), "g.db")
    _write(
        src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(1, 7))
    )

    def bad_generator(rec):
        yield (int(rec[0]), rec[1], True, 1.0)
        if rec[0] == "5":
            raise RuntimeError("generator exploded")

    conn = sqlite3.connect(db)
    with pytest.raises(RuntimeError, match="generator exploded"):
        load_csv_to_sqlite(
            src, conn, "out", SCHEMA, bad_generator, batch_size=4, rule_id="g"
        )
    # batch 1 is records 1-4 and would have committed, but it is consumed
    # into the same transaction window as the failing second batch only if
    # batch_size is large enough; with batch_size=4 the first batch commits
    # independently, so it survives while the failing batch rolls back
    assert [r[0] for r in _rows(conn)] == [1, 2, 3, 4]
    assert _state(conn)[0][0] == 4
    assert conn.in_transaction is False
    conn.close()


def test_sql_error_rolls_back_current_batch_only(tmpdir):
    src = os.path.join(str(tmpdir), "s.csv")
    db = os.path.join(str(tmpdir), "s.db")
    _write(
        src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(1, 7))
    )
    conn = sqlite3.connect(db)
    # first run loads and commits all records (batch 1-4, then tail 5-6)
    load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        batch_size=4,
        rule_id="sql",
        load_key="job-sql",
    )
    # simulate a crash after only the first batch had committed
    conn.execute(
        f"UPDATE {STATE_TABLE} SET finished=0, logical_offset=4 "
        f"WHERE load_key='job-sql'"
    )
    conn.execute("DELETE FROM out WHERE id > 4")
    conn.commit()

    # make every remaining INSERT fail while leaving the target table in place
    conn.execute(
        "CREATE TRIGGER fail_insert BEFORE INSERT ON out "
        "BEGIN SELECT RAISE(ABORT, 'injected sql failure'); END"
    )
    conn.commit()
    with pytest.raises(sqlite3.Error, match="injected sql failure"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            SCHEMA,
            passthrough_rule,
            batch_size=4,
            rule_id="sql",
            load_key="job-sql",
        )
    # progress must not have advanced past 4, prior batch rows survive
    assert _state(conn, "job-sql")[0] == 4
    assert [r[0] for r in _rows(conn)] == [1, 2, 3, 4]
    assert conn.in_transaction is False
    conn.close()


def test_rerun_completed_load_is_idempotent(tmpdir):
    src = os.path.join(str(tmpdir), "i.csv")
    db = os.path.join(str(tmpdir), "i.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n2,b,0,2.0\n")
    conn = sqlite3.connect(db)
    first = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, rule_id="idem"
    )
    assert first.already_finished is False
    second = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, rule_id="idem"
    )
    assert second.already_finished is True
    assert second.resumed is True
    assert second.logical_loaded == 0
    assert second.rows_inserted == 0
    # no duplicated target rows, single state row
    assert len(_rows(conn)) == 2
    assert conn.execute(f"SELECT COUNT(*) FROM {STATE_TABLE}").fetchone()[0] == 1
    conn.close()


def test_empty_file_finalizes_without_rows(tmpdir):
    src = os.path.join(str(tmpdir), "empty.csv")
    db = os.path.join(str(tmpdir), "empty.db")
    _write(src, "")
    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, has_header=False, rule_id="e"
    )
    assert res.logical_total == 0
    assert res.logical_loaded == 0
    assert res.rows_inserted == 0
    assert _rows(conn) == []
    assert _state(conn)[0] == (0, 1, 0)
    # rerun is idempotent
    res2 = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, has_header=False, rule_id="e"
    )
    assert res2.already_finished is True
    conn.close()


def test_header_only_file_finalizes(tmpdir):
    src = os.path.join(str(tmpdir), "h.csv")
    db = os.path.join(str(tmpdir), "h.db")
    _write(src, "id,name,active,score\n")
    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(src, conn, "out", SCHEMA, passthrough_rule, rule_id="h")
    assert res.logical_total == 1
    assert res.logical_loaded == 0
    assert _state(conn)[0] == (0, 1, 1)
    assert _rows(conn) == []
    conn.close()


def test_no_header_counts_all_records(tmpdir):
    src = os.path.join(str(tmpdir), "nh.csv")
    db = os.path.join(str(tmpdir), "nh.db")
    _write(src, "1,a,1,1.0\n2,b,0,2.0\n")
    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        has_header=False,
        batch_size=1,
        rule_id="nh",
    )
    assert res.logical_total == 2
    assert res.logical_loaded == 2
    assert len(_rows(conn)) == 2
    conn.close()


def test_changed_input_content_blocks_resume(tmpdir):
    src = os.path.join(str(tmpdir), "c.csv")
    db = os.path.join(str(tmpdir), "c.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        rule_id="c",
        load_key="job-c",
    )
    # same path, different bytes
    _write(src, "id,name,active,score\n1,a,1,1.0\n2,b,0,2.0\n")
    with pytest.raises(ValueError, match="source content changed"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            SCHEMA,
            passthrough_rule,
            rule_id="c",
            load_key="job-c",
        )
    # failed validation must not have inserted anything
    assert len(_rows(conn)) == 1
    conn.close()


def test_changed_rule_identity_blocks_resume(tmpdir):
    src = os.path.join(str(tmpdir), "r.csv")
    db = os.path.join(str(tmpdir), "r.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        rule_id="v1",
        load_key="job-r",
    )
    with pytest.raises(ValueError, match="cleaning rule identity changed"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            SCHEMA,
            passthrough_rule,
            rule_id="v2",
            load_key="job-r",
        )
    conn.close()


def test_changed_schema_blocks_resume(tmpdir):
    src = os.path.join(str(tmpdir), "sc.csv")
    db = os.path.join(str(tmpdir), "sc.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        rule_id="sc",
        load_key="job-sc",
    )

    def string_score_rule(rec):
        return [(int(rec[0]), rec[1], rec[2] == "1", rec[3])]

    with pytest.raises(ValueError, match="declared schema changed"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            "id:long,name:str,active:bool,score:str",
            string_score_rule,
            rule_id="sc",
            load_key="job-sc",
        )
    conn.close()


def test_changed_csv_dialect_blocks_resume(tmpdir):
    src = os.path.join(str(tmpdir), "d.csv")
    db = os.path.join(str(tmpdir), "d.db")
    _write(src, "id;name;active;score\n1;a;1;1.0\n")
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        rule_id="d",
        load_key="job-d",
        delimiter=";",
    )
    with pytest.raises(ValueError, match="csv dialect changed"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            SCHEMA,
            passthrough_rule,
            rule_id="d",
            load_key="job-d",
            delimiter="|",
        )
    conn.close()


def test_explicit_load_key_isolates_distinct_loads(tmpdir):
    src = os.path.join(str(tmpdir), "k.csv")
    db = os.path.join(str(tmpdir), "k.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    r1 = load_csv_to_sqlite(src, conn, "t1", SCHEMA, passthrough_rule, load_key="job-1")
    r2 = load_csv_to_sqlite(src, conn, "t2", SCHEMA, passthrough_rule, load_key="job-2")
    assert r1.resumed is False and r2.resumed is False
    assert len(_rows(conn, "t1")) == 1
    assert len(_rows(conn, "t2")) == 1
    assert conn.execute(f"SELECT COUNT(*) FROM {STATE_TABLE}").fetchone()[0] == 2
    conn.close()


def test_lambda_without_rule_id_is_rejected(tmpdir):
    src = os.path.join(str(tmpdir), "l.csv")
    db = os.path.join(str(tmpdir), "l.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    with pytest.raises(ValueError, match="rule identity cannot be derived"):
        load_csv_to_sqlite(src, conn, "out", SCHEMA, lambda rec: [tuple(rec)])
    conn.close()


def test_dict_rows_are_converted_and_validated(tmpdir):
    src = os.path.join(str(tmpdir), "dr.csv")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")

    # happy path: dict rows converted according to the declared schema
    db = os.path.join(str(tmpdir), "dr.db")
    conn = sqlite3.connect(db)

    def dict_rule(rec):
        yield {
            "id": int(rec[0]),
            "name": rec[1],
            "active": rec[2] == "1",
            "score": float(rec[3]),
        }

    load_csv_to_sqlite(src, conn, "out", SCHEMA, dict_rule, rule_id="dr")
    assert _rows(conn) == [(1, "a", True, 1.0)]
    conn.close()

    # unknown columns are rejected against a fresh database/load
    db2 = os.path.join(str(tmpdir), "dr2.db")
    conn2 = sqlite3.connect(db2)

    def unknown_column(rec):
        yield {"id": 1, "name": "a", "active": True, "score": 1.0, "extra": 9}

    with pytest.raises(RuntimeError, match="not in schema") as exc_info:
        load_csv_to_sqlite(src, conn2, "out", SCHEMA, unknown_column, rule_id="drx")
    assert isinstance(exc_info.value.__cause__, ValueError)
    conn2.close()


def test_bad_row_arity_and_type_errors_fail_the_batch(tmpdir):
    src = os.path.join(str(tmpdir), "b.csv")
    db = os.path.join(str(tmpdir), "b.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n2,b,1,notanumber\n")
    conn = sqlite3.connect(db)
    with pytest.raises(RuntimeError):
        load_csv_to_sqlite(
            src, conn, "out", SCHEMA, passthrough_rule, batch_size=10, rule_id="b"
        )
    # whole batch rolled back, nothing committed
    assert _rows(conn) == []
    conn.close()


def test_binary_stream_input_and_no_close_by_loader(tmpdir):
    src = os.path.join(str(tmpdir), "bin.csv")
    db = os.path.join(str(tmpdir), "bin.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n2,b,0,2.0\n")
    conn = sqlite3.connect(db)
    stream = open(src, "rb")
    try:
        res = load_csv_to_sqlite(
            stream,
            conn,
            "out",
            SCHEMA,
            passthrough_rule,
            rule_id="bin",
            source_name="named-input.csv",
        )
        assert res.rows_inserted == 2
        assert not stream.closed  # loader must not close caller-owned stream
    finally:
        stream.close()
    conn.close()


def test_blank_lines_vs_empty_field_records(tmpdir):
    src = os.path.join(str(tmpdir), "bl.csv")
    db = os.path.join(str(tmpdir), "bl.db")
    # fully blank line -> no record; quoted empty fields -> a real record
    _write(src, 'id,name,active,score\n1,a,1,1.0\n\n2,b,0,2.0\n3,"","",""\n')
    conn = sqlite3.connect(db)

    def rule(rec):
        # a short/empty-field record is still a real logical record but
        # produces no output rows
        if len(rec) < 4 or rec[0] == "3":
            return []
        return [(int(rec[0]), rec[1], rec[2] == "1", float(rec[3]))]

    res = load_csv_to_sqlite(src, conn, "out", SCHEMA, rule, rule_id="bl")
    # csv.reader yields the blank physical line as an empty-string record;
    # either way every parsed record advances logical progress and the
    # zero-output record (id "3") is committed even though it inserts nothing
    assert res.logical_loaded == 4
    assert res.rows_inserted == 2
    assert [r[0] for r in _rows(conn)] == [1, 2]
    assert _state(conn)[0][0] == 4
    conn.close()


def test_connection_is_left_open_and_usable_after_load(tmpdir):
    src = os.path.join(str(tmpdir), "co.csv")
    db = os.path.join(str(tmpdir), "co.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(src, conn, "out", SCHEMA, passthrough_rule, rule_id="co")
    # connection still usable, no dangling transaction
    assert conn.in_transaction is False
    conn.execute("SELECT 1").fetchall()
    conn.close()


def test_invalid_batch_size(tmpdir):
    src = os.path.join(str(tmpdir), "ib.csv")
    db = os.path.join(str(tmpdir), "ib.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        load_csv_to_sqlite(
            src, conn, "out", SCHEMA, passthrough_rule, batch_size=0, rule_id="ib"
        )
    conn.close()


def test_resume_skips_prefix_even_with_smaller_batch_size(tmpdir):
    src = os.path.join(str(tmpdir), "rs.csv")
    db = os.path.join(str(tmpdir), "rs.db")
    _write(
        src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(1, 11))
    )
    conn = sqlite3.connect(db)
    load_csv_to_sqlite(
        src, conn, "out", SCHEMA, passthrough_rule, batch_size=10, rule_id="rs"
    )
    # interrupt simulation: truncate progress manually to 6 committed records
    conn.execute(f"UPDATE {STATE_TABLE} SET finished=0, logical_offset=6")
    conn.execute("DELETE FROM out WHERE id > 6")
    conn.commit()

    seen = []

    def tracking(rec):
        seen.append(int(rec[0]))
        return [(int(rec[0]), rec[1], True, 1.0)]

    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, tracking, batch_size=2, rule_id="rs"
    )
    assert seen == [7, 8, 9, 10]  # prefix skipped, no gap no duplicate
    assert res.logical_loaded == 4
    assert [r[0] for r in _rows(conn)] == list(range(1, 11))
    assert _state(conn)[0] == (10, 1, 11)
    conn.close()


def test_custom_csv_dialect_object(tmpdir):
    class PipeDialect(csv.Dialect):
        delimiter = "|"
        quotechar = '"'
        doublequote = True
        skipinitialspace = False
        lineterminator = "\r\n"
        quoting = csv.QUOTE_MINIMAL

    src = os.path.join(str(tmpdir), "pipe.csv")
    db = os.path.join(str(tmpdir), "pipe.db")
    _write(src, "id|name|active|score\n1|a|1|1.0\n2|b|0|2.0\n")
    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src,
        conn,
        "out",
        SCHEMA,
        passthrough_rule,
        csv_dialect=PipeDialect,
        rule_id="pipe",
    )
    assert res.rows_inserted == 2
    assert [r[0] for r in _rows(conn)] == [1, 2]
    conn.close()


def test_bytesio_like_stream_is_supported(tmpdir):
    db = os.path.join(str(tmpdir), "bio.db")
    conn = sqlite3.connect(db)
    stream = io.BytesIO(b"id,name,active,score\n1,a,1,1.0\n")
    res = load_csv_to_sqlite(
        stream, conn, "out", SCHEMA, passthrough_rule, rule_id="bio"
    )
    assert res.rows_inserted == 1
    assert _rows(conn) == [(1, "a", True, 1.0)]
    conn.close()


def test_all_records_filtered_commits_progress_without_rows(tmpdir):
    src = os.path.join(str(tmpdir), "af.csv")
    db = os.path.join(str(tmpdir), "af.db")
    _write(
        src, "id,name,active,score\n" + "".join(f"{i},n,1,1.0\n" for i in range(1, 7))
    )

    def drop_all(rec):
        return []

    conn = sqlite3.connect(db)
    res = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, drop_all, batch_size=2, rule_id="af"
    )
    # progress is measured in logical records even when every batch is empty
    assert res.logical_loaded == 6
    assert res.rows_inserted == 0
    assert _rows(conn) == []
    assert _state(conn)[0] == (6, 1, 7)
    # rerun stays a no-op with no inserts
    res2 = load_csv_to_sqlite(
        src, conn, "out", SCHEMA, drop_all, batch_size=2, rule_id="af"
    )
    assert res2.already_finished is True
    assert _rows(conn) == []
    conn.close()


def test_recovery_after_hard_process_crash(tmpdir):
    """Simulate kill -9 with os._exit while a batch transaction is open."""
    src = os.path.join(str(tmpdir), "crash.csv")
    db = os.path.join(str(tmpdir), "crash.db")
    script = os.path.join(str(tmpdir), "crash_worker.py")
    _write(
        src,
        "id,name,active,score\n" + "".join(f"{i},n,1,{i}.0\n" for i in range(1, 21)),
    )
    _write(
        script,
        "import os, sqlite3, sys\n"
        "from triad.utils.sqlite_loader import load_csv_to_sqlite\n"
        "csv_path, db_path = sys.argv[1], sys.argv[2]\n"
        "def rule(rec):\n"
        "    if int(rec[0]) == 8:\n"
        "        os._exit(17)  # hard exit: no cleanup, like SIGKILL\n"
        "    return [{'id': int(rec[0]), 'name': rec[1],\n"
        "             'active': True, 'score': float(rec[3])}]\n"
        "conn = sqlite3.connect(db_path)\n"
        "load_csv_to_sqlite(csv_path, conn, 'out',\n"
        "    'id:long,name:str,active:bool,score:double', rule,\n"
        "    batch_size=5, rule_id='crash-rule', load_key='crash-job')\n",
    )

    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-I", script, src, db],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 17, proc.stderr

    # SQLite must have rolled back the open transaction on its own; only the
    # first committed batch (records 1-5) is visible.
    conn = sqlite3.connect(db)
    assert _state(conn, "crash-job") == (5, 0, 21)
    assert [r[0] for r in _rows(conn)] == [1, 2, 3, 4, 5]

    # resume in a new process/connection: no gap, no duplicate
    res = load_csv_to_sqlite(
        src,
        conn,
        "out",
        "id:long,name:str,active:bool,score:double",
        passthrough_rule,
        batch_size=5,
        rule_id="crash-rule",
        load_key="crash-job",
    )
    assert res.resumed is True
    assert res.logical_loaded == 15
    assert [r[0] for r in _rows(conn)] == list(range(1, 21))
    assert _state(conn, "crash-job") == (20, 1, 21)

    # a further run is idempotent
    again = load_csv_to_sqlite(
        src,
        conn,
        "out",
        "id:long,name:str,active:bool,score:double",
        passthrough_rule,
        batch_size=5,
        rule_id="crash-rule",
        load_key="crash-job",
    )
    assert again.already_finished is True
    assert len(_rows(conn)) == 20
    conn.close()


def test_type_conversions_for_all_supported_affinities(tmpdir):
    import datetime as dt

    from triad.collections.schema import Schema as _Schema
    from triad.utils.sqlite_loader import _convert_value

    def conv(value, type_expr):
        field = _Schema(type_expr).fields[0]
        return _convert_value(value, field)

    assert conv("yes", "b:bool") is True
    assert conv("FALSE", "b:bool") is False
    assert conv("", "b:bool") is False
    assert conv(True, "b:bool") is True
    assert conv(5, "b:bool") is True
    with pytest.raises(TypeError, match="to bool"):
        conv("maybe", "b:bool")
    assert conv(True, "i:long") == 1
    assert conv("42", "i:int") == 42
    with pytest.raises(TypeError):
        conv("x", "i:long")
    assert conv("1.5", "d:double") == 1.5
    assert conv(7, "s:str") == "7"
    assert conv("abc", "bin:binary") == b"abc"
    assert conv(b"raw", "bin:binary") == b"raw"
    day = dt.date(2026, 11, 18)
    assert conv(day, "d:date") == "2026-11-18"
    assert conv("2026-11-18", "d:date") == "2026-11-18"
    moment = dt.datetime(2026, 11, 18, 9, 30, 0)
    assert conv(moment, "ts:datetime") == "2026-11-18 09:30:00"
    assert conv("2026-11-18 09:30:00", "ts:datetime") == "2026-11-18 09:30:00"
    assert conv("12.345", "amt:decimal(10,3)") == "12.345"
    assert conv(None, "i:long") is None
    assert conv("anything", "n:null") is None


def test_rule_with_uuid_method_uses_its_identity(tmpdir):
    src = os.path.join(str(tmpdir), "u.csv")
    db = os.path.join(str(tmpdir), "u.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")

    class UuidRule:
        def __uuid__(self):
            return "rule-version-7"

        def __call__(self, rec):
            return [(int(rec[0]), rec[1], True, 1.0)]

    conn = sqlite3.connect(db)
    rule = UuidRule()
    load_csv_to_sqlite(src, conn, "out", SCHEMA, rule, load_key="uuid-job")
    stored = conn.execute(
        f"SELECT rule_id FROM {STATE_TABLE} WHERE load_key='uuid-job'"
    ).fetchone()[0]
    assert stored == "uuid:rule-version-7"

    # another instance reporting the same __uuid__ is treated as the same rule
    same = UuidRule()
    res = load_csv_to_sqlite(src, conn, "out", SCHEMA, same, load_key="uuid-job")
    assert res.already_finished is True
    conn.close()


def test_unsupported_column_type_is_rejected(tmpdir):
    src = os.path.join(str(tmpdir), "r.csv")
    db = os.path.join(str(tmpdir), "r.db")
    _write(src, "x\n1\n")
    conn = sqlite3.connect(db)
    with pytest.raises(NotImplementedError, match="does not support type"):
        load_csv_to_sqlite(
            src,
            conn,
            "out",
            "x:[long]",
            lambda rec: [rec],
            rule_id="unsupported",
        )
    conn.close()


def test_existing_table_with_mismatched_columns_is_rejected(tmpdir):
    src = os.path.join(str(tmpdir), "m.csv")
    db = os.path.join(str(tmpdir), "m.db")
    _write(src, "id,name,active,score\n1,a,1,1.0\n")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE out (other TEXT)")
    conn.commit()
    with pytest.raises(ValueError, match="do not match declared schema"):
        load_csv_to_sqlite(
            src, conn, "out", SCHEMA, passthrough_rule, rule_id="mismatch"
        )
    conn.close()


def test_non_seekable_stream_is_rejected(tmpdir):
    db = os.path.join(str(tmpdir), "ns.db")
    conn = sqlite3.connect(db)

    class NoSeek:
        def read(self, n=-1):
            return b""

    with pytest.raises(ValueError, match="seekable binary stream"):
        load_csv_to_sqlite(
            NoSeek(), conn, "out", SCHEMA, passthrough_rule, rule_id="ns"
        )
    conn.close()
