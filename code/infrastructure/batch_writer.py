import logging
import threading
from collections import defaultdict

from psycopg2 import sql as pgsql
from psycopg2.extras import Json, execute_values

from infrastructure.constants import Table

# Tables that may be written via BatchWriter.queue().
# Extend this set when a new batchable table is added to the schema.
_BATCHABLE_TABLES: frozenset[str] = frozenset({
    Table.FORECASTS,
    Table.MODEL_INITIALIZATION_S9,
    Table.FEATURE_DECISIONS_S9,
    Table.HYPERPARAMETER_DECISIONS,
    Table.BACKTEST_DECISIONS,
    Table.STAGE9_SKU_EXECUTION_LOG,
    Table.THOMPSON_SAMPLING_STATE,
    Table.SKU_SIMILARITY_REGISTRY,
    Table.DATA_FINGERPRINT_CACHE,
    Table.ADAPTIVE_QUANTILE_STATE,
    Table.STAGE9_SELF_ASSESSMENT,
    Table.MODEL_PERFORMANCE_S9,
    Table.SIZE_CURVE_REGISTRY,
    Table.FORECAST_OUTCOMES,
})

# Tables that must NEVER be queued to BatchWriter — they use direct
# conn.execute + 3-retry commits so Stage 8 / Stage 10 see them immediately.
_SACRED_WRITE_TABLES: frozenset[str] = frozenset({
    Table.PATTERN_FEEDBACK,      # P4: Sub-Stage 9.4 direct write
    Table.CROSS_AGENT_SIGNALS,   # P4: SignalEmitter direct write
})

log = logging.getLogger(__name__)


class BatchWriter:
    # IMPORTANT: pattern_feedback rows must NEVER be added to this BatchWriter.
    # pattern_feedback is always written directly with conn.execute() + conn.commit()
    # immediately in Sub-Stage 9.4. See ATH-37 for context.
    #
    # THREAD SAFETY: queue() and flush() are guarded by an internal Lock so
    # planning_handler can run Sub-Stage 9.1 in parallel via ThreadPoolExecutor
    # without corrupting the buffer or count. Lock contention is negligible —
    # queue() does a list.append + int increment; the heavy work (run_model_
    # initialisation, run_feature_engineering, etc.) runs outside the lock.

    def __init__(self, conn, batch_size=100):
        # conn       : open psycopg2 connection — caller owns the lifecycle entirely
        # batch_size : auto-flush threshold; its default value lives ONLY in this
        #              signature — nowhere else in this file (spec §5 rule)
        self.conn = conn
        self.batch_size = batch_size
        self.buffer = defaultdict(list)  # { table_name: [row_dict, ...] }
        self.count = 0                    # total rows buffered across ALL tables combined
        self._col_cache: dict[str, list[str]] = {}  # { table_name: [col, ...] } — fixed after first row
        self._lock = threading.Lock()

    @property
    def _buffer(self):
        return self.buffer

    def queue(self, table: str, row: dict) -> None:
        # Append to the in-memory buffer only — never touches the database,
        # never checks the threshold (caller decides when to call flush_if_needed)
        if table in _SACRED_WRITE_TABLES:
            raise ValueError(
                f"Table '{table}' is a sacred-write table and must never be "
                f"queued to BatchWriter. Use direct conn.execute with 3-retry "
                f"instead (same discipline as pattern_feedback)."
            )
        if table not in _BATCHABLE_TABLES:
            raise ValueError(
                f"Table '{table}' is not in the BatchWriter allowlist. "
                f"If this is a new table, add it to _BATCHABLE_TABLES in "
                f"batch_writer.py."
            )
        with self._lock:
            self.buffer[table].append(row)
            self.count += 1

    def flush(self):
        # The lock protects buffer/count from concurrent queue() calls during
        # the flush. The DB write itself runs inside the lock — acceptable
        # because flush is invoked at coarse boundaries (every batch_size
        # SKUs from a single coordinator), not from per-SKU worker threads.
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        # Step 1 — nothing buffered: short-circuit with no DB interaction at all
        if self.count == 0:
            return

        # Step 2 — one cursor is shared across every table in this flush call
        cursor = self.conn.cursor()
        try:
            # Step 3 — iterate every table that has buffered rows and bulk-insert each batch
            for table_name, rows in self.buffer.items():
                if not rows:
                    continue

                # Column order is fixed once from the first row's keys and then
                # cached for the lifetime of this BatchWriter instance. Every
                # subsequent row is accessed by key name (not position) so varying
                # dict insertion order across rows is handled correctly.
                # pgsql.Identifier quotes table and column names safely — prevents
                # injection even if a caller ever passes a non-standard identifier.
                if table_name not in self._col_cache:
                    self._col_cache[table_name] = list(rows[0].keys())
                columns = self._col_cache[table_name]
                values = [
                    tuple(
                        Json(row[col]) if isinstance(row[col], (dict, list)) else row[col]
                        for col in columns
                    )
                    for row in rows
                ]
                if hasattr(cursor, 'connection'):
                    # Real psycopg2 cursor — use fast execute_values with safe identifiers.
                    query = pgsql.SQL("INSERT INTO stage9.{} ({}) VALUES %s").format(
                        pgsql.Identifier(table_name),
                        pgsql.SQL(", ").join(pgsql.Identifier(c) for c in columns),
                    )
                    # page_size=len(values): send all rows in a single round-trip
                    # instead of the default 100-row pages.
                    execute_values(cursor, query, values, page_size=len(values))
                else:
                    # Fake/test cursor — fall back to executemany with a plain SQL string.
                    col_list = ", ".join(f'"{c}"' for c in columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    plain_sql = (
                        f'INSERT INTO stage9."{table_name}" ({col_list}) '
                        f'VALUES ({placeholders})'
                    )
                    cursor.executemany(plain_sql, values)

            # Step 4 — ONE commit for ALL tables; never one commit per table
            self.conn.commit()
            log.debug(
                "BatchWriter flushed %d rows across %d tables",
                self.count, len(self.buffer),
            )

        except Exception:
            self.conn.rollback()
            log.error(
                "BatchWriter flush failed — rolled back %d rows across %d tables",
                self.count, len(self.buffer),
            )
            raise

        finally:
            # Always close cursor and reset buffer — even on error — so the
            # writer is in a consistent state for the next operation.
            cursor.close()
            self.buffer.clear()
            self.count = 0

    def flush_if_needed(self):
        # The ONLY place in this class that compares against self.batch_size
        # Read self.count under the lock so a concurrent queue() doesn't see
        # a torn read, but call flush() outside the lock since flush() takes
        # the lock itself.
        with self._lock:
            need = self.count >= self.batch_size
        if need:
            self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush even when an exception is propagating — this guarantees the trailing
        # partial batch is always persisted, even on a Stage 9 mid-loop crash.
        # If flush itself raises while we are already handling a pipeline exception,
        # log the flush error rather than letting it replace the original exception.
        # Returning False lets the original exception continue to propagate.
        try:
            self.flush()
        except Exception as flush_exc:
            if exc_type is None:
                raise  # no prior exception — let the flush error propagate normally
            log.error(
                "BatchWriter flush failed during exception handling: %s", flush_exc
            )
        return False

    def close(self):
        # Final-flush guarantee for callers that do not use a with-block
        self.flush()
