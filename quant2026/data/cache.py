"""Local cache layer: SQLite metadata + Parquet storage."""

import sqlite3
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from .base import DataProvider

_DEFAULT_CACHE_DIR = Path.home() / ".quant2026" / "cache"
_INDUSTRY_TTL_DAYS = 7


class CachedProvider(DataProvider):
    """Decorator that caches any DataProvider with SQLite index + Parquet files."""

    def __init__(self, upstream: DataProvider, cache_dir: Path | str | None = None):
        self._upstream = upstream
        self._dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        for sub in ("quotes", "index_quotes", "financials", "stock_list", "industry"):
            (self._dir / sub).mkdir(exist_ok=True)
        self._db_path = self._dir / "meta.db"
        self._init_db()

    # ── DB setup ────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS quotes_range (
                    stock_code TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date   TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (stock_code)
                );
                CREATE TABLE IF NOT EXISTS index_range (
                    index_code TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date   TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (index_code)
                );
                CREATE TABLE IF NOT EXISTS stock_list_meta (
                    cache_date TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS financial_meta (
                    stock_code  TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (stock_code, report_date)
                );
                CREATE TABLE IF NOT EXISTS industry_meta (
                    id         INTEGER PRIMARY KEY CHECK (id = 1),
                    updated_at TEXT NOT NULL
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    # ── Parquet helpers ─────────────────────────────────────────

    @staticmethod
    def _read_pq(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    @staticmethod
    def _write_pq(df: pd.DataFrame, path: Path):
        df.to_parquet(path, index=False)

    # ── get_daily_quotes ────────────────────────────────────────

    def get_daily_quotes(
        self, stock_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch daily quotes with incremental caching per stock."""
        all_frames: list[pd.DataFrame] = []

        with self._conn() as conn:
            for code in stock_codes:
                pq = self._dir / "quotes" / f"{code}.parquet"
                row = conn.execute(
                    "SELECT start_date, end_date FROM quotes_range WHERE stock_code=?",
                    (code,),
                ).fetchone()

                if row:
                    cached_start, cached_end = date.fromisoformat(row[0]), date.fromisoformat(row[1])
                    # Determine missing ranges
                    gaps: list[tuple[date, date]] = []
                    if start < cached_start:
                        gaps.append((start, cached_start - timedelta(days=1)))
                    if end > cached_end:
                        gaps.append((cached_end + timedelta(days=1), end))

                    cached = self._read_pq(pq)
                    new_frames = [cached]
                    for gs, ge in gaps:
                        logger.debug(f"Cache miss {code} [{gs}..{ge}]")
                        fetched = self._upstream.get_daily_quotes([code], gs, ge)
                        if not fetched.empty:
                            new_frames.append(fetched)

                    if gaps:
                        merged = pd.concat(new_frames, ignore_index=True).drop_duplicates(
                            subset=["stock_code", "date"]
                        )
                        self._write_pq(merged, pq)
                        new_start = min(start, cached_start)
                        new_end = max(end, cached_end)
                        conn.execute(
                            "UPDATE quotes_range SET start_date=?, end_date=?, updated_at=? WHERE stock_code=?",
                            (new_start.isoformat(), new_end.isoformat(), self._now(), code),
                        )
                    else:
                        merged = cached

                else:
                    logger.debug(f"Cache miss {code} (full)")
                    fetched = self._upstream.get_daily_quotes([code], start, end)
                    if not fetched.empty:
                        self._write_pq(fetched, pq)
                    merged = fetched
                    conn.execute(
                        "INSERT INTO quotes_range VALUES (?,?,?,?)",
                        (code, start.isoformat(), end.isoformat(), self._now()),
                    )

                if not merged.empty:
                    mask = merged["date"].apply(
                        lambda d: start <= (d if isinstance(d, date) else date.fromisoformat(str(d)[:10])) <= end
                    )
                    all_frames.append(merged[mask])

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # ── get_index_quotes ────────────────────────────────────────

    def get_index_quotes(
        self, index_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Same incremental logic as daily quotes, for indices."""
        all_frames: list[pd.DataFrame] = []

        with self._conn() as conn:
            for code in index_codes:
                pq = self._dir / "index_quotes" / f"{code}.parquet"
                row = conn.execute(
                    "SELECT start_date, end_date FROM index_range WHERE index_code=?",
                    (code,),
                ).fetchone()

                if row:
                    cached_start, cached_end = date.fromisoformat(row[0]), date.fromisoformat(row[1])
                    gaps: list[tuple[date, date]] = []
                    if start < cached_start:
                        gaps.append((start, cached_start - timedelta(days=1)))
                    if end > cached_end:
                        gaps.append((cached_end + timedelta(days=1), end))

                    cached = self._read_pq(pq)
                    new_frames = [cached]
                    for gs, ge in gaps:
                        fetched = self._upstream.get_index_quotes([code], gs, ge)
                        if not fetched.empty:
                            new_frames.append(fetched)

                    if gaps:
                        merged = pd.concat(new_frames, ignore_index=True).drop_duplicates()
                        self._write_pq(merged, pq)
                        conn.execute(
                            "UPDATE index_range SET start_date=?, end_date=?, updated_at=? WHERE index_code=?",
                            (min(start, cached_start).isoformat(), max(end, cached_end).isoformat(), self._now(), code),
                        )
                    else:
                        merged = cached
                else:
                    fetched = self._upstream.get_index_quotes([code], start, end)
                    if not fetched.empty:
                        self._write_pq(fetched, pq)
                    merged = fetched
                    conn.execute(
                        "INSERT INTO index_range VALUES (?,?,?,?)",
                        (code, start.isoformat(), end.isoformat(), self._now()),
                    )

                if not merged.empty:
                    all_frames.append(merged)

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # ── get_stock_list ──────────────────────────────────────────

    def get_stock_list(self, market_date: date | None = None) -> pd.DataFrame:
        """Cache stock list per day (at most one upstream call per day)."""
        cache_date = (market_date or date.today()).isoformat()
        pq = self._dir / "stock_list" / f"{cache_date}.parquet"

        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM stock_list_meta WHERE cache_date=?", (cache_date,)
            ).fetchone()

            if row and pq.exists():
                logger.debug(f"stock_list cache hit {cache_date}")
                return pd.read_parquet(pq)

            logger.debug(f"stock_list cache miss {cache_date}")
            df = self._upstream.get_stock_list(market_date)
            if not df.empty:
                self._write_pq(df, pq)
            conn.execute(
                "INSERT OR REPLACE INTO stock_list_meta VALUES (?,?)",
                (cache_date, self._now()),
            )
        return df

    # ── get_financial_data ──────────────────────────────────────

    def get_financial_data(
        self, stock_codes: list[str], report_date: date
    ) -> pd.DataFrame:
        """Cache financial data per stock_code + report_date."""
        rd = report_date.isoformat()
        to_fetch: list[str] = []
        cached_frames: list[pd.DataFrame] = []

        with self._conn() as conn:
            for code in stock_codes:
                pq = self._dir / "financials" / f"{code}_{rd}.parquet"
                row = conn.execute(
                    "SELECT 1 FROM financial_meta WHERE stock_code=? AND report_date=?",
                    (code, rd),
                ).fetchone()
                if row and pq.exists():
                    cached_frames.append(pd.read_parquet(pq))
                else:
                    to_fetch.append(code)

            if to_fetch:
                logger.debug(f"financial cache miss: {to_fetch} @ {rd}")
                fetched = self._upstream.get_financial_data(to_fetch, report_date)
                if not fetched.empty:
                    for code, grp in fetched.groupby("stock_code"):
                        pq = self._dir / "financials" / f"{code}_{rd}.parquet"
                        self._write_pq(grp, pq)
                        conn.execute(
                            "INSERT OR REPLACE INTO financial_meta VALUES (?,?,?)",
                            (code, rd, self._now()),
                        )
                    cached_frames.append(fetched)

        return pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()

    # ── get_industry_classification ─────────────────────────────

    def get_industry_classification(self) -> pd.DataFrame:
        """Cache with TTL (default 7 days)."""
        pq = self._dir / "industry" / "classification.parquet"

        with self._conn() as conn:
            row = conn.execute(
                "SELECT updated_at FROM industry_meta WHERE id=1"
            ).fetchone()

            if row and pq.exists():
                updated = datetime.fromisoformat(row[0])
                if datetime.now() - updated < timedelta(days=_INDUSTRY_TTL_DAYS):
                    logger.debug("industry cache hit")
                    return pd.read_parquet(pq)

            logger.debug("industry cache miss / expired")
            df = self._upstream.get_industry_classification()
            if not df.empty:
                self._write_pq(df, pq)
            conn.execute(
                "INSERT OR REPLACE INTO industry_meta VALUES (1, ?)", (self._now(),)
            )
        return df

    # ── Cache management ────────────────────────────────────────

    def invalidate(self, stock_code: str | None = None):
        """Remove cache entries. If stock_code given, only that stock; else all."""
        with self._conn() as conn:
            if stock_code:
                conn.execute("DELETE FROM quotes_range WHERE stock_code=?", (stock_code,))
                conn.execute("DELETE FROM financial_meta WHERE stock_code=?", (stock_code,))
                for pq in self._dir.rglob(f"{stock_code}*"):
                    pq.unlink(missing_ok=True)
                logger.info(f"Invalidated cache for {stock_code}")
            else:
                for tbl in ("quotes_range", "index_range", "stock_list_meta", "financial_meta", "industry_meta"):
                    conn.execute(f"DELETE FROM {tbl}")
                for sub in ("quotes", "index_quotes", "financials", "stock_list", "industry"):
                    for f in (self._dir / sub).iterdir():
                        f.unlink(missing_ok=True)
                logger.info("Invalidated all cache")

    def clear(self):
        """Remove entire cache directory."""
        shutil.rmtree(self._dir, ignore_errors=True)
        logger.info(f"Cleared cache dir {self._dir}")
