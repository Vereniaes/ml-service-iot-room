# ml-service/app/database/connection.py
#
# -> koneksi ke PostgreSQL database
#    -> pakai psycopg2 langsung (bukan ORM, biar simple dan ga bergantung ke Drizzle)
#    -> tabel yang dipakai: face_embeddings (baru), users (existing - read only)
# -> koneksi dibuat sekali saat startup dan di-reuse (connection pool via context manager)

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ambil DATABASE_URL dari environment
# format: postgresql://user:password@host:5432/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "")


# helper ---------------------------------------------------------------------------------

# fungsi buat buka koneksi ke database
# output : psycopg2 connection object
# note   : selalu pakai dengan context manager (with get_connection() as conn)
def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL tidak ada di environment")
    conn = psycopg2.connect(DATABASE_URL)
    return conn


# fungsi buat execute query dan return semua rows sebagai list of dict
# input param : query  -> SQL query string
#               params -> tuple parameter (default None)
# output : list of dict { column_name: value }
def fetch_all(query: str, params: tuple = None) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


# fungsi buat execute query INSERT/UPDATE/DELETE dan return satu row
# input param : query  -> SQL query string
#               params -> tuple parameter
# output : dict hasil returning atau None
def execute_returning(query: str, params: tuple) -> dict | None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            conn.commit()
            row = cur.fetchone()
            return dict(row) if row else None


# fungsi buat execute batch INSERT (tanpa return)
# input param : query  -> SQL query string
#               params -> list of tuples
def execute_batch(query: str, params: list[tuple]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, query, params)
        conn.commit()

# end of helper --------------------------------------------------------------------------
