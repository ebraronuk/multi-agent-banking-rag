"""db/schema.sql'i DATABASE_URL'e karşı uygular.

`python scripts/seed_postgres.py` olarak çalıştırılır (`make seed-db`).
Docker'sız lokal Postgres kullanımı içindir — docker-compose'daki postgres
servisi aynı dosyayı zaten kendi ilk açılışında otomatik uyguluyor. Dosyadaki
her ifade idempotent, yeniden çalıştırmak sorun çıkarmaz.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


async def main() -> None:
    import asyncpg

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL ayarlı değil — uygulanacak bir şey yok. Bkz. .env.example.")
        return

    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()

    print(f"{_SCHEMA_PATH.relative_to(_SCHEMA_PATH.parent.parent)} DATABASE_URL'e uygulandı.")


if __name__ == "__main__":
    asyncio.run(main())
