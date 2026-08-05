"""Paylaşılan rate limiter örneği.

main.py'de yaşamak yerine kendi modülünde duruyor ki route modülleri (ör.
`app/api/routes/chat.py`) `main`'i import etmeden `limiter`'ı endpoint'lerini
dekore etmek için import edebilsin — `main.py` router'ları import ediyor,
yani bir route'un `main`'den geri import etmesi döngüsel bir import olurdu.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
