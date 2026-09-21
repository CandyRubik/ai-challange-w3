from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    kind: str
    content: str
    created_at: datetime
    refusal: bool = False
