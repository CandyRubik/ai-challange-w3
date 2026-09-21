from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Protocol
from uuid import uuid4

from ..schemas import UserProfile, UserProfileCreateRequest, UserProfileUpdateRequest
from ..state.onboarding import OnboardingState


DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "Основной"


class ProfileNotFound(LookupError):
    pass


class ProfileDeletionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredProfile:
    id: str
    name: str
    description: str
    language: str
    tone: str
    detail_level: str
    response_format: str
    constraints: tuple[str, ...]
    onboarding: OnboardingState
    created_at: datetime
    updated_at: datetime

    @property
    def onboarding_step(self) -> int:
        return self.onboarding.step

    @property
    def onboarding_complete(self) -> bool:
        return self.onboarding.complete


class ProfileRepository(Protocol):
    def create(self, request: UserProfileCreateRequest) -> StoredProfile: ...

    def list(self) -> list[StoredProfile]: ...

    def create_auto(self) -> StoredProfile: ...

    def get(self, profile_id: str) -> StoredProfile: ...

    def update(
        self,
        profile_id: str,
        request: UserProfileUpdateRequest,
    ) -> StoredProfile: ...

    def delete(self, profile_id: str) -> None: ...

    def apply_interview_update(
        self,
        profile_id: str,
        values: dict[str, object],
        *,
        next_step: int,
        complete: bool,
    ) -> StoredProfile: ...


def ensure_profile_schema(connection: sqlite3.Connection) -> None:
    """Create the profile table and its stable migration target."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'ru',
            tone TEXT NOT NULL DEFAULT 'neutral'
                CHECK (tone IN ('neutral', 'friendly', 'formal', 'technical')),
            detail_level TEXT NOT NULL DEFAULT 'balanced'
                CHECK (detail_level IN ('brief', 'balanced', 'detailed')),
            response_format TEXT NOT NULL DEFAULT 'plain'
                CHECK (response_format IN ('plain', 'bullets', 'steps')),
            constraints_json TEXT NOT NULL DEFAULT '[]',
            onboarding_step INTEGER NOT NULL DEFAULT 0,
            onboarding_complete INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(user_profiles)")
    }
    added_onboarding_state = "onboarding_complete" not in columns
    if "onboarding_step" not in columns:
        connection.execute(
            "ALTER TABLE user_profiles ADD COLUMN onboarding_step INTEGER NOT NULL DEFAULT 0",
        )
    if "onboarding_complete" not in columns:
        connection.execute(
            "ALTER TABLE user_profiles ADD COLUMN onboarding_complete INTEGER NOT NULL DEFAULT 0",
        )
    if added_onboarding_state:
        connection.execute(
            """
            UPDATE user_profiles
            SET onboarding_step = 3, onboarding_complete = 1
            WHERE id != ? OR description != '' OR language != 'ru'
               OR tone != 'neutral' OR detail_level != 'balanced'
               OR response_format != 'plain' OR constraints_json != '[]'
            """,
            (DEFAULT_PROFILE_ID,),
        )

    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        """
        INSERT OR IGNORE INTO user_profiles
            (id, name, description, language, tone, detail_level,
             response_format, constraints_json, onboarding_step,
             onboarding_complete, created_at, updated_at)
        VALUES (?, ?, '', 'ru', 'neutral', 'balanced', 'plain', '[]', 0, 0, ?, ?)
        """,
        (DEFAULT_PROFILE_ID, DEFAULT_PROFILE_NAME, now, now),
    )


class SQLiteProfileRepository:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            ensure_profile_schema(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @classmethod
    def _stored(cls, row: sqlite3.Row) -> StoredProfile:
        constraints = json.loads(row["constraints_json"])
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            constraints = []
        return StoredProfile(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            language=row["language"],
            tone=row["tone"],
            detail_level=row["detail_level"],
            response_format=row["response_format"],
            constraints=tuple(constraints),
            onboarding=OnboardingState(
                step=int(row["onboarding_step"]),
                complete=bool(row["onboarding_complete"]),
            ),
            created_at=cls._datetime(row["created_at"]),
            updated_at=cls._datetime(row["updated_at"]),
        )

    def create(self, request: UserProfileCreateRequest) -> StoredProfile:
        profile_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles
                    (id, name, description, language, tone, detail_level,
                     response_format, constraints_json, onboarding_step,
                     onboarding_complete, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 3, 1, ?, ?)
                """,
                (
                    profile_id,
                    request.name,
                    request.description,
                    request.language,
                    request.tone,
                    request.detail_level,
                    request.response_format,
                    json.dumps(request.constraints, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get(profile_id)

    def create_auto(self) -> StoredProfile:
        profile_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            sequence = connection.execute(
                "SELECT COUNT(*) FROM user_profiles",
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO user_profiles
                    (id, name, description, language, tone, detail_level,
                     response_format, constraints_json, onboarding_step,
                     onboarding_complete, created_at, updated_at)
                VALUES (?, ?, '', 'ru', 'neutral', 'balanced', 'plain', '[]',
                        0, 0, ?, ?)
                """,
                (profile_id, f"Пользователь {sequence + 1}", timestamp, timestamp),
            )
        return self.get(profile_id)

    def list(self) -> list[StoredProfile]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_profiles
                ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END, created_at, id
                """,
                (DEFAULT_PROFILE_ID,),
            ).fetchall()
        return [self._stored(row) for row in rows]

    def get(self, profile_id: str) -> StoredProfile:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM user_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
        if row is None:
            raise ProfileNotFound(profile_id)
        return self._stored(row)

    def update(
        self,
        profile_id: str,
        request: UserProfileUpdateRequest,
    ) -> StoredProfile:
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE user_profiles
                SET name = ?, description = ?, language = ?, tone = ?,
                    detail_level = ?, response_format = ?, constraints_json = ?,
                    onboarding_step = 3, onboarding_complete = 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    request.name,
                    request.description,
                    request.language,
                    request.tone,
                    request.detail_level,
                    request.response_format,
                    json.dumps(request.constraints, ensure_ascii=False),
                    self._timestamp(now),
                    profile_id,
                ),
            )
        if cursor.rowcount == 0:
            raise ProfileNotFound(profile_id)
        return self.get(profile_id)

    def delete(self, profile_id: str) -> None:
        if profile_id == DEFAULT_PROFILE_ID:
            raise ProfileDeletionError("Основной профиль нельзя удалить")

        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM user_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if exists is None:
                raise ProfileNotFound(profile_id)

            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'",
                )
            }
            if "chat_sessions" in tables:
                connection.execute(
                    "DELETE FROM chat_sessions WHERE profile_id = ?",
                    (profile_id,),
                )
            if "long_term_memory" in tables:
                connection.execute(
                    "DELETE FROM long_term_memory WHERE profile_id = ?",
                    (profile_id,),
                )
            connection.execute(
                "DELETE FROM user_profiles WHERE id = ?",
                (profile_id,),
            )

    def apply_interview_update(
        self,
        profile_id: str,
        values: dict[str, object],
        *,
        next_step: int,
        complete: bool,
    ) -> StoredProfile:
        allowed = {
            "name",
            "description",
            "language",
            "tone",
            "detail_level",
            "response_format",
            "constraints",
        }
        if not set(values).issubset(allowed):
            raise ValueError("Unsupported inferred profile field")

        assignments: list[str] = []
        parameters: list[object] = []
        for key in (
            "name",
            "description",
            "language",
            "tone",
            "detail_level",
            "response_format",
            "constraints",
        ):
            if key not in values:
                continue
            column = "constraints_json" if key == "constraints" else key
            value = values[key]
            if key == "constraints":
                value = json.dumps(list(value), ensure_ascii=False)
            assignments.append(f"{column} = ?")
            parameters.append(value)

        assignments.extend(
            ["onboarding_step = ?", "onboarding_complete = ?", "updated_at = ?"],
        )
        parameters.extend(
            [next_step, int(complete), self._timestamp(datetime.now(timezone.utc)), profile_id],
        )
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE user_profiles SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
        if cursor.rowcount == 0:
            raise ProfileNotFound(profile_id)
        return self.get(profile_id)


class ProfileService:
    def __init__(self, repository: ProfileRepository) -> None:
        self._repository = repository

    @staticmethod
    def _profile(profile: StoredProfile) -> UserProfile:
        return UserProfile(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            language=profile.language,
            tone=profile.tone,
            detail_level=profile.detail_level,
            response_format=profile.response_format,
            constraints=list(profile.constraints),
            onboarding_step=profile.onboarding_step,
            onboarding_complete=profile.onboarding_complete,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    def create(self, request: UserProfileCreateRequest) -> UserProfile:
        return self._profile(self._repository.create(request))

    def list(self) -> list[UserProfile]:
        return [self._profile(profile) for profile in self._repository.list()]

    def create_auto(self) -> UserProfile:
        return self._profile(self._repository.create_auto())

    def get(self, profile_id: str) -> UserProfile:
        return self._profile(self._repository.get(profile_id))

    def update(
        self,
        profile_id: str,
        request: UserProfileUpdateRequest,
    ) -> UserProfile:
        return self._profile(self._repository.update(profile_id, request))

    def delete(self, profile_id: str) -> None:
        self._repository.delete(profile_id)
