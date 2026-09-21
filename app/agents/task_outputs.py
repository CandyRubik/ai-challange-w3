from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


ShortText = Annotated[str, Field(min_length=1, max_length=1_000)]


class TaskModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class PlanOutput(TaskModelOutput):
    summary: ShortText
    plan: Annotated[list[ShortText], Field(min_length=1, max_length=8)]
    criteria: Annotated[list[ShortText], Field(min_length=1, max_length=8)]


class ValidationOutput(TaskModelOutput):
    passed: bool
    report: Annotated[str, Field(min_length=1, max_length=12_000)]
    repair_steps: Annotated[list[ShortText], Field(max_length=4)]

    @model_validator(mode="after")
    def check_repair_steps(self) -> "ValidationOutput":
        if self.passed and self.repair_steps:
            raise ValueError("Успешная проверка не должна содержать исправления")
        if not self.passed and not self.repair_steps:
            raise ValueError("Неуспешная проверка должна содержать шаги исправления")
        return self
