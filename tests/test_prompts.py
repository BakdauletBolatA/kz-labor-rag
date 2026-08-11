"""Реестр промптов.

Требование проекта: промпт меняется только вместе с бампом версии. Эти тесты
проверяют, что требование обеспечено механически, а не честным словом.
"""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.eval import prompts as P


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Изолированная директория промптов, чтобы не трогать реестр репозитория."""
    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(P, "REGISTRY_PATH", tmp_path / "REGISTRY.json")
    (tmp_path / "REGISTRY.json").write_text("{}", encoding="utf-8")
    return tmp_path


def write(sandbox, name: str, text: str) -> None:
    (sandbox / name).write_text(text, encoding="utf-8")


class TestRegistry:
    def test_registered_prompt_loads(self, sandbox):
        write(sandbox, "judge.v1.txt", "оцени {answer}")
        P.register_prompt("judge", "v1")

        prompt = P.load_prompt("judge", "v1")
        assert prompt.label == "judge@v1"
        assert prompt.text == "оцени {answer}"
        assert len(prompt.sha256) == 64

    def test_edit_without_version_bump_is_fatal(self, sandbox):
        write(sandbox, "judge.v1.txt", "оцени {answer}")
        P.register_prompt("judge", "v1")

        # Ровно тот случай, который ломает сравнимость итераций:
        # формулировку «просто уточнили», версию не тронули.
        write(sandbox, "judge.v1.txt", "оцени очень строго {answer}")

        with pytest.raises(P.PromptRegistryError, match="изменён без бампа версии"):
            P.load_prompt("judge", "v1")

    def test_unregistered_prompt_refuses_to_load(self, sandbox):
        write(sandbox, "judge.v1.txt", "оцени {answer}")
        with pytest.raises(P.PromptRegistryError, match="не зарегистрирован"):
            P.load_prompt("judge", "v1")

    def test_missing_file_says_new_version_is_a_new_file(self, sandbox):
        with pytest.raises(P.PromptRegistryError, match="новый файл"):
            P.load_prompt("judge", "v7")

    def test_registered_version_is_immutable(self, sandbox):
        write(sandbox, "judge.v1.txt", "первый вариант")
        P.register_prompt("judge", "v1")
        write(sandbox, "judge.v1.txt", "второй вариант")

        with pytest.raises(P.PromptRegistryError, match="Версии неизменяемы"):
            P.register_prompt("judge", "v1")

    def test_versions_are_independent(self, sandbox):
        write(sandbox, "judge.v1.txt", "первый")
        write(sandbox, "judge.v2.txt", "второй")
        P.register_prompt("judge", "v1")
        P.register_prompt("judge", "v2")

        assert P.load_prompt("judge", "v1").text == "первый"
        assert P.load_prompt("judge", "v2").text == "второй"

    def test_generator_and_judge_version_separately(self, sandbox):
        write(sandbox, "answer.v1.txt", "ответь")
        write(sandbox, "faithfulness.v3.txt", "оцени")
        P.register_prompt("answer", "v1")
        P.register_prompt("faithfulness", "v3")

        registry = json.loads((sandbox / "REGISTRY.json").read_text(encoding="utf-8"))
        assert set(registry) == {"answer", "faithfulness"}
        assert list(registry["answer"]) == ["v1"]
        assert list(registry["faithfulness"]) == ["v3"]


class TestRepositoryPrompts:
    """Промпты, реально лежащие в репозитории, обязаны быть валидны."""

    def test_judge_prompt_is_registered_and_intact(self):
        prompt = P.load_prompt("faithfulness_ru", "v1")
        for placeholder in ("{question}", "{context}", "{answer}"):
            assert placeholder in prompt.text

    def test_generator_prompt_is_registered_and_intact(self):
        prompt = P.load_prompt("answer_ru", "v1")
        assert "{question}" in prompt.text
        assert "{context}" in prompt.text
