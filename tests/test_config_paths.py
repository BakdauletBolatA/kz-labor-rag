"""Пути из конфига разрешаются от корня репозитория.

Раньше и сам конфиг, и все пути внутри него разрешались относительно текущего
каталога, поэтому любая команда из домашнего каталога падала на «конфиг не
найден». Инструмент, который работает только из одного каталога, — это
инструмент, о котором надо помнить лишнее.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kz_labor_rag.config import Config, ConfigError, find_repo_root, load_config


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "default.yaml").write_text(
        "version: t\n"
        "eval:\n  dataset: evals/datasets/x.jsonl\n"
        "corpus:\n  raw_html: data/raw/x.html\n",
        encoding="utf-8",
    )
    (tmp_path / "deep" / "nested").mkdir(parents=True)
    return tmp_path


class TestFindRoot:
    def test_finds_root_from_nested_directory(self, fake_repo):
        assert find_repo_root(fake_repo / "deep" / "nested") == fake_repo.resolve()

    def test_returns_none_outside_repo(self, tmp_path):
        assert find_repo_root(tmp_path) is None


class TestLoadFromAnywhere:
    def test_loads_when_cwd_is_repo_root(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo)
        assert load_config().version == "t"

    def test_loads_from_nested_directory(self, fake_repo, monkeypatch):
        # Ровно тот случай, который падал: запуск не из корня.
        monkeypatch.chdir(fake_repo / "deep" / "nested")
        assert load_config().version == "t"

    def test_missing_config_file_is_an_error(self, fake_repo, monkeypatch):
        # Корень найден, но самого файла нет — это по-прежнему ошибка.
        monkeypatch.chdir(fake_repo)
        (fake_repo / "config" / "default.yaml").unlink()
        with pytest.raises(ConfigError, match="конфиг не найден"):
            load_config()

    def test_explicit_absolute_path_still_wins(self, fake_repo, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_config(fake_repo / "config" / "default.yaml").version == "t"


class TestPathOf:
    def test_relative_path_resolves_from_root(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo / "deep" / "nested")
        config = load_config()
        assert config.path_of("eval.dataset") == fake_repo.resolve() / "evals/datasets/x.jsonl"

    def test_absolute_path_left_alone(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo)
        config = load_config()
        config.data["eval"]["dataset"] = "/tmp/somewhere/x.jsonl"
        assert config.path_of("eval.dataset") == Path("/tmp/somewhere/x.jsonl")

    def test_without_root_falls_back_to_plain_path(self):
        config = Config(data={"a": {"b": "rel/path"}})
        assert config.path_of("a.b") == Path("rel/path")

    def test_missing_key_still_raises(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo)
        with pytest.raises(ConfigError):
            load_config().path_of("eval.nope")


class TestFallbackToPackageLocation:
    """Запуск вообще вне репозитория — например, из домашнего каталога.

    Вверх от него подниматься некуда, и единственная зацепка — каталог
    установленного пакета: при `pip install -e` он лежит в src/ того же
    репозитория. Именно этот случай и падал: запуск CLI из ~.
    """

    def test_root_found_outside_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # tmp_path вне репозитория, но пакет установлен из него
        assert find_repo_root() == Path(__file__).resolve().parents[1]

    def test_explicit_start_does_not_fall_back(self, tmp_path):
        # Явно указанный каталог означает «искать здесь», а не «где получится».
        assert find_repo_root(tmp_path) is None

    def test_config_loads_from_outside_the_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.path_of("eval.dataset").is_absolute()
        assert config.path_of("eval.dataset").exists()


class TestEnvOverridesStayStrings:
    """Переопределения из окружения берутся как есть, без разбора YAML.

    Все шесть строковые, а разбор менял им тип: метка прогона `1.0`
    становилась числом, а `2026-08-13` — датой, на которой падал отпечаток
    конфига, потому что date не сериализуется в JSON.
    """

    def test_date_like_version_survives(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo)
        monkeypatch.setenv("KZRAG_VERSION", "2026-08-13")
        config = load_config()
        assert config.data["version"] == "2026-08-13"
        assert isinstance(config.data["version"], str)

    def test_fingerprint_computes_on_date_like_version(self, fake_repo, monkeypatch):
        # Раньше здесь был TypeError: Object of type date is not JSON serializable.
        monkeypatch.chdir(fake_repo)
        monkeypatch.setenv("KZRAG_VERSION", "2026-08-13")
        assert len(load_config().fingerprint) == 16

    def test_number_like_version_is_not_a_float(self, fake_repo, monkeypatch):
        monkeypatch.chdir(fake_repo)
        monkeypatch.setenv("KZRAG_VERSION", "1.0")
        assert load_config().data["version"] == "1.0"

    def test_dsn_is_passed_through_untouched(self, fake_repo, monkeypatch):
        dsn = "postgresql://kzrag:kzrag@db:5432/kzrag"
        monkeypatch.chdir(fake_repo)
        monkeypatch.setenv("KZRAG_DATABASE_URL", dsn)
        assert load_config().get("vector_store.dsn") == dsn


class TestSearchDepthIsOneKnob:
    """eval.k и retrieval.top_k — одна глубина поиска, названная дважды.

    Прогон ищет с eval.k, а /search, /ask и kzrag-search — с retrieval.top_k.
    Разъедься они, и метрики описывали бы выдачу, которой не отдаёт ни одна
    точка входа.
    """

    def _config(self, eval_k: int, serve_k: int) -> Config:
        return Config(data={"eval": {"k": eval_k}, "retrieval": {"top_k": serve_k}})

    def test_matching_depths_pass(self):
        from kz_labor_rag.cli import assert_search_depth_matches

        assert assert_search_depth_matches(self._config(5, 5)) is None

    def test_divergent_depths_are_refused(self):
        from kz_labor_rag.cli import assert_search_depth_matches

        with pytest.raises(ConfigError, match="одна и та же"):
            assert_search_depth_matches(self._config(10, 5))

    def test_shipped_config_is_consistent(self):
        from kz_labor_rag.cli import assert_search_depth_matches

        assert assert_search_depth_matches(load_config()) is None
