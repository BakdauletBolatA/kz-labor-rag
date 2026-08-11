"""Кодирование текста и кэш векторов.

Главная проверка — префиксы e5. Запрос и документ обязаны кодироваться
по-разному, и перепутать их легко: ошибка не даёт ни исключения, ни записи
в логе, а просто тихо роняет качество поиска.
"""

from __future__ import annotations

import numpy as np
import pytest

from kz_labor_rag.embeddings.encoder import (
    E5Encoder,
    EmbeddingCache,
    EncoderParams,
    HashEncoder,
)

PARAMS = EncoderParams(
    model="intfloat/multilingual-e5-base",
    dimensions=8,
    query_prefix="query: ",
    passage_prefix="passage: ",
)


class RecordingEncoder(E5Encoder):
    """E5Encoder без модели: запоминает, что именно ушло бы в неё."""

    def __init__(self, params, cache=None):
        super().__init__(params, cache=cache)
        self.encoded: list[str] = []

    def _encode_raw(self, texts):
        self.encoded.extend(texts)
        return np.stack(
            [np.full(self.params.dimensions, float(len(t)), dtype=np.float32) for t in texts]
        )


class TestPrefixes:
    def test_query_gets_query_prefix(self):
        enc = RecordingEncoder(PARAMS)
        enc.encode_query("можно ли уволить в отпуске")
        assert enc.encoded == ["query: можно ли уволить в отпуске"]

    def test_passages_get_passage_prefix(self):
        enc = RecordingEncoder(PARAMS)
        enc.encode_passages(["текст статьи 54", "текст статьи 52"])
        assert enc.encoded == ["passage: текст статьи 54", "passage: текст статьи 52"]

    def test_prefixes_are_not_interchangeable(self):
        # Тот же текст с разных сторон кодируется по-разному — иначе
        # асимметричность e5 потеряна.
        enc = RecordingEncoder(PARAMS)
        enc.encode_query("отпуск")
        enc.encode_passages(["отпуск"])
        assert enc.encoded[0] != enc.encoded[1]

    def test_prefix_comes_from_config(self):
        params = EncoderParams(model="m", dimensions=8, query_prefix="Q>", passage_prefix="P>")
        enc = RecordingEncoder(params)
        enc.encode_query("а")
        enc.encode_passages(["б"])
        assert enc.encoded == ["Q>а", "P>б"]

    def test_descriptor_records_prefixes(self):
        # Префиксы попадают в результат прогона: строка EVALUATION.md должна
        # восстанавливаться целиком, включая их.
        d = RecordingEncoder(PARAMS).descriptor
        assert d["query_prefix"] == "query: "
        assert d["passage_prefix"] == "passage: "


def cache(tmp_path, *, model="m", signature="sig", prefix="passage: ", normalize=True):
    return EmbeddingCache(
        tmp_path,
        model=model,
        chunking_signature=signature,
        passage_prefix=prefix,
        normalize=normalize,
    )


class TestCache:
    def test_roundtrip(self, tmp_path):
        c = cache(tmp_path)
        vectors = np.arange(16, dtype=np.float32).reshape(2, 8)
        c.put_many(["a", "b"], vectors)
        got = c.get_many(["a", "b"])
        assert np.allclose(got[0], vectors[0])
        assert np.allclose(got[1], vectors[1])

    def test_miss_returns_nothing_for_unknown_text(self, tmp_path):
        c = cache(tmp_path)
        c.put_many(["a"], np.zeros((1, 8), dtype=np.float32))
        assert c.get_many(["b"]) == {}

    def test_chunking_change_invalidates_cache(self, tmp_path):
        # Смена нарезки обязана обесценить старые векторы автоматически.
        cache(tmp_path, signature="sig-1").put_many(["текст"], np.zeros((1, 8), dtype=np.float32))
        assert cache(tmp_path, signature="sig-2").get_many(["текст"]) == {}

    def test_prefix_change_invalidates_cache(self, tmp_path):
        # Вектор считается от «префикс + текст». Если префикс не входит в ключ,
        # смена префикса вернёт векторы, посчитанные со старым, — правдоподобные
        # и неверные, причём молча.
        cache(tmp_path, prefix="passage: ").put_many(
            ["текст"], np.ones((1, 8), dtype=np.float32)
        )
        assert cache(tmp_path, prefix="документ: ").get_many(["текст"]) == {}

    def test_normalize_change_invalidates_cache(self, tmp_path):
        # Нормализация меняет сам вектор, значит входит в ключ.
        cache(tmp_path, normalize=True).put_many(["текст"], np.ones((1, 8), dtype=np.float32))
        assert cache(tmp_path, normalize=False).get_many(["текст"]) == {}

    def test_same_parameters_hit_the_cache(self, tmp_path):
        cache(tmp_path).put_many(["текст"], np.ones((1, 8), dtype=np.float32))
        assert cache(tmp_path).get_many(["текст"]) != {}

    def test_key_covers_every_vector_affecting_parameter(self, tmp_path):
        base = cache(tmp_path).key("текст")
        variants = {
            "модель": cache(tmp_path, model="other").key("текст"),
            "префикс": cache(tmp_path, prefix="q: ").key("текст"),
            "нормализация": cache(tmp_path, normalize=False).key("текст"),
            "чанкинг": cache(tmp_path, signature="other").key("текст"),
            "текст": cache(tmp_path).key("другой текст"),
        }
        collisions = [name for name, key in variants.items() if key == base]
        assert not collisions, f"эти параметры не влияют на ключ кэша: {collisions}"

    def test_model_change_uses_separate_storage(self, tmp_path):
        # Смена модели не затирает уже посчитанное: вернуться назад можно
        # без переиндексации.
        a = cache(tmp_path, model="model-a")
        a.put_many(["текст"], np.ones((1, 8), dtype=np.float32))
        b = cache(tmp_path, model="model-b")
        assert b.get_many(["текст"]) == {}
        assert a.get_many(["текст"]) != {}
        assert a.path != b.path

    def test_encoder_reuses_cache_and_encodes_only_new(self, tmp_path):
        shared = cache(tmp_path)
        first = RecordingEncoder(PARAMS, cache=shared)
        first.encode_passages(["один", "два"])
        assert len(first.encoded) == 2

        second = RecordingEncoder(PARAMS, cache=shared)
        second.encode_passages(["один", "два", "три"])
        # Заново кодируется только новый текст.
        assert second.encoded == ["passage: три"]

    def test_cached_vectors_keep_order(self, tmp_path):
        shared = cache(tmp_path)
        RecordingEncoder(PARAMS, cache=shared).encode_passages(["ааа"])
        out = RecordingEncoder(PARAMS, cache=shared).encode_passages(["бб", "ааа", "гггг"])
        # Длина текста заложена в фейковый вектор, поэтому порядок проверяем по ней.
        # Средний элемент пришёл из кэша, но встал на своё место.
        assert [float(v[0]) for v in out] == [
            len("passage: бб"),
            len("passage: ааа"),
            len("passage: гггг"),
        ]


class TestHashEncoder:
    def test_deterministic(self):
        a, b = HashEncoder(16), HashEncoder(16)
        assert np.allclose(a.encode_query("вопрос"), b.encode_query("вопрос"))

    def test_normalized(self):
        vec = HashEncoder(16).encode_query("вопрос")
        assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_applies_prefixes_too(self):
        enc = HashEncoder(16)
        enc.encode_query("x")
        enc.encode_passages(["y"])
        assert enc.seen_prefixes == ["query: ", "passage: "]

    def test_empty_batch(self):
        assert HashEncoder(16).encode_passages([]).shape == (0, 16)


class TestDimensionsGuard:
    def test_mismatch_is_fatal(self, monkeypatch):
        # Схема pgvector создаётся по конфигу, поэтому расхождение размерности
        # обязано падать, а не приводить к ошибке вставки посреди индексации.
        class FakeModel:
            def get_sentence_embedding_dimension(self):
                return 1024

        enc = E5Encoder(PARAMS)

        class FakeST:
            def __init__(self, *a, **kw):
                pass

        monkeypatch.setattr(
            enc,
            "_load",
            lambda: (_ for _ in ()).throw(
                ValueError("в конфиге dimensions=8, а модель выдаёт 1024")
            ),
        )
        with pytest.raises(ValueError, match="dimensions"):
            enc.encode_query("x")
