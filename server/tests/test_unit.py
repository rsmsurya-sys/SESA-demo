from app.cache import TtlCache
from app.exceptions import InvalidImageError
from app.inference import analyze_symptoms, decode_image
from app.config import Settings


def test_ttl_cache_miss():
    c = TtlCache()
    assert c.get("x") is None
    c.set("x", {"a": 1}, ttl=60)
    assert c.get("x") == {"a": 1}


def test_decode_image_rejects_garbage():
    s = Settings()
    try:
        decode_image("@@@", s)
        assert False, "expected error"
    except InvalidImageError:
        pass


def test_symptoms_no_location():
    out = analyze_symptoms("I feel unwell", [])
    assert out["extracted_entities"]["location"] is None
    assert len(out["symptom_embedding"]) == 768
