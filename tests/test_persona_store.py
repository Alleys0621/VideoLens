from src.agent.persona_store import (
    DEFAULT_PERSONA_ID,
    build_persona_system_prompt,
    get_persona_meta,
)


def test_get_persona_meta_du():
    meta = get_persona_meta("du")
    assert meta["id"] == "du"
    assert meta["name"] == "阿毒"
    assert meta["tagline"] == "毒舌损友"


def test_unknown_falls_back_to_default():
    meta = get_persona_meta("not_exists")
    assert meta["id"] == DEFAULT_PERSONA_ID


def test_system_prompt_has_shared_boundary():
    for persona_id in ("alleys", "du", "lao_ju"):
        prompt = build_persona_system_prompt(persona_id)
        assert "不假装看到" in prompt
        assert "不编造" in prompt


def test_personas_are_distinct():
    alleys = build_persona_system_prompt("alleys")
    du = build_persona_system_prompt("du")
    lao_ju = build_persona_system_prompt("lao_ju")
    assert "知心搭子" in alleys
    assert "损友搭子" in du
    assert "剧评搭子" in lao_ju
