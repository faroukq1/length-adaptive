from src.data.promptcraft import (
    PROMPT_STYLES,
    build_item_texts_for_style,
    format_item_prompt,
)


def test_format_item_prompt_all_styles():
    title = "Toy Story (1995)"
    genres = ["Animation", "Children", "Comedy", "Adventure"]

    p1 = format_item_prompt(title, genres, "P1_title_only")
    p2 = format_item_prompt(title, genres, "P2_title_genre")
    p3 = format_item_prompt(title, genres, "P3_user_centric")
    p4 = format_item_prompt(title, genres, "P4_hybrid")

    assert p1 == "Toy Story (1995)"
    assert "Genre:" in p2
    assert "Users who like" in p3
    assert "For fans of:" in p4


def test_build_item_texts_for_style_includes_pad_row():
    processed_data = {
        "config": {"num_items": 2},
        "mappings": {"idx_to_item": {1: 101, 2: 102}},
    }
    item_meta = {
        101: {"title": "Movie A", "genres": ["Drama"]},
        102: {"title": "Movie B", "genres": ["Action", "Thriller"]},
    }

    for style in PROMPT_STYLES:
        texts, missing = build_item_texts_for_style(processed_data, item_meta, style)
        assert len(texts) == 3
        assert texts[0] == "[PAD]"
        assert missing == 0
