import os
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

PROMPT_STYLES = [
    "P1_title_only",
    "P2_title_genre",
    "P3_user_centric",
    "P4_hybrid",
]


def validate_style(style: str) -> str:
    if style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown style '{style}'. Supported styles: {', '.join(PROMPT_STYLES)}"
        )
    return style


def load_item_metadata(raw_data_dir: str = "data/ml-100k/raw") -> Dict[int, Dict[str, List[str]]]:
    """Load ML-100K item metadata from u.item into {movie_id: {title, genres}}."""
    item_file = os.path.join(raw_data_dir, "ml-100k", "u.item")
    if not os.path.exists(item_file):
        raise FileNotFoundError(
            f"Item metadata file not found: {item_file}. "
            "Run ML-100K preprocessing first to download/extract raw data."
        )

    items_df = pd.read_csv(
        item_file,
        sep="|",
        encoding="latin-1",
        header=None,
        names=["movie_id", "title", "release_date", "video_release", "imdb_url"] + GENRE_COLUMNS,
    )

    item_meta: Dict[int, Dict[str, List[str]]] = {}
    for _, row in items_df.iterrows():
        genres = [genre for genre in GENRE_COLUMNS if row[genre] == 1]
        item_meta[int(row["movie_id"])] = {
            "title": str(row["title"]).strip(),
            "genres": genres,
        }

    return item_meta


def format_item_prompt(title: str, genres: List[str], style: str) -> str:
    """Create a prompt-style-specific text for one item."""
    style = validate_style(style)

    main_genre = genres[0] if genres else "Movie"
    top_genres = ", ".join(genres[:4]) if genres else "entertainment"

    if style == "P1_title_only":
        return title

    if style == "P2_title_genre":
        if genres:
            return f"{title} | Genre: {', '.join(genres[:3])}"
        return title

    if style == "P3_user_centric":
        return f"Users who like {title} enjoy: {top_genres}"

    # style == "P4_hybrid"
    return f"{title} | Genre: {main_genre} | For fans of: {top_genres}"


def build_item_texts_for_style(
    processed_data: dict,
    item_meta: Dict[int, Dict[str, List[str]]],
    style: str,
) -> Tuple[List[str], int]:
    """Build ordered prompt texts for IDs [0..num_items], where index 0 is [PAD]."""
    style = validate_style(style)

    num_items = processed_data["config"]["num_items"]
    idx_to_item = processed_data["mappings"]["idx_to_item"]

    item_texts = ["[PAD]"]
    missing_metadata = 0

    for item_id in range(1, num_items + 1):
        raw_item_id = idx_to_item.get(item_id)
        if raw_item_id is None:
            raw_item_id = idx_to_item.get(str(item_id))

        meta = item_meta.get(raw_item_id, {})
        title = meta.get("title", f"Unknown Movie {item_id}")
        genres = meta.get("genres", [])

        if raw_item_id not in item_meta:
            missing_metadata += 1

        item_texts.append(format_item_prompt(title=title, genres=genres, style=style))

    return item_texts, missing_metadata


def encode_with_bge_m3(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 256,
    max_length: int = 128,
    use_fp16: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Encode text list into dense vectors using BGE-M3 via Transformers."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformers-based encoding requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        use_amp = device == "cuda" and use_fp16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                outputs = model(**tokenized)

        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        mask = tokenized["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.vstack(vectors).astype(np.float32)


def load_or_generate_raw_embeddings(
    processed_data: dict,
    raw_data_dir: str,
    style: str,
    cache_dir: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 256,
    max_length: int = 128,
    use_fp16: bool = True,
    device: str = "cpu",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Return raw PromptCraft embeddings for one style, loading from cache when available.
    """
    style = validate_style(style)
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{style}_raw.npy")
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached raw embeddings: {cache_path}")
        return np.load(cache_path, allow_pickle=False), cache_path

    print(f"Generating raw embeddings for style: {style}")
    item_meta = load_item_metadata(raw_data_dir=raw_data_dir)
    item_texts, missing = build_item_texts_for_style(
        processed_data=processed_data,
        item_meta=item_meta,
        style=style,
    )

    if missing > 0:
        print(f"Warning: {missing} items had no metadata match; fallback text was used.")

    raw_embeddings = encode_with_bge_m3(
        texts=item_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        device=device,
    )

    np.save(cache_path, raw_embeddings)
    print(f"Saved raw embeddings: {cache_path} (shape={raw_embeddings.shape})")

    return raw_embeddings, cache_path


def pca_compress_embeddings(
    raw_embeddings: np.ndarray,
    out_dim: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    PCA compress raw embeddings and restore row 0 as zero-padding.

    Returns:
        compressed: shape [num_items+1, out_dim]
        explained_variance: float
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for PCA compression. "
            "Install it with: pip install scikit-learn"
        ) from exc

    if raw_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={raw_embeddings.shape}")

    if raw_embeddings.shape[0] < 2:
        raise ValueError("Raw embeddings must include at least [PAD] + 1 item row.")

    max_components = min(raw_embeddings.shape[1], raw_embeddings.shape[0] - 1)
    if out_dim > max_components:
        raise ValueError(
            f"Cannot project to out_dim={out_dim}; max possible components={max_components}."
        )

    pca = PCA(n_components=out_dim, random_state=random_state)
    items_only = raw_embeddings[1:]
    compressed_items = pca.fit_transform(items_only).astype(np.float32)

    pad_row = np.zeros((1, out_dim), dtype=np.float32)
    compressed = np.vstack([pad_row, compressed_items]).astype(np.float32)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    return compressed, explained_variance
