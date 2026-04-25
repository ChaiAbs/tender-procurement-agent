"""
rag/domain_indexer.py — Build ChromaDB index over domain knowledge documents.

Indexes:
  - UNSPSC commodity codes (Excel): each code as a chunk with full hierarchy path
  - COFOG classifications (PDF): each class definition as a chunk
  - AusTender valid field values: one chunk per field listing all valid values

Run:
    python rag/domain_indexer.py
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COFOG_FILE,
    DOMAIN_RAG_COLLECTION_NAME,
    DOMAIN_RAG_DIR,
    UNSPSC_FILE,
)


def _get_collection(reset: bool = False):
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    client = chromadb.PersistentClient(path=DOMAIN_RAG_DIR)
    ef     = ONNXMiniLM_L6_V2()

    if reset:
        try:
            client.delete_collection(DOMAIN_RAG_COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=DOMAIN_RAG_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── UNSPSC ─────────────────────────────────────────────────────────────────────

def _build_hierarchy(df: pd.DataFrame) -> dict[str, dict]:
    """Build a key → row lookup for navigating parent relationships."""
    return {str(int(row["Key"])): row for _, row in df.iterrows() if pd.notna(row["Key"])}


def _get_ancestors(key: str, lookup: dict) -> list[str]:
    """Walk up parent chain, return list of titles from segment → commodity."""
    titles = []
    current = lookup.get(key)
    while current is not None:
        titles.append(str(current["Title"]))
        parent_key = current["Parent key"]
        if pd.isna(parent_key):
            break
        current = lookup.get(str(int(parent_key)))
    return list(reversed(titles))


def index_unspsc(collection, verbose: bool = True) -> int:
    """Index UNSPSC 8-digit commodity codes with full hierarchy context."""
    if verbose:
        print("[UNSPSC] Loading …")
    df = pd.read_excel(UNSPSC_FILE)

    # Only 8-digit commodity codes (leaf level), deduplicated
    codes_df = df[df["Code"].astype(str).str.match(r"^\d{8}$")].copy()
    codes_df = codes_df.drop_duplicates(subset=["Code"])
    lookup   = _build_hierarchy(df)

    documents, ids, metadatas = [], [], []

    for _, row in codes_df.iterrows():
        code  = str(row["Code"])
        title = str(row["Title"])
        key   = str(int(row["Key"]))

        ancestors = _get_ancestors(key, lookup)
        hierarchy = " > ".join(ancestors) if ancestors else title

        text = (
            f"UNSPSC Code: {code}\n"
            f"Name: {title}\n"
            f"Hierarchy: {hierarchy}"
        )
        documents.append(text)
        ids.append(f"unspsc_{code}")
        metadatas.append({
            "source":    "unspsc",
            "code":      code,
            "title":     title,
            "hierarchy": hierarchy,
        })

    # Batch upsert
    batch = 500
    for i in range(0, len(documents), batch):
        collection.upsert(
            documents=documents[i:i+batch],
            ids=ids[i:i+batch],
            metadatas=metadatas[i:i+batch],
        )
        if verbose:
            print(f"[UNSPSC] Indexed {min(i+batch, len(documents)):,}/{len(documents):,}", end="\r")

    if verbose:
        print(f"\n[UNSPSC] Done — {len(documents):,} codes indexed.")
    return len(documents)


# ── COFOG ──────────────────────────────────────────────────────────────────────

def _extract_cofog_chunks(pdf_path: str) -> list[dict]:
    """
    Extract COFOG class definitions from PDF.
    Each chunk = one class entry (e.g. "01.1.1 Executive and legislative organs").
    """
    import pdfplumber

    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        # COFOG definitions start around page 39, end around page 80
        for page in pdf.pages[38:85]:
            text = page.extract_text()
            if text:
                full_text.append(text)

    combined = "\n".join(full_text)

    # Split on COFOG class codes like "01.1.1", "02.3.0", etc.
    pattern = r"(?=\n\d{2}\.\d+[\.\d]*\s+[A-Z])"
    raw_chunks = re.split(pattern, combined)

    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Extract code and title from first line
        m = re.match(r"^(\d{2}(?:\.\d+)*)\s+(.+?)(?:\s*\(CS\))?\s*\n", chunk)
        if m:
            code  = m.group(1)
            title = m.group(2).strip()
            body  = chunk[m.end():].strip()
            # Determine level: "01" = division, "01.1" = group, "01.1.1" = class
            depth = len(code.split("."))
            level = {1: "division", 2: "group", 3: "class"}.get(depth, "class")
            chunks.append({
                "code":  code,
                "title": title,
                "level": level,
                "text":  f"COFOG {level.title()}: {code} — {title}\n{body}",
            })

    return chunks


def index_cofog(collection, verbose: bool = True) -> int:
    """Index COFOG classification definitions from PDF."""
    if verbose:
        print("[COFOG] Extracting from PDF …")

    chunks = _extract_cofog_chunks(COFOG_FILE)
    if verbose:
        print(f"[COFOG] {len(chunks)} chunks extracted.")

    # Deduplicate by code — PDF may contain repeated sections
    seen = {}
    for c in chunks:
        key = c["code"]
        if key not in seen:
            seen[key] = c
    chunks = list(seen.values())

    documents = [c["text"]  for c in chunks]
    ids       = [f"cofog_{c['code'].replace('.', '_')}" for c in chunks]
    metadatas = [{
        "source": "cofog",
        "code":   c["code"],
        "title":  c["title"],
        "level":  c["level"],
    } for c in chunks]

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    if verbose:
        print(f"[COFOG] Done — {len(documents)} entries indexed.")
    return len(documents)


# ── AusTender valid values ─────────────────────────────────────────────────────

_AUSTENDER_VALID_VALUES = {
    "procurement_method": [
        "open tender", "limited tender", "demand driven", "closed non-competitive",
        "prequalified tender", "open competitive", "targeted or restricted competitive",
        "limited", "not required", "selective", "not indicated", "open non-competitive",
        "open", "open offer process", "limited offer process", "selective offer process",
        "quotation", "select", "public tender", "restricted tender", "single select",
        "limited sourcing", "request for tender (rft)", "non tender", "direct negotiation",
        "open advertisement", "direct sourcing", "selective tender", "sole source",
        "request for tender", "invited tender", "open ito", "request for quote (rfq)",
        "request for quotation", "request for proposal (rfp)",
    ],
    "disposition": [
        "contract notice", "grant", "request", "planned", "grant opportunity",
    ],
    "publisher_gov_type": [
        "fed", "qld", "nsw", "vic", "wa", "act", "sa", "tas", "nt",
    ],
    "publisher_cofog_level": ["1.0", "2.0"],
    "is_consultancy_services": ["no"],
}


def index_austender_fields(collection, verbose: bool = True) -> int:
    """Index AusTender field valid values as lookup chunks."""
    descriptions = {
        "procurement_method":      "How the contract was procured. Valid values (use exact string, lowercase):",
        "disposition":             "Type of AusTender notice. Valid values (use exact string, lowercase):",
        "publisher_gov_type":      "Government type of the publishing agency. 'fed' = federal. State codes: qld, nsw, vic, wa, act, sa, tas, nt. Valid values:",
        "publisher_cofog_level":   "COFOG classification level of the publisher. Only two values exist in the data:",
        "is_consultancy_services": "Whether this is a consultancy services contract. Always 'no' in the dataset:",
    }

    documents, ids, metadatas = [], [], []
    for field, values in _AUSTENDER_VALID_VALUES.items():
        desc   = descriptions.get(field, f"Valid values for {field}:")
        text   = f"AusTender field: {field}\n{desc}\n" + "\n".join(f"  - {v}" for v in values)
        documents.append(text)
        ids.append(f"austender_{field}")
        metadatas.append({"source": "austender", "field": field})

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    if verbose:
        print(f"[AusTender] Done — {len(documents)} field entries indexed.")
    return len(documents)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_domain_index(verbose: bool = True) -> None:
    """Build (or rebuild) the full domain knowledge index."""
    if verbose:
        print(f"Building domain RAG index → {DOMAIN_RAG_DIR}")

    collection = _get_collection(reset=True)

    n_unspsc    = index_unspsc(collection, verbose=verbose)
    n_cofog     = index_cofog(collection, verbose=verbose)
    n_austender = index_austender_fields(collection, verbose=verbose)

    if verbose:
        print(f"\nDomain index complete: {n_unspsc + n_cofog + n_austender:,} total chunks.")


if __name__ == "__main__":
    build_domain_index()
