"""
CRRA Lab C1 — Policy Knowledge Base Setup

Loads the BizOps procurement policy articles into a ChromaDB collection so the
Analysis Agent (Lab C3) can cite policy when it makes a recommendation.

Run from the project root:
    python data/kb_setup.py
"""

from pathlib import Path

import chromadb

KB_DIR = Path(__file__).parent / "kb"
COLLECTION_NAME = "crra_policy"

TEST_QUERIES = [
    "who approves a 60 lakh contract",
    "contract auto renews next month and we missed the notice deadline",
    "two monitoring tools with low licence usage",
    "vendor wants a 20 percent price increase at renewal",
    "we no longer need this tool at all, how do we exit",
]


def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a policy article into one chunk per '## ' section.

    Each section is one self-contained rule. Embedding a whole file as a single
    chunk buries the rule you need among unrelated ones and weakens every match,
    so each section is stored and searched on its own.
    """
    sections: list[tuple[str, list[str]]] = []
    heading, body = None, []

    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, body))
            heading, body = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # document title, not a rule
        elif heading is not None:
            body.append(line)

    if heading is not None:
        sections.append((heading, body))

    chunks = []
    for heading, lines in sections:
        text_body = "\n".join(lines).strip()
        if not text_body:
            continue
        chunks.append(
            {
                "id": f"{filename}::{len(chunks)}",
                # Heading is embedded too: it often names the rule better than the body
                "document": f"{heading}\n{text_body}",
                "metadata": {"source": filename, "heading": heading},
            }
        )
    return chunks


def main() -> None:
    client = chromadb.Client()

    # Start clean so re-running the script does not stack duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Cosine distance keeps (1 - distance) in a readable 0..1 range
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR}, check the folder path.")

    ids, docs, metas = [], [], []
    print("Loading policy articles\n" + "=" * 60)
    for md_file in md_files:
        chunks = chunk_article(md_file.read_text(encoding="utf-8"), md_file.name)
        print(f"  {md_file.name:<32} {len(chunks)} chunks")
        for c in chunks:
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print("=" * 60)
    print(f"  TOTAL: {len(ids)} chunks from {len(md_files)} articles\n")

    print("Testing retrieval\n" + "=" * 60)
    for q in TEST_QUERIES:
        res = collection.query(query_texts=[q], n_results=1)
        meta = res["metadatas"][0][0]
        confidence = 1 - res["distances"][0][0]
        print(f'  "{q}"')
        print(f"     -> {meta['source']}  §{meta['heading']}   confidence {confidence:.0%}\n")

    print("=" * 60)
    print("KB ready. Lab C3's Analysis Agent will query this collection.")


if __name__ == "__main__":
    main()
