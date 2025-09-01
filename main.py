from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
import requests, json, csv, io, os

from database import SessionLocal, init_db
from models import Site, Page
from embedder import embed_texts, build_faiss, save_faiss, load_faiss

# OPTIONAL LLM Integration
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
if USE_LLM:
    from openai import OpenAI
    openai_client = OpenAI()

app = FastAPI(title="Scraper + DB (BLOB + Search + Semantic)")
templates = Jinja2Templates(directory="templates")

init_db()
Path("scraped_sites").mkdir(exist_ok=True)
Path("indexes").mkdir(exist_ok=True)

# ============ Scraper helpers ============
def get_visible_text(soup: BeautifulSoup):
    visible_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']
    lines = []
    for tag in soup.find_all(visible_tags):
        text = tag.get_text(strip=True)
        if text:
            lines.append(text)
    return lines

def extract_links(soup: BeautifulSoup, base_url: str):
    return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]

def scrape_site(start_url: str, max_pages: int = 20, same_domain_only: bool = True):
    seen = set()
    queue = deque([start_url])
    all_lines = []
    domain = urlparse(start_url).netloc

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for line in get_visible_text(soup):
                all_lines.append((url, line))
            for link in extract_links(soup, url):
                if same_domain_only and urlparse(link).netloc != domain:
                    continue
                if link not in seen:
                    queue.append(link)
        except Exception as e:
            print("scrape error:", e)
    return all_lines

# ============ Persistence ============
def save_to_file_and_db(data, format_choice, base_url):
    output_dir = Path("scraped_sites")
    output_dir.mkdir(exist_ok=True)

    ext = "csv" if format_choice.lower() == "csv" else "json"
    file_path = output_dir / f"{urlparse(base_url).netloc}.{ext}"

    if ext == "csv":
        with open(file_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Page URL", "Line"])
            writer.writerows(data)
    else:
        grouped = {}
        for u, line in data:
            grouped.setdefault(u, []).append(line)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)

    db: Session = SessionLocal()
    try:
        site = db.query(Site).filter(Site.base_url == base_url).first()
        if not site:
            site = Site(base_url=base_url)
            db.add(site)
            db.commit()
            db.refresh(site)

        db.query(Page).filter(Page.site_id == site.id).delete()

        db.add_all([Page(site_id=site.id, url=u, text_line=t) for u, t in data])

        with open(file_path, "rb") as f:
            site.file_data = f.read()
            site.file_name = file_path.name

        db.commit()
        return site.id, file_path
    finally:
        db.close()

# ============ Routes ============
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    db = SessionLocal()
    sites = db.query(Site).all()
    db.close()
    return templates.TemplateResponse("index.html", {"request": request, "sites": sites})

@app.post("/scrape")
def scrape(urls: str = Form(...), format_choice: str = Form("json")):
    try:
        urls_list = [u.strip() for u in urls.split(",") if u.strip()]
        if not urls_list:
            return HTMLResponse("No valid URLs provided.", status_code=400)

        last_path = None
        last_site_id = None
        for url in urls_list:
            lines = scrape_site(url, max_pages=50, same_domain_only=True)
            site_id, path = save_to_file_and_db(lines, format_choice, url)
            last_site_id, last_path = site_id, path

        return FileResponse(last_path, media_type="application/octet-stream", filename=last_path.name)
    except Exception as e:
        return HTMLResponse(f"<h1>Error: {e}</h1>", status_code=500)

@app.get("/sites")
def list_sites():
    db = SessionLocal()
    sites = db.query(Site).all()
    db.close()
    return [
        {
            "id": s.id,
            "base_url": s.base_url,
            "file_name": s.file_name,
            "download_url": f"/download/{s.id}",
            "view_url": f"/view-data/{s.id}",
        } for s in sites
    ]

@app.get("/download/{site_id}")
def download_file(site_id: int):
    db = SessionLocal()
    site = db.query(Site).filter(Site.id == site_id).first()
    db.close()
    if not site or not site.file_data:
        return HTMLResponse("<h3>No file found for this site.</h3>", status_code=404)

    return StreamingResponse(
        io.BytesIO(site.file_data),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={site.file_name or 'file.bin'}"}
    )

@app.get("/view-data/{site_id}")
def view_data(site_id: int):
    db = SessionLocal()
    site = db.query(Site).filter(Site.id == site_id).first()
    db.close()
    if not site or not site.file_data or not site.file_name:
        return JSONResponse({"error": "Data not found"}, status_code=404)

    name = site.file_name.lower()
    blob = site.file_data
    if name.endswith(".json"):
        try:
            data = json.loads(blob.decode("utf-8"))
        except Exception:
            return JSONResponse({"error": "Invalid JSON in BLOB"}, status_code=400)
        return JSONResponse(data)
    elif name.endswith(".csv"):
        text = blob.decode("utf-8", errors="replace")
        rows, headers = [], None
        for i, row in enumerate(csv.reader(text.splitlines())):
            if i == 0:
                headers = row
            else:
                rows.append({headers[j] if j < len(headers) else f"col{j}": v for j, v in enumerate(row)})
        return JSONResponse(rows)
    return JSONResponse({"error": "Unknown file format"}, status_code=400)

@app.get("/search")
def keyword_search(
    q: str = Query(..., description="keyword or phrase"),
    site_id: int | None = Query(None)
):
    db = SessionLocal()
    query = db.query(Page)
    if site_id is not None:
        query = query.filter(Page.site_id == site_id)
    results = query.filter(Page.text_line.ilike(f"%{q}%")).limit(200).all()
    db.close()
    return [{"url": r.url, "text": r.text_line} for r in results]

@app.post("/embed/{site_id}")
def embed_site(site_id: int):
    db = SessionLocal()
    pages = db.query(Page).filter(Page.site_id == site_id).order_by(Page.id.asc()).all()
    db.close()
    if not pages:
        return JSONResponse({"error": "No pages to embed"}, status_code=404)

    texts = [p.text_line for p in pages]
    vecs = embed_texts(texts)
    index = build_faiss(vecs)
    id_map = {i: pages[i].id for i in range(len(pages))}

    base = Path("indexes") / f"site_{site_id}"
    save_faiss(index, id_map, base)
    return {"status": "ok", "site_id": site_id, "vectors": len(texts)}

@app.get("/semantic_search")
def semantic_search(
    q: str = Query(..., description="natural language query"),
    site_id: int = Query(...),
    k: int = Query(5, ge=1, le=50)
):
    base = Path("indexes") / f"site_{site_id}"
    index, id_map = load_faiss(base)
    if index is None:
        return JSONResponse({"error": "Index not found. Run POST /embed/{site_id} first."}, status_code=404)

    qv = embed_texts([q])
    scores, idxs = index.search(qv, k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    db = SessionLocal()
    out = []
    for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
        if i == -1:
            continue
        page_id = id_map.get(i)
        if not page_id:
            continue
        p = db.query(Page).filter(Page.id == page_id).first()
        if p:
            out.append({"rank": rank, "score": float(score), "url": p.url, "text": p.text_line})
    db.close()
    return out

@app.post("/ask")
def ask_llm(
    q: str = Form(..., description="your question"),
    site_id: int = Form(...),
    k: int = Form(6)
):
    context_hits = semantic_search(q=q, site_id=site_id, k=k)
    if isinstance(context_hits, dict) and "error" in context_hits:
        return context_hits

    context_text = "\n\n".join([f"- {h['text']}" for h in context_hits])

    if not USE_LLM:
        return {
            "message": "No OPENAI_API_KEY set. Returning context only.",
            "question": q,
            "context": context_text,
            "hits": context_hits
        }

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {q}\n\n"
        "Answer:"
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    return {"answer": answer, "hits": context_hits}
