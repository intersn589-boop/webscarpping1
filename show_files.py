from sqlalchemy.orm import Session
from models import engine, Site, Page
from pathlib import Path
from tabulate import tabulate

db = Session(bind=engine)

rows = []
for site in db.query(Site).all():
    for page in site.pages:
        file_path = Path(page.file_path)
        if file_path.exists():
            file_size_kb = file_path.stat().st_size / 1024
            file_type = file_path.suffix.lstrip(".")
        else:
            file_size_kb = None
            file_type = None
        rows.append([
            site.base_url,
            file_type or "N/A",
            str(file_path),
            f"{file_size_kb:.2f} KB" if file_size_kb else "File missing"
        ])

print(tabulate(rows, headers=["Website", "File Type", "File Path", "Size"], tablefmt="grid"))
