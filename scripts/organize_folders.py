"""Приведение папок датасета к НАСТОЯЩИМ именам + подписи «кто есть кто».

Переименовывает старые ключи (я/мама/папа/сестра1/сестра2) в реальные имена
и кладёт в каждую папку data/datasets/raw/<имя>/ файл README.txt, чтобы
было ясно чьи фото туда кидать.

Запуск: python scripts/organize_folders.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/datasets/raw"
FACES = ROOT / "data/datasets/faces"

# старое имя -> новое (настоящее) имя. Меняй под свою семью!
LEGACY = {
    "я": "Алексей",
    "мама": "Ольга",
    "папа": "Сергей",
    "сестра1": "Аня",
    "сестра2": "Катя",
}

HINT = (
    "ПОЛОЖИ СЮДА ФОТОГРАФИИ ЭТОГО ЧЕЛОВЕКА\n"
    "=======================================\n"
    "{who}\n\n"
    "Что делать:\n"
    "  1) положи сюда 20-25 чётких фото (анфас, повороты, разное освещение);\n"
    "  2) python scripts/collect_faces.py\n"
    "  3) python scripts/register_faces.py\n"
    "Сколько фото и как снимать: docs/DATASET.md\n"
)


def main() -> int:
    names = set(LEGACY.values())
    # создаём/переименовываем
    for old, new in LEGACY.items():
        for base in (RAW, FACES):
            old_p = base / old
            new_p = base / new
            if old_p.exists() and not new_p.exists():
                old_p.rename(new_p)
                print(f"[переименовано] {old_p.relative_to(ROOT)} -> {new_p.relative_to(ROOT)}")
            new_p.mkdir(parents=True, exist_ok=True)

    # подписи «кто есть кто»
    for name in sorted(names):
        folder = RAW / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.txt").write_text(
            HINT.format(who=f"Это папка: {name} — твои фото этого человека идут именно сюда."),
            encoding="utf-8")
        print(f"[ok] {folder.relative_to(ROOT)}  (подписано README.txt)")

    print("\nТекущий список людей (папки):")
    for p in sorted(x for x in RAW.iterdir() if x.is_dir()):
        print("   -", p.name)
    print("\nДалее обязательно:")
    print("  1) в config/*.yaml -> people.names впиши те же имена;")
    print("  2) пни фото в папки;")
    print("  3) python scripts/collect_faces.py && python scripts/register_faces.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())