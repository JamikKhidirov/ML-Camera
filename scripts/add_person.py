"""Добавление нового человека в датасет (можно сколько угодно людей).

Запуск: python scripts/add_person.py Толя
Затем:
  1) положи 15-25 фото в data/datasets/raw/Толя/
  2) впиши в config/*.yaml -> people.names:  Толя: "Анатолий"
  3) python scripts/collect_faces.py
  4) python scripts/register_faces.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Добавить человека в датасет")
    parser.add_argument("name", help="имя папки (напр. Толя, Бабушка, Сосед)")
    args = parser.parse_args()
    name = args.name.strip()
    if not name:
        print("Пустое имя.")
        return 1
    for sub in ("raw", "faces"):
        p = ROOT / "data/datasets" / sub / name
        p.mkdir(parents=True, exist_ok=True)
        print(f"[ok] {p.relative_to(ROOT)}")
    hint = ROOT / "data/datasets/raw" / name / "README.txt"
    if not hint.exists():
        hint.write_text(
            f"ПОЛОЖИ СЮДА ФОТОГРАФИИ: {name}\n"
            "=======================================\n\n"
            "Что делать:\n"
            "  1) положи сюда 15-25 чётких фото (анфас, повороты, разное освещение);\n"
            "  2) добавь имя в config/*.yaml -> people.names;\n"
            "  3) python scripts/collect_faces.py\n"
            "  4) python scripts/register_faces.py\n"
            "Сколько фото и как снимать: docs/DATASET.md\n",
            encoding="utf-8",
        )

    print("\nДальше:")
    print(f"  1) кидай фото: data/datasets/raw/{name}/")
    print(f"  2) в config/*.yaml -> people.names добавь строку:  {name}: \"Имя для уведомлений\"")
    print("  3) python scripts/collect_faces.py")
    print("  4) python scripts/register_faces.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())