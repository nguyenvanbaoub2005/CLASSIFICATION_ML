#!/usr/bin/env python3
"""
Tool quan ly va thong ke dataset.
Chuc nang:
1. Thong ke so luong anh tung class trong train, validation, test
2. Don dep anh loi bang cach chuyen sang thu muc backup, khong xoa vinh vien
3. Can bang du lieu train bang cach chuyen anh thua sang thu muc backup, khong xoa vinh vien
"""

import os
import sys
import random
import shutil
from PIL import Image

# Them thu muc goc vao sys.path de import cau hinh
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.config import CLASSES, COLORS

DATASET_DIR = os.path.join(project_root, 'dataset')
BACKUP_DIR = os.path.join(project_root, 'dataset_backup')
SPLITS = ['train', 'validation', 'test']
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def color(name, text):
    """Tra ve chuoi co mau neu COLORS co key tuong ung."""
    return f"{COLORS.get(name, '')}{text}{COLORS.get('reset', '')}"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def is_image_file(file_name):
    return file_name.lower().endswith(IMAGE_EXTENSIONS)


def safe_move(src, dst_dir):
    """
    Chuyen file sang thu muc backup.
    Neu trung ten file, tu dong them hau to _1, _2, ...
    """
    os.makedirs(dst_dir, exist_ok=True)

    base_name = os.path.basename(src)
    name, ext = os.path.splitext(base_name)
    dst = os.path.join(dst_dir, base_name)

    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(src, dst)
    return dst


def get_statistics():
    """Lay thong ke so luong anh."""
    stats = {}

    for split in SPLITS:
        split_path = os.path.join(DATASET_DIR, split)
        stats[split] = {}

        for cls in CLASSES:
            cls_path = os.path.join(split_path, cls)

            if os.path.exists(cls_path):
                files = [
                    f for f in os.listdir(cls_path)
                    if is_image_file(f)
                ]
                stats[split][cls] = len(files)
            else:
                stats[split][cls] = 0

    return stats


def show_statistics(stats):
    """Hien thi bang thong ke dataset."""
    clear_screen()
    print("\n" + "=" * 70)
    print(color('green', 'BANG THONG KE DU LIEU DATASET'))
    print("=" * 70 + "\n")

    total_train = 0
    total_val = 0
    total_test = 0

    print(f"{'Class':15s} | {'Train':10s} | {'Validation':10s} | {'Test':10s} | {'Tong':10s}")
    print("-" * 70)

    for cls in CLASSES:
        tr = stats['train'].get(cls, 0)
        va = stats['validation'].get(cls, 0)
        te = stats['test'].get(cls, 0)
        total = tr + va + te

        total_train += tr
        total_val += va
        total_test += te

        print(f"{cls.upper():15s} | {tr:10d} | {va:10d} | {te:10d} | {total:10d}")

    print("-" * 70)
    print(
        f"{'TONG CONG':15s} | {total_train:10d} | {total_val:10d} | "
        f"{total_test:10d} | {total_train + total_val + total_test:10d}"
    )
    print("-" * 70)

    print("\nPhan tich can bang du lieu tap train:")
    train_counts = [stats['train'][c] for c in CLASSES]

    if not train_counts or max(train_counts) == 0:
        print(color('red', 'Thu muc train dang trong.'))
        input("\nNhan Enter de quay lai menu...")
        return

    positive_counts = [c for c in train_counts if c > 0]
    min_val = min(positive_counts, default=0)
    max_val = max(train_counts)

    if min_val == 0:
        print(color('red', 'Canh bao: Co class chua co anh.'))
    elif max_val > min_val * 2:
        print(color('red', 'Canh bao: Du lieu train dang mat can bang nghiem trong.'))
        print(f"  - It nhat: {min_val} anh")
        print(f"  - Nhieu nhat: {max_val} anh")
        print("  - Khuyen nghi can bang du lieu hoac bo sung anh cho class it du lieu.")
    else:
        print(color('green', 'Du lieu tuong doi can bang.'))

    input("\nNhan Enter de quay lai menu...")


def clean_dataset():
    """
    Kiem tra file loi, file 0KB, sai dinh dang.
    Khong xoa vinh vien, chi chuyen sang dataset_backup/corrupted/...
    """
    clear_screen()
    print("\n" + "=" * 70)
    print(color('yellow', 'DON DEP DU LIEU LOI CORRUPTED IMAGES'))
    print("=" * 70 + "\n")
    print("Dang quet toan bo dataset...\n")

    corrupt_files = []
    total_scanned = 0

    for split in SPLITS:
        split_path = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_path):
            continue

        for cls in CLASSES:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                continue

            for file_name in os.listdir(cls_path):
                if file_name.startswith('.'):
                    continue

                file_path = os.path.join(cls_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                total_scanned += 1

                try:
                    if os.path.getsize(file_path) == 0:
                        corrupt_files.append((file_path, split, cls, 'Dung luong 0KB'))
                        continue
                except OSError:
                    corrupt_files.append((file_path, split, cls, 'Khong doc duoc file'))
                    continue

                if not is_image_file(file_name):
                    corrupt_files.append((file_path, split, cls, 'Sai dinh dang'))
                    continue

                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception:
                    corrupt_files.append((file_path, split, cls, 'File anh hong'))

    print(f"Da quet tong cong: {total_scanned} files.")

    if not corrupt_files:
        print(color('green', 'Dataset sach, khong tim thay file loi.'))
        input("\nNhan Enter de quay lai menu...")
        return

    print(color('red', f"Phat hien {len(corrupt_files)} file loi:"))
    for file_path, _, _, reason in corrupt_files[:10]:
        print(f"  - {os.path.basename(file_path)}: {reason}")

    if len(corrupt_files) > 10:
        print(f"  ... va {len(corrupt_files) - 10} file khac.")

    print("\nCac file loi se duoc chuyen sang thu muc backup, khong xoa vinh vien.")
    confirm = input("Ban co muon chuyen cac file loi sang backup khong? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Da huy thao tac.")
        input("\nNhan Enter de quay lai menu...")
        return

    moved = 0
    for file_path, split, cls, _ in corrupt_files:
        try:
            backup_class_dir = os.path.join(BACKUP_DIR, 'corrupted', split, cls)
            safe_move(file_path, backup_class_dir)
            moved += 1
        except Exception as e:
            print(f"Loi khi chuyen {file_path}: {e}")

    print(color('green', f"Da chuyen {moved} file loi sang backup."))
    print(f"Thu muc backup: {os.path.join(BACKUP_DIR, 'corrupted')}")
    input("\nNhan Enter de quay lai menu...")


def balance_dataset():
    """
    Can bang du lieu train bang undersampling.
    Anh thua duoc chuyen sang dataset_backup/balanced_removed/train/class_name,
    khong xoa vinh vien.
    """
    clear_screen()
    print("\n" + "=" * 70)
    print(color('cyan', 'CAN BANG DU LIEU TRAIN UNDERSAMPLING'))
    print("=" * 70 + "\n")

    train_path = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_path):
        print("Khong tim thay thu muc train.")
        input("\nNhan Enter de quay lai menu...")
        return

    stats = get_statistics()['train']

    print("So luong anh hien tai trong tap train:")
    for cls in CLASSES:
        print(f"  - {cls.upper():12s}: {stats.get(cls, 0)}")

    valid_counts = [count for count in stats.values() if count > 0]
    if not valid_counts:
        print("\nTap train dang trong.")
        input("\nNhan Enter de quay lai menu...")
        return

    min_count = min(valid_counts)

    print(f"\nNeu muon can bang tuyet doi, moi class nen giu: {min_count} anh")
    print("Ban co the nhap so khac, vi du 2000 hoac 3000 anh/class.")

    choice = input(f"\nNhap so anh muon giu lai cho moi class (Enter = {min_count}): ").strip()

    if choice == '':
        target_count = min_count
    else:
        try:
            target_count = int(choice)
        except ValueError:
            print(color('red', 'So khong hop le.'))
            input("\nNhan Enter de quay lai menu...")
            return

    if target_count <= 0:
        print("So luong phai lon hon 0.")
        input("\nNhan Enter de quay lai menu...")
        return

    total_to_move = 0
    move_plan = {}

    for cls in CLASSES:
        current_count = stats.get(cls, 0)
        if current_count > target_count:
            to_move = current_count - target_count
            total_to_move += to_move
            move_plan[cls] = to_move

    if total_to_move == 0:
        print(color('green', f"Du lieu da dat gioi han {target_count}, khong can chuyen them."))
        input("\nNhan Enter de quay lai menu...")
        return

    print(color('yellow', f"\nSe chuyen ngau nhien {total_to_move} anh sang backup:"))
    for cls, num in move_plan.items():
        print(f"  - {cls.upper()}: chuyen {num} anh")

    print("\nLuu y: Anh khong bi xoa vinh vien.")
    print(f"Thu muc backup: {os.path.join(BACKUP_DIR, 'balanced_removed', 'train')}")

    confirm = input(f"\nBan co chac muon chuyen {total_to_move} anh sang backup khong? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Da huy thao tac can bang.")
        input("\nNhan Enter de quay lai menu...")
        return

    moved = 0
    for cls, to_move in move_plan.items():
        cls_path = os.path.join(train_path, cls)
        files = [
            f for f in os.listdir(cls_path)
            if is_image_file(f) and os.path.isfile(os.path.join(cls_path, f))
        ]

        files_to_move = random.sample(files, to_move)
        backup_class_dir = os.path.join(BACKUP_DIR, 'balanced_removed', 'train', cls)

        for file_name in files_to_move:
            src = os.path.join(cls_path, file_name)
            try:
                safe_move(src, backup_class_dir)
                moved += 1
            except Exception as e:
                print(f"Loi khi chuyen {src}: {e}")

    print(color('green', f"\nDa chuyen thanh cong {moved} anh sang backup."))
    print("Du lieu train da duoc can bang theo muc da chon.")
    input("\nNhan Enter de quay lai menu...")


def main_menu():
    while True:
        clear_screen()
        print("=" * 70)
        print(color('primary', 'TOOL QUAN LY VA DON DEP DATASET'))
        print("=" * 70)
        print("1. Xem thong ke dataset")
        print("2. Don dep anh loi, chuyen sang backup")
        print("3. Can bang train, chuyen anh thua sang backup")
        print("4. Thoat")
        print("=" * 70)

        choice = input("Nhap lua chon cua ban (1-4): ").strip()

        if choice == '1':
            stats = get_statistics()
            show_statistics(stats)
        elif choice == '2':
            clean_dataset()
        elif choice == '3':
            balance_dataset()
        elif choice == '4':
            print("Da thoat tool.")
            break
        else:
            print("Lua chon khong hop le, vui long nhap lai.")
            input("Nhan Enter de tiep tuc...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nDa thoat.")
        sys.exit(0)
