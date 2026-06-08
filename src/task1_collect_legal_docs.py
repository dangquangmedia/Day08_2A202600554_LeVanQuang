"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}
MIN_REQUIRED_FILES = 3
MIN_FILE_SIZE_BYTES = 1024

EXPECTED_DOCUMENTS = [
    {
        "name": "luat-phong-chong-ma-tuy-2021.pdf",
        "description": "Luật Phòng, chống ma túy 2021 (73/2021/QH15)",
    },
    {
        "name": "nghi-dinh-105-2021.pdf",
        "description": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
    },
    {
        "name": "bo-luat-hinh-su-2015-sua-doi-2017.pdf",
        "description": "Văn bản hợp nhất Bộ luật Hình sự, gồm các tội phạm về ma túy",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def list_legal_files() -> list[Path]:
    """Liệt kê các file pháp luật hợp lệ trong data/landing/legal/."""
    if not DATA_DIR.exists():
        return []

    return sorted(
        [
            filepath
            for filepath in DATA_DIR.iterdir()
            if filepath.is_file() and filepath.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def validate_legal_files() -> dict:
    """Kiểm tra Task 1 đã có đủ tối thiểu 3 file pháp luật không rỗng."""
    setup_directory()
    files = list_legal_files()
    small_files = [
        filepath.name
        for filepath in files
        if filepath.stat().st_size <= MIN_FILE_SIZE_BYTES
    ]
    missing_expected = [
        doc["name"]
        for doc in EXPECTED_DOCUMENTS
        if not (DATA_DIR / doc["name"]).exists()
    ]
    missing_count = max(0, MIN_REQUIRED_FILES - len(files))

    return {
        "is_ready": (
            len(files) >= MIN_REQUIRED_FILES
            and not small_files
        ),
        "file_count": len(files),
        "missing_count": missing_count,
        "small_files": small_files,
        "missing_expected": missing_expected,
        "files": files,
    }


def print_report(result: dict):
    """In báo cáo ngắn gọn để người học biết Task 1 đang ở trạng thái nào."""
    print("\nTask 1: Thu thập văn bản pháp luật")
    print(f"Yêu cầu: >= {MIN_REQUIRED_FILES} file PDF/DOC/DOCX, mỗi file > 1KB")
    print(f"Tìm thấy: {result['file_count']} file")

    for filepath in result["files"]:
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✓ {filepath.name} ({size_kb:.1f} KB)")

    if result["missing_expected"]:
        print("\nThiếu file gợi ý:")
        for filename in result["missing_expected"]:
            print(f"  - {filename}")

    if result["small_files"]:
        print("\nFile quá nhỏ, cần kiểm tra lại:")
        for filename in result["small_files"]:
            print(f"  - {filename}")

    if result["is_ready"]:
        print("\n✓ Task 1 đã sẵn sàng cho các bước convert/chunking tiếp theo.")
    else:
        print("\n! Task 1 chưa đủ dữ liệu. Hãy bổ sung file pháp luật hợp lệ.")


def run_collection():
    """
    Hoàn thành Task 1 bằng cách kiểm kê bộ tài liệu đã thu thập.

    Các PDF hiện đã nằm trong data/landing/legal/, nên script không tải lại qua
    mạng để tránh phụ thuộc link ngoài. Nếu muốn thay nguồn, chỉ cần đặt thêm
    PDF/DOC/DOCX vào thư mục này và chạy lại file.
    """
    result = validate_legal_files()
    print_report(result)
    return result


if __name__ == "__main__":
    run_collection()
