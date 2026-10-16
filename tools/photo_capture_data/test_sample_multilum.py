import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from sample_multilum import sample, extract_photos, DIRECTIONS


class LightingCaptureTests(unittest.TestCase):
    def test_budget_fails_before_creating_output_or_contacting_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "photos"
            with self.assertRaisesRegex(ValueError, "budget"):
                sample(output, 2**20)
            self.assertFalse(output.exists())

    def test_selected_jpeg_bytes_survive_extraction_and_previous_work_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            buffer = BytesIO()
            Image.new("RGB", (3000, 2000), (100, 90, 80)).save(buffer, format="JPEG")
            data = buffer.getvalue()
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as source:
                for direction in DIRECTIONS:
                    source.writestr(f"scene/dir_{direction}_mip1.jpg", data)
                source.writestr("../../do-not-extract.txt", b"ignored")
            rows = extract_photos(archive, root / "review", "scene")
            self.assertEqual(len(rows), 5)
            self.assertEqual((root / "review" / rows[0]["file"]).read_bytes(), data)
            self.assertFalse((root / "do-not-extract.txt").exists())
            with self.assertRaisesRegex(ValueError, "Preserving"):
                extract_photos(archive, root / "review", "scene")
            self.assertEqual((root / "review" / rows[0]["file"]).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
