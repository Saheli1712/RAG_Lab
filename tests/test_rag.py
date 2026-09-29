import unittest
from unittest.mock import AsyncMock, patch
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from api.index import app
from api.rag import DocumentError, cosine, extract_pages, make_chunks, top_matches


class RagUnitTests(unittest.TestCase):
    def test_chunk_overlap_and_page_provenance(self):
        chunks = make_chunks([{"page": 3, "text": "word " * 450}])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["page"] == 3 for c in chunks))
        self.assertTrue(set(chunks[0]["text"].split()) & set(chunks[1]["text"].split()))

    def test_cosine_ranking(self):
        chunks = [{"id": 1, "page": 1, "text": "A"}, {"id": 2, "page": 2, "text": "B"}]
        self.assertEqual(top_matches(chunks, [[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0])[0]["page"], 2)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0)

    def test_bad_pdf(self):
        with self.assertRaises(DocumentError):
            extract_pages(b"not a PDF")

    def test_empty_pdf_explains_ocr(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buffer = BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(DocumentError, "OCR"):
            extract_pages(buffer.getvalue())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertTrue(self.client.get("/api/health").json()["ok"])

    def test_invalid_upload(self):
        response = self.client.post("/api/ingest", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
        self.assertEqual(response.status_code, 400)

    def test_missing_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            response = self.client.post("/api/embed-question", json={"question": "What happened?"})
        self.assertEqual(response.status_code, 401)

    def test_question_embedding_mocked_provider(self):
        with patch("api.index.provider_post", new=AsyncMock(return_value={"data": [{"index": 0, "embedding": [0.1, 0.9]}]})):
            response = self.client.post("/api/embed-question", json={"question": "Where?"}, headers={"X-API-Key": "sk-test"})
        self.assertEqual(response.json()["vector"], [0.1, 0.9])

    def test_pdf_to_retrieval_to_answer_with_mocked_provider(self):
        pdf = (Path(__file__).parent / "fixtures" / "sample.pdf").read_bytes()
        with patch("api.index.provider_post", new=AsyncMock(side_effect=[
            {"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [0.0, 1.0]}]},
            {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            {"choices": [{"message": {"content": "Five business days [p. 1]."}}]},
        ])):
            ingested = self.client.post("/api/ingest", files={"file": ("sample.pdf", pdf, "application/pdf")}, headers={"X-API-Key": "sk-test"})
            self.assertEqual(ingested.status_code, 200, ingested.text)
            doc = ingested.json()
            self.assertEqual(doc["pages"], 2)
            self.assertEqual(doc["chunks"][0]["page"], 1)
            question = "What is the approval target?"
            vector = self.client.post("/api/embed-question", json={"question": question}, headers={"X-API-Key": "sk-test"}).json()["vector"]
            matches = top_matches(doc["chunks"], doc["vectors"], vector)
            self.assertEqual(matches[0]["page"], 1)
            answer = self.client.post("/api/answer", json={"question": question, "passages": matches}, headers={"X-API-Key": "sk-test"})
            self.assertEqual(answer.status_code, 200)
            self.assertIn("[p. 1]", answer.json()["answer"])


if __name__ == "__main__":
    unittest.main()
