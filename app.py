import argparse
import math
import re
from collections import Counter

from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import sent_tokenize, word_tokenize


def summarize_dated_list(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries = [
        line
        for line in lines
        if re.search(r"\([^)]*(?:พ\.ศ\.|\d{4}|[๐-๙]{4})", line)
        and re.search(r"(?:–|—|-|ถึง)", line)
    ]
    if len(entries) < 4 or len(entries) / max(1, len(lines)) < 0.5:
        return None

    years = re.findall(r"(?<!\d)(?:\d{4}|[๐-๙]{4})(?!\d)", text)
    thai_digits = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    normalized_years = [year.translate(thai_digits) for year in years]
    if not normalized_years:
        return "เนื้อหานี้รวบรวมรายชื่อบุคคลพร้อมช่วงเวลาของแต่ละรายการ"

    end = "ปัจจุบัน" if "ปัจจุบัน" in text else f"พ.ศ. {normalized_years[-1]}"
    return f"เนื้อหานี้รวบรวมรายชื่อบุคคลพร้อมช่วงเวลาของแต่ละรายการ ตั้งแต่ พ.ศ. {normalized_years[0]} ถึง {end}"


def summarize_extractive(text: str, max_sentences: int, max_summary_chars: int) -> str:
    sentences = [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
    if len(sentences) <= 1:
        return text.strip()

    stopwords = thai_stopwords()
    sentence_tokens = []
    for sentence in sentences:
        tokens = [
            token.lower()
            for token in word_tokenize(sentence, keep_whitespace=False)
            if len(token) > 1
            and any(char.isalnum() for char in token)
            and token not in stopwords
        ]
        sentence_tokens.append(tokens)

    document_frequency = Counter(
        token for tokens in sentence_tokens for token in set(tokens)
    )
    term_frequency = Counter(token for tokens in sentence_tokens for token in tokens)
    sentence_scores = []
    sentence_count = len(sentences)
    for index, tokens in enumerate(sentence_tokens):
        if not tokens:
            score = 0.0
        else:
            score = sum(
                (1 + math.log(term_frequency[token]))
                * (1 + math.log((sentence_count + 1) / (document_frequency[token] + 1)))
                for token in set(tokens)
            ) / math.sqrt(len(tokens))
        sentence_scores.append(score * (1 + 0.12 / (index + 1)))

    summary_budget = min(max_summary_chars, max(160, int(len(text) * 0.38)))
    ranked_indexes = sorted(range(sentence_count), key=lambda index: sentence_scores[index], reverse=True)
    selected_indexes = []
    selected_tokens = []
    selected_length = 0

    for index in ranked_indexes:
        tokens = set(sentence_tokens[index])
        if tokens and any(
            len(tokens & existing) / max(1, len(tokens | existing)) >= 0.72
            for existing in selected_tokens
        ):
            continue

        sentence_length = len(sentences[index])
        if selected_indexes and selected_length + sentence_length + 1 > summary_budget:
            continue

        selected_indexes.append(index)
        selected_tokens.append(tokens)
        selected_length += sentence_length + 1
        if len(selected_indexes) >= max_sentences:
            break

    if not selected_indexes:
        selected_indexes = [max(range(sentence_count), key=lambda index: sentence_scores[index])]
    return " ".join(sentences[index] for index in sorted(selected_indexes))


def validate_summary(source: str, summary: str) -> str | None:
    if not summary:
        return "โมเดลไม่ได้สร้างบทสรุป"

    source_has_thai = any(0x0E00 <= ord(char) <= 0x0E7F for char in source)
    if source_has_thai and any(
        char.isalpha() and not 0x0E00 <= ord(char) <= 0x0E7F
        for char in summary
    ):
        return "ผลลัพธ์มีอักษรจากภาษาอื่นปะปน"

    number_pattern = r"\d+(?:[.,]\d+)*|[๐-๙]+"
    thai_digits = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

    def numbers(text: str) -> list[str]:
        return [
            value.translate(thai_digits).replace(",", "")
            for value in re.findall(number_pattern, text)
        ]

    if numbers(source) != numbers(summary):
        return "ตัวเลขหรือวันที่ในบทสรุปไม่ตรงกับต้นฉบับ"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="ย่อบทความภาษาไทยโดยคัดประโยคสำคัญจากต้นฉบับ")
    parser.add_argument("--text", help="เนื้อหาที่ต้องการสรุป (ถ้าไม่ระบุจะถามใน terminal)")
    parser.add_argument("--max-sentences", type=int, default=5)
    parser.add_argument("--max-summary-chars", type=int, default=1200)
    args = parser.parse_args()

    text = (args.text or input("เนื้อหาที่ต้องการสรุป: ")).strip()
    if not text:
        parser.error("กรุณาระบุเนื้อหาด้วย --text หรือพิมพ์เมื่อโปรแกรมถาม")
    if args.max_sentences < 1 or args.max_summary_chars < 1:
        parser.error("--max-sentences และ --max-summary-chars ต้องมากกว่า 0")

    summary = summarize_dated_list(text) or summarize_extractive(
        text,
        args.max_sentences,
        args.max_summary_chars,
    )
    print("\nสรุป:")
    print(summary)


if __name__ == "__main__":
    main()