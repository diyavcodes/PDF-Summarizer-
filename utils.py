import PyPDF2
import re
from transformers import AutoTokenizer
import yake


tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return full_text.strip()

def simple_sentence_splitter(text):
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_text(text, max_tokens=800):
    sentences = simple_sentence_splitter(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_len = len(sentence_tokens)

        if token_len > max_tokens:
            sentence_tokens = sentence_tokens[:max_tokens]
            sentence = tokenizer.decode(sentence_tokens, skip_special_tokens=True)
            token_len = len(sentence_tokens)

        if current_tokens + token_len <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += token_len
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = token_len

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if len(tokenizer.encode(chunk)) <= max_tokens]

def extract_keywords(text, max_keywords=20):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]
