from PyPDF2 import PdfReader
import pickle

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Create a PdfReader object to read the PDF file
pdf_path = "./hrmanagement.pdf"
pdf_reader = PdfReader(pdf_path)
# If the pdf is uploaded
if pdf_reader:
    reader = pdf_reader
    _raw_text = []

    # Looping through the pdf
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            _raw_text.append(text)
    raw_text = ''.join(_raw_text)

    # Splitting the text into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(text, embeddings)
    docsearch.save_local('./document_index')
