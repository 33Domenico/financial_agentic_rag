from pathlib import Path
from pypdf import PdfReader

def pdf_reader(pdf_path):
    try:
        reader=PdfReader(pdf_path)
        pages=reader.pages
        doc_name= Path(pdf_path).stem
        pages_records=[]
        for i,p in enumerate(pages):
            try:
                text=p.extract_text()
                page_dic={"page_num":i,
                        "doc_name":doc_name,
                        "n_chars":len(text),
                        "text":text}
                pages_records.append(page_dic)
            except Exception as e:
                print(f"Error reading page {i} of {pdf_path}: {e}")
                continue
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        pages_records=[]
    return pages_records