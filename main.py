import io
import os
import re
import cv2
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from google import genai

app = FastAPI(
    title="Enterprise Financial Document Extraction API",
    description="Production-grade, memory-optimized document parser with automated LLM schema healing.",
    version="3.6.3"
)

# FIXED: Explicit environment lookup resolves internal container variable mapping issues
RENDER_KEY = os.environ.get("GEMINI_API_KEY")
if not RENDER_KEY:
    raise ValueError("CRITICAL: GEMINI_API_KEY environment variable is missing on Render dashboard configuration panels!")

ai_client = genai.Client(api_key=RENDER_KEY)  

class W2TaxData(BaseModel):
    box_a_ssn: Optional[str] = Field(None, description="Employee Social Security Number (format: XXX-XX-XXXX)")
    box_b_ein: Optional[str] = Field(None, description="Employer Identification Number (format: XX-XXXXXXX)")
    box_1_wages: Optional[float] = Field(None, description="Wages, tips, other compensation")
    box_2_federal_tax: Optional[float] = Field(None, description="Federal income tax withheld")

class W2ParserResponse(BaseModel):
    status: str
    extraction_engine: str
    document_type: str
    data: W2TaxData

class SECBalanceSheetRow(BaseModel):
    line_item_name: str
    current_year_value: Optional[float] = Field(None, description="Most recent year value")
    prior_year_value: Optional[float] = Field(None, description="Previous year value")

class SECParserResponse(BaseModel):
    status: str
    extraction_engine: str
    balance_sheet: List[SECBalanceSheetRow]

def clean_currency(val: Optional[str]) -> Optional[float]:
    """Converts dirty ledger and accounting strings safely into mathematical floats."""
    if not val: 
        return None
    try:
        cleaned = re.sub(r'[^\d\.\-\(\)]', '', val.strip())
        if '(' in cleaned or ')' in cleaned:
            cleaned = '-' + cleaned.replace('(', '').replace(')', '')
        return float(cleaned) if cleaned else None
    except ValueError:
        return None

def process_scanned_pdf_via_ocr(pdf_bytes: bytes) -> str:
    """Memory-optimized rasterization using PyMuPDF and lightweight Tesseract OCR."""
    flattened_text = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY) if pix.n == 3 else img_data
        
        text = pytesseract.image_to_string(img_gray)
        if text:
            flattened_text.append(text)
                
    return " ".join(flattened_text)

def llm_fallback_w2(text_context: str) -> W2TaxData:
    """Uses structured schema enforcement to heal parsing anomalies instantly."""
    truncated_context = text_context[:8000]
    prompt = f"Extract W-2 tax variables exactly as specified by the schema from this raw text content: {truncated_context}"
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": W2TaxData,
            "temperature": 0.0
        }
    )
    return W2TaxData.model_validate_json(response.text)

def llm_fallback_sec(text_context: str) -> List[SECBalanceSheetRow]:
    """Extracts balance sheets from unstructured texts using semantic matrix matching."""
    class SECContainer(BaseModel):
        rows: List[SECBalanceSheetRow]

    truncated_context = text_context[:25000]
    prompt = f"Locate the balance sheet statements and extract the line items, matching current and prior years: {truncated_context}"
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SECContainer,
            "temperature": 0.0
        }
    )
    container = SECContainer.model_validate_json(response.text)
    return container.rows

@app.post("/v1/parse/w2", response_model=W2ParserResponse)
async def parse_w2(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Supported payload type is strictly PDF.")
        
    pdf_content = await file.read()
    raw_text_stream = ""
    engine_used = "Deterministic_Native"
    
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        all_pages_text = []
        for page in pdf.pages:
            words = page.extract_words()
            if words:
                tolerance = 3
                lines = {}
                for w in words:
                    top_rounded = int(w['top'] / tolerance) * tolerance
                    lines.setdefault(top_rounded, []).append(w)
                
                sorted_lines = []
                for top in sorted(lines.keys()):
                    sorted_row = sorted(lines[top], key=lambda w: w['x0'])
                    sorted_lines.append(" ".join([w['text'] for w in sorted_row]))
                all_pages_text.append(" ".join(sorted_lines))
        raw_text_stream = " ".join(all_pages_text)
        
    if len(raw_text_stream.strip()) < 20:
        engine_used = "Deterministic_OCR"
        raw_text_stream = process_scanned_pdf_via_ocr(pdf_content)

    ssn_match = re.search(r'\b\d{3}\s*-\s*\d{2}\s*-\s*\d{4}\b', raw_text_stream)
    ein_match = re.search(r'\b\d{2}\s*-\s*\d{7}\b', raw_text_stream)
    wages_match = re.search(r'(?:Wages|Box 1|box 1)[\s\S]*?([\d,]+\.\d{2})', raw_text_stream, re.IGNORECASE)
    fed_tax_match = re.search(r'(?:Federal|Box 2|box 2)[\s\S]*?([\d,]+\.\d{2})', raw_text_stream, re.IGNORECASE)

    extracted_data = W2TaxData(
        box_a_ssn=re.sub(r'\s+', '', ssn_match.group(0)) if ssn_match else None,
        box_b_ein=re.sub(r'\s+', '', ein_match.group(0)) if ein_match else None,
        box_1_wages=clean_currency(wages_match.group(1)) if wages_match else None,
        box_2_federal_tax=clean_currency(fed_tax_match.group(1)) if fed_tax_match else None
    )

    if not extracted_data.box_1_wages or not extracted_data.box_2_federal_tax:
        try:
            extracted_data = llm_fallback_w2(raw_text_stream)
            engine_used += " + LLM_Healing_Layer"
        except Exception:
            pass

    return W2ParserResponse(
        status="success",
        extraction_engine=engine_used,
        document_type="IRS_FORM_W2",
        data=extracted_data
    )

@app.post("/v1/parse/sec-10k", response_model=SECParserResponse)
async def parse_sec_10k(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Supported payload type is strictly PDF.")
        
    pdf_content = await file.read()
    parsed_rows = []
    engine_used = "Tabular_Geometry"
    
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages[:8]:  
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean_row = [cell.strip() for cell in row if cell and cell.strip() != ""]
                    if len(clean_row) >= 2:
                        label = clean_row
                        if any(k in label.lower() for k in ["cash", "assets", "liabilities", "equity", "goodwill"]):
                            numeric_candidates = [clean_currency(c) for c in clean_row[1:] if clean_currency(c) is not None]
                            current_val = numeric_candidates if len(numeric_candidates) > 0 else None
                            prior_val = numeric_candidates if len(numeric_candidates) > 1 else None
                            
                            parsed_rows.append(SECBalanceSheetRow(
                                line_item_name=label,
                                current_year_value=current_val,
                                prior_year_value=prior_val
                            ))
            if parsed_rows: 
                break

    if not parsed_rows:
        engine_used = "Semantic_LLM_Table_Extraction"
        full_text_buffer = ""
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            full_text_buffer = " ".join([p.extract_text() or "" for p in pdf.pages[:8]])
        try:
            parsed_rows = llm_fallback_sec(full_text_buffer)
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=f"All parsing extraction strategies exhausted: {str(err)}"
            )

    return SECParserResponse(
        status="success",
        extraction_engine=engine_used,
        balance_sheet=parsed_rows
    )
