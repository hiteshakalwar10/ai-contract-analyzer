# AI Contract Analyzer

An AI tool that reads a contract PDF and pulls out the important stuff automatically - parties involved, dates, payment terms, key clauses, obligations and risky terms. You can also download the analysis as a PDF or DOCX report.

## Why I built this

Going through long contracts manually to find key dates, payment terms and risky clauses takes time. This automates that first pass so you get everything structured upfront.

## Features

- Upload a contract PDF
- Executive summary
- Parties involved
- Important dates
- Payment terms
- Key clauses
- Obligations for each party
- Risk analysis (auto-renewal, liability caps, indemnification, etc.)
- Download report as PDF
- Download report as DOCX

## Tech stack

Streamlit for the frontend, Groq (Llama 3.3 70B) as the LLM, PyMuPDF for text extraction and ReportLab/python-docx for report generation.

## How it works

The PDF is parsed with PyMuPDF. The full contract text is sent to the LLM with a structured prompt to extract the summary, parties, dates, payment terms, clauses, obligations and risks as JSON. The final analysis can be exported as a PDF or DOCX.

## Running locally

pip install -r requirements.txt

Set your Groq API key:

export GROQ_API_KEY=your_key_here

Run the app:

streamlit run app.py

## Live demo

https://ai-contract-analyzer-5z6o9evc3dso4rwjectfyx.streamlit.app/

## Note

Built as a portfolio project to explore RAG pipelines and structured extraction from unstructured legal documents. Not meant for real legal use - always have an actual lawyer review real contracts.
