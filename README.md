# MedRAG --- Medical Retrieval-Augmented Generation

**Student:** GACEMI RAIDA\
**Project:** Medical question-answering prototype using
Retrieval-Augmented Generation (RAG)\
**Notebook:** `MedRAG_Annotated.ipynb`

## Introduction

MedRAG is an educational prototype that combines semantic retrieval and
large language model generation to answer questions using a selected
collection of medical PDF documents.

Instead of asking a language model to answer from its pretrained
knowledge alone, the system first searches the medical document
collection for relevant passages. It then supplies those passages,
together with the user's question and source metadata, to an
instruction-tuned language model to generate a response.

The project demonstrates the main stages of a RAG pipeline: PDF text
extraction, chunking, embedding generation, vector indexing, semantic
retrieval, prompt construction and answer generation.

## Architecture

``` text
Medical PDF documents
        |
        v
Text extraction (PyMuPDF)
        |
        v
Medical document selection
        |
        v
Chunking (900 characters, 150 overlap)
        |
        v
Embeddings (multilingual MiniLM, 384 dimensions)
        |
        v
FAISS vector index
        |
User question
        |
        v
Question embedding + similarity search
        |
        v
Retrieved passages with source and page metadata
        |
        v
Prompt construction
        |
        v
Qwen2.5-3B-Instruct
        |
        v
Generated answer
```

## Technologies

  -----------------------------------------------------------------------------
  Technology                                Purpose
  ----------------------------------------- -----------------------------------
  Python                                    Main programming language

  Google Colab                              Notebook execution environment

  PyMuPDF                                   Extract selectable text from PDFs

  LangChain text splitters                  Split page text into overlapping
                                            chunks

  Sentence Transformers                     Generate semantic embeddings

  `paraphrase-multilingual-MiniLM-L12-v2`   Embedding model (384 dimensions)

  FAISS                                     Vector similarity search

  Hugging Face Transformers                 Load and run the language model

  Qwen2.5-3B-Instruct                       Generate answers from retrieved
                                            context

  PyTorch                                   Tensor operations and model
                                            inference
  -----------------------------------------------------------------------------

## Current corpus and configuration

The notebook discovers PDF files in a Google Drive folder and extracts
selectable text. It excludes `State-of-wildfires-2024-2025.pdf` from the
medical corpus and uses the three medical PDFs.

The recorded run produced:

-   **1,584** extracted medical pages.
-   **6,615** text chunks.
-   **900 characters** per chunk (configured maximum).
-   **150 characters** of chunk overlap.
-   **384 dimensions** per embedding.
-   **Top 3** retrieved passages by default.
-   FAISS `IndexFlatIP` with normalized vectors.

These are results from the notebook's saved run, not guaranteed counts
for other input files or future executions.

## Repository contents

``` text
MedRAG/
├── MedRAG_Annotated.ipynb
├── README.md
└── requirements.txt
```

The notebook includes explanations of the workflow and code cells for
each stage.

## Running the project

### 1. Open the notebook

Upload `MedRAG_Annotated.ipynb` to Google Colab or open it from your
GitHub repository using Colab.

### 2. Install dependencies

The first notebook cell installs the required packages:

``` bash
pip install pymupdf langchain-text-splitters sentence-transformers faiss-cpu transformers accelerate
```

### 3. Prepare Google Drive

Create the following folder structure in Google Drive, or update
`PDF_FOLDER` in the notebook to match your own location:

``` text
MyDrive/
└── MedRAG/
    └── medrag_data/
        ├── medical_document_1.pdf
        ├── medical_document_2.pdf
        └── medical_document_3.pdf
```

The notebook discovers PDF files in the configured folder and its
subfolders. It currently excludes a PDF named
`State-of-wildfires-2024-2025.pdf` from the medical corpus. If you use
different files, review the corpus-selection cell.

### 4. Select a GPU runtime

A GPU is recommended for loading and running Qwen2.5-3B-Instruct. The
saved experiment used a Tesla T4 GPU with FP16 model weights.
Availability of GPU hardware depends on the Colab account and runtime.

### 5. Run the notebook in order

Execute the cells from top to bottom. The notebook will:

1.  Install dependencies and import libraries.
2.  Mount Google Drive and locate PDF files.
3.  Inspect PDFs and extract selectable text.
4.  Select the medical documents.
5.  Split the text into chunks.
6.  Generate and validate embeddings.
7.  Build the FAISS index.
8.  Retrieve passages for a question.
9.  Load Qwen2.5-3B-Instruct.
10. Construct the context and prompt.
11. Generate an answer from the retrieved context.

## How retrieval works

The same Sentence Transformers model encodes both document chunks and
user questions. FAISS searches for the closest stored vectors. The
notebook uses normalized embeddings with `IndexFlatIP`; therefore,
inner-product scores correspond to cosine similarity.

A similarity score is a measure of vector closeness. It does **not**
prove that a passage answers the question. The retrieved text should
always be inspected.

## How generation works

The retrieved passages and their source metadata are formatted into a
context string. A prompt then combines this context with the user's
question and instructions for Qwen.

Qwen2.5-3B-Instruct generates a natural-language response based on the
prompt. The model is not fine-tuned on the PDFs in this project; the
documents are provided as context at inference time.

## Limitations and responsible use

-   The notebook extracts selectable PDF text only. OCR is not
    implemented, so scanned image-only pages may not be processed.
-   Retrieval can return passages that mention a topic without directly
    answering the question.
-   Qwen can misinterpret or overgeneralize retrieved information,
    including confusing a complication's symptoms with general disease
    symptoms.
-   Prompt instructions do not guarantee factual correctness or reliable
    citations.
-   The system has not undergone clinical validation and is not intended
    for diagnosis, treatment decisions or emergency guidance.
-   Always verify generated answers against the original documents and
    qualified medical sources.

**MedRAG is an educational prototype, not a substitute for professional
medical judgment.**

## Data and model access

The medical PDFs and Qwen model weights are not included in this
repository. Users must provide their own authorized PDF files and allow
Transformers to download the model from Hugging Face when the notebook
runs.

Do not upload copyrighted books, private documents, credentials, API
tokens or large model-weight files to the repository unless you have the
rights and a clear reason to do so.

## Author

**GACEMI RAIDA**
