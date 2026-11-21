import math

import os

import re

import uuid

from pathlib import Path

from typing import Optional



import chromadb

import pymupdf

from dotenv import load_dotenv

from fastapi import FastAPI, File, HTTPException, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from google import genai

from google.genai import types

from neo4j import GraphDatabase

from pydantic import BaseModel, Field



# ------------------------------------------------------------

# Configuration

# ------------------------------------------------------------



BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)



MAX_FILE_SIZE = 20 * 1024 * 1024

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSIONS = 768

CHROMA_COLLECTION_NAME = "document_chunks_gemini_v2"



load_dotenv(BASE_DIR / ".env")



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")



gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None



NEO4J_URI = os.getenv("NEO4J_URI")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")



neo4j_driver = None

if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:

    try:

        neo4j_driver = GraphDatabase.driver(

            NEO4J_URI,

            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),

        )

    except Exception as exc:

        print(f"Neo4j driver initialization failed: {exc}")



chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

chunks_collection = chroma_client.get_or_create_collection(

    name=CHROMA_COLLECTION_NAME,

    metadata={"hnsw:space": "cosine"},

)



app = FastAPI(

    title="GraphRAG Knowledge Explorer",

    description="AI-powered document and knowledge graph explorer",

    version="4.0.0",

)



allowed_origins = [

    "http://localhost:5173",

    "http://127.0.0.1:5173",
    "https://lustrous-croquembouche-55c58f.netlify.app",
]

# Add your deployed Netlify URL as an environment variable on Render:

# FRONTEND_URL=https://your-site.netlify.app

frontend_url = os.getenv("FRONTEND_URL")

if frontend_url:

    allowed_origins.append(frontend_url.rstrip("/"))



app.add_middleware(

    CORSMiddleware,

    allow_origins=allowed_origins,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)





# ------------------------------------------------------------

# Request models

# ------------------------------------------------------------



class SearchRequest(BaseModel):

    query: str = Field(..., min_length=2, max_length=1000)

    document_id: Optional[str] = None

    top_k: int = Field(default=5, ge=1, le=10)





class AskRequest(BaseModel):

    question: str = Field(..., min_length=2, max_length=4000)

    document_id: Optional[str] = None

    top_k: int = Field(default=5, ge=1, le=10)





# ------------------------------------------------------------

# Common helpers

# ------------------------------------------------------------



def validate_document_id(document_id: str) -> str:

    try:

        return str(uuid.UUID(document_id))

    except (ValueError, TypeError, AttributeError):

        raise HTTPException(status_code=400, detail="Invalid document ID")





def document_pdf_path(document_id: str) -> Path:

    return UPLOAD_DIR / f"{document_id}.pdf"





def document_text_path(document_id: str) -> Path:

    return UPLOAD_DIR / f"{document_id}.txt"





def normalize_vector(values):

    vector = [float(value) for value in values]

    magnitude = math.sqrt(sum(value * value for value in vector))

    if magnitude == 0:

        return vector

    return [value / magnitude for value in vector]





def require_gemini():

    if gemini_client is None:

        raise HTTPException(

            status_code=503,

            detail="Gemini API is not configured. Add GEMINI_API_KEY to the backend environment.",

        )





def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:

    """Create normalized Gemini embeddings in small batches."""

    require_gemini()

    vectors = []

    batch_size = 50



    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        response = gemini_client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=batch,

            config=types.EmbedContentConfig(

                task_type=task_type,

                output_dimensionality=EMBEDDING_DIMENSIONS,

            ),

        )

        if not response.embeddings:

            raise RuntimeError("Gemini returned no embeddings.")

        vectors.extend(normalize_vector(item.values) for item in response.embeddings)



    if len(vectors) != len(texts):

        raise RuntimeError("The number of embeddings did not match the number of texts.")

    return vectors





def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:

    try:

        with pymupdf.open(pdf_path) as pdf:

            pages = [page.get_text("text") for page in pdf]

            return "\n".join(pages).strip(), len(pdf)

    except Exception as exc:

        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")





def split_text_into_chunks(text: str) -> list[str]:

    text = re.sub(r"\s+", " ", text).strip()

    if not text:

        return []



    chunks = []

    start = 0

    while start < len(text):

        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):

            boundary = text.rfind(" ", start + int(CHUNK_SIZE * 0.7), end)

            if boundary > start:

                end = boundary

        piece = text[start:end].strip()

        if piece:

            chunks.append(piece)

        if end >= len(text):

            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks





def get_document_chunk_count(document_id: str) -> int:

    try:

        result = chunks_collection.get(

            where={"document_id": document_id},

            include=[],

        )

        return len(result.get("ids", []))

    except Exception:

        return 0





def process_document_chunks(document_id: str, text: str) -> int:

    chunks = split_text_into_chunks(text)

    if not chunks:

        return 0



    embeddings = embed_texts(chunks, "RETRIEVAL_DOCUMENT")

    ids = [f"{document_id}_chunk_{index}" for index in range(len(chunks))]

    metadatas = [

        {

            "document_id": document_id,

            "chunk_index": index,

            "total_chunks": len(chunks),

        }

        for index in range(len(chunks))

    ]



    # Replace this document's previous vectors in the new Gemini collection.

    try:

        chunks_collection.delete(where={"document_id": document_id})

    except Exception:

        pass



    chunks_collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings,

        metadatas=metadatas,

    )

    return len(chunks)





def load_document_text(document_id: str) -> str:

    path = document_text_path(document_id)

    if not path.exists():

        raise HTTPException(status_code=404, detail="Document text was not found")

    return path.read_text(encoding="utf-8")





# ------------------------------------------------------------

# Rule-based entity and relationship extraction

# ------------------------------------------------------------



KNOWN_ENTITIES = {

    "Northstar Analytics Pvt. Ltd.": "Organization",

    "BluePeak University": "Organization",

    "Meridian Devices Ltd.": "Organization",

    "City of Lakewood": "Organization",

    "Applied AI Laboratory": "Organization",

    "BluePeak University's Applied AI Laboratory": "Organization",

    "Computer Science Department": "Organization",

    "Meera Kulkarni": "Person",

    "Dr Kabir Shah": "Person",

    "Asha Rao": "Person",

    "Rohan Iyer": "Person",

    "Elena D'Souza": "Person",

    "Nikhil Verma": "Person",

    "Sara Fernandes": "Person",

    "RiverWatch": "Project",

    "GreenGrid": "Project",

    "Pune": "Location",

    "Maharashtra": "Location",

    "Nagpur": "Location",

    "Bengaluru": "Location",

    "Karnataka": "Location",

    "Aster River": "Location",

}





def extract_entities(text: str) -> list[dict]:

    found = []

    lowered = text.lower()

    for name, entity_type in KNOWN_ENTITIES.items():

        if name.lower() in lowered:

            found.append({"name": name, "type": entity_type})

    return found





def extract_relationships(text: str, entities: list[dict]) -> list[dict]:

    """Small, intentionally rule-based relationship extractor."""

    relationships = []

    entity_names = [entity["name"] for entity in entities]



    def add(source, target, relation, evidence):

        if source in entity_names and target in entity_names:

            item = {

                "source": source,

                "target": target,

                "type": relation,

                "evidence": evidence[:500],

            }

            if item not in relationships:

                relationships.append(item)



    patterns = [

        (r"(.{0,100})RiverWatch(.{0,180})", "RiverWatch", "RELATED_TO"),

        (r"(.{0,100})GreenGrid(.{0,180})", "GreenGrid", "RELATED_TO"),

    ]

    for pattern, project, relation in patterns:

        for match in re.finditer(pattern, text, re.IGNORECASE):

            sentence = match.group(0).strip()

            for name in entity_names:

                if name != project and name.lower() in sentence.lower():

                    add(name, project, relation, sentence)



    role_patterns = [

        (r"([A-Z][A-Za-z.' ]{2,50})\s+(?:is|was|serves as)\s+(?:the\s+)?CEO\s+of\s+([A-Z][A-Za-z0-9&.' ]{2,70})", "CEO_OF"),

        (r"([A-Z][A-Za-z.' ]{2,50})\s+(?:is|was)\s+(?:the\s+)?managing director\s+of\s+([A-Z][A-Za-z0-9&.' ]{2,70})", "MANAGING_DIRECTOR_OF"),

        (r"([A-Z][A-Za-z.' ]{2,50})\s+(?:is|was)\s+(?:the\s+)?sustainability officer\s+at\s+([A-Z][A-Za-z0-9&.' ]{2,70})", "WORKS_AT"),

        (r"([A-Z][A-Za-z.' ]{2,50})\s+(?:is|was)\s+(?:the\s+)?research director\s+at\s+([A-Z][A-Za-z0-9&.' ]{2,70})", "WORKS_AT"),

    ]

    for pattern, relation in role_patterns:

        for match in re.finditer(pattern, text):

            source_guess, target_guess = match.group(1).strip(), match.group(2).strip()

            source = next((n for n in entity_names if n.lower() == source_guess.lower()), None)

            target = next((n for n in entity_names if n.lower() == target_guess.lower()), None)

            if source and target:

                add(source, target, relation, match.group(0))



    # Co-occurrence-based links are useful for the small demonstration corpus.

    # They are labeled RELATED_TO, not asserted as a specific real-world relation.

    for sentence in re.split(r"(?<=[.!?])\s+", text):

        present = [name for name in entity_names if name.lower() in sentence.lower()]

        if len(present) >= 2:

            for index in range(len(present) - 1):

                add(present[index], present[index + 1], "RELATED_TO", sentence)



    return relationships





def store_graph_in_neo4j(document_id: str, entities: list[dict], relationships: list[dict]):

    if neo4j_driver is None:

        raise HTTPException(status_code=503, detail="Neo4j is not configured")



    with neo4j_driver.session(database=NEO4J_DATABASE) as session:

        session.run(

            "MATCH (n:Entity {document_id: $document_id}) DETACH DELETE n",

            document_id=document_id,

        )

        for entity in entities:

            session.run(

                """

                MERGE (e:Entity {name: $name, document_id: $document_id})

                SET e.type = $type

                """,

                name=entity["name"],

                document_id=document_id,

                type=entity["type"],

            )

        for relation in relationships:

            session.run(

                """

                MATCH (a:Entity {name: $source, document_id: $document_id})

                MATCH (b:Entity {name: $target, document_id: $document_id})

                MERGE (a)-[r:RELATED_TO {type: $type, document_id: $document_id}]->(b)

                SET r.evidence = $evidence

                """,

                source=relation["source"],

                target=relation["target"],

                document_id=document_id,

                type=relation["type"],

                evidence=relation.get("evidence", ""),

            )





# ------------------------------------------------------------

# Basic endpoints

# ------------------------------------------------------------



@app.get("/")

def home():

    return {

        "success": True,

        "message": "GraphRAG Knowledge Explorer API is running",

        "version": "4.0.0",

    }





@app.get("/health")

def health():

    return {

        "success": True,

        "status": "healthy",

        "gemini_configured": gemini_client is not None,

        "neo4j_configured": neo4j_driver is not None,

    }





@app.get("/api/neo4j/status")

def neo4j_status():

    if neo4j_driver is None:

        return {

            "success": False,

            "connected": False,

            "message": "Neo4j environment variables are not configured",

        }

    try:

        with neo4j_driver.session(database=NEO4J_DATABASE) as session:

            result = session.run("RETURN 1 AS result").single()

            return {

                "success": True,

                "connected": True,

                "message": "Successfully connected to Neo4j AuraDB!",

                "database": NEO4J_DATABASE,

                "connection_test": result["result"],

            }

    except Exception as exc:

        return {

            "success": False,

            "connected": False,

            "message": f"Neo4j connection failed: {str(exc)}",

        }





# ------------------------------------------------------------

# PDF upload and document management

# ------------------------------------------------------------



@app.post("/api/documents/upload")

async def upload_document(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(status_code=400, detail="Only PDF files are allowed")



    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(status_code=413, detail="Maximum file size is 20 MB")

    if not contents:

        raise HTTPException(status_code=400, detail="The uploaded file is empty")



    document_id = str(uuid.uuid4())

    pdf_path = document_pdf_path(document_id)

    pdf_path.write_bytes(contents)



    extracted_text, total_pages = extract_pdf_text(pdf_path)

    if not extracted_text:

        pdf_path.unlink(missing_ok=True)

        raise HTTPException(status_code=400, detail="No readable text was found in this PDF")



    text_path = document_text_path(document_id)

    text_path.write_text(extracted_text, encoding="utf-8")



    processing_message = "Text extracted. Embedding processing is pending."

    processed_chunks = 0

    try:

        processed_chunks = process_document_chunks(document_id, extracted_text)

        processing_message = "PDF uploaded, text extracted, and chunks embedded successfully."

    except Exception as exc:

        processing_message = f"PDF uploaded and text extracted; embedding processing failed: {exc}"



    return {

        "success": True,

        "message": processing_message,

        "document": {

            "document_id": document_id,

            "id": document_id,

            "filename": file.filename,

            "total_pages": total_pages,

            "total_characters": len(extracted_text),

            "total_chunks": processed_chunks,

            "processed": processed_chunks > 0,

        },

    }





@app.get("/api/documents")

def list_documents():

    documents = []

    for pdf_path in UPLOAD_DIR.glob("*.pdf"):

        document_id = pdf_path.stem

        text_path = document_text_path(document_id)

        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""

        try:

            with pymupdf.open(pdf_path) as pdf:

                total_pages = len(pdf)

        except Exception:

            total_pages = 0



        documents.append({

            "document_id": document_id,

            "id": document_id,

            "filename": pdf_path.name,

            "total_pages": total_pages,

            "total_characters": len(text),

            "total_chunks": get_document_chunk_count(document_id),

            "processed": get_document_chunk_count(document_id) > 0,

        })



    documents.sort(key=lambda item: item["filename"].lower())

    return {"success": True, "total_documents": len(documents), "documents": documents}





@app.delete("/api/documents/{document_id}")

def delete_document(document_id: str):

    document_id = validate_document_id(document_id)



    try:

        chunks_collection.delete(where={"document_id": document_id})

    except Exception:

        pass



    if neo4j_driver is not None:

        try:

            with neo4j_driver.session(database=NEO4J_DATABASE) as session:

                session.run(

                    "MATCH (n:Entity {document_id: $document_id}) DETACH DELETE n",

                    document_id=document_id,

                )

        except Exception as exc:

            print(f"Could not remove graph data for {document_id}: {exc}")



    document_pdf_path(document_id).unlink(missing_ok=True)

    document_text_path(document_id).unlink(missing_ok=True)

    return {"success": True, "message": "Document deleted successfully", "document_id": document_id}





@app.get("/api/documents/{document_id}/text")

def get_document_text(document_id: str):

    document_id = validate_document_id(document_id)

    text = load_document_text(document_id)

    return {"success": True, "document_id": document_id, "text": text}





@app.post("/api/documents/{document_id}/process")

def process_document(document_id: str):

    document_id = validate_document_id(document_id)

    text = load_document_text(document_id)

    try:

        count = process_document_chunks(document_id, text)

    except Exception as exc:

        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}")

    return {

        "success": True,

        "message": "Document processed successfully",

        "document_id": document_id,

        "total_chunks": count,

        "embedding_model": EMBEDDING_MODEL,

    }





@app.get("/api/documents/{document_id}/chunks")

def get_document_chunks(document_id: str):

    document_id = validate_document_id(document_id)

    result = chunks_collection.get(

        where={"document_id": document_id},

        include=["documents", "metadatas"],

    )

    items = []

    for index, chunk_id in enumerate(result.get("ids", [])):

        items.append({

            "id": chunk_id,

            "text": result["documents"][index],

            "metadata": result["metadatas"][index],

        })

    items.sort(key=lambda item: item["metadata"].get("chunk_index", 0))

    return {

        "success": True,

        "document_id": document_id,

        "total_chunks": len(items),

        "chunks": items,

    }





# ------------------------------------------------------------

# Semantic search and AI question answering

# ------------------------------------------------------------



@app.post("/api/knowledge/search")

def semantic_search(request: SearchRequest):

    query_vector = embed_texts([request.query], "RETRIEVAL_QUERY")[0]

    where = {"document_id": request.document_id} if request.document_id else None



    results = chunks_collection.query(

        query_embeddings=[query_vector],

        n_results=request.top_k,

        where=where,

        include=["documents", "metadatas", "distances"],

    )



    matches = []

    ids = results.get("ids", [[]])[0]

    docs = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    distances = results.get("distances", [[]])[0]



    for index, chunk_id in enumerate(ids):

        matches.append({

            "id": chunk_id,

            "text": docs[index],

            "document_id": metadatas[index].get("document_id"),

            "chunk_index": metadatas[index].get("chunk_index"),

            "distance": distances[index] if index < len(distances) else None,

        })



    return {

        "success": True,

        "query": request.query,

        "total_results": len(matches),

        "results": matches,

    }





@app.post("/api/knowledge/ask")

def ask_question(request: AskRequest):

    require_gemini()



    if request.document_id:

        document_id = validate_document_id(request.document_id)

        if get_document_chunk_count(document_id) == 0:

            text = load_document_text(document_id)

            try:

                process_document_chunks(document_id, text)

            except Exception as exc:

                raise HTTPException(status_code=500, detail=f"Could not process document: {exc}")



    query_vector = embed_texts([request.question], "RETRIEVAL_QUERY")[0]

    where = {"document_id": request.document_id} if request.document_id else None



    results = chunks_collection.query(

        query_embeddings=[query_vector],

        n_results=request.top_k,

        where=where,

        include=["documents", "metadatas", "distances"],

    )



    ids = results.get("ids", [[]])[0]

    docs = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    distances = results.get("distances", [[]])[0]



    if not docs:

        raise HTTPException(

            status_code=404,

            detail="No processed document chunks were found. Upload and process a PDF first.",

        )



    context_parts = []

    sources = []

    for index, chunk_text in enumerate(docs):

        metadata = metadatas[index]

        context_parts.append(f"[Source {index + 1}]\n{chunk_text}")

        sources.append({

            "id": ids[index],

            "document_id": metadata.get("document_id"),

            "chunk_index": metadata.get("chunk_index"),

            "text": chunk_text,

            "distance": distances[index] if index < len(distances) else None,

        })



    prompt = f"""You are the AI assistant in a document knowledge explorer.

Answer the user's question using only the supplied document context.

If the context does not contain the answer, clearly say that the documents do not provide enough information.

Do not invent facts. Keep the answer clear and concise.



DOCUMENT CONTEXT:

{chr(10).join(context_parts)}



USER QUESTION:

{request.question}

"""



    try:

        response = gemini_client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

        )

        answer = response.text or "I could not generate an answer from the available context."

    except Exception as exc:

        raise HTTPException(status_code=502, detail=f"Gemini answer generation failed: {exc}")



    return {

        "success": True,

        "question": request.question,

        "answer": answer,

        "sources": sources,

        "model": GEMINI_MODEL,

    }





@app.get("/api/knowledge/status")

def knowledge_status():

    return {

        "success": True,

        "total_stored_chunks": chunks_collection.count(),

        "storage": "ChromaDB",

        "embedding_model": EMBEDDING_MODEL,

        "embedding_dimensions": EMBEDDING_DIMENSIONS,

        "collection": CHROMA_COLLECTION_NAME,

    }





# ------------------------------------------------------------

# Knowledge graph endpoints

# ------------------------------------------------------------



@app.post("/api/documents/{document_id}/build-graph")

def build_document_graph(document_id: str):

    document_id = validate_document_id(document_id)

    if neo4j_driver is None:

        raise HTTPException(status_code=503, detail="Neo4j is not configured")



    text = load_document_text(document_id)

    entities = extract_entities(text)

    relationships = extract_relationships(text, entities)



    try:

        store_graph_in_neo4j(document_id, entities, relationships)

    except Exception as exc:

        raise HTTPException(status_code=500, detail=f"Could not save graph to Neo4j: {exc}")



    return {

        "success": True,

        "message": "Knowledge graph built successfully",

        "document_id": document_id,

        "total_entities": len(entities),

        "total_relationships": len(relationships),

        "entities": entities,

        "relationships": relationships,

    }





@app.get("/api/knowledge/graph")

def get_knowledge_graph(document_id: Optional[str] = None):

    if neo4j_driver is None:

        raise HTTPException(status_code=503, detail="Neo4j is not configured")



    query = """

    MATCH (a:Entity)

    WHERE $document_id IS NULL OR a.document_id = $document_id

    OPTIONAL MATCH (a)-[r:RELATED_TO]->(b:Entity)

    WHERE b IS NULL OR ($document_id IS NULL OR b.document_id = $document_id)

    RETURN a.name AS source, a.type AS source_type,

           a.document_id AS document_id,

           type(r) AS relationship,

           r.type AS relationship_type,

           r.evidence AS evidence,

           b.name AS target, b.type AS target_type

    """

    try:

        with neo4j_driver.session(database=NEO4J_DATABASE) as session:

            records = list(session.run(query, document_id=document_id))

    except Exception as exc:

        raise HTTPException(status_code=500, detail=f"Could not retrieve graph: {exc}")



    nodes_by_name = {}

    edges = []

    for record in records:

        source = record["source"]

        if source:

            nodes_by_name[source] = {

                "id": source,

                "label": source,

                "name": source,

                "type": record["source_type"] or "Entity",

                "document_id": record["document_id"],

            }

        target = record["target"]

        if target:

            nodes_by_name[target] = {

                "id": target,

                "label": target,

                "name": target,

                "type": record["target_type"] or "Entity",

                "document_id": record["document_id"],

            }

            edges.append({

                "id": f"{source}-{target}-{len(edges)}",

                "source": source,

                "target": target,

                "label": record["relationship_type"] or "RELATED_TO",

                "type": record["relationship_type"] or "RELATED_TO",

                "evidence": record["evidence"] or "",

            })



    nodes = list(nodes_by_name.values())

    return {

        "success": True,

        "document_id": document_id,

        "total_nodes": len(nodes),

        "total_edges": len(edges),

        "nodes": nodes,

        "edges": edges,

    }





# ------------------------------------------------------------

# Shutdown

# ------------------------------------------------------------



@app.on_event("shutdown")

def shutdown_event():

    if neo4j_driver is not None:

        neo4j_driver.close()
