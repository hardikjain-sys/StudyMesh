import os
import json
import re
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def sanitize_text(text: str) -> str:
    """Removes null bytes and unprintable control characters that crash API serialization."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")

def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    cleaned_fallback = text.replace('\\', '\\\\')
    cleaned_fallback = re.sub(r'\\\\(["\\/bfnrt])', r'\\\1', cleaned_fallback)
    return json.loads(cleaned_fallback, strict=False)

def build_mermaid_from_json(data: dict) -> str:
    """Builds clean Mermaid syntax."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    lines = ["graph TD"]
    valid_ids = set()

    for node in nodes:
        node_id = str(node.get("id", "")).strip().replace("-", "_").replace(" ", "_")
        if not node_id:
            continue
        valid_ids.add(node_id)
        label = str(node.get("label", "")).replace('"', "'").replace("[", "(").replace("]", ")")
        lines.append(f'    {node_id}["{label}"]')

    for edge in edges:
        src = str(edge.get("from", "")).strip().replace("-", "_").replace(" ", "_")
        dst = str(edge.get("to", "")).strip().replace("-", "_").replace(" ", "_")

        if src in valid_ids and dst in valid_ids:
            relation = str(edge.get("relation", "")).replace('"', "'").replace("[", "(").replace("]", ")")
            if relation:
                lines.append(f'    {src} -->|"{relation}"| {dst}')
            else:
                lines.append(f'    {src} --> {dst}')

    return "\n".join(lines)

def clean_arxiv_query(raw_query: str) -> str:
    cleaned = re.sub(r"(?i)\bthier\b", "their", raw_query)
    
    fillers = [
        r"\bi want to research in\b", r"\bi want to research\b", r"\bi want to learn\b",
        r"\bi need papers on\b", r"\bresearch on\b", r"\bpapers about\b", 
        r"\bfind papers for\b", r"\bbased on\b", r"\bcompared to\b", r"\bversus\b", r"\bvs\b"
    ]
    for f in fillers:
        cleaned = re.sub(f, " ", cleaned, flags=re.IGNORECASE)

    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", cleaned)
    stopwords = {"want", "need", "like", "normal", "their", "this", "that", "with", "from", "about"}
    keywords = [w for w in words if w.lower() not in stopwords]

    if not keywords:
        return "all:" + urllib.parse.quote(raw_query)

    return "+AND+".join([f"all:{w}" for w in keywords])

def fetch_arxiv_papers_text(query: str, max_papers: int = 2, max_pages_per_paper: int = 2) -> tuple[str, list[dict]]:
    formatted_query = clean_arxiv_query(query)
    url = f"https://export.arxiv.org/api/query?search_query={formatted_query}&start=0&max_results={max_papers}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    compiled_research_text = ""
    papers_info = []

    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_url = link.attrib['href']
                break
        
        if not pdf_url:
            continue
            
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"

        papers_info.append({
            "title": title,
            "summary": summary[:200] + "...",
            "url": pdf_url
        })

        pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(pdf_req) as pdf_resp:
            pdf_bytes = pdf_resp.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(pdf_bytes)
            temp_pdf_path = temp_pdf_file.name

        try:
            doc = fitz.open(temp_pdf_path)
            paper_content = f"\n\n--- RESEARCH PAPER: {title} ---\nAbstract: {summary}\n\nContent Excerpt:\n"
            num_pages = min(max_pages_per_paper, len(doc))
            for p in range(num_pages):
                paper_content += doc[p].get_text() + "\n"
            doc.close()
            compiled_research_text += sanitize_text(paper_content)
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Cap context length at 15,000 chars to avoid server-side timeouts
    if len(compiled_research_text) > 15000:
        compiled_research_text = compiled_research_text[:15000] + "\n[Excerpt truncated for size]"

    return compiled_research_text, papers_info

def process_study_materials(image_paths: list[str], text_notes: str, research_text: str = "") -> dict:
    contents = []

    for path in image_paths:
        if os.path.exists(path):
            uploaded_file = client.files.upload(file=path)
            contents.append(uploaded_file)

    context_str = ""
    if text_notes:
        context_str += f"User Additional Context / Notes:\n{sanitize_text(text_notes)}\n\n"
    if research_text:
        context_str += f"Fetched Research Papers Content:\n{sanitize_text(research_text)}\n\n"

    if context_str:
        contents.append(context_str)

    prompt = """
    Analyze all provided study materials and research paper content together. 
    Synthesize them into a single, cohesive Knowledge Graph.

    RULES FOR JSON FORMATTING:
    - Return plain text string values only.
    - Do NOT use backslashes or raw LaTeX formulas.
    - Replace all math symbols with plain words (e.g., write 'alpha' instead of '\\alpha', 'implies' instead of '\\rightarrow').

    Return ONLY a valid JSON object matching this schema:
    {
      "topic": "Main overall topic of the materials",
      "summary": "2-3 sentence overview of the combined content",
      "nodes": [
        {
          "id": "node1",
          "label": "Concept Name",
          "description": "Short explanation of this concept"
        }
      ],
      "edges": [
        {
          "from": "node1",
          "to": "node2",
          "relation": "label describing relationship"
        }
      ]
    }
    """
    contents.append(prompt)

    # Retry loop with backoff to handle transient 500 errors gracefully
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high")
                )
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback request without thinking_config if gemma-4 endpoint encounters transient server issues
                response = client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=contents
                )
            else:
                time.sleep(2 * (attempt + 1))

    data = extract_json_from_text(response.text)
    data["mermaid"] = build_mermaid_from_json(data)
    return data

def ask_gemma_about_concept(topic: str, concept_label: str, concept_desc: str, user_question: str) -> str:
    prompt = f"""
    You are an AI study assistant analyzing a specific concept from a knowledge graph.

    Overall Topic: {topic}
    Concept: {concept_label}
    Definition: {concept_desc}

    User Question: {user_question}

    Provide a concise, detailed, and direct answer explaining this question.
    """
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    )
    return response.text
