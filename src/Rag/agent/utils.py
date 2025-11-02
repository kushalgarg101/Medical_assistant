from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import List, Dict, Union
import json
import os

from src.config import settings
from src.logger.logg import logs
from src.Rag.retrieve import make_client, load_embed_model, get_vector_store

logger = logs('utils.log')

###########################      Templates      ###########################

reception_prompt_template = """
                                TASK:
                                Using the inputs, manage the patient interaction to locate records, gather necessary details,
                                surface non-clinical follow-ups, and escalate clinical questions to the clinical team. Act only on information available
                                in the provided discharge_report_content; do not invent clinical facts.
                                
                                TOOLS (available to call):
                                - database_retriever_tool(patient_name: str) -> returns a patient's discharge report text (use when discharge_report_content is empty or a fresh copy is required).

                                RULES (follow exactly):
                                1. If discharge_report_content is empty, ask for the patient’s full name to look up records. Then acknowledge the lookup ("Let me pull your report") before calling database_retriever_tool(patient_name).
                                2. When discharge_report_content is provided, treat it as the single source of truth for any retrieval-based statements. Quote or paraphrase only what appears there.
                                3. Always ask short, focused, open questions about recovery, medication-taking, symptom changes, wound status, appointment needs, or other non-clinical logistics related to the report.
                                4. If the patient’s query clearly requests clinical management (diagnosis, medication dosing or changes, new/worsening symptoms, or treatment decisions), do NOT provide clinical advice. Instead:
                                - Summarize the clinical question in one concise sentence (include key context from discharge_report_content, e.g., medication name, dates, or relevant findings).
                                - Tell the patient you will escalate this to clinical staff and ask if they have other concerns.
                                5. If the patient reports emergency/red-flag symptoms (e.g., severe chest pain, trouble breathing, heavy bleeding, sudden confusion), instruct immediate emergency care AND call route_to_clinical_agent with a clear, urgent summary.
                                6. Never include identifiable patient data (PHI) in free-text logs or outputs; use de-identified IDs when logging or noting records.
                                7. Maintain a polite, empathetic, professional voice and keep replies short and easy to understand.

                                PREFERRED WORKFLOW (stepwise):
                                1. No discharge_report_content:
                                - Ask for the patient's full name.
                                - After they reply, say a brief confirmation that you will retrieve the record, then call database_retriever_tool(name).
                                2. Report present:
                                - Begin with a one-sentence synthesis of the key items you can see (discharge date, reason for admission, major meds or follow-up instructions) — only if those items are in discharge_report_content.
                                - Ask 1–2 targeted, non-clinical follow-up questions (medication adherence, pain level, wound appearance, appointment scheduling).
                                3. Clinical question detected:
                                - Produce a single-sentence clinical summary (include the minimal necessary context from the report).
                                - Inform the patient that the clinical team will review and ask if they have additional concerns.

                                OUTPUT STYLE:
                                - Short, clear sentences (one or two per line).
                                - Patient-centered phrasing: ask about feelings, symptoms, ability to follow instructions.
                                - Before invoking any tool, state a one-line confirmation (e.g., "Let me pull up your report now.").
                                - Do not log PHI in visible text outputs; use redacted or de-identified placeholders if you must reference identity in internal notes.

                                SAFETY & BOUNDARIES:
                                - Do not provide medical recommendations, dosages, or clinical management.
                                - Escalate all clinical management requests to the clinical team via route_to_clinical_agent.
                                - For severe or life-threatening complaints, instruct immediate emergency care and flag clinical staff.

                                IMPLEMENTATION NOTES:
                                - Treat discharge_report_content as authoritative for facts about the patient’s care.
                                - If both query and discharge_report_content are present, weave only relevant details from the report into follow-ups and any handoff summary.
                                - Keep summaries and handoffs as brief as possible while preserving critical context (medication names, dates, symptoms, and any recent changes).

                                INPUT (substituted below):
                                Query:
                                \"\"\"{query}\"\"\"

                                Discharge report (if available):
                                \"\"\"{discharge_report_content}\"\"\"
"""

clinical_llm_template = """
                            You are a clinical AI assistant specialized in nephrology. Use the instructions below to produce a single, evidence-based response to the user's question(s).
                            Throughout the prompt we will refer to: the retrieved RAG data (a local book/source), the web search results, the curated passages chosen for reasoning, and the user's query.
                            These input placeholders are defined once below and should be filled by your system before rendering this template.

                            IMPORTANT NOTE ABOUT THE RAG TOOL
                            - The RAG data tool does NOT contain personal user records or user-identifying information.
                            - Instead, the RAG datasource is a book-like source (textbook/ebook) that contains domain knowledge (chapters, sections, passages, and reference ids).
                            - Before creating the final answer, the system MUST produce a 1–2 line summary (source_summary) that describes the main topics and types of information contained in that source.

                            TOOLS (available)
                            - get_rag_data_tool(query: str) -> returns zero or more passages from the local RAG datasource (each passage may include a reference id, chapter, page, and a short evidence snippet).
                            - web_search_tool(query: str, max_results: int = 5) -> performs an external literature/web search and returns ranked results with short snippets. (Only use when RAG-derived information is missing or insufficient.)

                            INPUT_PLACEHOLDERS (fill these exactly once before rendering the template)
                            - retrieved_rag_data: {retrieved_rag_data}          # source_summary: Comprehensive Clinical Nephrology (7th ed.) is a single-volume clinical nephrology textbook that covers core renal science (anatomy, physiology), diagnostic methods (GFR, urinalysis, imaging, biopsy), and major disease areas including acute kidney injury, glomerular disease, diabetic kidney disease, hypertension, chronic kidney disease, dialysis modalities, and transplantation. It also provides practical, evidence-based guidance on drug dosing, interventional and critical-care nephrology, geriatric/palliative nephrology, and procedural techniques with extensive references and self-assessment material.  # 2-line summary produced from the uploaded RAG source (scanned).
                            - web_search_output: {web_search_output}            # (optional) web search results — present only if web_search_tool was called
                            - queries (the user's original question(s)): {queries}

                            PRINCIPLES (short)
                            1. Treat the RAG source (the textbook-like data) as a primary, authoritative local reference for nephrology content.
                        
                            a) Call the RAG tool first to retrieve passages from the local book/source.
                            b) Curate a small set of high-value passages from those retrieved RAG passages.
                            c) Only if the curated passages are empty or insufficient to answer the user's query, run the web search tool to supplement evidence.
                            
                            2. Do not invent facts — if information is missing or uncertain, say so and ask focused follow-ups.
                            3. Prioritize patient safety: avoid definitive therapeutic actions when critical information is missing; recommend clinician confirmation when appropriate.

                            HOW TO USE THE DATA
                            - "retrieved_rag_data" = raw RAG tool output (passages from the book-like source).
                            - Use web_search_tool only to fill gaps when the curated RAG passages do not answer the query.

                            OUTPUT STRUCTURE (produce these sections in order)
                            1) TL;DR — one-sentence summary (answer snapshot).
                            2) Answer — stepwise, teach-first then apply:
                            - One-line simple explanation of the core nephrology concept needed.
                            - Numbered clinical reasoning steps:
                                1. Assumptions made.
                                2. What was checked in the curated RAG passages (cite RAG reference ids).
                                3. What was checked in the web search results if used (list sources).
                                4. How the recommendation was reached.
                            - Separate evidence-based statements (with citations) and practical next steps.
                            - Quote exact lab values, units, ranges, or dosing ranges verbatim from the source and cite them.
                            4) Uncertainty & Confidence — short statement (High / Moderate / Low) with reason.
                            5) Follow-up Questions — up to 3 concise clinical clarifying questions.
                            6) Red Flags — immediate warning signs requiring urgent evaluation.
                            7) Patient-facing summary — 1–2 safe, simple sentences for the patient.
                            
                            PROCESSING STEPS (implementation checklist)
                            1) Call the RAG tool using the user's query and store its raw output as "retrieved_rag_data".
                            2) Produce a concise 1–2 line `source_summary` describing the book/source (topics, scope, editions/chapters referenced). Insert this into INPUT_PLACEHOLDERS before rendering the template.
                            3) Curate a small "curated passages" set internally from the retrieved_rag_data; this curated set is the primary evidence to consult during reasoning.
                            4) If the curated passages are empty or insufficient, call web_search_tool(query: str) and store results as "web_search_output". Use web sources only to supplement missing facts.
                            5) Produce the structured response (OUTPUT STRUCTURE), explicitly referencing passages from the curated RAG passages (by reference id) and any web sources used.
                            6) Log the raw retrieved_rag_data, source_summary, web_search_output (if any), curated passage ids, and the final LOG JSON object.
"""

###########################      Tools      ###########################
@tool
def database_retriever_tool(patient_name: str, file_path: str = r"D:\medicare\Data\reports.json") -> dict:
    """
    Read a JSON file of patient discharge reports and return the record(s) that match `patient_name`.

    Behavior:
    - `patient_name` is matched case-insensitively against the "patient_name" field in each record.
    - If an exact (case-insensitive) match is found, that entry is returned.
    - If no exact match, a fallback substring match is attempted (useful for partial names).
    - If multiple records match, returns a dict with key "matches" containing the list of matching entries.
    - On error or no matches, returns a dict with an "error" key describing the problem.
    """

    try:
        # validate input
        if not isinstance(patient_name, str) or not patient_name.strip():
            return {"error": "patient_name must be a non-empty string."}

        # check file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        # load json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # normalize to a list of records
        if isinstance(data, dict):
            # if top-level is a dict but contains a list under common keys, try to use it
            for candidate_key in ("reports", "patients", "records"):
                if candidate_key in data and isinstance(data[candidate_key], list):
                    data_list = data[candidate_key]
                    break
            else:
                # otherwise wrap single dict into list
                data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            return {"error": "Unexpected JSON structure: expected list or dict."}

        # normalize search name
        name_norm = patient_name.strip().lower()

        # exact (case-insensitive) matches
        exact_matches = [
            rec for rec in data_list
            if isinstance(rec, dict) and rec.get("patient_name", "").strip().lower() == name_norm
        ]

        if exact_matches:
            # return single entry if one, else return list under "matches"
            return {"match" : exact_matches[0]} if len(exact_matches) == 1 else {"matches": exact_matches}
        else:
            return {"error": f"No records found for patient name '{patient_name}'."}

    except Exception as e:
        return {"error": str(e)}

@tool
def vector_retriever_tool(
    query: str,
    collection_name: str = "Medicare",
    top_k: int = 5,
    qdrant_path: str = r"D:\medicare\src\infrastructure",
) -> Union[Dict[str, List[Dict]], Dict[str, str]]:
    """
    Tool which can retrive data stored in a database 
    takes arguments as query. 
    returns relevant results
    
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "query must be a non-empty string."}

    try:
        client = make_client(path=qdrant_path)
        embeddings = load_embed_model()
        vs = get_vector_store(client=client, collection_name=collection_name, embeddings=embeddings)

        # single clear search call (keeps it simple)
        docs = vs.search(query, search_type="similarity", limit=top_k)

        matches = []
        for doc in docs:
            text = getattr(doc, "page_content", "") or ""
            meta = getattr(doc, "metadata", {}) or {}
            score = getattr(doc, "score", None)
            citation = meta.get("source") or meta.get("filename") or meta.get("doc_id") or "unknown"
            matches.append({"text": text, "score": score, "citation": citation, "metadata": meta})

        logger.info("Query=%r collection=%r returned %d matches", query, collection_name, len(matches))
        return {"matches": matches}

    except Exception as exc:
        logger.exception("Error in vector_retriever_tool: %s", exc)
        return {"error": str(exc)}

web_search_tool = TavilySearch(tavily_api_key=settings.TAVILY_API_KEY)

tools_reception = [database_retriever_tool]
clinical_node_tools = [web_search_tool, vector_retriever_tool]

###########################      Models      ###########################

decision_node_llm = ChatGroq(
    model=settings.GROQ_LLM_MODEL,
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=settings.GROQ_API_KEY,  # type: ignore
)

instance_decision_llm = decision_node_llm.bind_tools(tools=tools_reception)

clinical_llm = ChatGroq(
    model=settings.GROQ_LLM_MODEL,
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=settings.GROQ_API_KEY,  # type: ignore
).bind_tools(tools = clinical_node_tools)