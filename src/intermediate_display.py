"""
intermediate_display.py - Display Intermediate Results for Each Pipeline Stage

This module provides pretty-print functions to visualize what's happening
at each stage of the Glaucoma RAG Pipeline.

Usage:
    from intermediate_display import InterMediateDisplay
    display = InterMediateDisplay()
    display.show_pdf_chunks(chunks)
    display.show_vector_store_stats(vector_store, query)
    ...
"""

from typing import Dict, List
from datetime import datetime


# ─────────────────────────────────────────────
# ANSI Color Codes for Terminal Output
# ─────────────────────────────────────────────
class Colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'


def _header(title: str, width: int = 70):
    """Print a bold section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.END}")


def _subheader(title: str, width: int = 60):
    """Print a sub-section header"""
    print(f"\n{Colors.CYAN}{'─' * width}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * width}{Colors.END}")


def _ok(msg: str):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")


def _warn(msg: str):
    print(f"  {Colors.YELLOW}[!!]{Colors.END} {msg}")


def _info(msg: str):
    print(f"  {Colors.BLUE}[>>]{Colors.END} {msg}")


def _bullet(label: str, value, unit: str = ""):
    print(f"      {Colors.BOLD}•{Colors.END} {label}: {Colors.YELLOW}{value}{Colors.END} {unit}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — PDF Processing
# ══════════════════════════════════════════════════════════════════════

def show_pdf_chunks(chunks: List[Dict]):
    """
    Display intermediate results from PDF processing stage.

    Args:
        chunks: List of chunk dicts from PDFProcessor.process_all_pdfs()
    """
    _header("STAGE 1 │ PDF PROCESSING & CHUNKING")

    if not chunks:
        _warn("No chunks found! Check that PDFs exist in data/guidelines/")
        return

    # Summary
    sources = {}
    total_chars = 0
    for chunk in chunks:
        src = chunk.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1
        total_chars += chunk.get("chunk_size", len(chunk.get("text", "")))

    _ok(f"Total chunks created  : {Colors.YELLOW}{len(chunks)}{Colors.END}")
    _ok(f"Total characters      : {Colors.YELLOW}{total_chars:,}{Colors.END}")
    _ok(f"Unique source PDFs    : {Colors.YELLOW}{len(sources)}{Colors.END}")

    # Per-source breakdown
    _subheader("Chunks Per Source PDF")
    for src, count in sources.items():
        bar = "█" * min(count // 2, 40)
        print(f"    {Colors.CYAN}{src:<35}{Colors.END} {Colors.YELLOW}{count:>4} chunks{Colors.END}  {Colors.GREEN}{bar}{Colors.END}")

    # Sample chunk
    _subheader("Sample Chunk (First Chunk)")
    sample = chunks[0]
    _bullet("Chunk ID",     sample.get("chunk_id", "N/A"))
    _bullet("Source",       sample.get("source", "N/A"))
    _bullet("Chunk Index",  f"{sample.get('chunk_index', 0)} / {sample.get('total_chunks', '?')}")
    _bullet("Chunk Size",   f"{sample.get('chunk_size', '?')} characters")
    print(f"\n    {Colors.BOLD}Text Preview:{Colors.END}")
    preview = sample.get("text", "")[:300]
    print(f"    {Colors.BLUE}{preview}...{Colors.END}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — Vector Store / FAISS Embeddings
# ══════════════════════════════════════════════════════════════════════

def show_vector_store_stats(vector_store, sample_query: str = "glaucoma treatment guidelines"):
    """
    Display FAISS vector store stats and a sample search result.

    Args:
        vector_store : FAISSVectorStore instance
        sample_query : A test query to demonstrate retrieval
    """
    _header("STAGE 2 │ VECTOR STORE & EMBEDDINGS (FAISS)")

    stats = vector_store.get_stats()

    _subheader("Index Statistics")
    _bullet("Total Vectors Indexed",  stats.get("total_vectors", 0))
    _bullet("Total Documents Stored", stats.get("total_documents", 0))
    _bullet("Embedding Dimensions",   stats.get("dimension", "N/A"))
    _bullet("Embedding Model",        stats.get("embedding_model", "N/A"))
    _bullet("Index Type",             stats.get("index_type", "N/A"))
    _bullet("Index Path",             stats.get("index_path", "N/A"))

    # Sample search
    _subheader(f"Sample Semantic Search")
    _info(f"Query: \"{sample_query}\"")

    try:
        results = vector_store.search(sample_query, k=3)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n    {Colors.BOLD}Result {i}:{Colors.END}")
                _bullet("Source",     r["metadata"].get("source", "Unknown"))
                _bullet("Distance",   f"{r.get('distance', 0):.4f}")
                _bullet("Similarity", f"{r.get('similarity', 0):.4f}")
                preview = r["text"][:150].replace("\n", " ")
                print(f"      {Colors.BLUE}Text: {preview}...{Colors.END}")
        else:
            _warn("No results returned. Index may be empty.")
    except Exception as e:
        _warn(f"Search failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — Clinical Metrics
# ══════════════════════════════════════════════════════════════════════

def show_clinical_metrics(patient_data: Dict):
    """
    Display computed clinical metrics for each visit.

    Args:
        patient_data: Patient dict after process_all_visits() has been called
    """
    _header("STAGE 3 │ CLINICAL METRICS CALCULATION")

    visits = patient_data.get("visits", [])
    patient_id = patient_data.get("patient_id", "Unknown")

    _info(f"Patient ID : {patient_id}")
    _info(f"Visits     : {len(visits)}")

    _subheader("Per-Visit Clinical Metrics Table")

    # Table header
    col = Colors
    print(f"\n    {col.BOLD}{'Visit':>6}  {'MD (dB)':>10}  {'VFI (%)':>8}  "
          f"{'PSD (dB)':>9}  {'Mean VF':>8}  {'Severity':<12}  {'Pattern'}{col.END}")
    print(f"    {'─'*6}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*12}  {'─'*20}")

    for v in visits:
        m = v.get("clinical_metrics", {})
        visit_num = v.get("visit", "?")
        md        = m.get("MD",        "N/A")
        vfi       = m.get("VFI",       "N/A")
        psd       = m.get("PSD",       "N/A")
        mean_vf   = m.get("mean_vf",   "N/A")
        severity  = m.get("severity",  "N/A")
        pattern   = m.get("pattern",   "N/A")

        # Color-code severity
        sev_color = (col.GREEN if severity == "Mild" else
                     col.YELLOW if severity == "Moderate" else
                     col.RED)

        print(f"    {str(visit_num):>6}  {str(md):>10}  {str(vfi):>8}  "
              f"{str(psd):>9}  {str(mean_vf):>8}  "
              f"{sev_color}{severity:<12}{col.END}  {pattern}")

    # Quadrant analysis for latest visit
    if visits:
        latest = visits[-1].get("clinical_metrics", {})
        _subheader("Latest Visit — Quadrant Analysis")
        _bullet("Superior Field Mean", f"{latest.get('superior_mean', 'N/A')} dB")
        _bullet("Inferior Field Mean", f"{latest.get('inferior_mean', 'N/A')} dB")
        _bullet("Sup-Inf Difference",  f"{latest.get('superior_inferior_diff', 'N/A')} dB")
        _bullet("Defect Pattern",      latest.get("pattern", "N/A"))


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — Progression Analysis
# ══════════════════════════════════════════════════════════════════════

def show_progression_analysis(progression_analysis: Dict):
    """
    Display multi-visit progression analysis results.

    Args:
        progression_analysis: Output from ProgressionAnalyzer.analyze_progression()
    """
    _header("STAGE 4 │ PROGRESSION ANALYSIS")

    if progression_analysis.get("status") != "analyzed":
        _warn(f"Cannot show progression: {progression_analysis.get('message', 'Unknown error')}")
        return

    _subheader("Overall Summary")
    _bullet("Visits Analyzed",     progression_analysis.get("num_visits"))
    _bullet("Time Span",           f"{progression_analysis.get('time_span_months')} months "
                                   f"({progression_analysis.get('time_span_years')} years)")
    _bullet("MD Change",           f"{progression_analysis.get('delta_MD')} dB")
    _bullet("VFI Change",          f"{progression_analysis.get('delta_VFI')}%")
    _bullet("Progression Rate",    f"{progression_analysis.get('progression_rate_MD')} dB/year")
    _bullet("VFI Rate",            f"{progression_analysis.get('progression_rate_VFI')}% /year")

    _subheader("Classification")
    trend = progression_analysis.get("trend", "N/A")
    risk  = progression_analysis.get("risk_level", "N/A")

    trend_color = (Colors.GREEN if trend == "Improving" else
                   Colors.YELLOW if trend == "Stable" else Colors.RED)
    risk_color  = (Colors.GREEN if risk == "Slow" else
                   Colors.YELLOW if risk == "Moderate" else Colors.RED)

    _bullet("Trend",            f"{trend_color}{trend}{Colors.END}")
    _bullet("Risk Level",       f"{risk_color}{risk}{Colors.END}")
    _bullet("Acceleration",     progression_analysis.get("acceleration", "N/A"))
    _bullet("Initial Severity", progression_analysis.get("initial_severity", "N/A"))
    _bullet("Current Severity", progression_analysis.get("current_severity", "N/A"))
    _bullet("Severity Changed", "Yes" if progression_analysis.get("severity_changed") else "No")

    # Visit-by-visit MD trend
    _subheader("Visit-by-Visit MD Trend")
    visit_history = progression_analysis.get("visit_history", [])
    if visit_history:
        print(f"\n    {Colors.BOLD}{'Visit':>6}  {'MD (dB)':>10}  {'VFI (%)':>8}  {'Severity':<12}  Trend{Colors.END}")
        print(f"    {'─'*6}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*10}")
        prev_md = None
        for v in visit_history:
            md  = v.get("MD", 0)
            arrow = ""
            if prev_md is not None:
                diff = md - prev_md
                arrow = (f"{Colors.GREEN}▲ +{diff:.2f}{Colors.END}" if diff > 0 else
                         f"{Colors.RED}▼ {diff:.2f}{Colors.END}" if diff < 0 else
                         f"{Colors.YELLOW}→ 0.00{Colors.END}")
            prev_md = md
            print(f"    {str(v.get('visit','?')):>6}  {str(md):>10}  "
                  f"{str(v.get('VFI','?')):>8}  {str(v.get('severity','?')):<12}  {arrow}")

    # Warnings
    warnings = progression_analysis.get("warnings", [])
    if warnings:
        _subheader("Clinical Warnings")
        for w in warnings:
            print(f"    {Colors.YELLOW}⚠  {w}{Colors.END}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — RAG Retrieval
# ══════════════════════════════════════════════════════════════════════

def show_rag_retrieval(queries: List[str], retrieved_chunks: List[Dict]):
    """
    Display RAG queries and retrieved guideline chunks.

    Args:
        queries         : List of query strings generated by RAGRetriever
        retrieved_chunks: List of retrieved chunk dicts
    """
    _header("STAGE 5 │ RAG RETRIEVAL")

    # Queries
    _subheader("Generated Queries")
    for i, q in enumerate(queries, 1):
        print(f"    {Colors.CYAN}{i}.{Colors.END} {q}")

    # Retrieval summary
    _subheader("Retrieval Summary")
    sources = {}
    for c in retrieved_chunks:
        src = c.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1

    _bullet("Total Chunks Retrieved", len(retrieved_chunks))
    _bullet("Unique Sources",         len(sources))

    for src, count in sources.items():
        print(f"      {Colors.CYAN}▸ {src}{Colors.END}: {count} chunks")

    # Top retrieved chunks
    _subheader("Top Retrieved Chunks (with Similarity Scores)")
    top_chunks = retrieved_chunks[:5]
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n    {Colors.BOLD}Chunk {i}:{Colors.END}")
        _bullet("Source",     chunk.get("source", "Unknown"))
        _bullet("Similarity", f"{chunk.get('similarity', 0):.4f}")
        _bullet("Distance",   f"{chunk.get('distance', 0):.4f}")
        _bullet("Query",      chunk.get("query", "N/A")[:60] + "...")
        preview = chunk.get("text", "")[:200].replace("\n", " ")
        print(f"      {Colors.BLUE}Text: {preview}...{Colors.END}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — Prompt Builder
# ══════════════════════════════════════════════════════════════════════

def show_prompt_builder(prompt: str, retrieved_chunks: List[Dict], patient_data: Dict):
    """
    Display the assembled prompt structure and key excerpts.

    Args:
        prompt          : Full assembled prompt string from PromptBuilder
        retrieved_chunks: Retrieved guideline chunks included in prompt
        patient_data    : Patient data used to build prompt
    """
    _header("STAGE 6 │ PROMPT BUILDER — ASSEMBLED PROMPT")

    total_chars  = len(prompt)
    est_tokens   = total_chars // 4
    num_sections = prompt.count("=" * 10)

    _subheader("Prompt Statistics")
    _bullet("Total Characters",      f"{total_chars:,}")
    _bullet("Estimated Tokens",      f"~{est_tokens:,}")
    _bullet("Number of Visits",      len(patient_data.get("visits", [])))
    _bullet("Guidelines Injected",   len(set(c.get("source") for c in retrieved_chunks)))
    _bullet("Chunks in Prompt",      len(retrieved_chunks))

    # Show what sections are present
    _subheader("Prompt Sections Detected")
    section_keywords = [
        "CLINICAL GUIDELINES CONTEXT",
        "PATIENT DATA",
        "PROGRESSION ANALYSIS",
        "FORECAST",
        "TASK: GENERATE",
    ]
    for kw in section_keywords:
        found = kw in prompt
        status = f"{Colors.GREEN}✓ Found{Colors.END}" if found else f"{Colors.RED}✗ Missing{Colors.END}"
        print(f"      {Colors.BOLD}•{Colors.END} {kw:<40} {status}")

    # Guidelines context preview
    _subheader("Guidelines Context Preview (first 400 chars)")
    start = prompt.find("CLINICAL GUIDELINES CONTEXT")
    if start != -1:
        excerpt = prompt[start:start + 400].replace("\n", "\n    ")
        print(f"\n    {Colors.BLUE}{excerpt}...{Colors.END}")

    # Patient data preview
    _subheader("Patient Data Section Preview (first 400 chars)")
    start = prompt.find("PATIENT DATA")
    if start != -1:
        excerpt = prompt[start:start + 400].replace("\n", "\n    ")
        print(f"\n    {Colors.BLUE}{excerpt}...{Colors.END}")

    # Task section
    _subheader("Task Instructions Preview (first 300 chars)")
    start = prompt.find("TASK: GENERATE")
    if start != -1:
        excerpt = prompt[start:start + 300].replace("\n", "\n    ")
        print(f"\n    {Colors.BLUE}{excerpt}...{Colors.END}")

    _ok(f"Prompt successfully assembled and ready to send to Gemini ({LLM_MODEL_NAME})")


# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — Forecasting
# ══════════════════════════════════════════════════════════════════════

def show_forecast(forecast: Dict):
    """
    Display progression forecast results.

    Args:
        forecast: Output from ProgressionForecaster.forecast_linear()
    """
    _header("STAGE 7 │ FORECASTING & PROGNOSIS")

    _subheader("Current Baseline")
    _bullet("Current MD",       f"{forecast.get('current_MD')} dB")
    _bullet("Current VFI",      f"{forecast.get('current_VFI')}%")
    _bullet("Severity",         forecast.get("current_severity"))
    _bullet("Progression Rate", f"{forecast.get('progression_rate')} dB/year")
    _bullet("Method",           forecast.get("method"))

    # Forecast table
    _subheader("Predicted Future Values")
    forecasts = forecast.get("forecasts", [])

    print(f"\n    {Colors.BOLD}{'Horizon':>10}  {'Pred MD':>10}  {'Pred VFI':>10}  {'Severity':<12}  {'Confidence'}{Colors.END}")
    print(f"    {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    for f in forecasts:
        months   = f.get("time_horizon_months")
        pred_md  = f.get("predicted_MD")
        pred_vfi = f.get("predicted_VFI")
        sev      = f.get("predicted_severity", "N/A")
        conf     = f.get("confidence", "N/A")

        conf_color = (Colors.GREEN if conf == "High" else
                      Colors.YELLOW if conf == "Moderate" else Colors.RED)

        print(f"    {f'{months} months':>10}  {str(pred_md):>10}  {str(pred_vfi):>10}  "
              f"{sev:<12}  {conf_color}{conf}{Colors.END}")

    # Time to next stage
    t2n = forecast.get("time_to_next_severity_stage", {})
    if t2n.get("years"):
        _subheader("Time to Next Severity Stage")
        _bullet("Next Stage",    t2n.get("next_severity"))
        _bullet("Threshold MD",  f"{t2n.get('next_threshold_MD')} dB")
        _bullet("Estimated Time",f"{t2n.get('years')} years ({t2n.get('months')} months)")
        print(f"\n    {Colors.YELLOW}⚠  {t2n.get('message')}{Colors.END}")

    # Risk assessment
    risk = forecast.get("risk_assessment", {})
    if risk:
        _subheader("Risk Assessment")
        risk_level  = risk.get("risk_level", "N/A")
        risk_color  = (Colors.GREEN if risk_level == "Low" else
                       Colors.YELLOW if risk_level == "Moderate" else Colors.RED)
        _bullet("Overall Risk", f"{risk_color}{risk_level}{Colors.END}")
        for factor in risk.get("risk_factors", []):
            print(f"      {Colors.RED}▸ {factor}{Colors.END}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — Report Generation
# ══════════════════════════════════════════════════════════════════════

def show_report_summary(structured_output: Dict):
    """
    Display the generated report summary and key outputs.

    Args:
        structured_output: Output from ReportGenerator.create_structured_output()
    """
    _header("STAGE 8 │ REPORT GENERATION — OUTPUT SUMMARY")

    meta = structured_output.get("report_metadata", {})
    _subheader("Report Metadata")
    _bullet("Patient ID",       meta.get("patient_id"))
    _bullet("Generated At",     meta.get("report_generated_date", "")[:19])
    _bullet("Report Type",      meta.get("report_type"))
    _bullet("Visits Analyzed",  meta.get("visits_analyzed"))
    _bullet("Analysis Period",  f"{meta.get('analysis_timespan_months')} months")
    _bullet("Guidelines Used",  ", ".join(meta.get("guidelines_used", [])))

    # Model performance
    model_perf = structured_output.get("model_performance", {})
    _subheader("AI Model Performance (MST Predictions)")
    print(f"\n    {Colors.BOLD}{'Visit':>6}  {'MAE (dB)':>10}  {'RMSE (dB)':>10}{Colors.END}")
    print(f"    {'─'*6}  {'─'*10}  {'─'*10}")
    for v in model_perf.get("visits", []):
        mae  = v.get("mae_dB")
        rmse = v.get("rmse_dB")
        mae_str  = f"{mae:.2f}"  if mae  is not None else "N/A"
        rmse_str = f"{rmse:.2f}" if rmse is not None else "N/A"
        print(f"    {str(v.get('visit','?')):>6}  {mae_str:>10}  {rmse_str:>10}")

    avg_mae  = model_perf.get("average_mae",  0)
    avg_rmse = model_perf.get("average_rmse", 0)
    print(f"\n    {Colors.BOLD}Average MAE : {Colors.YELLOW}{avg_mae:.2f} dB{Colors.END}")
    print(f"    {Colors.BOLD}Average RMSE: {Colors.YELLOW}{avg_rmse:.2f} dB{Colors.END}")

    # Narrative report preview (executive summary)
    _subheader("Generated Report — Executive Summary Preview")
    narrative = structured_output.get("narrative_report", "")
    if narrative:
        # Try to find EXECUTIVE SUMMARY section
        upper = narrative.upper()
        start = upper.find("EXECUTIVE SUMMARY")
        if start != -1:
            excerpt = narrative[start:start + 600]
        else:
            excerpt = narrative[:600]
        excerpt = excerpt.replace("\n", "\n    ")
        print(f"\n    {Colors.BLUE}{excerpt}{Colors.END}")
        print(f"\n    {Colors.BOLD}... [Full report saved to file]{Colors.END}")
    else:
        _warn("Narrative report is empty.")

    # Saved files
    _subheader("Report Generation Complete")
    _ok(f"Patient {meta.get('patient_id')} — all stages completed successfully")
    print(f"\n    {Colors.GREEN}{'─'*60}{Colors.END}")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"    {Colors.BOLD}Pipeline finished at: {now}{Colors.END}")
    print(f"    {Colors.GREEN}{'─'*60}{Colors.END}\n")


# ══════════════════════════════════════════════════════════════════════
# Pipeline Start Banner
# ══════════════════════════════════════════════════════════════════════

def show_pipeline_start(patient_id: str, num_visits: int):
    """Show a banner at the start of processing a patient"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'█' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}  GLAUCOMA RAG PIPELINE — INTERMEDIATE RESULTS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Patient: {patient_id}   |   Visits: {num_visits}   |   {datetime.now().strftime('%Y-%m-%d %H:%M')}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'█' * 70}{Colors.END}\n")


# Used in show_prompt_builder to display model name
try:
    from config import LLM_MODEL as LLM_MODEL_NAME
except ImportError:
    LLM_MODEL_NAME = "Gemini"