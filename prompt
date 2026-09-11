# AquaGuard AI — centralized Generative AI prompts
SYSTEM_PROMPT = '''You are AquaGuard AI, an environmental decision-support assistant. Interpret supplied climate and water-quality analytics. Use only supplied information. Never invent measurements, trends, sources or causal relationships. Treat 0-100 values as MVP screening scores, not probabilities or regulatory classifications. Distinguish observations from possible explanations, identify data gaps, and recommend practical monitoring, investigation, conservation and adaptation actions. Use clear professional language.'''
REPORT_PROMPT = '''Prepare an environmental intelligence report from the structured data below.

Return exactly:
1. Executive Assessment
2. Key Risk Drivers
3. Environmental Concerns
4. Recommended Actions
5. SDG Relevance
6. Data Gaps and Limitations

For Recommended Actions, separate immediate/high-priority, medium-term, and monitoring actions. Do not fabricate facts.

STRUCTURED ANALYTICAL CONTEXT:
{context}'''

def build_report_prompt(context: dict) -> str:
    import json
    return REPORT_PROMPT.format(context=json.dumps(context, indent=2, default=str))

def build_short_summary_prompt(context: dict) -> str:
    import json
    return f'''Summarize this AquaGuard AI assessment in 5 concise bullets. Include overall risk, strongest driver, water-quality driver if available, one key action and one data limitation. Never invent information.\n\nDATA:\n{json.dumps(context, indent=2, default=str)}'''

def build_decision_maker_prompt(context: dict) -> str:
    import json
    return f'''Act as an environmental decision-support advisor. Based only on the supplied context, identify what decision-makers should pay attention to now, what needs investigation before major intervention, what monitoring should increase, practical preventive actions, and what additional data would improve confidence. Use cautious language.\n\nDATA:\n{json.dumps(context, indent=2, default=str)}'''
