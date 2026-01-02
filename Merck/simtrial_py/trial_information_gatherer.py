"""
Extract recruitment duration and trial information.

This module helps gather data for clinical trial simulation:
1. Recruitment duration (via Selenium scraping of ClinicalTrials.gov)
2. Design and Scenario Hazard Ratios (via Gemini AI)

Usage:
    from trial_information_gatherer import gather_trial_info
"""

from datetime import datetime
from typing import Optional
import re
import argparse
import sys
import time
import json
import requests

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def get_ctgov_data(nct_id: str) -> dict:
    """
    Fetch trial data from ClinicalTrials.gov API v2.
    """
    nct_id = nct_id.upper().strip()
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant fields for AI context
        protocol = data.get('protocolSection', {})
        id_module = protocol.get('identificationModule', {})
        status_module = protocol.get('statusModule', {})
        design_module = protocol.get('designModule', {})
        conditions_module = protocol.get('conditionsModule', {})
        eligibility_module = protocol.get('eligibilityModule', {})
        arms_interventions = protocol.get('armsInterventionsModule', {})
        
        extracted = {
            'nct_id': nct_id,
            'brief_title': id_module.get('briefTitle', ''),
            'official_title': id_module.get('officialTitle', ''),
            'overall_status': status_module.get('overallStatus', ''),
            'start_date': status_module.get('startDateStruct', {}).get('date', ''),
            'primary_completion_date': status_module.get('primaryCompletionDateStruct', {}).get('date', ''),
            'conditions': conditions_module.get('conditions', []),
            'study_type': design_module.get('studyType', ''),
            'phases': design_module.get('phases', []),
            'enrollment': design_module.get('enrollmentInfo', {}).get('count', 0),
            'intervention_model': design_module.get('designInfo', {}).get('interventionModel', ''),
            'primary_purpose': design_module.get('designInfo', {}).get('primaryPurpose', ''),
            'interventions': [i.get('name') for i in arms_interventions.get('interventions', [])],
            'inclusion_criteria': eligibility_module.get('eligibilityCriteria', '')[:1000] + "..." # Truncate for token limit
        }
        return extracted
        
    except Exception as e:
        print(f"Error fetching CT.gov API data: {e}")
        return {}


def get_recruitment_changes_from_url(nct_id: str, headless: bool = True) -> list[dict]:
    """
    Navigate to ClinicalTrials.gov history page and extract recruitment status changes.
    """
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium not installed. Run: pip install selenium")
    
    # Normalize NCT ID
    nct_id = nct_id.upper().strip()
    if not nct_id.startswith('NCT'):
        nct_id = f'NCT{nct_id}'
    
    url = f"https://clinicaltrials.gov/study/{nct_id}?tab=history"
    
    # Setup Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=chrome_options)
    changes = []
    
    try:
        print(f"Navigating to {url}...")
        driver.get(url)
        
        # Wait for the history table to load
        wait = WebDriverWait(driver, 15)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.usa-table")))
        except:
            print("History table not found or timed out.")
            return []

        time.sleep(3)
        
        # Find all rows and identify those with "Recruitment Status" changes
        rows = driver.find_elements(By.CSS_SELECTOR, "table.usa-table tbody tr")
        print(f"Found {len(rows)} version rows in history table")
        
        # Collect version numbers that have Recruitment Status changes
        versions_with_recruitment = []
        for i, row in enumerate(rows):
            row_text = row.text
            # Check for "Recruitment Status" (case insensitive)
            if 'Recruitment Status' in row_text:
                # Extract date from the row text
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', row_text)
                if date_match:
                    date_str = date_match.group(1)
                    version_num = i + 1  # 1-based version number
                    versions_with_recruitment.append({
                        'version': version_num,
                        'date_str': date_str,
                        'row_text': row_text[:200]
                    })
                    print(f"  Row {version_num} ({date_str}): Has Recruitment Status change")
        
        if not versions_with_recruitment:
            print("No rows with 'Recruitment Status' changes found.")
            return []
        
        # Now visit each version page to get the actual status value
        print(f"\nFetching status from {len(versions_with_recruitment)} version pages...")
        
        for v in versions_with_recruitment:
            version_url = f"https://clinicaltrials.gov/study/{nct_id}?tab=history&a={v['version']}#version-content-panel"
            print(f"  Checking version {v['version']}...")
            
            try:
                driver.get(version_url)
                time.sleep(3)
                
                # Find the version-content-panel element
                try:
                    panel = driver.find_element(By.ID, 'version-content-panel')
                    panel_text = panel.text
                except:
                    panel_text = driver.find_element(By.TAG_NAME, 'body').text
                
                # Parse lines to find "Overall Status" followed by actual status
                lines = panel_text.split('\n')
                status = None
                
                for i, line in enumerate(lines):
                    if 'Overall Status' in line and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Check if next line is a status value
                        if next_line in ['Recruiting', 'Not yet recruiting', 'Completed', 'Terminated', 'Suspended', 'Withdrawn']:
                            status = next_line
                            break
                        elif 'Active, not recruiting' in next_line or 'Active not recruiting' in next_line:
                            status = 'Active, not recruiting'
                            break
                
                if status:
                    parsed_date = datetime.strptime(v['date_str'], '%Y-%m-%d')
                    changes.append({
                        'date': parsed_date,
                        'status': status,
                        'version': v['version']
                    })
                    print(f"    → {status}")
                else:
                    print(f"    → Could not determine status")
                    
            except Exception as e:
                print(f"    → Error: {e}")
                continue
        
    finally:
        driver.quit()
    
    # Remove duplicates and sort by date
    seen = set()
    unique_changes = []
    for change in changes:
        key = (change['date'], change['status'])
        if key not in seen:
            seen.add(key)
            unique_changes.append(change)
    
    unique_changes.sort(key=lambda x: x['date'])
    return unique_changes


def calculate_enrollment_duration(changes: list[dict]) -> dict:
    """
    Calculate enrollment duration from recruitment status changes.
    """
    if not changes:
        return None
    
    # Sort by date
    changes = sorted(changes, key=lambda x: x['date'])
    
    # Track recruiting periods
    recruiting_periods = []
    current_recruiting_start = None
    first_recruiting_date = None
    last_end_date = None
    
    for change in changes:
        status = change['status']
        change_date = change['date']
        
        if status == 'Recruiting':
            if current_recruiting_start is None:
                current_recruiting_start = change_date
                if first_recruiting_date is None:
                    first_recruiting_date = change_date
        elif status in ('Suspended', 'Active, not recruiting', 'Completed', 'Terminated', 'Withdrawn'):
            if current_recruiting_start is not None:
                # End of a recruiting period
                recruiting_periods.append({
                    'start': current_recruiting_start,
                    'end': change_date,
                    'days': (change_date - current_recruiting_start).days
                })
                current_recruiting_start = None
            if status in ('Active, not recruiting', 'Completed'):
                last_end_date = change_date
    
    if not recruiting_periods:
        return None
    
    # Calculate totals
    total_recruiting_days = sum(p['days'] for p in recruiting_periods)
    
    # Overall span (first recruiting to last end)
    if last_end_date is None:
        last_end_date = recruiting_periods[-1]['end']
    overall_span_days = (last_end_date - first_recruiting_date).days
    
    # Non-recruiting time
    non_recruiting_days = overall_span_days - total_recruiting_days
    
    return {
        'start_date': first_recruiting_date,
        'end_date': last_end_date,
        'recruiting_periods': recruiting_periods,
        'total_recruiting_days': total_recruiting_days,
        'total_recruiting_months': round(total_recruiting_days / 30.44, 1),
        'overall_span_days': overall_span_days,
        'overall_span_months': round(overall_span_days / 30.44, 1),
        'non_recruiting_days': non_recruiting_days,
        'non_recruiting_months': round(non_recruiting_days / 30.44, 1)
    }


def get_enrollment_duration(nct_id: str, headless: bool = True) -> Optional[dict]:
    """
    Get enrollment duration for a trial via Selenium.
    """
    try:
        changes = get_recruitment_changes_from_url(nct_id, headless=headless)
        if not changes:
            return None
        
        duration = calculate_enrollment_duration(changes)
        if not duration:
            return None
        
        return {
            'enrollment_months': duration['overall_span_months'],
            'start_date': duration['start_date'],
            'end_date': duration['end_date'],
            'active_recruiting_months': duration['total_recruiting_months'],
            'interruption_months': duration['non_recruiting_months'],
            'details': duration
        }
    except Exception as e:
        print(f"Error getting enrollment duration for {nct_id}: {e}")
        return None


def get_ai_design_hr(api_key, nct_id, ct_data=None):
    """Call Gemini to get Design HR."""
    if not HAS_GENAI:
        return None, "Google GenAI library not installed."
    
    try:
        client = genai.Client(api_key=api_key)
        
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        context = ""
        if ct_data:
            context = f"""
            TRIAL CONTEXT:
            Title: {ct_data.get('brief_title')}
            Conditions: {ct_data.get('conditions')}
            Interventions: {ct_data.get('interventions')}
            Phases: {ct_data.get('phases')}
            Enrollment: {ct_data.get('enrollment')}
            """
        
        prompt = f"""
        Research the clinical trial {nct_id} extensively.
        {context}
        
        Your goal is to estimate the 'Design Hazard Ratio' used for the Power/Sample Size calculation in the protocol.
        
        1. **Analyze the Protocol**: Look for "assumed hazard ratio", "detect a hazard ratio of", or "reduction in risk of X%".
        2. **Search for Recent Benchmarks**: If not explicitly stated, find the MOST RELEVANT and RECENT Phase 3 trials in this exact indication ({ct_data.get('conditions', '') if ct_data else ''} - {ct_data.get('interventions', '') if ct_data else ''}).
           - Example: "Keynote-ABC design HR", "CheckMate-XYZ power calculation".
        3. **Infer from Sample Size**: A sample size of {ct_data.get('enrollment', 'Unknown') if ct_data else 'Unknown'} suggests a specific effect size.
        
        CRITICAL: 
        - Do NOT generic ranges (e.g. "0.7-0.8"). Give a specific POINT ESTIMATE (e.g. 0.72).
        - Cite the SPECIFIC benchmark trial or source you used (e.g. "Based on the similar design of trial NCT12345...").
        
        Return a JSON object with two keys:
        - "design_hr": float (e.g. 0.73)
        - "rationale": string (Concise explanation citing the specific benchmark/source. Max 2 sentences.)
        """
        
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                response_mime_type="application/json"
            )
        )
        
        print(f"\n[DEBUG] Gemini Design HR Response:\n{response.text}\n")
        data = json.loads(response.text)
        return data.get("design_hr"), data.get("rationale")
        
    except Exception as e:
        return None, str(e)


def get_ai_scenario_hr(api_key, nct_id, ct_data=None):
    """Call Gemini to get Scenario HR (True HR)."""
    if not HAS_GENAI:
        return None, "Google GenAI library not installed."
    
    try:
        client = genai.Client(api_key=api_key)
        
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        context = ""
        if ct_data:
            context = f"""
            TRIAL CONTEXT:
            Title: {ct_data.get('brief_title')}
            Conditions: {ct_data.get('conditions')}
            Interventions: {ct_data.get('interventions')}
            """
        
        prompt = f"""
        Research the clinical trial {nct_id} for the LATEST RESULTS.
        {context}
        
        1. **Find Actual Results**: Search for "{nct_id} results", "{nct_id} hazard ratio", or conference presentations (ASCO, ESMO) from 2023-2025.
        2. **Find Competitor Benchmarks**: If no results, find the most recent Phase 3 results for this specific drug class/indication.
           - Example: "Latest Phase 3 results for [Drug Name] in [Condition]".
        3. **Estimate Scenario HR**: 
           - If results exist: Use the reported HR for the primary endpoint.
           - If no results: Estimate based on Phase 2 data or competitor performance.
        
        CRITICAL:
        - Prioritize ACTUAL reported results over assumptions.
        - If estimating, cite the SPECIFIC trial (e.g. "Phase 2 EV-103 Cohort K showed HR 0.XX").
        - Be precise.
        
        Return a JSON object with two keys:
        - "scenario_hr": float (e.g. 0.47)
        - "rationale": string (Concise explanation citing the specific result source or benchmark. Max 2 sentences.)
        """
        
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                response_mime_type="application/json"
            )
        )
        
        print(f"\n[DEBUG] Gemini Scenario HR Response:\n{response.text}\n")
        data = json.loads(response.text)
        return data.get("scenario_hr"), data.get("rationale")
        
    except Exception as e:
        return None, str(e)


def get_ai_enrollment_estimate(api_key, nct_id, ct_data):
    """
    If recruitment is ongoing, ask AI to estimate duration based on Start Date and PCD.
    """
    if not HAS_GENAI:
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        
        start_date = ct_data.get('start_date', 'Unknown')
        pcd_date = ct_data.get('primary_completion_date', 'Unknown')
        
        prompt = f"""
        The clinical trial {nct_id} is currently recruiting (or not completed).
        Start Date: {start_date}
        Primary Completion Date (PCD): {pcd_date}
        
        Estimate the likely 'Enrollment Duration' (in months).
        
        Context:
        - Enrollment usually ends BEFORE the Primary Completion Date (PCD), as PCD includes follow-up time for the primary endpoint.
        - Typically, Enrollment Duration = (PCD - Start Date) - Follow-up Time.
        - Look for "follow-up" duration in the protocol or similar trials (e.g., 6-12 months for PFS/OS).
        
        If dates are missing, search for "Start date" and "PCD" for {nct_id}.
        
        Return a JSON object:
        - "estimated_duration_months": float (e.g. 24.5)
        - "rationale": string (Explain why, e.g. "PCD is Jan 2026, assuming 12m follow-up...")
        """
        
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        print(f"\n[DEBUG] Gemini Enrollment Estimate:\n{response.text}\n")
        data = json.loads(response.text)
        return data
        
    except Exception as e:
        print(f"Enrollment AI error: {e}")
        return None


def get_ai_trial_info(api_key, nct_id, ct_data=None):
    """
    Get general trial info (Sample Size, Control Median, Events) via AI.
    Used to fill gaps if scraping fails or for deeper insights.
    """
    if not HAS_GENAI:
        return None, "Google GenAI library not installed."
    
    try:
        client = genai.Client(api_key=api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        context = ""
        if ct_data:
            context = f"""
            TRIAL CONTEXT:
            Title: {ct_data.get('brief_title')}
            Conditions: {ct_data.get('conditions')}
            Interventions: {ct_data.get('interventions')}
            """
        
        prompt = f"""
        Research the clinical trial {nct_id} extensively.
        {context}
        
        Extract the following design parameters for the PRIMARY ENDPOINT:
        1. **Total Enrollment**: Target or Actual sample size.
        2. **Control Arm Median Survival**: 
           - Look for the median PFS or OS (whichever is primary) for the CONTROL arm (Standard of Care).
           - If results are not out, find the historical median survival for this control therapy in this indication (benchmark).
        3. **Target Events**: For Interim and Final Analysis.
        4. **Primary Completion Date**: Current estimate.
        
        Return a JSON object with these keys:
        - "sample_size": int
        - "control_median": float (months) -> CRITICAL: Must be specific to this trial or its control arm benchmark.
        - "interim_events": int (or null)
        - "final_events": int (or null)
        - "pcd_date": string (YYYY-MM-DD)
        """
        
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                response_mime_type="application/json"
            )
        )
        
        print(f"\n[DEBUG] Gemini Trial Info Response:\n{response.text}\n")
        return json.loads(response.text), None
        
    except Exception as e:
        return None, str(e)


def gather_trial_info(nct_id: str, api_key: str = None) -> dict:
    """
    Consolidated function to gather all available trial info.
    
    Steps:
    1. CT.gov API: Get metadata (title, conditions, interventions).
    2. Selenium: Get enrollment duration/dates.
    3. AI (if key): Get Design HR, Scenario HR, Sample Size, Control Median (using CT.gov context).
    
    Returns a dictionary with all found parameters.
    """
    print(f"Gathering info for {nct_id}...")
    result = {
        'nct_id': nct_id,
        'logs': []
    }
    
    # 1. CT.gov API Data
    ct_data = get_ctgov_data(nct_id)
    if ct_data:
        result['ct_data'] = ct_data
        result['logs'].append(f"✅ Found trial: {ct_data.get('brief_title')[:50]}...")
    else:
        result['logs'].append("⚠️ Could not fetch CT.gov API data.")
    
    # 2. Enrollment Duration (Scraping)
    enroll_data = None
    try:
        enroll_data = get_enrollment_duration(nct_id, headless=True)
        if enroll_data:
            result['enrollment'] = enroll_data
            result['logs'].append("✅ Found enrollment duration via history scraping.")
            
            # Check if still recruiting
            is_recruiting = False
            if 'details' in enroll_data and enroll_data['details'].get('end_date') is None:
                is_recruiting = True
            
            # If recruiting, ask AI for estimate
            if is_recruiting and api_key and ct_data:
                result['logs'].append("ℹ️ Trial is still recruiting. Asking AI for duration estimate...")
                ai_enroll = get_ai_enrollment_estimate(api_key, nct_id, ct_data)
                if ai_enroll:
                    # Update enrollment months with estimate
                    est_months = ai_enroll.get('estimated_duration_months')
                    if est_months:
                        result['enrollment']['enrollment_months'] = est_months
                        result['enrollment']['is_estimate'] = True
                        result['enrollment']['estimate_rationale'] = ai_enroll.get('rationale')
                        result['logs'].append(f"✅ AI Estimated Enrollment: {est_months} mo")
        else:
            result['logs'].append("⚠️ Could not scrape enrollment duration.")
            
            # Fallback to AI estimate if scraping failed completely
            if api_key and ct_data:
                ai_enroll = get_ai_enrollment_estimate(api_key, nct_id, ct_data)
                if ai_enroll:
                    est_months = ai_enroll.get('estimated_duration_months')
                    if est_months:
                        result['enrollment'] = {
                            'enrollment_months': est_months,
                            'is_estimate': True,
                            'estimate_rationale': ai_enroll.get('rationale')
                        }
                        result['logs'].append(f"✅ AI Estimated Enrollment (Fallback): {est_months} mo")

    except Exception as e:
        result['logs'].append(f"❌ Scraping error: {e}")
    
    # 3. AI Gathering
    if api_key:
        try:
            # Design HR
            d_hr, d_rationale = get_ai_design_hr(api_key, nct_id, ct_data)
            if d_hr:
                result['design_hr'] = d_hr
                result['design_hr_rationale'] = d_rationale
                result['logs'].append("✅ AI found Design HR.")
            
            # Scenario HR
            s_hr, s_rationale = get_ai_scenario_hr(api_key, nct_id, ct_data)
            if s_hr:
                result['scenario_hr'] = s_hr
                result['scenario_hr_rationale'] = s_rationale
                result['logs'].append("✅ AI found Scenario HR.")
                
            # General Info
            info, error = get_ai_trial_info(api_key, nct_id, ct_data)
            if info:
                result['ai_info'] = info
                result['logs'].append("✅ AI found General Trial Info.")
            
        except Exception as e:
            result['logs'].append(f"❌ AI Error: {e}")
    else:
        result['logs'].append("ℹ️ AI skipped (No API Key).")
        
    return result

if __name__ == '__main__':
    # Simple CLI test
    if len(sys.argv) > 1:
        nct = sys.argv[1]
        key = sys.argv[2] if len(sys.argv) > 2 else None
        res = gather_trial_info(nct, key)
        print(json.dumps(res, indent=2, default=str))
