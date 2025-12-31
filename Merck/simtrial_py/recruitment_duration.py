"""
Extract recruitment duration from ClinicalTrials.gov history page.

This module can be used as:
1. A standalone CLI tool: python recruitment_duration.py NCT04223856
2. An importable module: from recruitment_duration import get_enrollment_duration

Usage:
    python recruitment_duration.py NCT04223856
    python recruitment_duration.py --nct NCT04223856
"""

from datetime import datetime
from typing import Optional
import re
import argparse
import sys
import time

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


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
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.usa-table")))
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
    
    Handles complex patterns like:
    - Recruiting → Suspended → Recruiting → Active, not recruiting
    - Multiple recruitment phases
    
    Returns total time actually spent in 'Recruiting' status.
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
    High-level API to get enrollment duration for a trial.
    
    Args:
        nct_id: NCT number (e.g., 'NCT04223856')
        headless: Run browser in headless mode (default True)
    
    Returns:
        dict with keys:
            - enrollment_months: float (overall span, use this for modeling)
            - start_date: datetime
            - end_date: datetime
            - active_recruiting_months: float
            - interruption_months: float
            - details: full duration dict
        None if duration could not be determined
    
    Example:
        >>> result = get_enrollment_duration('NCT04223856')
        >>> if result:
        ...     print(f"Enrollment: {result['enrollment_months']} months")
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract recruitment duration from ClinicalTrials.gov'
    )
    parser.add_argument('nct', nargs='?', type=str, help='NCT number (e.g., NCT04223856)')
    parser.add_argument('--nct', dest='nct_flag', type=str, help='NCT number (alternative)')
    parser.add_argument('--no-headless', action='store_true', help='Show browser window')
    
    args = parser.parse_args()
    
    nct_id = args.nct or args.nct_flag
    
    if not nct_id:
        print("Error: NCT number required")
        print("Usage: python recruitment_duration.py NCT04223856")
        sys.exit(1)
    
    if not SELENIUM_AVAILABLE:
        print("Error: Selenium not installed")
        print("Install with: pip install selenium")
        sys.exit(1)
    
    try:
        changes = get_recruitment_changes_from_url(nct_id, headless=not args.no_headless)
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print(f"RECRUITMENT STATUS CHANGES FOR {nct_id.upper()}")
    print("=" * 60)
    
    if not changes:
        print("No recruitment status changes found.")
        sys.exit(0)
    
    for change in changes:
        print(f"  {change['date'].strftime('%Y-%m-%d')} : {change['status']}")
    
    duration = calculate_enrollment_duration(changes)
    
    if duration:
        print("\n" + "-" * 60)
        print("ENROLLMENT DURATION ANALYSIS")
        print("-" * 60)
        print(f"  First Recruiting:  {duration['start_date'].strftime('%B %d, %Y')}")
        print(f"  Last End Date:     {duration['end_date'].strftime('%B %d, %Y')}")
        print()
        print(f"  *** ENROLLMENT DURATION: {duration['overall_span_days']} days ({duration['overall_span_months']} months) ***")
        print(f"      (Use this for event prediction modeling)")
        print()
        print(f"  Status Changes Timeline:")
        for i, period in enumerate(duration['recruiting_periods'], 1):
            print(f"    {i}. RECRUITING: {period['start'].strftime('%Y-%m-%d')} → {period['end'].strftime('%Y-%m-%d')} ({period['days']} days)")
        print()
        print(f"  Breakdown:")
        print(f"    Active recruiting time:  {duration['total_recruiting_days']} days ({duration['total_recruiting_months']} months)")
        print(f"    Interruptions/pauses:    {duration['non_recruiting_days']} days ({duration['non_recruiting_months']} months)")
    else:
        print("\nCould not calculate enrollment duration.")
        print("Need both 'Recruiting' and 'Active, not recruiting' status changes.")


if __name__ == '__main__':
    main()
