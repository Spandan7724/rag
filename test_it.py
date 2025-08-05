
#!/usr/bin/env python3

import requests
import time

# Configuration
BASE_URL = "http://localhost:8000/hackrx/run"
BEARER_TOKEN = "8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b"
PDF_PATH = "file:///home/spandan/projects/bajaj/pdfs/Arogya.pdf"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

def run_test(number, description, expected, question):
    """
    Send a single question to the RAG pipeline and print the result.
    """
    print(f"Test {number}: {description}")
    print(f"Expected: {expected}")
    payload = {
        "documents": PDF_PATH,
        "questions": [question]
    }
    try:
        response = requests.post(BASE_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("responses", [{}])[0].get("answer", "No answer returned")
    except Exception as e:
        answer = f"Error during request: {e}"
    print(f"Answer: {answer}\n")
    time.sleep(2)


def main():
    # Intro
    print("Testing Improved RAG Pipeline with Arogya Sanjeevani Policy")
    print("Pipeline Improvements:")
    print("  - Adaptive retrieval: 30-60 chunks (was 20)")
    print("  - Synchronized chunking: 100 token overlap (was 75)")
    print("  - Expanded token budget: 768 tokens (was 200)")
    print("  - Relaxed early stopping: 0.6 threshold (was 0.85)\n")

    # Define all tests
    tests = [
        (1, "Grace Period Query", "15 days grace period details", 
         "What is the grace period for premium payment in Arogya Sanjeevani policy?"),
        (2, "Sum Insured Limits", "Rs 1 lakh to Rs 5 lakhs range", 
         "What are the minimum and maximum sum insured limits available under this policy?"),
        (3, "Room Rent Sub-limits", "2% of Sum Insured per day", 
         "What is the room rent sub-limit in this policy?"),
        (4, "Pre-existing Disease Waiting Period", "2-4 year waiting period", 
         "What is the waiting period for pre-existing diseases?"),
        (5, "Age Limits for Entry", "18-65 years entry, renewal till 75", 
         "What are the age limits for entry and renewal in this policy?"),
        (6, "AYUSH Treatment Coverage", "Ayurveda, Yoga, Unani, Siddha, Homeopathy conditions", 
         "Does this policy cover AYUSH treatments? What are the conditions?"),
        (7, "Hospital Definition", "Bed requirements and registration criteria", 
         "How is hospital defined in this policy?"),
        (8, "Claim Settlement Procedure", "Step-by-step claim process with timeline", 
         "What is the procedure for settling claims under this policy?"),
        (9, "Major Exclusions", "List of excluded treatments and conditions", 
         "What are the major exclusions in this policy?"),
        (10, "Co-payment Requirements", "Percentage and conditions by age/location", 
         "What are the co-payment requirements in this policy? How does it vary by age and location?"),
    ]

    # Run each test
    for number, desc, exp, q in tests:
        run_test(number, desc, exp, q)

    # Summary
    print("Test Questions Designed to Validate:")
    summary = [
        "Grace Period - Critical term accuracy",
        "Sum Insured - Numerical range extraction",
        "Room Rent - Percentage sub-limit precision",
        "Waiting Period - Time-based information",
        "Age Limits - Eligibility criteria",
        "AYUSH Coverage - Coverage scope",
        "Hospital Definition - Complex definition extraction",
        "Claim Procedure - Multi-step process",
        "Exclusions - Comprehensive negative coverage",
        "Co-payment - Multi-conditional requirements"
    ]
    for i, item in enumerate(summary, 1):
        print(f"  {i}. {item}")
    print("\nThese questions test:")
    print("  - Adaptive retrieval with complex queries")
    print("  - Complete clause extraction with 768 token budget")
    print("  - Multi-part answers with relaxed early stopping")
    print("  - Proper chunk overlap for sentence continuity")

if __name__ == "__main__":
    main()
