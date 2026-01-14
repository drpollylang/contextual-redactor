#!/usr/bin/env python3
"""
Test script for the extract_emails method in utils.
This script analyzes a sample PDF document, extracts the emails, and returns them as a List of Dicts containing info about
each email (content, span, bounding region).
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from redactor.azure_client import AzureAIClient
from redactor.utils import extract_emails, parse_thread, contains_email_message

def test_extract_emails(client, test_doc_filename: str):
    # Path to a test document containing NO emails
    # test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', 'CareActAssessmentTest.pdf')
    test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', test_doc_filename)

    if not os.path.exists(test_doc_path):
        print(f"Test document not found: {test_doc_path}")
        return

    print(f"Analyzing document: {test_doc_path}")

    # Analyze the document
    try:
        result = client.analyse_document(test_doc_path)
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return None

    # Contains emails?
    if contains_email_message(result):
        print("This document contains at least one email message.")
    else:
        print("No full email messages detected.")
        return

    # Extract emails
    try:
        # emails = extract_emails(result)
        emails = parse_thread(result.content)
    except Exception as e:
        print(f"Error extracting emails: {e}")
        return

    print(f"Found {len(emails)} emails.")

    # Print results
    for i, email in enumerate(emails):
        print(f"\nEmail {i+1}:")
        # print(f"  Content: {email['content'][:100]}...")  # Truncate for readability
        # print(f"  Content: {email['content']}")
        # print(f"  Span: {email['span']}")
        # print(f"  Bounding Region: {email['bounding_region']}")
        # print()
        # print(email)
        # print(f"  From: {email['from_raw']}")
        # print(f"  To: {email['to_raw']} - {email['to']}")
        # print(f"  CC: {email['cc_raw']} - {email['cc']}")
        # print(f"  BCC: {email['bcc_raw']} - {email['bcc']}")
        # print(f"  subject: {email['subject']}")
        # print(f"  date: {email['date']}")
        # print(f"  message id: {email['message_id']}")
        # print(f"  in reply to: {email['in_reply_to']}")
        # print(f"  references: {email['references']}")
        # print(f"  content type: {email['content_type']}")
        print(f"  headers: {email['headers']}")
        print(f"  body: {email['body']}")
        print(f"  spans: {email['span']}")

    return len(emails)


def test_detect_and_extract_emails_wrapper():
    load_dotenv()
    
    # Initialize the client
    try:
        client = AzureAIClient()
    except Exception as e:
        print(f"Error initializing AzureAIClient: {e}")
        return

    # Test doc with no emails
    print('Testing document with no emails:')
    test_doc_no_emails = 'CareActAssessmentTest.pdf'
    no_emails_test = test_extract_emails(client, test_doc_no_emails)
    assert no_emails_test is None, f"Expected 0 emails, found {no_emails_test}"

    # # Test doc with 5 emails
    print('-'*50 + '\nTesting document with 5 emails:')
    test_doc_with_emails = 'Telefonica Redaction Email Example 2 Parental Custody.pdf'
    five_emails_test = test_extract_emails(client, test_doc_with_emails)
    assert five_emails_test == 5, f"Expected 5 emails, found {five_emails_test}"

    # Different test doc with 2 emails
    print('-'*50 + '\nTesting document with 2 emails:')
    test_doc_with_2emails = 'Telefonica Redaction Email Example 3 Disability.pdf'
    two_emails_test = test_extract_emails(client, test_doc_with_2emails)
    assert two_emails_test == 2, f"Expected 2 emails, found {two_emails_test}"

    print('-'*50 + '\nTesting stacked email thread with 5 (stacked) emails:')
    stacked_test_doc = 'FakeStackedEmailThread.pdf'
    stacked_test = test_extract_emails(client, stacked_test_doc)
    assert stacked_test == 5, f"Expected 5 emails, found {stacked_test}"

if __name__ == "__main__":
    test_detect_and_extract_emails_wrapper()