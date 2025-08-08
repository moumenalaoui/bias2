#!/usr/bin/env python3
"""
Hybrid PDF extraction script that combines Marker and custom parsing
Uses Marker for comprehensive extraction, then parses markdown into structured JSONL
"""

import os
import sys
import re
import json
import pandas as pd
import subprocess
import time
import shutil
from pathlib import Path

def clean_text_noise(text: str) -> str:
    """
    Remove noise from text while preserving data quality
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text with noise removed
    """
    if not text:
        return text
    
    # Remove figure references and image placeholders
    text = re.sub(r'#{1,6}\s*Figure\s+[IVX]+', '', text)
    text = re.sub(r'#{1,6}\s*\*\*Figure\s+[IVX]+[^*]*\*\*', '', text)
    text = re.sub(r'!\[\]\([^)]+\.(jpeg|jpg|png|gif)\)', '', text)
    
    # Remove source citations
    text = re.sub(r'\*Source\*:\s*[^.]*\.', '', text)
    text = re.sub(r'\*Source\*:\s*[^.]*', '', text)
    
    # Remove page references and anchors
    text = re.sub(r'\[[^]]*\]\(#[^)]*\)', '', text)
    text = re.sub(r'<span[^>]*></span>', '', text)
    
    # Remove formatting artifacts
    text = re.sub(r'\*\*_{10,}\*\*', '', text)  # Remove underscore lines
    text = re.sub(r'<sup>\*</sup>', '', text)   # Remove asterisk superscripts
    text = re.sub(r'<span[^>]*id="[^"]*"[^>]*></span>', '', text)
    
    # Remove standalone figure numbers
    text = re.sub(r'^#{1,6}\s*Figure\s+[IVX]+[^#\n]*$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines and excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple empty lines to double
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    
    # Remove leading/trailing whitespace from lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_with_marker(pdf_path: str, output_dir: str, report_id: str = ""):
    """
    Extract text from PDF using Marker/Nougat
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        report_id: Optional report ID for naming
    
    Returns:
        dict: Results including output path, processing time, and stats
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary folder for Marker (it expects a folder, not a single file)
    temp_folder = os.path.join(output_dir, "temp_pdf_folder")
    
    # Clean up any existing temp folder
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    
    # Create fresh temp folder
    os.makedirs(temp_folder, exist_ok=True)
    
    # Copy PDF to temp folder
    pdf_name = os.path.basename(pdf_path)
    temp_pdf_path = os.path.join(temp_folder, pdf_name)
    shutil.copy2(pdf_path, temp_pdf_path)
    
    # Verify the temp folder and PDF exist
    if not os.path.exists(temp_folder):
        raise FileNotFoundError(f"Failed to create temp folder: {temp_folder}")
    
    if not os.path.exists(temp_pdf_path):
        raise FileNotFoundError(f"Failed to copy PDF to temp folder: {temp_pdf_path}")
    
    print(f"📁 Temp folder created: {temp_folder}")
    print(f"📄 PDF copied to temp folder: {temp_pdf_path}")
    
    print(f"🔍 Processing PDF with Marker: {pdf_path}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        start_time = time.time()
        
        # Run Marker conversion
        cmd = ["marker", temp_folder, "--output_format", "markdown"]
        
        print("🔄 Running Marker conversion...")
        print(f"🔧 Command: {' '.join(cmd)}")
        print(f"📁 Working directory: {os.getcwd()}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Marker conversion completed in {processing_time:.2f} seconds")
            
            # Find the output file in Marker's default location
            marker_output_dir = os.path.expanduser("~/Desktop/bias2/bias_env2/lib/python3.12/site-packages/conversion_results")
            pdf_name_no_ext = os.path.splitext(pdf_name)[0]
            
            # Marker creates a folder with the PDF name (without extension)
            marker_output_path = os.path.join(marker_output_dir, pdf_name_no_ext, f"{pdf_name_no_ext}.md")
            
            # If that doesn't exist, try the alternative naming pattern
            if not os.path.exists(marker_output_path):
                # Sometimes Marker uses different naming patterns
                possible_paths = [
                    os.path.join(marker_output_dir, pdf_name_no_ext, f"{pdf_name_no_ext}.md"),
                    os.path.join(marker_output_dir, pdf_name_no_ext, "output.md"),
                    os.path.join(marker_output_dir, pdf_name_no_ext, "document.md")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        marker_output_path = path
                        break
            
            if os.path.exists(marker_output_path):
                # Copy to our output directory
                final_output_path = os.path.join(output_dir, f"{pdf_name_no_ext}_marker.md")
                shutil.copy2(marker_output_path, final_output_path)
                
                # Add report ID at the top for easy detection
                with open(final_output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add report ID at the very top
                report_id_line = f"# Report ID: {report_id}\n\n"
                with open(final_output_path, 'w', encoding='utf-8') as f:
                    f.write(report_id_line + content)
                
                # Get file stats
                file_size = os.path.getsize(final_output_path)
                with open(final_output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"📄 Output saved to: {final_output_path}")
                print(f"📊 File size: {file_size:,} bytes")
                print(f"📊 Characters extracted: {len(content):,}")
                
                # Clean up temp folder
                shutil.rmtree(temp_folder)
                
                return {
                    'success': True,
                    'output_path': final_output_path,
                    'processing_time': processing_time,
                    'file_size': file_size,
                    'characters': len(content),
                    'error': None
                }
            else:
                print(f"❌ Marker output file not found: {marker_output_path}")
                return {
                    'success': False,
                    'output_path': None,
                    'processing_time': processing_time,
                    'file_size': 0,
                    'characters': 0,
                    'error': f"Output file not found: {marker_output_path}"
                }
        else:
            print(f"❌ Marker conversion failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            print(f"Command output: {result.stdout}")
            print(f"Temp folder contents: {os.listdir(temp_folder) if os.path.exists(temp_folder) else 'Folder not found'}")
            return {
                'success': False,
                'output_path': None,
                'processing_time': processing_time,
                'file_size': 0,
                'characters': 0,
                'error': f"Marker failed: {result.stderr}\nCommand output: {result.stdout}"
            }
    
    except subprocess.TimeoutExpired:
        print("❌ Marker conversion timed out after 10 minutes")
        return {
            'success': False,
            'output_path': None,
            'processing_time': 600,
            'file_size': 0,
            'characters': 0,
            'error': "Conversion timed out"
        }
    except Exception as e:
        print(f"❌ Marker conversion failed: {e}")
        return {
            'success': False,
            'output_path': None,
            'processing_time': 0,
            'file_size': 0,
            'characters': 0,
            'error': str(e)
        }
    finally:
        # Clean up temp folder if it exists
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def extract_with_fallback(pdf_path: str, output_dir: str, report_id: str = ""):
    """
    Fallback PDF extraction using PyPDF2 when Marker fails
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        report_id: Optional report ID for naming
    
    Returns:
        dict: Results including output path, processing time, and stats
    """
    
    try:
        import PyPDF2
        
        print(f"📄 Using PyPDF2 fallback extraction for: {pdf_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract text from PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"## Page {page_num}\n\n{text}\n")
            
            full_text = '\n'.join(text_content)
        
        # Save as markdown
        pdf_name = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_name)[0]
        output_path = os.path.join(output_dir, f"{pdf_name_no_ext}_marker.md")
        
        # Add report ID at the top
        report_id_line = f"# Report ID: {report_id}\n\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_id_line + full_text)
        
        # Get file stats
        file_size = os.path.getsize(output_path)
        
        print(f"✅ Fallback extraction completed")
        print(f"📄 Output saved to: {output_path}")
        print(f"📊 File size: {file_size:,} bytes")
        print(f"📊 Characters extracted: {len(full_text):,}")
        
        return {
            'success': True,
            'output_path': output_path,
            'processing_time': 0,  # Fast extraction
            'file_size': file_size,
            'characters': len(full_text),
            'error': None
        }
        
    except ImportError:
        return {
            'success': False,
            'output_path': None,
            'processing_time': 0,
            'file_size': 0,
            'characters': 0,
            'error': "PyPDF2 not installed. Install with: pip install PyPDF2"
        }
    except Exception as e:
        return {
            'success': False,
            'output_path': None,
            'processing_time': 0,
            'file_size': 0,
            'characters': 0,
            'error': f"Fallback extraction failed: {str(e)}"
        }

def extract_metadata_from_markdown(markdown_path: str, pdf_path: str = None):
    """
    Extract metadata (report_id, publication_date) from markdown content
    Falls back to PDF extraction if markdown doesn't contain the header
    
    Args:
        markdown_path: Path to the markdown file
        pdf_path: Path to original PDF for fallback extraction
    
    Returns:
        dict: Extracted metadata
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metadata = {}
    
    # Extract publication date (look for date patterns)
    date_patterns = [
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}/\d{1,2}/\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content[:2000])  # Search first 2000 chars
        if match:
            metadata['publication_date'] = match.group(1)
            break
    
    # Extract report ID from PDF since Marker doesn't capture the document header properly
    # The actual document header (S/2024/548) is not in the markdown
    if pdf_path:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pdf_text = doc[0].get_text()[:1000]  # First 1000 chars of first page
            doc.close()
            
            # Look for report ID in PDF header
            report_id_pattern = r'(S/\d{4}/\d+)'
            pdf_matches = re.findall(report_id_pattern, pdf_text)
            if pdf_matches:
                metadata['report_id'] = pdf_matches[0]
                print(f"📄 Extracted report ID from PDF: {metadata['report_id']}")
        except Exception as e:
            print(f"⚠️ Could not extract from PDF: {e}")
    
    return metadata

def parse_markdown_to_structured(markdown_path: str, report_id: str = "", publication_date: str = ""):
    """
    Parse Marker's markdown output into structured JSONL format
    
    Args:
        markdown_path: Path to the markdown file from Marker
        report_id: Optional report ID
        publication_date: Optional publication date
    
    Returns:
        list: List of structured paragraph dictionaries
    """
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean noise from the entire content first
    content = clean_text_noise(content)
    
    # Normalize content to handle line breaks within URLs
    # Fix URLs that are split across lines and missing closing parentheses
    content = re.sub(r'\]\(([^)]*)\n([^)]*)\)', r'](\1\2)', content)
    
    # Fix URLs that are missing the closing parenthesis (common with resolution URLs)
    content = re.sub(r'\]\(https://docs\.un\.org/en/S/RES/(\d+)\((\d{4})\)([^)]*)\)', r'](https://docs.un.org/en/S/RES/\1(\2))\3)', content)
    content = re.sub(r'\]\(https://undocs\.org/en/S/RES/(\d+)\((\d{4})\)([^)]*)\)', r'](https://undocs.org/en/S/RES/\1(\2))\3)', content)
    
    # Fix truncated resolution URLs that are missing closing parenthesis
    content = re.sub(r'\]\(https://docs\.un\.org/en/S/RES/(\d+)\((\d{4})\)([^)]*)$', r'](https://docs.un.org/en/S/RES/\1(\2))\3)', content, flags=re.MULTILINE)
    content = re.sub(r'\]\(https://undocs\.org/en/S/RES/(\d+)\((\d{4})\)([^)]*)$', r'](https://undocs.org/en/S/RES/\1(\2))\3)', content, flags=re.MULTILINE)
    
    # Extract all links from the entire content first (handles multi-line links)
    all_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    
    paragraphs = []
    lines = content.split('\n')
    
    current_header = ""
    current_subheader = ""
    current_paragraph_text = ""
    current_paragraph_id = None
    current_links = []
    
    # Enhanced patterns for UN reports - handle all markdown levels
    header_pattern = r'^#{1,6}\s*\*\*([IVX]+\.\s+[^*]+)\*\*'
    subheader_pattern = r'^#{1,6}\s*\*\*([A-Z]\.\s+[^*]+)\*\*'
    paragraph_pattern = r'^(\d+)\.\s+(.+)$'
    annex_pattern = r'^#{1,6}\s*\*\*(Annex\s+[IVX]+|[A-Z]\.\s+Annex\s+[IVX]+)\*\*'
    # Link pattern that handles URLs with parentheses by looking for the end of the markdown link
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    # Additional patterns for different header formats - handle all markdown levels
    alt_header_pattern = r'^#{1,6}\s*([IVX]+\.\s+[^#\n]+)$'
    alt_subheader_pattern = r'^#{1,6}\s*([A-Z]\.\s+[^#\n]+)$'
    
    # Track current section context
    current_section = ""
    current_subsection = ""
    
    def flush_paragraph():
        nonlocal current_paragraph_text, current_paragraph_id, current_links
        if current_paragraph_id and current_paragraph_text.strip():
            # Clean the paragraph text before adding
            cleaned_text = clean_text_noise(current_paragraph_text.strip())
            if cleaned_text:  # Only add if there's still content after cleaning
                paragraphs.append({
                    "report_id": report_id,
                    "publication_date": publication_date,
                    "paragraph_id": current_paragraph_id,
                    "header": current_header,
                    "subheader": current_subheader,
                    "text": cleaned_text,
                    "links": current_links.copy()
                })
            current_paragraph_text = ""
            current_paragraph_id = None
            current_links = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for headers (I., II., III., etc.) - try multiple patterns
        header_match = re.match(header_pattern, line) or re.match(alt_header_pattern, line)
        if header_match:
            flush_paragraph()
            current_header = header_match.group(1).strip()
            current_subheader = ""
            current_section = current_header
            continue
        
        # Check for subheaders (A., B., C., etc.) - try multiple patterns
        subheader_match = re.match(subheader_pattern, line) or re.match(alt_subheader_pattern, line)
        if subheader_match:
            flush_paragraph()
            current_subheader = subheader_match.group(1).strip()
            current_subsection = current_subheader
            continue
        
        # Check for annexes
        if re.match(annex_pattern, line):
            flush_paragraph()
            current_header = "Annex"
            current_subheader = line.strip()
            continue
        
        # Check for numbered paragraphs
        para_match = re.match(paragraph_pattern, line)
        if para_match:
            flush_paragraph()
            current_paragraph_id = para_match.group(1)
            current_paragraph_text = para_match.group(2)
            continue
        
        # If we have a current paragraph, append to it
        if current_paragraph_id is not None:
            if current_paragraph_text:
                current_paragraph_text += " " + line
            else:
                current_paragraph_text = line
    
    # Flush the last paragraph
    flush_paragraph()
    
    # Assign links to paragraphs based on text matching
    for paragraph in paragraphs:
        paragraph_text = paragraph['text']
        paragraph_links = []
        
        # Find links that appear in this paragraph's text
        for link_text, link_url in all_links:
            # Clean the link text for comparison (remove markdown escaping)
            clean_link_text = link_text.replace('\\', '')
            
            # Try multiple matching strategies
            matched = False
            
            # Strategy 1: Direct text match
            if clean_link_text in paragraph_text:
                matched = True
            
            # Strategy 2: Match resolution number (e.g., "1701" or "2591")
            elif link_url.startswith('https://undocs.org/en/S/RES/'):
                # Extract resolution number from URL
                res_match = re.search(r'https://undocs\.org/en/S/RES/(\d+)', link_url)
                if res_match:
                    res_num = res_match.group(1)
                    # Check if resolution number appears in paragraph text
                    if res_num in paragraph_text:
                        matched = True
            
            # Strategy 3: Match with different parentheses formats
            elif '(' in clean_link_text and ')' in clean_link_text:
                # Try matching with unescaped parentheses
                unescaped_text = clean_link_text.replace('\\(', '(').replace('\\)', ')')
                if unescaped_text in paragraph_text:
                    matched = True
            
            if matched:
                # Fix truncated URLs by adding missing closing parentheses
                if link_url.startswith('https://docs.un.org/en/S/RES/') and not link_url.endswith(')'):
                    # Extract resolution number and year
                    match = re.search(r'https://docs\.un\.org/en/S/RES/(\d+)\((\d{4})\)?', link_url)
                    if match:
                        res_num = match.group(1)
                        year = match.group(2)
                        link_url = f'https://docs.un.org/en/S/RES/{res_num}({year})'
                
                # Fix truncated undocs.org URLs
                if link_url.startswith('https://undocs.org/en/S/RES/') and not link_url.endswith(')'):
                    # Extract resolution number and year
                    match = re.search(r'https://undocs\.org/en/S/RES/(\d+)\((\d{4})\)?', link_url)
                    if match:
                        res_num = match.group(1)
                        year = match.group(2)
                        link_url = f'https://undocs.org/en/S/RES/{res_num}({year})'
                
                paragraph_links.append({"text": link_text, "url": link_url})
        
        # Remove duplicate URLs while preserving unique link text variations
        unique_links = []
        seen_urls = set()
        
        for link in paragraph_links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        paragraph['links'] = unique_links
    
    return paragraphs

def extract_hybrid(pdf_path: str, output_dir: str, report_id: str = "", publication_date: str = ""):
    """
    Hybrid extraction: Marker + Custom parsing
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        report_id: Optional report ID
        publication_date: Optional publication date
    
    Returns:
        dict: Results including output paths and stats
    """
    
    print(f"🔍 Starting hybrid extraction: {pdf_path}")
    
    # Extract report info from filename if not provided
    pdf_name = os.path.basename(pdf_path)
    if not report_id:
        # Let the metadata extraction handle this
        report_id = ""
    
    # Step 1: Extract with Marker
    print("📥 Step 1: Extracting with Marker...")
    marker_result = extract_with_marker(pdf_path, output_dir, report_id)
    
    if not marker_result['success']:
        print(f"❌ Marker extraction failed: {marker_result['error']}")
        print("🔄 Trying fallback extraction method...")
        
        # Try fallback extraction using PyPDF2 or similar
        fallback_result = extract_with_fallback(pdf_path, output_dir, report_id)
        if not fallback_result['success']:
            print(f"❌ Fallback extraction also failed: {fallback_result['error']}")
            return marker_result  # Return original error
        
        print("✅ Fallback extraction successful!")
        marker_result = fallback_result
    
    # Step 2: Parse markdown to structured format
    print("🔄 Step 2: Parsing markdown to structured format...")
    markdown_path = marker_result['output_path']
    
    try:
        # Extract metadata from markdown content
        extracted_metadata = extract_metadata_from_markdown(markdown_path, pdf_path)
        
        # Use extracted metadata if not provided
        if not publication_date and extracted_metadata.get('publication_date'):
            publication_date = extracted_metadata['publication_date']
        if not report_id and extracted_metadata.get('report_id'):
            report_id = extracted_metadata['report_id']
        
        paragraphs = parse_markdown_to_structured(
            markdown_path, 
            report_id, 
            publication_date
        )
        
        # Step 3: Save as JSONL
        jsonl_path = markdown_path.replace('_marker.md', '_hybrid.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for para in paragraphs:
                f.write(json.dumps(para, ensure_ascii=False) + '\n')
        
        # Step 4: Save as CSV for compatibility
        csv_path = markdown_path.replace('_marker.md', '_hybrid.csv')
        df = pd.DataFrame(paragraphs)
        df.to_csv(csv_path, index=False)
        
        # Step 5: Also save to JSONL_outputs directory for API processing
        jsonl_outputs_dir = os.path.join(os.path.dirname(output_dir), "JSONL_outputs")
        os.makedirs(jsonl_outputs_dir, exist_ok=True)
        
        # Create structured JSONL file for API processing
        structured_jsonl_path = os.path.join(jsonl_outputs_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}-structured.jsonl")
        with open(structured_jsonl_path, 'w', encoding='utf-8') as f:
            for para in paragraphs:
                f.write(json.dumps(para, ensure_ascii=False) + '\n')
        
        # Step 5: Generate statistics
        total_links = sum(len(para.get('links', [])) for para in paragraphs)
        unique_headers = len(set(para['header'] for para in paragraphs if para['header']))
        unique_subheaders = len(set(para['subheader'] for para in paragraphs if para['subheader']))
        
        print(f"✅ Hybrid extraction completed!")
        print(f"📄 JSONL output: {jsonl_path}")
        print(f"📄 CSV output: {csv_path}")
        print(f"📄 Structured JSONL for API: {structured_jsonl_path}")
        print(f"📊 Statistics:")
        print(f"   • Paragraphs: {len(paragraphs)}")
        print(f"   • Links extracted: {total_links}")
        print(f"   • Unique headers: {unique_headers}")
        print(f"   • Unique subheaders: {unique_subheaders}")
        
        return {
            'success': True,
            'markdown_path': markdown_path,
            'jsonl_path': jsonl_path,
            'csv_path': csv_path,
            'paragraphs': len(paragraphs),
            'links': total_links,
            'headers': unique_headers,
            'subheaders': unique_subheaders,
            'processing_time': marker_result['processing_time'],
            'error': None
        }
        
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return {
            'success': False,
            'error': f"Parsing failed: {str(e)}",
            'processing_time': marker_result['processing_time']
        }

def main():
    """
    Main function for command line usage
    """
    if len(sys.argv) < 3:
        print("Usage: python hybrid_extractor.py <pdf_path> <output_dir> [report_id] [publication_date]")
        print("Example: python hybrid_extractor.py extraction/reports_pdf/report.pdf extraction/hybrid_output")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    report_id = sys.argv[3] if len(sys.argv) > 3 else ""
    publication_date = sys.argv[4] if len(sys.argv) > 4 else ""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    result = extract_hybrid(pdf_path, output_dir, report_id, publication_date)
    
    if result['success']:
        print(f"\n🎉 Hybrid extraction successful!")
        print(f"📊 Total processing time: {result['processing_time']:.2f} seconds")
    else:
        print(f"\n❌ Hybrid extraction failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main() 