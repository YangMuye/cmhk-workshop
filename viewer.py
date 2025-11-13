import streamlit as st
import fitz  # PyMuPDF
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PDF to Markdown Viewer",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Helper functions
def get_pdf_files(directory="pdfs"):
    """Get list of PDF files in the specified directory"""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        return []
    return sorted([f.name for f in pdf_dir.glob("*.pdf")])

def get_markdown_path(pdf_path):
    """Get the corresponding markdown file path for a PDF"""
    return Path(pdf_path).with_suffix('.md')

def load_markdown(md_path):
    """Load markdown content from file"""
    if Path(md_path).exists():
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def render_pdf_page(pdf_path, page_num):
    """Render a specific page of PDF as image"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if page_num >= total_pages:
            page_num = total_pages - 1
        if page_num < 0:
            page_num = 0
        
        page = doc.load_page(page_num)
        # Render at higher resolution for better quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes, total_pages
    except Exception as e:
        st.error(f"Error rendering PDF: {str(e)}")
        return None, 0

# Main UI
st.title("📄 PDF to Markdown Viewer")

# File selector - in main pane
pdf_files = get_pdf_files()
if not pdf_files:
    st.error("No PDF files found in the 'pdfs' directory")
    st.stop()

col_selector, col_mode = st.columns([3, 1])
with col_selector:
    selected_pdf = st.selectbox("Select PDF", pdf_files)
with col_mode:
    md_display_mode = st.radio(
        "Markdown Display",
        ["Rendered", "Raw Text"],
        help="Choose how to display the markdown",
        horizontal=True
    )

pdf_path = Path("pdfs") / selected_pdf

# Check if markdown exists
md_path = get_markdown_path(pdf_path)
md_exists = md_path.exists()

if not md_exists:
    st.warning("⚠️ Markdown file not found. Please extract markdown first.")
    st.info(f"Expected: `{md_path}`")

# Get total pages
try:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()
except:
    total_pages = 1

# Main content area - two columns
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.header("Original PDF")
    
    # Page navigation controls
    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        page_input = st.number_input(
            "Go to page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page + 1,
            step=1,
            key="page_number",
            label_visibility="collapsed"
        )
        st.session_state.current_page = page_input - 1
    
    with nav_col2:
        st.markdown(f"**of {total_pages}**")
    
    # Render and display PDF page
    img_bytes, _ = render_pdf_page(str(pdf_path), st.session_state.current_page)
    if img_bytes:
        st.image(img_bytes, use_container_width=True)

with col_right:
    st.header("Markdown Output")
    
    if md_exists:
        md_content = load_markdown(md_path)
        
        if md_content:
            if md_display_mode == "Rendered":
                # Display rendered markdown in a container with scroll
                with st.container(height=800):
                    st.markdown(md_content)
            else:
                # Display raw markdown text
                st.text_area(
                    "Raw Markdown",
                    md_content,
                    height=800,
                    label_visibility="collapsed"
                )
        else:
            st.error("Failed to load markdown content")
    else:
        st.info("No markdown file available. Use the helper script to extract markdown from the PDF.")
        st.code("""
# Extract markdown using helper script:
python extract_markdown.py
        """, language="bash")

# Footer
st.divider()
st.caption("Built with Streamlit and PyMuPDF")

