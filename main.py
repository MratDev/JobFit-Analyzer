import tempfile
import streamlit as st
import json
import re
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="JobFit Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .candidate-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("JobFit Analyzer")
st.markdown("*Intelligent candidate evaluation powered by AI*")

# ===============================
# SESSION STATE INICIALIZÁCIA
# ===============================
if "desc" not in st.session_state:
    st.session_state["desc"] = ""
if "users" not in st.session_state:
    st.session_state["users"] = []
if "cvs" not in st.session_state:
    st.session_state["cvs"] = []
if "job_extracted" not in st.session_state:
    st.session_state["job_extracted"] = False
if "cvs_extracted" not in st.session_state:
    st.session_state["cvs_extracted"] = False
if "matching_results" not in st.session_state:
    st.session_state["matching_results"] = []
if "job_title" not in st.session_state:
    st.session_state["job_title"] = ""

# ===============================
# HELPER FUNCTIONS
# ===============================
def parse_json_safe(text):
    """Bezpečné parsovanie JSON z LLM outputu"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return {
        "error": "Could not parse JSON",
        "raw_response": text,
        "name": "Parse Error",
        "email": "",
        "phone": "",
        "education": "Could not extract",
        "experience": [],
        "skills": []
    }

def extract_score_from_text(text):
    """Extraktuje skóre z textu"""
    score_patterns = [
        r'MATCH SCORE:\s*(\d+)',
        r'Score:\s*(\d+)',
        r'(\d+)%',
        r'(\d+)/100'
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 50  # Default score

def get_score_color_class(score):
    """Vráti CSS triedu podľa skóre"""
    if score >= 80:
        return "score-high"
    elif score >= 60:
        return "score-medium"
    else:
        return "score-low"

def create_skills_comparison_chart(job_skills, candidate_skills):
    """Vytvorí chart porovnania skills"""
    # Jednoduchá logika pre demo - v realite by si mohol použiť NLP similarity
    job_skills_lower = [skill.lower().strip() for skill in job_skills]
    candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
    
    matched = []
    missing = []
    
    for skill in job_skills:
        if any(skill.lower() in cs for cs in candidate_skills_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    
    fig = go.Figure(data=[
        go.Bar(name='Matched Skills', x=['Skills'], y=[len(matched)], marker_color='green'),
        go.Bar(name='Missing Skills', x=['Skills'], y=[len(missing)], marker_color='red')
    ])
    
    fig.update_layout(barmode='stack', title='Skills Match Analysis', height=300)
    return fig

# ===============================
# ENHANCED FUNCTIONS
# ===============================

def Enhanced_JD_Upload():
    with st.container():
        st.subheader("📋 Job Description Analysis")
        
        if st.session_state["job_extracted"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"✅ Job: {st.session_state.get('job_title', 'Position')} - Analysis Complete")
            with col2:
                if st.button("🔄 New Job", key="reset_jd"):
                    st.session_state["job_extracted"] = False
                    st.session_state["desc"] = ""
                    st.session_state["job_title"] = ""
                    st.session_state["matching_results"] = []
                    st.rerun()
            
            with st.expander("📋 View Job Requirements"):
                st.write(st.session_state["desc"])
            return
        
        # Job title input
        job_title = st.text_input("Job Title (Optional)", key="job_title_input", 
                                placeholder="e.g., Senior Python Developer")
        
        JD = st.text_area("Paste Job Description here", height=300, key="jd_input",
                         placeholder="Paste the complete job description here...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            load = st.button("🔍 Analyze Job", key="jd_extract", type="primary")
        
        if load and JD:
            with st.spinner("🔍 Analyzing job requirements..."):
                # Enhanced job analysis prompt
                messages = [
                    ("system", "You are an expert HR analyst specializing in job requirement extraction."),
                    ("human", f"""
                    Analyze this job description and provide a comprehensive breakdown:
                    
                    1. REQUIRED SKILLS (technical & soft skills)
                    2. EXPERIENCE REQUIREMENTS (years, specific experience)
                    3. EDUCATION REQUIREMENTS 
                    4. KEY RESPONSIBILITIES
                    5. NICE-TO-HAVE QUALIFICATIONS
                    6. COMPANY CULTURE FIT INDICATORS
                    
                    Format your response clearly with headers and bullet points.
                    
                    Job Description:
                    {JD}
                    """),
                ]
                job_description = llm.invoke(messages)
                st.session_state["desc"] = job_description.content
                st.session_state["job_title"] = job_title if job_title else "Position"
                st.session_state["job_extracted"] = True
            
            st.success("✅ Job analysis completed!")
            st.write("**Extracted Requirements:**")
            st.write(st.session_state["desc"])

def Enhanced_CV_Upload():
    with st.container():
        st.subheader("📄 Candidate CV Processing")
        
        # Current status
        if st.session_state["cvs"]:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f"📂 {len(st.session_state['cvs'])} CV(s) loaded")
            with col2:
                if st.button("➕ Add More", key="add_more"):
                    pass  # Just refresh to show uploader
            with col3:
                if st.button("🗑️ Clear All", key="clear_cvs"):
                    st.session_state["cvs"] = []
                    st.session_state["users"] = []
                    st.session_state["cvs_extracted"] = False
                    st.session_state["matching_results"] = []
                    st.rerun()
            
            # Show loaded files
            with st.expander("📋 View Loaded Files"):
                for i, cv in enumerate(st.session_state["cvs"]):
                    for filename in cv.keys():
                        st.write(f"📄 {filename}")
        
        # File uploader
        CVs = st.file_uploader(
            "Upload CV files", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            key="cv_uploader",
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button("🚀 Process CVs", key="cv_extract", type="primary")
        
        if process_button and CVs:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_files_count = 0
            
            for i, file in enumerate(CVs):
                progress_bar.progress((i + 1) / len(CVs))
                status_text.text(f"Processing {file.name}...")
                
                # Check for duplicates
                file_exists = any(file.name in cv for cv in st.session_state["cvs"])
                
                if not file_exists:
                    with tempfile.NamedTemporaryFile(delete=False) as temporary_file:
                        temporary_file.write(file.read())
                    
                    try:
                        filename = file.name.lower()
                        if filename.endswith(".pdf"):
                            loader = PyPDFLoader(temporary_file.name)
                        elif filename.endswith(".docx"):
                            loader = Docx2txtLoader(temporary_file.name)
                        elif filename.endswith(".txt"):
                            loader = TextLoader(temporary_file.name, encoding="utf-8")
                        
                        loaded_file = loader.load()
                        st.session_state["cvs"].append({file.name: loaded_file})
                        new_files_count += 1
                        
                    except Exception as e:
                        st.error(f"❌ Error loading {file.name}: {e}")
                        
            status_text.text("Processing complete!")
            progress_bar.progress(1.0)
            
            if new_files_count > 0:
                st.success(f"📂 {new_files_count} new file(s) processed")
                Enhanced_CV_Extraction()

def Enhanced_CV_Extraction():
    """Enhanced CV extraction s lepším parsovaním"""
    if not st.session_state["cvs"]:
        return
    
    # Enhanced CV extraction prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert CV parser with deep understanding of various CV formats and structures.
    
    Extract comprehensive information from this CV text and respond with ONLY valid JSON.
    
    Be thorough in extracting:
    - Full name (look for name at the top, in headers, or signatures)
    - Contact information (email, phone, LinkedIn if present)
    - Education details (degrees, institutions, years)
    - Work experience (roles, companies, duration, key achievements)
    - Technical and soft skills
    - Certifications, languages, or other relevant info
    
    CV Text:
    {cv_text}
    
    Return ONLY this JSON format:
    {{
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "phone number or empty string",
        "linkedin": "LinkedIn profile or empty string",
        "education": "Comprehensive education summary",
        "experience": [
            {{
                "role": "Job Title",
                "company": "Company Name", 
                "years": "YYYY-YYYY or YYYY-Present",
                "achievements": "Key achievements or responsibilities"
            }}
        ],
        "skills": ["skill1", "skill2", "skill3"],
        "certifications": ["cert1", "cert2"],
        "languages": ["language1", "language2"],
        "summary": "Brief professional summary"
    }}
    """)
    
    users_data = []
    
    with st.spinner("🔍 Extracting candidate information..."):
        extraction_progress = st.progress(0)
        
        for i, cv in enumerate(st.session_state["cvs"]):
            extraction_progress.progress((i + 1) / len(st.session_state["cvs"]))
            
            for filename, content in cv.items():
                try:
                    cv_info = llm.invoke(prompt_template.format_prompt(cv_text=content[0].page_content).to_messages())
                    parsed_data = parse_json_safe(cv_info.content)
                    parsed_data["filename"] = filename
                    parsed_data["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    users_data.append(parsed_data)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {e}")
                    users_data.append({
                        "filename": filename,
                        "error": str(e),
                        "name": "Processing Error",
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
        
        extraction_progress.progress(1.0)
    
    st.session_state["users"] = users_data
    st.session_state["cvs_extracted"] = True
    
    # Display extracted candidates
    st.success(f"✅ Processed {len(users_data)} candidate(s)")
    
    with st.expander("👥 View Extracted Candidates"):
        for user in users_data:
            if "error" not in user:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{user.get('name', 'Unknown')}**")
                    st.write(f"📧 {user.get('email', 'N/A')}")
                    st.write(f"📞 {user.get('phone', 'N/A')}")
                with col2:
                    st.write(f"🎓 {user.get('education', 'N/A')}")
                    st.write(f"💼 {len(user.get('experience', []))} work experiences")
                    st.write(f"🛠️ {len(user.get('skills', []))} skills")
                st.write("---")

def Enhanced_Matching_Analysis():
    """Pokročilá analýza kandidátov s vizualizáciami"""
    if not st.session_state["desc"] or not st.session_state["users"]:
        st.warning("⚠️ Please complete job analysis and CV processing first.")
        return
    
    st.subheader("🎯 Advanced Candidate Analysis")
    
    # Enhanced matching prompt
    matching_prompt = ChatPromptTemplate.from_template("""
    You are a senior HR consultant and talent acquisition expert. 
    Conduct a comprehensive analysis of this candidate against the job requirements.
    
    Provide your analysis in this EXACT format:
    
    OVERALL_SCORE: [number from 0-100]
    SKILLS_MATCH: [number from 0-100] 
    EXPERIENCE_MATCH: [number from 0-100]
    EDUCATION_MATCH: [number from 0-100]
    CULTURE_FIT: [number from 0-100]
    
    STRENGTHS:
    • [Specific strength 1]
    • [Specific strength 2] 
    • [Specific strength 3]
    
    CONCERNS:
    • [Specific concern 1]
    • [Specific concern 2]
    
    RECOMMENDATION: [STRONG_FIT/GOOD_FIT/WEAK_FIT/NO_FIT]
    
    DETAILED_ANALYSIS:
    [2-3 sentences providing detailed reasoning for the scores and recommendation]
    
    Job Requirements:
    {job_requirements}
    
    Candidate Profile:
    Name: {name}
    Education: {education}
    Experience: {experience}
    Skills: {skills}
    Summary: {summary}
    """)
    
    valid_candidates = [user for user in st.session_state["users"] 
                       if "error" not in user and user.get("name") != "Parse Error"]
    
    if not valid_candidates:
        st.error("❌ No valid candidates to analyze.")
        return
    
    # Analyze all candidates
    matching_results = []
    analysis_progress = st.progress(0)
    
    for i, user in enumerate(valid_candidates):
        analysis_progress.progress((i + 1) / len(valid_candidates))
        
        try:
            # Prepare candidate data
            experience_str = "; ".join([
                f"{exp.get('role', '')} at {exp.get('company', '')} ({exp.get('years', '')})" 
                for exp in user.get('experience', [])
            ])
            skills_str = ", ".join(user.get('skills', []))
            
            match_result = llm.invoke(matching_prompt.format_prompt(
                job_requirements=st.session_state["desc"],
                name=user.get('name', ''),
                education=user.get('education', ''),
                experience=experience_str,
                skills=skills_str,
                summary=user.get('summary', '')
            ).to_messages())
            
            # Parse results
            result_text = match_result.content
            
            # Extract scores and data
            analysis_data = {
                'candidate': user,
                'analysis_text': result_text,
                'overall_score': extract_score_from_text(result_text),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            # Extract other scores
            for score_type in ['SKILLS_MATCH', 'EXPERIENCE_MATCH', 'EDUCATION_MATCH', 'CULTURE_FIT']:
                pattern = f'{score_type}:\\s*(\\d+)'
                match = re.search(pattern, result_text)
                analysis_data[score_type.lower()] = int(match.group(1)) if match else 50
            
            matching_results.append(analysis_data)
            
        except Exception as e:
            st.error(f"❌ Error analyzing {user.get('name', 'candidate')}: {str(e)}")
    
    analysis_progress.progress(1.0)
    st.session_state["matching_results"] = matching_results
    
    # Display results with enhanced UI
    Display_Enhanced_Results(matching_results)

def Display_Enhanced_Results(matching_results):
    """Zobrazí výsledky s pokročilými vizualizáciami"""
    
    # Sort by overall score
    sorted_results = sorted(matching_results, key=lambda x: x['overall_score'], reverse=True)
    
    # Overview dashboard
    st.subheader("📊 Analysis Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sum(r['overall_score'] for r in matching_results) / len(matching_results)
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    with col2:
        top_candidate = sorted_results[0] if sorted_results else None
        top_score = top_candidate['overall_score'] if top_candidate else 0
        st.metric("Top Candidate", f"{top_score}%")
    
    with col3:
        strong_fits = len([r for r in matching_results if r['overall_score'] >= 80])
        st.metric("Strong Fits", strong_fits)
    
    with col4:
        candidates_count = len(matching_results)
        st.metric("Total Analyzed", candidates_count)
    
    # Score distribution chart
    if len(matching_results) > 1:
        scores_df = pd.DataFrame([
            {
                'Candidate': r['candidate'].get('name', 'Unknown')[:20],
                'Overall Score': r['overall_score'],
                'Skills': r.get('skills_match', 50),
                'Experience': r.get('experience_match', 50),
                'Education': r.get('education_match', 50),
                'Culture Fit': r.get('culture_fit', 50)
            }
            for r in sorted_results
        ])
        
        fig = px.bar(scores_df, x='Candidate', y='Overall Score',
                    title='Candidate Scores Comparison',
                    color='Overall Score',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed candidate analysis
    st.subheader("👥 Detailed Candidate Analysis")
    
    for i, result in enumerate(sorted_results):
        candidate = result['candidate']
        score = result['overall_score']
        
        # Candidate card
        with st.container():
            st.markdown(f"### #{i+1} {candidate.get('name', 'Unknown')} - **{score}%**")
            st.write(f"**File:** {candidate.get('filename', 'Unknown')}")
            
            # Score breakdown
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Overall", f"{score}%")
            with col2:
                st.metric("Skills", f"{result.get('skills_match', 50)}%")
            with col3:
                st.metric("Experience", f"{result.get('experience_match', 50)}%")
            with col4:
                st.metric("Education", f"{result.get('education_match', 50)}%")
            with col5:
                st.metric("Culture", f"{result.get('culture_fit', 50)}%")
            
            # Detailed analysis
            with st.expander(f"📋 Detailed Analysis - {candidate.get('name', 'Unknown')}"):
                st.write(result['analysis_text'])
                
                # Candidate details
                st.write("**Candidate Details:**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"📧 **Email:** {candidate.get('email', 'N/A')}")
                    st.write(f"📞 **Phone:** {candidate.get('phone', 'N/A')}")
                    st.write(f"🎓 **Education:** {candidate.get('education', 'N/A')}")
                
                with col_b:
                    st.write(f"💼 **Experience:** {len(candidate.get('experience', []))} positions")
                    st.write(f"🛠️ **Skills:** {len(candidate.get('skills', []))} skills")
                    if candidate.get('certifications'):
                        st.write(f"📜 **Certifications:** {len(candidate.get('certifications', []))}")
            
            st.write("---")

def Export_Results():
    """Export výsledkov do CSV/Excel"""
    if not st.session_state.get("matching_results"):
        return
    
    st.subheader("📤 Export Results")
    
    # Prepare data for export
    export_data = []
    for result in st.session_state["matching_results"]:
        candidate = result['candidate']
        export_data.append({
            'Name': candidate.get('name', ''),
            'Email': candidate.get('email', ''),
            'Phone': candidate.get('phone', ''),
            'Overall_Score': result['overall_score'],
            'Skills_Match': result.get('skills_match', ''),
            'Experience_Match': result.get('experience_match', ''),
            'Education_Match': result.get('education_match', ''),
            'Culture_Fit': result.get('culture_fit', ''),
            'Filename': candidate.get('filename', ''),
            'Analysis_Date': result['timestamp'],
            'Job_Title': st.session_state.get('job_title', 'Position')
        })
    
    df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"candidate_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel by potreboval openpyxl, ale CSV stačí pre väčšinu prípadov
        st.info("💡 CSV format contains all analysis data")

# ===============================
# MAIN APP LAYOUT
# ===============================

# Main content area
Enhanced_JD_Upload()
st.write("---")
Enhanced_CV_Upload()

# Results section
if st.session_state.get("cvs_extracted") and st.session_state.get("job_extracted"):
    st.write("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🎯 Run Advanced Analysis", key="advanced_analysis", type="primary"):
            Enhanced_Matching_Analysis()
    
    # Export section
    if st.session_state.get("matching_results"):
        with col2:
            Export_Results()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("🔧 Control Panel")
    
    # Status overview
    st.subheader("📊 Status")
    job_status = "✅" if st.session_state["job_extracted"] else "⏳"
    cv_status = "✅" if st.session_state["cvs"] else "⏳"
    analysis_status = "✅" if st.session_state.get("matching_results") else "⏳"
    
    st.write(f"{job_status} Job Analysis")
    st.write(f"{cv_status} CV Processing ({len(st.session_state['cvs'])} files)")
    st.write(f"{analysis_status} Candidate Matching")
    
    st.write("---")
    
    # Quick stats
    if st.session_state.get("matching_results"):
        st.subheader("📈 Quick Stats")
        results = st.session_state["matching_results"]
        avg_score = sum(r['overall_score'] for r in results) / len(results)
        
        st.metric("Average Score", f"{avg_score:.1f}%")
        st.metric("Candidates", len(results))
        
        strong_candidates = len([r for r in results if r['overall_score'] >= 80])
        st.metric("Strong Fits", strong_candidates)
    
    st.write("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    if st.button("🗑️ Reset All Data"):
        for key in ["desc", "users", "cvs", "job_extracted", "cvs_extracted", "matching_results", "job_title"]:
            if key in ["users", "cvs", "matching_results"]:
                st.session_state[key] = []
            elif key in ["desc", "job_title"]:
                st.session_state[key] = ""
            else:
                st.session_state[key] = False
        st.rerun()
    
    # Help section
    st.write("---")
    st.subheader("❓ Help")
    with st.expander("How to use"):
        st.write("""
        1. **Add Job Description**: Paste job posting and analyze
        2. **Upload CVs**: Add candidate resume files
        3. **Run Analysis**: Compare candidates against job requirements  
        4. **Export Results**: Download analysis as CSV
        """)
    
    with st.expander("Supported formats"):
        st.write("📄 PDF, DOCX, TXT files")
    
    # About
    st.write("---")
    st.caption("🎯 Resume Tailor v2.0")
    st.caption("Powered by OpenAI GPT-4")
