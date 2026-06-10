import streamlit as st
import pandas as pd
import re
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import io

# -------------------------------------------------------------
# 1. Page Setup & Cinematic SaaS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="HireSight AI | Premium Talent Engine",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HIGH-END LUXURY CSS (F1/Cyber Themes)
st.markdown("""
<style>
    /* Global Background - Deep luxury Space */
    .stApp {
        background-color: #060B14;
        background-image: radial-gradient(circle at 50% 0%, #111A2E 0%, #060B14 70%);
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    /* Cinematic Hero Banner */
    .hero-container {
        text-align: center;
        padding: 60px 0 40px 0;
        margin-bottom: 20px;
        position: relative;
        animation: fadeDown 1s ease-out;
    }
    .neon-title {
        font-size: 5.5rem;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #FFFFFF 0%, #00E5FF 30%, #2563FF 70%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1;
        text-transform: uppercase;
    }
    .neon-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        color: #94A3B8;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 15px;
    }
    .hero-glow {
        height: 2px;
        width: 120px;
        background: #00E5FF;
        margin: 25px auto;
        box-shadow: 0 0 20px #00E5FF, 0 0 40px #00E5FF;
        border-radius: 5px;
    }

    /* Premium Glassmorphism Cards */
    .glass-card {
        background: rgba(13, 18, 30, 0.72);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 25px;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .card-title::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00E5FF;
        border-radius: 50%;
        box-shadow: 0 0 15px #00E5FF;
    }

    /* Animated Stat Cards (F1 Telemetry Vibe) */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-top: 2px solid #2563FF;
        padding: 25px 20px;
        border-radius: 12px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        border-top: 2px solid #00E5FF;
        box-shadow: 0 10px 30px rgba(0, 229, 255, 0.1);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    .metric-glow-cyan { text-shadow: 0 0 25px rgba(0, 229, 255, 0.6); color: #00E5FF;}
    .metric-glow-violet { text-shadow: 0 0 25px rgba(124, 58, 237, 0.6); color: #8B5CF6;}
    .metric-glow-green { text-shadow: 0 0 25px rgba(16, 185, 129, 0.6); color: #10B981;}

    /* Premium Skill Badges */
    .badge-wrap { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; margin-bottom: 25px;}
    .badge {
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.2s;
    }
    .badge-core { background: rgba(0, 229, 255, 0.1); color: #00E5FF; border: 1px solid rgba(0, 229, 255, 0.4); box-shadow: 0 0 15px rgba(0,229,255,0.1); }
    .badge-core:hover { background: rgba(0, 229, 255, 0.2); transform: scale(1.05); }
    
    .badge-support { background: rgba(148, 163, 184, 0.1); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.4); }
    
    .badge-missing { background: rgba(239, 68, 68, 0.1); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);}
    
    .badge-bonus { background: rgba(217, 70, 239, 0.1); color: #D946EF; border: 1px solid rgba(217, 70, 239, 0.4); }

    /* New Role Recommendation Cards */
    .role-card {
        background: linear-gradient(90deg, rgba(37, 99, 255, 0.1) 0%, transparent 100%);
        border-left: 4px solid #2563FF;
        border-radius: 0 8px 8px 0;
        padding: 18px 20px;
        margin-bottom: 15px;
        transition: transform 0.2s;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 15px;
    }
    .role-card:hover {
        transform: translateX(5px);
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.15) 0%, transparent 100%);
        border-left: 4px solid #00E5FF;
    }
    .role-header {
        font-size: 1.2rem;
        font-weight: 900;
        color: #F8FAFC;
        margin-bottom: 3px;
        letter-spacing: 0.5px;
    }
    .role-reason {
        color: #94A3B8;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .role-score {
        background: rgba(0, 229, 255, 0.1);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.4);
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 900;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }

    /* Custom Checklists & Bullet Points */
    .audit-list { list-style: none; padding: 0; margin: 0; }
    .audit-item { padding: 12px 0; border-bottom: 1px dashed rgba(255,255,255,0.05); font-size: 0.95rem; display: flex; justify-content: space-between; font-weight: 500;}
    .audit-item:last-child { border-bottom: none; }
    .audit-pass { color: #10B981; font-weight: 800; background: rgba(16, 185, 129, 0.1); padding: 2px 10px; border-radius: 12px; font-size:0.8rem;}
    .audit-fail { color: #EF4444; font-weight: 800; background: rgba(239, 68, 68, 0.1); padding: 2px 10px; border-radius: 12px; font-size:0.8rem;}

    .insight-bullet { margin-bottom: 15px; color: #CBD5E1; font-size: 0.95rem; line-height: 1.6; padding-left: 24px; position: relative;}
    .insight-bullet::before { content: '▹'; position: absolute; left: 0; color: #00E5FF; font-weight: 900; font-size: 1.2rem; top: -3px;}
    
    .summary-text { font-size: 1.15rem; line-height: 1.7; color: #E2E8F0; font-weight: 300; border-left: 4px solid #8B5CF6; padding-left: 20px; background: linear-gradient(90deg, rgba(139, 92, 246, 0.05) 0%, transparent 100%); padding-top: 15px; padding-bottom: 15px; border-radius: 0 8px 8px 0; }

    /* Match Circle */
    .match-ring {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%);
        border: 2px solid rgba(16, 185, 129, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto 20px auto;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.1);
    }
    .match-ring-value {
        font-size: 4rem;
        font-weight: 900;
        color: #10B981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        line-height: 1;
    }
    
    /* JD Component Analysis Table Rows */
    .jd-component-row { display:flex; justify-content:space-between; margin-bottom:12px; align-items:center; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 8px; }
    .jd-component-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0;}

    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #030509 !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Sidebar Navigation (Recruiter Dashboard Feel)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #F8FAFC; font-weight: 900; letter-spacing: -0.5px;'>HIRESIGHT<span style='color:#00E5FF;'>AI</span></h2>", unsafe_allow_html=True)
    st.caption("INTELLIGENT SCREENING PLATFORM")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); padding: 15px; border-radius: 8px; margin-bottom: 25px;'>
            <div style='color: #00E5FF; font-size: 0.75rem; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase;'>Network Status</div>
            <div style='color: #10B981; font-weight: 900; font-size: 1.1rem; margin-top: 5px; text-shadow: 0 0 10px rgba(16, 185, 129, 0.4);'>● SECURE & ONLINE</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='color: #94A3B8; font-size: 0.85rem; line-height: 2.2;'>
            <b style="color: #FFFFFF; font-size: 0.8rem; letter-spacing: 1.5px; text-transform:uppercase;">Platform Modules Active</b><br>
            ▹ Signal Extraction Matrix<br>
            ▹ Job Alignment Engine<br>
            ▹ Hybrid Role Recommendation<br>
            ▹ ATS Optimization Scanner<br>
            ▹ Intelligent Reporting
        </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. Fast Data Models Initialization
# -------------------------------------------------------------
@st.cache_resource
def load_hiring_models():
    data = {"Resume_Text": [], "Category": []}
    sample_resumes = [
        ("Python SQL Machine Learning Data Visualization Statistics Deep Learning NLP pandas scikit learn", "Data Science"),
        ("HTML CSS JavaScript React Nodejs Web Development Frontend Backend UI UX responsive design", "Web Developer"),
        ("Java Spring Boot Hibernate REST API SQL Object Oriented Programming Microservices Maven", "Java Developer"),
        ("Recruitment Employee Relations Payroll HR Policies Communication Talent Acquisition Interviewing", "HR"),
        ("Artificial Intelligence NLP Computer Vision TensorFlow PyTorch Deep Learning Neural Networks models optimization", "AI Engineer")
    ]
    for text, cat in sample_resumes:
        data["Resume_Text"].append(text)
        data["Category"].append(cat)
        
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["Resume_Text"])
    y = df["Category"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return vectorizer, model

engine_vectorizer, engine_model = load_hiring_models()


# -------------------------------------------------------------
# 4. Core Intelligence Algorithms & Hybrid Role Engine
# -------------------------------------------------------------
ALL_SKILLS = [
    "python", "sql", "machine learning", "data visualization", "statistics", "deep learning",
    "pandas", "scikit learn", "nlp", "tensorflow", "pytorch", "artificial intelligence",
    "html", "css", "javascript", "react", "nodejs", "web development", "frontend", "backend", "ui ux",
    "java", "spring boot", "hibernate", "rest api", "microservices", "maven",
    "recruitment", "employee relations", "payroll", "hr policies", "communication", "talent acquisition",
    "computer vision", "aws", "docker", "kubernetes", "agile", "scrum", "git", "cloud computing",
    "c++", "c#", "ruby", "go", "php", "jira", "numpy", "api", "database", "vue", "angular", "excel"
]

# EXPLICIT ROLE MAPPINGS FOR HYBRID MATCHING
ROLE_MAPPINGS = {
    "Full Stack Developer": ["html", "css", "javascript", "react", "nodejs", "backend", "frontend", "api", "database", "git", "sql", "mongodb"],
    "Web Developer": ["html", "css", "javascript", "frontend", "react", "web development", "ui ux", "backend"],
    "Data Science": ["python", "pandas", "machine learning", "statistics", "numpy", "sklearn", "sql", "deep learning", "nlp"],
    "AI Engineer": ["deep learning", "pytorch", "tensorflow", "nlp", "computer vision", "artificial intelligence", "python"],
    "Java Developer": ["java", "spring boot", "hibernate", "rest api", "sql", "microservices", "maven"],
    "Data Analyst": ["python", "sql", "data visualization", "pandas", "statistics", "data", "excel"],
    "Frontend Engineer": ["html", "css", "javascript", "react", "ui", "vue", "angular", "ui ux"],
    "HR": ["recruitment", "payroll", "employee relations", "hr policies", "talent acquisition", "communication", "hr"],
    "Cloud Engineer": ["aws", "docker", "kubernetes", "cloud computing", "agile", "git", "linux", "ci/cd"],
    "Software Engineer": ["python", "java", "c++", "c#", "javascript", "sql", "data structures", "git", "agile"]
}

def clean_resume_text(text):
    text = text.lower()
    text = text.replace("node.js", "nodejs").replace("scikit-learn", "scikit learn").replace("ai engineer", "ai")
    text = re.sub(r"[^\w\s\.\@\+]", " ", text)
    words = text.split()
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)

def extract_registered_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if skill in text_lower:
            found.add(skill)
    return sorted(list(found))

def hybrid_role_engine(raw_text_lower, ml_predicted_role):
    scores = {}
    for role, keywords in ROLE_MAPPINGS.items():
        hits = sum(1 for kw in keywords if kw in raw_text_lower)
        base_percentage = (hits / len(keywords)) * 100
        
        if role == ml_predicted_role and hits > 0:
            base_percentage += 30
            
        base_percentage += (hits * 5)
        scores[role] = min(100, int(base_percentage))
        
    ranked_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if ranked_roles[0][1] < 20:
        primary_role = "Software Engineer"
        ranked_roles.insert(0, ("Software Engineer", 75))
    else:
        primary_role = ranked_roles[0][0]
        
    suggestions = []
    reasonings = {
        "Data Analyst": "Features practical data extraction, processing capabilities, and database querying.",
        "Data Science": "Demonstrates statistical modeling capabilities along with core Python methodologies.",
        "Web Developer": "Solid alignment with web fundamentals and standard architecture.",
        "Frontend Engineer": "Highly specialized signature in user interfaces and rendering.",
        "Java Developer": "Highlights corporate backend design patterns via the JVM.",
        "HR": "Organizational personnel terminology signifies non-technical placement.",
        "AI Engineer": "Showcases heavy mathematics and specialized neural learning models.",
        "Cloud Engineer": "Concentrated signals targeting deployment pipelines and cloud infrastructure.",
        "Full Stack Developer": "Shows operative alignment across both UI generation and backend servers.",
        "Software Engineer": "Demonstrates broad structural engineering knowledge."
    }
    
    for role, score in ranked_roles:
        if score >= 35:
            reason = reasonings.get(role, "Matches technical keyword footprints associated with this role.")
            suggestions.append((role, score, reason))
            
    if len(suggestions) == 0:
        suggestions.append(("Software Engineer", 60, "Foundational knowledge applicable to broad roles."))
        
    return primary_role, suggestions[:4]

# ----------- IMPROVED MATCH CALCULATION LOGIC -----------------
def expand_short_jd_skills(jd_text_lower):
    """If the JD text is short (role-input), intelligently expand into required skill cluster."""
    expanded_set = set()
    jd_clean = jd_text_lower.lower().strip()
    
    if "full stack" in jd_clean or "fullstack" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Full Stack Developer"])
    if "web" in jd_clean or "frontend" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Web Developer"])
    if "data scien" in jd_clean or "machine learning" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Data Science"])
    if "ai" in jd_clean or "artificial" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["AI Engineer"])
    if "java" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Java Developer"])
    if "analyst" in jd_clean or "data" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Data Analyst"])
    if "software" in jd_clean:
        expanded_set.update(ROLE_MAPPINGS["Software Engineer"])
    
    return list(expanded_set)

def hybrid_jd_match_engine(resume_text_lower, jd_text_lower, resume_skills, jd_skills_extracted, primary_role):
    # Debug Safety Check #1: Ensure clean operation
    words_len = len(jd_text_lower.split())
    if words_len == 0:
        return 0, "No Input", "#94A3B8", "None", [], [], [], 0
    
    expanded_jd_skills = list(set(jd_skills_extracted))
    
    # If the user typed a super short JD (like "web developer"), dynamically expand it into skill cluster
    if words_len < 10:
        expanded_clusters = expand_short_jd_skills(jd_text_lower)
        expanded_jd_skills.extend(expanded_clusters)
        expanded_jd_skills = list(set(expanded_jd_skills))
        
    # Guard against 0 divisor if no recognized skills were populated
    if not expanded_jd_skills:
        expanded_jd_skills = ["communication", "teamwork", "software"] # fallback baseline
        
    matched_skills = [s for s in expanded_jd_skills if s in resume_skills]
    missing_skills = [s for s in expanded_jd_skills if s not in resume_skills]

    # COMPONENT 1: Skill Overlap Score (Weighted 50%)
    skill_coverage_ratio = len(matched_skills) / len(expanded_jd_skills)
    skill_score = skill_coverage_ratio * 50
    
    # COMPONENT 2: Semantic TF-IDF Similarity (Weighted 30%)
    tfidf_score = 0
    if words_len > 8:
        try:
            match_vectorizer = TfidfVectorizer()
            tfidf_matrix = match_vectorizer.fit_transform([resume_text_lower, jd_text_lower])
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            tfidf_score = min(30, similarity * 40)
        except:
            pass
    else:
        # Prevent penalizing deeply short text inputs (role-based inputs)
        tfidf_score = skill_coverage_ratio * 30
        
    # COMPONENT 3: Role Alignment Score (Weighted 20%)
    role_score = 0
    jd_clean = jd_text_lower.replace("developer", "").replace("engineer", "").replace("specialist", "").strip()
    primary_role_lower = primary_role.lower()
    
    if any(k in jd_text_lower for k in primary_role_lower.split() if len(k) > 2):
        role_score = 20
    else:
        role_kws = ROLE_MAPPINGS.get(primary_role, [])
        hits = sum(1 for kw in role_kws if kw in jd_text_lower)
        if hits >= 2: role_score = 15
        elif hits == 1: role_score = 10
            
    # Final Tabulation
    final_score = int(skill_score + tfidf_score + role_score)
    final_score = max(0, min(100, final_score))
    
    # Recruiter Evaluation Output
    if final_score >= 75: label, color = "Strong Fit", "#10B981"
    elif final_score >= 50: label, color = "Good Potential", "#00E5FF"
    elif final_score >= 30: label, color = "Partial Match", "#FBBF24"
    else: label, color = "Low Alignment", "#EF4444"
    
    role_align_text = "High" if role_score >= 15 else ("Medium" if role_score >= 10 else "Low")
    coverage_perc = int(skill_coverage_ratio * 100)
    
    return final_score, label, color, role_align_text, expanded_jd_skills, matched_skills, missing_skills, coverage_perc

# ----------- REALISTIC ATS ENGINE -----------------
def evaluate_ats_readiness(raw_text, words_count, skill_count):
    text = raw_text.lower()
    email_found = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    phone_found = bool(re.search(r'\+?\d[\d -]{8,12}\d', text))
    skills_found = "skill" in text
    edu_found = any(w in text for w in ["education", "degree", "university", "college", "btech", "b.tech", "bsc", "b.sc"])
    exp_found = any(w in text for w in ["experience", "employment", "work history", "internship"])
    proj_found = any(w in text for w in ["project", "portfolio"])
    cert_found = any(w in text for w in ["certificat", "course"])
    
    audit = {
        "Contact Data (Email)": email_found,
        "Contact Data (Phone)": phone_found,
        "Education Credentials": edu_found,
        "Employment History": exp_found,
        "Project Architecture": proj_found,
        "Technical Skills Block": skills_found,
        "Training / Certifications": cert_found
    }
    
    ats_score = 0
    if email_found and phone_found: ats_score += 15
    elif email_found or phone_found: ats_score += 5
    
    if edu_found: ats_score += 15
    if exp_found: ats_score += 25
    if proj_found: ats_score += 15
    if skills_found: ats_score += 10
    if cert_found: ats_score += 5
    
    if words_count < 150: ats_score -= 15
    elif words_count > 600: ats_score -= 5
    
    if skill_count >= 8: ats_score += 15
    elif skill_count >= 4: ats_score += 5
    
    ats_score = max(0, min(100, ats_score))
    if ats_score == 100 and skill_count < 15: ats_score = 94
        
    return audit, ats_score

def calculate_resume_strength(words_count, skill_count, ats_score, raw_text):
    score = 0
    if words_count > 350: score += 25
    elif words_count > 200: score += 15
    else: score += 5
    if skill_count >= 12: score += 35
    elif skill_count >= 6: score += 20
    else: score += 10
    score += (ats_score * 0.4)
    return min(100, int(score))

# ----------- REALISTIC PROFESSIONAL INSIGHTS -----------------
def compile_recruiter_insights(audit, strong_skills, words, match_perc=None):
    insights = []
    if len(strong_skills) >= 4:
        insights.append("Profile demonstrates a solid technical foundation with identifiable core competencies.")
    else:
        insights.append("Technical keyword presence is low; may require further assessment to verify capabilities.")
        
    if audit["Project Architecture"] and audit["Employment History"]:
        insights.append("Candidate effectively validates skills with both hands-on projects and work history.")
    elif not audit["Project Architecture"] and not audit["Employment History"]:
        insights.append("Resume currently lacks formal experiential validation (projects/employment).")
        
    if words < 150:
        insights.append("Content volume is unusually low, reducing keyword visibility for ATS platforms.")
        
    if match_perc is not None:
        if match_perc >= 75:
            insights.append("Presents excellent conceptual and practical alignment with the required job parameters.")
        elif match_perc < 45:
            insights.append("Noticeable gaps identified between candidate capabilities and direct job requirements.")
            
    if not insights:
        insights.append("Candidate passes standard automated checks with expected formatting.")
        
    return insights

def generate_candidate_summary(role, strength, skills, ats_score):
    if len(skills) > 0:
        skill_snippet = f"demonstrating practical exposure to modern tooling including {', '.join(skills[:3])}."
    else:
        skill_snippet = "exhibiting foundational technical proficiencies."
        
    if strength >= 80: tier = "strong potential fit"
    elif strength >= 55: tier = "moderate fit"
    else: tier = "developing candidate"
        
    summary = f"Based on structural and semantic analysis, this candidate shows a {tier} geared towards {role} positions. The resume achieves an ATS readiness score of {ats_score}%, {skill_snippet}"
    return summary

# -------------------------------------------------------------
# 5. Dashboard Layout & Presentation
# -------------------------------------------------------------
# SECTION 1: Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="neon-title">HIRESIGHT AI</div>
    <div class="neon-subtitle">ENTERPRISE ROLE & TALENT EVALUATION SUITE</div>
    <div class="hero-glow"></div>
</div>
""", unsafe_allow_html=True)

# SECTION 2: Input Controls
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>TALENT INGESTION CONSOLE</div>", unsafe_allow_html=True)

col_in1, col_in2 = st.columns(2, gap="large")

with col_in1:
    st.markdown("<div style='color: #94A3B8; font-weight: 700; margin-bottom:12px; font-size: 0.8rem; letter-spacing:1px;'>1. UPLOAD PRIMARY RESUME PAYLOAD</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF / TXT", type=["pdf", "txt"], label_visibility="collapsed")

with col_in2:
    st.markdown("<div style='color: #94A3B8; font-weight: 700; margin-bottom:12px; font-size: 0.8rem; letter-spacing:1px;'>2. INJECT JOB DESCRIPTION TARGET (OPTIONAL)</div>", unsafe_allow_html=True)
    job_description = st.text_area(
        "JD Input",
        height=100,
        placeholder="Paste role requirements here to activate the Compatibility Matrix...",
        label_visibility="collapsed"
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 6. Execution Pipeline
# -------------------------------------------------------------
if uploaded_file is not None:
    
    with st.spinner("Initializing Diagnostics Workbench..."):
        time.sleep(0.6)

    if uploaded_file.name.endswith(".pdf"):
        try:
            pdf = PdfReader(uploaded_file)
            raw_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except:
            raw_text = ""
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    if not raw_text.strip():
        st.error("SYSTEM HALT: Text extraction failed. Ensure the document is not an image-based PDF wrapper or empty file.")
        st.stop()
        
    cleaned_resume = clean_resume_text(raw_text)
    detected_skills = extract_registered_skills(raw_text)
    word_count = len(raw_text.split())
    
    audit_results, ats_score = evaluate_ats_readiness(raw_text, word_count, len(detected_skills))
    strength_score = calculate_resume_strength(word_count, len(detected_skills), ats_score, raw_text)
    
    X_input = engine_vectorizer.transform([cleaned_resume])
    ml_predicted_role = engine_model.predict(X_input)[0]
    
    final_primary_role, role_suggestions = hybrid_role_engine(raw_text.lower(), ml_predicted_role)
    
    # JD Match Engine Check
    jd_provided = bool(job_description.strip())
    match_percentage = 0
    jd_skills_raw = missing_skills = bonus_skills = core_skills = expanded_jd_skills = matched_jd_skills = []
    coverage_perc = 0
    
    core_mapping = set([k for val in ROLE_MAPPINGS.values() for k in val])

    if jd_provided:
        cleaned_jd = clean_resume_text(job_description)
        jd_skills_raw = extract_registered_skills(job_description)
        
        match_percentage, verdict_label, verdict_color, role_align_text, expanded_jd_skills, matched_jd_skills, missing_skills, coverage_perc = hybrid_jd_match_engine(
            cleaned_resume, cleaned_jd, detected_skills, jd_skills_raw, final_primary_role
        )
        
        core_skills = matched_jd_skills
        bonus_skills = [s for s in detected_skills if s not in expanded_jd_skills]
        
        # FEATURE IMPROVEMENT: RE-RANK ROLE SUGGESTIONS BASED ON JD
        jd_lower_clean = job_description.lower()
        adjusted_suggestions = []
        for r_title, r_score, r_reason in role_suggestions:
            boost = 0
            if r_title.lower().replace("developer", "").replace("engineer", "").strip() in jd_lower_clean:
                boost += 25
            r_kws = ROLE_MAPPINGS.get(r_title, [])
            hits = sum(1 for kw in r_kws if kw in jd_lower_clean)
            boost += (hits * 5)
            new_score = min(100, r_score + boost)
            adjusted_suggestions.append((r_title, new_score, r_reason))
        
        # Sort by updated JD-influenced rank
        role_suggestions = sorted(adjusted_suggestions, key=lambda x: x[1], reverse=True)[:4]

    else:
        core_skills = [s for s in detected_skills if s in core_mapping]
        bonus_skills = [s for s in detected_skills if s not in core_mapping]
        
    insights = compile_recruiter_insights(audit_results, core_skills, word_count, match_percentage if jd_provided else None)
    summary_text = generate_candidate_summary(final_primary_role, strength_score, core_skills, ats_score)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP TIER: Intelligence Stat Cards
    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value metric-glow-cyan">{strength_score}%</div>
            <div class="metric-label">Resume Strength</div>
        </div>
        <div class="metric-box" style="border-top: 2px solid #8B5CF6;">
            <div class="metric-value metric-glow-violet">{ats_score}%</div>
            <div class="metric-label">ATS Readiness</div>
        </div>
        <div class="metric-box" style="border-top: 2px solid #D946EF;">
            <div class="metric-value" style="font-size: 1.6rem; padding-top: 15px; color: #D946EF; text-shadow: 0 0 20px rgba(217, 70, 239, 0.4);">{final_primary_role}</div>
            <div class="metric-label" style="margin-top: 5px;">Best-Fit Category</div>
        </div>
        <div class="metric-box" style="border-top: 2px solid #10B981;">
            <div class="metric-value metric-glow-green">{match_percentage if jd_provided else '--'}%</div>
            <div class="metric-label">JD Match Engine</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div><br>", unsafe_allow_html=True)

    # MID TIER: Summary & AI Role Engine
    col_ci1, col_ci2 = st.columns([1.5, 1], gap="large")
    
    with col_ci1:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>CANDIDATE INTELLIGENCE SUMMARY</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-text'>\"{summary_text}\"</div>", unsafe_allow_html=True)
        
        st.markdown("<br><div style='color: #94A3B8; font-weight: 800; margin-bottom: 20px; font-size: 0.85rem; letter-spacing: 1.5px;'>RECRUITER INSIGHTS</div>", unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f"<div class='insight-bullet'>{ins}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_ci2:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        if jd_provided:
            st.markdown("<div class='card-title'>ROLE SUGGESTIONS (JD CALIBRATED)</div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #CBD5E1; font-size: 0.9rem; margin-bottom: 20px;'>Dynamically shifted recommendations aligning Resume telemetry with provided Job targets:</p>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-title'>TOP ROLE RECOMMENDATIONS</div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #CBD5E1; font-size: 0.9rem; margin-bottom: 20px;'>Evaluated algorithmic confidence correlating to specific engineering domains:</p>", unsafe_allow_html=True)
            
        for role_title, conf_score, role_reason in role_suggestions:
            st.markdown(f"""
            <div class="role-card">
                <div>
                    <div class="role-header">{role_title}</div>
                    <div class="role-reason">{role_reason}</div>
                </div>
                <div class="role-score">{conf_score}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # LOWER TIER: Skill Matrices & Audits & Match Score
    col_low1, col_low2 = st.columns([1.1, 1], gap="large")

    with col_low1:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>RESUME SIGNAL MATRIX</div>", unsafe_allow_html=True)
        
        if jd_provided:
            st.markdown("<span style='color: #94A3B8; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;'>CORE SKILLS (REQUIRED BY JOB TARGET)</span>", unsafe_allow_html=True)
            if core_skills:
                ch = "<div class='badge-wrap'>" + "".join([f"<span class='badge badge-core'>{s}</span>" for s in core_skills]) + "</div>"
                st.markdown(ch, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #EF4444; font-size: 0.85rem;'>0 Matches Found.</p>", unsafe_allow_html=True)
                
            st.markdown("<br><span style='color: #94A3B8; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;'>BONUS / ADDITIONAL TECH DETECTED</span>", unsafe_allow_html=True)
            if bonus_skills:
                bh = "<div class='badge-wrap'>" + "".join([f"<span class='badge badge-bonus'>{s}</span>" for s in bonus_skills]) + "</div>"
                st.markdown(bh, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748B;'>None</p>", unsafe_allow_html=True)

            st.markdown("<br><span style='color: #EF4444; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;'>DEFICIENT SKILLS DETECTED (MISSING)</span>", unsafe_allow_html=True)
            if missing_skills:
                mh = "<div class='badge-wrap'>" + "".join([f"<span class='badge badge-missing'>{s}</span>" for s in missing_skills]) + "</div>"
                st.markdown(mh, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #10B981; font-weight: bold;'>✓ No specific keyword deficiencies identified against JD.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #94A3B8; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;'>GLOBAL CORE SKILLS RECOGNIZED</span>", unsafe_allow_html=True)
            if core_skills:
                ch = "<div class='badge-wrap'>" + "".join([f"<span class='badge badge-core'>{s}</span>" for s in core_skills]) + "</div>"
                st.markdown(ch, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748B;'>None</p>", unsafe_allow_html=True)
                
            st.markdown("<br><span style='color: #94A3B8; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;'>SUPPORTING TECHNICAL SKILLS</span>", unsafe_allow_html=True)
            if bonus_skills:
                bh = "<div class='badge-wrap'>" + "".join([f"<span class='badge badge-support'>{s}</span>" for s in bonus_skills]) + "</div>"
                st.markdown(bh, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748B;'>None</p>", unsafe_allow_html=True)
                
        with st.expander("🔍 EXPAND RAW RESUME PAYLOAD"):
            st.text_area("Extraction Stream", raw_text, height=200, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_low2:
        st.markdown("<div class='glass-card' style='height: 100%; border-top: 3px solid #10B981;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>MATCH INTELLIGENCE DASHBOARD</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 25px;'>", unsafe_allow_html=True)
        
        if jd_provided:
            st.markdown(f"""
            <div style="display:flex; justify-content:center;">
                <div class="match-ring" style="border: 2px solid {verdict_color}; box-shadow: 0 0 30px {verdict_color}30;">
                    <div class="metric-label" style="margin-bottom: 15px;">Overall Match</div>
                    <div class="match-ring-value" style="color: {verdict_color}; text-shadow: 0 0 20px {verdict_color}50;">{match_percentage}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Professional Recruiter Match Banner
            st.markdown(f"<div style='text-align:center; padding: 15px; border-radius: 8px; background: {verdict_color}15; color: {verdict_color}; font-weight: 900; font-size: 1.4rem; border: 1px solid {verdict_color}; margin-bottom: 20px;'>{verdict_label.upper()}</div>", unsafe_allow_html=True)
            
            # Additional logic output for poor matches
            if match_percentage <= 30:
                st.markdown(f"<div style='padding:12px; background:rgba(239, 68, 68, 0.05); border-left: 3px solid #EF4444; color:#FCA5A5; font-size:0.85rem; line-height:1.4;'><b>ALIGNMENT ALERT:</b> Minimal overlap detected between candidate profile and job requirements. The resume fundamentally lacks the expected technical footprint.</div>", unsafe_allow_html=True)

            # Advanced Match Component Breakdown Section
            st.markdown("<div style='margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px dashed rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#CBD5E1; font-weight: bold; font-size:0.95rem; margin-bottom:15px; letter-spacing: 1px;'>SCORE COMPONENT ANALYSIS</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='jd-component-row'><span style='color:#94A3B8;'>Expected Skill Clusters</span><span style='color:#CBD5E1; font-weight:bold;'>{len(expanded_jd_skills)} Evaluated</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='jd-component-row'><span style='color:#94A3B8;'>Role Semantic Alignment</span><span style='color:#00E5FF; font-weight:bold;'>{role_align_text}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='jd-component-row'><span style='color:#94A3B8;'>Successfully Matched Skills</span><span style='color:#10B981; font-weight:bold;'>{len(core_skills)}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='jd-component-row'><span style='color:#94A3B8;'>Identified Skill Gaps</span><span style='color:#EF4444; font-weight:bold;'>{len(missing_skills)}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='jd-component-row'><span style='color:#94A3B8;'>JD Skill Coverage Vector</span><span style='color:#FBBF24; font-weight:bold;'>{coverage_perc}%</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style='padding: 40px 20px; text-align: center; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;'>
                <h4 style="color: #64748B; margin-bottom: 10px;">MATCH ENGINE STANDBY</h4>
                <p style="color: #475569; font-size: 0.9rem;">Input Job Description to unlock visual alignment telemetry.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ATS Check list logically shifted to feel real
        st.markdown("<div style='color: #F8FAFC; font-weight: bold; margin-top: 30px; margin-bottom: 15px; font-size: 0.95rem; text-transform: uppercase;'>ATS STRUCTURAL SCANNER</div>", unsafe_allow_html=True)
        st.markdown("<ul class='audit-list'>", unsafe_allow_html=True)
        for key, passed in audit_results.items():
            status = "<span class='audit-pass'>✓ INCLUDED</span>" if passed else "<span class='audit-fail'>✗ MISSING</span>"
            st.markdown(f"<li class='audit-item'><span style='color: #CBD5E1;'>{key}</span> {status}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 7: Download / Export
    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    report_text = (
        "==============================================\n"
        "      HIRESIGHT AI - CANDIDATE DOSSIER       \n"
        "==============================================\n\n"
        f"Primary Alignment:      {final_primary_role}\n"
        f"Resume Strength Score:  {strength_score}/100\n"
        f"ATS System Readiness:   {ats_score}/100\n"
        f"JD Alignment Vector:    {match_percentage}% \n\n"
        "--- HIRE INTELLIGENCE SUMMARY ---\n"
        f"{summary_text}\n\n"
        "--- TOP ROLE RECOMMENDATIONS ---\n"
        f"{chr(10).join([f'- {r[0]} ({r[1]}%): {r[2]}' for r in role_suggestions])}\n\n"
        "--- BEHAVIORAL & STRUCTURAL INSIGHTS ---\n"
        f"{chr(10).join(['* ' + i for i in insights])}\n\n"
        "==============================================\n"
        "Dossier automatically generated by HireSight AI.\n"
    )

    st.download_button(
        label="📥 EXTRACT SECURE DOSSIER (.TXT)",
        data=report_text,
        file_name="HireSight_Enterprise_Profile.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Global Footer
# -------------------------------------------------------------
st.markdown("""
<div style='text-align: center; color: #334155; font-size: 0.8rem; margin-top: 50px; padding-bottom: 20px;'>
    <b>HIRESIGHT AI // ENTERPRISE DATA PLATFORM v4.0</b><br>
    Powered by Hybrid Skill-Matching Architecture and Scikit-Learn.
</div>
""", unsafe_allow_html=True)