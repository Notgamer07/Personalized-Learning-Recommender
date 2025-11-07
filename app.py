import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import upload_file, list_repo_files, HfApi, hf_hub_download
import os
from io import StringIO

# --- 1. CONFIGURATION, MODELS, AND DATA LOADING ---

# Streamlit Resource Caching for models (loading only once)
@st.cache_resource
def load_models():
    """Loads the pre-trained ML model and encoders."""
    try:
        pipeline = joblib.load("recommendation_model.pkl")
        le_subtopic = joblib.load("subtopic_encoder.pkl")
        le_course = joblib.load("course_encoder.pkl")
        return pipeline, le_subtopic, le_course
    except FileNotFoundError:
        st.error("Error: Required model files (pkl) not found. Please ensure they are uploaded or exist in the execution directory.")
        st.stop()

# Load models and encoders
knn_pipeline, le_subtopic, le_course = load_models()

# Data from gui.py
TOTAL_NUM_QUESTIONS = {
    "Algebra": 4, "Calculus": 4, "Coordinate geometry": 2,
    "Networking": 4, "Cybersecurity": 4, "Other_tech": 2,
    "Physics": 4, "Chemistry": 4, "Biology": 2, "Computer science": 10
}

# Advanced/Harder 12th Grade MCQ Questions (Ensures question count matches TOTAL_NUM_QUESTIONS)
MCQ_QUESTIONS = {
    "Algebra": [
        ["If a non-singular matrix A is idempotent (A² = A), what is the inverse of A?", ["A", "I - A", "I", "A²"], 2],
        ["For a sequence to be convergent, which condition must it satisfy?", ["Monotonic and unbounded", "Cauchy but not bounded", "Bounded and monotonic", "Unbounded and non-monotonic"], 2],
        ["If roots of the equation x² - bx + c = 0 are two consecutive integers, then b² - 4c is equal to:", ["4", "3", "2", "1"], 3],
        ["The greatest integer function f(x) = [x] is continuous at:", ["Integers only", "Non-integers only", "All real numbers", "Nowhere"], 1],
    ],
    "Calculus": [
        ["What is the value of the definite integral $\int_{0}^{\pi} \cos^2(x) dx$?", ["$\pi$", "$\pi/2$", "0", "$2\pi$"], 1],
        ["If $y = e^{\sin(x)}$, what is $dy/dx$?", ["$e^{\cos(x)}$", "$\cos(x) e^{\sin(x)}$", "$e^{\sin(x)} / \cos(x)$", "$\sin(x) e^{\sin(x)}$"], 1],
        ["For which function does the Mean Value Theorem not apply on the interval [0, 4]?", ["$f(x) = |x-2|$", "$f(x) = x^2$", "$f(x) = \sin(x)$", "$f(x) = e^x$"], 0],
        ["Which test determines local extrema for a single-variable function?", ["Integral Test", "Ratio Test", "First Derivative Test", "Comparison Test"], 2],
    ],
    "Coordinate geometry": [
        ["The general equation of a sphere with center (h, k, l) and radius r is given by:", ["$(x-h)^2 + (y-k)^2 = r^2$", "$(x-h)^2 + (y-k)^2 + (z-l)^2 = r$", "$x^2 + y^2 + z^2 + 2gx + 2fy + 2hz + c = 0$", "$x^2 + y^2 + 2gx + 2fy + c = 0$"], 2],
        ["What is the eccentricity of a rectangular hyperbola?", ["$1$", "$\sqrt{2}$", "$1/\sqrt{2}$", "$\infty$"], 1],
    ],
    "Networking": [
        ["Which TCP/IP model layer combines the OSI's Presentation and Session layers?", ["Application Layer", "Transport Layer", "Network Layer", "Internet Layer"], 0],
        ["What is the primary function of the ARP protocol?", ["Mapping IP to TCP port numbers", "Mapping MAC to UDP port numbers", "Mapping IP to MAC addresses", "Mapping hostnames to IP addresses"], 2],
        ["The term 'three-way handshake' is associated with:", ["UDP connection establishment", "TCP connection termination", "ARP request/reply", "TCP connection establishment"], 3],
        ["In CIDR notation, what does the suffix /24 represent in an IPv4 address?", ["A Class C network", "The first 24 bits are the network ID", "The last 24 bits are the host ID", "24 subnets"], 1],
    ],
    "Cybersecurity": [
        ["A zero-day exploit targets:", ["A patchable vulnerability known for years", "A newly discovered vulnerability with no patch available", "Software running with elevated privileges", "Misconfigured database access"], 1],
        ["What mathematical function underpins public-key cryptography like RSA?", ["Matrix Inversion", "One-way Hash Function", "Prime Factorization Difficulty", "Discrete Fourier Transform"], 2],
        ["A 'Man-in-the-Middle' (MITM) attack relies primarily on:", ["DDoS to overload the server", "Eavesdropping on a secure channel", "Impersonating one party to both communicating parties", "Exploiting a buffer overflow"], 2],
        ["What is the term for a logical grouping of users or resources that share security policies within an Active Directory environment?", ["Domain", "Workgroup", "Forest", "Realm"], 0],
    ],
    "Other_tech": [
        ["Which technology uses a decentralized, distributed ledger that is immutable?", ["Cloud Computing", "NoSQL Database", "Blockchain", "Virtualization"], 2],
        ["What is the fundamental difference between a virtual machine (VM) and a container (e.g., Docker)?", ["VMs use kernel-level isolation; containers use hardware emulation", "VMs emulate hardware; containers share the host OS kernel", "VMs are platform-independent; containers are not", "VMs are significantly faster than containers"], 1],
    ],
    "Physics": [
        ["In an ideal transformer, the voltage ratio is proportional to the:", ["Current ratio", "Power ratio", "Number of turns ratio", "Frequency ratio"], 2],
        ["According to de Broglie, the wavelength $\lambda$ associated with a particle of momentum $p$ is:", ["$\lambda = p/h$", "$\lambda = h/p$", "$\lambda = h \cdot p$", "$\lambda = h^2/p$"], 1],
        ["What does the term 'critical angle' relate to in optics?", ["Reflection", "Dispersion", "Total Internal Reflection", "Polarization"], 2],
        ["The property of a semiconductor that determines its conductivity at absolute zero (0 K) is its:", ["Doping concentration", "Band gap energy", "Hole mobility", "Effective mass"], 1],
    ],
    "Chemistry": [
        ["What is the oxidation state of sulfur in sulfuric acid ($H_2SO_4$)?", ["+2", "+4", "+6", "+7"], 2],
        ["The process of converting vegetable oil into solid fat is an example of which type of chemical reaction?", ["Hydrolysis", "Saponification", "Hydrogenation", "Esterification"], 2],
        ["Which factor increases the rate of reaction by lowering the activation energy without being consumed?", ["Inhibitor", "Catalyst", "Reagent", "Solvent"], 1],
        ["According to VSEPR theory, a molecule with four bonding pairs and zero lone pairs around the central atom has what geometry?", ["Trigonal planar", "Tetrahedral", "Bent", "Trigonal bipyramidal"], 1],
    ],
    "Biology": [
        ["Which enzyme is responsible for synthesizing a new strand of DNA during replication in the 5' to 3' direction?", ["DNA Ligase", "Helicase", "DNA Polymerase III", "Primase"], 2],
        ["In eukaryotic cells, where does the Krebs cycle (Citric Acid Cycle) occur?", ["Cytosol", "Mitochondrial matrix", "Ribosome", "Nucleus"], 1],
    ],
    "Computer science": [
        ["The complexity $O(n^2)$ algorithm is generally preferred over $O(n \log n)$ for large datasets. (True/False)", ["True", "False", "Only if $n > 1000$", "Depends on constant factors"], 1],
        ["In SQL, which keyword is used to sort the result-set of a query?", ["SORT BY", "ARRANGE", "ORDER BY", "GROUP BY"], 2],
        ["What is the purpose of normalization in database design?", ["To speed up query execution", "To reduce data redundancy and dependency issues", "To encrypt sensitive data", "To manage user permissions"], 1],
        ["Which search algorithm requires the graph edges to have non-negative weights?", ["Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Dijkstra's Algorithm", "A* Search"], 2],
        ["A 'race condition' is characteristic of which programming paradigm?", ["Functional programming", "Concurrent programming", "Procedural programming", "Object-Oriented programming"], 1],
        ["What is the primary role of a Hypervisor (VMM)?", ["Managing network traffic flow", "Creating and running virtual machines", "Providing secure browser environment", "Compiling high-level code"], 1],
        ["Which of the following is a non-volatile memory type?", ["SRAM", "DRAM", "Cache memory", "Flash memory"], 3],
        ["What data structure is used to implement a recursive function call stack?", ["Queue", "Hash Table", "Binary Tree", "Stack"], 3],
        ["Which of the following protocols operates at the Transport Layer of the OSI model, guaranteeing reliable, ordered data delivery?", ["IP", "HTTP", "TCP", "UDP"], 2],
        ["The technique used to handle multiple interrupts by assigning priority levels is called:", ["Polling", "Vectored interrupt", "Interrupt chaining", "Daisy chaining"], 3],
    ]
}

# --- 2. RECOMMENDATION LOGIC ---

def recommend_courses_batch(subtopics, num_questions, correct_answers):
    """
    Predicts and returns course recommendations based on user performance.
    Matches the original logic from gui.py.
    """
    # Use correct_answers as the 'Score out of 10' feature, matching the original script.
    scores = correct_answers
    
    data = pd.DataFrame({
        'Sub topics': subtopics,
        'Num of questions': num_questions,
        'Correct answers': correct_answers,
        'Score out of 10': scores,
    })
    
    # Feature Engineering: Calculate Accuracy
    data['Accuracy'] = data['Correct answers'] / data['Num of questions']
    
    # Encoding: Transform categorical feature 'Sub topics'
    # Check if a subtopic wasn't in the original training data (which shouldn't happen here)
    for topic in data['Sub topics'].unique():
        if topic not in le_subtopic.classes_:
            st.warning(f"Subtopic '{topic}' not recognized by the model encoder. Skipping recommendation.")
            return ["Error: Subtopic not found in model vocabulary."]
    
    data['Sub topics'] = le_subtopic.transform(data['Sub topics'])
    
    # [cite_start]Ensure feature order matches the model training data [cite: 5]
    X_predict = data[['Sub topics', 'Num of questions', 'Correct answers', 'Score out of 10', 'Accuracy']]
    
    # Predict courses
    try:
        preds = knn_pipeline.predict(X_predict)
        courses = le_course.inverse_transform(preds)
        
        # Return unique courses
        return list(set(courses))
    except Exception as e:
        st.error(f"Error during model prediction: {e}. Check model integrity.")
        return ["Error: Model failed to generate recommendation."]


# --- 3. STREAMLIT APP STATE MANAGEMENT ---

# Initialize session state for navigation and data storage
if 'page' not in st.session_state:
    st.session_state['page'] = 'start'
if 'selected_topics' not in st.session_state:
    st.session_state['selected_topics'] = []
if 'mcq_results' not in st.session_state:
    st.session_state['mcq_results'] = {}

def set_page(page_name):
    """Function to change the page/state."""
    st.session_state['page'] = page_name

# --- 4. STREAMLIT PAGE FUNCTIONS ---

def start_screen():
    """Step 1: Welcome Screen"""
    st.markdown("<h1 style='text-align: center; color: #0078D7;'>🎓 Personalised Course Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.image(
        "https://images.unsplash.com/photo-1541807084534-c3199b0c2e6f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg"
        "&ixid=M3wzOTU3MzUzfDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&ixlib=rb-4.0.3&q=80&w=400", 
        width=400, 
        caption="Knowledge is power", 
        use_column_width='auto'
    )

    st.markdown("""
        Welcome to the Course Recommendation System.
        
        [cite_start]This system uses a machine learning model (K-Nearest Neighbors) [cite: 8] 
        to predict suitable courses based on your performance in core academic and technical subtopics.
        
        **Let's test your knowledge!**
    """)
    
    # --- New Section: User Info ---
    st.markdown("### 🧾 Enter Your Details")
    name = st.text_input("Full Name", key="user_name")
    email = st.text_input("Email ID", key="user_email")

    # Basic validation feedback
    if name.strip() == "":
        st.warning("Please enter your full name to continue.")
    elif email.strip() == "":
        st.warning("Please enter your email address to continue.")

    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Quiz", use_container_width=True, type="primary"):
            if not st.session_state.user_name.strip() or not st.session_state.user_email.strip():
                st.error("⚠️ Please fill in both your **Name** and **Email ID** before starting the quiz.")
            else:
                # Reset other states
                st.session_state['selected_topics'] = []
                st.session_state['mcq_results'] = {}
                st.session_state['user_name_value'] = name
                st.session_state['user_email_value'] = email
                set_page('select_subtopic')


def select_subtopic_screen():
    """Step 2: Subtopic Selection"""
    st.markdown("<h2 style='text-align: center; color: #0078D7;'>Select Subtopics for Assessment</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.write("Please select the subtopics you wish to be tested on. The quiz questions are of an advanced difficulty level (approx. 12th grade/Entry-level).")
    
    # Use multiselect widget for compact selection
    all_topics = list(TOTAL_NUM_QUESTIONS.keys())
    selected_topics = st.multiselect(
        "Choose Topics:",
        options=all_topics,
        default=[]
    )
    
    # Display selected topics and question count
    if selected_topics:
        st.subheader("Selected Topics:")
        for topic in selected_topics:
            st.markdown(f"• **{topic}**: {TOTAL_NUM_QUESTIONS[topic]} Questions")

    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Proceed to Quiz", use_container_width=True, type="primary"):
            if not selected_topics:
                st.warning("Please select at least one subtopic to continue.")
            else:
                st.session_state['selected_topics'] = selected_topics
                set_page('input_scores')
        
        if st.button("Back to Start", use_container_width=True):
             set_page('start')


def mcq_quiz_screen():
    """Step 3: MCQ Quiz and Scoring"""
    st.markdown("<h2 style='text-align: center; color: #0078D7;'>Take the Quiz</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_topics = st.session_state['selected_topics']
    results = {}

    # Use a form to manage state and submit all answers together
    with st.form(key='mcq_form'):
        
        for topic in selected_topics:
            st.subheader(f"📚 Topic: {topic} ({TOTAL_NUM_QUESTIONS[topic]} Questions)")
            questions = MCQ_QUESTIONS.get(topic, [])
            
            with st.container(border=True):
                for i, (question, options, correct_idx) in enumerate(questions):
                    key = f"{topic}_{i}"
                    
                    # Radio button for MCQ selection
                    user_choice = st.radio(
                        f"**Q{i+1}.** {question}",
                        options,
                        key=key,
                        index=None,  # Start with no option selected
                        horizontal=False
                    )
                    
                    # Store the actual correct option text for later validation
                    correct_option_text = options[correct_idx]
                    results[key] = (user_choice, correct_option_text)
                    st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_quiz = st.form_submit_button("Submit Answers & Get Recommendations", type="primary", use_container_width=True)

    if submit_quiz:
        
        # --- SCORING LOGIC ---
        topic_scores = {}
        
        # Validate that all questions were attempted
        not_answered_count = 0
        for topic in selected_topics:
            for i in range(TOTAL_NUM_QUESTIONS[topic]):
                 key = f"{topic}_{i}"
                 if results[key][0] is None:
                     not_answered_count += 1
                     
        if not_answered_count > 0:
            st.error(f"Please answer all {TOTAL_NUM_QUESTIONS[topic]} questions to submit the quiz. {not_answered_count} question(s) remain unanswered.")
            return

        # Calculate score for each topic
        for topic in selected_topics:
            num_q = TOTAL_NUM_QUESTIONS[topic]
            correct_count = 0
            
            for i in range(num_q):
                key = f"{topic}_{i}"
                user_selection, actual_answer = results[key]
                
                if user_selection == actual_answer:
                    correct_count += 1
            
            topic_scores[topic] = correct_count
        
        # Prepare data for the recommendation model
        subtopics_list = list(topic_scores.keys())
        correct_answers_list = list(topic_scores.values())
        num_questions_list = [TOTAL_NUM_QUESTIONS[t] for t in subtopics_list]
        
        st.session_state['mcq_results'] = {
            'subtopics': subtopics_list,
            'corrects': correct_answers_list,
            'num_questions': num_questions_list
        }
        
        # Move to results page
        set_page('show_results')

    st.markdown("---")
    if st.button("Go Back to Topic Selection", use_container_width=False):
        set_page('select_subtopic')


from datetime import datetime

def result_screen():
    st.markdown("<h2 style='text-align: center; color: #00ff7f;'>Recommended Courses 🚀</h2>", unsafe_allow_html=True)
    st.markdown("---")
    data = st.session_state.get('mcq_results', None)
    if not data or not data['subtopics']:
        st.error("No quiz data found.")
        if st.button("Go Back"):
            set_page('start')
        return

    st.subheader("Your Performance Analysis")
    results_df = pd.DataFrame({
        'Subtopic': data['subtopics'],
        'Correct Answers': data['corrects'],
        'Total Questions': data['num_questions'],
    })
    results_df['Accuracy (%)'] = (results_df['Correct Answers'] / results_df['Total Questions'] * 100).round(1)
    total_correct = results_df['Correct Answers'].sum()
    total_questions = results_df['Total Questions'].sum()
    overall_accuracy = (total_correct / total_questions * 100).round(1)
    st.metric(label="Overall Accuracy", value=f"{overall_accuracy}%", delta=f"{total_correct}/{total_questions}")
    st.dataframe(results_df.set_index('Subtopic'), use_container_width=True)
    st.markdown("---")

    st.subheader("Course Recommendations")
    with st.spinner('Analyzing performance...'):
        recommended_courses = recommend_courses_batch(
            data['subtopics'], data['num_questions'], data['corrects']
        )

    if recommended_courses and not recommended_courses[0].startswith("Error"):
        st.success(f"Recommended {len(recommended_courses)} course(s):")
        st.markdown("\n".join([f"• **{c}**" for c in recommended_courses]))
    else:
        st.info("No specific courses recommended.")

    # --- 3. SAVE TO HUGGING FACE DATASET ---
    # --- 3. SAVE TO HUGGING FACE DATASET ---
    st.markdown("---")
    user_name = st.session_state.get('user_name_value', st.session_state.get('user_name', 'Unknown User'))
    user_email = st.session_state.get('user_email_value', st.session_state.get('user_email', 'No Email'))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_rows = []
    for subtopic, correct, total, course in zip(
        data['subtopics'], data['corrects'], data['num_questions'], recommended_courses
    ):
        accuracy = round((correct / total) * 100, 1)
        save_rows.append({
            "Timestamp": timestamp,
            "User Name": user_name,
            "Email ID": user_email,
            "Subtopic": subtopic,
            "Correct Answers": correct,
            "Total Questions": total,
            "Accuracy (%)": accuracy,
            "Courses Recommended": course
        })

    save_df = pd.DataFrame(save_rows)

    # Hugging Face Dataset Repo Info
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "jagrath01/Recommender-System-Dataset"
    dataset_file = "user_quiz_results.csv"
    
    if hf_token:
        try:
            api = HfApi()
            files = [f.split("/")[-1] for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)]
    
            if dataset_file in files:
                print("Found existing files")
                try:
                    existing_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=dataset_file,
                        repo_type="dataset",
                        token=hf_token
                    )
                    existing_df = pd.read_csv(existing_path)
                except Exception as e:
                    print("Could not read existing files")
                    existing_df = pd.DataFrame()
            else:
                print("Could not find existing files")
                existing_df = pd.DataFrame()
    
            # Combine and upload
            combined_df = pd.concat([existing_df, save_df], ignore_index=True)
    
            csv_buffer = StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            upload_file(
                path_or_fileobj=csv_buffer.getvalue().encode("utf-8"),
                path_in_repo=dataset_file,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )
            print("results updates")
        except Exception as e:
            print("Results not updated")
    else:
        st.warning("⚠️ HF_TOKEN not found. Results not saved persistently.")
    
        # --- 4. Navigation & Feedback ---
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Restart Quiz", type="primary", use_container_width=True):
                set_page('start')
        with col2:
            if st.button("Change Topics", use_container_width=True):
                set_page('select_subtopic')
        with col3:
            form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfbyB86J1zF99yq78LUvt7B8-GzVYMUM7hU1cTcmnqT5hQACw/viewform?usp=publish-editor"
            st.markdown(
                f"<a href='{form_url}' target='_blank'><button style='background-color:#00b050; color:white; border:none; border-radius:8px; padding:10px 20px; cursor:pointer;'>📝 Give Feedback</button></a>",
                unsafe_allow_html=True
            )
    # --- 4. Navigation ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Restart Quiz", type="primary", use_container_width=True):
            set_page('start')
    with col2:
        if st.button("Change Topics", use_container_width=True):
            set_page('select_subtopic')
    with col3:
        # 🔗 Redirect to Google Form
        form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfbyB86J1zF99yq78LUvt7B8-GzVYMUM7hU1cTcmnqT5hQACw/viewform?usp=publish-editor"
        st.markdown(
            f"<a href='{form_url}' target='_blank'><button style='background-color:#00b050; color:white; border:none; border-radius:8px; padding:10px 20px; cursor:pointer;'>📝 Give Feedback</button></a>",
            unsafe_allow_html=True
        )

# --- 5. MAIN APP EXECUTION ---

# Apply Streamlit Theme
st.set_page_config(layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* Mimic dark background */
    .stApp {
        background-color: #1e1e1e; 
        color: white;
    }
    /* Style headers (matching Tkinter blue/green theme) */
    h1 { color: #0078D7; }
    h2 { color: #0078D7; }
    h3 { color: #00ff7f; } /* Result screen accent color */
    .stMarkdown h2 { color: #0078D7; }
    .stMarkdown h3 { color: #00ff7f; }
    
    /* Input/Checkbox Text Color */
    label, .st-b5 { 
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Routing logic
if st.session_state['page'] == 'start':
    start_screen()
    st.markdown("---")
    st.caption("Application built for Course Recommendation using a pre-trained KNN Pipeline.")
elif st.session_state['page'] == 'select_subtopic':
    select_subtopic_screen()
elif st.session_state['page'] == 'input_scores':
    mcq_quiz_screen()
elif st.session_state['page'] == 'show_results':
    result_screen()