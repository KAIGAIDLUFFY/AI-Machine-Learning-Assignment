import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rapidfuzz import fuzz
import hashlib
from datetime import datetime

# -------------------------------
#        ENHANCED DATASET
# -------------------------------

# Significantly expanded dataset for better accuracy
data = {
    "drug_name": [
        # Authentic drugs (35 samples)
        "Panadol", "Paracetamol", "Aspirin", "Augmentin", "Amoxil", "Ibuprofen",
        "Cetirizine", "Flagyl", "Metformin", "Amlodipine", "Omeprazole", "Ciprofloxacin",
        "Amoxicillin", "Doxycycline", "Azithromycin", "Diclofenac", "Loratadine",
        "Atorvastatin", "Losartan", "Levothyroxine", "Metoprolol", "Lisinopril",
        "Simvastatin", "Clopidogrel", "Montelukast", "Pantoprazole", "Prednisone",
        "Albuterol", "Gabapentin", "Hydrochlorothiazide", "Warfarin", "Insulin",
        "Tramadol", "Codeine", "Morphine",
        # Suspicious drugs (20 samples)
        "UnknownPain", "GenericX", "QuickCure", "MiraclePill", "FastHeal",
        "ChemPlus", "MediGeneric", "PharmaSol", "HealthFix", "CureAll",
        "DrugPlus", "MediQuick", "FastRelief", "GenericCure", "QuickMed",
        "PainAway", "SuperCure", "MegaHealth", "PowerMed", "InstantFix",
        # Counterfeit drugs (20 samples)
        "FakeMedX", "SuspiciousDrug", "FakePain", "FakeCough", "CounterfeitAspro",
        "FakeAugment", "BadMedicine", "FraudPharma", "CounterfeitMed", "FakeGeneric",
        "ShadyPills", "FraudCure", "FakeRelief", "BadPharma", "CounterfeitX",
        "FakePanadol", "FakeIbuprofen", "BadAspirin", "FraudMeds", "CounterfeitDrug"
    ],
    "manufacturer": [
        # Authentic manufacturers
        "GSK", "Beta Healthcare", "Bayer", "GSK", "Beecham", "Pfizer",
        "Cipla", "Sanofi", "Merck", "Pfizer", "AstraZeneca", "Bayer",
        "GSK", "Pfizer", "Pfizer", "Novartis", "Merck",
        "Pfizer", "Merck", "AbbVie", "AstraZeneca", "Novartis",
        "Merck", "Sanofi", "Merck", "Takeda", "Pfizer",
        "GSK", "Pfizer", "Merck", "Bristol-Myers", "Novo Nordisk",
        "Janssen", "GSK", "Pfizer",
        # Suspicious manufacturers
        "Unknown Labs", "Generic Inc", "Quick Pharma", "Miracle Co", "Fast Labs",
        "Chem Plus Ltd", "Medi Generic", "Pharma Solutions", "Health Fix Inc", "Cure All Co",
        "Drug Plus", "Medi Quick", "Fast Relief Ltd", "Generic Cure", "Quick Meds",
        "Pain Solutions", "Super Pharma", "Mega Health Co", "Power Meds", "Instant Cure",
        # Counterfeit manufacturers
        "Unknown", "Unknown Inc", "ShadyPharma", "ShadyCure", "Fake Co",
        "Bad Labs", "Fraud Meds", "Counterfeit Inc", "Fake Pharma", "Generic Fake",
        "Shady Pills Ltd", "Fraud Cure Co", "Fake Relief", "Bad Pharma Inc", "Counterfeit X",
        "Fake GSK", "Fake Pfizer", "Bad Bayer", "Fraud Labs", "Counterfeit Co"
    ],
    "authentic": [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 35 Authentic
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 20 Suspicious
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1  # 20 Counterfeit
    ]
}

df = pd.DataFrame(data)
df["combined"] = df["drug_name"] + " " + df["manufacturer"]

# -------------------------------
#      ENHANCED ML MODEL SETUP
# -------------------------------

@st.cache_resource
def train_models():
    """Train both ML models with improved parameters"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Increased to trigrams
        max_features=150,
        min_df=1
    )
    X = vectorizer.fit_transform(df["combined"])
    y = df["authentic"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # KNN Model with optimized parameters
    knn_model = KNeighborsClassifier(
        n_neighbors=3,  # Reduced for smaller dataset
        weights='distance',
        metric='cosine'
    )
    knn_model.fit(X_train, y_train)
    
    # Random Forest Model with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate both models
    knn_preds = knn_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    knn_acc = accuracy_score(y_test, knn_preds)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    knn_f1 = f1_score(y_test, knn_preds, average="macro")
    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    
    # Cross-validation scores
    knn_cv = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy').mean()
    rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy').mean()
    
    metrics = {
        'knn': {'accuracy': knn_acc, 'f1': knn_f1, 'cv_score': knn_cv},
        'rf': {'accuracy': rf_acc, 'f1': rf_f1, 'cv_score': rf_cv}
    }
    
    return vectorizer, knn_model, rf_model, metrics

vectorizer, knn_model, rf_model, metrics = train_models()

# -------------------------------
#      OPENFDA API CHECK
# -------------------------------

@st.cache_data(ttl=3600)
def check_openfda(drug_name):
    """Check OpenFDA database with improved error handling"""
    try:
        # Clean the drug name
        drug_name_clean = drug_name.strip().replace('"', '').replace("'", "")
        url = f"https://api.fda.gov/drug/ndc.json?search=brand_name:{drug_name_clean}&limit=5"
        
        r = requests.get(url, timeout=10)
        
        # Handle different status codes
        if r.status_code == 404:
            return None
        elif r.status_code == 429:
            st.warning("⚠️ API rate limit reached. Please try again later.")
            return None
        elif r.status_code != 200:
            return None
        
        results = r.json().get("results", [])
        if len(results) == 0:
            return None
        return results
        
    except requests.exceptions.Timeout:
        st.warning("⚠️ OpenFDA API timeout. Connection slow.")
        return None
    except requests.exceptions.RequestException:
        st.warning("⚠️ Could not connect to OpenFDA. Check your internet.")
        return None
    except Exception as e:
        st.warning(f"⚠️ OpenFDA API error: {str(e)[:50]}")
        return None

# -------------------------------
#      ENHANCED KENYA PPB SIMULATION
# -------------------------------

ppb_data = {
    "Panadol": {"manufacturer": "GSK", "registration": "PPB/2021/001"},
    "Augmentin": {"manufacturer": "GSK", "registration": "PPB/2020/045"},
    "Flagyl": {"manufacturer": "Sanofi", "registration": "PPB/2019/123"},
    "Cetirizine": {"manufacturer": "Cipla", "registration": "PPB/2021/078"},
    "Aspirin": {"manufacturer": "Bayer", "registration": "PPB/2018/034"},
    "Amoxil": {"manufacturer": "Beecham", "registration": "PPB/2020/089"},
    "Ibuprofen": {"manufacturer": "Pfizer", "registration": "PPB/2021/012"},
    "Metformin": {"manufacturer": "Merck", "registration": "PPB/2020/156"},
    "Amlodipine": {"manufacturer": "Pfizer", "registration": "PPB/2021/098"},
    "Omeprazole": {"manufacturer": "AstraZeneca", "registration": "PPB/2019/234"}
}

def check_ppb(drug_name, manufacturer):
    """Enhanced PPB check with fuzzy matching"""
    best_match = None
    best_score = 0
    
    for name, info in ppb_data.items():
        name_score = fuzz.ratio(drug_name.lower(), name.lower())
        if name_score > 75 and name_score > best_score:
            best_match = (name, info)
            best_score = name_score
    
    if best_match:
        name, info = best_match
        manuf_score = fuzz.ratio(manufacturer.lower(), info["manufacturer"].lower())
        if manuf_score > 75:
            return {"status": "verified", "registration": info["registration"], "confidence": name_score}
        else:
            return {"status": "mismatch", "expected": info["manufacturer"], "confidence": manuf_score}
    
    return {"status": "not_found"}

# -------------------------------
#      BLOCKCHAIN VERIFICATION (SIMULATED)
# -------------------------------

def generate_drug_hash(drug_name, manufacturer, batch_number=""):
    """Simulate blockchain verification using hash"""
    data = f"{drug_name}{manufacturer}{batch_number}".lower().strip()
    return hashlib.sha256(data.encode()).hexdigest()[:16]

# Known authentic hashes (simulated blockchain)
blockchain_registry = {
    generate_drug_hash("Panadol", "GSK"): True,
    generate_drug_hash("Augmentin", "GSK"): True,
    generate_drug_hash("Aspirin", "Bayer"): True,
    generate_drug_hash("Flagyl", "Sanofi"): True,
    generate_drug_hash("Ibuprofen", "Pfizer"): True,
}

# -------------------------------
#      RISK SCORING SYSTEM
# -------------------------------

def calculate_risk_score(ml_pred, ml_confidence, ppb_result, fda_result, blockchain_verified):
    """Calculate comprehensive risk score (0-100)"""
    risk = 50  # Neutral starting point
    
    # ML prediction impact (weighted by confidence)
    confidence_weight = ml_confidence
    
    if ml_pred == 1:  # Authentic
        risk -= (30 * confidence_weight)
    elif ml_pred == 0:  # Suspicious
        risk += (15 * confidence_weight)
    else:  # -1 Counterfeit
        risk += (40 * confidence_weight)
    
    # PPB verification impact
    if ppb_result["status"] == "verified":
        risk -= 25
    elif ppb_result["status"] == "mismatch":
        risk += 30
    elif ppb_result["status"] == "not_found":
        risk += 10
    
    # FDA verification impact
    if fda_result:
        risk -= 20
    else:
        risk += 5
    
    # Blockchain verification impact
    if blockchain_verified:
        risk -= 15
    
    return max(0, min(100, int(risk)))

# -------------------------------
#       STREAMLIT APP UI
# -------------------------------

def main():
    st.set_page_config(page_title="Drug Authenticity Checker", page_icon="💊", layout="wide")
    
    st.title("🏥 Drug Authenticity Checker")
    st.markdown("### Supporting UN SDG 3: Good Health and Well-being")
    
    st.markdown("""
    This tool uses multiple verification methods:
    - 🤖 **Machine Learning** (KNN + Random Forest)
    - 🇰🇪 **Kenya PPB Registry** (simulated)
    - 🌐 **OpenFDA Database**
    - 🔗 **Blockchain Verification** (simulated)
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Drug Information")
        drug = st.text_input("Drug Name:", placeholder="e.g., Panadol")
        manufacturer = st.text_input("Manufacturer:", placeholder="e.g., GSK")
        batch_number = st.text_input("Batch Number (optional):", placeholder="e.g., B12345")
        
        model_choice = st.radio("ML Model:", ["Random Forest (Recommended)", "K-Nearest Neighbors"])
    
    with col2:
        st.subheader("Quick Info")
        st.info("💡 Enter the exact drug name and manufacturer as shown on the package.")
        st.warning("⚠️ This is a demonstration tool. Always consult healthcare professionals.")
    
    if st.button("🔍 Verify Drug", type="primary"):
        if not drug.strip() or not manufacturer.strip():
            st.error("❌ Please enter both drug name and manufacturer.")
            return
        
        # Input validation
        if len(drug) > 100 or len(manufacturer) > 100:
            st.error("❌ Input too long. Please enter valid drug information.")
            return
        
        with st.spinner("🔄 Analyzing drug authenticity..."):
            # ML Prediction
            user_combo = drug.strip() + " " + manufacturer.strip()
            user_vec = vectorizer.transform([user_combo])
            
            if "Random Forest" in model_choice:
                ml_pred = rf_model.predict(user_vec)[0]
                ml_proba = rf_model.predict_proba(user_vec)[0]
                model_name = "Random Forest"
                classes = rf_model.classes_
            else:
                ml_pred = knn_model.predict(user_vec)[0]
                ml_proba = knn_model.predict_proba(user_vec)[0]
                model_name = "KNN"
                classes = knn_model.classes_
            
            # CRITICAL FIX: Map probabilities to correct classes
            class_prob_dict = dict(zip(classes, ml_proba))
            prob_authentic = class_prob_dict.get(1, 0)
            prob_suspicious = class_prob_dict.get(0, 0)
            prob_counterfeit = class_prob_dict.get(-1, 0)
            
            # Get confidence (max probability)
            ml_confidence = max(ml_proba)
            
            # Other verifications
            ppb_result = check_ppb(drug, manufacturer)
            fda_result = check_openfda(drug)
            drug_hash = generate_drug_hash(drug, manufacturer, batch_number)
            blockchain_verified = drug_hash in blockchain_registry
            
            # Calculate risk score
            risk_score = calculate_risk_score(ml_pred, ml_confidence, ppb_result, fda_result, blockchain_verified)
            
            # Display Results
            st.markdown("---")
            st.header("📊 Verification Results")
            
            # Create tabs for organized results
            tab1, tab2, tab3, tab4 = st.tabs(["ML Analysis", "PPB Check", "FDA Check", "Blockchain"])
            
            with tab1:
                st.subheader(f"{model_name} Prediction")
                
                # Display probabilities correctly
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Authentic Probability", f"{prob_authentic*100:.1f}%")
                with col_b:
                    st.metric("Suspicious Probability", f"{prob_suspicious*100:.1f}%")
                with col_c:
                    st.metric("Counterfeit Probability", f"{prob_counterfeit*100:.1f}%")
                
                st.markdown("---")
                
                # Classification result
                if ml_pred == 1:
                    st.success(f"✅ **Classification: Authentic** (Confidence: {ml_confidence*100:.1f}%)")
                elif ml_pred == 0:
                    st.warning(f"⚠️ **Classification: Suspicious** (Confidence: {ml_confidence*100:.1f}%)")
                else:
                    st.error(f"❌ **Classification: Counterfeit** (Confidence: {ml_confidence*100:.1f}%)")
            
            with tab2:
                st.subheader("Kenya PPB Registry")
                if ppb_result["status"] == "verified":
                    st.success(f"✅ **Verified** - Registration: {ppb_result['registration']}")
                    st.write(f"Match confidence: {ppb_result.get('confidence', 'N/A')}%")
                elif ppb_result["status"] == "mismatch":
                    st.warning(f"⚠️ **Manufacturer Mismatch**")
                    st.write(f"Expected: {ppb_result['expected']}")
                    st.write(f"Provided: {manufacturer}")
                    st.error("⚠️ This could indicate a counterfeit product!")
                else:
                    st.info("ℹ️ No PPB record found")
                    st.write("This drug may not be registered in Kenya.")
            
            with tab3:
                st.subheader("OpenFDA Database")
                if fda_result:
                    st.success(f"✅ Found {len(fda_result)} matching record(s)")
                    with st.expander("View FDA Details"):
                        st.json(fda_result[0])
                else:
                    st.warning("⚠️ No matching record in OpenFDA")
                    st.write("The drug might not be marketed in the USA or listed under a different name.")
            
            with tab4:
                st.subheader("Blockchain Verification")
                st.code(f"Hash: {drug_hash}", language="text")
                if blockchain_verified:
                    st.success("✅ Hash verified on blockchain")
                    st.write("This drug's digital signature matches our blockchain registry.")
                else:
                    st.warning("⚠️ Hash not found in blockchain registry")
                    st.write("This drug has not been registered in our blockchain system.")
            
            # Risk Score & Final Verdict
            st.markdown("---")
            st.header("🎯 Final Assessment")
            
            col_x, col_y = st.columns([1, 2])
            
            with col_x:
                # Create color-coded risk display
                if risk_score <= 20:
                    risk_color = "🟢"
                elif risk_score <= 40:
                    risk_color = "🟡"
                elif risk_score <= 60:
                    risk_color = "🟠"
                else:
                    risk_color = "🔴"
                
                st.metric(
                    "Risk Score", 
                    f"{risk_color} {risk_score}/100", 
                    delta=f"{risk_score-50:+d} from baseline",
                    delta_color="inverse"
                )
            
            with col_y:
                if risk_score <= 20:
                    st.success("### ✅ HIGH CONFIDENCE: Authentic Drug")
                    st.write("✓ Multiple verification methods confirm authenticity.")
                    st.write("✓ Safe to use as prescribed.")
                elif risk_score <= 40:
                    st.success("### ✔️ MODERATE-HIGH CONFIDENCE: Likely Authentic")
                    st.write("✓ Most verification methods indicate authenticity.")
                    st.write("✓ Minor concerns, but generally safe.")
                elif risk_score <= 60:
                    st.warning("### ⚠️ MODERATE CONFIDENCE: Requires Caution")
                    st.write("⚠ Mixed signals detected.")
                    st.write("⚠ Verify with a pharmacist before use.")
                elif risk_score <= 80:
                    st.error("### ⚠️ LOW CONFIDENCE: Likely Suspicious")
                    st.write("⚠ Multiple red flags detected.")
                    st.write("⚠ Do NOT use. Seek verification.")
                else:
                    st.error("### ❌ VERY LOW CONFIDENCE: Likely Counterfeit")
                    st.write("❌ Strong indicators of counterfeit drug.")
                    st.write("❌ DO NOT USE. Report immediately.")
            
            # Recommendations
            st.markdown("---")
            st.subheader("📋 Recommendations")
            
            if risk_score > 60:
                st.error("""
                ### ⚠️ HIGH RISK - DO NOT USE
                - 🔴 **DO NOT** consume this medication
                - 📞 Report to Kenya Pharmacy and Poisons Board (PPB)
                - 🏥 Consult a healthcare professional immediately
                - 📸 Take photos of package and batch number
                - 🗑️ Dispose of safely through a pharmacy
                - 🚨 Report seller to authorities
                """)
            elif risk_score > 40:
                st.warning("""
                ### ⚠️ MODERATE RISK - VERIFY BEFORE USE
                - ⚠️ Do not consume until verified
                - 🏪 Return to place of purchase for verification
                - 💊 Consult a licensed pharmacist
                - 🔍 Check physical security features (holograms, seals)
                - 📞 Contact manufacturer for verification
                """)
            else:
                st.info("""
                ### ✅ LOW RISK - STANDARD PRECAUTIONS
                - ✅ Drug appears legitimate
                - 🔍 Still verify packaging integrity and seals
                - 📱 Check manufacturer's security features
                - 🌡️ Store according to package instructions
                - 📅 Always check expiration date
                - 💊 Follow prescribed dosage strictly
                - 🏥 Report any adverse effects to doctor
                """)
            
            # Important disclaimer
            st.markdown("---")
            st.warning("""
            **⚠️ IMPORTANT DISCLAIMER:**
            This is a demonstration tool for educational purposes. It should NOT be used as the sole method 
            for verifying drug authenticity. Always:
            - Purchase drugs from licensed pharmacies
            - Consult healthcare professionals
            - Verify with official regulatory bodies
            - Report suspected counterfeits to authorities
            """)
            
            # Log verification
            st.caption(f"🕐 Verification completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar
    st.sidebar.title("📈 Model Performance")
    
    model_display = st.sidebar.selectbox("View Metrics For:", ["Random Forest", "KNN"])
    
    if "Random Forest" in model_display:
        m = metrics['rf']
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        st.sidebar.metric("F1 Score", f"{m['f1']:.2%}")
        st.sidebar.metric("CV Score", f"{m['cv_score']:.2%}")
    else:
        m = metrics['knn']
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        st.sidebar.metric("F1 Score", f"{m['f1']:.2%}")
        st.sidebar.metric("CV Score", f"{m['cv_score']:.2%}")
    
    st.sidebar.markdown("---")
    
    # Dataset info
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Total Samples:** {len(df)}")
    st.sidebar.write(f"**Authentic:** {len(df[df['authentic']==1])}")
    st.sidebar.write(f"**Suspicious:** {len(df[df['authentic']==0])}")
    st.sidebar.write(f"**Counterfeit:** {len(df[df['authentic']==-1])}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 About")
    st.sidebar.info("""
    This application demonstrates how technology can combat counterfeit drugs, 
    supporting UN Sustainable Development Goal 3: Good Health and Well-being.
    
    **Technologies:**
    - Machine Learning (Scikit-learn)
    - TF-IDF Vectorization
    - Fuzzy String Matching
    - OpenFDA API
    - Blockchain (simulated)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Test Samples")
    with st.sidebar.expander("Try these"):
        st.code("✅ Panadol + GSK")
        st.code("✅ Augmentin + GSK")
        st.code("✅ Aspirin + Bayer")
        st.code("⚠️ QuickCure + Quick Pharma")
        st.code("❌ FakeMedX + Unknown")
        st.code("❌ FakePain + Fake Co")

if __name__ == "__main__":
    main()