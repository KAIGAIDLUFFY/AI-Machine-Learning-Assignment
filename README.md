💊 Drug Authenticity Verification System

> Combating counterfeit pharmaceuticals with AI-powered multi-layered verification

An intelligent drug authenticity checker that uses machine learning, regulatory databases, and blockchain technology to verify pharmaceutical products and protect public health.


 🎯 About the Project

The Problem

Counterfeit drugs kill over 1 million people annually worldwide. The World Health Organization estimates that 10% of medicines in developing countries are substandard or falsified. This crisis:

- Threatens public health and patient safety
- Undermines trust in healthcare systems  
- Costs the global economy over $200 billion per year
- Disproportionately affects vulnerable populations in low- and middle-income countries

Solution

This project addresses UN Sustainable Development Goal 3 (Good Health and Well-being) by providing an accessible, AI-powered tool that verifies drug authenticity through multiple independent verification layers.

SDG Alignment

- Primary: SDG 3 - Good Health and Well-being
- Secondary: SDG 9 - Industry, Innovation and Infrastructure
- Tertiary: SDG 17 - Partnerships for the Goals

---

 ✨ Features

Core Capabilities

- 🤖 **Dual ML Models** - Random Forest (90-95% accuracy) and KNN (85-90% accuracy)
- 🇰🇪 **PPB Registry Integration** - Fuzzy matching with Kenya Pharmacy and Poisons Board database
- 🌐 **Real-time API Verification** - OpenFDA database cross-reference
- 🔗 **Blockchain Verification** - SHA-256 hash validation (simulated)
- 📊 **Risk Scoring System** - Comprehensive 0-100 risk assessment
- 📱 **Responsive Web Interface** - Works on desktop, tablet, and mobile
- 🔍 **Multi-layered Verification** - Four independent verification methods

### User Experience

- ✅ No account required - immediate access
- ✅ Free and open-source
- ✅ Color-coded risk indicators (🟢🟡🟠🔴)
- ✅ Clear, actionable recommendations
- ✅ Interactive visualizations
- ✅ Ethical transparency and limitations disclosure


 Screenshots

 Main Interface 
 <img width="1920" height="1080" alt="main interface" src="https://github.com/user-attachments/assets/f2117e1a-363a-421f-8614-767ab3e51f07" />



Verification Results
<img width="1920" height="1080" alt="verification results ppb check" src="https://github.com/user-attachments/assets/06e7d72d-62fd-4c54-bd62-e715949e47da" />
<img width="1920" height="1080" alt="verification-results-mlanalysis" src="https://github.com/user-attachments/assets/d5acf9c5-59dd-443a-9299-ee13fa0c140b" />
<img width="1920" height="1080" alt="verification results fda check" src="https://github.com/user-attachments/assets/79fbdc94-b451-4696-8977-d614762a274f" />

 Risk Assessment
 [Risk Score]
<img width="1920" height="1080" alt="risk-assessment" src="https://github.com/user-attachments/assets/6952337e-6c12-474e-a047-3d8c4401c1b4" />
The Risk Score is calculated by:

Starting at 50 (neutral)
Adjusting based on ML prediction (-30 to +40, weighted by confidence)
Adjusting based on PPB registry (-25 to +30)
Adjusting based on FDA database (-20 to +5)
Adjusting based on blockchain (-15 to 0)
Capping between 0-100
Final score interpretation:

0-20: ✅ Authentic (multiple confirmations)
21-40: ✔️ Likely authentic (mostly positive signals)
41-60: ⚠️ Uncertain (mixed signals - verify further)
61-80: ⚠️ Suspicious (multiple red flags)
81-100: ❌ Counterfeit (strong negative indicators)
This approach provides a balanced, transparent, and actionable assessment of drug authenticity! 🎯

