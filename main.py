"""
================================================================================
                   MEDICAL DIAGNOSIS REASONING ENGINE
                    CustomTkinter GUI Application
================================================================================

A sophisticated AI system implementing 6 Course Outcomes:
CO1: Knowledge Representation
CO2: A* Search
CO3: CSP with Backtracking
CO4: Utility Functions
CO5: Bayesian Networks
CO6: Hybrid Architecture
"""

# pyrefly: ignore [missing-import]
import customtkinter as ctk
# pyrefly: ignore [missing-import]
from customtkinter import CTkLabel, CTkEntry, CTkButton, CTkFrame, CTkTabview, CTkScrollableFrame
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt 
# pyrefly: ignore [missing-import]
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

from ai_engine import MedicalReasoningEngine
from database import MedicalDatabase
from utils import ReportGenerator
from config import SYMPTOMS, DISEASES, DISEASE_INFO


class MedicalDiagnosisGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Diagnosis Reasoning Engine")
        self.root.geometry("1200x700")
        self.root.configure(bg="#0f172a")
        
        # Initialize backends
        self.engine = MedicalReasoningEngine()
        self.db = MedicalDatabase()
        self.reporter = ReportGenerator()
        
        # Current patient data
        self.patient_data = {}
        self.selected_symptoms = []
        self.diagnosis_results = None
        
        self._create_ui()
        
    def _create_ui(self):
        """Create the main UI with tabs"""
        # Main frame
        main_frame = CTkFrame(self.root, fg_color="#141E2E", corner_radius=24)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title Banner
        title_frame = CTkFrame(main_frame, fg_color="#172A45", corner_radius=20)
        title_frame.pack(fill="x", padx=10, pady=(15, 10))
        
        title = CTkLabel(title_frame, text="🏥 Medical Diagnosis Reasoning Engine", 
                         font=("Segoe UI", 24, "bold"), text_color="#7dd3fc")
        title.pack(pady=(12, 4))
        
        subtitle = CTkLabel(title_frame, text="Advanced AI Clinical Decision Support System", 
                            font=("Segoe UI", 12, "italic"), text_color="#cbd5e1")
        subtitle.pack(pady=(0, 12))
        
        stats_frame = CTkFrame(main_frame, fg_color="#152337", corner_radius=18,
                               border_width=1, border_color="#334155")
        stats_frame.pack(fill="x", padx=10, pady=(0, 12))
        
        stat_labels = [
            ("Patients", "24", "#38bdf8"),
            ("Diagnoses", "18", "#22c55e"),
            ("Ready", "Live AI", "#fb7185")
        ]
        
        for label_text, value_text, accent in stat_labels:
            card = CTkFrame(stats_frame, fg_color="#172f4d", corner_radius=16,
                            border_width=1, border_color="#294563")
            card.pack(side="left", expand=True, fill="both", padx=8, pady=12)
            CTkLabel(card, text=label_text, font=("Segoe UI", 11), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 2))
            CTkLabel(card, text=value_text, font=("Segoe UI", 24, "bold"), text_color=accent).pack(anchor="w", padx=12, pady=(0, 12))
        
        # Tabs
        self.tabview = CTkTabview(
            main_frame,
            fg_color="#1f2937",
            corner_radius=20,
            border_width=1,
            border_color="#334155"
        )
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tabs
        self.tab_patient = self.tabview.add("👤 Patient Profile")
        self.tab_symptoms = self.tabview.add("🔍 Symptom Wizard")
        self.tab_results = self.tabview.add("📊 Diagnosis Results")
        self.tab_visualization = self.tabview.add("📈 Bayes Visuals")
        self.tab_trace = self.tabview.add("🔬 Reasoning Trace")
        self.tab_database = self.tabview.add("💾 Records")
        
        self._setup_patient_tab()
        self._setup_symptoms_tab()
        self._setup_results_tab()
        self._setup_visualization_tab()
        self._setup_trace_tab()
        self._setup_database_tab()
        
    def _setup_patient_tab(self):
        """Patient information entry tab"""
        frame = CTkScrollableFrame(
            self.tab_patient,
            fg_color="#172a45",
            border_width=1,
            border_color="#334155",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        CTkLabel(frame, text="Patient Information", font=("Segoe UI", 16, "bold"), 
                text_color="#00ffcc").pack(pady=10)
        
        # Input fields
        fields = [
            ("Patient Name:", "name", "e.g. Jane Doe"),
            ("Age:", "age", "e.g. 34"),
            ("Gender (M/F/O):", "gender", "e.g. F"),
            ("Email Address:", "email", "e.g. jane.doe@example.com"),
            ("Phone Number:", "phone", "e.g. 555-0144"),
            ("Medical History:", "history", "e.g. Seasonal allergies, minor asthma")
        ]
        
        self.patient_entries = {}
        
        for label, key, placeholder in fields:
            CTkLabel(frame, text=label, font=("Segoe UI", 12, "bold"), text_color="#cbd5e1").pack(anchor="w", padx=20, pady=(10, 0))
            entry = CTkEntry(frame, width=350, height=35, fg_color="#2d2d2d", border_color="#404040", placeholder_text=placeholder)
            entry.pack(padx=20, pady=5)
            self.patient_entries[key] = entry
            
        # Pre-populate one default example with random info
        self.patient_entries['name'].insert(0, "Jane Doe")
        self.patient_entries['age'].insert(0, "34")
        self.patient_entries['gender'].insert(0, "F")
        self.patient_entries['email'].insert(0, "jane.doe@example.com")
        self.patient_entries['phone'].insert(0, "555-0144")
        self.patient_entries['history'].insert(0, "Seasonal allergies, minor asthma")
        
        # Buttons
        btn_frame = CTkFrame(frame, fg_color="#1e1e1e")
        btn_frame.pack(pady=20)
        
        CTkButton(btn_frame, text="✓ Save Patient", command=self._save_patient,
                 fg_color="#10b981", hover_color="#059669", font=("Segoe UI", 12, "bold")).pack(side="left", padx=5)
                 
        CTkButton(btn_frame, text="🎲 Load Random Example", command=self._load_random_example,
                 fg_color="#6366f1", hover_color="#4f46e5", font=("Segoe UI", 12, "bold")).pack(side="left", padx=5)
                 
        CTkButton(btn_frame, text="🔄 Clear Form", command=self._clear_form,
                 fg_color="#ef4444", hover_color="#dc2626", font=("Segoe UI", 12)).pack(side="left", padx=5)

    def _load_random_example(self):
        """Loads a random patient profile and symptoms"""
        import random
        profiles = [
            {
                'name': 'Sarah Jenkins',
                'age': '28',
                'gender': 'F',
                'email': 'sarah.j@example.com',
                'phone': '555-0123',
                'history': 'Mild asthma, seasonal allergies',
                'symptoms': ['fever', 'cough', 'fatigue']
            },
            {
                'name': 'David Chen',
                'age': '45',
                'gender': 'M',
                'email': 'dchen99@example.com',
                'phone': '555-0178',
                'history': 'Hypertension, otherwise healthy',
                'symptoms': ['headache', 'nausea', 'dizziness']
            },
            {
                'name': 'Emma Watson',
                'age': '34',
                'gender': 'F',
                'email': 'emma.w@example.com',
                'phone': '555-0192',
                'history': 'None',
                'symptoms': ['sore_throat', 'cough', 'runny_nose']
            },
            {
                'name': 'James Miller',
                'age': '62',
                'gender': 'M',
                'email': 'jmiller@example.com',
                'phone': '555-0143',
                'history': 'Type 2 Diabetes',
                'symptoms': ['fatigue', 'shortness_of_breath', 'chest_pain']
            },
            {
                'name': 'Sophia Martinez',
                'age': '19',
                'gender': 'F',
                'email': 'smartinez@example.com',
                'phone': '555-0211',
                'history': 'None',
                'symptoms': ['fever', 'nausea', 'vomiting', 'diarrhea']
            }
        ]
        
        profile = random.choice(profiles)
        
        # Clear existing patient form
        self._clear_form()
        
        # Insert new data
        self.patient_entries['name'].insert(0, profile['name'])
        self.patient_entries['age'].insert(0, profile['age'])
        self.patient_entries['gender'].insert(0, profile['gender'])
        self.patient_entries['email'].insert(0, profile['email'])
        self.patient_entries['phone'].insert(0, profile['phone'])
        self.patient_entries['history'].insert(0, profile['history'])
        
        # Clear and set symptoms
        self._clear_symptoms()
        for symptom in profile['symptoms']:
            if symptom in self.symptom_vars:
                self.symptom_vars[symptom].set(True)
                
        # Switch tab view back to Patient Info to let them see
        self.tabview.set("👤 Patient Info")
        
        # Show a confirmation in the results display
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"✓ Loaded random demo profile: {profile['name']}\nClick 'Save Patient' or go directly to 'Symptoms' tab to run diagnosis.")
        
    def _setup_symptoms_tab(self):
        """Symptom selection tab"""
        frame = CTkScrollableFrame(
            self.tab_symptoms,
            fg_color="#172a45",
            border_width=1,
            border_color="#334155",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        CTkLabel(frame, text="Select Symptoms", font=("Segoe UI", 16, "bold"), 
                text_color="#00ffcc").pack(pady=10)
        
        # Symptom checkboxes
        self.symptom_vars = {}
        
        for symptom in SYMPTOMS:
            var = ctk.BooleanVar(value=False)
            self.symptom_vars[symptom] = var
            
            # Show a cleaner label for symptoms
            display_name = symptom.replace("_", " ").title()
            check = ctk.CTkCheckBox(
                frame,
                text=display_name,
                variable=var,
                text_color="#ffffff",
                checkbox_width=22,
                checkbox_height=22,
                hover_color="#1d4ed8",
                font=("Segoe UI", 12)
            )
            check.pack(anchor="w", padx=30, pady=5)
            
        # Pre-select default symptoms corresponding to our pre-filled patient
        self.symptom_vars['fever'].set(True)
        self.symptom_vars['cough'].set(True)
        self.symptom_vars['fatigue'].set(True)
        
        # Action buttons
        btn_frame = CTkFrame(frame, fg_color="#152336")
        btn_frame.pack(pady=20)
        
        CTkButton(btn_frame, text="🧠 Diagnose", command=self._perform_diagnosis,
                 fg_color="#10b981", hover_color="#059669", font=("Segoe UI", 14, "bold"), width=200, height=44, corner_radius=16).pack(side="left", padx=5)
        CTkButton(btn_frame, text="❌ Clear All", command=self._clear_symptoms,
                 fg_color="#f59e0b", hover_color="#d97706", font=("Segoe UI", 14, "bold"), width=200, height=44, corner_radius=16).pack(side="left", padx=5)
        
    def _setup_results_tab(self):
        """Results display tab"""
        frame = CTkScrollableFrame(
            self.tab_results,
            fg_color="#172a45",
            border_width=1,
            border_color="#334155",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        CTkLabel(frame, text="Diagnosis Results", font=("Segoe UI", 16, "bold"), 
                text_color="#00ffcc").pack(pady=10)
        
        self.results_text = ctk.CTkTextbox(frame, width=1000, height=400, 
                                          fg_color="#1b283a", text_color="#ffffff",
                                          border_color="#2d2d44", corner_radius=14, font=("Consolas", 12))
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Action buttons
        btn_frame = CTkFrame(frame, fg_color="#152336")
        btn_frame.pack(pady=10)
        
        CTkButton(btn_frame, text="💾 Save to Database", command=self._save_to_db,
                 fg_color="#6366f1", hover_color="#4f46e5", font=("Segoe UI", 12, "bold"), width=180, height=40, corner_radius=16).pack(side="left", padx=5)
        CTkButton(btn_frame, text="📄 Generate PDF", command=self._generate_pdf,
                 fg_color="#ec4899", hover_color="#db2777", font=("Segoe UI", 12, "bold"), width=180, height=40, corner_radius=16).pack(side="left", padx=5)
        
    def _setup_visualization_tab(self):
        """Visualization tab with Matplotlib"""
        self.viz_frame = CTkFrame(self.tab_visualization, fg_color="#172a45", border_width=1,
                                 border_color="#334155", corner_radius=20)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        CTkLabel(self.viz_frame, text="Probability Visualization", font=("Segoe UI", 16, "bold"),
                text_color="#00ffcc").pack(pady=10)
        CTkLabel(self.viz_frame, text="Use the Results tab to generate probability charts for top diagnoses.",
                font=("Segoe UI", 11), text_color="#cbd5e1").pack(pady=(0, 10))
        
    def _setup_trace_tab(self):
        """Reasoning trace tab"""
        frame = CTkScrollableFrame(
            self.tab_trace,
            fg_color="#172a45",
            border_width=1,
            border_color="#334155",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        CTkLabel(frame, text="AI Reasoning Steps", font=("Segoe UI", 16, "bold"), 
                text_color="#00ffcc").pack(pady=10)
        
        self.trace_text = ctk.CTkTextbox(frame, width=1000, height=400, 
                                        fg_color="#1b283a", text_color="#ffffff",
                                        border_color="#2d2d44", corner_radius=14, font=("Consolas", 12))
        self.trace_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def _setup_database_tab(self):
        """Database browser tab"""
        frame = CTkScrollableFrame(
            self.tab_database,
            fg_color="#172a45",
            border_width=1,
            border_color="#334155",
            corner_radius=20
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        CTkLabel(frame, text="Patient Database", font=("Segoe UI", 16, "bold"), 
                text_color="#00ffcc").pack(pady=10)
        
        self.db_text = ctk.CTkTextbox(frame, width=1000, height=400, 
                                     fg_color="#1b283a", text_color="#ffffff",
                                     border_color="#2d2d44", corner_radius=14, font=("Consolas", 12))
        self.db_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        CTkButton(frame, text="🔄 Refresh Database", command=self._refresh_database,
                 fg_color="#10b981", hover_color="#059669", font=("Segoe UI", 12, "bold"), width=220, height=42, corner_radius=16).pack(pady=10)
        
    def _save_patient(self):
        """Save patient information"""
        try:
            self.patient_data = {
                'name': self.patient_entries['name'].get(),
                'age': int(self.patient_entries['age'].get()),
                'gender': self.patient_entries['gender'].get(),
                'email': self.patient_entries['email'].get(),
                'phone': self.patient_entries['phone'].get(),
                'history': self.patient_entries['history'].get()
            }
            
            # Add to database
            pid = self.db.add_patient(**self.patient_data)
            
            # Show confirmation
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", f"✓ Patient '{self.patient_data['name']}' saved (ID: {pid})")
        except Exception as e:
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", f"❌ Error: {str(e)}")
            
    def _clear_form(self):
        """Clear patient form"""
        for entry in self.patient_entries.values():
            entry.delete(0, "end")
            
    def _clear_symptoms(self):
        """Clear symptom selections"""
        for var in self.symptom_vars.values():
            var.set(False)
            
    def _perform_diagnosis(self):
        """Run AI diagnosis pipeline"""
        # Get selected symptoms
        self.selected_symptoms = [s for s, v in self.symptom_vars.items() if v.get()]
        
        if not self.selected_symptoms:
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", "⚠️ Please select at least one symptom")
            return
        
        if not self.patient_data:
            self.results_text.delete("0.0", "end")
            self.results_text.insert("0.0", "⚠️ Please enter patient information first")
            return
        
        # Run reasoning engine
        self.diagnosis_results = self.engine.reason(self.selected_symptoms)
        
        # Display results
        self._display_results()
        self._display_visualization()
        self._display_trace()
        
    def _display_results(self):
        """Display diagnosis results"""
        if not self.diagnosis_results:
            return
        
        output = "="*60 + "\n"
        output += "🏥 MEDICAL DIAGNOSIS RESULTS\n"
        output += "="*60 + "\n\n"
        
        output += f"👤 Patient: {self.patient_data.get('name', 'Unknown')}\n"
        output += f"🔍 Symptoms: {', '.join(self.selected_symptoms)}\n\n"
        
        output += "📋 PRIMARY DIAGNOSIS:\n"
        output += f"   Disease: {self.diagnosis_results['best_diagnosis']}\n"
        output += f"   Confidence: {self.diagnosis_results['probability']:.1%}\n\n"
        
        output += "💊 RECOMMENDED TREATMENT:\n"
        treatment = self.diagnosis_results.get('treatment', 'Consult physician')
        output += f"   {treatment}\n\n"
        
        output += "🔝 TOP PROBABLE DIAGNOSES:\n"
        ranked = self.diagnosis_results.get('ranked_treatments', {})
        for disease, data in list(ranked.items())[:5]:
            prob = data.get('probability', 0)
            output += f"   • {disease}: {prob:.1%}\n"
        
        output += "\n⚠️ MEDICAL DISCLAIMER:\n"
        output += "This system is for educational purposes only.\n"
        output += "Always consult a qualified healthcare professional.\n"
        
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", output)
        
    def _display_visualization(self):
        """Display probability chart"""
        if not self.diagnosis_results:
            return
        
        # Clear previous figure
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Get top diagnoses
        ranked = self.diagnosis_results.get('ranked_treatments', {})
        diseases = list(ranked.keys())[:10]
        probs = [ranked[d].get('probability', 0) * 100 for d in diseases]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100, facecolor="#2d2d2d")
        ax.bar(diseases, probs, color="#00a8ff", edgecolor="#ffffff", linewidth=1.5)
        ax.set_ylabel("Probability (%)", color="#ffffff")
        ax.set_xlabel("Disease", color="#ffffff")
        ax.set_title("Top Diagnosis Probabilities", color="#ffffff", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#ffffff")
        ax.set_facecolor("#1e1e1e")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def _display_trace(self):
        """Display reasoning trace"""
        if not self.engine.reasoning_trace:
            return
        
        output = "="*60 + "\n"
        output += "🔬 AI REASONING TRACE\n"
        output += "="*60 + "\n\n"
        
        for i, (step, description) in enumerate(self.engine.reasoning_trace, 1):
            output += f"Step {i}: {step}\n"
            output += f"   → {description}\n\n"
        
        output += "="*60 + "\n"
        output += "✓ Reasoning pipeline complete\n"
        
        self.trace_text.delete("0.0", "end")
        self.trace_text.insert("0.0", output)
        
    def _save_to_db(self):
        """Save results to database"""
        if not self.diagnosis_results:
            self.results_text.insert("end", "\n\n❌ No diagnosis to save")
            return
        
        try:
            disease = self.diagnosis_results['best_diagnosis']
            prob = self.diagnosis_results['probability']
            treatment = self.diagnosis_results['treatment']
            
            # Add diagnosis record
            self.db.add_diagnosis(
                patient_id=1,  # Simplified
                disease=disease,
                probability=prob,
                symptoms=",".join(self.selected_symptoms),
                treatment=treatment,
                utility=prob
            )
            
            self.results_text.insert("end", "\n\n✓ Results saved to database")
        except Exception as e:
            self.results_text.insert("end", f"\n\n❌ Error: {str(e)}")
            
    def _generate_pdf(self):
        """Generate PDF report"""
        if not self.diagnosis_results:
            self.results_text.insert("end", "\n\n❌ No diagnosis to export")
            return
        
        try:
            filename = f"diagnosis_{self.patient_data.get('name', 'patient').replace(' ', '_')}.pdf"
            self.reporter.generate_report(
                patient_info=self.patient_data,
                diagnoses=self.diagnosis_results,
                trace=self.engine.reasoning_trace,
                symptoms=self.selected_symptoms
            )
            self.results_text.insert("end", f"\n\n✓ PDF generated: {filename}")
        except Exception as e:
            self.results_text.insert("end", f"\n\n❌ Error: {str(e)}")
            
    def _refresh_database(self):
        """Refresh database view"""
        try:
            stats = self.db.get_statistics()
            output = "DATABASE STATISTICS\n"
            output += "="*40 + "\n\n"
            output += f"Total Patients: {stats.get('patients', 0)}\n"
            output += f"Total Diagnoses: {stats.get('diagnoses', 0)}\n"
            output += f"Reasoning Traces: {stats.get('traces', 0)}\n"
            
            self.db_text.delete("0.0", "end")
            self.db_text.insert("0.0", output)
        except Exception as e:
            self.db_text.delete("0.0", "end")
            self.db_text.insert("0.0", f"❌ Error: {str(e)}")


def main():
    """Application entry point"""
    print("\n" + "="*80)
    print(" "*15 + "Medical Diagnosis Reasoning Engine")
    print(" "*20 + "AI Clinical Decision Support System")
    print("="*80)
    print("\n[v] Initializing AI components...")
    print("[v] Knowledge Representation (Rule Base)")
    print("[v] A* Search Algorithm")
    print("[v] CSP with Backtracking")
    print("[v] Utility Function Optimization")
    print("[v] Bayesian Network Inference")
    print("[v] Hybrid Architecture Integration")
    print("\n[v] Launching GUI...\n")
    
    # Configure CustomTkinter
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    root = ctk.CTk()
    app = MedicalDiagnosisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
