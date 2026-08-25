import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ==========================================
# 1. SISTEMA DE AUDITORÍA CLÍNICA
# ==========================================
LOG_FILE = "auditoria_clinica.log"

def registrar_auditoria(estado: str, mensaje: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{timestamp}] | STATUS: {estado} | {mensaje}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(linea)
    except Exception as e:
        print(f"[ERROR DE LOGS] {e}")

# ==========================================
# 2. ARQUITECTURAS NEURONALES TABULARES
# ==========================================
class MedicalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.output(x))

modelos_patologias = {
    "diabetes": MedicalNet(input_dim=8),
    "cardio": MedicalNet(input_dim=13),
    "oncologia": MedicalNet(input_dim=30)
}

for m in modelos_patologias.values():
    m.eval()

# ==========================================
# 3. MOTOR DE TRIAJE NLP BILINGÜE
# ==========================================
def analizar_sintomas_nlp(texto: str):
    texto_lower = texto.lower()
    
    criticos = ["pecho", "ahogo", "respirar", "infarto", "desmayo", "sangre", "chest", "breath", "heart attack", "fainting", "blood"]
    moderados = ["fiebre", "vómito", "dolor fuerte", "mareo", "infección", "fever", "vomit", "strong pain", "dizziness", "infection"]
    
    score_urgencia = 1 
    especialidad = "Medicina General / General Practice"
    
    if any(palabra in texto_lower for palabra in criticos):
        score_urgencia = 3  
        especialidad = "Urgencias / Cardiología / UCI (Emergency / Cardiology)"
    elif any(palabra in texto_lower for palabra in moderados):
        score_urgencia = 2  
        especialidad = "Medicina Interna / Especialidad (Internal Medicine)"
    
    if "corazón" in texto_lower or "heart" in texto_lower or "palpitaciones" in texto_lower:
        especialidad = "Cardiología / Cardiology"
    elif "hueso" in texto_lower or "bone" in texto_lower or "caída" in texto_lower or "fall" in texto_lower:
        especialidad = "Traumatología / Orthopedics"
    elif "cabeza" in texto_lower or "head" in texto_lower or "migraña" in texto_lower:
        especialidad = "Neurología / Neurology"

    niveles = {
        1: ("LEVE / RUTINARIO (MILD / ROUTINE)", "VERDE"), 
        2: ("MODERADO / OBSERVACIÓN (MODERATE / OBSERVATION)", "AMARILLO"), 
        3: ("CRÍTICO / ATENCIÓN INMEDIATA (CRITICAL / IMMEDIATE)", "ROJO")
    }
    urgencia_txt, color = niveles[score_urgencia]
    
    return {
        "urgencia": urgencia_txt,
        "nivel_color": color,
        "especialidad_sugerida": especialidad,
        "recomendacion_es": "Derivar de inmediato a sala de urgencias." if score_urgencia == 3 else "Programar cita prioritaria con especialista." if score_urgencia == 2 else "Tratamiento ambulatorio o control rutinario.",
        "recomendacion_en": "Derive immediately to the emergency room." if score_urgencia == 3 else "Schedule a priority appointment with a specialist." if score_urgencia == 2 else "Outpatient treatment or routine follow-up."
    }

# ==========================================
# 4. FASTAPI Y SEGURIDAD
# ==========================================
app = FastAPI(title="Clinical Intelligence Platform", version="4.2.0")
TOKEN_MEDICO_AUTORIZADO = os.getenv("TOKEN_MEDICO_AUTORIZADO", "hospital-med-token-2026-secure")
security = HTTPBearer()

def verificar_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TOKEN_MEDICO_AUTORIZADO:
        raise HTTPException(status_code=403, detail="Token inválido.")
    return credentials.credentials

class PeticionTabular(BaseModel):
    patologia: str
    variables: list[float]

class PeticionTriaje(BaseModel):
    sintomas_texto: str

# ==========================================
# 5. INTERFAZ WEB (AHORA CON 3 PESTAÑAS)
# ==========================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clinical Intelligence Platform</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 20px; }
            .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0px 4px 20px rgba(0,0,0,0.08); position: relative; }
            .lang-switcher { position: absolute; top: 25px; right: 30px; }
            .lang-btn { background: #e9ecef; border: 1px solid #ced4da; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; }
            .lang-btn:hover { background: #dee2e6; }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 5px; }
            .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 25px; font-size: 14px; }
            .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 25px; overflow-x: auto; }
            .tab-btn { flex: 1; padding: 12px; background: none; border: none; font-size: 15px; font-weight: bold; cursor: pointer; color: #7f8c8d; transition: 0.3s; white-space: nowrap; }
            .tab-btn.active { color: #007bff; border-bottom: 3px solid #007bff; margin-bottom: -2px; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .form-group { margin-bottom: 20px; }
            label { display: block; font-weight: bold; margin-bottom: 8px; color: #34495e; }
            input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
            textarea { resize: vertical; height: 100px; }
            .btn { width: 100%; padding: 12px; background-color: #007bff; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; }
            .btn:hover { background-color: #0056b3; }
            .result-card { margin-top: 25px; padding: 20px; border-radius: 8px; display: none; background: #e9ecef; }
            .faq-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #007bff; }
            .faq-section h4 { margin-top: 0; color: #2c3e50; }
            ul { line-height: 1.6; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="lang-switcher">
                <button class="lang-btn" onclick="toggleLanguage()" id="langBtn">🇬🇧 EN / 🇪🇸 ES</button>
            </div>

            <h1 id="ui-title">🏥 Centro de Inteligencia Clínica</h1>
            <p class="subtitle" id="ui-subtitle">Sistemas de Apoyo al Diagnóstico (CDS) - Tabular & NLP</p>
            
            <div class="form-group" style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <label for="token" id="lbl-token">🔑 Token de Autorización (Bearer Token):</label>
                <input type="password" id="token" value="hospital-med-token-2026-secure">
            </div>

            <div class="tabs">
                <button class="tab-btn active" onclick="cambiarTab(event, 'tab-tabular')" id="tab1-btn">1. Diagnóstico Tabular</button>
                <button class="tab-btn" onclick="cambiarTab(event, 'tab-nlp')" id="tab2-btn">2. Triaje Inteligente</button>
                <button class="tab-btn" onclick="cambiarTab(event, 'tab-faq')" id="tab3-btn">3. Marco Teórico (FAQ)</button>
            </div>

            <!-- TAB 1: TABULAR -->
            <div id="tab-tabular" class="tab-content active">
                <div class="form-group">
                    <label for="patologia" id="lbl-pat">Seleccionar Patología a Evaluar:</label>
                    <select id="patologia" onchange="actualizarCamposTabulares()">
                        <option value="diabetes">Diabetes Tipo 2 (8 variables)</option>
                        <option value="cardio">Riesgo Cardíaco / Heart Risk (13 variables)</option>
                        <option value="oncologia">Oncología / Cáncer de Mama (30 variables)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="variablesInput" id="lbl-vars">Vector de Variables (Separadas por comas):</label>
                    <input type="text" id="variablesInput" placeholder="Ej: 148, 72, 35, 0, 0, 33.6, 0.627, 50">
                </div>
                <button class="btn" onclick="enviarTabular()" id="btn-tab-exec">Ejecutar Inferencia Tabular</button>
                <div id="resultadoTabular" class="result-card">
                    <h3 style="margin-top:0;" id="res-tab-title">Resultado del Análisis</h3>
                    <p><strong><span id="lbl-diag">Diagnóstico</span>:</strong> <span id="res-tab-diag"></span></p>
                    <p><strong><span id="lbl-prob">Probabilidad de Riesgo</span>:</strong> <span id="res-tab-prob"></span>%</p>
                </div>
            </div>

            <!-- TAB 2: NLP -->
            <div id="tab-nlp" class="tab-content">
                <div class="form-group">
                    <label for="sintomasInput" id="lbl-sintomas">Describe los síntomas (Español o Inglés):</label>
                    <textarea id="sintomasInput" placeholder="Ej: Dolor punzante en el pecho con dificultad para respirar..."></textarea>
                </div>
                <button class="btn" onclick="enviarTriaje()" id="btn-nlp-exec">Procesar Triaje con NLP</button>
                <div id="resultadoTriaje" class="result-card">
                    <h3 style="margin-top:0;" id="res-nlp-title">Evaluación de Triaje</h3>
                    <div id="badge-urgencia" style="margin-bottom: 12px; padding:10px; border-radius:6px; font-weight:bold;"></div>
                    <p><strong><span id="lbl-esp">Especialidad Sugerida</span>:</strong> <span id="res-nlp-esp"></span></p>
                    <p><strong><span id="lbl-rec">Protocolo Clínico</span>:</strong> <span id="res-nlp-rec"></span></p>
                </div>
            </div>

            <!-- TAB 3: FAQ / MARCO TEÓRICO -->
            <div id="tab-faq" class="tab-content">
                
                <!-- FAQ EN ESPAÑOL -->
                <div id="faq-es">
                    <h2>📌 FAQ: Entendiendo los Vectores Clínicos</h2>
                    
                    <div class="faq-section">
                        <h4>1. ¿Qué es exactamente un "Vector de Variables"?</h4>
                        <p>En nuestra plataforma, un vector representa el <strong>perfil fisiológico completo de un paciente</strong> en un momento dado. Por ejemplo, al evaluar diabetes, la red recibe un vector <i>X = [148.0, 72.0, 35.0, 33.6...]</i> donde cada posición corresponde estrictamente a una variable (Glucosa, Presión, Edad, IMC).</p>
                    </div>

                    <div class="faq-section">
                        <h4>2. ¿Cómo procesa la Red Neuronal este vector?</h4>
                        <p>La red neuronal posee sus propios vectores internos llamados <strong>Pesos (W)</strong> y <strong>Sesgos (b)</strong>. Al introducir el vector del paciente (X), la red calcula algebraicamente el producto punto: <i>z = (W &middot; X) + b</i>. El resultado (z) es el <strong>logit</strong>, que luego se comprime en un porcentaje del 0% al 100% mediante una función Sigmoide.</p>
                    </div>

                    <div class="faq-section">
                        <h4>3. Composición de los Vectores de Enfermedades</h4>
                        <ul>
                            <li><strong>Diabetes Tipo 2 (8 dim):</strong> V1: Embarazos, V2: Glucosa, V3: Presión diastólica, V4: Pliegue cutáneo, V5: Insulina, V6: IMC, V7: Función genética, V8: Edad.</li>
                            <li><strong>Riesgo Cardiovascular (13 dim):</strong> V1: Edad, V2: Sexo, V3: Tipo de dolor de pecho, V4: Presión en reposo, V5: Colesterol, V6: Azúcar en ayunas, V7-V13: Resultados de ECG, frecuencia máxima, angina inducida y defectos sanguíneos.</li>
                            <li><strong>Oncología Mamaria (30 dim):</strong> Analiza 10 características del núcleo celular extraídas por biopsia (Radio, Textura, Perímetro, Área, Suavidad, Compacidad, Concavidad, Puntos cóncavos, Simetría, Dimensión fractal), calculando la media, error estándar y el peor valor para cada una (10 x 3 = 30).</li>
                        </ul>
                    </div>
                </div>

                <!-- FAQ EN INGLÉS -->
                <div id="faq-en" style="display: none;">
                    <h2>📌 FAQ: Understanding Clinical Vectors</h2>
                    
                    <div class="faq-section">
                        <h4>1. What exactly is a "Variable Vector"?</h4>
                        <p>On our platform, a vector represents the <strong>complete physiological profile of a patient</strong> at a given moment. For example, when evaluating diabetes, the network receives a vector <i>X = [148.0, 72.0, 35.0, 33.6...]</i> where each position corresponds strictly to a variable (Glucose, Blood Pressure, Age, BMI).</p>
                    </div>

                    <div class="faq-section">
                        <h4>2. How does the Neural Network process this vector?</h4>
                        <p>The neural network has its own internal vectors called <strong>Weights (W)</strong> and <strong>Biases (b)</strong>. When introducing the patient's vector (X), the network algebraically calculates the dot product: <i>z = (W &middot; X) + b</i>. The result (z) is the <strong>logit</strong>, which is then compressed into a 0% to 100% percentage using a Sigmoid function.</p>
                    </div>

                    <div class="faq-section">
                        <h4>3. Composition of Disease Vectors</h4>
                        <ul>
                            <li><strong>Type 2 Diabetes (8 dim):</strong> V1: Pregnancies, V2: Glucose, V3: Diastolic pressure, V4: Skinfold thickness, V5: Insulin, V6: BMI, V7: Genetic pedigree function, V8: Age.</li>
                            <li><strong>Cardiovascular Risk (13 dim):</strong> V1: Age, V2: Sex, V3: Chest pain type, V4: Resting pressure, V5: Cholesterol, V6: Fasting sugar, V7-V13: ECG results, max heart rate, induced angina, and blood defects.</li>
                            <li><strong>Breast Oncology (30 dim):</strong> Analyzes 10 cell nucleus features extracted via biopsy (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension), calculating the mean, standard error, and worst value for each (10 x 3 = 30).</li>
                        </ul>
                    </div>
                </div>

            </div>
        </div>

        <script>
            let currentLang = 'es';
            const translations = {
                es: {
                    title: "🏥 Centro de Inteligencia Clínica", subtitle: "Sistemas de Apoyo al Diagnóstico (CDS)",
                    lblToken: "🔑 Token de Autorización:", tab1: "1. Diagnóstico Tabular", tab2: "2. Triaje Inteligente",
                    tab3: "3. Marco Teórico (FAQ)", lblPat: "Seleccionar Patología:", lblVars: "Vector de Variables (Separadas por comas):",
                    btnTabExec: "Ejecutar Inferencia Tabular", resTabTitle: "Resultado del Análisis", lblDiag: "Diagnóstico",
                    lblProb: "Probabilidad de Riesgo", lblSintomas: "Describe los síntomas (Español o Inglés):",
                    btnNlpExec: "Procesar Triaje con NLP", resNlpTitle: "Evaluación de Triaje", lblEsp: "Especialidad Sugerida",
                    lblRec: "Protocolo Clínico", langBtnText: "🇬🇧 EN / 🇪🇸 ES"
                },
                en: {
                    title: "🏥 Clinical Intelligence Center", subtitle: "Clinical Decision Support (CDS)",
                    lblToken: "🔑 Authorization Token:", tab1: "1. Tabular Diagnosis", tab2: "2. Smart NLP Triage",
                    tab3: "3. Theoretical Framework (FAQ)", lblPat: "Select Pathology:", lblVars: "Variables Vector (Comma-separated):",
                    btnTabExec: "Run Tabular Inference", resTabTitle: "Tabular Analysis Result", lblDiag: "Diagnosis",
                    lblProb: "Risk Probability", lblSintomas: "Describe symptoms (Spanish or English):",
                    btnNlpExec: "Process NLP Triage", resNlpTitle: "Triage Evaluation", lblEsp: "Suggested Specialty",
                    lblRec: "Clinical Protocol", langBtnText: "🇪🇸 ES / 🇬🇧 EN"
                }
            };

            function toggleLanguage() {
                currentLang = currentLang === 'es' ? 'en' : 'es';
                const t = translations[currentLang];
                document.getElementById('ui-title').innerText = t.title;
                document.getElementById('ui-subtitle').innerText = t.subtitle;
                document.getElementById('lbl-token').innerText = t.lblToken;
                document.getElementById('tab1-btn').innerText = t.tab1;
                document.getElementById('tab2-btn').innerText = t.tab2;
                document.getElementById('tab3-btn').innerText = t.tab3;
                document.getElementById('lbl-pat').innerText = t.lblPat;
                document.getElementById('lbl-vars').innerText = t.lblVars;
                document.getElementById('btn-tab-exec').innerText = t.btnTabExec;
                document.getElementById('res-tab-title').innerText = t.resTabTitle;
                document.getElementById('lbl-diag').innerText = t.lblDiag;
                document.getElementById('lbl-prob').innerText = t.lblProb;
                document.getElementById('lbl-sintomas').innerText = t.lblSintomas;
                document.getElementById('btn-nlp-exec').innerText = t.btnNlpExec;
                document.getElementById('res-nlp-title').innerText = t.resNlpTitle;
                document.getElementById('lbl-esp').innerText = t.lblEsp;
                document.getElementById('lbl-rec').innerText = t.lblRec;
                document.getElementById('langBtn').innerText = t.langBtnText;
                
                // Alternar vista del FAQ
                document.getElementById('faq-es').style.display = currentLang === 'es' ? 'block' : 'none';
                document.getElementById('faq-en').style.display = currentLang === 'en' ? 'block' : 'none';
            }

            function cambiarTab(evt, tabId) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                evt.currentTarget.classList.add('active');
            }

            function actualizarCamposTabulares() {
                const pat = document.getElementById("patologia").value;
                const input = document.getElementById("variablesInput");
                if (pat === "diabetes") input.placeholder = "Ej: 148, 72, 35, 0, 0, 33.6, 0.627, 50";
                else if (pat === "cardio") input.placeholder = "Ej: 63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1";
                else input.placeholder = "Introduce 30 valores separados por comas...";
            }

            async function enviarTabular() {
                const token = document.getElementById("token").value;
                const patologia = document.getElementById("patologia").value;
                const vars = document.getElementById("variablesInput").value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                try {
                    const response = await fetch('/predecir/tabular', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                        body: JSON.stringify({ patologia, variables: vars })
                    });
                    const data = await response.json();
                    if (!response.ok) { alert("Error: " + data.detail); return; }
                    document.getElementById("resultadoTabular").style.display = "block";
                    document.getElementById("res-tab-diag").innerText = data.diagnostico;
                    document.getElementById("res-tab-prob").innerText = data.probabilidad_porcentaje;
                } catch (e) { alert("Network error: " + e); }
            }

            async function enviarTriaje() {
                const token = document.getElementById("token").value;
                const sintomas_texto = document.getElementById("sintomasInput").value;
                if (!sintomas_texto.trim()) return;
                try {
                    const response = await fetch('/triaje/sintomas', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                        body: JSON.stringify({ sintomas_texto })
                    });
                    const data = await response.json();
                    if (!response.ok) { alert("Error: " + data.detail); return; }
                    document.getElementById("resultadoTriaje").style.display = "block";
                    const badge = document.getElementById("badge-urgencia");
                    badge.style.backgroundColor = data.nivel_color === "ROJO" ? "#f8d7da" : data.nivel_color === "AMARILLO" ? "#fff3cd" : "#d4edda";
                    badge.style.color = data.nivel_color === "ROJO" ? "#721c24" : data.nivel_color === "AMARILLO" ? "#856404" : "#155724";
                    badge.innerText = "NIVEL DE URGENCIA: " + data.urgencia;
                    document.getElementById("res-nlp-esp").innerText = data.especialidad_sugerida;
                    document.getElementById("res-nlp-rec").innerText = currentLang === 'es' ? data.recomendacion_es : data.recomendacion_en;
                } catch (e) { alert("Network error: " + e); }
            }
        </script>
    </body>
    </html>
    """

# ==========================================
# 6. ENDPOINTS DE LA API
# ==========================================
@app.post("/predecir/tabular")
def predecir_tabular(datos: PeticionTabular, token: str = Depends(verificar_token)):
    if datos.patologia not in modelos_patologias:
        raise HTTPException(status_code=400, detail="Patología no soportada.")
    modelo = modelos_patologias[datos.patologia]
    if len(datos.variables) != modelo.layer1.in_features:
        raise HTTPException(status_code=400, detail=f"Requiere {modelo.layer1.in_features} variables.")
    
    tensor_entrada = torch.tensor([datos.variables], dtype=torch.float32)
    with torch.no_grad():
        probabilidad = modelo(tensor_entrada).item()
    es_riesgo = probabilidad >= 0.5
    registrar_auditoria("EXITOSO_TABULAR", f"{datos.patologia} | Prob: {round(probabilidad, 4)}")
    return {
        "probabilidad_porcentaje": round(probabilidad * 100, 2),
        "diagnostico": "Alto Riesgo / High Risk" if es_riesgo else "Normal / Healthy",
        "status": "ALERTA" if es_riesgo else "NORMAL"
    }

@app.post("/triaje/sintomas")
def procesar_triaje(datos: PeticionTriaje, token: str = Depends(verificar_token)):
    resultado = analizar_sintomas_nlp(datos.sintomas_texto)
    registrar_auditoria("EXITOSO_TRIAJE_NLP", f"Texto procesado exitosamente.")
    return resultado
