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
# 3. MOTOR DE TRIAJE NLP BILINGÜE (ES / EN)
# ==========================================
def analizar_sintomas_nlp(texto: str):
    texto_lower = texto.lower()
    
    # Palabras clave en Español e Inglés
    criticos = [
        "pecho", "ahogo", "respirar", "infarto", "desmayo", "sangre", "súbito", "inconsciente", "parálisis",
        "chest", "breath", "heart attack", "fainting", "blood", "sudden", "unconscious", "paralysis"
    ]
    moderados = [
        "fiebre", "vómito", "dolor fuerte", "mareo", "infección", "fractura", "presión alta",
        "fever", "vomit", "strong pain", "dizziness", "infection", "fracture", "high blood pressure"
    ]
    
    score_urgencia = 1 
    especialidad = "Medicina General / General Practice"
    
    if any(palabra in texto_lower for palabra in criticos):
        score_urgencia = 3  
        especialidad = "Urgencias / Cardiología / UCI (Emergency / Cardiology)"
    elif any(palabra in texto_lower for palabra in moderados):
        score_urgencia = 2  
        especialidad = "Medicina Interna / Especialidad (Internal Medicine)"
    
    if "corazón" in texto_lower or "heart" in texto_lower or "palpitaciones" in texto_lower or "palpitations" in texto_lower:
        especialidad = "Cardiología / Cardiology"
    elif "hueso" in texto_lower or "bone" in texto_lower or "rodilla" in texto_lower or "knee" in texto_lower or "caída" in texto_lower or "fall" in texto_lower:
        especialidad = "Traumatología / Orthopedics"
    elif "cabeza" in texto_lower or "head" in texto_lower or "migraña" in texto_lower or "migraine" in texto_lower:
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
# 4. FASTAPI Y CONFIGURACIÓN
# ==========================================
app = FastAPI(
    title="Clinical Intelligence Platform (Hybrid Tabular + NLP)",
    description="Multilingual Medical Decision Support System (CDS)",
    version="4.1.0"
)

TOKEN_MEDICO_AUTORIZADO = os.getenv("TOKEN_MEDICO_AUTORIZADO", "hospital-med-token-2026-secure")
security = HTTPBearer()

def verificar_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TOKEN_MEDICO_AUTORIZADO:
        registrar_auditoria("FALLIDO_403", "Intento de acceso con Token Bearer inválido.")
        raise HTTPException(status_code=403, detail="Acceso denegado: Token inválido.")
    return credentials.credentials

class PeticionTabular(BaseModel):
    patologia: str
    variables: list[float]

class PeticionTriaje(BaseModel):
    sintomas_texto: str

# ==========================================
# 5. INTERFAZ WEB BILINGÜE
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
            
            /* Selector de Idioma */
            .lang-switcher { position: absolute; top: 25px; right: 30px; }
            .lang-btn { background: #e9ecef; border: 1px solid #ced4da; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; }
            .lang-btn:hover { background: #dee2e6; }

            h1 { color: #2c3e50; text-align: center; margin-bottom: 5px; }
            .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 25px; font-size: 14px; }
            
            .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 25px; }
            .tab-btn { flex: 1; padding: 12px; background: none; border: none; font-size: 16px; font-weight: bold; cursor: pointer; color: #7f8c8d; transition: 0.3s; }
            .tab-btn.active { color: #007bff; border-bottom: 3px solid #007bff; margin-bottom: -2px; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            .form-group { margin-bottom: 20px; }
            label { display: block; font-weight: bold; margin-bottom: 8px; color: #34495e; }
            input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
            textarea { resize: vertical; height: 100px; }
            .btn { width: 100%; padding: 12px; background-color: #007bff; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.3s; }
            .btn:hover { background-color: #0056b3; }
            
            .result-card { margin-top: 25px; padding: 20px; border-radius: 8px; display: none; background: #e9ecef; }
            .badge-rojo { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 10px; border-radius: 6px; font-weight: bold; }
            .badge-amarillo { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 10px; border-radius: 6px; font-weight: bold; }
            .badge-verde { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; padding: 10px; border-radius: 6px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="lang-switcher">
                <button class="lang-btn" onclick="toggleLanguage()" id="langBtn">🇬🇧 EN / 🇪🇸 ES</button>
            </div>

            <h1 id="ui-title">🏥 Centro de Inteligencia Clínica</h1>
            <p class="subtitle" id="ui-subtitle">Sistemas de Apoyo al Diagnóstico (CDS) - Tabular & NLP Bilingüe</p>
            
            <div class="form-group" style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <label for="token" id="lbl-token">🔑 Token de Autorización (Bearer Token):</label>
                <input type="password" id="token" value="hospital-med-token-2026-secure">
            </div>

            <div class="tabs">
                <button class="tab-btn active" onclick="cambiarTab(event, 'tab-tabular')" id="tab1-btn">1. Diagnóstico Tabular (Lab)</button>
                <button class="tab-btn" onclick="cambiarTab(event, 'tab-nlp')" id="tab2-btn">2. Triaje Inteligente NLP</button>
            </div>

            <!-- TAB 1 -->
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
                    <h3 style="margin-top:0;" id="res-tab-title">Resultado del Análisis Tabular</h3>
                    <p><strong><span id="lbl-diag">Diagnóstico</span>:</strong> <span id="res-tab-diag"></span></p>
                    <p><strong><span id="lbl-prob">Probabilidad de Riesgo</span>:</strong> <span id="res-tab-prob"></span>%</p>
                </div>
            </div>

            <!-- TAB 2 -->
            <div id="tab-nlp" class="tab-content">
                <div class="form-group">
                    <label for="sintomasInput" id="lbl-sintomas">Describe los síntomas (Español o Inglés):</label>
                    <textarea id="sintomasInput" placeholder="Ej: Dolor punzante en el pecho con dificultad para respirar / Sharp chest pain with shortness of breath..."></textarea>
                </div>

                <button class="btn" onclick="enviarTriaje()" id="btn-nlp-exec">Procesar Triaje con NLP</button>

                <div id="resultadoTriaje" class="result-card">
                    <h3 style="margin-top:0;" id="res-nlp-title">Evaluación de Triaje</h3>
                    <div id="badge-urgencia" style="margin-bottom: 12px;"></div>
                    <p><strong><span id="lbl-esp">Especialidad Sugerida</span>:</strong> <span id="res-nlp-esp"></span></p>
                    <p><strong><span id="lbl-rec">Protocolo Clínico</span>:</strong> <span id="res-nlp-rec"></span></p>
                </div>
            </div>
        </div>

        <script>
            let currentLang = 'es';

            const translations = {
                es: {
                    title: "🏥 Centro de Inteligencia Clínica",
                    subtitle: "Sistemas de Apoyo al Diagnóstico (CDS) - Tabular & NLP Bilingüe",
                    lblToken: "🔑 Token de Autorización (Bearer Token):",
                    tab1: "1. Diagnóstico Tabular (Lab)",
                    tab2: "2. Triaje Inteligente NLP",
                    lblPat: "Seleccionar Patología a Evaluar:",
                    lblVars: "Vector de Variables (Separadas por comas):",
                    btnTabExec: "Ejecutar Inferencia Tabular",
                    resTabTitle: "Resultado del Análisis Tabular",
                    lblDiag: "Diagnóstico",
                    lblProb: "Probabilidad de Riesgo",
                    lblSintomas: "Describe los síntomas (Español o Inglés):",
                    btnNlpExec: "Procesar Triaje con NLP",
                    resNlpTitle: "Evaluación de Triaje",
                    lblEsp: "Especialidad Sugerida",
                    lblRec: "Protocolo Clínico",
                    langBtnText: "🇬🇧 EN / 🇪🇸 ES"
                },
                en: {
                    title: "🏥 Clinical Intelligence Center",
                    subtitle: "Clinical Decision Support (CDS) - Tabular & Bilingual NLP",
                    lblToken: "🔑 Authorization Token (Bearer Token):",
                    tab1: "1. Tabular Diagnosis (Lab)",
                    tab2: "2. Smart NLP Triage",
                    lblPat: "Select Pathology to Evaluate:",
                    lblVars: "Variables Vector (Comma-separated):",
                    btnTabExec: "Run Tabular Inference",
                    resTabTitle: "Tabular Analysis Result",
                    lblDiag: "Diagnosis",
                    lblProb: "Risk Probability",
                    lblSintomas: "Describe symptoms (Spanish or English):",
                    btnNlpExec: "Process NLP Triage",
                    resNlpTitle: "Triage Evaluation",
                    lblEsp: "Suggested Specialty",
                    lblRec: "Clinical Protocol",
                    langBtnText: "🇪🇸 ES / 🇬🇧 EN"
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
                const textoVars = document.getElementById("variablesInput").value;
                const variables = textoVars.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                
                try {
                    const response = await fetch('/predecir/tabular', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                        body: JSON.stringify({ patologia, variables })
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
                if (!sintomas_texto.trim()) { alert("Por favor ingresa síntomas / Please enter symptoms"); return; }
                
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
                    badge.className = data.nivel_color === "ROJO" ? "badge-rojo" : data.nivel_color === "AMARILLO" ? "badge-amarillo" : "badge-verde";
                    badge.innerText = "URGENCY LEVEL: " + data.urgencia;
                    
                    document.getElementById("res-nlp-esp").innerText = data.especialidad_sugerida;
                    document.getElementById("res-nlp-rec").innerText = currentLang === 'es' ? data.recomendacion_es : data.recomendacion_en;
                } catch (e) { alert("Network error: " + e); }
            }
        </script>
    </body>
    </html>
    """

# ==========================================
# 6. ENDPOINTS
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
    registrar_auditoria("EXITOSO_TRIAJE_NLP", f"Texto: {datos.sintomas_texto[:30]}...")
    return resultado