import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher, HtmlDiff
from dataclasses import dataclass, asdict
import hashlib
import logging

# Imports opcionais com tratamento de erro
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly não instalado. Gráficos não estarão disponíveis.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.error("PyMuPDF não instalado. Por favor, instale com: pip install PyMuPDF")

try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("SentenceTransformers não instalado. Apenas comparação rápida estará disponível.")

# Configuração da página
st.set_page_config(
    page_title="BurlaCheck - Detector de Alterações em Bulas",
    page_icon="🥼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SectionComparison:
    """Classe para armazenar resultado da comparação de uma seção"""
    similarity_score: float
    assessment: str
    text_old: str
    text_new: str
    method_used: str

@dataclass
class BulaComparison:
    """Classe para armazenar resultado completo da comparação de uma bula"""
    product_id: str
    file_current: str
    file_reference: str
    sections: Dict[str, SectionComparison]
    overall_assessment: str
    processing_time: float
    pages_filtered: Dict[str, int]
    bula_index: int
    bula_name: str

@dataclass
class MultiBulaComparison:
    """Armazena resultado da comparação de múltiplas bulas"""
    file_reference: str
    file_current: str
    total_bulas_reference: int
    total_bulas_current: int
    comparisons: List[BulaComparison]
    unmatched_bulas: Dict[str, List[str]]
    processing_time: float

class StreamlitBulaExtractor:
    """Versão adaptada do extrator para Streamlit com suporte a múltiplas bulas"""
    
    STANDARD_SECTIONS = {
        "1": "PARA QUE ESTE MEDICAMENTO É INDICADO",
        "2": "COMO ESTE MEDICAMENTO FUNCIONA",
        "3": "QUANDO NÃO DEVO USAR ESTE MEDICAMENTO",
        "4": "O QUE DEVO SABER ANTES DE USAR ESTE MEDICAMENTO",
        "5": "ONDE, COMO E POR QUANTO TEMPO POSSO GUARDAR ESTE MEDICAMENTO",
        "6": "COMO DEVO USAR ESTE MEDICAMENTO",
        "7": "O QUE DEVO FAZER QUANDO EU ME ESQUECER DE USAR ESTE MEDICAMENTO",
        "8": "QUAIS OS MALES QUE ESTE MEDICAMENTO PODE ME CAUSAR",
        "9": "O QUE FAZER SE ALGUÉM USAR UMA QUANTIDADE MAIOR DO QUE A INDICADA DESTE MEDICAMENTO"
    }
    
    THRESHOLDS = {
        'excellent': 0.95,
        'good': 0.85,
        'attention': 0.70,
        'critical': 0.50
    }
    
    FILTER_PATTERNS = {
        'processo_anvisa': r'\d{7}/\d{2}-\d',
        'texto_bula': r'de\s+texto\s+de\s+bula'
    }
    
    def __init__(self, use_semantic_comparison: bool = True):
        self.use_semantic_comparison = use_semantic_comparison and TRANSFORMERS_AVAILABLE
        self.semantic_model = None
        self._cache = {}
        
        if self.use_semantic_comparison:
            self.semantic_model = self._load_semantic_model()
    
    @st.cache_resource
    def _load_semantic_model(_self):
        """Carrega modelo semântico com cache do Streamlit"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        try:
            with st.spinner("Carregando modelo de análise semântica..."):
                return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            st.error(f"Erro ao carregar modelo semântico: {e}")
            return None
    
    def _should_filter_page(self, page_text: str) -> tuple[bool, str]:
        """Verifica se uma página deve ser filtrada"""
        page_text_lower = page_text.lower()
        
        if re.search(self.FILTER_PATTERNS['processo_anvisa'], page_text):
            return True, "Padrão processo Anvisa"
        
        if re.search(self.FILTER_PATTERNS['texto_bula'], page_text_lower):
            return True, "Texto administrativo"
        
        return False, ""
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, filename: str) -> tuple[str, int]:
        """Extrai texto de PDF filtrando páginas indesejadas"""
        if not PYMUPDF_AVAILABLE:
            st.error("PyMuPDF não está instalado. Não é possível processar PDFs.")
            return "", 0
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            try:
                text = ""
                total_pages = 0
                filtered_pages = 0
                
                with fitz.open(tmp_path) as pdf:
                    total_pages = len(pdf)
                    
                    for page_num, page in enumerate(pdf):
                        page_text = page.get_text()
                        
                        should_filter, filter_reason = self._should_filter_page(page_text)
                        
                        if should_filter:
                            filtered_pages += 1
                            logger.info(f"📄 Página {page_num + 1} de {filename} filtrada: {filter_reason}")
                        else:
                            text += page_text
                
                if filtered_pages > 0:
                    st.info(f"📋 {filename}: {filtered_pages} de {total_pages} páginas filtradas")
                
                text = re.sub(r'\s+', ' ', text.strip())
                return text, filtered_pages
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Erro ao extrair texto de {filename}: {e}")
            return "", 0
    
    def split_multiple_bulas(self, text: str) -> List[Tuple[str, str]]:
        """Divide o texto em múltiplas bulas baseado na presença da palavra 'composição'"""
        bulas = []
        
        composicao_pattern = r'composiç[aã]o'
        composicao_matches = list(re.finditer(composicao_pattern, text, re.IGNORECASE))
        
        if not composicao_matches:
            bula_name = self._extract_bula_name(text)
            return [(bula_name or "Bula 1", text)]
        
        start_points = [match.start() for match in composicao_matches]
        end_points = start_points[1:] + [len(text)]
        
        bula_counter = 1
        
        for i in range(len(start_points)):
            start = start_points[i]
            end = end_points[i]
            bula_text = text[start:end].strip()
            
            if bula_text:
                bula_name = self._extract_bula_name(bula_text) or f"Bula {bula_counter}"
                bulas.append((bula_name, bula_text))
                bula_counter += 1
        
        logger.info(f"🔍 Identificadas {len(bulas)} bulas no documento")
        return bulas
    
    def _extract_bula_name(self, text: str) -> Optional[str]:
        """Extrai o nome/identificação da bula do texto"""
        patterns = [
            r'([A-ZÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝ]+(?:\s+[A-ZÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝ]+)*)\s*(?:®|™)?\s*composição',
            r'^([A-ZÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝ\s]{3,50}?)(?:\s+\d|\s+composição|\s+indicações)',
            r'medicamento:\s*([A-Za-zÀ-ÿ\s]{3,50}?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                name = re.sub(r'[^\w\s\-]', '', name)
                if len(name) > 3 and len(name) < 50:
                    return name.title()
        
        return None
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extrai seções do texto da bula"""
        sections = {}
        
        composition = self._extract_composition(text)
        if composition:
            sections["COMPOSIÇÃO"] = composition
        
        numbered_sections = self._extract_numbered_sections(text)
        sections.update(numbered_sections)
        
        return sections
    
    def _extract_composition(self, text: str) -> Optional[str]:
        """Extrai seção de composição"""
        match = re.search(r'COMPOSIÇ[AÃ]O', text, re.IGNORECASE)
        if not match:
            return None
        
        start = match.end()
        window = text[start:start+2000]
        
        stop_match = re.search(r'((?:[IVXLCDM]+|\d+)\s*[-.:])', window)
        if stop_match:
            return window[:stop_match.start()].strip()
        
        cap_match = re.search(r'([A-ZÁÉÄÂÇÌÖÄÇ]{3,}\s*){2,}', window)
        if cap_match:
            return window[:cap_match.start()].strip()
        
        return window.strip()
    
    def _extract_numbered_sections(self, text: str) -> Dict[str, str]:
        """Extrai seções numeradas"""
        pattern = r"((?:[IVXLCDM]+|\d+)\s*[-.:][\s]*[A-ZÁÉÄÂÇÌÖÄÇ\s\-]+)"
        splits = re.split(pattern, text, flags=re.IGNORECASE)
        
        standard_sections = {}
        fragments = []
        
        i = 1
        while i < len(splits):
            if i % 2 == 1:
                title = splits[i].strip()
                content = splits[i + 1].strip() if i + 1 < len(splits) else ""
                
                section_number = self._extract_section_number(title)
                position = sum(len(splits[j]) for j in range(i))
                
                fragments.append({
                    'title': title,
                    'content': content,
                    'section_number': section_number,
                    'position': position,
                    'is_standard': section_number in self.STANDARD_SECTIONS if section_number else False
                })
            i += 1
        
        fragments.sort(key=lambda x: x['position'])
        
        current_standard_section = None
        accumulated_content = ""
        
        for fragment in fragments:
            if fragment['is_standard']:
                if current_standard_section and accumulated_content:
                    if current_standard_section in standard_sections:
                        standard_sections[current_standard_section] += "\n\n" + accumulated_content
                    else:
                        standard_sections[current_standard_section] = accumulated_content
                    accumulated_content = ""
                
                section_key = f"{fragment['section_number']}. {self.STANDARD_SECTIONS[fragment['section_number']]}"
                current_standard_section = section_key
                standard_sections[section_key] = fragment['content']
                
            else:
                if current_standard_section:
                    accumulated_content += f"\n\n{fragment['title']}\n{fragment['content']}" if accumulated_content else f"{fragment['title']}\n{fragment['content']}"
        
        if current_standard_section and accumulated_content:
            if current_standard_section in standard_sections:
                standard_sections[current_standard_section] += "\n\n" + accumulated_content
            else:
                standard_sections[current_standard_section] = accumulated_content
        
        return standard_sections
    
    def _extract_section_number(self, title: str) -> Optional[str]:
        """Extrai número da seção do título"""
        match = re.match(r"(\d+)", title.strip())
        if match:
            return match.group(1)
        
        roman_match = re.match(r"([IVXLCDM]+)", title.strip(), re.IGNORECASE)
        if roman_match:
            roman = roman_match.group(1).upper()
            roman_to_int = {
                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
            }
            if roman in roman_to_int:
                return str(roman_to_int[roman])
        
        return None
    
    def compare_sections(self, sections_old: Dict[str, str], 
                        sections_new: Dict[str, str]) -> Dict[str, SectionComparison]:
        """Compara seções entre duas versões"""
        comparisons = {}
        
        for title, old_text in sections_old.items():
            new_text = self._find_matching_section(title, sections_new)
            
            if new_text is not None:
                comparison = self._compare_text_pair(old_text, new_text)
                comparisons[title] = comparison
            else:
                comparisons[title] = SectionComparison(
                    similarity_score=0.0,
                    assessment="❌ Seção removida",
                    text_old=old_text,
                    text_new="",
                    method_used="none"
                )
        
        for title, new_text in sections_new.items():
            if title not in sections_old:
                old_text = self._find_matching_section(title, sections_old)
                if old_text is None:
                    comparisons[title] = SectionComparison(
                        similarity_score=0.0,
                        assessment="➕ Seção adicionada",
                        text_old="",
                        text_new=new_text,
                        method_used="none"
                    )
        
        return comparisons
    
    def _find_matching_section(self, title: str, sections: Dict[str, str]) -> Optional[str]:
        """Encontra seção correspondente pelo número ou título similar"""
        if title in sections:
            return sections[title]
        
        match = re.match(r"(\d+)", title)
        if match:
            num = match.group(1)
            for sect_title, content in sections.items():
                if sect_title.startswith(f"{num}."):
                    return content
        
        return None
    
    def _compare_text_pair(self, text1: str, text2: str) -> SectionComparison:
        """Compara par de textos usando método mais apropriado"""
        if not text1 or not text2:
            return SectionComparison(0.0, "❌ Texto vazio", text1, text2, "none")
        
        if self.use_semantic_comparison and self.semantic_model:
            try:
                embeddings1 = self.semantic_model.encode([text1])
                embeddings2 = self.semantic_model.encode([text2])
                similarity = float(util.cos_sim(embeddings1, embeddings2)[0][0])
                method = "semantic"
            except Exception as e:
                logger.warning(f"Erro na comparação semântica: {e}")
                similarity = SequenceMatcher(None, text1, text2).ratio()
                method = "textual"
        else:
            similarity = SequenceMatcher(None, text1, text2).ratio()
            method = "textual"
        
        assessment = self._assess_similarity(similarity)
        
        return SectionComparison(
            similarity_score=similarity,
            assessment=assessment,
            text_old=text1,
            text_new=text2,
            method_used=method
        )
    
    def _assess_similarity(self, score: float) -> str:
        """Avalia score de similaridade"""
        if score >= self.THRESHOLDS['excellent']:
            return "✅ Excelente - Sem alterações significativas"
        elif score >= self.THRESHOLDS['good']:
            return "🟢 Boa - Alterações menores"
        elif score >= self.THRESHOLDS['attention']:
            return "🟡 Atenção - Alterações moderadas"
        elif score >= self.THRESHOLDS['critical']:
            return "🟠 Crítico - Alterações significativas"
        else:
            return "🔴 Muito crítico - Alterações substanciais"
    
    def compare_multiple_bulas(self, pdf_bytes_ref: bytes, filename_ref: str,
                             pdf_bytes_cur: bytes, filename_cur: str) -> MultiBulaComparison:
        """Compara múltiplas bulas entre dois PDFs"""
        start_time = time.time()
        
        # Extrai texto dos PDFs
        text_ref, filtered_ref = self.extract_text_from_pdf(pdf_bytes_ref, filename_ref)
        text_cur, filtered_cur = self.extract_text_from_pdf(pdf_bytes_cur, filename_cur)
        
        # Divide em múltiplas bulas
        bulas_ref = self.split_multiple_bulas(text_ref)
        bulas_cur = self.split_multiple_bulas(text_cur)
        
        comparisons = []
        unmatched_bulas = {"reference": [], "current": []}
        
        # Compara bulas na ordem
        max_bulas = max(len(bulas_ref), len(bulas_cur))
        
        for i in range(max_bulas):
            if i < len(bulas_ref) and i < len(bulas_cur):
                # Ambas existem - compara
                ref_name, ref_text = bulas_ref[i]
                cur_name, cur_text = bulas_cur[i]
                
                comparison = self._compare_single_bula(
                    ref_text, cur_text, ref_name, cur_name,
                    filename_ref, filename_cur, i,
                    {"reference": filtered_ref, "current": filtered_cur}
                )
                comparisons.append(comparison)
                
            elif i < len(bulas_ref):
                # Só existe na referência
                ref_name, _ = bulas_ref[i]
                unmatched_bulas["reference"].append(ref_name)
                
            elif i < len(bulas_cur):
                # Só existe na atual
                cur_name, _ = bulas_cur[i]
                unmatched_bulas["current"].append(cur_name)
        
        processing_time = time.time() - start_time
        
        return MultiBulaComparison(
            file_reference=filename_ref,
            file_current=filename_cur,
            total_bulas_reference=len(bulas_ref),
            total_bulas_current=len(bulas_cur),
            comparisons=comparisons,
            unmatched_bulas=unmatched_bulas,
            processing_time=processing_time
        )
    
    def _compare_single_bula(self, text_ref: str, text_cur: str, 
                           name_ref: str, name_cur: str,
                           file_ref: str, file_cur: str, index: int,
                           pages_filtered: dict) -> BulaComparison:
        """Compara uma bula individual"""
        start_time = time.time()
        
        sections_ref = self.extract_sections(text_ref)
        sections_cur = self.extract_sections(text_cur)
        
        section_comparisons = self.compare_sections(sections_ref, sections_cur)
        
        # Avaliação geral
        scores = [comp.similarity_score for comp in section_comparisons.values() 
                 if comp.similarity_score is not None]
        
        if scores:
            avg_score = sum(scores) / len(scores)
            overall_assessment = self._assess_similarity(avg_score)
        else:
            overall_assessment = "⚠️ Não foi possível avaliar"
        
        processing_time = time.time() - start_time
        
        return BulaComparison(
            product_id=f"{name_ref}_{index}",
            file_current=file_cur,
            file_reference=file_ref,
            sections=section_comparisons,
            overall_assessment=overall_assessment,
            processing_time=processing_time,
            pages_filtered=pages_filtered,
            bula_index=index,
            bula_name=name_ref or name_cur or f"Bula {index + 1}"
        )

def generate_html_diff(text1: str, text2: str, title1: str = "Anterior", title2: str = "Atual") -> str:
    """Gera diff HTML entre dois textos"""
    differ = HtmlDiff(wrapcolumn=80)
    
    # Divide em linhas para melhor visualização
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    html_diff = differ.make_file(lines1, lines2, title1, title2)
    
    # Adiciona estilos customizados
    #custom_styles = """
   # Adiciona estilos customizados para modo dark - todas as cores legíveis
    custom_styles = '''
<style>
.diff_header {
    background-color: #2d3748; 
    color: #e2e8f0;
    padding: 10px; 
    font-weight: bold;
    border: 1px solid #4a5568;
}
.diff_next {
    background-color: #4a5568;
    color: #e2e8f0;
}
.diff_add {
    background-color: #90ee90;
    color: #1a202c;
    border-left: 3px solid #22543d;
}
.diff_chg {
    background-color: #fff2a8;
    color: #1a202c;
    border-left: 3px solid #b7791f;
}
.diff_sub {
    background-color: #ffb3b3;
    color: #1a202c;
    border-left: 3px solid #c53030;
}
table.diff {
    font-family: courier; 
    border: 1px solid #4a5568;
    background-color: #1a202c;
    color: #e2e8f0;
}
.diff_header {
    text-align: center;
}
td.diff_header {
    text-align: right; 
    padding-right: 10px;
}
table.diff td {
    color: #e2e8f0;
    padding: 8px;
    border: 1px solid #4a5568;
}
table.diff td.diff_add {
    color: #1a202c;
    font-weight: 600;
}
table.diff td.diff_chg {
    color: #1a202c;
    font-weight: 600;
}
table.diff td.diff_sub {
    color: #1a202c;
    font-weight: 600;
}
table.diff th {
    background-color: #2d3748;
    color: #e2e8f0;
    padding: 10px;
    border: 1px solid #4a5568;
}
</style>
'''


   # """
    
    # Insere estilos no HTML
    html_diff = html_diff.replace('<head>', f'<head>{custom_styles}')
    
    return html_diff

def display_multi_bula_results(result: MultiBulaComparison):
    """Exibe resultados da comparação de múltiplas bulas"""
    try:
        st.header("📊 Resultados da Análise Multi-Bula")
        
        # Resumo geral
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bulas Referência", result.total_bulas_reference)
        with col2:
            st.metric("Bulas Atuais", result.total_bulas_current)
        with col3:
            st.metric("Comparações", len(result.comparisons))
        with col4:
            st.metric("Tempo Processamento", f"{result.processing_time:.2f}s")
        
        # Alerta para bulas não correspondentes
        if result.unmatched_bulas['reference'] or result.unmatched_bulas['current']:
            st.warning("⚠️ Bulas sem correspondência encontradas:")
            
            if result.unmatched_bulas['reference']:
                st.write(f"**Apenas em {result.file_reference}:**")
                for bula in result.unmatched_bulas['reference']:
                    st.write(f"- {bula}")
            
            if result.unmatched_bulas['current']:
                st.write(f"**Apenas em {result.file_current}:**")
                for bula in result.unmatched_bulas['current']:
                    st.write(f"- {bula}")
        
        # Resultados por bula
        if result.comparisons:
            st.subheader("📋 Análise Detalhada por Bula")
            
            for i, comparison in enumerate(result.comparisons):
                display_single_bula_result(comparison, i)
        
    except Exception as e:
        st.error(f"❌ Erro ao exibir resultados: {e}")

def display_single_bula_result(result: BulaComparison, bula_index: int):
    """Exibe resultado de uma bula individual"""
    try:
        bula_suffix = f"_bula_{bula_index}"
        
        with st.expander(f"📖 {result.bula_name} - {result.overall_assessment}", expanded=False):
            
            # Métricas da bula
            col1, col2, col3 = st.columns(3)
            
            with col1:
                scores = [comp.similarity_score for comp in result.sections.values() 
                         if comp.similarity_score is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    st.metric("Similaridade Média", f"{avg_score:.2%}")
            
            with col2:
                st.metric("Seções Analisadas", len(result.sections))
            
            with col3:
                st.metric("Tempo Processamento", f"{result.processing_time:.2f}s")
            
            # Controles de visualização
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_only_changes_key = f"show_only_changes{bula_suffix}"
                if show_only_changes_key not in st.session_state:
                    st.session_state[show_only_changes_key] = False
                
                st.session_state[show_only_changes_key] = st.checkbox(
                    "🔍 Mostrar apenas alterações",
                    value=st.session_state[show_only_changes_key],
                    key=f"checkbox_changes{bula_suffix}"
                )
            
            with col2:
                show_text_diff_key = f"show_text_diff{bula_suffix}"
                if show_text_diff_key not in st.session_state:
                    st.session_state[show_text_diff_key] = False
                
                st.session_state[show_text_diff_key] = st.checkbox(
                    "📝 Mostrar diferenças textuais",
                    value=st.session_state[show_text_diff_key],
                    key=f"checkbox_text{bula_suffix}"
                )
            
            with col3:
                show_html_diff_key = f"show_html_diff{bula_suffix}"
                if show_html_diff_key not in st.session_state:
                    st.session_state[show_html_diff_key] = False
                
                st.session_state[show_html_diff_key] = st.checkbox(
                                            "Visualização Detalhada",
                                            value=st.session_state.get(show_html_diff_key, True),  # True é o valor padrão
                                            key=f"checkbox_html{bula_suffix}"
                                            )
            
            # Detalhes das seções
            sections_container = st.container()
            
            with sections_container:
                sections_list = list(result.sections.items())
                
                for idx, (section_title, comparison) in enumerate(sections_list):
                    try:
                        # Filtra baseado na opção selecionada
                        if (st.session_state[f"show_only_changes{bula_suffix}"] and 
                            comparison.similarity_score is not None and 
                            comparison.similarity_score >= 0.85):
                            continue
                        
                        section_key = f"section_{result.bula_index}_{idx}"
                        
                        with st.expander(f"{section_title} - {comparison.assessment}", 
                                       expanded=False):
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if comparison.similarity_score is not None:
                                    st.metric("Similaridade", f"{comparison.similarity_score:.2%}")
                            with col2:
                                st.metric("Método", comparison.method_used)
                            with col3:
                                st.write(f"**Status:** {comparison.assessment}")
                            
                            # Mostrar diferenças textuais apenas se habilitado
                            if (st.session_state[f"show_text_diff{bula_suffix}"] and 
                                comparison.text_old and comparison.text_new):
                                
                                st.subheader("Comparação Textual")
                                
                                # Verifica se deve mostrar diff HTML
                                if st.session_state[f"show_html_diff{bula_suffix}"]:
                                    try:
                                        max_chars = 3000
                                        text_old_limited = (comparison.text_old[:max_chars] + "..." 
                                                          if len(comparison.text_old) > max_chars 
                                                          else comparison.text_old)
                                        text_new_limited = (comparison.text_new[:max_chars] + "..." 
                                                          if len(comparison.text_new) > max_chars 
                                                          else comparison.text_new)
                                        
                                        html_diff = generate_html_diff(
                                            text_old_limited, 
                                            text_new_limited,
                                            "Versão Anterior",
                                            "Versão Atual"
                                        )
                                        
                                        st.subheader("🔍 Diferenças Visuais")
                                        st.markdown("""
                                        **Legenda:**
                                        - 🟢 **Verde**: Texto adicionado
                                        - 🔴 **Vermelho**: Texto removido  
                                        - 🟡 **Amarelo**: Texto modificado
                                        """)
                                        
                                        components.html(html_diff, height=600, scrolling=True)
                                        
                                        if (len(comparison.text_old) > max_chars or 
                                            len(comparison.text_new) > max_chars):
                                            st.warning(f"⚠️ Texto limitado a {max_chars} caracteres para melhor performance")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Erro ao gerar diff visual: {e}")
                                        st.info("💡 Tente usar a comparação textual simples")
                                
                                else:
                                    tab1, tab2 = st.tabs(["📄 Versão Anterior", "📄 Versão Atual"])
                                    
                                    with tab1:
                                        old_text_display = (comparison.text_old[:2000] + "..." 
                                                          if len(comparison.text_old) > 2000 
                                                          else comparison.text_old)
                                        st.text_area(
                                            "Versão Anterior", 
                                            old_text_display, 
                                            height=300, 
                                            key=f"old_text_{section_key}",
                                            disabled=True
                                        )
                                    
                                    with tab2:
                                        new_text_display = (comparison.text_new[:2000] + "..." 
                                                          if len(comparison.text_new) > 2000 
                                                          else comparison.text_new)
                                        st.text_area(
                                            "Versão Atual", 
                                            new_text_display, 
                                            height=300, 
                                            key=f"new_text_{section_key}",
                                            disabled=True
                                        )
                                
                                # Botão para ver texto completo se necessário
                                if (len(comparison.text_old) > 2000 or 
                                    len(comparison.text_new) > 2000):
                                    
                                    if st.button(f"📄 Ver texto completo", 
                                               key=f"full_text_{section_key}"):
                                        
                                        st.markdown("---")
                                        st.subheader("Texto Completo - Versão Anterior")
                                        st.text(comparison.text_old)
                                        
                                        st.subheader("Texto Completo - Versão Atual")
                                        st.text(comparison.text_new)
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao processar seção '{section_title}': {e}")
                        continue
    
    except Exception as e:
        st.error(f"❌ Erro ao exibir resultados da bula: {e}")

def export_multi_bula_results(result: MultiBulaComparison):
    """Permite exportar resultados de múltiplas bulas"""
    st.subheader("💾 Exportar Resultados Multi-Bula")
    
    # Converte para JSON
    result_dict = asdict(result)
    json_str = json.dumps(result_dict, ensure_ascii=False, indent=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Baixar JSON Completo",
            data=json_str,
            file_name=f"comparacao_multi_bulas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Cria relatório resumido em texto
        report = f"""RELATÓRIO DE COMPARAÇÃO DE MÚLTIPLAS BULAS
==========================================

Arquivos Comparados:
- Referência: {result.file_reference} ({result.total_bulas_reference} bula(s))
- Atual: {result.file_current} ({result.total_bulas_current} bula(s))

Tempo de Processamento Total: {result.processing_time:.2f}s
Comparações Realizadas: {len(result.comparisons)}

BULAS SEM CORRESPONDÊNCIA:
-------------------------"""
        
        if result.unmatched_bulas['reference']:
            report += f"\nApenas em {result.file_reference}:"
            for bula in result.unmatched_bulas['reference']:
                report += f"\n- {bula}"
        
        if result.unmatched_bulas['current']:
            report += f"\nApenas em {result.file_current}:"
            for bula in result.unmatched_bulas['current']:
                report += f"\n- {bula}"
        
        report += f"\n\nDETALHES POR BULA:\n" + "="*50
        
        for comparison in result.comparisons:
            report += f"\n\n{comparison.bula_name.upper()}\n" + "-"*40
            report += f"\nAvaliação Geral: {comparison.overall_assessment}"
            report += f"\nSeções Analisadas: {len(comparison.sections)}"
            
            scores = [comp.similarity_score for comp in comparison.sections.values() 
                     if comp.similarity_score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                report += f"\nSimilaridade Média: {avg_score:.2%}"
            
            report += f"\n\nSeções:"
            for section, comp in comparison.sections.items():
                report += f"\n  - {section}:"
                if comp.similarity_score is not None:
                    report += f"\n    Similaridade: {comp.similarity_score:.2%}"
                report += f"\n    Avaliação: {comp.assessment}"
                report += f"\n    Método: {comp.method_used}"
        
        st.download_button(
            label="📃 Baixar Relatório",
            data=report,
            file_name=f"relatorio_multi_bulas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def main():
    """Interface principal"""
    try:
        # Header
        st.title("🥼 BurlaCheck Multi-Bula")
        st.subheader("Detector de Alterações em Bulas Digitalizadas com Suporte a Múltiplas Bulas")
        
        st.markdown("""
        Esta ferramenta utiliza técnicas de processamento de linguagem natural para detectar 
        alterações entre versões de bulas de medicamentos, conforme a **RDC nº 885/2024** da Anvisa.
               
        **Recursos avançados:**
        - Consolidação automática de seções não padrão nas seções regulamentares correspondentes
        - Filtragem inteligente de páginas administrativas
        - Identificação automática de bulas usando o padrão da seção "composição"
        """)
        
        # Verifica dependências
        missing_deps = []
        if not PYMUPDF_AVAILABLE:
            missing_deps.append("PyMuPDF (pip install PyMuPDF)")
        
        if missing_deps:
            st.error("❌ Dependências obrigatórias não instaladas:")
            for dep in missing_deps:
                st.write(f"- {dep}")
            st.stop()
        
        # Sidebar com configurações
        st.sidebar.header("⚙️ Configurações")
        
        # Verifica se transformers está disponível
        if TRANSFORMERS_AVAILABLE:
            use_semantic = st.sidebar.selectbox(
                "Método de Comparação",
                options=[True, False],
                format_func=lambda x: "🧠 Análise Semântica (Recomendado)" if x else "⚡ Comparação Rápida",
                index=0
            )
        else:
            use_semantic = False
            st.sidebar.warning("⚠️ Análise semântica não disponível")
            st.sidebar.info("Para usar análise semântica, instale: pip install sentence-transformers")
        
        st.sidebar.info("""
        **Análise Semântica:** Detecta alterações mesmo quando 
        o texto foi reescrito com palavras diferentes.
        
        **Comparação Rápida:** Mais rápida, mas pode não detectar 
        alterações sutis no significado.
        """)
        
        # Informações sobre múltiplas bulas
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Múltiplas Bulas")
        st.sidebar.info("""
        **Detecção automática:**
        - Identifica múltiplas bulas no mesmo PDF
        - **Padrão:** O texto de cada bula é delimitado por uma nova seção "composição"
        - Comparação ordenada: 1ª com 1ª, 2ª com 2ª, etc.
        - Alertas para bulas sem correspondência
        """)
        
        # Informações sobre consolidação
        st.sidebar.markdown("---")
        st.sidebar.subheader("📃 Consolidação de Seções")
        st.sidebar.info("""
        O sistema agora consolida automaticamente:
        - Subseções não regulamentares
        - Anexos e apêndices  
        - Seções numeradas fora do padrão
        
        Tudo é integrado às 9 seções padrão da Anvisa.
        """)
        
        # Informações sobre filtragem
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filtragem de Páginas")
        st.sidebar.info("""
        **Páginas filtradas automaticamente:**
        - Padrão de processo: XXXXXXX/XX-X
        - Texto "de texto de bula"
        - Páginas administrativas da Anvisa
        
        Foco exclusivo no conteúdo da bula.
        """)
        
        # Upload de arquivos
        st.header("📂 Upload dos Arquivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Bula(s) de Referência")
            pdf1 = st.file_uploader(
                "Selecione o arquivo com bula(s) de referência (versão anterior/aprovada)",
                type=['pdf'],
                key='pdf1'
            )
            
            if pdf1:
                st.success(f"✅ {pdf1.name} carregado")
        
        with col2:
            st.subheader("📄 Bula(s) para Comparação")
            pdf2 = st.file_uploader(
                "Selecione o arquivo com bula(s) atual(is) para comparação",
                type=['pdf'],
                key='pdf2'
            )
            
            if pdf2:
                st.success(f"✅ {pdf2.name} carregado")
        
        # Botão de análise
        if pdf1 and pdf2:
            if st.button("🔄 Analisar Bulas", type="primary", use_container_width=True):
                
                if not PYMUPDF_AVAILABLE:
                    st.error("❌ PyMuPDF não está instalado. Instale com: pip install PyMuPDF")
                    return
                
                # Limpar session_state ao iniciar nova análise
                for key in list(st.session_state.keys()):
                    if any(prefix in key for prefix in ['expanded_', 'old_text_', 'new_text_', 'full_text_', 'show_only_changes', 'show_text_diff', 'show_html_diff', 'section_']):
                        del st.session_state[key]
                
                # Inicializa o extrator
                extractor = StreamlitBulaExtractor(use_semantic_comparison=use_semantic)
                
                try:
                    # Realiza a comparação de múltiplas bulas
                    with st.spinner("Analisando múltiplas bulas..."):
                        result = extractor.compare_multiple_bulas(
                            pdf1.getvalue(), pdf1.name,
                            pdf2.getvalue(), pdf2.name
                        )
                    
                    # Armazenar resultado no session_state
                    st.session_state.multi_analysis_result = result
                    
                except Exception as e:
                    st.error(f"❌ Erro durante a análise: {e}")
                    st.exception(e)
        
        # Exibir resultados apenas se existirem no session_state
        if hasattr(st.session_state, 'multi_analysis_result'):
            # Exibe resultados
            display_multi_bula_results(st.session_state.multi_analysis_result)
            
            # Opções de export
            export_multi_bula_results(st.session_state.multi_analysis_result)
        
        elif not (pdf1 and pdf2):
            st.info("📋 Carregue dois arquivos PDF para começar a análise")
        
        # Footer com informações
        st.markdown("---")
        st.markdown("""
        **Desenvolvido para atender à RDC nº 885/2024 da Anvisa**
        
        Esta ferramenta auxilia na fiscalização de bulas digitalizadas, detectando automaticamente 
        alterações que podem comprometer a segurança do paciente.
        
        ### 🔍 Como funciona a detecção de múltiplas bulas:
        1. **Identificação:** Procura pelo padrão "composição" no texto
        2. **Divisão:** Quando encontra uma nova seção "composição", identifica uma nova bula
        3. **Nomeação:** Extrai automaticamente o nome do medicamento de cada bula
        4. **Comparação:** Compara as bulas na ordem encontrada nos arquivos
        5. **Alertas:** Notifica sobre bulas que existem apenas em um dos arquivos
        
        ### 📃 Padrões de filtragem utilizados:
        - **Processos Anvisa:** Padrão XXXXXXX/XX-X (7 dígitos/2 dígitos-1 dígito)
        - **Texto administrativo:** Expressão "de texto de bula" (case-insensitive)
        """)
        
    except Exception as e:
        st.error(f"❌ Erro na aplicação: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
