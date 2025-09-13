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
    bula_index: int  # NOVO: Índice da bula no arquivo
    bula_name: str   # NOVO: Nome identificado da bula

@dataclass
class MultiBulaComparison:
    """NOVA CLASSE: Armazena resultado da comparação de múltiplas bulas"""
    file_reference: str
    file_current: str
    total_bulas_reference: int
    total_bulas_current: int
    comparisons: List[BulaComparison]
    unmatched_bulas: Dict[str, List[str]]  # Bulas sem correspondência
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
    
    # Padrões regex para filtrar páginas
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
        
        # Verifica padrão de processo Anvisa
        if re.search(self.FILTER_PATTERNS['processo_anvisa'], page_text):
            return True, "Padrão processo Anvisa"
        
        # Verifica texto "de texto de bula"
        if re.search(self.FILTER_PATTERNS['texto_bula'], page_text_lower):
            return True, "Texto administrativo"
        
        return False, ""
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, filename: str) -> tuple[str, int]:
        """Extrai texto de PDF filtrando páginas indesejadas"""
        if not PYMUPDF_AVAILABLE:
            st.error("PyMuPDF não está instalado. Não é possível processar PDFs.")
            return "", 0
            
        try:
            # Cria arquivo temporário
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
                        
                        # Verifica se deve filtrar esta página
                        should_filter, filter_reason = self._should_filter_page(page_text)
                        
                        if should_filter:
                            filtered_pages += 1
                            logger.info(f"📄 Página {page_num + 1} de {filename} filtrada: {filter_reason}")
                        else:
                            text += page_text
                
                # Log do resultado da filtragem
                if filtered_pages > 0:
                    st.info(f"📝 {filename}: {filtered_pages} de {total_pages} páginas filtradas")
                
                # Normaliza espaços em branco
                text = re.sub(r'\s+', ' ', text.strip())
                
                return text, filtered_pages
                
            finally:
                # Remove arquivo temporário
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Erro ao extrair texto de {filename}: {e}")
            return "", 0
    
    def split_multiple_bulas(self, text: str) -> List[Tuple[str, str]]:
        """
        Divide o texto em múltiplas bulas baseado na presença da palavra 'composição' no texto.
        
        Args:
            text (str): O texto completo do PDF após a filtragem de páginas.

        Returns:
            List[Tuple[str, str]]: Uma lista de tuplas (nome_bula, texto_bula)
        """
        bulas = []
        
        # Padrão para identificar "composição" (case insensitive)
        composicao_pattern = r'composiç[aã]o'
        
        # Encontra todas as posições de "composição"
        composicao_matches = list(re.finditer(composicao_pattern, text, re.IGNORECASE))
        
        if not composicao_matches:
            # Se não encontrou "composição", trata como bula única
            bula_name = self._extract_bula_name(text)
            return [(bula_name or "Bula 1", text)]
        
        # A primeira bula começa no primeiro "composição"
        start_points = [match.start() for match in composicao_matches]
        
        # A última bula termina no final do texto
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
        """
        NOVA FUNÇÃO: Extrai o nome/identificação da bula do texto
        """
        # Procura por padrões comuns de identificação de medicamentos
        patterns = [
            # Nome comercial seguido de composição
            r'([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÎÏÐÑÒÓÔÕÖ]+(?:\s+[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ]+)*)\s*(?:\®|\™)?\s*composição',
            
            # Nome em caixa alta no início
            r'^([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ\s]{3,50}?)(?:\s+\d|\s+composição|\s+indicações)',
            
            # Padrão "medicamento: NOME"
            r'medicamento:\s*([A-Za-zÀ-ÿ\s]{3,50}?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Limpa o nome (remove caracteres especiais, normaliza espaços)
                name = re.sub(r'\s+', ' ', name)
                name = re.sub(r'[^\w\s\-]', '', name)
                if len(name) > 3 and len(name) < 50:
                    return name.title()
        
        return None
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extrai seções do texto da bula"""
        sections = {}
        
        # Extrai composição
        composition = self._extract_composition(text)
        if composition:
            sections["COMPOSIÇÃO"] = composition
        
        # Extrai seções numeradas
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
        
        # Para na próxima seção numerada
        stop_match = re.search(r'((?:[IVXLCDM]+|\d+)\s*[-.:])', window)
        if stop_match:
            return window[:stop_match.start()].strip()
        
        # Para em texto em caixa alta
        cap_match = re.search(r'([A-ZÇÃÊÉÍÓÚÀÂÕ]{3,}\s*){2,}', window)
        if cap_match:
            return window[:cap_match.start()].strip()
        
        return window.strip()
    
    def _extract_numbered_sections(self, text: str) -> Dict[str, str]:
        """Extrai seções numeradas, consolidando conteúdo não padrão"""
        
        pattern = r"((?:[IVXLCDM]+|\d+)\s*[-.:][\s]*[A-ZÇÃÊÉÍÓÚÀÂÕ\s\-]+)"
        splits = re.split(pattern, text, flags=re.IGNORECASE)
        
        # Dicionário para armazenar seções padrão
        standard_sections = {}
        
        # Lista para armazenar todos os fragmentos encontrados com suas posições
        fragments = []
        
        i = 1
        while i < len(splits):
            if i % 2 == 1:  # Título
                title = splits[i].strip()
                content = splits[i + 1].strip() if i + 1 < len(splits) else ""
                
                # Extrai número da seção
                section_number = self._extract_section_number(title)
                
                # Calcula posição aproximada no texto original
                position = sum(len(splits[j]) for j in range(i))
                
                fragments.append({
                    'title': title,
                    'content': content,
                    'section_number': section_number,
                    'position': position,
                    'is_standard': section_number in self.STANDARD_SECTIONS if section_number else False
                })
            i += 1
        
        # Ordena fragments por posição no texto
        fragments.sort(key=lambda x: x['position'])
        
        # Processa fragments, consolidando não padrão nas seções padrão anteriores
        current_standard_section = None
        accumulated_content = ""
        
        for fragment in fragments:
            if fragment['is_standard']:
                # Se havia conteúdo acumulado, adiciona à seção padrão anterior
                if current_standard_section and accumulated_content:
                    if current_standard_section in standard_sections:
                        standard_sections[current_standard_section] += "\n\n" + accumulated_content
                    else:
                        standard_sections[current_standard_section] = accumulated_content
                    accumulated_content = ""
                
                # Define nova seção padrão atual
                section_key = f"{fragment['section_number']}. {self.STANDARD_SECTIONS[fragment['section_number']]}"
                current_standard_section = section_key
                standard_sections[section_key] = fragment['content']
                
                logger.debug(f"✅ Seção padrão encontrada: {section_key}")
            
            else:
                # Acumula conteúdo de seções não padrão
                if current_standard_section:
                    accumulated_content += f"\n\n{fragment['title']}\n{fragment['content']}" if accumulated_content else f"{fragment['title']}\n{fragment['content']}"
                    logger.debug(f"📄 Conteúdo não padrão acumulado para {current_standard_section}: {fragment['title'][:50]}...")
                else:
                    # Se não há seção padrão anterior, cria uma entrada temporária
                    logger.warning(f"⚠️ Conteúdo não padrão sem seção anterior: {fragment['title'][:50]}...")
        
        # Adiciona último conteúdo acumulado se existir
        if current_standard_section and accumulated_content:
            if current_standard_section in standard_sections:
                standard_sections[current_standard_section] += "\n\n" + accumulated_content
            else:
                standard_sections[current_standard_section] = accumulated_content
        
        return standard_sections
    
    def _extract_section_number(self, title: str) -> Optional[str]:
        """Extrai número da seção do título"""
        # Procura por números arábicos no início
        match = re.match(r"(\d+)", title.strip())
        if match:
            return match.group(1)
        
        # Procura por números romanos
        roman_match = re.match(r"([IVXLCDM]+)", title.strip(), re.IGNORECASE)
        if roman_match:
            roman = roman_match.group(1).upper()
            # Converte romano para arábico
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
        
        # Compara seções presentes na versão antiga
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
        
        # Verifica seções novas
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
        # Busca exata
        if title in sections:
            return sections[title]
        
        # Busca por número da seção
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
        
        # Usa comparação semântica se disponível e textos não são muito grandes
        if (self.use_semantic_comparison and self.semantic_model and 
            len(text1) < 5000 and len(text2) < 5000):
            
            try:
                with st.spinner("Analisando semanticamente..."):
                    emb1 = self.semantic_model.encode(text1)
                    emb2 = self.semantic_model.encode(text2)
                    score = util.cos_sim(emb1, emb2).item()
                method = "transformers"
            except Exception as e:
                st.warning(f"Erro na comparação semântica: {e}")
                score = SequenceMatcher(None, text1, text2).ratio()
                method = "difflib"
        else:
            # Usa difflib
            score = SequenceMatcher(None, text1, text2).ratio()
            method = "difflib"
        
        assessment = self._classify_change(score)
        
        return SectionComparison(
            similarity_score=round(score, 4),
            assessment=assessment,
            text_old=text1,
            text_new=text2,
            method_used=method
        )
    
    def _classify_change(self, score: float) -> str:
        """Classifica mudança baseada no score"""
        if score >= self.THRESHOLDS['excellent']:
            return "✅ Sem alteração relevante"
        elif score >= self.THRESHOLDS['good']:
            return "🟢 Alteração mínima"
        elif score >= self.THRESHOLDS['attention']:
            return "⚠️ Revisar - Alteração moderada"
        elif score >= self.THRESHOLDS['critical']:
            return "🟡 Alteração significativa"
        else:
            return "🔴 Possível alteração importante"
    
    def compare_single_bula(self, text_old: str, text_new: str, 
                           bula_name_old: str, bula_name_new: str,
                           file_old: str, file_new: str, 
                           bula_index: int, filtered_pages: Dict[str, int]) -> BulaComparison:
        """
        NOVA FUNÇÃO: Compara uma única bula
        """
        start_time = time.time()
        
        # Extrai seções
        with st.spinner(f"Identificando seções da {bula_name_old}..."):
            sections_old = self.extract_sections(text_old)
            sections_new = self.extract_sections(text_new)
        
        # Compara seções
        with st.spinner(f"Comparando seções da {bula_name_old}..."):
            comparisons = self.compare_sections(sections_old, sections_new)
        
        # Avaliação geral
        overall_assessment = self._calculate_overall_assessment(comparisons)
        
        processing_time = time.time() - start_time
        
        return BulaComparison(
            product_id=f"{bula_name_old} ({file_old} vs {file_new})",
            file_current=file_new,
            file_reference=file_old,
            sections=comparisons,
            overall_assessment=overall_assessment,
            processing_time=processing_time,
            pages_filtered=filtered_pages,
            bula_index=bula_index,
            bula_name=bula_name_old
        )
    
    def compare_multiple_bulas(self, pdf1_bytes: bytes, pdf1_name: str, 
                              pdf2_bytes: bytes, pdf2_name: str) -> MultiBulaComparison:
        """
        NOVA FUNÇÃO PRINCIPAL: Compara múltiplas bulas entre dois arquivos
        """
        start_time = time.time()
        
        # Extrai textos com filtragem
        with st.spinner(f"Extraindo texto de {pdf1_name}..."):
            text_old, filtered_old = self.extract_text_from_pdf(pdf1_bytes, pdf1_name)
        
        with st.spinner(f"Extraindo texto de {pdf2_name}..."):
            text_new, filtered_new = self.extract_text_from_pdf(pdf2_bytes, pdf2_name)
        
        # Divide em múltiplas bulas
        with st.spinner("Identificando bulas individuais..."):
            bulas_old = self.split_multiple_bulas(text_old)
            bulas_new = self.split_multiple_bulas(text_new)
        
        # Mostra estatísticas
        st.info(f"📋 **{pdf1_name}**: {len(bulas_old)} bula(s) identificada(s)")
        st.info(f"📋 **{pdf2_name}**: {len(bulas_new)} bula(s) identificada(s)")
        
        # Lista para armazenar comparações
        comparisons = []
        unmatched_bulas = {'reference': [], 'current': []}
        
        # Compara bulas em ordem (primeira com primeira, segunda com segunda, etc.)
        max_bulas = max(len(bulas_old), len(bulas_new))
        filtered_pages = {pdf1_name: filtered_old, pdf2_name: filtered_new}
        
        for i in range(max_bulas):
            if i < len(bulas_old) and i < len(bulas_new):
                # Ambas as bulas existem - compara normalmente
                bula_name_old, bula_text_old = bulas_old[i]
                bula_name_new, bula_text_new = bulas_new[i]
                
                st.info(f"🔄 Comparando: **{bula_name_old}** ↔ **{bula_name_new}**")
                
                comparison = self.compare_single_bula(
                    bula_text_old, bula_text_new,
                    bula_name_old, bula_name_new,
                    pdf1_name, pdf2_name,
                    i + 1, filtered_pages
                )
                comparisons.append(comparison)
                
            elif i < len(bulas_old):
                # Bula existe apenas no arquivo de referência
                bula_name_old, bula_text_old = bulas_old[i]
                unmatched_bulas['reference'].append(bula_name_old)
                st.warning(f"⚠️ **{bula_name_old}** existe apenas em {pdf1_name}")
                
            elif i < len(bulas_new):
                # Bula existe apenas no arquivo atual
                bula_name_new, bula_text_new = bulas_new[i]
                unmatched_bulas['current'].append(bula_name_new)
                st.warning(f"⚠️ **{bula_name_new}** existe apenas em {pdf2_name}")
        
        processing_time = time.time() - start_time
        
        return MultiBulaComparison(
            file_reference=pdf1_name,
            file_current=pdf2_name,
            total_bulas_reference=len(bulas_old),
            total_bulas_current=len(bulas_new),
            comparisons=comparisons,
            unmatched_bulas=unmatched_bulas,
            processing_time=processing_time
        )
    
    def _calculate_overall_assessment(self, comparisons: Dict[str, SectionComparison]) -> str:
        """Calcula avaliação geral baseada nas comparações"""
        if not comparisons:
            return "❌ Nenhuma seção encontrada"
        
        scores = [comp.similarity_score for comp in comparisons.values() 
                 if comp.similarity_score is not None]
        
        if not scores:
            return "❌ Não foi possível calcular similaridade"
        
        avg_score = sum(scores) / len(scores)
        critical_changes = sum(1 for comp in comparisons.values() 
                             if "importante" in comp.assessment.lower())
        
        if critical_changes > 3:
            return f"🔴 CRÍTICO: {critical_changes} mudanças importantes (média: {avg_score:.2%})"
        elif critical_changes > 1:
            return f"🟡 ATENÇÃO: {critical_changes} mudanças importantes (média: {avg_score:.2%})"
        elif avg_score >= self.THRESHOLDS['good']:
            return f"✅ ESTÁVEL: Poucas alterações (média: {avg_score:.2%})"
        else:
            return f"⚠️ MODIFICADO: Alterações moderadas (média: {avg_score:.2%})"

def normalize_text_for_diff(text: str) -> str:
    """Normaliza texto para comparação, removendo formatação irrelevante"""
    # Remove múltiplos pontos
    text = re.sub(r'\.{2,}', ' ', text)
    
    # Normaliza diferentes tipos de travessão/hífen para o mesmo caractere
    text = re.sub(r'[—–−-]', '-', text)  # Unifica todos os tipos de traço
    
    # Normaliza espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    # Remove espaços antes/depois de pontuação
    text = re.sub(r'\s*([,.;:!?])\s*', r'\1 ', text)
    
    # Converte para minúsculas para ignorar diferenças de capitalização
    text = text.lower()
    
    # Remove acentos para comparação mais robusta (opcional)
    # Substitui caracteres acentuados pelos equivalentes sem acento
    replacements = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    for accented, normal in replacements.items():
        text = text.replace(accented, normal)
    
    return text.strip()

def generate_html_diff(text1: str, text2: str, title1: str = "Versão Anterior", title2: str = "Versão Atual") -> str:
    """
    Gera diff focado em diferenças textuais relevantes
    """
    # Normaliza textos
    norm1 = normalize_text_for_diff(text1)
    norm2 = normalize_text_for_diff(text2)
    
    # Divide em palavras
    words1 = norm1.split()
    words2 = norm2.split()
    
    # Usa SequenceMatcher para identificar diferenças
    matcher = SequenceMatcher(None, words1, words2)
    
    # Gera HTML com diferenças destacadas
    html_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Palavras iguais - texto normal
            html_parts.extend(words1[i1:i2])
        elif tag == 'delete':
            # Palavras removidas - vermelho
            for word in words1[i1:i2]:
                html_parts.append(f'<span style="background-color: #ffebee; color: #c62828; text-decoration: line-through; padding: 2px 4px; border-radius: 3px; margin: 0 1px;">{word}</span>')
        elif tag == 'insert':
            # Palavras adicionadas - verde
            for word in words2[j1:j2]:
                html_parts.append(f'<span style="background-color: #e8f5e8; color: #2e7d32; padding: 2px 4px; border-radius: 3px; margin: 0 1px;">{word}</span>')
        elif tag == 'replace':
            # Palavras modificadas - amarelo para removidas, verde para novas
            for word in words1[i1:i2]:
                html_parts.append(f'<span style="background-color: #fff3e0; color: #ef6c00; text-decoration: line-through; padding: 2px 4px; border-radius: 3px; margin: 0 1px;">{word}</span>')
            for word in words2[j1:j2]:
                html_parts.append(f'<span style="background-color: #e8f5e8; color: #2e7d32; padding: 2px 4-px; border-radius: 3px; margin: 0 1px;">{word}</span>')
    
    # Monta HTML final
    html_content = ' '.join(html_parts)
    
    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                background-color: #fafafa;
            }}
            .diff-container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .legend {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }}
            .legend h4 {{
                margin: 0 0 10px 0;
                color: #1976D2;
            }}
            .content {{
                font-size: 14px;
                line-height: 2;
                text-align: justify;
            }}
        </style>
    </head>
    <body>
        <div class="diff-container">
            <div class="legend">
                <h4>📖 Legenda das Diferenças:</h4>
                <span style="background-color: #e8f5e8; color: #2e7d32; padding: 2px 4px; border-radius: 3px;">Texto Adicionado</span> |
                <span style="background-color: #ffebee; color: #c62828; text-decoration: line-through; padding: 2px 4px; border-radius: 3px;">Texto Removido</span> |
                <span style="background-color: #fff3e0; color: #ef6c00; padding: 2px 4px; border-radius: 3px;">Texto Modificado</span>
            </div>
            <div class="content">
                {html_content}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

def display_multi_bula_results(result: MultiBulaComparison):
    """
    NOVA FUNÇÃO: Exibe os resultados da comparação de múltiplas bulas
    """
    
    try:
        # Header com resultado geral
        st.header("📊 Resultado da Análise Multi-Bula")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tempo de Processamento", f"{result.processing_time:.2f}s")
        with col2:
            st.metric("Bulas no Arquivo Referência", result.total_bulas_reference)
        with col3:
            st.metric("Bulas no Arquivo Atual", result.total_bulas_current)
        with col4:
            st.metric("Comparações Realizadas", len(result.comparisons))
        
        # Alerta sobre bulas não correspondidas
        if result.unmatched_bulas['reference'] or result.unmatched_bulas['current']:
            st.warning("⚠️ **Bulas sem correspondência encontradas!**")
            
            if result.unmatched_bulas['reference']:
                st.write(f"**Apenas em {result.file_reference}:**")
                for bula in result.unmatched_bulas['reference']:
                    st.write(f"- {bula}")
            
            if result.unmatched_bulas['current']:
                st.write(f"**Apenas em {result.file_current}:**")
                for bula in result.unmatched_bulas['current']:
                    st.write(f"- {bula}")
        
        # Resumo geral de todas as comparações
        if result.comparisons:
            st.subheader("🎯 Resumo Geral das Bulas")
            
            # Calcula estatísticas gerais
            all_scores = []
            critical_assessments = []
            
            for comparison in result.comparisons:
                scores = [comp.similarity_score for comp in comparison.sections.values() 
                         if comp.similarity_score is not None]
                if scores:
                    all_scores.extend(scores)
                
                if "CRÍTICO" in comparison.overall_assessment or "ATENÇÃO" in comparison.overall_assessment:
                    critical_assessments.append(comparison.bula_name)
            
            # Exibe métricas gerais
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                st.metric("Similaridade Média Geral", f"{avg_score:.2%}")
            
            with col2:
                st.metric("Bulas com Problemas", len(critical_assessments))
            
            with col3:
                total_sections = sum(len(comp.sections) for comp in result.comparisons)
                st.metric("Total de Seções Analisadas", total_sections)
            
            # Lista de bulas com problemas
            if critical_assessments:
                st.error(f"🚨 **Bulas que requerem atenção:** {', '.join(critical_assessments)}")
            
            # Gráfico consolidado (se plotly disponível)
            if PLOTLY_AVAILABLE and len(result.comparisons) > 1:
                try:
                    st.subheader("📈 Comparação Geral entre Bulas")
                    
                    # Prepara dados para gráfico
                    chart_data = []
                    for comparison in result.comparisons:
                        scores = [comp.similarity_score for comp in comparison.sections.values() 
                                 if comp.similarity_score is not None]
                        if scores:
                            avg_similarity = sum(scores) / len(scores)
                            chart_data.append({
                                'Bula': comparison.bula_name,
                                'Similaridade Média': avg_similarity,
                                'Seções': len(comparison.sections),
                                'Status': comparison.overall_assessment
                            })
                    
                    # if chart_data:
                    #     df = pd.DataFrame(chart_data)
                        
                    #     # Gráfico de barras comparativo
                    #     fig = px.bar(
                    #         df, 
                    #         x='Bula', 
                    #         y='Similaridade Média',
                    #         color='Similaridade Média',
                    #         color_continuous_scale='RdYlGn',
                    #         range_color=[0, 1],
                    #         title="Similaridade Média por Bula",
                    #         hover_data=['Seções', 'Status']
                    #     )
                    #     fig.update_layout(height=500)
                    #     st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"⚠️ Erro ao gerar gráfico consolidado: {e}")
        
        # Detalhamento individual de cada bula
        st.subheader("📋 Detalhamento por Bula")
        
        for i, comparison in enumerate(result.comparisons):
            with st.expander(f"📄 **{comparison.bula_name}** - {comparison.overall_assessment}", 
                           expanded=i == 0):  # Primeira bula expandida por padrão
                
                # Chama a função de exibição original para cada bula individual
                display_single_bula_results(comparison, show_header=False)
    
    except Exception as e:
        st.error(f"❌ Erro ao exibir resultados: {e}")
        st.exception(e)

def display_single_bula_results(result: BulaComparison, show_header: bool = True):
    """
    FUNÇÃO MODIFICADA: Exibe os resultados de uma única bula (versão original adaptada)
    """
    
    try:
        if show_header:
            # Header com resultado geral
            st.header(f"📊 Resultado da Análise - {result.bula_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tempo de Processamento", f"{result.processing_time:.2f}s")
        with col2:
            st.metric("Total de Seções", len(result.sections))
        with col3:
            scores = [comp.similarity_score for comp in result.sections.values() 
                     if comp.similarity_score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            st.metric("Similaridade Média", f"{avg_score:.2%}")
        with col4:
            # Total de páginas filtradas
            total_filtered = sum(result.pages_filtered.values())
            st.metric("Páginas Filtradas", total_filtered)
        
        # Detalhes das páginas filtradas
        if total_filtered > 0:
            with st.expander("📝 Detalhes da Filtragem de Páginas", expanded=False):
                for filename, count in result.pages_filtered.items():
                    if count > 0:
                        st.write(f"📄 **{filename}**: {count} página(s) filtrada(s)")
                
                st.info("""
                **Páginas filtradas automaticamente:**
                - Páginas com padrão de processo Anvisa (XXXXXXX/XX-X)
                - Páginas com texto administrativo ("de texto de bula")
                
                Essas páginas são excluídas da análise para focar apenas no conteúdo da bula.
                """)
        
        # Avaliação geral
        st.subheader("🎯 Avaliação Geral")
        
        # Determina cor baseada na avaliação
        if "CRÍTICO" in result.overall_assessment:
            st.error(result.overall_assessment)
        elif "ATENÇÃO" in result.overall_assessment:
            st.warning(result.overall_assessment)
        elif "ESTÁVEL" in result.overall_assessment:
            st.success(result.overall_assessment)
        else:
            st.info(result.overall_assessment)
        
        # Gráfico de similaridade por seção
        if result.sections and PLOTLY_AVAILABLE:
            try:
                st.subheader("📈 Similaridade por Seção")
                
                # Prepara dados para o gráfico
                chart_data = []
                for section, comp in result.sections.items():
                    if comp.similarity_score is not None:
                        chart_data.append({
                            'Seção': section[:50] + "..." if len(section) > 50 else section,
                            'Similaridade': comp.similarity_score,
                            'Avaliação': comp.assessment,
                            'Método': comp.method_used
                        })
                
                if chart_data:
                    df = pd.DataFrame(chart_data)
                    
                    # Gráfico de barras
                    fig = px.bar(
                        df, 
                        x='Similaridade', 
                        y='Seção',
                        color='Similaridade',
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 1],
                        title=f"Similaridade por Seção - {result.bula_name}",
                        hover_data=['Avaliação', 'Método']
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erro ao gerar gráfico: {e}")
        
        # Detalhes das seções
        st.subheader("🔍 Detalhamento por Seção")
        
        # Usar session_state para manter o estado dos filtros (com sufixo da bula)
        bula_suffix = f"_bula_{result.bula_index}"
        
        if f"show_only_changes{bula_suffix}" not in st.session_state:
            st.session_state[f"show_only_changes{bula_suffix}"] = True
        if f"show_text_diff{bula_suffix}" not in st.session_state:
            st.session_state[f"show_text_diff{bula_suffix}"] = True
        if f"show_html_diff{bula_suffix}" not in st.session_state:
            st.session_state[f"show_html_diff{bula_suffix}"] = True
        
        # Filtros com session_state
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state[f"show_only_changes{bula_suffix}"] = st.checkbox(
                "Mostrar apenas alterações significativas", 
                value=st.session_state[f"show_only_changes{bula_suffix}"],
                key=f"checkbox_show_changes{bula_suffix}"
            )
        with col2:
            st.session_state[f"show_text_diff{bula_suffix}"] = st.checkbox(
                "Mostrar diferenças textuais", 
                value=st.session_state[f"show_text_diff{bula_suffix}"],
                key=f"checkbox_show_diff{bula_suffix}"
            )
        with col3:
            st.session_state[f"show_html_diff{bula_suffix}"] = st.checkbox(
                "Diff visual (HTML)", 
                value=st.session_state[f"show_html_diff{bula_suffix}"],
                key=f"checkbox_show_html_diff{bula_suffix}"
            )
        
        # Container para os detalhes das seções
        sections_container = st.container()
        
        with sections_container:
            # Gera lista estável de seções ordenadas
            sections_list = list(result.sections.items())
            
            for idx, (section_title, comparison) in enumerate(sections_list):
                try:
                    # Filtra baseado na opção selecionada
                    if (st.session_state[f"show_only_changes{bula_suffix}"] and 
                        comparison.similarity_score is not None and 
                        comparison.similarity_score >= 0.85):
                        continue
                    
                    # Usar índice numerico com sufixo da bula para chaves
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
                                    # Limita textos para evitar problemas de performance
                                    max_chars = 3000
                                    text_old_limited = (comparison.text_old[:max_chars] + "..." 
                                                      if len(comparison.text_old) > max_chars 
                                                      else comparison.text_old)
                                    text_new_limited = (comparison.text_new[:max_chars] + "..." 
                                                      if len(comparison.text_new) > max_chars 
                                                      else comparison.text_new)
                                    
                                    # Gera diff HTML
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
                                    
                                    # Mostra diff HTML
                                    components.html(html_diff, height=600, scrolling=True)
                                    
                                    # Aviso se texto foi limitado
                                    if (len(comparison.text_old) > max_chars or 
                                        len(comparison.text_new) > max_chars):
                                        st.warning(f"⚠️ Texto limitado a {max_chars} caracteres para melhor performance")
                                
                                except Exception as e:
                                    st.error(f"❌ Erro ao gerar diff visual: {e}")
                                    st.info("💡 Tente usar a comparação textual simples")
                            
                            else:
                                # Usar tabs ao invés de colunas para melhor organização
                                tab1, tab2 = st.tabs(["📄 Versão Anterior", "📄 Versão Atual"])
                                
                                with tab1:
                                    # Limita o texto para evitar problemas de performance
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
                                    # Limita o texto para evitar problemas de performance
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
        st.error(f"❌ Erro ao exibir resultados: {e}")
        st.exception(e)

def export_multi_bula_results(result: MultiBulaComparison):
    """
    NOVA FUNÇÃO: Permite exportar resultados de múltiplas bulas
    """
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
        report = f"""
RELATÓRIO DE COMPARAÇÃO DE MÚLTIPLAS BULAS
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
                report += f"\n    Similaridade: {comp.similarity_score:.2%}"
                report += f"\n    Avaliação: {comp.assessment}"
                report += f"\n    Método: {comp.method_used}"
        
        st.download_button(
            label="📋 Baixar Relatório",
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
        
        🆕 **Nova funcionalidade:** Suporte a arquivos com **múltiplas bulas**, comparando automaticamente 
        na ordem (primeira com primeira, segunda com segunda, etc.).
        
        🎯 **Recursos avançados:** - Consolidação automática de seções não padrão nas seções regulamentares correspondentes
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
        st.sidebar.subheader("📋 Consolidação de Seções")
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
        st.header("📁 Upload dos Arquivos")
        
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
            if st.button("🚀 Analisar Bulas", type="primary", use_container_width=True):
                
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
            st.info("👆 Carregue dois arquivos PDF para começar a análise")
        
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
        
        ### 📋 Padrões de filtragem utilizados:
        - **Processos Anvisa:** Padrão XXXXXXX/XX-X (7 dígitos/2 dígitos-1 dígito)
        - **Texto administrativo:** Expressão "de texto de bula" (case-insensitive)
        """)
        st.markdown("---")
        st.markdown("""
        **Desenvolvido e elaborado por Cyro Caldeira, Daniel Dourado, Gláucia Lima e Leonardo Santos**
        
        Com a finalidade de obtenção de certificado de especialista em Ciência de Dados e Inteligência Artificial
        """)
        
    except Exception as e:
        st.error(f"❌ Erro na aplicação: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()




