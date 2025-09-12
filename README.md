# BurlaCheck Multi-Bula

Detector de Alterações em Bulas Digitalizadas com Suporte a Múltiplas Bulas

## 📋 Descrição

Ferramenta desenvolvida para atender à **RDC nº 885/2024** da Anvisa, utilizando técnicas de processamento de linguagem natural para detectar alterações entre versões de bulas de medicamentos.

### ✨ Funcionalidades Principais

- **🔍 Detecção Automática de Múltiplas Bulas**: Identifica e processa automaticamente múltiplas bulas no mesmo arquivo PDF
- **🧠 Análise Semântica Avançada**: Detecta alterações mesmo quando o texto foi reescrito com palavras diferentes
- **📋 Consolidação Inteligente**: Integra automaticamente subseções não regulamentares às 9 seções padrão da Anvisa
- **🔧 Filtragem de Páginas**: Remove automaticamente páginas administrativas irrelevantes
- **📊 Relatórios Detalhados**: Gera análises completas com visualizações e estatísticas

### 🆕 Nova Funcionalidade Multi-Bula

- Identifica bulas usando o padrão da seção "composição"
- Comparação ordenada (primeira com primeira, segunda com segunda, etc.)
- Alertas para bulas sem correspondência entre arquivos
- Nomenclatura automática de medicamentos

## 🚀 Como Usar

1. **Upload dos Arquivos**: Carregue duas versões de bulas em formato PDF
2. **Configuração**: Escolha entre análise semântica (recomendado) ou comparação rápida
3. **Análise**: Clique em "Analisar Bulas" e aguarde o processamento
4. **Resultados**: Visualize diferenças, estatísticas e gráficos comparativos
5. **Export**: Baixe relatórios em JSON ou texto

## 🛠️ Instalação Local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/bulacheck-multi-bula.git
cd bulacheck-multi-bula

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run bulacheck_multi2.py
```

## 📦 Dependências

- Python 3.8+
- Streamlit
- PyMuPDF (processamento de PDF)
- Sentence Transformers (análise semântica)
- Plotly (visualizações)
- Pandas (manipulação de dados)

## 🎯 Casos de Uso

- **Fiscalização Sanitária**: Verificação de conformidade de bulas digitalizadas
- **Indústria Farmacêutica**: Controle de qualidade em alterações de bulas
- **Pesquisa Acadêmica**: Análise de evolução de informações médicas
- **Consultoria Regulatória**: Suporte em processos regulatórios

## 📖 Seções Analisadas

Conforme RDC nº 885/2024:
1. Para que este medicamento é indicado
2. Como este medicamento funciona
3. Quando não devo usar este medicamento
4. O que devo saber antes de usar este medicamento
5. Onde, como e por quanto tempo posso guardar este medicamento
6. Como devo usar este medicamento
7. O que devo fazer quando eu me esquecer de usar este medicamento
8. Quais os males que este medicamento pode me causar
9. O que fazer se alguém usar uma quantidade maior do que a indicada deste medicamento

## 🔍 Algoritmos de Detecção

### Padrões de Filtragem
- Processos Anvisa: `XXXXXXX/XX-X`
- Texto administrativo: "de texto de bula"

### Métodos de Comparação
- **Análise Semântica**: Sentence Transformers (multilingual)
- **Comparação Rápida**: SequenceMatcher (difflib)

## ⚡ Performance

- Suporte a arquivos de até 50MB
- Processamento otimizado para múltiplas bulas
- Cache inteligente para modelos semânticos
- Interface responsiva com feedback em tempo real

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Add: MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 📞 Suporte

Para dúvidas ou suporte, abra uma issue neste repositório.

---

**Desenvolvido para auxiliar na fiscalização de bulas digitalizadas conforme RDC nº 885/2024 da Anvisa**
