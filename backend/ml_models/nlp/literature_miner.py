"""
Scientific Literature Mining
Real implementation for mining scientific literature and extracting knowledge
"""

import asyncio
import logging
from typing import Dict, List, Optional
import aiohttp
import xml.etree.ElementTree as ET
import re
import spacy
from datetime import datetime

logger = logging.getLogger(__name__)


class LiteratureMiner:
    """Real scientific literature mining using PubMed and NLP"""
    
    def __init__(self):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.nlp = None
        
    async def initialize(self):
        """Initialize NLP models"""
        try:
            # Load scientific NLP model
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("Literature miner initialized")
        except OSError:
            logger.warning("Scientific spaCy model not found, using basic English model")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available")
                self.nlp = None
    
    async def search_pubmed(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search PubMed for scientific articles"""
        try:
            # Step 1: Search for PMIDs
            search_url = f"{self.pubmed_base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pmids = data.get("esearchresult", {}).get("idlist", [])
                    else:
                        return []
            
            if not pmids:
                return []
            
            # Step 2: Fetch article details
            fetch_url = f"{self.pubmed_base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, params=fetch_params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        articles = self._parse_pubmed_xml(xml_data)
                        return articles
            
            return []
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict]:
        """Parse PubMed XML response"""
        articles = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract basic information
                    pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else ""
                    title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else ""
                    
                    # Extract abstract
                    abstract_parts = []
                    for abstract_text in article.findall(".//AbstractText"):
                        if abstract_text.text:
                            abstract_parts.append(abstract_text.text)
                    abstract = " ".join(abstract_parts)
                    
                    # Extract authors
                    authors = []
                    for author in article.findall(".//Author"):
                        last_name = author.find("LastName")
                        first_name = author.find("ForeName")
                        if last_name is not None and first_name is not None:
                            authors.append(f"{last_name.text}, {first_name.text}")
                    
                    # Extract journal
                    journal = article.find(".//Journal/Title")
                    journal_name = journal.text if journal is not None else ""
                    
                    # Extract publication date
                    pub_date = self._extract_publication_date(article)
                    
                    articles.append({
                        "pubmed_id": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "journal": journal_name,
                        "publication_date": pub_date
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        
        return articles
    
    def _extract_publication_date(self, article) -> Optional[str]:
        """Extract publication date from XML"""
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            
            if year is not None:
                date_str = year.text
                if month is not None:
                    date_str += f"-{month.text}"
                    if day is not None:
                        date_str += f"-{day.text}"
                return date_str
        return None
    
    async def extract_entities(self, text: str) -> List[Dict]:
        """Extract biomedical entities from text"""
        if not self.nlp:
            return []
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8
                })
        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    async def extract_relationships(self, text: str) -> List[Dict]:
        """Extract relationships from text"""
        relationships = []
        
        # Simple pattern-based relationship extraction
        patterns = [
            (r'(\w+)\s+inhibits?\s+(\w+)', 'inhibits'),
            (r'(\w+)\s+activates?\s+(\w+)', 'activates'),
            (r'(\w+)\s+binds?\s+to\s+(\w+)', 'binds_to'),
            (r'(\w+)\s+causes?\s+(\w+)', 'causes')
        ]
        
        for pattern, relation_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    "subject": match.group(1),
                    "predicate": relation_type,
                    "object": match.group(2),
                    "confidence": 0.7
                })
        
        return relationships


# Global instance
literature_miner = LiteratureMiner()
