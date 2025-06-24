# tools/content_grounding_tool.py
# Specialized tool for content grounding and verification

from smolagents import Tool
from typing import Optional, Dict, Tuple
import re
from datetime import datetime, timezone
from urllib.parse import urlparse


class ContentGroundingTool(Tool):
    """
    Specialized tool for grounding and verifying content.
    
    Provides:
    - Source authority assessment
    - Temporal relevance analysis  
    - Content verification and fact-checking support
    - Multi-source comparison capabilities
    
    Use this to assess the reliability and context of content
    retrieved from any source (ContentRetrieverTool, web search, etc.)
    """
    
    name = "ground_content"
    description = """Assess and verify content for authority, temporal relevance, and reliability.

This tool analyzes content to provide:
- Source authority ranking (government, academic, news tier assessment)
- Temporal context analysis (recent vs historical relevance)
- Content verification insights
- Bias and reliability indicators

Perfect for fact-checking, source verification, and content quality assessment."""

    inputs = {
        "url": {
            "type": "string",
            "description": "The source URL to assess for authority and credibility.",
        },
        "content": {
            "type": "string",
            "description": "The content text to analyze for temporal relevance and verification.",
        },
        "analysis_type": {
            "type": "string",
            "description": "Type of grounding analysis: 'authority' for source assessment, 'temporal' for time relevance, 'verification' for fact-checking, or 'full' for comprehensive analysis.",
            "default": "full"
        }
    }
    output_type = "string"

    # Source authority patterns for different domains
    AUTHORITY_PATTERNS = {
        'government': [
            'gov', 'gov.uk', 'europa.eu', 'un.org', 'who.int',
            'census.gov', 'sec.gov', 'fda.gov', 'cdc.gov', 'nih.gov'
        ],
        'academic': [
            'edu', 'ac.uk', 'nature.com', 'science.org', 'pubmed',
            'arxiv.org', 'scholar.google', 'jstor.org', 'ieee.org'
        ],
        'financial': [
            'bloomberg.com', 'reuters.com', 'sec.gov', 'nasdaq.com',
            'nyse.com', 'imf.org', 'worldbank.org', 'federalreserve.gov'
        ],
        'news_tier1': [
            'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
            'pbs.org', 'economist.com', 'nytimes.com', 'wsj.com'
        ],
        'international': [
            'un.org', 'who.int', 'worldbank.org', 'imf.org',
            'oecd.org', 'wto.org', 'unesco.org'
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assess_source_authority(self, url: str) -> Dict[str, any]:
        """
        Comprehensive source authority assessment
        
        Returns:
            Dictionary with authority analysis
        """
        url_lower = url.lower()
        
        # Check for authoritative domains
        for category, domains in self.AUTHORITY_PATTERNS.items():
            for auth_domain in domains:
                if auth_domain in url_lower:
                    confidence_scores = {
                        'government': 0.95,
                        'academic': 0.90, 
                        'financial': 0.85,
                        'news_tier1': 0.80,
                        'international': 0.90
                    }
                    
                    return {
                        'authority_level': category,
                        'confidence_score': confidence_scores.get(category, 0.75),
                        'domain_match': auth_domain,
                        'reasoning': f"{category.replace('_', ' ').title()} source: {auth_domain}",
                        'trustworthiness': 'high'
                    }
        
        # Check for general patterns
        if any(pattern in url_lower for pattern in ['wikipedia', 'wiki']):
            return {
                'authority_level': 'reference',
                'confidence_score': 0.70,
                'domain_match': 'wikipedia',
                'reasoning': "Wikipedia/reference source - collaborative editing",
                'trustworthiness': 'medium-high'
            }
        elif any(pattern in url_lower for pattern in ['news', 'times', 'post', 'guardian']):
            return {
                'authority_level': 'news_general',
                'confidence_score': 0.60,
                'domain_match': 'news_media',
                'reasoning': "General news source - verify independently",
                'trustworthiness': 'medium'
            }
        elif '.edu' in url_lower or '.ac.' in url_lower:
            return {
                'authority_level': 'educational',
                'confidence_score': 0.75,
                'domain_match': 'educational',
                'reasoning': "Educational institution",
                'trustworthiness': 'medium-high'
            }
        
        return {
            'authority_level': 'unknown',
            'confidence_score': 0.30,
            'domain_match': 'unverified',
            'reasoning': "Source authority unclear - verify claims independently",
            'trustworthiness': 'low'
        }

    def _analyze_temporal_relevance(self, content: str) -> Dict[str, any]:
        """
        Analyze temporal aspects of content
        
        Returns:
            Dictionary with temporal analysis
        """
        current_year = datetime.now(timezone.utc).year
        
        # Extract dates from content
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'   # YYYY-MM-DD
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_dates.extend(matches)
        
        # Analyze temporal indicators
        temporal_keywords = {
            'current': ['current', 'latest', 'recent', 'now', 'today', 'this year'],
            'historical': ['historical', 'past', 'previous', 'former', 'old'],
            'future': ['future', 'upcoming', 'next', 'planned', 'projected'],
            'trending': ['trending', 'changing', 'evolving', 'developing']
        }
        
        content_lower = content.lower()
        detected_temporal_context = []
        
        for context_type, keywords in temporal_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_temporal_context.append(context_type)
        
        # Assess recency
        years = []
        for date in found_dates:
            year_match = re.search(r'\b(\d{4})\b', date)
            if year_match:
                try:
                    year = int(year_match.group(1))
                    if 1900 <= year <= current_year + 1:
                        years.append(year)
                except ValueError:
                    continue
        
        temporal_assessment = "unknown"
        if years:
            latest_year = max(years)
            if latest_year >= current_year - 1:
                temporal_assessment = "very_recent"
            elif latest_year >= current_year - 3:
                temporal_assessment = "recent"
            elif latest_year >= current_year - 10:
                temporal_assessment = "moderately_recent"
            else:
                temporal_assessment = "historical"
        
        return {
            'temporal_context': detected_temporal_context,
            'found_dates': found_dates[:5],
            'latest_year': max(years) if years else None,
            'temporal_assessment': temporal_assessment,
            'currency_score': self._calculate_currency_score(years, current_year),
            'temporal_indicators': len(detected_temporal_context)
        }

    def _calculate_currency_score(self, years: list, current_year: int) -> float:
        """Calculate how current/recent the content appears to be"""
        if not years:
            return 0.0
        
        latest_year = max(years)
        age = current_year - latest_year
        
        if age <= 1:
            return 1.0
        elif age <= 3:
            return 0.8
        elif age <= 5:
            return 0.6
        elif age <= 10:
            return 0.4
        else:
            return 0.2

    def _verify_content(self, content: str, url: str) -> Dict[str, any]:
        """
        Basic content verification analysis
        
        Returns:
            Dictionary with verification insights
        """
        content_lower = content.lower()
        
        # Check for verification indicators
        verification_signals = {
            'strong_claims': ['proven', 'confirmed', 'verified', 'established', 'documented'],
            'weak_claims': ['allegedly', 'reportedly', 'claims', 'suggests', 'might'],
            'uncertainty': ['unclear', 'unknown', 'unconfirmed', 'disputed', 'controversial'],
            'citations': ['source:', 'according to', 'study shows', 'research indicates', 'data from']
        }
        
        signal_counts = {}
        for signal_type, indicators in verification_signals.items():
            count = sum(1 for indicator in indicators if indicator in content_lower)
            signal_counts[signal_type] = count
        
        # Calculate verification score
        verification_score = (
            signal_counts['strong_claims'] * 0.3 +
            signal_counts['citations'] * 0.4 -
            signal_counts['weak_claims'] * 0.2 -
            signal_counts['uncertainty'] * 0.1
        ) / max(1, len(content.split()) / 100)  # Normalize by content length
        
        verification_score = max(0, min(1, verification_score))
        
        return {
            'verification_signals': signal_counts,
            'verification_score': verification_score,
            'content_length': len(content),
            'claim_density': (signal_counts['strong_claims'] + signal_counts['weak_claims']) / max(1, len(content.split()) / 100),
            'citation_density': signal_counts['citations'] / max(1, len(content.split()) / 100)
        }

    def forward(self, url: str, content: str, analysis_type: str = "full") -> str:
        """
        Perform grounding analysis on content
        
        Args:
            url: Source URL for authority assessment
            content: Content text to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Formatted grounding report
        """
        print(f"🔍 Grounding analysis ({analysis_type}) for content from: {url[:50]}...")
        
        results = {}
        
        # Perform requested analyses
        if analysis_type in ["authority", "full"]:
            results['authority'] = self._assess_source_authority(url)
        
        if analysis_type in ["temporal", "full"]:
            results['temporal'] = self._analyze_temporal_relevance(content)
        
        if analysis_type in ["verification", "full"]:
            results['verification'] = self._verify_content(content, url)
        
        # Format comprehensive report
        return self._format_grounding_report(url, content, results, analysis_type)

    def _format_grounding_report(self, url: str, content: str, results: Dict, analysis_type: str) -> str:
        """Format the grounding analysis report"""
        
        report_parts = []
        report_parts.append(f"🔍 CONTENT GROUNDING ANALYSIS")
        report_parts.append("=" * 50)
        report_parts.append(f"📄 SOURCE: {url}")
        report_parts.append(f"📊 CONTENT LENGTH: {len(content)} characters")
        report_parts.append("")
        
        # Authority Analysis
        if 'authority' in results:
            auth = results['authority']
            confidence_emoji = "🥇" if auth['confidence_score'] >= 0.8 else "🥈" if auth['confidence_score'] >= 0.6 else "🥉"
            report_parts.append(f"{confidence_emoji} SOURCE AUTHORITY:")
            report_parts.append(f"   Level: {auth['authority_level']}")
            report_parts.append(f"   Confidence: {auth['confidence_score']:.1%}")
            report_parts.append(f"   Trustworthiness: {auth['trustworthiness']}")
            report_parts.append(f"   Reasoning: {auth['reasoning']}")
            report_parts.append("")
        
        # Temporal Analysis
        if 'temporal' in results:
            temp = results['temporal']
            currency_emoji = "🕰️" if temp['currency_score'] >= 0.8 else "📅" if temp['currency_score'] >= 0.5 else "📜"
            report_parts.append(f"{currency_emoji} TEMPORAL ANALYSIS:")
            report_parts.append(f"   Assessment: {temp['temporal_assessment']}")
            report_parts.append(f"   Currency Score: {temp['currency_score']:.1%}")
            if temp['latest_year']:
                report_parts.append(f"   Latest Year: {temp['latest_year']}")
            if temp['temporal_context']:
                report_parts.append(f"   Context: {', '.join(temp['temporal_context'])}")
            report_parts.append("")
        
        # Verification Analysis
        if 'verification' in results:
            verif = results['verification']
            verif_emoji = "✅" if verif['verification_score'] >= 0.7 else "⚠️" if verif['verification_score'] >= 0.4 else "❌"
            report_parts.append(f"{verif_emoji} VERIFICATION ANALYSIS:")
            report_parts.append(f"   Verification Score: {verif['verification_score']:.1%}")
            report_parts.append(f"   Strong Claims: {verif['verification_signals']['strong_claims']}")
            report_parts.append(f"   Weak Claims: {verif['verification_signals']['weak_claims']}")
            report_parts.append(f"   Citations: {verif['verification_signals']['citations']}")
            report_parts.append("")
        
        # Overall Assessment
        if analysis_type == "full":
            overall_score = 0
            factors = 0
            
            if 'authority' in results:
                overall_score += results['authority']['confidence_score']
                factors += 1
            if 'temporal' in results:
                overall_score += results['temporal']['currency_score']
                factors += 1
            if 'verification' in results:
                overall_score += results['verification']['verification_score']
                factors += 1
            
            if factors > 0:
                overall_score /= factors
                overall_emoji = "🌟" if overall_score >= 0.8 else "👍" if overall_score >= 0.6 else "👎"
                report_parts.append(f"{overall_emoji} OVERALL GROUNDING SCORE: {overall_score:.1%}")
        
        return "\n".join(report_parts)