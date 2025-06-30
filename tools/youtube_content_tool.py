from smolagents import Tool
import yt_dlp
import re
import requests
from urllib.parse import urlparse, parse_qs
from typing import Optional


class YouTubeContentTool(Tool):
    name = "extract_youtube_content"
    description = """Extract transcript, metadata, and content from YouTube videos using yt-dlp. 
    Supports both manual subtitles and auto-generated captions. Can handle video URLs or video IDs."""
    inputs = {
        "url_or_id": {
            "type": "string",
            "description": "YouTube video URL (https://youtube.com/watch?v=...) or video ID (dQw4w9WgXcQ)",
        },
        "query": {
            "type": "string", 
            "description": "Optional: specific topic or information to look for in the transcript",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        self._state_question = None
        super().__init__(**kwargs)
    
    def configure_from_state(self, question: str):
        """Store question for potential query enhancement"""
        self._state_question = question
        print(f"🎥 YouTubeContentTool noted question context: {question[:50]}...")

    def forward(self, url_or_id: str, query: Optional[str] = None) -> str:
        """
        OFFICIAL WORKAROUND: Use Optional[str] = None in signature
        Runtime validation handled by making query truly optional
        """
        # Validate required parameter
        if not url_or_id or not url_or_id.strip():
            raise ValueError("url_or_id parameter is required")
        
        # Handle optional query parameter (None is valid)
        if query is None:
            query = ""
        
        # Extract YouTube video content with yt-dlp
        try:
            # Normalize input to video URL
            video_url = self._normalize_video_url(url_or_id)
            if not video_url:
                return "Error: Invalid YouTube URL or video ID"
            
            # Extract video content using yt-dlp
            video_content = self._extract_video_content(video_url)
            if not video_content:
                return "Error: Failed to extract video content"
            
            # Format output based on query
            if query.strip():
                return self._format_targeted_response(video_content, query)
            else:
                return self._format_full_response(video_content)
                
        except Exception as e:
            return f"Error extracting YouTube content: {str(e)}"

    def _normalize_video_url(self, url_or_id: str) -> Optional[str]:
        """Convert various YouTube URL formats to standard format"""
        try:
            url_or_id = url_or_id.strip()
            
            # If it's already a full YouTube URL, return as-is
            if 'youtube.com/watch' in url_or_id:
                return url_or_id
            elif 'youtu.be/' in url_or_id:
                # Convert youtu.be short URL to full URL
                video_id = url_or_id.split('/')[-1].split('?')[0]
                return f"https://www.youtube.com/watch?v={video_id}"
            elif len(url_or_id) == 11 and url_or_id.isalnum():
                # Assume it's a video ID
                return f"https://www.youtube.com/watch?v={url_or_id}"
            else:
                # Try to extract video ID from various URL formats
                video_id = self._extract_video_id(url_or_id)
                if video_id:
                    return f"https://www.youtube.com/watch?v={video_id}"
                    
            return None
            
        except Exception:
            return None

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        try:
            parsed = urlparse(url)
            
            if 'youtube.com' in parsed.netloc:
                return parse_qs(parsed.query).get('v', [None])[0]
            elif 'youtu.be' in parsed.netloc:
                return parsed.path.lstrip('/')
                
            # Try regex as fallback
            match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
            return match.group(1) if match else None
            
        except Exception:
            return None

    def _extract_video_content(self, video_url: str) -> Optional[dict]:
        """Extract video content using yt-dlp"""
        try:
            # Configure yt-dlp for subtitle and metadata extraction
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,  # Include auto-generated captions
                'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-AU'],
                'skip_download': True,  # Don't download video file
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info and subtitles
                info = ydl.extract_info(video_url, download=False)
                
                # Get transcript from subtitles
                transcript = self._extract_transcript(info)
                
                return {
                    'video_id': info.get('id'),
                    'title': info.get('title', 'Unknown Title'),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'uploader_id': info.get('uploader_id', ''),
                    'transcript': transcript,
                    'has_transcript': transcript is not None and len(transcript.strip()) > 0,
                    'url': video_url
                }
                
        except Exception as e:
            print(f"yt-dlp extraction error: {e}")
            return None

    def _extract_transcript(self, video_info: dict) -> Optional[str]:
        """Extract transcript from video info"""
        try:
            subtitles = video_info.get('subtitles', {})
            automatic_captions = video_info.get('automatic_captions', {})
            
            # Priority order for language selection
            language_priority = ['en', 'en-US', 'en-GB', 'en-AU']
            
            # Try manual subtitles first (higher quality)
            for lang in language_priority:
                if lang in subtitles:
                    transcript = self._download_subtitle(subtitles[lang])
                    if transcript:
                        return transcript
            
            # Fallback to auto-generated captions
            for lang in language_priority:
                if lang in automatic_captions:
                    transcript = self._download_subtitle(automatic_captions[lang])
                    if transcript:
                        return transcript
            
            return None
            
        except Exception as e:
            print(f"Transcript extraction error: {e}")
            return None

    def _download_subtitle(self, subtitle_formats: list) -> Optional[str]:
        """Download and clean subtitle content"""
        try:
            # Prefer VTT format, then SRT, then others
            format_priority = ['vtt', 'srt', 'ttml', 'srv1', 'srv2', 'srv3']
            
            for preferred_format in format_priority:
                for subtitle_format in subtitle_formats:
                    if subtitle_format.get('ext') == preferred_format:
                        subtitle_url = subtitle_format.get('url')
                        if subtitle_url:
                            # Download subtitle content
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                            response = requests.get(subtitle_url, headers=headers, timeout=10)
                            
                            if response.status_code == 200:
                                return self._clean_subtitle_text(response.text)
            
            return None
            
        except Exception as e:
            print(f"Subtitle download error: {e}")
            return None

    def _clean_subtitle_text(self, subtitle_content: str) -> str:
        """Clean subtitle format to readable text"""
        try:
            # Remove VTT headers and metadata
            content = re.sub(r'WEBVTT.*?\n\n', '', subtitle_content, flags=re.DOTALL)
            content = re.sub(r'NOTE.*?\n\n', '', content, flags=re.DOTALL)
            
            # Remove SRT numbering and timestamps
            content = re.sub(r'^\d+\n', '', content, flags=re.MULTILINE)
            content = re.sub(r'\d{2}:\d{2}:\d{2}[,\.]\d{3} --> \d{2}:\d{2}:\d{2}[,\.]\d{3}.*?\n', '', content)
            content = re.sub(r'\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}.*?\n', '', content)
            
            # Remove HTML/XML tags
            content = re.sub(r'<[^>]+>', '', content)
            
            # Remove speaker labels and timestamps
            content = re.sub(r'\[.*?\]', '', content)
            content = re.sub(r'\(.*?\)', '', content)
            
            # Clean up whitespace and newlines
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not re.match(r'^[\d\s:,.-]+$', line):  # Skip timestamp-only lines
                    lines.append(line)
            
            # Join lines and clean up extra spaces
            text = ' '.join(lines)
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
            
            return text.strip()
            
        except Exception as e:
            print(f"Subtitle cleaning error: {e}")
            return subtitle_content

    def _format_targeted_response(self, video_content: dict, query: str) -> str:
        """Format response focused on specific query"""
        output_lines = [
            f"📺 **{video_content['title']}**",
            f"👤 **Uploader:** {video_content['uploader']}",
            f"⏱️ **Duration:** {self._format_duration(video_content['duration'])}",
            ""
        ]
        
        if video_content['has_transcript']:
            # Search for query-relevant parts of transcript
            transcript = video_content['transcript']
            relevant_parts = self._find_relevant_transcript_parts(transcript, query)
            
            if relevant_parts:
                output_lines.extend([
                    f"🎯 **Content related to '{query}':**",
                    "",
                    relevant_parts
                ])
            else:
                output_lines.extend([
                    f"⚠️ No specific content found for '{query}' in transcript.",
                    "",
                    "📝 **Full transcript:**",
                    transcript[:1000] + "..." if len(transcript) > 1000 else transcript
                ])
        else:
            output_lines.extend([
                "❌ **No transcript available**",
                "",
                f"📝 **Description:** {video_content['description'][:500]}..."
            ])
        
        return "\n".join(output_lines)

    def _format_full_response(self, video_content: dict) -> str:
        """Format complete video information"""
        output_lines = [
            f"📺 **{video_content['title']}**",
            f"👤 **Uploader:** {video_content['uploader']}",
            f"⏱️ **Duration:** {self._format_duration(video_content['duration'])}",
            f"👀 **Views:** {self._format_number(video_content['view_count'])}",
            f"📅 **Upload Date:** {self._format_date(video_content['upload_date'])}",
            f"🔗 **URL:** {video_content['url']}",
            ""
        ]
        
        if video_content['has_transcript']:
            transcript = video_content['transcript']
            output_lines.extend([
                "📝 **Full Transcript:**",
                "",
                transcript
            ])
        else:
            output_lines.extend([
                "❌ **No transcript available**",
                "",
                "📝 **Description:**",
                video_content['description'][:1000] + "..." if len(video_content['description']) > 1000 else video_content['description']
            ])
        
        return "\n".join(output_lines)

    def _find_relevant_transcript_parts(self, transcript: str, query: str) -> str:
        """Find parts of transcript relevant to query"""
        try:
            query_words = query.lower().split()
            sentences = re.split(r'[.!?]+', transcript)
            
            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short fragments
                    continue
                    
                sentence_lower = sentence.lower()
                # Check if sentence contains query words
                matches = sum(1 for word in query_words if word in sentence_lower)
                if matches > 0:
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return ". ".join(relevant_sentences[:5])  # Limit to top 5 relevant sentences
            
            return ""
            
        except Exception:
            return ""

    def _format_duration(self, duration: int) -> str:
        """Format duration in seconds to human readable"""
        if not duration:
            return "Unknown"
        
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _format_number(self, number: int) -> str:
        """Format large numbers with commas"""
        if not number:
            return "Unknown"
        return f"{number:,}"

    def _format_date(self, date_str: str) -> str:
        """Format YYYYMMDD date string"""
        if not date_str or len(date_str) != 8:
            return "Unknown"
        
        try:
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{year}-{month}-{day}"
        except Exception:
            return date_str