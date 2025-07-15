from smolagents import Tool
import yt_dlp
import re
import requests
import os
import tempfile
import json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import logging


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

    def __init__(self, cookies_file: Optional[str] = None, **kwargs):
        """
        Initialize YouTube tool with optional cookie support
        
        Args:
            cookies_file: Path to cookies.txt file for authentication
        """
        # Initialize instance variables
        self._state_question: Optional[str] = None
        self.cookies_file: Optional[str] = cookies_file
        self.browser_cookies: Optional[Dict[str, Optional[str]]] = None
        
        self._setup_logging()
        super().__init__(**kwargs)
    
    def _setup_logging(self):
        """Setup logging to help debug extraction issues"""
        logging.getLogger('yt_dlp').setLevel(logging.WARNING)
    
    def configure_from_state(self, question: str):
        """Store question for potential query enhancement"""
        self._state_question = question
        print(f"🎥 YouTubeContentTool noted question context: {question[:50]}...")
        
    def set_cookies_from_browser(self, browser: str = "chrome", profile: Optional[str] = None) -> None:
        """
        Configure tool to extract cookies from browser
        
        Args:
            browser: Browser name ("chrome", "firefox", "safari", "edge", "opera")
            profile: Optional browser profile name
        """
        self.browser_cookies = {"browser": browser, "profile": profile}
        print(f"🍪 Configured to use {browser} cookies" + (f" (profile: {profile})" if profile else ""))

    def set_cookies_from_file(self, cookies_file: str) -> None:
        """
        Set cookies from Netscape format cookies.txt file
        
        Args:
            cookies_file: Path to cookies.txt file
        """
        if os.path.exists(cookies_file):
            self.cookies_file = cookies_file
            print(f"🍪 Using cookies from file: {cookies_file}")
        else:
            print(f"⚠️ Cookie file not found: {cookies_file}")

    def create_cookies_from_env(self) -> Optional[str]:
        """
        Create cookies file from environment variables/secrets for HF Spaces deployment
        Expects YOUTUBE_COOKIES secret/environment variable with cookie data
        
        Returns:
            Path to temporary cookies file or None
        """
        try:
            # Check for cookie data in environment variable/secret
            cookie_data = os.environ.get('YOUTUBE_COOKIES')
            if not cookie_data:
                print("ℹ️ No YOUTUBE_COOKIES secret/environment variable found")
                return None
            
            # Create temporary cookies file
            temp_dir = tempfile.gettempdir()
            cookies_path = os.path.join(temp_dir, 'youtube_cookies.txt')
            
            # Write cookies to file
            with open(cookies_path, 'w') as f:
                f.write(cookie_data)
            
            print(f"🍪 Created temporary cookies file: {cookies_path}")
            return cookies_path
            
        except Exception as e:
            print(f"⚠️ Failed to create cookies from secret/environment: {e}")
            return None

    def _should_try_simple_first(self, video_url: str) -> bool:
        """
        Determine if we should try simple extraction first
        Some videos are more likely to work without authentication
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return True  # Default to simple approach
        
        # Simple heuristics - could be expanded
        # Most YouTube videos work with simple approach unless:
        # - They're very popular (more protection)
        # - They're from major channels (more protection)
        # - They're recent (more protection)
        
        # For now, always try simple first (fastest)
        return True

    def forward(self, url_or_id: str, query: Optional[str] = None) -> str:
        """
        Extract YouTube content with enhanced cookie support
        """
        # Validate required parameter
        if not url_or_id or not url_or_id.strip():
            raise ValueError("url_or_id parameter is required")
        
        # Handle optional query parameter
        if query is None:
            query = ""
        
        try:
            # Normalize input to video URL
            video_url = self._normalize_video_url(url_or_id)
            if not video_url:
                return "Error: Invalid YouTube URL or video ID"
            
            # Try multiple extraction strategies with smart ordering
            video_content = None
            
            # Option 1: Simple public video approach (fastest, works for most content)
            if not video_content:
                print("🎬 Trying simple public video extraction...")
                video_content = self._extract_public_video_content(video_url)
            
            # Option 2: Try with environment cookies (HF Spaces)
            if not video_content:
                env_cookies = self.create_cookies_from_env()
                if env_cookies:
                    print("🍪 Trying with environment cookies...")
                    video_content = self._extract_video_content(video_url, cookies_file=env_cookies)
                    # Clean up temporary file
                    try:
                        os.unlink(env_cookies)
                    except:
                        pass
            
            # Option 3: Try with configured cookies file
            if not video_content and self.cookies_file:
                print(f"🍪 Trying with cookies file: {self.cookies_file}")
                video_content = self._extract_video_content(video_url, cookies_file=self.cookies_file)
            
            # Option 4: Try with browser cookies (if configured)
            if not video_content and self.browser_cookies:
                print(f"🍪 Trying browser cookies: {self.browser_cookies['browser']}")
                video_content = self._extract_video_content(
                    video_url, 
                    cookiesfrom_browser=self.browser_cookies
                )
            
            # Option 5: Final fallback - no authentication
            if not video_content:
                print("⚠️ Trying final fallback without authentication...")
                video_content = self._extract_video_content(video_url)
            
            if not video_content:
                return self._format_extraction_failure(video_url)
            
            # Format output based on query
            if query.strip():
                return self._format_targeted_response(video_content, query)
            else:
                return self._format_full_response(video_content)
                
        except Exception as e:
            return f"Error extracting YouTube content: {str(e)}\n\n{self._get_troubleshooting_tips()}"

    def _extract_video_content(self, video_url: str, cookies_file: Optional[str] = None, 
                             cookiesfrom_browser: Optional[Dict] = None) -> Optional[dict]:
        """Enhanced video content extraction with cookie support"""
        try:
            # Base yt-dlp options inspired by working YoutubeVideoTool
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-AU'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'forceurl': True,  # Force URL extraction (helps avoid some bot detection)
                'noplaylist': True,  # Don't process playlists
                # Simpler format selection (less suspicious)
                'format': 'best[height<=720]/best',  # Reasonable quality limit
                # Basic headers (don't over-engineer)
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
            }
            
            # Add cookie authentication if available
            if cookies_file and os.path.exists(cookies_file):
                ydl_opts['cookiefile'] = cookies_file
                print(f"🍪 Using cookies from file: {cookies_file}")
            elif cookiesfrom_browser:
                ydl_opts['cookiesfrombrowser'] = (
                    cookiesfrom_browser['browser'], 
                    cookiesfrom_browser.get('profile')
                )
                print(f"🍪 Using {cookiesfrom_browser['browser']} browser cookies")
            
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
                    'url': video_url,
                    'extraction_method': 'authenticated'
                }
                
        except Exception as e:
            print(f"yt-dlp extraction error: {e}")
            return None

    def _extract_public_video_content(self, video_url: str) -> Optional[dict]:
        """
        Simple extraction for public videos using YoutubeVideoTool approach
        This method targets lower-quality content that's less protected
        """
        try:
            # Minimal configuration inspired by working YoutubeVideoTool
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'vtt',
                'subtitleslangs': ['en'],
                'forceurl': True,  # Key flag from working tool
                'noplaylist': True,
                # Target lower quality content (less protected)
                'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                # Minimal headers to avoid detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
                    'url': video_url,
                    'extraction_method': 'public_simple'
                }
                
        except Exception as e:
            print(f"Public video extraction failed: {e}")
            return None

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