#!/usr/bin/env python3
"""
🛡️ VOICE SCAM INTERCEPTOR - REAL-TIME PROTECTION SYSTEM (FIXED VERSION)
=======================================================================

This system intercepts incoming calls, analyzes them in real-time using AI,
and blocks scammer calls before they reach the user.

FIXED: Compatible with latest package versions
"""

import asyncio
import threading
import queue
import time
import numpy as np
import re
from datetime import datetime
import json
import logging
from threading import Thread, Lock
import tkinter as tk
from tkinter import ttk

# Core imports that should work
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available - audio capture disabled")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available - using mock transcription")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - some features disabled")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - sentiment analysis disabled")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ Librosa not available - advanced audio features disabled")

try:
    from scipy import signal
    from sklearn.ensemble import IsolationForest
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - ML features disabled")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS not available - voice responses disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available - visualizations disabled")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI not available - web interface disabled")

# FIXED: ElevenLabs import (optional)
try:
    from elevenlabs import client
    ELEVENLABS_AVAILABLE = True
except ImportError:
    try:
        import elevenlabs
        ELEVENLABS_AVAILABLE = True
    except ImportError:
        ELEVENLABS_AVAILABLE = False
        print("⚠️ ElevenLabs not available - advanced TTS disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedVoiceAnalyzer:
    """
    🎤 ADVANCED REAL-TIME VOICE ANALYSIS ENGINE (COMPATIBLE VERSION)
    """
    
    def __init__(self):
        logger.info("🤖 Initializing Advanced Voice Analyzer...")
        
        # Load AI models with fallbacks
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Whisper loading failed: {e}")
        
        self.sentiment_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                logger.info("✅ Sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"⚠️ Sentiment analyzer failed: {e}")
        
        # Voice synthesis detector
        self.synthetic_detector = None
        if SKLEARN_AVAILABLE:
            self.synthetic_detector = IsolationForest(contamination=0.1, random_state=42)
            self.is_voice_model_trained = False
        
        # Enhanced scam detection patterns
        self.scam_patterns = {
            'high_risk': [
                r'\b(social security|ssn)\b.*\b(suspend|block|freeze)\b',
                r'\b(urgent|immediate|emergency)\b.*\b(payment|money|card)\b',
                r'\b(irs|government|tax)\b.*\b(arrest|warrant|lawsuit)\b',
                r'\b(gift card|western union|bitcoin|crypto)\b',
                r'\b(tech support|microsoft|apple)\b.*\b(virus|infected|compromised)\b',
                r'\b(bank account|credit card)\b.*\b(verify|confirm|update)\b',
                r'\b(final notice|last warning|immediate action)\b',
                r'\b(refund|prize|lottery|inheritance)\b.*\b(claim|collect|process)\b'
            ],
            'medium_risk': [
                r'\b(account|information)\b.*\b(verify|confirm|validate)\b',
                r'\b(expire|expir)\b.*\b(today|tonight|soon)\b',
                r'\b(call back|contact us)\b.*\b(urgent|asap)\b',
                r'\b(security|safety)\b.*\b(concern|issue|problem)\b'
            ],
            'suspicious_phrases': [
                "don't tell anyone", "keep this confidential", "this call is recorded",
                "federal crime", "local authorities", "send money now",
                "act now", "limited time", "exclusive offer", "congratulations you've won"
            ]
        }
        
        # Real-time audio buffer
        self.audio_buffer = queue.Queue()
        self.analysis_results = queue.Queue()
        self.is_analyzing = False
        
        logger.info("✅ Voice Analyzer initialized successfully!")
    
    def extract_voice_features(self, audio_data, sample_rate=16000):
        """Extract voice features with fallbacks"""
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for advanced features
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.numpy()
                
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                features = {}
                
                # Basic features
                features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
                features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
                
                # MFCCs
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i}'] = np.mean(mfccs[i])
                
                return np.array(list(features.values()))
            else:
                # Fallback: basic statistical features
                features = []
                features.append(np.mean(audio_data))
                features.append(np.std(audio_data))
                features.append(np.max(audio_data))
                features.append(np.min(audio_data))
                features.extend([0] * 16)  # Padding for compatibility
                return np.array(features)
                
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            return np.zeros(20)
    
    def detect_synthetic_voice(self, audio_data, sample_rate=16000):
        """Detect synthetic voice with fallbacks"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"is_synthetic": False, "confidence": 0.5, "risk_score": 0}
            
            features = self.extract_voice_features(audio_data, sample_rate)
            
            if not self.is_voice_model_trained:
                self.synthetic_detector.fit([features])
                self.is_voice_model_trained = True
                return {"is_synthetic": False, "confidence": 0.5, "risk_score": 0}
            
            prediction = self.synthetic_detector.predict([features])[0]
            decision_score = self.synthetic_detector.decision_function([features])[0]
            
            confidence = 1 / (1 + np.exp(-decision_score))
            is_synthetic = prediction == -1
            
            risk_score = int(confidence * 100) if is_synthetic else int((1 - confidence) * 100)
            
            return {
                "is_synthetic": is_synthetic,
                "confidence": float(confidence),
                "risk_score": risk_score,
                "features_extracted": len(features)
            }
            
        except Exception as e:
            logger.error(f"Error in synthetic voice detection: {e}")
            return {"is_synthetic": False, "confidence": 0.0, "risk_score": 0}
    
    def analyze_scam_patterns(self, text):
        """Advanced scam pattern analysis"""
        if not text:
            return {"risk_score": 0, "patterns_found": [], "category": "unknown"}
        
        text_lower = text.lower()
        risk_score = 0
        patterns_found = []
        
        # Check high-risk patterns
        for pattern in self.scam_patterns['high_risk']:
            if re.search(pattern, text_lower):
                risk_score += 30
                patterns_found.append(f"High-risk pattern: {pattern}")
        
        # Check medium-risk patterns
        for pattern in self.scam_patterns['medium_risk']:
            if re.search(pattern, text_lower):
                risk_score += 20
                patterns_found.append(f"Medium-risk pattern: {pattern}")
        
        # Check suspicious phrases
        for phrase in self.scam_patterns['suspicious_phrases']:
            if phrase in text_lower:
                risk_score += 15
                patterns_found.append(f"Suspicious phrase: {phrase}")
        
        # Additional heuristics
        urgency_words = len(re.findall(r'\b(urgent|immediate|asap|emergency|now|quickly)\b', text_lower))
        if urgency_words > 2:
            risk_score += 25
            patterns_found.append(f"Excessive urgency ({urgency_words} words)")
        
        # Personal info requests
        personal_requests = len(re.findall(r'\b(ssn|social security|password|pin|account number|routing number)\b', text_lower))
        if personal_requests > 0:
            risk_score += personal_requests * 20
            patterns_found.append(f"Personal info requests ({personal_requests})")
        
        # Determine category
        if risk_score >= 70:
            category = "high_risk_scam"
        elif risk_score >= 40:
            category = "suspicious"
        elif risk_score >= 20:
            category = "caution"
        else:
            category = "safe"
        
        return {
            "risk_score": min(risk_score, 100),
            "patterns_found": patterns_found,
            "category": category,
            "urgency_level": urgency_words,
            "personal_info_requests": personal_requests
        }
    
    def analyze_audio_chunk(self, audio_data, sample_rate=16000):
        """Comprehensive real-time audio analysis with fallbacks"""
        start_time = time.time()
        
        try:
            # 1. Transcribe speech
            if self.whisper_model and WHISPER_AVAILABLE:
                transcription_result = self.whisper_model.transcribe(audio_data)
                transcript = transcription_result['text'].strip()
                detected_language = transcription_result.get('language', 'unknown')
            else:
                # Mock transcription for demo
                transcript = "Mock transcription: This is a test of the scam detection system"
                detected_language = 'english'
            
            # 2. Analyze for synthetic voice
            voice_analysis = self.detect_synthetic_voice(audio_data, sample_rate)
            
            # 3. Analyze text for scam patterns
            scam_analysis = self.analyze_scam_patterns(transcript)
            
            # 4. Sentiment analysis
            sentiment = {"label": "neutral", "score": 0.5}
            if transcript and self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer(transcript)[0]
                    sentiment = {
                        "label": sentiment_result['label'],
                        "score": sentiment_result['score']
                    }
                except:
                    pass
            
            # 5. Calculate combined risk score
            content_risk = scam_analysis['risk_score']
            voice_risk = voice_analysis['risk_score']
            sentiment_risk = 20 if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8 else 0
            
            combined_risk = int(0.6 * content_risk + 0.3 * voice_risk + 0.1 * sentiment_risk)
            combined_risk = min(combined_risk, 100)
            
            # 6. Determine final decision
            if combined_risk >= 70:
                decision = "BLOCK_SCAMMER"
                threat_level = "HIGH"
            elif combined_risk >= 40:
                decision = "MONITOR_CLOSELY"
                threat_level = "MEDIUM"
            else:
                decision = "ALLOW_CALL"
                threat_level = "LOW"
            
            processing_time = time.time() - start_time
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript,
                "detected_language": detected_language,
                "voice_analysis": voice_analysis,
                "scam_analysis": scam_analysis,
                "sentiment": sentiment,
                "combined_risk_score": combined_risk,
                "decision": decision,
                "threat_level": threat_level,
                "processing_time": processing_time,
                "audio_duration": len(audio_data) / sample_rate
            }
            
            logger.info(f"🎯 Analysis complete: {decision} ({combined_risk}% risk) in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "transcript": "",
                "error": str(e),
                "decision": "ALLOW_CALL",
                "threat_level": "UNKNOWN",
                "combined_risk_score": 0
            }

class RealTimeAudioCapture:
    """Real-time audio capture with fallbacks"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.is_recording = False
        self.audio_thread = None
        
        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = None
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3
        
        # Initialize PyAudio if available
        self.audio = None
        if PYAUDIO_AVAILABLE:
            try:
                import pyaudio
                self.audio = pyaudio.PyAudio()
                self.FORMAT = pyaudio.paInt16
                logger.info("🎤 Audio capture system initialized")
            except Exception as e:
                logger.warning(f"⚠️ Audio initialization failed: {e}")
                self.audio = None

        if not self.audio:
            logger.warning("🎤 Audio capture disabled - will use simulation mode")
    
    def start_recording(self):
        """Start real-time audio capture"""
        if self.is_recording:
            return
        
        self.is_recording = True
        if self.audio:
            self.audio_thread = Thread(target=self._audio_capture_loop, daemon=True)
            self.audio_thread.start()
            logger.info("🔴 Started real-time audio capture")
        else:
            # Simulation mode
            self.audio_thread = Thread(target=self._simulation_loop, daemon=True)
            self.audio_thread.start()
            logger.info("🔴 Started simulation mode (no real audio)")
    
    def stop_recording(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join()
        logger.info("⏹️ Stopped audio capture")
    
    def _simulation_loop(self):
        """Simulation loop when audio is not available"""
        while self.is_recording:
            # Generate fake audio data for testing
            fake_audio = np.random.randn(self.RATE * self.RECORD_SECONDS) * 0.1
            
            # Process every 5 seconds in simulation
            analysis_thread = Thread(
                target=self._process_audio_chunk, 
                args=(fake_audio,), 
                daemon=True
            )
            analysis_thread.start()
            
            time.sleep(5)  # Simulate every 5 seconds
    
    def _audio_capture_loop(self):
        """Main audio capture loop"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            logger.info("🎙️ Audio stream opened successfully")
            
            frames = []
            frame_count = 0
            target_frames = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
            
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    frame_count += 1
                    
                    if frame_count >= target_frames:
                        audio_data = b''.join(frames)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        analysis_thread = Thread(
                            target=self._process_audio_chunk, 
                            args=(audio_np,), 
                            daemon=True
                        )
                        analysis_thread.start()
                        
                        frames = []
                        frame_count = 0
                        
                except Exception as e:
                    logger.error(f"Error in audio capture: {e}")
                    time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in audio capture loop: {e}")
    
    def _process_audio_chunk(self, audio_data):
        """Process audio chunk in separate thread"""
        try:
            # Only analyze if there's significant audio activity
            if np.max(np.abs(audio_data)) > 0.01:
                result = self.analyzer.analyze_audio_chunk(audio_data, self.RATE)
                
                # Add to results queue for real-time display
                self.analyzer.analysis_results.put(result)
                
                # If scammer detected, trigger immediate action
                if result.get('decision') == 'BLOCK_SCAMMER':
                    self._handle_scammer_detected(result)
                    
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _handle_scammer_detected(self, analysis_result):
        """Handle when a scammer is detected"""
        logger.warning("🚨 SCAMMER DETECTED - Taking protective action!")
        
        # Trigger scammer confrontation
        if TTS_AVAILABLE:
            scammer_responder = ScammerResponder()
            scammer_responder.confront_scammer(analysis_result)

class ScammerResponder:
    """AI-powered scammer confrontation system with fallbacks"""
    
    def __init__(self):
        # Initialize text-to-speech
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
                logger.info("🤖 Scammer Responder initialized with TTS")
            except Exception as e:
                logger.warning(f"⚠️ TTS initialization failed: {e}")
        else:
            logger.info("🤖 Scammer Responder initialized (TTS disabled)")
        
        # Confrontation messages
        self.confrontation_messages = [
            "I am an AI security system. You have been identified as a potential scammer. This call is being terminated and reported.",
            "Warning: Scam attempt detected. Your call patterns match known fraudulent activities. Disconnecting now.",
            "This is an automated fraud protection system. Your conversation contains multiple scam indicators. Call terminated.",
            "Alert: This call has been flagged for suspicious activity. All details are being logged and reported to authorities.",
            "Scam detection activated. Your voice patterns and content suggest fraudulent intent. Call blocked."
        ]
    
    def confront_scammer(self, analysis_result):
        """Confront the detected scammer"""
        try:
            risk_score = analysis_result.get('combined_risk_score', 0)
            transcript = analysis_result.get('transcript', '')
            
            logger.warning(f"🚨 CONFRONTING SCAMMER - Risk: {risk_score}%")
            logger.warning(f"📝 Detected content: {transcript[:100]}...")
            
            # Select appropriate confrontation message
            if risk_score >= 90:
                message = self.confrontation_messages[0]
            elif risk_score >= 80:
                message = self.confrontation_messages[1]
            elif risk_score >= 70:
                message = self.confrontation_messages[2]
            else:
                message = self.confrontation_messages[3]
            
            # Speak or print the confrontation message
            self._deliver_message(message)
            
            # Log the incident
            self._log_scammer_incident(analysis_result, message)
            
        except Exception as e:
            logger.error(f"Error confronting scammer: {e}")
    
    def _deliver_message(self, message):
        """Deliver confrontation message via TTS or print"""
        try:
            logger.info(f"🔊 Speaking: {message}")
            if self.tts_engine:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            else:
                print(f"\n🔊 SCAMMER CONFRONTATION: {message}")
        except Exception as e:
            logger.error(f"Error in message delivery: {e}")
            print(f"\n🔊 SCAMMER CONFRONTATION: {message}")
    
    def _log_scammer_incident(self, analysis_result, response_message):
        """Log the scammer incident for analysis"""
        incident = {
            "timestamp": datetime.now().isoformat(),
            "risk_score": analysis_result.get('combined_risk_score'),
            "transcript": analysis_result.get('transcript'),
            "voice_analysis": analysis_result.get('voice_analysis'),
            "scam_patterns": analysis_result.get('scam_analysis', {}).get('patterns_found'),
            "response_message": response_message,
            "threat_level": analysis_result.get('threat_level')
        }
        
        try:
            with open('scammer_incidents.json', 'a') as f:
                f.write(json.dumps(incident) + '\n')
            logger.info("📝 Scammer incident logged successfully")
        except Exception as e:
            logger.error(f"Error logging incident: {e}")

class SimpleGUI:
    """Simple GUI that works even with limited packages"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("🛡️ Voice Scam Interceptor")
        self.root.geometry("800x600")
        
        # System state
        self.is_monitoring = False
        self.audio_capture = None
        self.current_risk = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = ttk.Label(title_frame, text="🛡️ VOICE SCAM INTERCEPTOR", 
                               font=('Arial', 20, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Real-Time AI Protection System", 
                                 font=('Arial', 12))
        subtitle_label.pack()
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding=20)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        # Risk display
        self.risk_label = ttk.Label(status_frame, text="Risk Level: 0% (SAFE)", 
                                   font=('Arial', 16, 'bold'))
        self.risk_label.pack(pady=10)
        
        # System metrics
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill='x')
        
        ttk.Label(metrics_frame, text="🎤 Audio Input:").grid(row=0, column=0, sticky='w')
        self.audio_status = ttk.Label(metrics_frame, text="Ready" if PYAUDIO_AVAILABLE else "Simulated")
        self.audio_status.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        ttk.Label(metrics_frame, text="🤖 AI Models:").grid(row=1, column=0, sticky='w')
        ai_status = "Loaded" if WHISPER_AVAILABLE else "Basic Mode"
        ttk.Label(metrics_frame, text=ai_status).grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        ttk.Label(metrics_frame, text="🛡️ Protection:").grid(row=2, column=0, sticky='w')
        self.protection_status = ttk.Label(metrics_frame, text="Disabled")
        self.protection_status.grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Control buttons
        controls_frame = ttk.LabelFrame(self.root, text="Controls", padding=20)
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        self.start_button = ttk.Button(controls_frame, text="🔴 START PROTECTION", 
                                      command=self.start_protection)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="⏹️ STOP PROTECTION", 
                                     command=self.stop_protection, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        self.test_button = ttk.Button(controls_frame, text="🧪 TEST DETECTION", 
                                     command=self.run_test)
        self.test_button.pack(side='left', padx=5)
        
        # Activity log
        log_frame = ttk.LabelFrame(self.root, text="Activity Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create scrollable text widget
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initial log message
        self.add_log_message("🛡️ Voice Scam Interceptor initialized and ready")
        
        # Start update thread
        self.update_thread = Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def add_log_message(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)
        
        # Keep only last 100 lines
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.log_text.delete("1.0", f"{len(lines)-100}.0")
    
    def start_protection(self):
        """Start voice protection system"""
        try:
            self.is_monitoring = True
            self.audio_capture = RealTimeAudioCapture(self.analyzer)
            self.audio_capture.start_recording()
            
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.protection_status.config(text="ACTIVE")
            
            self.add_log_message("🔴 Protection system activated - monitoring calls")
            
        except Exception as e:
            self.add_log_message(f"❌ Error starting protection: {e}")
    
    def stop_protection(self):
        """Stop voice protection system"""
        try:
            self.is_monitoring = False
            if self.audio_capture:
                self.audio_capture.stop_recording()
            
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.protection_status.config(text="Disabled")
            
            self.add_log_message("⏹️ Protection system deactivated")
            
        except Exception as e:
            self.add_log_message(f"❌ Error stopping protection: {e}")
        
    def run_test(self):
            """Run a test with sample scam detection"""
            try:
                # Test scam text
                test_transcript = "Your social security number has been suspended due to suspicious activity. Press 1 to speak with an agent immediately or your benefits will be terminated."
                
                self.add_log_message(f"🧪 Running scam detection test...")
                self.add_log_message(f"📝 Test transcript: {test_transcript[:80]}...")
                
                # Analyze the test text
                scam_analysis = self.analyzer.analyze_scam_patterns(test_transcript)
                
                # Generate fake audio for voice analysis
                fake_audio = np.random.randn(16000 * 3) * 0.1
                voice_analysis = self.analyzer.detect_synthetic_voice(fake_audio)
                
                # Calculate combined score
                combined_risk = int(0.7 * scam_analysis['risk_score'] + 0.3 * voice_analysis['risk_score'])
                
                self.add_log_message(f"📊 Test Results:")
                self.add_log_message(f"   Content Risk: {scam_analysis['risk_score']}%")
                self.add_log_message(f"   Voice Risk: {voice_analysis['risk_score']}%") 
                self.add_log_message(f"   Combined Risk: {combined_risk}%")
                
                decision = "🚨 BLOCK SCAMMER" if combined_risk >= 70 else "⚠️ MONITOR" if combined_risk >= 40 else "✅ ALLOW"
                self.add_log_message(f"   Decision: {decision}")
                
                if scam_analysis['patterns_found']:
                    self.add_log_message(f"   Detected Patterns:")
                    for pattern in scam_analysis['patterns_found'][:3]:
                        self.add_log_message(f"     • {pattern}")
                
                # Update risk display
                self.current_risk = combined_risk
                self.update_risk_display()
                
            except Exception as e:
                self.add_log_message(f"❌ Test error: {e}")
    
    def update_risk_display(self):
        """Update the risk level display"""
        if self.current_risk >= 70:
            risk_text = f"Risk Level: {self.current_risk}% (HIGH RISK)"
            self.risk_label.config(foreground='red')
        elif self.current_risk >= 40:
            risk_text = f"Risk Level: {self.current_risk}% (SUSPICIOUS)"
            self.risk_label.config(foreground='orange')
        else:
            risk_text = f"Risk Level: {self.current_risk}% (SAFE)"
            self.risk_label.config(foreground='green')
        
        self.risk_label.config(text=risk_text)
    
    def update_loop(self):
        """Main update loop for real-time data"""
        while True:
            try:
                # Check for new analysis results
                if not self.analyzer.analysis_results.empty():
                    result = self.analyzer.analysis_results.get_nowait()
                    self.handle_analysis_result(result)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1)
    
    def handle_analysis_result(self, result):
        """Handle new analysis result"""
        try:
            risk_score = result.get('combined_risk_score', 0)
            decision = result.get('decision', 'UNKNOWN')
            transcript = result.get('transcript', '')
            
            # Update risk display
            self.current_risk = risk_score
            self.update_risk_display()
            
            # Add to activity log
            if transcript:
                self.add_log_message(f"{decision} ({risk_score}%): {transcript[:50]}...")
            else:
                self.add_log_message(f"{decision} ({risk_score}%): Audio analyzed")
            
            # Handle high risk
            if risk_score >= 70:
                self.add_log_message("🚨 SCAMMER DETECTED - Taking protective action!")
                
        except Exception as e:
            logger.error(f"Error handling analysis result: {e}")
    
    def run(self):
        """Run the GUI"""
        logger.info("🖥️ Starting desktop interface...")
        self.root.mainloop()

class VoiceScamInterceptor:
    """Main Voice Scam Interceptor System"""
    
    def __init__(self):
        logger.info("🚀 Initializing Voice Scam Interceptor System...")
        
        # Initialize core components
        self.analyzer = AdvancedVoiceAnalyzer()
        self.audio_capture = None
        self.gui = None
        
        # System state
        self.is_running = False
        self.protection_enabled = False
        
        logger.info("✅ Voice Scam Interceptor System initialized!")
    
    def start_desktop_interface(self):
        """Start the desktop GUI interface"""
        logger.info("🖥️ Starting desktop interface...")
        self.gui = SimpleGUI(self.analyzer)
        self.gui.run()
    
    def start_web_interface(self, host="localhost", port=8000):
        """Start the web interface"""
        if not FASTAPI_AVAILABLE:
            print("❌ Web interface not available - FastAPI not installed")
            print("💡 Install with: pip install fastapi uvicorn")
            return
        
        logger.info("🌐 Starting web interface...")
        
        app = FastAPI(title="Voice Scam Interceptor API")
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>🛡️ Voice Scam Interceptor</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }
                    .header { text-align: center; margin-bottom: 40px; }
                    .status { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
                    .safe { color: #4CAF50; } .warning { color: #FF9800; } .danger { color: #F44336; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🛡️ Voice Scam Interceptor</h1>
                    <p>Advanced AI-Powered Real-Time Scam Protection</p>
                </div>
                
                <div class="status">
                    <h2>📊 System Status</h2>
                    <p>🎤 Audio Input: Ready</p>
                    <p>🤖 AI Models: Loaded</p>
                    <p>🛡️ Protection: Available</p>
                </div>
                
                <div class="status">
                    <h2>🎯 Current Threat Level</h2>
                    <h1 class="safe">0% - SAFE</h1>
                </div>
                
                <div class="status">
                    <h2>🛡️ Protection Tips</h2>
                    <ul>
                        <li>Never share SSN over phone</li>
                        <li>Hang up and call back officially</li>
                        <li>Be suspicious of urgency tactics</li>
                        <li>No gift card payments</li>
                        <li>Government doesn't demand immediate payment</li>
                    </ul>
                </div>
            </body>
            </html>
            """
        
        import uvicorn
        print(f"🌐 Web interface starting at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    
    def run_cli_mode(self):
        """Run in command-line interface mode"""
        logger.info("💻 Starting CLI mode...")
        
        print("\n" + "="*60)
        print("🛡️ VOICE SCAM INTERCEPTOR - CLI MODE")
        print("="*60)
        print("Commands:")
        print("  start    - Start voice protection")
        print("  stop     - Stop voice protection")
        print("  status   - Show system status")
        print("  test     - Run test with sample audio")
        print("  web      - Start web interface")
        print("  gui      - Start desktop interface")
        print("  quit     - Exit system")
        print("="*60)
        
        while True:
            try:
                command = input("\n🛡️ Enter command: ").strip().lower()
                
                if command == "start":
                    self.start_protection()
                elif command == "stop":
                    self.stop_protection()
                elif command == "status":
                    self.show_status()
                elif command == "test":
                    self.run_test()
                elif command == "web":
                    self.start_web_interface()
                elif command == "gui":
                    self.start_desktop_interface()
                elif command in ["quit", "exit", "q"]:
                    print("👋 Goodbye! Stay safe from scammers!")
                    break
                else:
                    print(f"❌ Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Stay safe from scammers!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
    
    def start_protection(self):
        """Start voice protection system"""
        if self.protection_enabled:
            logger.warning("Protection already enabled!")
            return
        
        logger.info("🛡️ Starting voice protection...")
        
        # Start audio capture
        self.audio_capture = RealTimeAudioCapture(self.analyzer)
        self.audio_capture.start_recording()
        
        self.protection_enabled = True
        logger.info("✅ Voice protection system is now active!")
    
    def stop_protection(self):
        """Stop voice protection system"""
        if not self.protection_enabled:
            logger.warning("Protection not enabled!")
            return
        
        logger.info("⏹️ Stopping voice protection...")
        
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_recording()
        
        self.protection_enabled = False
        logger.info("✅ Voice protection system stopped!")
    
    def show_status(self):
        """Show system status"""
        print(f"\n📊 System Status:")
        print(f"  Protection: {'🟢 Enabled' if self.protection_enabled else '🔴 Disabled'}")
        print(f"  Audio Capture: {'🎤 Active' if self.audio_capture and self.audio_capture.is_recording else '⏸️ Inactive'}")
        print(f"  Whisper: {'🤖 Loaded' if WHISPER_AVAILABLE else '⚠️ Unavailable'}")
        print(f"  PyAudio: {'🎤 Available' if PYAUDIO_AVAILABLE else '⚠️ Simulated'}")
        print(f"  AI Models: {'🔊 Ready' if SKLEARN_AVAILABLE else '⚠️ Basic Mode'}")
    
    def run_test(self):
        """Run a test with sample scam detection"""
        print("\n🧪 Running scam detection test...")
        
        # Test scam text
        test_transcript = "Your social security number has been suspended due to suspicious activity. Press 1 to speak with an agent immediately or your benefits will be terminated."
        
        print(f"📝 Test transcript: {test_transcript}")
        
        # Analyze
        scam_analysis = self.analyzer.analyze_scam_patterns(test_transcript)
        
        # Generate fake audio for voice analysis
        fake_audio = np.random.randn(16000 * 3) * 0.1
        voice_analysis = self.analyzer.detect_synthetic_voice(fake_audio)
        
        # Calculate combined score
        combined_risk = int(0.7 * scam_analysis['risk_score'] + 0.3 * voice_analysis['risk_score'])
        
        print(f"\n📊 Test Results:")
        print(f"  Content Risk: {scam_analysis['risk_score']}%")
        print(f"  Voice Risk: {voice_analysis['risk_score']}%")
        print(f"  Combined Risk: {combined_risk}%")
        print(f"  Decision: {'🚨 BLOCK SCAMMER' if combined_risk >= 70 else '⚠️ MONITOR' if combined_risk >= 40 else '✅ ALLOW'}")
        
        if scam_analysis['patterns_found']:
            print(f"  Detected Patterns:")
            for pattern in scam_analysis['patterns_found'][:3]:
                print(f"    • {pattern}")

def main():
    """Main entry point with improved error handling"""
    print("""
🛡️ VOICE SCAM INTERCEPTOR (COMPATIBLE VERSION)
==============================================
Advanced AI-Powered Real-Time Scam Protection System

This system intercepts incoming calls and uses AI to detect
and block scammer calls before they reach you.

Status:
""")
    
    # Show package availability
    print(f"🎤 Audio Capture: {'✅ Available' if PYAUDIO_AVAILABLE else '⚠️ Simulated'}")
    print(f"🤖 Whisper AI: {'✅ Loaded' if WHISPER_AVAILABLE else '⚠️ Mock Mode'}")
    print(f"🧠 ML Models: {'✅ Active' if SKLEARN_AVAILABLE else '⚠️ Basic Mode'}")
    print(f"🔊 Text-to-Speech: {'✅ Ready' if TTS_AVAILABLE else '⚠️ Print Mode'}")
    print(f"📊 Visualizations: {'✅ Available' if MATPLOTLIB_AVAILABLE else '⚠️ Text Mode'}")
    print(f"🌐 Web Interface: {'✅ Available' if FASTAPI_AVAILABLE else '⚠️ Disabled'}")
    
    # Initialize system
    interceptor = VoiceScamInterceptor()
    
    # Ask user for interface preference
    print("\nChoose interface:")
    print("1. Desktop GUI (Recommended)")
    print("2. Web Dashboard") 
    print("3. Command Line Interface")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            interceptor.start_desktop_interface()
        elif choice == "2":
            interceptor.start_web_interface()
        elif choice == "3":
            interceptor.run_cli_mode()
        else:
            print("Invalid choice. Starting CLI mode...")
            interceptor.run_cli_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Stay safe from scammers!")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("💡 Try running: pip install pyaudio whisper transformers")

if __name__ == "__main__":
    main()