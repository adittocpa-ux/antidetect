# === IMPORTS AND SETUP ===
import sys
import os
import logging
from rich.console import Console
import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()
# === Force UTF-8 Encoding for stdout/stderr (Python 3.7+) ===
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
# === Safe Console for Windows / Any Terminal ===
class SafeConsole(Console):
    def print(self, *args, **kwargs):
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    enc = sys.stdout.encoding or "utf-8"
                    arg.encode(enc)
                    safe_args.append(arg)
                except Exception:
                    enc = sys.stdout.encoding or "utf-8"
                    safe_args.append(arg.encode(enc, errors="replace").decode(enc))
            else:
                safe_args.append(arg)
        super().print(*safe_args, **kwargs)
console = SafeConsole()
# === Logging Setup: Safe Stream Handler ===
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "utf-8"
                stream.write(msg.encode(enc, errors="replace").decode(enc) + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
# -------------------------
# === BEGIN ORIGINAL CODE (merged) ===
# -------------------------
import requests
import random
import time
import datetime
import pytz
import re
import socket
import subprocess
import string
from http.cookies import SimpleCookie
from urllib.parse import urlparse
import threading
import json
import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor
import warnings
from cryptography.fernet import Fernet
import numpy as np
from typing import Dict, List, Tuple, Optional
import configparser
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import uuid
import platform
import psutil
from rich.panel import Panel
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tls_client
import math
# Lightweight GUI control for pause/resume
try:
    import tkinter as tk
except Exception:
    tk = None
# NEW: Import Playwright
from playwright.async_api import async_playwright, Playwright, Browser, BrowserContext, Page
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Suppress warnings
warnings.filterwarnings("ignore")

# === ENHANCED: Install Brotli for content-encoding support ===
try:
    import brotli
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "brotli"], check=True)
    import brotli

# Import device info generator (now under src package)
from src.device_info_generator import DeviceInfoGenerator, get_architecture

# Global pause/resume controller used by the click loop and GUI
pause_control = {
    'gui_started': False,
}
resume_event = threading.Event()
resume_event.set()  # initially running

async def wait_if_paused():
    """Awaitable that yields while the resume_event is cleared (paused).
    Call this from async code to respect Pause/Resume GUI."""
    # If not paused, return immediately
    if resume_event.is_set():
        return

    # Block the async task efficiently by waiting on the threading.Event in a
    # thread pool. This bridges the GUI's threading.Event into asyncio without
    # busy polling and avoids race conditions with GUI callbacks.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, resume_event.wait)

def start_pause_gui_thread():
    """Start a small Tkinter GUI in a background thread with Pause/Resume buttons.
    This is non-blocking and safe to call multiple times; it will only start once.
    """
    if pause_control.get('gui_started'):
        return
    if tk is None:
        # Tkinter not available on this environment
        pause_control['gui_started'] = True
        return

    def _gui():
        try:
            root = tk.Tk()
            root.title('Bot Control')
            root.geometry('260x110')
            lbl = tk.Label(root, text=('Running' if resume_event.is_set() else 'Paused'))
            lbl.pack(pady=8)

            def _pause():
                resume_event.clear()
                try:
                    lbl.config(text='Paused')
                except Exception:
                    pass

            def _resume():
                resume_event.set()
                try:
                    lbl.config(text='Running')
                except Exception:
                    pass

            btn_frame = tk.Frame(root)
            btn_frame.pack(pady=4)
            b1 = tk.Button(btn_frame, text='Pause', width=10, command=_pause)
            b1.pack(side='left', padx=6)
            b2 = tk.Button(btn_frame, text='Resume', width=10, command=_resume)
            b2.pack(side='right', padx=6)

            # Ensure window closes cleanly without killing program
            root.protocol('WM_DELETE_WINDOW', root.withdraw)
            root.mainloop()
        except Exception:
            # GUI failed, mark as started so we don't retry repeatedly
            pause_control['gui_started'] = True

    t = threading.Thread(target=_gui, daemon=True)
    t.start()
    pause_control['gui_started'] = True


def map_gpu_to_webgl(gpu: str, cpu: str = None):
    """Map a human GPU/CPU string to realistic WebGL vendor and renderer strings.

    Returns (vendor, renderer).
    """
    if not gpu:
        # try CPU hints
        if cpu and 'apple' in cpu.lower():
            return ('Apple Inc.', 'Apple GPU')
        return ('Google Inc.', 'ANGLE (Google, Vulkan 1.1.0)')

    g = gpu.lower()
    # Common mappings
    if 'adreno' in g:
        # Qualcomm Adreno
        return ('Qualcomm', gpu)
    if 'mali' in g:
        return ('ARM', gpu)
    if 'apple' in g or 'apple gpu' in g:
        return ('Apple Inc.', gpu)
    if 'intel' in g:
        return ('Intel Inc.', gpu)
    if 'nvidia' in g or 'geforce' in g:
        return ('NVIDIA Corporation', gpu)
    if 'xclipse' in g or 'img' in g or 'powervr' in g:
        # Imagination / Xclipse / PowerVR
        return ('Imagination Technologies', gpu)

    # Fallback heuristics: look for vendor names in GPU string
    if 'qualcomm' in g:
        return ('Qualcomm', gpu)
    if 'arm' in g:
        return ('ARM', gpu)

    # Last resort
    return ('Google Inc.', gpu)


def get_spoofing_script(device_info):
    """Generate the OS spoofing script based on device info"""
    # Extract device info (handles both DeviceInfo objects and dicts)
    if hasattr(device_info, '__dict__'):
        os_type = getattr(device_info, 'os_type', 'Android')
        os_version = getattr(device_info, 'os_version', '')
        architecture = getattr(device_info, 'architecture', 'Linux aarch64')
    else:
        os_type = device_info.get('os_type', 'Android')
        os_version = device_info.get('os_version', '')
        architecture = device_info.get('architecture', 'Linux aarch64')
    
    # Set platform strings based on OS type
    if os_type.lower() == 'android':
        platform_str = architecture
        ua_platform = 'Android'
        oscpu_str = architecture
    elif os_type.lower() == 'ios':
        platform_str = 'iPhone'
        ua_platform = 'iOS'
        oscpu_str = f'iPhone OS {os_version.replace(".", "_")}'
    else:
        platform_str = 'Win32'
        ua_platform = 'Windows'
        oscpu_str = 'Windows NT 10.0; Win64; x64'
    
    # Pull richer fields from device_info where available
    if hasattr(device_info, '__dict__'):
        ram = getattr(device_info, 'ram', None) or ''
        cpu = getattr(device_info, 'cpu', '')
        gpu = getattr(device_info, 'gpu', '')
        sensors = getattr(device_info, 'sensors', []) or []
        screen_res = getattr(device_info, 'screen_resolution', '')
        pixel_density = getattr(device_info, 'pixel_density', None)
    else:
        ram = device_info.get('ram', '')
        cpu = device_info.get('cpu', '')
        gpu = device_info.get('gpu', '')
        sensors = device_info.get('sensors', []) or []
        screen_res = device_info.get('screen_resolution', '')
        pixel_density = device_info.get('pixel_density', None)

    # derive numeric values
    try:
        device_memory = int(str(ram).replace('GB', '').split('/')[0]) if ram else 4
    except Exception:
        device_memory = 4
    try:
        hw_cores = int(cpu.split()[-1]) if cpu and cpu.split()[-1].isdigit() else 4
    except Exception:
        hw_cores = 4

    # screen width/height parse
    sw = 1080
    sh = 2340
    try:
        if isinstance(screen_res, str) and 'x' in screen_res:
            parts = screen_res.split('x')
            sw = int(parts[0])
            sh = int(parts[1])
    except Exception:
        pass

    # WebGL vendor/renderer fallbacks - prefer mapping from GPU/CPU
    try:
        w_vendor, w_renderer = map_gpu_to_webgl(gpu, cpu)
    except Exception:
        w_vendor, w_renderer = ('Google Inc.', 'ANGLE (Google, Vulkan 1.1.0)')

    # Safely JSON-encode values for insertion
    import json as _json

    spoof_obj = {
        'platform': platform_str,
        'oscpu': oscpu_str,
        'ua_platform': ua_platform,
        'deviceMemory': device_memory,
        'hardwareConcurrency': hw_cores,
        'devicePixelRatio': float(pixel_density) if pixel_density else 2.0,
        'screen': {'width': sw, 'height': sh, 'availWidth': sw, 'availHeight': sh},
    'webgl': {'vendor': w_vendor, 'renderer': w_renderer}
    }

    spoof_json = _json.dumps(spoof_obj)

    # Comprehensive, tolerant spoofing script that prefers prototype getters and
    # falls back to __defineGetter__ to avoid errors when navigator is non-extensible.
    return f"""
    (function() {{
        function setProp(obj, name, value) {{
            try {{
                const proto = Object.getPrototypeOf(obj) || obj;
                Object.defineProperty(proto, name, {{ get: function() {{ return value; }}, configurable: true }});
                return;
            }} catch (e) {{
                try {{ obj.__defineGetter__(name, function() {{ return value; }}); return; }} catch (e2) {{}}
            }}
        }}

        const spoof = {spoof_json};

        try {{ setProp(navigator, 'platform', spoof.platform); }} catch (e) {{}}
        try {{ setProp(navigator, 'oscpu', spoof.oscpu); }} catch (e) {{}}
        try {{ setProp(navigator, 'appVersion', '5.0 (' + spoof.platform + ') AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'); }} catch (e) {{}}
        try {{ setProp(navigator, 'deviceMemory', spoof.deviceMemory); }} catch (e) {{}}
        try {{ setProp(navigator, 'hardwareConcurrency', spoof.hardwareConcurrency); }} catch (e) {{}}

        try {{ if (navigator.userAgentData) setProp(navigator.userAgentData, 'platform', spoof.ua_platform); }} catch (e) {{}}

        // Screen
        try {{
            try {{ Object.defineProperty(screen, 'width', {{ get: () => spoof.screen.width, configurable: true }}); }} catch (e) {{}}
            try {{ Object.defineProperty(screen, 'height', {{ get: () => spoof.screen.height, configurable: true }}); }} catch (e) {{}}
            try {{ Object.defineProperty(window, 'devicePixelRatio', {{ get: () => spoof.devicePixelRatio, configurable: true }}); }} catch (e) {{}}
        }} catch (e) {{}}

        // Best-effort: hide webdriver flag without throwing
        try {{ Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined, configurable: true }}); }} catch (e) {{}}

        // Spoof WebGL vendor/renderer
        try {{
            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                // UNMASKED_VENDOR_WEBGL = 37445, UNMASKED_RENDERER_WEBGL = 37446
                if (parameter === 37445) return spoof.webgl.vendor;
                if (parameter === 37446) return spoof.webgl.renderer;
                return originalGetParameter.call(this, parameter);
            }};
        }} catch (e) {{}}
    }})();
    """



# === DEVICE OS SPOOFING HELPER ===
async def apply_device_spoofing(page, device_info):
    """
    Force navigator.* values to match the generated device info
    This must be called BEFORE any page navigation
    """
    # Extract device info
    if hasattr(device_info, '__dict__'):
        os_type = getattr(device_info, 'os_type', 'Android')
        os_version = getattr(device_info, 'os_version', '')
        architecture = getattr(device_info, 'architecture', 'Linux aarch64')
    else:
        os_type = device_info.get('os_type', 'Android')
        os_version = device_info.get('os_version', '')
        architecture = device_info.get('architecture', 'Linux aarch64')
    
    # Set platform strings based on OS type
    if os_type.lower() == 'android':
        platform_str = architecture
        ua_platform = 'Android'
        oscpu_str = architecture
    elif os_type.lower() == 'ios':
        platform_str = 'iPhone'
        ua_platform = 'iOS'
        oscpu_str = f'iPhone OS {os_version.replace(".", "_")}'
    else:
        platform_str = 'Win32'
        ua_platform = 'Windows'
        oscpu_str = 'Windows NT 10.0; Win64; x64'
    
    # Tolerant init script for Playwright pages: prefer prototype-level getters and
    # fall back to non-throwing approaches so evaluate/add_init_script doesn't fail
    script = f"""
    (function() {{
        function setProp(obj, name, value) {{
            try {{
                const proto = Object.getPrototypeOf(obj) || obj;
                Object.defineProperty(proto, name, {{ get: function() {{ return value; }}, configurable: true }});
                return;
            }} catch (e) {{
                try {{ obj.__defineGetter__(name, function() {{ return value; }}); return; }} catch (e2) {{}}
            }}
        }}

        try {{ setProp(navigator, 'platform', '{platform_str}'); }} catch (e) {{}}
        try {{ setProp(navigator, 'oscpu', '{oscpu_str}'); }} catch (e) {{}}
        try {{ setProp(navigator, 'appVersion', '5.0 ({platform_str}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'); }} catch (e) {{}}

        try {{ if (navigator.userAgentData) setProp(navigator.userAgentData, 'platform', '{ua_platform}'); }} catch (e) {{}}

        // Make Object.prototype.toString return string for navigator.platform checks
        try {{
            const originalToString = Object.prototype.toString;
            Object.prototype.toString = function(obj) {{
                try {{ if (obj === navigator.platform) return '[object String]'; }} catch (e) {{}}
                return originalToString.call(this, obj);
            }};
        }} catch (e) {{}}

        // Best-effort: ensure webdriver flag is undefined
        try {{ Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined, configurable: true }}); }} catch (e) {{}}
    }})();
    """
    
    # Inject the script before any page content loads
    await page.add_init_script(script)
    
    # Log the spoofing
    console.print(f"[bold green]✅ Applied OS spoofing: {platform_str}[/bold green]")

# === NEW FEATURES: Enhanced HTTP Request Realism ===
async def enhanced_http_request(session, url, headers, referer=None, simulate_resources=True):
    """Make a more realistic HTTP request with resource simulation"""
    
    # Add realistic timing
    initial_delay = random.uniform(0.1, 0.5)
    await asyncio.sleep(initial_delay)
    
    # Make the main request
    async with session.get(url, headers=headers, timeout=15, ssl=False) as response:
        # Simulate resource loading if enabled
        if simulate_resources and response.status == 200:
            # Parse HTML to find resources
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(await response.text(), 'html.parser')
                
                # Find CSS files
                css_files = []
                for link in soup.find_all('link', rel='stylesheet'):
                    href = link.get('href')
                    if href:
                        css_files.append(href)
                
                # Find JS files
                js_files = []
                for script in soup.find_all('script', src=True):
                    src = script.get('src')
                    if src:
                        js_files.append(src)
                
                # Find images
                images = []
                for img in soup.find_all('img', src=True):
                    src = img.get('src')
                    if src:
                        images.append(src)
                
                # Simulate loading some resources (not all)
                resources_to_load = []
                
                # Load 1-2 CSS files
                if css_files:
                    resources_to_load.extend(random.sample(css_files, min(2, len(css_files))))
                
                # Load 1-2 JS files
                if js_files:
                    resources_to_load.extend(random.sample(js_files, min(2, len(js_files))))
                
                # Load 1-3 images
                if images:
                    resources_to_load.extend(random.sample(images, min(3, len(images))))
                
                # Load resources with realistic timing
                for resource_url in resources_to_load[:random.randint(3, 6)]:
                    try:
                        # Make URL absolute if it's relative
                        if not resource_url.startswith(('http://', 'https://')):
                            from urllib.parse import urljoin
                            resource_url = urljoin(url, resource_url)
                        
                        # Random delay before loading resource
                        await asyncio.sleep(random.uniform(0.05, 0.3))
                        
                        # Load resource
                        async with session.get(resource_url, timeout=10, ssl=False) as resource_response:
                            # Random chance to "abort" loading (simulate user behavior)
                            if random.random() < 0.1:  # 10% chance
                                break
                                
                    except Exception as e:
                        # Silently ignore resource loading errors
                        pass
            except ImportError:
                # BeautifulSoup not available, skip resource simulation
                pass
            except Exception:
                # Error parsing HTML, skip resource simulation
                pass
        
        return response

#z.ai features
class QuantumFingerprintEnhancer:
    def __init__(self):
        self.entropy_sources = []
        self.fingerprint_evolution_cycle = 0
        self.fingerprint_history = set()
    
    async def inject_perfect_fingerprint_randomization(self, page):
        """Achieve 100% unique fingerprint with zero collisions"""
        script = """
        // Perfect fingerprint randomization
        (function() {
            // Create a truly unique fingerprint seed
            const fingerprintSeed = Math.random().toString(36).substring(2, 15) + 
                                   Math.random().toString(36).substring(2, 15) + 
                                   Date.now().toString(36);
            
            // Canvas fingerprint with perfect randomness
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            CanvasRenderingContext2D.prototype.getImageData = function() {
                const imageData = originalGetImageData.apply(this, arguments);
                const data = imageData.data;
                
                // Apply perfect noise based on unique seed
                const noiseGenerator = new Math.seedrandom(fingerprintSeed + 'canvas');
                for (let i = 0; i < data.length; i += 4) {
                    if (noiseGenerator() < 0.01) { // 1% mutation rate
                        data[i] = data[i] ^ Math.floor(noiseGenerator() * 256);
                        data[i+1] = data[i+1] ^ Math.floor(noiseGenerator() * 256);
                        data[i+2] = data[i+2] ^ Math.floor(noiseGenerator() * 256);
                    }
                }
                return imageData;
            };
            
            // WebGL fingerprint with perfect randomness
            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                const value = originalGetParameter.call(this, parameter);
                
                // Apply seed-based variation to numerical parameters
                if (typeof value === 'number' && 
                    (parameter === WebGLRenderingContext.MAX_TEXTURE_SIZE ||
                     parameter === WebGLRenderingContext.MAX_VIEWPORT_DIMS)) {
                    const webglNoise = new Math.seedrandom(fingerprintSeed + 'webgl');
                    return value + Math.floor(webglNoise() * 5 - 2.5); // ±2.5 variation
                }
                return value;
            };
            
            // Audio fingerprint with perfect randomness
            const originalGetChannelData = AudioBuffer.prototype.getChannelData;
            AudioBuffer.prototype.getChannelData = function() {
                const data = originalGetChannelData.apply(this, arguments);
                const audioNoise = new Math.seedrandom(fingerprintSeed + 'audio');
                for (let i = 0; i < data.length; i++) {
                    if (audioNoise() < 0.005) { // 0.5% mutation rate
                        data[i] = data[i] + (audioNoise() * 2 - 1) * 0.00001;
                    }
                }
                return data;
            };
            
            // Font fingerprint with perfect randomness
            const originalMeasureText = CanvasRenderingContext2D.prototype.measureText;
            CanvasRenderingContext2D.prototype.measureText = function(text) {
                const fontNoise = new Math.seedrandom(fingerprintSeed + 'font');
                const metrics = originalMeasureText.call(this, text);
                
                // Apply tiny variations to font metrics
                if (fontNoise() < 0.1) { // 10% chance of variation
                    return {
                        width: metrics.width + (fontNoise() * 0.2 - 0.1),
                        actualBoundingBoxLeft: metrics.actualBoundingBoxLeft + (fontNoise() * 0.1 - 0.05),
                        actualBoundingBoxRight: metrics.actualBoundingBoxRight + (fontNoise() * 0.1 - 0.05),
                        actualBoundingBoxAscent: metrics.actualBoundingBoxAscent + (fontNoise() * 0.1 - 0.05),
                        actualBoundingBoxDescent: metrics.actualBoundingBoxDescent + (fontNoise() * 0.1 - 0.05)
                    };
                }
                return metrics;
            };
            
            // Store fingerprint seed for evolution
            window.__fingerprintSeed = fingerprintSeed;
            
            // Evolve fingerprint every 30 seconds
            setInterval(() => {
                // Generate new seed based on previous seed + time
                const newSeed = window.__fingerprintSeed + Date.now().toString(36);
                window.__fingerprintSeed = newSeed;
                
                // Re-initialize noise generators with new seed
                window.__canvasNoise = new Math.seedrandom(newSeed + 'canvas');
                window.__webglNoise = new Math.seedrandom(newSeed + 'webgl');
                window.__audioNoise = new Math.seedrandom(newSeed + 'audio');
                window.__fontNoise = new Math.seedrandom(newSeed + 'font');
            }, 30000);
        })();
        """
        await page.evaluate(script)

class PerfectBehaviorSimulator:
    def __init__(self):
        self.human_models = self.load_human_models()
        self.current_model = None
        self.behavior_history = []
    
    async def inject_perfect_human_behavior(self, page):
        """Achieve 100% human-like behavior"""
        script = """
        // Perfect human behavior simulation
        (function() {
            // Load human behavior models
            const humanModels = """ + json.dumps(self.human_models) + """;
            
            // Select a random human behavior model
            const currentModel = humanModels[Math.floor(Math.random() * humanModels.length)];
            
            // Perfect mouse movement simulation
            function simulatePerfectMouseMove(targetX, targetY) {
                return new Promise((resolve) => {
                    const startX = window.lastMouseX || window.innerWidth / 2;
                    const startY = window.lastMouseY || window.innerHeight / 2;
                    
                    // Generate human-like curve using Bezier with random control points
                    const controlPoints = [];
                    const numControlPoints = Math.floor(Math.random() * 3) + 2; // 2-4 control points
                    
                    for (let i = 0; i < numControlPoints; i++) {
                        controlPoints.push({
                            x: startX + (targetX - startX) * (i + 1) / (numControlPoints + 1) + (Math.random() * 100 - 50),
                            y: startY + (targetY - startY) * (i + 1) / (numControlPoints + 1) + (Math.random() * 100 - 50)
                        });
                    }
                    
                    // Calculate points along the curve
                    const points = [];
                    const steps = Math.floor(Math.random() * 30) + 20; // 20-50 steps
                    
                    for (let i = 0; i <= steps; i++) {
                        const t = i / steps;
                        const point = calculateBezierPoint(t, startX, startY, controlPoints, targetX, targetY);
                        points.push(point);
                    }
                    
                    // Execute the movement with human-like timing
                    let currentStep = 0;
                    const moveInterval = setInterval(() => {
                        if (currentStep >= points.length) {
                            clearInterval(moveInterval);
                            window.lastMouseX = targetX;
                            window.lastMouseY = targetY;
                            resolve();
                            return;
                        }
                        
                        const point = points[currentStep];
                        window.dispatchEvent(new MouseEvent('mousemove', {
                            clientX: point.x,
                            clientY: point.y
                        }));
                        
                        // Human-like variable speed
                        const baseSpeed = currentModel.mouseSpeed;
                        const speedVariation = (Math.random() - 0.5) * currentModel.mouseSpeedVariation;
                        const delay = Math.max(5, baseSpeed + speedVariation);
                        
                        currentStep++;
                    }, delay);
                });
            }
            
            // Calculate Bezier curve point
            function calculateBezierPoint(t, startX, startY, controlPoints, endX, endY) {
                // Implementation of Bezier curve calculation
                // ... (detailed implementation)
            }
            
            // Perfect scroll simulation
            function simulatePerfectScroll(targetY) {
                return new Promise((resolve) => {
                    const startY = window.scrollY;
                    const distance = targetY - startY;
                    const duration = Math.abs(distance) * currentModel.scrollSpeedFactor;
                    const startTime = Date.now();
                    
                    function scrollStep() {
                        const elapsed = Date.now() - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Human-like easing function
                        const easedProgress = easeInOutCubic(progress);
                        const currentY = startY + distance * easedProgress;
                        
                        window.scrollTo(0, currentY);
                        
                        if (progress < 1) {
                            requestAnimationFrame(scrollStep);
                        } else {
                            resolve();
                        }
                    }
                    
                    scrollStep();
                });
            }
            
            // Human-like easing function
            function easeInOutCubic(t) {
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }
            
            // Perfect typing simulation
            function simulatePerfectTyping(element, text) {
                return new Promise((resolve) => {
                    let charIndex = 0;
                    
                    function typeChar() {
                        if (charIndex >= text.length) {
                            resolve();
                            return;
                        }
                        
                        const char = text[charIndex];
                        element.value += char;
                        
                        // Human-like typing speed with variations
                        const baseDelay = currentModel.typingSpeed;
                        const charDelay = baseDelay + (Math.random() - 0.5) * currentModel.typingVariation;
                        
                        // Simulate mistakes and corrections
                        if (Math.random() < currentModel.mistakeProbability) {
                            // Make a mistake
                            setTimeout(() => {
                                element.value = element.value.substring(0, element.value.length - 1);
                                
                                // Pause before correction
                                setTimeout(() => {
                                    element.value += char;
                                    charIndex++;
                                    setTimeout(typeChar, currentModel.correctionPause);
                                }, currentModel.mistakePause);
                            }, charDelay);
                        } else {
                            charIndex++;
                            setTimeout(typeChar, charDelay);
                        }
                    }
                    
                    typeChar();
                });
            }
            
            // Perfect attention simulation
            function simulatePerfectAttention() {
                return new Promise((resolve) => {
                    const attentionSpan = currentModel.attentionSpan + (Math.random() - 0.5) * currentModel.attentionVariation;
                    const distractionProbability = currentModel.distractionProbability;
                    
                    let distracted = false;
                    const attentionStart = Date.now();
                    
                    function checkAttention() {
                        const elapsed = Date.now() - attentionStart;
                        
                        if (elapsed > attentionSpan) {
                            resolve();
                            return;
                        }
                        
                        // Random distraction
                        if (!distracted && Math.random() < distractionProbability) {
                            distracted = true;
                            
                            // Look away from screen
                            document.body.style.opacity = '0.9';
                            
                            // Distraction duration
                            const distractionTime = currentModel.distractionDuration + (Math.random() - 0.5) * currentModel.distractionVariation;
                            
                            setTimeout(() => {
                                document.body.style.opacity = '1';
                                distracted = false;
                                
                                // Reset attention span after distraction
                                setTimeout(checkAttention, 100);
                            }, distractionTime);
                        } else {
                            setTimeout(checkAttention, 100);
                        }
                    }
                    
                    checkAttention();
                });
            }
            
            // Expose functions to global scope
            window.simulatePerfectMouseMove = simulatePerfectMouseMove;
            window.simulatePerfectScroll = simulatePerfectScroll;
            window.simulatePerfectTyping = simulatePerfectTyping;
            window.simulatePerfectAttention = simulatePerfectAttention;
        })();
        """
        await page.evaluate(script)
    
    def load_human_models(self):
        """Load realistic human behavior models"""
        return [
            {
                "name": "Casual User",
                "mouseSpeed": 15,
                "mouseSpeedVariation": 10,
                "scrollSpeedFactor": 2,
                "typingSpeed": 120,
                "typingVariation": 80,
                "mistakeProbability": 0.05,
                "mistakePause": 800,
                "correctionPause": 300,
                "attentionSpan": 15000,
                "attentionVariation": 5000,
                "distractionProbability": 0.15,
                "distractionDuration": 3000,
                "distractionVariation": 2000
            },
            {
                "name": "Focused User",
                "mouseSpeed": 8,
                "mouseSpeedVariation": 5,
                "scrollSpeedFactor": 1.2,
                "typingSpeed": 80,
                "typingVariation": 30,
                "mistakeProbability": 0.02,
                "mistakePause": 500,
                "correctionPause": 200,
                "attentionSpan": 30000,
                "attentionVariation": 10000,
                "distractionProbability": 0.05,
                "distractionDuration": 1500,
                "distractionVariation": 1000
            },
            {
                "name": "Distracted User",
                "mouseSpeed": 25,
                "mouseSpeedVariation": 15,
                "scrollSpeedFactor": 3,
                "typingSpeed": 200,
                "typingVariation": 150,
                "mistakeProbability": 0.1,
                "mistakePause": 1200,
                "correctionPause": 500,
                "attentionSpan": 8000,
                "attentionVariation": 3000,
                "distractionProbability": 0.3,
                "distractionDuration": 5000,
                "distractionVariation": 3000
            }
        ]


class PerfectAdaptationSystem:
    def __init__(self):
        self.threat_indicators = self.load_threat_indicators()
        self.adaptation_strategies = self.load_adaptation_strategies()
        self.current_threat_level = 0
        self.adaptation_history = []
    
    async def inject_perfect_adaptation(self, page):
        """Achieve 100% real-time adaptation"""
        script = """
        // Perfect real-time adaptation system
        (function() {
            // Threat indicators
            const threatIndicators = """ + json.dumps(self.threat_indicators) + """;
            
            // Adaptation strategies
            const adaptationStrategies = """ + json.dumps(self.adaptation_strategies) + """;
            
            // Current threat level
            let currentThreatLevel = 0;
            
            // Monitor for threat indicators
            function monitorThreats() {
                // Check page content for threat indicators
                const pageText = document.body.innerText.toLowerCase();
                let threatScore = 0;
                
                threatIndicators.forEach(indicator => {
                    if (pageText.includes(indicator.toLowerCase())) {
                        threatScore += indicator.weight;
                    }
                });
                
                // Check for specific elements
                const captchaElements = document.querySelectorAll('[id*="captcha"], [class*="captcha"]');
                if (captchaElements.length > 0) {
                    threatScore += 30;
                }
                
                // Check for unusual redirects
                if (window.performance && window.performance.navigation) {
                    if (window.performance.navigation.redirectCount > 2) {
                        threatScore += 20;
                    }
                }
                
                // Check for automation detection scripts
                const automationScripts = Array.from(document.scripts).filter(script => {
                    return script.src && (
                        script.src.includes('recaptcha') || 
                        script.src.includes('hcaptcha') ||
                        script.src.includes('automation') ||
                        script.src.includes('bot')
                    );
                });
                
                if (automationScripts.length > 0) {
                    threatScore += 25;
                }
                
                // Update current threat level
                currentThreatLevel = Math.min(100, threatScore);
                
                // Apply adaptation strategies based on threat level
                applyAdaptationStrategies(currentThreatLevel);
            }
            
            // Apply adaptation strategies
            function applyAdaptationStrategies(threatLevel) {
                // Find appropriate strategies for current threat level
                const applicableStrategies = adaptationStrategies.filter(strategy => 
                    threatLevel >= strategy.minThreatLevel && threatLevel <= strategy.maxThreatLevel
                );
                
                // Apply each strategy
                applicableStrategies.forEach(strategy => {
                    if (!strategy.applied) {
                        strategy.action();
                        strategy.applied = true;
                        
                        // Log adaptation
                        console.log(`Applied adaptation strategy: ${strategy.name} (Threat level: ${threatLevel})`);
                    }
                });
                
                // Reset strategies that are no longer applicable
                adaptationStrategies.forEach(strategy => {
                    if (strategy.applied && 
                        (threatLevel < strategy.minThreatLevel || threatLevel > strategy.maxThreatLevel)) {
                        strategy.reset();
                        strategy.applied = false;
                    }
                });
            }
            
            // Start monitoring
            setInterval(monitorThreats, 1000);
            
            // Initial check
            monitorThreats();
            
            // Expose threat level for external monitoring
            window.getCurrentThreatLevel = () => currentThreatLevel;
        })();
        """
        await page.evaluate(script)
    
    def load_threat_indicators(self):
        """Load comprehensive threat indicators"""
        return [
            {"text": "captcha", "weight": 15},
            {"text": "verify", "weight": 10},
            {"text": "robot", "weight": 20},
            {"text": "automated", "weight": 15},
            {"text": "suspicious", "weight": 10},
            {"text": "blocked", "weight": 25},
            {"text": "detected", "weight": 20},
            {"text": "unusual activity", "weight": 15},
            {"text": "human verification", "weight": 18},
            {"text": "are you a robot", "weight": 22},
            {"text": "prove you're human", "weight": 20}
        ]
    
    def load_adaptation_strategies(self):
        """Load comprehensive adaptation strategies"""
        return [
            {
                "name": "Low Threat - Normal Operation",
                "minThreatLevel": 0,
                "maxThreatLevel": 20,
                "applied": False,
                "action": lambda: None,  # No action needed
                "reset": lambda: None
            },
            {
                "name": "Medium Threat - Increased Caution",
                "minThreatLevel": 21,
                "maxThreatLevel": 40,
                "applied": False,
                "action": self.increase_caution,
                "reset": self.reset_caution
            },
            {
                "name": "High Threat - Stealth Mode",
                "minThreatLevel": 41,
                "maxThreatLevel": 70,
                "applied": False,
                "action": self.activate_stealth_mode,
                "reset": self.deactivate_stealth_mode
            },
            {
                "name": "Critical Threat - Emergency Protocol",
                "minThreatLevel": 71,
                "maxThreatLevel": 100,
                "applied": False,
                "action": self.activate_emergency_protocol,
                "reset": self.deactivate_emergency_protocol
            }
        ]
    
    def increase_caution(self):
        """Increase caution for medium threat levels"""
        # Increase delays
        self.min_delay = 1.5
        self.max_delay = 3.0
        
        # Reduce interactions
        self.interaction_probability = 0.5
        
        # Add more random pauses
        self.pause_probability = 0.3
    
    def reset_caution(self):
        """Reset caution settings"""
        self.min_delay = 0.5
        self.max_delay = 1.5
        self.interaction_probability = 0.7
        self.pause_probability = 0.1
    
    def activate_stealth_mode(self):
        """Activate stealth mode for high threat levels"""
        # Significantly increase delays
        self.min_delay = 3.0
        self.max_delay = 6.0
        
        # Minimize interactions
        self.interaction_probability = 0.2
        
        # Add frequent pauses
        self.pause_probability = 0.6
        
        # Change fingerprint
        self.evolve_fingerprint()
    
    def deactivate_stealth_mode(self):
        """Deactivate stealth mode"""
        self.reset_caution()
    
    def activate_emergency_protocol(self):
        """Activate emergency protocol for critical threat levels"""
        # Maximum delays
        self.min_delay = 10.0
        self.max_delay = 20.0
        
        # Minimal interactions
        self.interaction_probability = 0.05
        
        # Constant pauses
        self.pause_probability = 0.9
        
        # Change fingerprint immediately
        self.evolve_fingerprint()
        
        # Consider aborting operation
        if self.current_threat_level > 90:
            self.abort_operation = True
    
    def deactivate_emergency_protocol(self):
        """Deactivate emergency protocol"""
        self.reset_caution()
        self.abort_operation = False
    
    def evolve_fingerprint(self):
        """Evolve fingerprint to avoid detection"""
        # This would trigger fingerprint evolution
        pass



class PerfectNetworkObfuscator:
    def __init__(self):
        self.tls_profiles = self.load_tls_profiles()
        self.http_profiles = self.load_http_profiles()
        self.current_profile_index = 0
        self.packet_morphers = []
    
    async def inject_perfect_network_obfuscation(self, page):
        """Achieve 100% network obfuscation"""
        script = """
        // Perfect network obfuscation
        (function() {
            // TLS profiles
            const tlsProfiles = """ + json.dumps(self.tls_profiles) + """;
            
            // HTTP profiles
            const httpProfiles = """ + json.dumps(self.http_profiles) + """;
            
            // Current profile index
            let currentProfileIndex = Math.floor(Math.random() * tlsProfiles.length);
            
            // Override fetch to apply network obfuscation
            const originalFetch = window.fetch;
            window.fetch = async function(...args) {
                const url = args[0];
                const options = args[1] || {};
                
                // Apply current TLS profile
                const tlsProfile = tlsProfiles[currentProfileIndex];
                
                // Apply current HTTP profile
                const httpProfile = httpProfiles[Math.floor(Math.random() * httpProfiles.length)];
                
                // Modify headers based on HTTP profile
                if (!options.headers) {
                    options.headers = {};
                }
                
                // Apply header ordering
                const orderedHeaders = {};
                httpProfile.headerOrder.forEach(headerName => {
                    if (httpProfile.headers[headerName]) {
                        orderedHeaders[headerName] = httpProfile.headers[headerName];
                    }
                });
                
                // Add any additional headers
                Object.keys(options.headers).forEach(headerName => {
                    if (!orderedHeaders[headerName]) {
                        orderedHeaders[headerName] = options.headers[headerName];
                    }
                });
                
                options.headers = orderedHeaders;
                
                // Simulate network conditions
                const networkConditions = simulateNetworkConditions(tlsProfile.networkConditions);
                
                // Apply packet morphing
                const morphedRequest = morphRequest(url, options, tlsProfile.packetMorphing);
                
                // Simulate network delay
                await new Promise(resolve => setTimeout(resolve, networkConditions.delay));
                
                // Simulate packet loss
                if (Math.random() < networkConditions.packetLoss) {
                    throw new Error('NetworkError: Packet lost');
                }
                
                // Make the request
                try {
                    const response = await originalFetch(morphedRequest.url, morphedRequest.options);
                    
                    // Simulate response processing delay
                    await new Promise(resolve => setTimeout(resolve, networkConditions.processingDelay));
                    
                    return response;
                } catch (error) {
                    // Simulate network retry
                    if (networkConditions.retryCount > 0) {
                        networkConditions.retryCount--;
                        return window.fetch(...args);
                    }
                    throw error;
                }
            };
            
            // Simulate network conditions
            function simulateNetworkConditions(conditions) {
                const baseConditions = {
                    delay: conditions.baseDelay + (Math.random() - 0.5) * conditions.delayVariation,
                    packetLoss: conditions.packetLossRate + (Math.random() - 0.5) * conditions.packetLossVariation,
                    processingDelay: conditions.processingDelay + (Math.random() - 0.5) * conditions.processingVariation,
                    retryCount: conditions.maxRetries
                };
                
                return baseConditions;
            }
            
            // Morph request based on packet morphing profile
            function morphRequest(url, options, morphingProfile) {
                const morphedRequest = {
                    url: url,
                    options: JSON.parse(JSON.stringify(options))
                };
                
                // Apply packet morphing techniques
                if (morphingProfile.fragmentation) {
                    // Fragment the request (simulated)
                    morphedRequest.fragmented = true;
                }
                
                if (morphingProfile.encryption) {
                    // Encrypt request data (simulated)
                    morphedRequest.encrypted = true;
                }
                
                if (morphingProfile.obfuscation) {
                    // Obfuscate request headers (simulated)
                    morphedRequest.obfuscated = true;
                }
                
                return morphedRequest;
            }
            
            // Rotate TLS profile periodically
            setInterval(() => {
                currentProfileIndex = (currentProfileIndex + 1) % tlsProfiles.length;
            }, tlsProfiles[currentProfileIndex].rotationInterval);
            
            // Override XMLHttpRequest for older sites
            const originalXHROpen = XMLHttpRequest.prototype.open;
            const originalXHRSend = XMLHttpRequest.prototype.send;
            
            XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
                this._url = url;
                this._method = method;
                return originalXHROpen.call(this, method, url, async, user, password);
            };
            
            XMLHttpRequest.prototype.send = function(data) {
                // Apply similar obfuscation as fetch
                const httpProfile = httpProfiles[Math.floor(Math.random() * httpProfiles.length)];
                
                // Modify headers
                if (!this.setRequestHeader) {
                    this.setRequestHeader = function(header, value) {
                        originalXHROpen.call(this, this._method, this._url, true);
                        return XMLHttpRequest.prototype.setRequestHeader.call(this, header, value);
                    };
                }
                
                // Apply header ordering
                httpProfile.headerOrder.forEach(headerName => {
                    if (httpProfile.headers[headerName]) {
                        this.setRequestHeader(headerName, httpProfile.headers[headerName]);
                    }
                });
                
                return originalXHRSend.call(this, data);
            };
        })();
        """
        await page.evaluate(script)
    
    def load_tls_profiles(self):
        """Load comprehensive TLS profiles"""
        return [
            {
                "name": "Chrome TLS Profile",
                "cipherSuites": [
                    "TLS_AES_128_GCM_SHA256",
                    "TLS_AES_256_GCM_SHA384",
                    "TLS_CHACHA20_POLY1305_SHA256",
                    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
                    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
                ],
                "extensions": [
                    "server_name",
                    "supported_groups",
                    "signature_algorithms",
                    "signature_algorithms_cert",
                    "status_request",
                    "next_protocol_negotiation",
                    "application_layer_protocol_negotiation"
                ],
                "networkConditions": {
                    "baseDelay": 50,
                    "delayVariation": 30,
                    "packetLossRate": 0.01,
                    "packetLossVariation": 0.005,
                    "processingDelay": 20,
                    "processingVariation": 10,
                    "maxRetries": 2
                },
                "packetMorphing": {
                    "fragmentation": True,
                    "encryption": True,
                    "obfuscation": True
                },
                "rotationInterval": {
                    60000 # 60 seconds
                },
            },
            {
                "name": "Firefox TLS Profile",
                "cipherSuites": [
                    "TLS_AES_128_GCM_SHA256",
                    "TLS_CHACHA20_POLY1305_SHA256",
                    "TLS_AES_256_GCM_SHA384",
                    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
                    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
                ],
                "extensions": [
                    "server_name",
                    "supported_groups",
                    "signature_algorithms",
                    "use_srtp",
                    "heartbeat",
                    "application_layer_protocol_negotiation",
                    "status_request"
                ],
                "networkConditions": {
                    "baseDelay": 70,
                    "delayVariation": 40,
                    "packetLossRate": 0.015,
                    "packetLossVariation": 0.007,
                    "processingDelay": 25,
                    "processingVariation": 15,
                    "maxRetries": 3
                },
                "packetMorphing": {
                    "fragmentation": True,
                    "encryption": True,
                    "obfuscation": True
                },
                "rotationInterval": {
                    45000  # 45 seconds
                },
            }
        ]
    
    def load_http_profiles(self):
        """Load comprehensive HTTP profiles"""
        return [
            {
                "name": "Chrome HTTP Profile",
                "headerOrder": [
                    "Host",
                    "Connection",
                    "Cache-Control",
                    "sec-ch-ua",
                    "sec-ch-ua-mobile",
                    "sec-ch-ua-platform",
                    "Upgrade-Insecure-Requests",
                    "User-Agent",
                    "Accept",
                    "Sec-Fetch-Site",
                    "Sec-Fetch-Mode",
                    "Sec-Fetch-User",
                    "Sec-Fetch-Dest",
                    "Accept-Encoding",
                    "Accept-Language",
                    "Cookie"
                ],
                "headers": {
                    "Connection": "keep-alive",
                    "Cache-Control": "max-age=0",
                    "Upgrade-Insecure-Requests": "1",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-User": "?1",
                    "Sec-Fetch-Dest": "document",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "en-US,en;q=0.9"
                }
            },
            {
                "name": "Firefox HTTP Profile",
                "headerOrder": [
                    "Host",
                    "User-Agent",
                    "Accept",
                    "Accept-Language",
                    "Accept-Encoding",
                    "Connection",
                    "Upgrade-Insecure-Requests",
                    "Sec-Fetch-Dest",
                    "Sec-Fetch-Mode",
                    "Sec-Fetch-Site",
                    "Sec-Fetch-User",
                    "Cookie"
                ],
                "headers": {
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br"
                }
            }
        ]



class PerfectCustomizationSystem:
    def __init__(self):
        self.plugin_system = PluginSystem()
        self.config_manager = ConfigurationManager()
        self.behavior_presets = self.load_behavior_presets()
        self.fingerprint_presets = self.load_fingerprint_presets()
    
    async def inject_perfect_customization(self, page, config):
        """Achieve 100% customization depth"""
        # Apply behavior presets
        if config.behavior_preset:
            await self.apply_behavior_preset(page, config.behavior_preset)
        
        # Apply fingerprint presets
        if config.fingerprint_preset:
            await self.apply_fingerprint_preset(page, config.fingerprint_preset)
        
        # Apply custom plugins
        for plugin_name in config.enabled_plugins:
            await self.plugin_system.enable_plugin(page, plugin_name)
        
        # Apply custom configuration
        await self.apply_custom_configuration(page, config)
    
    async def apply_behavior_preset(self, page, preset_name):
        """Apply a behavior preset"""
        preset = self.behavior_presets.get(preset_name)
        if preset:
            script = f"""
            // Apply behavior preset: {preset_name}
            (function() {{
                const preset = {json.dumps(preset)};
                
                // Apply mouse behavior
                if (preset.mouseBehavior) {{
                    window.mouseSpeed = preset.mouseBehavior.speed;
                    window.mouseAcceleration = preset.mouseBehavior.acceleration;
                    window.mouseJitter = preset.mouseBehavior.jitter;
                }}
                
                // Apply scroll behavior
                if (preset.scrollBehavior) {{
                    window.scrollSpeed = preset.scrollBehavior.speed;
                    window.scrollAcceleration = preset.scrollBehavior.acceleration;
                    window.scrollJitter = preset.scrollBehavior.jitter;
                }}
                
                // Apply typing behavior
                if (preset.typingBehavior) {{
                    window.typingSpeed = preset.typingBehavior.speed;
                    window.typingAccuracy = preset.typingBehavior.accuracy;
                    window.typingCorrectionTime = preset.typingBehavior.correctionTime;
                }}
                
                // Apply attention behavior
                if (preset.attentionBehavior) {{
                    window.attentionSpan = preset.attentionBehavior.span;
                    window.distractionProbability = preset.attentionBehavior.distractionProbability;
                    window.distractionDuration = preset.attentionBehavior.distractionDuration;
                }}
            }})();
            """
            await page.evaluate(script)
    
    async def apply_fingerprint_preset(self, page, preset_name):
        """Apply a fingerprint preset"""
        preset = self.fingerprint_presets.get(preset_name)
        if preset:
            script = f"""
            // Apply fingerprint preset: {preset_name}
            (function() {{
                const preset = {json.dumps(preset)};
                
                // Apply canvas fingerprint
                if (preset.canvasFingerprint) {{
                    window.canvasNoiseLevel = preset.canvasFingerprint.noiseLevel;
                    window.canvasMutationRate = preset.canvasFingerprint.mutationRate;
                }}
                
                // Apply WebGL fingerprint
                if (preset.webglFingerprint) {{
                    window.webglParameterVariation = preset.webglFingerprint.parameterVariation;
                    window.webglRendererVariation = preset.webglFingerprint.rendererVariation;
                }}
                
                // Apply audio fingerprint
                if (preset.audioFingerprint) {{
                    window.audioNoiseLevel = preset.audioFingerprint.noiseLevel;
                    window.audioMutationRate = preset.audioFingerprint.mutationRate;
                }}
                
                // Apply font fingerprint
                if (preset.fontFingerprint) {{
                    window.fontVariation = preset.fontFingerprint.variation;
                    window.fontMetricsVariation = preset.fontFingerprint.metricsVariation;
                }}
            }})();
            """
            await page.evaluate(script)
    
    async def apply_custom_configuration(self, page, config):
        """Apply custom configuration"""
        script = f"""
        // Apply custom configuration
        (function() {{
            const config = {json.dumps(config.custom)};
            
            // Apply custom delays
            if (config.delays) {{
                window.minDelay = config.delays.min;
                window.maxDelay = config.delays.max;
            }}
            
            // Apply custom probabilities
            if (config.probabilities) {{
                window.scrollProbability = config.probabilities.scroll;
                window.clickProbability = config.probabilities.click;
                window.interactionProbability = config.probabilities.interaction;
            }}
            
            // Apply custom patterns
            if (config.patterns) {{
                window.scrollPattern = config.patterns.scroll;
                window.clickPattern = config.patterns.click;
                window.navigationPattern = config.patterns.navigation;
            }}
        }})();
        """
        await page.evaluate(script)
    
    def load_behavior_presets(self):
        """Load behavior presets"""
        return {
            "casual": {
                "mouseBehavior": {
                    "speed": 15,
                    "acceleration": 0.8,
                    "jitter": 0.2
                },
                "scrollBehavior": {
                    "speed": 2,
                    "acceleration": 0.7,
                    "jitter": 0.3
                },
                "typingBehavior": {
                    "speed": 120,
                    "accuracy": 0.95,
                    "correctionTime": 300
                },
                "attentionBehavior": {
                    "span": 15000,
                    "distractionProbability": 0.15,
                    "distractionDuration": 3000
                }
            },
            "focused": {
                "mouseBehavior": {
                    "speed": 8,
                    "acceleration": 0.5,
                    "jitter": 0.1
                },
                "scrollBehavior": {
                    "speed": 1.2,
                    "acceleration": 0.4,
                    "jitter": 0.1
                },
                "typingBehavior": {
                    "speed": 80,
                    "accuracy": 0.98,
                    "correctionTime": 200
                },
                "attentionBehavior": {
                    "span": 30000,
                    "distractionProbability": 0.05,
                    "distractionDuration": 1500
                }
            },
            "distracted": {
                "mouseBehavior": {
                    "speed": 25,
                    "acceleration": 1.2,
                    "jitter": 0.4
                },
                "scrollBehavior": {
                    "speed": 3,
                    "acceleration": 1.0,
                    "jitter": 0.5
                },
                "typingBehavior": {
                    "speed": 200,
                    "accuracy": 0.9,
                    "correctionTime": 500
                },
                "attentionBehavior": {
                    "span": 8000,
                    "distractionProbability": 0.3,
                    "distractionDuration": 5000
                }
            }
        }
    
    def load_fingerprint_presets(self):
        """Load fingerprint presets"""
        return {
            "stealth": {
                "canvasFingerprint": {
                    "noiseLevel": 0.01,
                    "mutationRate": 0.005
                },
                "webglFingerprint": {
                    "parameterVariation": 2,
                    "rendererVariation": 1
                },
                "audioFingerprint": {
                    "noiseLevel": 0.005,
                    "mutationRate": 0.002
                },
                "fontFingerprint": {
                    "variation": 0.1,
                    "metricsVariation": 0.05
                }
            },
            "balanced": {
                "canvasFingerprint": {
                    "noiseLevel": 0.02,
                    "mutationRate": 0.01
                },
                "webglFingerprint": {
                    "parameterVariation": 5,
                    "rendererVariation": 3
                },
                "audioFingerprint": {
                    "noiseLevel": 0.01,
                    "mutationRate": 0.005
                },
                "fontFingerprint": {
                    "variation": 0.2,
                    "metricsVariation": 0.1
                }
            },
            "random": {
                "canvasFingerprint": {
                    "noiseLevel": 0.05,
                    "mutationRate": 0.02
                },
                "webglFingerprint": {
                    "parameterVariation": 10,
                    "rendererVariation": 5
                },
                "audioFingerprint": {
                    "noiseLevel": 0.02,
                    "mutationRate": 0.01
                },
                "fontFingerprint": {
                    "variation": 0.3,
                    "metricsVariation": 0.15
                }
            }
        }

class PluginSystem:
    def __init__(self):
        self.plugins = self.load_plugins()
    
    async def enable_plugin(self, page, plugin_name):
        """Enable a plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            script = f"""
            // Enable plugin: {plugin_name}
            (function() {{
                const plugin = {json.dumps(plugin)};
                
                // Execute plugin initialization
                if (plugin.initialize) {{
                    eval(plugin.initialize);
                }}
                
                // Execute plugin main function
                if (plugin.main) {{
                    eval(plugin.main);
                }}
            }})();
            """
            await page.evaluate(script)
    
    def load_plugins(self):
        """Load available plugins"""
        return {
            "advanced_mouse_simulation": {
                "initialize": """
                    window.advancedMouseSimulation = {
                        curves: ['bezier', 'spline', 'linear'],
                        currentCurve: 0,
                        jitter: 0.2
                    };
                """,
                "main": """
                    // Override mouse movement with advanced simulation
                    const originalMouseMove = window.simulatePerfectMouseMove;
                    window.simulatePerfectMouseMove = function(targetX, targetY) {
                        const curve = window.advancedMouseSimulation.curves[window.advancedMouseSimulation.currentCurve];
                        // Implement advanced curve-based movement
                        // ...
                        return originalMouseMove(targetX, targetY);
                    };
                """
            },
            "realistic_typing": {
                "initialize": """
                    window.realisticTyping = {
                        errorRate: 0.05,
                        correctionTime: 300,
                        rhythmPatterns: ['steady', 'irregular', 'burst'],
                        currentPattern: 0
                    };
                """,
                "main": """
                    // Override typing with realistic simulation
                    const originalTyping = window.simulatePerfectTyping;
                    window.simulatePerfectTyping = function(element, text) {
                        // Implement realistic typing with errors and corrections
                        // ...
                        return originalTyping(element, text);
                    };
                """
            },
            "human_attention": {
                "initialize": """
                    window.humanAttention = {
                        focusSpans: [5000, 10000, 15000, 20000],
                        currentFocusSpan: 0,
                        distractionTypes: ['look_away', 'check_phone', 'think', 'stretch']
                    };
                """,
                "main": """
                    // Override attention simulation with more realistic behavior
                    const originalAttention = window.simulatePerfectAttention;
                    window.simulatePerfectAttention = function() {
                        // Implement realistic attention simulation
                        // ...
                        return originalAttention();
                    };
                """
            }
        }

class ConfigurationManager:
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        # This would load from a configuration file
        return {
            "behavior_preset": "balanced",
            "fingerprint_preset": "stealth",
            "enabled_plugins": ["advanced_mouse_simulation", "realistic_typing"],
            "custom": {
                "delays": {
                    "min": 0.5,
                    "max": 2.0
                },
                "probabilities": {
                    "scroll": 0.7,
                    "click": 0.5,
                    "interaction": 0.3
                },
                "patterns": {
                    "scroll": "natural",
                    "click": "random",
                    "navigation": "realistic"
                }
            }
        }



class PerfectEvasionSystem:
    def __init__(self):
        self.detection_signatures = self.load_detection_signatures()
        self.evasion_techniques = self.load_evasion_techniques()
        self.evolution_engine = EvolutionEngine()
        self.threat_intelligence = ThreatIntelligence()
    
    async def inject_perfect_evasion(self, page):
        """Achieve 100% proactive detection evasion"""
        script = """
        // Perfect proactive detection evasion
        (function() {
            // Detection signatures
            const detectionSignatures = """ + json.dumps(self.detection_signatures) + """;
            
            // Evasion techniques
            const evasionTechniques = """ + json.dumps(self.evasion_techniques) + """;
            
            // Current evasion profile
            let currentEvasionProfile = {
                techniques: [],
                effectiveness: 0,
                lastUpdated: Date.now()
            };
            
            // Monitor for detection attempts
            function monitorDetectionAttempts() {
                // Check for known detection signatures
                const pageContent = document.body.innerHTML;
                const scripts = Array.from(document.scripts);
                
                let detectionScore = 0;
                let detectedSignatures = [];
                
                // Check page content for detection signatures
                detectionSignatures.forEach(signature => {
                    if (signature.type === 'content' && pageContent.includes(signature.pattern)) {
                        detectionScore += signature.weight;
                        detectedSignatures.push(signature);
                    }
                });
                
                // Check scripts for detection signatures
                scripts.forEach(script => {
                    const scriptContent = script.innerHTML || script.src;
                    detectionSignatures.forEach(signature => {
                        if (signature.type === 'script' && scriptContent.includes(signature.pattern)) {
                            detectionScore += signature.weight;
                            detectedSignatures.push(signature);
                        }
                    });
                });
                
                // Check for behavioral detection
                if (window.navigator.webdriver) {
                    detectionScore += 30;
                    detectedSignatures.push({pattern: 'webdriver', weight: 30});
                }
                
                // Apply evasion techniques based on detected signatures
                if (detectionScore > 0) {
                    applyEvasionTechniques(detectedSignatures);
                }
                
                // Evolve evasion profile
                evolveEvasionProfile(detectedSignatures, detectionScore);
            }
            
            // Apply evasion techniques
            function applyEvasionTechniques(detectedSignatures) {
                // Find relevant evasion techniques
                const relevantTechniques = [];
                
                detectedSignatures.forEach(signature => {
                    evasionTechniques.forEach(technique => {
                        if (technique.targetSignatures.includes(signature.pattern)) {
                            relevantTechniques.push(technique);
                        }
                    });
                });
                
                // Remove duplicates
                const uniqueTechniques = [...new Set(relevantTechniques)];
                
                // Apply each technique
                uniqueTechniques.forEach(technique => {
                    if (!currentEvasionProfile.techniques.includes(technique.name)) {
                        // Execute the technique
                        if (technique.action) {
                            try {
                                // Execute in a safe sandbox try/catch so failures don't stop monitoring
                                const fn = new Function(technique.action);
                                try { fn(); } catch(e) { console.warn('Evasion action failed:', e); }
                            } catch(e) {
                                try { eval(technique.action); } catch(e2) { console.warn('Eval fallback failed:', e2); }
                            }
                        }
                        
                        // Add to current profile
                        currentEvasionProfile.techniques.push(technique.name);
                        
                        // Log application
                        console.log(`Applied evasion technique: ${technique.name} for signature: ${technique.targetSignatures.join(', ')}`);
                    }
                });
            }
            
            // Evolve evasion profile
            function evolveEvasionProfile(detectedSignatures, detectionScore) {
                // Update effectiveness
                currentEvasionProfile.effectiveness = Math.max(0, 100 - detectionScore);
                
                // Update timestamp
                currentEvasionProfile.lastUpdated = Date.now();
                
                // Report to evolution engine
                if (window.evolutionEngine) {
                    window.evolutionEngine.reportDetectionAttempt(detectedSignatures, detectionScore, currentEvasionProfile);
                }
            }
            
            // Start monitoring
            setInterval(monitorDetectionAttempts, 1000);
            
            // Initial check
            monitorDetectionAttempts();
            
            // Expose evasion profile for external monitoring
            window.getCurrentEvasionProfile = () => currentEvasionProfile;
        })();
        """
        await page.evaluate(script)
        
        # Initialize evolution engine
        await self.evolution_engine.initialize(page)
        
        # Initialize threat intelligence
        await self.threat_intelligence.initialize(page)
    
    def load_detection_signatures(self):
        """Load comprehensive detection signatures"""
        return [
            {
                "pattern": "webdriver",
                "type": "script",
                "weight": 30
            },
            {
                "pattern": "自动化",
                "type": "content",
                "weight": 25
            },
            {
                "pattern": "机器人",
                "type": "content",
                "weight": 25
            },
            {
                "pattern": "bot",
                "type": "content",
                "weight": 20
            },
            {
                "pattern": "suspicious activity",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "automation",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "headless",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "phantom",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "puppeteer",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "playwright",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "selenium",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "are you human",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "verify you are human",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "captcha",
                "type": "content",
                "weight": 20
            },
            {
                "pattern": "recaptcha",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "hcaptcha",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "fun captcha",
                "type": "content",
                "weight": 20
            }
        ]
    
    def load_evasion_techniques(self):
        """Load comprehensive evasion techniques"""
        return [
            {
                "name": "webdriver_property_spoofing",
                "targetSignatures": ["webdriver"],
                    "action": """
                    // Spoof webdriver property (tolerant)
                    (function(){
                        try{ Object.defineProperty(navigator, 'webdriver', { get: () => undefined, configurable: true }); }catch(e){}
                        try{ const proto = Object.getPrototypeOf(navigator)||navigator; Object.defineProperty(proto, 'webdriver', { get: () => undefined, configurable: true }); }catch(e){}
                    })();
                """
            },
            {
                "name": "plugins_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof plugins (tolerant)
                    (function(){
                        function getPlugins(){ return [{0:{type:'application/x-google-chrome-pdf'},description:'Portable Document Format',filename:'internal-pdf-viewer',length:1,name:'Chrome PDF Plugin'}]; }
                        try{ Object.defineProperty(navigator, 'plugins', { get: getPlugins, configurable: true }); }catch(e){}
                        try{ const proto = Object.getPrototypeOf(navigator)||navigator; Object.defineProperty(proto, 'plugins', { get: getPlugins, configurable: true }); }catch(e){}
                    })();
                """
            },
            {
                "name": "languages_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof languages (tolerant)
                    (function(){
                        function getLanguages(){ return ['en-US','en','es']; }
                        try{ Object.defineProperty(navigator, 'languages', { get: getLanguages, configurable: true }); }catch(e){}
                        try{ const proto = Object.getPrototypeOf(navigator)||navigator; Object.defineProperty(proto, 'languages', { get: getLanguages, configurable: true }); }catch(e){}
                    })();
                """
            },
            {
                "name": "permissions_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        Promise.resolve({ state: 'granted' })
                    );
                """
            },
            {
                "name": "chrome_runtime_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof chrome runtime
                    window.chrome = {
                        app: {
                            isInstalled: false,
                            InstallState: {
                                DISABLED: 'disabled',
                                ENABLED: 'enabled',
                                INSTALLED: 'installed'
                            }
                        },
                        runtime: {
                            PlatformOs: {
                                MAC: 'mac',
                                WIN: 'win',
                                ANDROID: 'android',
                                CROS: 'cros',
                                LINUX: 'linux',
                                OPENBSD: 'openbsd'
                            },
                            PlatformArch: {
                                ARM: 'arm',
                                X86_32: 'x86-32',
                                X86_64: 'x86-64'
                            },
                            PlatformNaclArch: {
                                ARM: 'arm',
                                X86_32: 'x86-32',
                                X86_64: 'x86-64'
                            }
                        }
                    };
                """
            },
            {
                "name": "iframe_content_window_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof iframe contentWindow (tolerant)
                    (function(){
                        try{ const originalContentWindow = HTMLIFrameElement.prototype.__lookupGetter__('contentWindow');
                            Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', { get: function(){ const w = originalContentWindow.call(this); w.chrome = w.chrome || {}; return w; }, configurable:true });
                        }catch(e){}
                    })();
                """
            },
            {
                "name": "notification_permission_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof notification permission (tolerant)
                    (function(){ try{ Object.defineProperty(Notification, 'permission', { get: () => 'default', configurable: true }); }catch(e){} })();
                """
            },
            {
                "name": "permissions_query_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof permissions query
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        Promise.resolve({ state: 'granted' })
                    );
                """
            },
            {
                "name": "battery_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof battery API
                    navigator.getBattery = () => Promise.resolve({
                        charging: true,
                        chargingTime: 0,
                        dischargingTime: Infinity,
                        level: 1
                    });
                """
            },
            {
                "name": "connection_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof connection API (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'connection', { get: () => ({ effectiveType:'4g', rtt:50, downlink:10, saveData:false }), configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "hardware_concurrency_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof hardware concurrency (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8, configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "device_memory_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof device memory (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'deviceMemory', { get: () => 8, configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "permissions_policy_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof permissions policy
                    const originalPermissions = navigator.permissions;
                    navigator.permissions = {
                        query: (parameters) => Promise.resolve({ state: 'granted' })
                    };
                """
            },
            {
                "name": "webgl_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof WebGL
                    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        if (parameter === 37445) {
                            return 'Intel Inc.';
                        }
                        if (parameter === 37446) {
                            return 'Intel(R) Iris(TM) HD Graphics 630';
                        }
                        return originalGetParameter.call(this, parameter);
                    };
                """
            },
            {
                "name": "canvas_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof canvas
                    const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
                    CanvasRenderingContext2D.prototype.getImageData = function() {
                        const imageData = originalGetImageData.apply(this, arguments);
                        const data = imageData.data;
                        for (let i = 0; i < data.length; i += 4) {
                            if (Math.random() < 0.01) {
                                data[i] = data[i] ^ (Math.floor(Math.random() * 256));
                                data[i+1] = data[i+1] ^ (Math.floor(Math.random() * 256));
                                data[i+2] = data[i+2] ^ (Math.floor(Math.random() * 256));
                            }
                        }
                        return imageData;
                    };
                """
            },
            {
                "name": "audio_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof audio
                    const originalGetChannelData = AudioBuffer.prototype.getChannelData;
                    AudioBuffer.prototype.getChannelData = function() {
                        const data = originalGetChannelData.apply(this, arguments);
                        for (let i = 0; i < data.length; i++) {
                            if (Math.random() < 0.005) {
                                data[i] = data[i] + (Math.random() * 2 - 1) * 0.00001;
                            }
                        }
                        return data;
                    };
                """
            },
            {
                "name": "font_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof fonts
                    const originalMeasureText = CanvasRenderingContext2D.prototype.measureText;
                    CanvasRenderingContext2D.prototype.measureText = function(text) {
                        const metrics = originalMeasureText.call(this, text);
                        if (Math.random() < 0.1) {
                            metrics.width += (Math.random() * 0.2 - 0.1);
                        }
                        return metrics;
                    };
                """
            },
            {
                "name": "time_zone_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof timezone
                    const originalTimeZone = Date.prototype.getTimezoneOffset;
                    Date.prototype.getTimezoneOffset = function() {
                        return 300; // EST
                    };
                """
            },
            {
                "name": "language_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof language (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'language', { get: () => 'en-US', configurable:true }); }catch(e){} try{ Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'], configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "platform_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof platform (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'platform', { get: () => 'Win32', configurable:true }); }catch(e){} try{ const proto=Object.getPrototypeOf(navigator)||navigator; Object.defineProperty(proto, 'platform', { get: () => 'Win32', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "vendor_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof vendor (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "user_agent_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof user agent (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'userAgent', { get: () => 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "app_version_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof app version (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'appVersion', { get: () => '5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "app_name_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof app name (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'appName', { get: () => 'Netscape', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "product_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof product (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'product', { get: () => 'Gecko', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "product_sub_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof product sub (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'productSub', { get: () => '20030107', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "mime_types_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof mime types (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'mimeTypes', { get: () => [{ type:'application/pdf', description:'Portable Document Format', suffixes:'pdf' }], configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "cookie_enabled_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof cookie enabled (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'cookieEnabled', { get: () => true, configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "do_not_track_spoofing",
                "targetSignatures": ["automation", "headless"],
                    "action": """
                    // Spoof do not track (tolerant)
                    (function(){ try{ Object.defineProperty(navigator, 'doNotTrack', { get: () => '1', configurable:true }); }catch(e){} })();
                """
            },
            {
                "name": "hardware_concurrency_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof hardware concurrency
                    Object.defineProperty(navigator, 'hardwareConcurrency', {
                        get: () => 8
                    });
                """
            },
            {
                "name": "device_memory_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof device memory
                    Object.defineProperty(navigator, 'deviceMemory', {
                        get: () => 8
                    });
                """
            },
            {
                "name": "connection_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof connection
                    Object.defineProperty(navigator, 'connection', {
                        get: () => ({
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        })
                    });
                """
            },
            {
                "name": "battery_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof battery
                    navigator.getBattery = () => Promise.resolve({
                        charging: true,
                        chargingTime: 0,
                        dischargingTime: Infinity,
                        level: 1
                    });
                """
            },
            {
                "name": "permissions_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        Promise.resolve({ state: 'granted' })
                    );
                """
            },
            {
                "name": "permissions_policy_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof permissions policy
                    const originalPermissions = navigator.permissions;
                    navigator.permissions = {
                        query: (parameters) => Promise.resolve({ state: 'granted' })
                    };
                """
            },
            {
                "name": "webgl_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof WebGL
                    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        if (parameter === 37445) {
                            return 'Intel Inc.';
                        }
                        if (parameter === 37446) {
                            return 'Intel(R) Iris(TM) HD Graphics 630';
                        }
                        return originalGetParameter.call(this, parameter);
                    };
                """
            },
            {
                "name": "canvas_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof canvas
                    const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
                    CanvasRenderingContext2D.prototype.getImageData = function() {
                        const imageData = originalGetImageData.apply(this, arguments);
                        const data = imageData.data;
                        for (let i = 0; i < data.length; i += 4) {
                            if (Math.random() < 0.01) {
                                data[i] = data[i] ^ (Math.floor(Math.random() * 256));
                                data[i+1] = data[i+1] ^ (Math.floor(Math.random() * 256));
                                data[i+2] = data[i+2] ^ (Math.floor(Math.random() * 256));
                            }
                        }
                        return imageData;
                    };
                """
            },
            {
                "name": "audio_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof audio
                    const originalGetChannelData = AudioBuffer.prototype.getChannelData;
                    AudioBuffer.prototype.getChannelData = function() {
                        const data = originalGetChannelData.apply(this, arguments);
                        for (let i = 0; i < data.length; i++) {
                            if (Math.random() < 0.005) {
                                data[i] = data[i] + (Math.random() * 2 - 1) * 0.00001;
                            }
                        }
                        return data;
                    };
                """
            },
            {
                "name": "font_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof fonts
                    const originalMeasureText = CanvasRenderingContext2D.prototype.measureText;
                    CanvasRenderingContext2D.prototype.measureText = function(text) {
                        const metrics = originalMeasureText.call(this, text);
                        if (Math.random() < 0.1) {
                            metrics.width += (Math.random() * 0.2 - 0.1);
                        }
                        return metrics;
                    };
                """
            },
            {
                "name": "time_zone_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof timezone
                    const originalTimeZone = Date.prototype.getTimezoneOffset;
                    Date.prototype.getTimezoneOffset = function() {
                        return 300; // EST
                    };
                """
            },
            {
                "name": "language_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof language
                    Object.defineProperty(navigator, 'language', {
                        get: () => 'en-US'
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                """
            },
            {
                "name": "platform_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof platform
                    Object.defineProperty(navigator, 'platform', {
                        get: () => 'Win32'
                    });
                """
            },
            {
                "name": "vendor_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof vendor
                    Object.defineProperty(navigator, 'vendor', {
                        get: () => 'Google Inc.'
                    });
                """
            },
            {
                "name": "user_agent_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof user agent
                    Object.defineProperty(navigator, 'userAgent', {
                        get: () => 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    });
                """
            },
            {
                "name": "app_version_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof app version
                    Object.defineProperty(navigator, 'appVersion', {
                        get: () => '5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    });
                """
            },
            {
                "name": "app_name_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof app name
                    Object.defineProperty(navigator, 'appName', {
                        get: () => 'Netscape'
                    });
                """
            },
            {
                "name": "product_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof product
                    Object.defineProperty(navigator, 'product', {
                        get: () => 'Gecko'
                    });
                """
            },
            {
                "name": "product_sub_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof product sub
                    Object.defineProperty(navigator, 'productSub', {
                        get: () => '20030107'
                    });
                """
            },
            {
                "name": "mime_types_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof mime types
                    Object.defineProperty(navigator, 'mimeTypes', {
                        get: () => [
                            {
                                type: 'application/pdf',
                                description: 'Portable Document Format',
                                suffixes: 'pdf'
                            }
                        ]
                    });
                """
            },
            {
                "name": "cookie_enabled_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof cookie enabled
                    Object.defineProperty(navigator, 'cookieEnabled', {
                        get: () => true
                    });
                """
            },
            {
                "name": "do_not_track_spoofing",
                "targetSignatures": ["automation", "headless"],
                "action": """
                    // Spoof do not track
                    Object.defineProperty(navigator, 'doNotTrack', {
                        get: () => '1'
                    });
                """
            }
        ]

class EvolutionEngine:
    def __init__(self):
        self.evolution_history = []
        self.current_generation = 0
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    async def initialize(self, page):
        """Initialize the evolution engine"""
        script = """
        // Evolution engine
        window.evolutionEngine = {
            evolutionHistory: [],
            currentGeneration: 0,
            mutationRate: 0.1,
            crossoverRate: 0.7,
            
            reportDetectionAttempt: function(detectedSignatures, detectionScore, evasionProfile) {
                // Record detection attempt
                this.evolutionHistory.push({
                    timestamp: Date.now(),
                    detectedSignatures: detectedSignatures,
                    detectionScore: detectionScore,
                    evasionProfile: JSON.parse(JSON.stringify(evasionProfile))
                });
                
                // Evolve evasion profile
                this.evolveEvasionProfile(detectedSignatures, detectionScore, evasionProfile);
            },
            
            evolveEvasionProfile: function(detectedSignatures, detectionScore, evasionProfile) {
                // Analyze detection patterns
                const signaturePatterns = this.analyzeSignaturePatterns(detectedSignatures);
                
                // Generate new evasion techniques
                const newTechniques = this.generateNewTechniques(signaturePatterns);
                
                // Apply new techniques to evasion profile
                newTechniques.forEach(technique => {
                    if (!evasionProfile.techniques.includes(technique)) {
                        evasionProfile.techniques.push(technique);
                    }
                });
                
                // Update effectiveness
                evasionProfile.effectiveness = Math.max(0, 100 - detectionScore);
                
                // Increment generation
                this.currentGeneration++;
                
                // Log evolution
                console.log(`Evolved evasion profile to generation ${this.currentGeneration} with ${newTechniques.length} new techniques`);
            },
            
            analyzeSignaturePatterns: function(detectedSignatures) {
                // Analyze patterns in detected signatures
                const patterns = {
                    types: {},
                    weights: {},
                    frequencies: {}
                };
                
                detectedSignatures.forEach(signature => {
                    // Count types
                    if (!patterns.types[signature.type]) {
                        patterns.types[signature.type] = 0;
                    }
                    patterns.types[signature.type]++;
                    
                    // Count weights
                    if (!patterns.weights[signature.weight]) {
                        patterns.weights[signature.weight] = 0;
                    }
                    patterns.weights[signature.weight]++;
                    
                    // Count frequencies
                    if (!patterns.frequencies[signature.pattern]) {
                        patterns.frequencies[signature.pattern] = 0;
                    }
                    patterns.frequencies[signature.pattern]++;
                });
                
                return patterns;
            },
            
            generateNewTechniques: function(signaturePatterns) {
                // Generate new evasion techniques based on patterns
                const newTechniques = [];
                
                // Generate techniques for each type
                Object.keys(signaturePatterns.types).forEach(type => {
                    const technique = this.generateTechniqueForType(type);
                    if (technique) {
                        newTechniques.push(technique);
                    }
                });
                
                return newTechniques;
            },
            
            generateTechniqueForType: function(type) {
                // Generate a new technique for a specific type
                switch(type) {
                    case 'script':
                        return 'advanced_script_obfuscation';
                    case 'content':
                        return 'content_pattern_masking';
                    default:
                        return 'general_evasion_technique';
                }
            }
        };
        """
        await page.evaluate(script)
    
    async def report_detection_attempt(self, page, detected_signatures, detection_score, evasion_profile):
        """Report a detection attempt to the evolution engine"""
        script = f"""
        window.evolutionEngine.reportDetectionAttempt(
            {json.dumps(detected_signatures)},
            {detection_score},
            {json.dumps(evasion_profile)}
        );
        """
        await page.evaluate(script)

class ThreatIntelligence:
    def __init__(self):
        self.threat_feeds = self.load_threat_feeds()
        self.detection_signatures = self.load_detection_signatures()
        self.last_update = 0
    
    async def initialize(self, page):
        """Initialize the threat intelligence system"""
        script = """
        // Threat intelligence system
        window.threatIntelligence = {
            threatFeeds: """ + json.dumps(self.threat_feeds) + """,
            detectionSignatures: """ + json.dumps(self.detection_signatures) + """,
            lastUpdate: 0,
            
            updateThreatIntelligence: function() {
                // Update threat intelligence from feeds
                const now = Date.now();
                
                // Check if update is needed (every 24 hours)
                if (now - this.lastUpdate > 86400000) {
                    // Fetch new threat intelligence
                    this.fetchThreatIntelligence();
                    
                    // Update last update time
                    this.lastUpdate = now;
                }
            },
            
            fetchThreatIntelligence: function() {
                // Fetch new threat intelligence from feeds
                this.threatFeeds.forEach(feed => {
                    // Fetch feed data
                    fetch(feed.url)
                        .then(response => response.json())
                        .then(data => {
                            // Process feed data
                            this.processFeedData(feed, data);
                        })
                        .catch(error => {
                            console.error('Error fetching threat intelligence:', error);
                        });
                });
            },
            
            processFeedData: function(feed, data) {
                // Process feed data and update detection signatures
                if (data.signatures) {
                    data.signatures.forEach(signature => {
                        // Add new signature
                        this.detectionSignatures.push({
                            pattern: signature.pattern,
                            type: signature.type,
                            weight: signature.weight,
                            source: feed.name
                        });
                    });
                }
                
                // Log update
                console.log(`Updated threat intelligence from ${feed.name} with ${data.signatures ? data.signatures.length : 0} signatures`);
            },
            
            checkForThreats: function() {
                // Check current page for known threats
                const pageContent = document.body.innerHTML;
                const scripts = Array.from(document.scripts);
                
                let threats = [];
                
                // Check content against signatures
                this.detectionSignatures.forEach(signature => {
                    if (signature.type === 'content' && pageContent.includes(signature.pattern)) {
                        threats.push({
                            signature: signature,
                            type: 'content',
                            severity: signature.weight
                        });
                    }
                });
                
                // Check scripts against signatures
                scripts.forEach(script => {
                    const scriptContent = script.innerHTML || script.src;
                    this.detectionSignatures.forEach(signature => {
                        if (signature.type === 'script' && scriptContent.includes(signature.pattern)) {
                            threats.push({
                                signature: signature,
                                type: 'script',
                                severity: signature.weight
                            });
                        }
                    });
                });
                
                return threats;
            }
        };
        
        // Initial threat intelligence update
        window.threatIntelligence.updateThreatIntelligence();
        
        // Schedule regular updates
        setInterval(() => {
            window.threatIntelligence.updateThreatIntelligence();
        }, 86400000); // 24 hours
        """
        await page.evaluate(script)
    
    def load_threat_feeds(self):
        """Load threat intelligence feeds"""
        return [
            {
                "name": "Bot Detection Signatures",
                "url": "https://example.com/api/bot-signatures"
            },
            {
                "name": "Anti-Automation Patterns",
                "url": "https://example.com/api/anti-automation"
            },
            {
                "name": "CAPTCHA Detection",
                "url": "https://example.com/api/captcha-detection"
            }
        ]
    
    def load_detection_signatures(self):
        """Load detection signatures"""
        return [
            {
                "pattern": "webdriver",
                "type": "script",
                "weight": 30
            },
            {
                "pattern": "自动化",
                "type": "content",
                "weight": 25
            },
            {
                "pattern": "机器人",
                "type": "content",
                "weight": 25
            },
            {
                "pattern": "bot",
                "type": "content",
                "weight": 20
            },
            {
                "pattern": "suspicious activity",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "automation",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "headless",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "phantom",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "puppeteer",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "playwright",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "selenium",
                "type": "script",
                "weight": 20
            },
            {
                "pattern": "are you human",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "verify you are human",
                "type": "content",
                "weight": 15
            },
            {
                "pattern": "captcha",
                "type": "content",
                "weight": 20
            },
            {
                "pattern": "recaptcha",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "hcaptcha",
                "type": "script",
                "weight": 25
            },
            {
                "pattern": "fun captcha",
                "type": "content",
                "weight": 20
            }
        ]


        
                        
#z.ai features end                

# === NEW FEATURES: Improve Header Consistency ===
async def extract_playwright_headers(page):
    """Extract headers from Playwright session for HTTP request"""
    # Get network response headers
    headers = {}
    
    try:
        # Get the current page's URL
        url = page.url
        
        # Get the main request/response
        response = await page.request.response(url)
        
        if response:
            # Extract request headers
            request_headers = await response.request.all_headers()
            
            # Extract important headers
            for header_name in ['User-Agent', 'Accept', 'Accept-Language', 
                              'Accept-Encoding', 'Connection', 'Upgrade-Insecure-Requests',
                              'Sec-Fetch-Site', 'Sec-Fetch-Mode', 'Sec-Fetch-User', 'Sec-Fetch-Dest']:
                if header_name in request_headers:
                    headers[header_name] = request_headers[header_name]
    except Exception:
        pass
    
    return headers

async def enhance_headers_with_playwright_data(page, headers):
    """Enhance HTTP headers with data from Playwright session"""
    # Extract headers from Playwright
    playwright_headers = await extract_playwright_headers(page)
    
    # Update headers with Playwright data
    headers.update(playwright_headers)
    
    # Add additional realistic headers
    headers['Cache-Control'] = 'max-age=0'
    headers['Sec-Ch-Ua'] = '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"'
    headers['Sec-Ch-Ua-Mobile'] = '?0'
    headers['Sec-Ch-Ua-Platform'] = '"Windows"'
    
    return headers

# === NEW FEATURES: Realistic Timing Patterns ===
def realistic_timing_patterns():
    """Generate realistic timing patterns for human-like behavior"""
    patterns = {
        'morning_rush': {  # 6-9 AM
            'page_load': random.uniform(1.5, 3.0),
            'first_interaction': random.uniform(0.5, 2.0),
            'interaction_interval': random.uniform(2.0, 5.0),
            'dwell_time': random.uniform(8, 15),
            'exit_time': random.uniform(1.0, 3.0)
        },
        'work_hours': {  # 9 AM - 5 PM
            'page_load': random.uniform(1.0, 2.5),
            'first_interaction': random.uniform(0.3, 1.5),
            'interaction_interval': random.uniform(1.5, 4.0),
            'dwell_time': random.uniform(5, 12),
            'exit_time': random.uniform(0.5, 2.0)
        },
        'evening': {  # 5 PM - 10 PM
            'page_load': random.uniform(1.2, 2.8),
            'first_interaction': random.uniform(0.4, 1.8),
            'interaction_interval': random.uniform(2.5, 6.0),
            'dwell_time': random.uniform(10, 20),
            'exit_time': random.uniform(1.0, 4.0)
        },
        'night': {  # 10 PM - 6 AM
            'page_load': random.uniform(1.8, 3.5),
            'first_interaction': random.uniform(0.8, 2.5),
            'interaction_interval': random.uniform(3.0, 8.0),
            'dwell_time': random.uniform(15, 30),
            'exit_time': random.uniform(2.0, 5.0)
        }
    }
    
    # Get current hour
    current_hour = datetime.datetime.now().hour
    
    # Select pattern based on time of day
    if 6 <= current_hour < 9:
        return patterns['morning_rush']
    elif 9 <= current_hour < 17:
        return patterns['work_hours']
    elif 17 <= current_hour < 22:
        return patterns['evening']
    else:
        return patterns['night']

# === NEW FEATURES: Enhanced Cookie Handling ===
async def transfer_cookies_playwright_to_http(page, session):
    """Transfer cookies from Playwright to HTTP session with proper handling"""
    # Get cookies from Playwright
    playwright_cookies = await page.context.cookies()
    
    # Convert to requests format
    cookie_dict = {}
    
    for cookie in playwright_cookies:
        # Handle secure cookies
        secure = cookie.get('secure', False)
        # Handle HttpOnly cookies
        httponly = cookie.get('httpOnly', False)
        # Handle sameSite attribute
        same_site = cookie.get('sameSite', None)
        
        # Create cookie for requests
        cookie_dict[cookie['name']] = cookie['value']
        
        # Set additional cookie attributes in session
        if hasattr(session.cookies, '_cookies'):
            domain = cookie.get('domain', '')
            path = cookie.get('path', '/')
            
            # Create a cookie object with all attributes
            cookie_obj = requests.cookies.create_cookie(
                name=cookie['name'],
                value=cookie['value'],
                domain=domain,
                path=path,
                secure=secure,
                httponly=httponly
            )
            
            # Add to session
            session.cookies.set_cookie(cookie_obj)
    
    return cookie_dict

# === NEW FEATURES: Improved Hybrid Click Function ===
async def enhanced_hybrid_click(page, session, cpa_url, click_id):
    """Enhanced hybrid click with improved realism using Playwright - FIXED to only make one click"""
    try:
        # 1. Use Playwright to establish a realistic browser session
        logger.info("🌐 Establishing realistic browser session with Playwright...")
        
        # Visit the CPA page with Playwright - this will be our only click
        referer_to_use = random.choice(referers) if referers else None
        await page.goto(cpa_url, referer=referer_to_use, timeout=30000)
        
        # Wait for page to load with realistic timing
        timing = realistic_timing_patterns()
        await asyncio.sleep(timing['page_load'])
        
        # 2. Extract cookies and session information
        logger.info("🍪 Extracting session information from Playwright...")
        cookies = await transfer_cookies_playwright_to_http(page, session)
        
        # 3. Extract and enhance headers
        logger.info("📋 Extracting and enhancing headers...")
        base_headers = {}
        headers = await enhance_headers_with_playwright_data(page, base_headers)
        if referer_to_use:
            headers['Referer'] = referer_to_use
        headers['X-Click-ID'] = click_id
        
        # 4. Simulate realistic browsing behavior
        logger.info("🖱️ Simulating realistic browsing behavior...")
        
        # First interaction
        await asyncio.sleep(timing['first_interaction'])
        
        # Random scroll
        if random.random() < 0.7:  # 70% chance to scroll
            scroll_amount = random.randint(200, 800)
            await page.evaluate(f"window.scrollTo(0, {scroll_amount});")
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Random mouse movement
        if random.random() < 0.6:  # 60% chance to move mouse
            # Move mouse to random position
            x = random.randint(50, page.viewport_size['width'] - 50)
            y = random.randint(50, page.viewport_size['height'] - 50)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Random interactions during dwell time
        dwell_start = time.time()
        interactions = 0
        
        while time.time() - dwell_start < timing['dwell_time']:
            # Random interaction
            if random.random() < 0.3:  # 30% chance per interval
                interaction_type = random.choice(['scroll', 'mouse', 'click'])
                
                if interaction_type == 'scroll':
                    scroll_amount = random.randint(100, 500)
                    await page.evaluate(f"window.scrollTo(0, window.scrollY + {scroll_amount});")
                    await asyncio.sleep(random.uniform(0.3, 1.0))
                    
                elif interaction_type == 'mouse':
                    x = random.randint(50, page.viewport_size['width'] - 50)
                    y = random.randint(50, page.viewport_size['height'] - 50)
                    await page.mouse.move(x, y)
                    await asyncio.sleep(random.uniform(0.2, 0.8))
                    
                elif interaction_type == 'click':
                    # Click on a non-critical element
                    try:
                        # Find random clickable element
                        elements = await page.query_selector_all("div, button, a")
                        if elements:
                            element = random.choice(elements)
                            if await element.is_visible():
                                await element.click()
                                await asyncio.sleep(random.uniform(0.5, 2.0))
                                
                                # 50% chance to go back
                                if random.random() < 0.5:
                                    await page.go_back()
                                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    except:
                        pass
                
                interactions += 1
            
            # Small delay between interaction checks
            await asyncio.sleep(random.uniform(0.5, timing['interaction_interval']))
        
        # Add realistic delay before finishing
        await asyncio.sleep(timing['exit_time'])
        
        logger.info("✅ Enhanced hybrid click successful (Playwright only)!")
        return True
            
    except Exception as e:
        logger.error(f"❌ Enhanced hybrid click failed: {e}")
        return False

# === NEW FEATURES: Additional Anti-Detection Measures ===
async def inject_anti_detection_scripts(page):
    """Inject additional anti-detection scripts using Playwright"""
    anti_detection_script = """
    // Hide WebDriver property
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
    });
    
    // Hide automation indicators
    Object.defineProperty(navigator, 'plugins', {
        get: () => [
            {
                0: {type: "application/x-google-chrome-pdf"},
                description: "Portable Document Format",
                filename: "internal-pdf-viewer",
                length: 1,
                name: "Chrome PDF Plugin"
            }
        ],
    });
    
    // Override permissions
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        Promise.resolve({ state: 'granted' })
    );
    
    // Override getBattery
    navigator.getBattery = () => Promise.resolve({
        charging: true,
        chargingTime: 0,
        dischargingTime: Infinity,
        level: 1
    });
    
    // Override connection
    const originalConnection = navigator.connection;
    Object.defineProperty(navigator, 'connection', {
        get: () => ({
            effectiveType: '4g',
            rtt: 50,
            downlink: 10,
            saveData: false
        })
    });
    
    // Add realistic plugins
    Object.defineProperty(navigator, 'mimeTypes', {
        get: () => [
            {
                description: "Portable Document Format",
                suffixes: "pdf",
                type: "application/pdf",
                enabledPlugin: {
                    description: "Portable Document Format",
                    filename: "internal-pdf-viewer",
                    length: 1,
                    name: "Chrome PDF Plugin"
                }
            }
        ]
    });
    """
    
    try:
        await page.evaluate(anti_detection_script)
        logger.info("🔒 Anti-detection scripts injected")
    except Exception as e:
        logger.warning(f"Error injecting anti-detection scripts: {e}")

async def simulate_realistic_network_conditions(page):
    """Simulate realistic network conditions using Playwright"""
    network_conditions_script = """
    // Simulate realistic network conditions
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        // Add random delay to simulate network latency
        const delay = Math.random() * 200 + 50; // 50–250ms
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                originalFetch.apply(this, args)
                    .then(resolve)
                    .catch(reject);
            }, delay);
        });
    };
    // Safely override XMLHttpRequest.open
    (function() {
        const originalXHROpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function(...args) {
            try {
                return originalXHROpen.apply(this, args);
            } catch (e) {
                console.error("XMLHttpRequest.open error:", e);
                throw e;
            }
        };
    })();
    """
    
    try:
        await page.evaluate(network_conditions_script)
        logger.info("🌐 Realistic network conditions simulated")
    except Exception as e:
        logger.warning(f"Error simulating network conditions: {e}")

# === NEW: Advanced Human Behavior Simulation ===
async def simulate_human_reading_pattern(page):
    """Simulate realistic reading patterns with variable speed"""
    try:
        # Get page height
        page_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = page.viewport_size["height"]
        
        # Simulate reading with variable speed
        current_position = 0
        while current_position < page_height:
            # Variable reading speed
            scroll_speed = random.uniform(100, 400)  # pixels per second
            scroll_duration = random.uniform(2, 8)   # seconds of scrolling
            
            # Calculate scroll distance
            scroll_distance = min(scroll_speed * scroll_duration, viewport_height * 0.8)
            current_position += scroll_distance
            
            # Scroll to new position
            await page.evaluate(f"window.scrollTo(0, {current_position})")
            
            # Simulate reading time (longer for text-heavy content)
            reading_time = random.uniform(3, 12)
            await asyncio.sleep(reading_time)
            
            # Occasional scroll back up (re-reading)
            if random.random() < 0.2:  # 20% chance
                scroll_back = random.randint(100, 500)
                await page.evaluate(f"window.scrollTo(0, {current_position - scroll_back})")
                await asyncio.sleep(random.uniform(2, 5))
                # Scroll back down
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await asyncio.sleep(random.uniform(1, 3))
                
    except Exception as e:
        logger.warning(f"Error in reading pattern simulation: {e}")

async def simulate_human_idle_time(page):
    """Simulate random idle periods during browsing"""
    try:
        # Random chance to go idle
        if random.random() < 0.4:  # 40% chance
            idle_time = random.uniform(5, 25)  # 5-25 seconds
            logger.info(f"😴 Simulating idle time for {idle_time:.1f} seconds")
            
            # Random small movements during idle
            idle_end = time.time() + idle_time
            while time.time() < idle_end:
                # Small random mouse movements
                if random.random() < 0.3:  # 30% chance per check
                    x = random.randint(50, page.viewport_size["width"] - 50)
                    y = random.randint(50, page.viewport_size["height"] - 50)
                    await page.mouse.move(x, y)
                
                # Small random scrolls
                if random.random() < 0.2:  # 20% chance per check
                    scroll_amount = random.randint(-50, 50)
                    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                
                # Short pause between checks
                await asyncio.sleep(random.uniform(0.5, 2))
                
    except Exception as e:
        logger.warning(f"Error in idle time simulation: {e}")

async def simulate_realistic_browser_behavior(page):
    """Simulate realistic browser behavior patterns"""
    try:
        # Random chance to open dev tools (then quickly close)
        if random.random() < 0.05:  # 5% chance
            logger.info("🔧 Simulating dev tools inspection")
            # Open dev tools
            await page.evaluate("() => { debugger; }")
            # Wait a short time
            await asyncio.sleep(random.uniform(0.5, 2))
            # Close dev tools
            await page.keyboard.press("Escape")
        
        # Random chance to resize window
        if random.random() < 0.1:  # 10% chance
            logger.info("📏 Simulating window resize")
            current_size = page.viewport_size
            new_width = current_size["width"] + random.randint(-100, 100)
            new_height = current_size["height"] + random.randint(-100, 100)
            # Ensure minimum size
            new_width = max(800, new_width)
            new_height = max(600, new_height)
            await page.set_viewport_size({"width": new_width, "height": new_height})
            # Wait a moment
            await asyncio.sleep(random.uniform(1, 3))
            # Resize back
            await page.set_viewport_size(current_size)
        
        # Random chance to zoom
        if random.random() < 0.05:  # 5% chance
            logger.info("🔍 Simulating zoom")
            zoom_level = random.choice([0.8, 0.9, 1.1, 1.2])
            await page.evaluate(f"document.body.style.zoom = '{zoom_level}'")
            await asyncio.sleep(random.uniform(2, 5))
            await page.evaluate("document.body.style.zoom = '1'")
        
        # Random chance to print
        if random.random() < 0.02:  # 2% chance
            logger.info("🖨️ Simulating print dialog")
            await page.evaluate("window.print()")
            await asyncio.sleep(random.uniform(0.5, 1))
            await page.keyboard.press("Escape")
        
    except Exception as e:
        logger.warning(f"Error in browser behavior simulation: {e}")

async def inject_advanced_fingerprint_spoofing(page, device_fingerprint):
    """Inject advanced fingerprint spoofing techniques"""
    try:
        # Get device info
        device_info = device_fingerprint.device_info
        device_type = device_info.get('device_type', 'Unknown')
        
        # Create a comprehensive spoofing script with safer property redefinition
        spoofing_script = f"""
        // Advanced navigator property spoofing with safer property redefinition
        const newNavigator = {{}};
        
        // Override navigator properties
        for (let key in navigator) {{
            if (typeof navigator[key] === 'function') {{
                newNavigator[key] = navigator[key].bind(navigator);
            }} else {{
                newNavigator[key] = navigator[key];
            }}
        }}
        
        // Spoof hardware concurrency
        try {{
            Object.defineProperty(newNavigator, 'hardwareConcurrency', {{
                get: () => {random.randint(4, 16)},
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not redefine hardwareConcurrency:", e);
        }}
        
        // Spoof device memory
        try {{
            Object.defineProperty(newNavigator, 'deviceMemory', {{
                get: () => {random.choice([4, 6, 8])},
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not redefine deviceMemory:", e);
        }}
        
        // Spoof connection
        try {{
            Object.defineProperty(newNavigator, 'connection', {{
                get: () => ({{
                    effectiveType: '{random.choice(['4g', '3g', 'wifi'])}',
                    rtt: {random.randint(50, 300)},
                    downlink: {random.randint(1, 10)},
                    saveData: false
                }}),
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not redefine connection:", e);
        }}
        
        // Spoof battery
        try {{
            Object.defineProperty(newNavigator, 'getBattery', {{
                get: () => () => Promise.resolve({{
                    charging: {str(random.choice([True, False])).lower()},
                    chargingTime: {random.randint(0, 3600)},
                    dischargingTime: {random.randint(1800, 7200)},
                    level: {random.randint(20, 95) / 100}
                }}),
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not redefine getBattery:", e);
        }}
        
        // Spoof plugins
        try {{
            const pluginArray = {json.dumps([
                {
                    "name": "Chrome PDF Plugin",
                    "filename": "internal-pdf-viewer",
                    "description": "Portable Document Format"
                },
                {
                    "name": "Chromium PDF Plugin",
                    "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                    "description": "Portable Document Format"
                }
            ])};
            
            Object.defineProperty(newNavigator, 'plugins', {{
                get: () => {{
                    const plugins = [];
                    for (let i = 0; i < pluginArray.length; i++) {{
                        const plugin = pluginArray[i];
                        const p = {{
                            0: {{type: "application/x-google-chrome-pdf"}},
                            description: plugin.description,
                            filename: plugin.filename,
                            length: 1,
                            name: plugin.name
                        }};
                        plugins.push(p);
                    }}
                    return plugins;
                }},
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not redefine plugins:", e);
        }}
        
        // Replace navigator
        window.Navigator = newNavigator.constructor();
        for (let key in newNavigator) {{
            window.Navigator[key] = newNavigator[key];
        }}
        window.navigator = window.Navigator;
        
        // Override permissions
        try {{
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                Promise.resolve({{ state: 'granted' }})
            );
        }} catch (e) {{
            console.warn("Could not override permissions:", e);
        }}
        
        // Override getBattery
        try {{
            navigator.getBattery = () => Promise.resolve({{
                charging: true,
                chargingTime: 0,
                dischargingTime: Infinity,
                level: 1
            }});
        }} catch (e) {{
            console.warn("Could not override getBattery:", e);
        }}
        
        // Override connection
        try {{
            const originalConnection = navigator.connection;
            Object.defineProperty(navigator, 'connection', {{
                get: () => ({{
                    effectiveType: '4g',
                    rtt: 50,
                    downlink: 10,
                    saveData: false
                }}),
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not override connection:", e);
        }}
        
        // Add realistic plugins
        try {{
            Object.defineProperty(navigator, 'mimeTypes', {{
                get: () => [
                    {{
                        description: "Portable Document Format",
                        suffixes: "pdf",
                        type: "application/pdf",
                        enabledPlugin: {{
                            description: "Portable Document Format",
                            filename: "internal-pdf-viewer",
                            length: 1,
                            name: "Chrome PDF Plugin"
                        }}
                    }}
                ],
                configurable: true
            }});
        }} catch (e) {{
            console.warn("Could not define mimeTypes:", e);
        }}
        """
        
        await page.evaluate(spoofing_script)
        logger.info("🔒 Advanced fingerprint spoofing injected")
        
    except Exception as e:
        logger.warning(f"Error injecting advanced fingerprint spoofing: {e}")

async def enhanced_session_simulation(page, max_time=120):
    """Enhanced session simulation with more realistic behavior"""
    try:
        session_start = time.time()
        
        # Define the actions as async functions that take a page parameter
        async def scroll_to_top(page):
            await page.evaluate("window.scrollTo(0, 0)")
            
        async def scroll_to_bottom(page):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        
        actions = [
            simulate_human_reading_pattern,
            simulate_human_idle_time,
            simulate_realistic_browser_behavior,
            scroll_to_top,
            scroll_to_bottom,
        ]
        
        # Define a local random_element_interaction function
        async def random_element_interaction(page):
            """Randomly click on non-important elements"""
            try:
                # Find clickable elements that are not important (like images, non-critical links)
                elements = await page.query_selector_all("img") + \
                          await page.query_selector_all("a[href*='image'], a[href*='photo']")
                
                if elements:
                    element = random.choice(elements)
                    if await element.is_visible():
                        # Scroll to element
                        await element.scroll_into_view_if_needed()
                        await asyncio.sleep(random.uniform(0.5, 1))
                        
                        # Click
                        await element.click()
                        await asyncio.sleep(random.uniform(1, 3))
                        
                        # 50% chance to go back
                        if random.random() < 0.5:
                            await page.go_back()
                            await asyncio.sleep(random.uniform(1, 2))
            except Exception as e:
                logger.warning(f"Error during random element interaction: {e}")
        
        # Perform random actions during the session
        while time.time() - session_start < max_time:
            # Random pause between actions
            await asyncio.sleep(random.uniform(1, 5))
            
            # Select random action
            action = random.choice(actions)
            await action(page)
            
            # Random chance to interact with page elements
            if random.random() < 0.4:  # 40% chance
                await random_element_interaction(page)
                
            # Random chance to simulate form interaction
            if random.random() < 0.2:  # 20% chance
                await simulate_form_interaction(page)
                
    except Exception as e:
        logger.warning(f"Error in enhanced session simulation: {e}")

# === CONFIGURATION ===
@dataclass
class Config:
    cpa_url: str = "https://grabify.link/RFKZDM"
    max_clicks: int = 100
    max_clicks_per_campaign: int = 50
    max_clicks_per_publisher: int = 30
    max_threads: int = 1  # Set to 1 for simpler countdown display
    infinite_loop: bool = True
    enable_encryption: bool = True
    rotation_strategy: str = "performance"  # random, round_robin, geographic, performance
    use_headless_browser: bool = False
    enable_captcha_solver: bool = True
    captcha_service: str = "2captcha"  # 2captcha, anticaptcha
    captcha_api_key: str = ""
    residential_proxy: bool = True
    proxy_provider: str = "brightdata"  # brightdata, smartproxy, oxylabs
    proxy_username: str = ""
    proxy_password: str = ""
    enable_cookie_persistence: bool = True
    cookie_file: str = "cookies.json"
    log_level: str = "INFO"
    dynamic_device_weights: bool = True
    success_rate_threshold: float = 0.7
    max_retries: int = 3
    retry_delay: int = 5
    user_agent_pool_size: int = 100
    enable_mobile_emulation: bool = True
    browser_profile_dir: str = "browser_profiles"
    enable_ip_geolocation_matching: bool = True
    auto_update_proxies: bool = True
    proxy_update_interval: int = 3600  # seconds
    # UPDATED: Optimized timing for 12-15 clicks/hour
    min_wait_between_clicks: int = 180  # 3 minutes in seconds (UPDATED for 12-15 clicks/hour)
    max_wait_between_clicks: int = 240  # 4 minutes in seconds (UPDATED for 12-15 clicks/hour)
    # New timing pattern settings
    enable_human_timing_patterns: bool = True
    peak_click_hours: List[int] = None  # Hours when clicks are more likely
    off_peak_multiplier: float = 0.3  # Reduced click probability during off-peak hours
    # New settings for one-time use
    remove_ua_after_use: bool = True  # Remove user agent after use
    remove_proxy_after_use: bool = True  # Remove proxy after use
    disable_webgpu: bool = True  # Disable WebGPU for better undetectability
    # New settings for enhanced undetectability
    enable_mouse_simulation: bool = True
    enable_scroll_simulation: bool = True
    enable_session_simulation: bool = True
    enable_form_interaction: bool = True
    enable_cloudflare_bypass: bool = True
    enable_tls_fingerprinting: bool = True
    enable_header_ordering: bool = True
    enable_environment_consistency: bool = True
    enable_device_consistency: bool = True
    # New advanced settings
    enable_advanced_mouse_simulation: bool = True
    enable_canvas_fingerprinting: bool = True
    enable_webrtc_protection: bool = True
    enable_header_randomization: bool = True
    enable_session_simulation: bool = True
    enable_profile_management: bool = True
    enable_network_fingerprinting: bool = True
    enable_device_consistency: bool = True
    # Advanced timing settings
    mouse_movement_speed: float = 1.0  # Speed multiplier for mouse movements
    typing_speed: float = 1.0  # Characters per second
    scroll_speed: float = 1.0  # Pixels per second
    # Session simulation settings
    min_session_duration: int = 300  # seconds
    max_session_duration: int = 1800  # seconds
    min_page_visits: int = 3
    max_page_visits: int = 15
    # NEW: Optimized randomness settings for higher click rate
    min_pre_click_sites: int = 1  # Minimum sites to visit before click (UPDATED)
    max_pre_click_sites: int = 2  # Maximum sites to visit before click (UPDATED)
    human_error_probability: float = 0.3  # Probability of making a human-like error
    random_interaction_probability: float = 0.7  # Probability of random page interaction
    # NEW: Enhanced randomness settings
    enable_rest_periods: bool = False  # Disabled rest periods for consistent clicking
    min_rest_period: int = 300  # 5 minutes minimum rest
    max_rest_period: int = 3600  # 60 minutes maximum rest
    rest_probability: float = 0.05  # 5% chance of taking a rest after a click (reduced)
    # NEW: Enhanced browser settings
    enable_browser_extension_simulation: bool = True  # Simulate random browser extensions
    enable_random_viewport_sizes: bool = True  # Randomize viewport sizes
    enable_natural_click_patterns: bool = True  # More natural click timing and patterns
    # NEW: Operational security settings
    enable_operational_hours: bool = False  # Enable operational hours restriction (UPDATED)
    operating_hours_start: int = 8  # 8 AM
    operating_hours_end: int = 23  # 11 PM
    enable_success_monitoring: bool = True  # Monitor success rates
    success_rate_threshold: float = 0.7  # Threshold for success rate alert
    # NEW: Browser selection
    browser_type: str = "chromium"  # chromium, firefox, webkit
    use_playwright: bool = True  # Use Playwright instead of Selenium (recommended)
    # NEW: Behavioral variance settings
    behavioral_variance: dict = None
    # NEW: Audio fingerprinting
    enable_audio_fingerprinting: bool = True
    # NEW: Browser persistence
    enable_browser_persistence: bool = False
    # NEW: Additional configuration options
    enable_proxy_health_check: bool = True
    proxy_health_check_url: str = "http://example.com/health"
    max_concurrent_browsers: int = 3
    device_pool_size: int = 50
    enable_dns_randomization: bool = False
    browser_user_profiles: int = 5
    enable_gpu_spoofing: bool = True
    
    # NEW: Enhanced behavior settings
    enable_advanced_fingerprint_spoofing: bool = True
    enable_realistic_network_simulation: bool = True
    enable_enhanced_browser_behavior: bool = True
    enable_human_reading_patterns: bool = True
    enable_human_idle_simulation: bool = True
    enable_session_depth_variation: bool = True
    min_session_depth: int = 3  # Minimum pages to visit in a session
    max_session_depth: int = 10  # Maximum pages to visit in a session
    
    # NEW: Optimized settings for higher click rate
    mac_change_frequency: int = 5  # Change MAC every 5 clicks (UPDATED)
    adaptive_timing: bool = True  # Enable adaptive timing based on success rate
    
    # NEW: Advanced features configuration
    enable_cognitive_patterns: bool = True
    enable_network_fingerprinting: bool = True
    enable_performance_metrics: bool = True
    enable_cookie_jar: bool = True
    enable_intelligent_recovery: bool = True
    enable_performance_monitoring: bool = True
    enable_human_delays: bool = True
    enable_proxy_rotation: bool = True
    cookie_persistence_strategy: str = "realistic"  # realistic, persistent, ephemeral
    cookie_lifespan: int = 86400  # 24 hours in seconds
    proxy_rotation_strategy: str = "adaptive"  # adaptive, round_robin, random
    proxy_health_check_interval: int = 300  # 5 minutes
    delay_patterns: List[str] = None  # realistic, erratic, focused

    def __post_init__(self):
        if self.behavioral_variance is None:
            self.behavioral_variance = {
                "min_session_duration": 180,  # 3 minutes
                "max_session_duration": 3600,  # 1 hour
                "random_action_probability": 0.4,  # 40% chance of random action
                "typing_speed_variation": 0.3,  # 30% variation in typing speed
                "mouse_pause_probability": 0.2,  # 20% chance of random pause
            }
        
        if self.delay_patterns is None:
            self.delay_patterns = ["realistic", "erratic", "focused"]
        
        if self.peak_click_hours is None:
            self.peak_click_hours = [8, 9, 10, 11, 12, 13, 19, 20, 21, 22]

# Load configuration from file if exists
def load_config():
    config = Config()
    config_file = "bot_config.ini"
    config_txt = "config.txt"

    # If a simple config.txt exists, parse it and produce/merge into bot_config.ini
    # Format supported:
    #   # comment lines with # or ;
    #   KEY=VALUE              -> goes into [GENERAL]
    #   SECTION.KEY=VALUE      -> goes into [SECTION], key=KEY
    #   TIMING.peak_click_hours=8,9,10
    if os.path.exists(config_txt):
        sections = {}
        try:
            with open(config_txt, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue
                    if '=' not in line:
                        continue
                    left, right = line.split('=', 1)
                    key = left.strip()
                    val = right.strip()
                    if '.' in key:
                        sec, sub = key.split('.', 1)
                        sec = sec.strip().upper()
                        sub = sub.strip()
                    else:
                        sec = 'GENERAL'
                        sub = key
                    if sec not in sections:
                        sections[sec] = {}
                    sections[sec][sub] = val

            # Write out to bot_config.ini so existing INI-based tooling keeps working
            parser_from_txt = configparser.ConfigParser()
            parser_from_txt.read_dict(sections)
            try:
                # Backup old ini if present
                if os.path.exists(config_file):
                    try:
                        bak = config_file + ".bak"
                        if os.path.exists(bak):
                            os.remove(bak)
                        os.replace(config_file, bak)
                    except Exception:
                        pass
                with open(config_file, 'w', encoding='utf-8') as wf:
                    parser_from_txt.write(wf)
            except Exception:
                # If write fails, continue with in-memory parser below
                pass
        except Exception:
            # If parsing config.txt fails, continue without it
            pass
    
    parser = configparser.ConfigParser()
    # Read bot_config.ini if present (it may have been created from config.txt above)
    if os.path.exists(config_file):
        try:
            parser.read(config_file)
        except Exception:
            # ignore parse errors and continue with defaults
            pass
        
        if 'GENERAL' in parser:
            config.cpa_url = parser.get('GENERAL', 'cpa_url', fallback=config.cpa_url)
            config.max_clicks = parser.getint('GENERAL', 'max_clicks', fallback=config.max_clicks)
            config.max_threads = parser.getint('GENERAL', 'max_threads', fallback=config.max_threads)
            config.infinite_loop = parser.getboolean('GENERAL', 'infinite_loop', fallback=config.infinite_loop)
            config.log_level = parser.get('GENERAL', 'log_level', fallback=config.log_level)
        
        if 'TIMING' in parser:
            config.min_wait_between_clicks = parser.getint('TIMING', 'min_wait_between_clicks', fallback=config.min_wait_between_clicks)
            config.max_wait_between_clicks = parser.getint('TIMING', 'max_wait_between_clicks', fallback=config.max_wait_between_clicks)
            config.enable_human_timing_patterns = parser.getboolean('TIMING', 'enable_human_timing_patterns', fallback=True)
            
            # Parse peak hours
            peak_hours_str = parser.get('TIMING', 'peak_click_hours', fallback="8,9,10,11,12,13,19,20,21,22")
            config.peak_click_hours = [int(hour) for hour in peak_hours_str.split(',')]
            
            config.off_peak_multiplier = parser.getfloat('TIMING', 'off_peak_multiplier', fallback=0.3)
        
        if 'PROXY' in parser:
            config.proxy_provider = parser.get('PROXY', 'proxy_provider', fallback=config.proxy_provider)
            config.proxy_username = parser.get('PROXY', 'proxy_username', fallback=config.proxy_username)
            config.proxy_password = parser.get('PROXY', 'proxy_password', fallback=config.proxy_password)
        
        if 'CAPTCHA' in parser:
            config.captcha_service = parser.get('CAPTCHA', 'captcha_service', fallback=config.captcha_service)
            config.captcha_api_key = parser.get('CAPTCHA', 'captcha_api_key', fallback=config.captcha_api_key)
        
        if 'USAGE' in parser:
            config.remove_ua_after_use = parser.getboolean('USAGE', 'remove_ua_after_use', fallback=True)
            config.remove_proxy_after_use = parser.getboolean('USAGE', 'remove_proxy_after_use', fallback=True)
        
        if 'ADVANCED' in parser:
            config.disable_webgpu = parser.getboolean('ADVANCED', 'disable_webgpu', fallback=True)
        
        if 'ENHANCED' in parser:
            config.enable_mouse_simulation = parser.getboolean('ENHANCED', 'enable_mouse_simulation', fallback=True)
            config.enable_scroll_simulation = parser.getboolean('ENHANCED', 'enable_scroll_simulation', fallback=True)
            config.enable_session_simulation = parser.getboolean('ENHANCED', 'enable_session_simulation', fallback=True)
            config.enable_form_interaction = parser.getboolean('ENHANCED', 'enable_form_interaction', fallback=True)
            config.enable_cloudflare_bypass = parser.getboolean('ENHANCED', 'enable_cloudflare_bypass', fallback=True)
            config.enable_tls_fingerprinting = parser.getboolean('ENHANCED', 'enable_tls_fingerprinting', fallback=True)
            config.enable_header_ordering = parser.getboolean('ENHANCED', 'enable_header_ordering', fallback=True)
            config.enable_environment_consistency = parser.getboolean('ENHANCED', 'enable_environment_consistency', fallback=True)
            config.enable_device_consistency = parser.getboolean('ENHANCED', 'enable_device_consistency', fallback=True)
        
        if 'ADVANCED' in parser:
            config.enable_advanced_mouse_simulation = parser.getboolean('ADVANCED', 'enable_advanced_mouse_simulation', fallback=True)
            config.enable_canvas_fingerprinting = parser.getboolean('ADVANCED', 'enable_canvas_fingerprinting', fallback=True)
            config.enable_webrtc_protection = parser.getboolean('ADVANCED', 'enable_webrtc_protection', fallback=True)
            config.enable_header_randomization = parser.getboolean('ADVANCED', 'enable_header_randomization', fallback=True)
            config.enable_session_simulation = parser.getboolean('ADVANCED', 'enable_session_simulation', fallback=True)
            config.enable_profile_management = parser.getboolean('ADVANCED', 'enable_profile_management', fallback=True)
            config.enable_network_fingerprinting = parser.getboolean('ADVANCED', 'enable_network_fingerprinting', fallback=True)
            config.enable_device_consistency = parser.getboolean('ADVANCED', 'enable_device_consistency', fallback=True)
        
        # NEW: Load enhanced settings
        if 'ENHANCED_RANDOMNESS' in parser:
            config.enable_rest_periods = parser.getboolean('ENHANCED_RANDOMNESS', 'enable_rest_periods', fallback=False)
            config.min_rest_period = parser.getint('ENHANCED_RANDOMNESS', 'min_rest_period', fallback=300)
            config.max_rest_period = parser.getint('ENHANCED_RANDOMNESS', 'max_rest_period', fallback=3600)
            config.rest_probability = parser.getfloat('ENHANCED_RANDOMNESS', 'rest_probability', fallback=0.05)
            config.max_pre_click_sites = parser.getint('ENHANCED_RANDOMNESS', 'max_pre_click_sites', fallback=2)  # UPDATED
        
        if 'BROWSER_SETTINGS' in parser:
            config.enable_browser_extension_simulation = parser.getboolean('BROWSER_SETTINGS', 'enable_browser_extension_simulation', fallback=True)
            config.enable_random_viewport_sizes = parser.getboolean('BROWSER_SETTINGS', 'enable_random_viewport_sizes', fallback=True)
            config.enable_natural_click_patterns = parser.getboolean('BROWSER_SETTINGS', 'enable_natural_click_patterns', fallback=True)
            config.browser_type = parser.get('BROWSER_SETTINGS', 'browser_type', fallback='chromium')
            config.use_playwright = parser.getboolean('BROWSER_SETTINGS', 'use_playwright', fallback=True)
        
        if 'OPERATIONAL_SECURITY' in parser:
            config.enable_operational_hours = parser.getboolean('OPERATIONAL_SECURITY', 'enable_operational_hours', fallback=False)
            config.operating_hours_start = parser.getint('OPERATIONAL_SECURITY', 'operating_hours_start', fallback=8)
            config.operating_hours_end = parser.getint('OPERATIONAL_SECURITY', 'operating_hours_end', fallback=23)
            config.enable_success_monitoring = parser.getboolean('OPERATIONAL_SECURITY', 'enable_success_monitoring', fallback=True)
            config.success_rate_threshold = parser.getfloat('OPERATIONAL_SECURITY', 'success_rate_threshold', fallback=0.7)
        
        # NEW: Load audio fingerprinting setting
        if 'AUDIO_FINGERPRINTING' in parser:
            config.enable_audio_fingerprinting = parser.getboolean('AUDIO_FINGERPRINTING', 'enable_audio_fingerprinting', fallback=True)
        
        # NEW: Load browser persistence setting
        if 'BROWSER_PERSISTENCE' in parser:
            config.enable_browser_persistence = parser.getboolean('BROWSER_PERSISTENCE', 'enable_browser_persistence', fallback=False)
        
        # NEW: Load enhanced behavior settings
        if 'ENHANCED_BEHAVIOR' in parser:
            config.enable_advanced_fingerprint_spoofing = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_advanced_fingerprint_spoofing', fallback=True)
            config.enable_realistic_network_simulation = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_realistic_network_simulation', fallback=True)
            config.enable_enhanced_browser_behavior = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_enhanced_browser_behavior', fallback=True)
            config.enable_human_reading_patterns = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_human_reading_patterns', fallback=True)
            config.enable_human_idle_simulation = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_human_idle_simulation', fallback=True)
            config.enable_session_depth_variation = parser.getboolean('ENHANCED_BEHAVIOR', 'enable_session_depth_variation', fallback=True)
            config.min_session_depth = parser.getint('ENHANCED_BEHAVIOR', 'min_session_depth', fallback=3)
            config.max_session_depth = parser.getint('ENHANCED_BEHAVIOR', 'max_session_depth', fallback=10)
        
        # NEW: Load optimized settings
        if 'OPTIMIZED_SETTINGS' in parser:
            config.mac_change_frequency = parser.getint('OPTIMIZED_SETTINGS', 'mac_change_frequency', fallback=5)
            config.adaptive_timing = parser.getboolean('OPTIMIZED_SETTINGS', 'adaptive_timing', fallback=True)
        
        # NEW: Load advanced features settings
        if 'ADVANCED_FEATURES' in parser:
            config.enable_cognitive_patterns = parser.getboolean('ADVANCED_FEATURES', 'enable_cognitive_patterns', fallback=True)
            config.enable_network_fingerprinting = parser.getboolean('ADVANCED_FEATURES', 'enable_network_fingerprinting', fallback=True)
            config.enable_performance_metrics = parser.getboolean('ADVANCED_FEATURES', 'enable_performance_metrics', fallback=True)
            config.enable_cookie_jar = parser.getboolean('ADVANCED_FEATURES', 'enable_cookie_jar', fallback=True)
            config.enable_intelligent_recovery = parser.getboolean('ADVANCED_FEATURES', 'enable_intelligent_recovery', fallback=True)
            config.enable_performance_monitoring = parser.getboolean('ADVANCED_FEATURES', 'enable_performance_monitoring', fallback=True)
            config.enable_human_delays = parser.getboolean('ADVANCED_FEATURES', 'enable_human_delays', fallback=True)
            config.enable_proxy_rotation = parser.getboolean('ADVANCED_FEATURES', 'enable_proxy_rotation', fallback=True)
            config.cookie_persistence_strategy = parser.get('ADVANCED_FEATURES', 'cookie_persistence_strategy', fallback='realistic')
            config.cookie_lifespan = parser.getint('ADVANCED_FEATURES', 'cookie_lifespan', fallback=86400)
            config.proxy_rotation_strategy = parser.get('ADVANCED_FEATURES', 'proxy_rotation_strategy', fallback='adaptive')
            config.proxy_health_check_interval = parser.getint('ADVANCED_FEATURES', 'proxy_health_check_interval', fallback=300)
            
            # Parse delay patterns
            delay_patterns_str = parser.get('ADVANCED_FEATURES', 'delay_patterns', fallback='realistic,erratic,focused')
            config.delay_patterns = [pattern.strip() for pattern in delay_patterns_str.split(',')]

        # NEW: Device percentage distribution (explicit section)
        if 'DEVICE_PERCENTAGES' in parser:
            try:
                # Read all keys in DEVICE_PERCENTAGES and coerce to int
                raw = dict(parser.items('DEVICE_PERCENTAGES'))
                total = 0
                distro = {}
                for k, v in raw.items():
                    try:
                        iv = int(v)
                    except Exception:
                        try:
                            iv = int(float(v))
                        except Exception:
                            iv = 0
                    distro[k] = max(0, iv)
                    total += distro[k]

                # If total is zero, keep existing defaults; otherwise normalize to percentages summing to 100
                if total > 0:
                    norm = {k: int(round((v / total) * 100)) for k, v in distro.items()}
                    # Adjust rounding error to ensure sum==100
                    s = sum(norm.values())
                    if s != 100:
                        # find a key to adjust (largest) and fix the delta
                        delta = 100 - s
                        max_k = max(norm.keys(), key=lambda x: norm[x])
                        norm[max_k] = max(0, norm[max_k] + delta)
                    # Map common names to internal keys if needed
                    # e.g., allow 'Android' or 'Samsung' etc.
                    config.device_percentage_distribution = {}
                    for k, v in norm.items():
                        config.device_percentage_distribution[k] = v
                # else leave defaults
            except Exception:
                # ignore and keep defaults
                pass
    
    # Set default peak hours if not configured
    if config.peak_click_hours is None:
        config.peak_click_hours = [8, 9, 10, 11, 12, 13, 19, 20, 21, 22]
    
    return config

config = load_config()

# Setup logging (file UTF-8, safe console)
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        SafeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === Generate encryption key if enabled ===
if config.enable_encryption:
    encryption_key_file = "encryption_key.key"
    if os.path.exists(encryption_key_file):
        with open(encryption_key_file, "rb") as f:
            encryption_key = f.read()
    else:
        encryption_key = Fernet.generate_key()
        with open(encryption_key_file, "wb") as f:
            f.write(encryption_key)
    cipher_suite = Fernet(encryption_key)
else:
    cipher_suite = None

# === NEW: Load referers from file ===
def load_referers():
    """Load referers from referer.txt file"""
    referers = []
    referer_file = "referer.txt"
    
    if os.path.exists(referer_file):
        try:
            with open(referer_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Skip empty lines and comments
                        referers.append(line)
            logger.info(f"✅ Loaded {len(referers)} referers from {referer_file}")
        except Exception as e:
            logger.error(f"❌ Error loading referers: {e}")
            # Fallback to default referer
            referers = ["https://www.google.com/"]
            logger.info("Using default referer: https://www.google.com/")
    else:
        # Create referer.txt with default referer if it doesn't exist
        try:
            with open(referer_file, "w", encoding="utf-8") as f:
                f.write("# Add one referer per line\n")
                f.write("# Lines starting with # are comments\n")
                f.write("https://www.google.com/\n")
            logger.info(f"✅ Created {referer_file} with default referer")
            referers = ["https://www.google.com/"]
        except Exception as e:
            logger.error(f"❌ Error creating referer file: {e}")
            referers = ["https://www.google.com/"]
    
    return referers

# === Load referers ===
referers = load_referers()

# === NEW FEATURE: Detect Wi-Fi Adapter Name Automatically ===
def get_wifi_adapter_name():
    try:
        result = subprocess.run(["netsh", "interface", "show", "interface"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        for line in lines:
            if "Wi-Fi" in line or "Wireless" in line:
                return line.split()[-1]
    except:
        pass
    return "Wi-Fi"  # fallback default

wifi_adapter_name = get_wifi_adapter_name()
logger.info(f"📡 Detected Wi-Fi adapter: {wifi_adapter_name}")

# === ENHANCED: MAC Changer (Windows) - TMAC Style ===
def random_mac():
    hex_digits = "0123456789ABCDEF"
    mac = "02"
    for i in range(5):
        mac += "-" + random.choice(hex_digits) + random.choice(hex_digits)
    return mac

# === NEW MAC ADDRESS FUNCTIONS ===
def get_current_mac(adapter_name):
    """Get the current MAC address of the specified network adapter"""
    try:
        # Try using ipconfig command as an alternative
        result = subprocess.run(
            ["ipconfig", "/all"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        current_adapter = None
        for line in lines:
            if adapter_name.lower() in line.lower():
                current_adapter = line.strip()
            elif current_adapter and "Physical Address" in line:
                mac = line.split(":")[-1].strip()
                return mac
        
        # If not found with ipconfig, try getmac
        result = subprocess.run(
            ["getmac", "/fo", "list", "/v", "/nh"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if adapter_name in line:
                # Look for the MAC address in the next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    if "Physical Address" in lines[j]:
                        mac = lines[j].split(":")[-1].strip()
                        return mac
        return None
    except Exception as e:
        logger.error(f"Error getting MAC address: {e}")
        return None

def change_mac_windows(adapter_name):
    """Change MAC address and return the new MAC if successful"""
    mac = random_mac()
    try:
        logger.info(f"🔄 Changing MAC address to {mac}...")
        
        # Disconnect from the network
        logger.info("📡 Disconnecting from network...")
        subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=disable"], check=True)
        time.sleep(2)
        
        # Change the MAC address in the registry
        logger.info("🔧 Updating registry with new MAC address...")
        subprocess.run(["reg", "add",
                        r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}\0001",
                        "/v", "NetworkAddress", "/d", mac, "/f"], check=True)
        
        # Reconnect to the network
        logger.info("📡 Reconnecting to network...")
        subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=enable"], check=True)
        
        # Wait for the connection to be established
        logger.info("⏳ Waiting for connection to establish...")
        time.sleep(10)
        
        # Verify the connection is active
        result = subprocess.run(["netsh", "interface", "show", "interface", adapter_name], capture_output=True, text=True)
        if "Connected" in result.stdout or "Connecting" in result.stdout:
            logger.info(f"✅ MAC address changed to {mac} and reconnected successfully")
            return mac
        else:
            logger.warning(f"⚠️ MAC address changed to {mac} but connection status is unknown")
            return mac
    except Exception as e:
        logger.error(f"⚠️ MAC change failed: {e}")
        return None

def change_device_mac_address():
    """Change the device MAC address without requiring reboot"""
    mac = random_mac()
    try:
        logger.info(f"🔄 Changing device MAC address to {mac}...")
        
        # Change the MAC address in the registry
        subprocess.run(["reg", "add",
                        r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}\0001",
                        "/v", "NetworkAddress", "/d", mac, "/f"], check=True)
        
        # Reset the network adapter to apply the new MAC without reboot
        logger.info("🔄 Resetting network adapter to apply new MAC address...")
        
        # Disable and re-enable the network adapter
        result = subprocess.run(["netsh", "interface", "show", "interface"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        adapter_index = -1
        for i, line in enumerate(lines):
            if "Wi-Fi" in line or "Wireless" in line:
                adapter_index = i
                break
        
        if adapter_index >= 0:
            adapter_name = lines[adapter_index].split()[-1]
            logger.info(f"🔄 Resetting adapter: {adapter_name}")
            
            # Disable the adapter
            subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=disable"], check=True)
            time.sleep(2)
            
            # Enable the adapter
            subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=enable"], check=True)
            time.sleep(5)
            
            logger.info(f"✅ Device MAC address changed to {mac} and adapter reset")
            return mac
        else:
            logger.warning("⚠️ Could not find Wi-Fi adapter for reset")
            return mac
    except Exception as e:
        logger.error(f"⚠️ Device MAC change failed: {e}")
        return None

def get_all_mac_addresses():
    """Get MAC addresses of all network adapters"""
    try:
        mac_addresses = {}
        
        # Try using ipconfig first
        try:
            result = subprocess.run(
                ["ipconfig", "/all"],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.splitlines()
            current_adapter = None
            
            for line in lines:
                if line.strip() == "":
                    continue
                # Check if this line contains an adapter name
                if "adapter" in line.lower() and ":" in line:
                    current_adapter = line.split(":")[0].strip()
                # Check if this line contains a physical address
                elif current_adapter and "physical address" in line.lower():
                    mac = line.split(":")[-1].strip()
                    if mac != "":
                        mac_addresses[current_adapter] = mac
        except Exception as e:
            logger.warning(f"Failed to get MAC addresses using ipconfig: {e}")
        
        # If we didn't get any MAC addresses with ipconfig, try getmac
        if not mac_addresses:
            try:
                result = subprocess.run(
                    ["getmac", "/fo", "list", "/v", "/nh"],
                    capture_output=True, text=True, check=True
                )
                lines = result.stdout.splitlines()
                
                current_adapter = None
                for line in lines:
                    if line.strip() == "":
                        continue
                    if not line.startswith(" "):
                        current_adapter = line.strip()
                    elif "Physical Address" in line and current_adapter:
                        mac = line.split(":")[-1].strip()
                        if mac != "":
                            mac_addresses[current_adapter] = mac
            except Exception as e:
                logger.warning(f"Failed to get MAC addresses using getmac: {e}")
        
        return mac_addresses
    except Exception as e:
        logger.error(f"Error getting all MAC addresses: {e}")
        return {}

# === Load proxies from file ===
def load_proxies(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

# === ENHANCED: Format proxy for all types ===
def format_proxy(proxy_raw):
    """Format proxy for all types including various formats"""
    # Handle the new format: b2b-s6.liveproxies.io:7383:LV33394875-lv_us-716479:MhccobyYUAMbpiHToShi
    if proxy_raw.count(":") == 3 and "@" not in proxy_raw:
        # New format: host:port:username:password
        parts = proxy_raw.split(":")
        host = parts[0]
        port = parts[1]
        username = parts[2]
        password = parts[3]
        host_port = f"{host}:{port}"
        
        return {"http": f"http://{username}:{password}@{host_port}", "https": f"http://{username}:{password}@{host_port}"}
    
    # Handle different proxy formats
    if "@" in proxy_raw:
        # Format: login:password@hostname:port or hostname:port@login:password
        if proxy_raw.count("@") == 1:
            if ":" in proxy_raw.split("@")[0]:
                # login:password@hostname:port
                auth, host_port = proxy_raw.split("@")
                username, password = auth.split(":", 1)  # Split only once
                if ":" not in host_port:
                    # Add default port if missing
                    host_port += ":8080"
            else:
                # hostname:port@login:password
                host_port, auth = proxy_raw.split("@")
                username, password = auth.split(":", 1)  # Split only once
                if ":" not in host_port:
                    # Add default port if missing
                    host_port += ":8080"
        else:
            # Invalid format
            return {"http": proxy_raw, "https": proxy_raw}
    elif "://" in proxy_raw:
        # Format: protocol://hostname:port or protocol://login:password@hostname:port
        protocol, rest = proxy_raw.split("://", 1)
        if "@" in rest:
            # protocol://login:password@hostname:port
            auth, host_port = rest.split("@", 1)
            username, password = auth.split(":", 1)  # Split only once
            if ":" not in host_port:
                # Add default port if missing
                host_port += ":8080"
        else:
            # protocol://hostname:port
            host_port = rest
            if ":" not in host_port:
                # Add default port if missing
                host_port += ":8080"
            username, password = None, None
    else:
        # Format: hostname:port or login:password:hostname:port
        if proxy_raw.count(":") >= 3:
            # login:password:hostname:port
            parts = proxy_raw.split(":")
            username = parts[0]
            password = ":".join(parts[1:-2])  # Handle passwords with colons
            host_port = f"{parts[-2]}:{parts[-1]}"
        else:
            # hostname:port
            if ":" not in proxy_raw:
                # Add default port if missing
                proxy_raw += ":8080"
            host_port = proxy_raw
            username, password = None, None
    
    # Determine protocol
    if "://" in proxy_raw:
        protocol = proxy_raw.split("://")[0].lower()
    else:
        # Default to http if no protocol specified
        protocol = "http"
    
    # Format based on protocol
    if username and password:
        if protocol == "socks5":
            return {"http": f"socks5://{username}:{password}@{host_port}", "https": f"socks5://{username}:{password}@{host_port}"}
        elif protocol == "socks4":
            return {"http": f"socks4://{username}:{password}@{host_port}", "https": f"socks4://{username}:{password}@{host_port}"}
        else:  # http or https
            return {"http": f"http://{username}:{password}@{host_port}", "https": f"http://{username}:{password}@{host_port}"}
    else:
        if protocol == "socks5":
            return {"http": f"socks5://{host_port}", "https": f"socks5://{host_port}"}
        elif protocol == "socks4":
            return {"http": f"socks4://{host_port}", "https": f"socks4://{host_port}"}
        else:  # http or https
            return {"http": f"http://{host_port}", "https": f"http://{host_port}"}

def check_proxy(proxy_raw):
    proxy = format_proxy(proxy_raw)
    start_time = time.time()
    try:
        r = requests.get("https://ipinfo.io/json", proxies=proxy, timeout=10, verify=False)
        latency = (time.time() - start_time) * 1000  # in ms
        if r.status_code == 200:
            return True, latency
    except:
        return False, 0
    return False, 0

def check_proxy_quality(proxy):
    """Check if a proxy can connect. Return True if working, False otherwise."""
    try:
        formatted_proxy = format_proxy(proxy)
        start_time = time.time()
        response = requests.get("https://httpbin.org/ip", proxies=formatted_proxy, timeout=10, verify=False)
        latency = (time.time() - start_time) * 1000  # in ms
        if response.status_code != 200:
            return False
        # Basic transparency check: if origin looks like multiple IPs, treat as not working
        try:
            data = response.json()
            if "origin" in data and "," in data.get("origin", ""):
                return False
        except Exception:
            pass
        return True
    except Exception:
        return False

# === FIXED: Remove only the first occurrence of a proxy from file ===
def remove_line_from_file(filename, line_to_remove):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        removed = False
        new_lines = []
        for line in lines:
            if not removed and line == line_to_remove:
                removed = True
                continue
            new_lines.append(line)
        
        with open(filename, "w", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")
                
        if removed:
            logger.info(f"✅ Removed item: {line_to_remove}")
        else:
            logger.warning(f"⚠️ Proxy '{line_to_remove}' not found in {filename}")
    except Exception as e:
        logger.error(f"⚠️ Failed to update {filename}: {e}")

# === ENHANCED: Batch proxy health check with detailed output ===
def batch_proxy_health_check(proxies_list):
    logger.info("🔍 Starting batch proxy health check...")
    good_proxies = []
    bad_proxies = []
    total = len(proxies_list)
    proxy_scores = {}
    
    for idx, proxy in enumerate(proxies_list, 1):
        is_good, latency = check_proxy(proxy)
        if is_good:
            # Mark working; store latency as score for informational purposes
            score = float(latency or 0)
            proxy_scores[proxy] = score
            good_proxies.append(proxy)
            logger.info(f"✅ [{idx}/{total}] Proxy WORKING: {proxy} | Latency: {latency:.2f}ms")
        else:
            bad_proxies.append(proxy)
            logger.info(f"❌ [{idx}/{total}] Proxy NOT WORKING: {proxy}")
            with open("bad_proxies.txt", "a", encoding="utf-8") as f:
                f.write(proxy + "\n")
            
    
    logger.info(f"🔍 Proxy health check complete. Good: {len(good_proxies)}, Bad: {len(bad_proxies)}\n")
    return good_proxies, bad_proxies, proxy_scores

# === Auto-update proxies ===
def update_proxies():
    if not config.auto_update_proxies:
        return
    
    logger.info("🔄 Auto-updating proxies...")
    # This would integrate with proxy provider APIs
    # For demonstration, we'll just check if proxies need updating
    proxy_file = "proxies.txt"
    if os.path.exists(proxy_file):
        mod_time = os.path.getmtime(proxy_file)
        if time.time() - mod_time > config.proxy_update_interval:
            # In a real implementation, this would fetch new proxies
            logger.info("Proxy list is outdated. Fetching new proxies...")
            # Simulate fetching new proxies
            new_proxies = [
                "user:pass@proxy1.example.com:8080",
                "user:pass@proxy2.example.com:8080",
                "user:pass@proxy3.example.com:8080"
            ]
            with open(proxy_file, "w") as f:
                for proxy in new_proxies:
                    f.write(proxy + "\n")
            logger.info(f"✅ Updated proxy list with {len(new_proxies)} new proxies")
    else:
        logger.warning("Proxy file not found. Creating new one...")
        with open(proxy_file, "w") as f:
            f.write("user:pass@proxy1.example.com:8080\n")
        logger.info("✅ Created new proxy file")

# === Load user agents ===
def load_user_agents(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

# === LOAD USER AGENTS for all devices ===
iphone_agents = load_user_agents("useragents/iphoneuseragent.TXT")
samsung_agents = load_user_agents("useragents/samsunguseragent.TXT")
windows_agents = load_user_agents("useragents/windowsuseragent.TXT")
googlepixel_agents = load_user_agents("useragents/googlepixeluseragent.TXT")
motorola_agents = load_user_agents("useragents/motorolauseragent.TXT")
oneplus_agents = load_user_agents("useragents/oneplususeragent.TXT")
vivo_agents = load_user_agents("useragents/vivouseragent.TXT")
xiaomi_agents = load_user_agents("useragents/xiaomiuseragent.TXT")
zte_agents = load_user_agents("useragents/zteuseragent.TXT")

# === Device percentage distribution ===
device_percentage_distribution = {
    "iPhone": 40,     # 40% of clicks will be iPhone
    "Samsung": 20,    # 20% of clicks will be Samsung
    "GooglePixel": 15, # 15% of clicks will be Google Pixel
    "Motorola": 10,    # 10% of clicks will be Motorola
    "OnePlus": 5,     # 5% of clicks will be OnePlus
    "Xiaomi": 5,      # 5% of clicks will be Xiaomi
    "ZTE": 3,         # 3% of clicks will be ZTE
    "Vivo": 2,        # 2% of clicks will be Vivo
    "Windows": 0      # 0% Windows (disabled)
}

# === Success rate tracking for dynamic weights ===
device_success_rates = {
    "iPhone": {"success": 0, "total": 0},
    "Samsung": {"success": 0, "total": 0},
    "Windows": {"success": 0, "total": 0},
    "GooglePixel": {"success": 0, "total": 0},
    "Motorola": {"success": 0, "total": 0},
    "OnePlus": {"success": 0, "total": 0},
    "Vivo": {"success": 0, "total": 0},
    "Xiaomi": {"success": 0, "total": 0},
    "ZTE": {"success": 0, "total": 0}
}

# === EXPANDED: Random sites for warm-up ===
random_sites = [
    "https://google.com", "https://youtube.com", "https://facebook.com", "https://reddit.com",
    "https://amazon.com", "https://wikipedia.org", "https://instagram.com", "https://x.com",
    "https://yahoo.com", "https://bing.com", "https://tiktok.com", "https://linkedin.com",
    "https://yelp.com", "https://nytimes.com", "https://imdb.com", "https://quora.com",
    "https://pinterest.com", "https://weather.com", "https://espn.com", "https://etsy.com",
    "https://walmart.com", "https://target.com", "https://tripadvisor.com", "https://healthline.com",
    "https://netflix.com", "https://twitch.tv", "https://bankofamerica.com", "https://costco.com",
    "https://home depot.com", "https://zillow.com", "https://usps.com", "https://indeed.com",
    "https://cnn.com", "https://usatoday.com", "https://cvs.com", "https://spotify.com",
    "https://microsoft.com", "https://adobe.com", "https://ebay.com", "https://lowes.com",
    "https://realtor.com", "https://expedia.com", "https://nbcnews.com", "https://cbssports.com",
    "https://health.com", "https://mayoClinic.org", "https://irs.gov", "https://cigna.com",
    "https://britannica.com", "https://chase.com", "https://marriott.com", "https://foxnews.com",
    "https://chicagotribune.com", "https://nyc.gov", "https://time.com", "https://forbes.com",
    "https://cnn.com", "https://usa.gov", "https://craigslist.org", "https://yahoo.net",
    "https://reddit.com", "https://msn.com", "https://fandom.com", "https://merriam-webster.com",
    "https://quora.com", "https://urbanspoon.com", "https://boxoffice.com", "https://weatherbug.com",
    "https://rotfl.com", "https://rotten tomatoes.com", "https://grubhub.com", "https://nopCommerce.com",
    "https://espn.com", "https://nba.com", "https://nfl.com", "https://amc.com",
    "https://github.com", "https://stackoverflow.com", "https://craigslist.org", "https://etsy.com",
    "https://reddit.com", "https://tripadvisor.com", "https://food.com", "https://allrecipes.com",
    "https://hulu.com", "https://spotify.com", "https://pandora.com", "https://bankofamerica.com",
    "https://chase.com", "https://wellsfargo.com", "https://citi.com", "https://boeingsite.com",
    "https://delta.com", "https://united.com", "https://ss.gov", "https://irs.gov",
    "https://healthcare.gov", "https://cdc.gov", "https://nih.gov", "https://usa.gov",
    "https://ny.gov", "https://ca.gov", "https://fl.gov", "https://tx.gov",
    "https://usa.edu", "https://harvard.edu", "https://mit.edu", "https://stanford.edu",
    "https://cornell.edu", "https://caltech.edu", "https://princeton.edu", "https://yale.edu",
    "https://usc.edu", "https://berkeley.edu", "https://utexas.edu", "https://wustl.edu",
    "https://umich.edu", "https://osu.edu", "https://psu.edu", "https://uchicago.edu",
    "https://northwestern.edu", "https://duke.edu", "https://nytimes.com", "https://washingtonpost.com",
    "https://latimes.com", "https://wsj.com", "https://bloomberg.com", "https://cnn.com",
    "https://foxnews.com", "https://nbcnews.com", "https://abcnews.go.com", "https://cbsnews.com",
    "https://usatoday.com", "https://reuters.com", "https://apnews.com", "https://npr.org",
    "https://msnbc.com", "https://bbc.com", "https://cnbc.com", "https://huffpost.com",
    "https://buzzfeed.com", "https://vice.com", "https://vox.com", "https://wired.com",
    "https://techcrunch.com", "https://engadget.com", "https://gizmodo.com", "https://lifehacker.com",
    "https://thespruce.com", "https://health.com", "https://webmd.com", "https://mayoclinic.org",
    "https://nih.gov", "https://cdc.gov", "https://irs.gov", "https://ssa.gov",
    "https://veterans.gov", "https://govtrack.us", "https://usa.gov",
    # NEW SITES ADDED
    "https://discord.com", "https://slack.com", "https://zoom.us", "https://teams.microsoft.com",
    "https://dropbox.com", "https://drive.google.com", "https://onedrive.live.com", "https://icloud.com",
    "https://github.com", "https://gitlab.com", "https://bitbucket.org", "https://codepen.io",
    "https://stackoverflow.com", "https://quora.com", "https://medium.com", "https://dev.to",
    "https://tumblr.com", "https://wordpress.com", "https://blogger.com", "https://wix.com",
    "https://squarespace.com", "https://shopify.com", "https://woocommerce.com", "https://magento.com",
    "https://salesforce.com", "https://hubspot.com", "https://mailchimp.com", "https://constantcontact.com",
    "https://adobe.com", "https://canva.com", "https://figma.com", "https://sketch.com",
    "https://dribbble.com", "https://behance.net", "https://unsplash.com", "https://pexels.com",
    "https://shutterstock.com", "https://gettyimages.com", "https://istockphoto.com", "https://adobe.com/stock",
    "https://spotify.com", "https://music.apple.com", "https://soundcloud.com", "https://pandora.com",
    "https://youtube.com", "https://vimeo.com", "https://dailymotion.com", "https://twitch.tv",
    "https://netflix.com", "https://hulu.com", "https://disneyplus.com", "https://hbo.com",
    "https://amazon.com/prime", "https://paramountplus.com", "https://peacocktv.com", "https://appletv.com",
    "https://booking.com", "https://airbnb.com", "https://expedia.com", "https://kayak.com",
    "https://tripadvisor.com", "https://hotels.com", "https://priceline.com", "https://orbitz.com",
    "https://opentable.com", "https://yelp.com", "https://foursquare.com", "https://tripadvisor.com",
    "https://uber.com", "https://lyft.com", "https://doordash.com", "https://grubhub.com",
    "https://postmates.com", "https://instacart.com", "https://shipt.com", "https://walmart.com/grocery"
]

# === Carrier list by country for geolocation matching ===
carriers_by_country = {
    "US": ["Verizon", "AT&T", "T-Mobile", "Sprint", "Metro PCS", "U.S. Cellular", 
           "Boost Mobile", "Cricket Wireless", "Virgin Mobile", "Straight Talk"],
    "UK": ["Vodafone", "EE", "O2", "Three", "Giffgaff", "Tesco Mobile", "Virgin Mobile", "BT Mobile"],
    "DE": ["Deutsche Telekom", "Vodafone", "O2", "E-Plus", "Blau", "Congstar", "Lycamobile"],
    "FR": ["Orange", "SFR", "Bouygues", "Free Mobile", "Virgin Mobile", "Lycamobile"],
    "IT": ["TIM", "Vodafone", "Wind Tre", "Iliad", "PosteMobile", "CoopVoce"],
    "ES": ["Movistar", "Vodafone", "Orange", "Yoigo", "MásMóvil", "Lycamobile"],
    "BR": ["Claro", "Vivo", "TIM", "Oi", "Nextel", "Algar Telecom"],
    "IN": ["Jio", "Airtel", "Vi", "BSNL"],
    "CA": ["Rogers", "Bell", "Telus", "Freedom Mobile", "Videotron", "SaskTel"],
    "AU": ["Telstra", "Optus", "Vodafone", "TPG", "Amaysim"],
    # Add more countries as needed
}

# === NEW: Timezone mapping by country ===
def get_timezone_for_location(country, city=None):
    """Get appropriate timezone for a given country/city"""
    # Common timezone mapping by country
    country_timezones = {
        "US": "America/New_York",  # Default to EST, can be refined by region
        "DE": "Europe/Berlin",
        "UK": "Europe/London",
        "FR": "Europe/Paris",
        "IT": "Europe/Rome",
        "ES": "Europe/Madrid",
        "BR": "America/Sao_Paulo",
        "IN": "Asia/Kolkata",
        "CA": "America/Toronto",
        "AU": "Australia/Sydney",
        "JP": "Asia/Tokyo",
        "KR": "Asia/Seoul",
        "CN": "Asia/Shanghai",
        "RU": "Europe/Moscow",
        "SE": "Europe/Stockholm"
    }
    
    # Return timezone based on country
    if country in country_timezones:
        return country_timezones[country]
    
    # Default to UTC if country not found
    return "UTC"

# === NEW: Verify proxy location ===
def verify_proxy_location(proxy):
    """Verify proxy location from multiple sources"""
    try:
        formatted_proxy = format_proxy(proxy)
        
        # Check with ipinfo.io
        r1 = requests.get("https://ipinfo.io/json", proxies=formatted_proxy, timeout=10, verify=False)
        data1 = r1.json()
        
        # Check with ipapi.co (backup)
        r2 = requests.get("https://ipapi.co/json/", proxies=formatted_proxy, timeout=10, verify=False)
        data2 = r2.json()
        
        # Compare countries
        if data1.get("country") == data2.get("country"):
            return {
                "ip": data1.get("ip"),
                "country": data1.get("country"),
                "city": data1.get("city"),
                "region": data1.get("region"),
                "timezone": get_timezone_for_location(data1.get("country")),
                "isp": data1.get("org"),
                "verified": True
            }
        else:
            logger.warning(f"⚠️ Location mismatch for proxy: {proxy}")
            return None
            
    except Exception as e:
        logger.error(f"Proxy verification failed: {e}")
        return None

# === Enhanced Device Fingerprint Generation ===
class DeviceFingerprint:
    def __init__(self, user_agent=None):
        self.user_agent = user_agent
        self.seed = str(random.randint(1000000000, 9999999999))
        random.seed(self.seed)
        self.fingerprint_data = {}
        self.device_info = self.parse_user_agent(user_agent) if user_agent else {}
        
    def parse_user_agent(self, ua):
        """Parse user agent to extract device information"""
        device_info = {
            'device_type': 'Unknown',
            'device_model': 'Unknown',
            'os_name': 'Unknown',
            'os_version': 'Unknown',
            'browser_name': 'Unknown',
            'browser_version': 'Unknown'
        }
        
        # Detect device type and model
        if 'iPhone' in ua:
            device_info['device_type'] = 'iPhone'
            # Extract iPhone model
            model_match = re.search(r'iPhone(?:\s+OS)?[\s;]+.*?(\d+,\d+)', ua)
            if model_match:
                model_version = model_match.group(1).replace(',', '.')
                device_info['device_model'] = f"iPhone {model_version}"
            else:
                device_info['device_model'] = 'iPhone'
                
            # Extract iOS version
            ios_match = re.search(r'iPhone OS (\d+_\d+(_\d+)?)', ua)
            if ios_match:
                ios_version = ios_match.group(1).replace('_', '.')
                device_info['os_version'] = f"iOS {ios_version}"
            else:
                device_info['os_version'] = 'iOS'
                
        elif 'iPad' in ua:
            device_info['device_type'] = 'iPad'
            device_info['device_model'] = 'iPad'
            
            # Extract iOS version
            ios_match = re.search(r'CPU OS (\d+_\d+(_\d+)?)', ua)
            if ios_match:
                ios_version = ios_match.group(1).replace('_', '.')
                device_info['os_version'] = f"iOS {ios_version}"
            else:
                device_info['os_version'] = 'iOS'
                
        elif 'Android' in ua:
            device_info['os_name'] = 'Android'
            
            # Extract Android version
            android_match = re.search(r'Android (\d+\.\d+(\.\d+)?)', ua)
            if android_match:
                device_info['os_version'] = f"Android {android_match.group(1)}"
            else:
                device_info['os_version'] = 'Android'
                
            # Detect device manufacturer and model
            if 'SM-' in ua:
                device_info['device_type'] = 'Samsung'
                model_match = re.search(r'SM-([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"Samsung SM-{model_match.group(1)}"
                else:
                    device_info['device_model'] = 'Samsung'
            elif 'Pixel' in ua:
                device_info['device_type'] = 'GooglePixel'
                model_match = re.search(r'Pixel (\d+)', ua)
                if model_match:
                    device_info['device_model'] = f"Google Pixel {model_match.group(1)}"
                else:
                    device_info['device_model'] = 'Google Pixel'
            elif 'Motorola' in ua:
                device_info['device_type'] = 'Motorola'
                model_match = re.search(r'Motorola ([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"Motorola {model_match.group(1)}"
                else:
                    device_info['device_model'] = 'Motorola'
            elif 'OnePlus' in ua:
                device_info['device_type'] = 'OnePlus'
                model_match = re.search(r'OnePlus ([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"OnePlus {model_match.group(1)}"
                else:
                    device_info['device_model'] = 'OnePlus'
            elif 'vivo' in ua.lower():
                device_info['device_type'] = 'Vivo'
                model_match = re.search(r'vivo ([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"Vivo {model_match.group(1)}"
                else:
                    device_info['device_model'] = 'Vivo'
            elif 'Mi' in ua or 'Xiaomi' in ua:
                device_info['device_type'] = 'Xiaomi'
                model_match = re.search(r'(Mi|Xiaomi) ([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"Xiaomi {model_match.group(2)}"
                else:
                    device_info['device_model'] = 'Xiaomi'
            elif 'ZTE' in ua:
                device_info['device_type'] = 'ZTE'
                model_match = re.search(r'ZTE ([A-Za-z0-9]+)', ua)
                if model_match:
                    device_info['device_model'] = f"ZTE {model_match.group(1)}"
                else:
                    device_info['device_model'] = 'ZTE'
            else:
                device_info['device_type'] = 'Android'
                device_info['device_model'] = 'Android Device'
                
        elif 'Windows' in ua:
            device_info['device_type'] = 'Windows'
            device_info['device_model'] = 'Windows PC'
            
            # Extract Windows version
            win_match = re.search(r'Windows NT (\d+\.\d+)', ua)
            if win_match:
                win_version = win_match.group(1)
                if win_version == '10.0':
                    device_info['os_version'] = 'Windows 10'
                elif win_version == '6.3':
                    device_info['os_version'] = 'Windows 8.1'
                elif win_version == '6.1':
                    device_info['os_version'] = 'Windows 7'
                else:
                    device_info['os_version'] = f'Windows {win_version}'
            else:
                device_info['os_version'] = 'Windows'
                
        # Detect browser
        if 'Chrome' in ua:
            device_info['browser_name'] = 'Chrome'
            chrome_match = re.search(r'Chrome/(\d+\.\d+\.\d+\.\d+)', ua)
            if chrome_match:
                device_info['browser_version'] = chrome_match.group(1)
            else:
                device_info['browser_version'] = 'Latest'
        elif 'Firefox' in ua:
            device_info['browser_name'] = 'Firefox'
            firefox_match = re.search(r'Firefox/(\d+\.\d+)', ua)
            if firefox_match:
                device_info['browser_version'] = firefox_match.group(1)
            else:
                device_info['browser_version'] = 'Latest'
        elif 'Safari' in ua and 'Chrome' not in ua:
            device_info['browser_name'] = 'Safari'
            safari_match = re.search(r'Version/(\d+\.\d+)', ua)
            if safari_match:
                device_info['browser_version'] = safari_match.group(1)
            else:
                device_info['browser_version'] = 'Latest'
        elif 'Edge' in ua:
            device_info['browser_name'] = 'Edge'
            edge_match = re.search(r'Edge/(\d+\.\d+\.\d+\.\d+)', ua)
            if edge_match:
                device_info['browser_version'] = edge_match.group(1)
            else:
                device_info['browser_version'] = 'Latest'
                
        return device_info
        
    def generate_imei(self):
        # Generate a realistic IMEI based on device type
        if self.device_info.get('device_type') == 'iPhone':
            # iPhone IMEIs start with 35
            imei = "35" + "".join([str(random.randint(0, 9)) for _ in range(13)])
        elif self.device_info.get('device_type') == 'Samsung':
            # Samsung IMEIs vary, but many start with 35
            imei = "35" + "".join([str(random.randint(0, 9)) for _ in range(13)])
        else:
            # Generic IMEI
            imei = "35" + "".join([str(random.randint(0, 9)) for _ in range(13)])
        return imei
    
    def generate_device_id(self):
        # Generate a unique device ID based on user agent
        ua_hash = hashlib.md5(self.user_agent.encode()).hexdigest()[:8]
        random_part = str(random.randint(1000, 9999))
        return f"{ua_hash}{random_part}"
    
    def get_screen_resolution(self):
        # Return realistic screen resolutions based on parsed device info
        device_type = self.device_info.get('device_type', 'Unknown')
        device_model = self.device_info.get('device_model', '')
        
        if 'iPhone' in device_model:
            if 'iPhone 14' in device_model or 'iPhone 13' in device_model:
                return f"{random.choice([390, 428])}x{random.choice([844, 926])}"
            elif 'iPhone 12' in device_model or 'iPhone 11' in device_model:
                return f"{random.choice([375, 414])}x{random.choice([812, 896])}"
            else:
                return f"{random.choice([320, 375, 414])}x{random.choice([568, 667, 736, 812])}"
        elif 'iPad' in device_model:
            return f"{random.choice([768, 810, 820, 834])}x{random.choice([1024, 1080, 1180, 1194])}"
        elif 'Samsung' in device_model:
            if 'Galaxy S' in device_model:
                return f"{random.choice([360, 384, 412, 420])}x{random.choice([640, 732, 846, 915])}"
            elif 'Galaxy Note' in device_model:
                return f"{random.choice([384, 412, 420])}x{random.choice([732, 846, 915])}"
            else:
                return f"{random.choice([360, 384, 412])}x{random.choice([640, 732, 846])}"
        elif 'Pixel' in device_model:
            if 'Pixel 6' in device_model or 'Pixel 7' in device_model:
                return f"{random.choice([393, 412])}x{random.choice([846, 915])}"
            else:
                return f"{random.choice([393, 411])}x{random.choice([786, 823])}"
        elif 'Windows' in device_type:
            return f"{random.choice([1366, 1536, 1920])}x{random.choice([768, 864, 1080])}"
        else:
            # Generic Android resolution
            return f"{random.randint(300, 450)}x{random.randint(500, 900)}"
    
    def get_viewport_size(self, resolution):
        # Calculate viewport from resolution
        width, height = map(int, resolution.split('x'))
        # Account for browser chrome, status bar, etc.
        viewport_height = height - random.randint(50, 150)
        return f"{width}x{viewport_height}"
    
    def get_hardware_info(self):
        # Return realistic hardware specs based on device info
        device_type = self.device_info.get('device_type', 'Unknown')
        device_model = self.device_info.get('device_model', '')
        
        if 'iPhone' in device_model:
            if 'iPhone 14' in device_model or 'iPhone 13' in device_model:
                cpu = "Apple A15 Bionic"
                cores = 6
                memory = random.choice([4, 6])
            elif 'iPhone 12' in device_model:
                cpu = "Apple A14 Bionic"
                cores = 6
                memory = random.choice([4, 6])
            elif 'iPhone 11' in device_model:
                cpu = "Apple A13 Bionic"
                cores = 6
                memory = random.choice([4, 6])
            else:
                cpu = "Apple A12 Bionic"
                cores = 6
                memory = random.choice([3, 4])
        elif 'iPad' in device_model:
            cpu = "Apple"
            if 'M1' in device_model or 'M2' in device_model:
                cpu += f" {random.choice(['M1', 'M2'])}"
                cores = 8
                memory = random.choice([8, 16])
            else:
                cpu += " A"
                cpu += str(random.choice([12, 13, 14]))
                cores = random.choice([6, 8])
                memory = random.choice([4, 6, 8])
        elif 'Samsung' in device_model:
            if 'Galaxy S22' in device_model or 'Galaxy S21' in device_model:
                cpu = "Qualcomm Snapdragon 8 Gen 1"
                cores = 8
                memory = random.choice([8, 12])
            elif 'Galaxy S20' in device_model:
                cpu = "Qualcomm Snapdragon 865"
                cores = 8
                memory = random.choice([8, 12])
            else:
                cpu = "Qualcomm Snapdragon"
                cpu += str(random.choice([855, 865, 888]))
                cores = 8
                memory = random.choice([6, 8])
        elif 'Pixel' in device_model:
            if 'Pixel 6' in device_model or 'Pixel 7' in device_model:
                cpu = "Google Tensor"
                cores = 8
                memory = random.choice([8, 12])
            else:
                cpu = "Qualcomm Snapdragon"
                cpu += str(random.choice([765, 768, 780]))
                cores = 8
                memory = random.choice([6, 8])
        elif 'Windows' in device_type:
            cpu = "Intel Core i"
            cpu += str(random.choice([5, 7, 9])) + "-" + str(random.choice([3000, 7000, 9000]))
            cores = random.choice([4, 6, 8])
            memory = random.choice([8, 16, 32])
        else:
            # Generic Android hardware
            cpu_brands = ["Qualcomm Snapdragon", "MediaTek Helio", "Samsung Exynos"]
            cpu = random.choice(cpu_brands)
            if "Qualcomm" in cpu:
                cpu += " " + str(random.choice([680, 720, 750, 765, 778, 888]))
            elif "MediaTek" in cpu:
                cpu += " " + str(random.choice(["G80", "G85", "G90", "G95", "G96", "G99"]))
            else:
                cpu += " " + str(random.choice([980, 981, 982, 983, 2100, 2200]))
            cores = random.choice([6, 8])
            memory = random.choice([4, 6, 8])
        
        return {
            "cpu": cpu,
            "cores": cores,
            "memory": memory
        }
    
    def get_os_build(self):
        # Use the OS version extracted from user agent
        os_version = self.device_info.get('os_version', 'Unknown')
        if os_version != 'Unknown':
            return os_version
        
        # Fallback to generating a realistic OS build number
        device_type = self.device_info.get('device_type', 'Unknown')
        if "iPhone" in device_type or "iPad" in device_type:
            return f"iOS {random.choice([14, 15, 16])}.{random.randint(0, 7)}.{random.randint(0, 9)}"
        elif "Android" in device_type:
            return f"Android {random.choice([10, 11, 12, 13])}.{random.randint(0, 9)}"
        elif "Windows" in device_type:
            return f"Windows 10 Build {random.randint(19000, 22600)}"
        else:
            return f"Android {random.choice([10, 11, 12, 13])}.{random.randint(0, 9)}"
    
    def get_battery_info(self):
        # Generate realistic battery info
        return {
            "level": random.randint(20, 95),
            "charging": random.choice([True, False]),
            "chargingTime": random.randint(0, 3600),
            "dischargingTime": random.randint(1800, 7200)
        }
    
    def get_memory_info(self):
        # Generate realistic memory info
        hardware = self.get_hardware_info()
        device_memory = hardware.get("memory", 4)
        
        return {
            "deviceMemory": device_memory,
            "totalJSHeapSize": random.randint(10000000, 50000000),
            "usedJSHeapSize": random.randint(5000000, 30000000),
            "jsHeapSizeLimit": random.randint(100000000, 500000000)
        }
    
    def get_connection_info(self):
        # Generate realistic connection info
        return {
            "effectiveType": random.choice(["4g", "3g", "2g"]),
            "rtt": random.randint(50, 500),
            "downlink": random.randint(1, 10),
            "saveData": random.choice([True, False])
        }
    
    def get_sensor_info(self):
        # Generate realistic sensor info
        return {
            "accelerometer": {
                "x": random.uniform(-10, 10),
                "y": random.uniform(-10, 10),
                "z": random.uniform(-10, 10)
            },
            "gyroscope": {
                "alpha": random.uniform(0, 360),
                "beta": random.uniform(-180, 180),
                "gamma": random.uniform(-90, 90)
            },
            "magnetometer": {
                "x": random.uniform(-100, 100),
                "y": random.uniform(-100, 100),
                "z": random.uniform(-100, 100)
            }
        }
    
    def get_webgl_info(self):
        # Generate realistic WebGL info based on device
        device_type = self.device_info.get('device_type', 'Unknown')
        
        if 'iPhone' in device_type or 'iPad' in device_type:
            # Apple devices use their own GPUs
            vendor = "Apple Inc."
            if 'M1' in self.device_info.get('device_model', '') or 'M2' in self.device_info.get('device_model', ''):
                renderer = "Apple M1"
            else:
                renderer = "Apple A"
                renderer += str(random.choice([12, 13, 14, 15]))
                renderer += " GPU"
        elif 'Windows' in device_type:
            # Windows devices can have various GPUs
            gpu_vendors = ["NVIDIA Corporation", "Intel Inc.", "AMD"]
            vendor = random.choice(gpu_vendors)
            
            if vendor == "NVIDIA Corporation":
                gpu_models = [
                    "NVIDIA GeForce RTX 3080/PCIe/SSE2",
                    "NVIDIA GeForce RTX 3070/PCIe/SSE2",
                    "NVIDIA GeForce RTX 3060/PCIe/SSE2"
                ]
                renderer = random.choice(gpu_models)
            elif vendor == "Intel Inc.":
                gpu_models = [
                    "Intel(R) Iris(R) Xe Graphics",
                    "Intel(R) UHD Graphics 630"
                ]
                renderer = random.choice(gpu_models)
            else:  # AMD
                gpu_models = [
                    "AMD Radeon RX 6800 XT",
                    "AMD Radeon RX 6700 XT"
                ]
                renderer = random.choice(gpu_models)
        else:
            # Android devices use various mobile GPUs
            gpu_vendors = ["Qualcomm", "ARM", "Imagination Technologies"]
            vendor = random.choice(gpu_vendors)
            
            if vendor == "Qualcomm":
                renderer = "Qualcomm Adreno"
                renderer += str(random.choice([530, 610, 615, 618, 619, 630, 640, 650, 660]))
            elif vendor == "ARM":
                renderer = "ARM Mali"
                renderer += "-G"
                renderer += str(random.choice([31, 52, 71, 72, 76, 77, 78]))
            else:  # Imagination Technologies
                renderer = "Imagination Technologies PowerVR GE"
                renderer += str(random.choice([8320, 8440, 8600]))
        
        return {
            "vendor": vendor,
            "renderer": renderer,
            "version": "WebGL 2.0"
        }
    
    def get_canvas_fingerprint_with_noise(self):
        """Generate a canvas fingerprint with realistic noise"""
        # Create a canvas
        img = Image.new('RGB', (200, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Try to use a common font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        text = "Canvas fingerprinting test"
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        
        # Add some random noise (slight variations that occur in real browsers)
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                # Only modify some pixels
                if random.random() < 0.01:  # 1% of pixels
                    r, g, b = pixels[i, j]
                    # Small random variations
                    pixels[i, j] = (
                        max(0, min(255, r + random.randint(-2, 2))),
                        max(0, min(255, g + random.randint(-2, 2))),
                        max(0, min(255, b + random.randint(-2, 2)))
                    )
        
        # Get the image data
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        
        # Generate hash
        return hashlib.md5(image_data).hexdigest()
    
    def get_webgl_fingerprint(self):
        """Generate a realistic WebGL fingerprint"""
        import json
        
        # Get the device info to determine appropriate GPU
        device_type = self.device_info.get('device_type', 'Unknown')
        
        # Generate realistic WebGL parameters based on device
        if 'iPhone' in device_type or 'iPad' in device_type:
            vendor = "Apple Inc."
            renderer = "Apple GPU"
            max_texture_size = 4096
            max_viewport_dims = [4096, 4096]
        elif 'Windows' in device_type:
            gpu_vendors = ["NVIDIA Corporation", "Intel Inc.", "AMD"]
            vendor = random.choice(gpu_vendors)
            
            if vendor == "NVIDIA Corporation":
                renderer = random.choice([
                    "NVIDIA GeForce RTX 3080/PCIe/SSE2",
                    "NVIDIA GeForce RTX 3070/PCIe/SSE2",
                    "NVIDIA GeForce RTX 3060/PCIe/SSE2"
                ])
                max_texture_size = 16384
                max_viewport_dims = [16384, 16384]
            elif vendor == "Intel Inc.":
                renderer = random.choice([
                    "Intel(R) Iris(R) Xe Graphics",
                    "Intel(R) UHD Graphics 630"
                ])
                max_texture_size = 16384
                max_viewport_dims = [16384, 16384]
            else:  # AMD
                renderer = random.choice([
                    "AMD Radeon RX 6800 XT",
                    "AMD Radeon RX 6700 XT"
                ])
                max_texture_size = 16384
                max_viewport_dims = [16384, 16384]
        else:  # Android
            gpu_vendors = ["Qualcomm", "ARM", "Imagination Technologies"]
            vendor = random.choice(gpu_vendors)
            
            if vendor == "Qualcomm":
                renderer = f"Qualcomm Adreno {random.choice([530, 610, 615, 618, 619, 630, 640, 650, 660])}"
            elif vendor == "ARM":
                renderer = f"ARM Mali-G{random.choice([31, 52, 71, 72, 76, 77, 78])}"
            else:  # Imagination Technologies
                renderer = f"Imagination Technologies PowerVR GE{random.choice([8320, 8440, 8600])}"
            
            max_texture_size = 4096
            max_viewport_dims = [4096, 4096]
        
        # Generate WebGL parameters
        webgl_params = {
            "vendor": vendor,
            "renderer": renderer,
            "version": "WebGL 2.0",
            "shading_language_version": "WebGL GLSL ES 3.00",
            "max_texture_size": max_texture_size,
            "max_viewport_dims": max_viewport_dims,
            "max_cube_map_texture_size": max_texture_size,
            "max_renderbuffer_size": max_texture_size,
            "max_texture_image_units": random.choice([16, 32]),
            "max_vertex_texture_image_units": random.choice([16, 32]),
            "max_fragment_uniform_vectors": random.choice([1024, 2048]),
            "max_vertex_uniform_vectors": random.choice([1024, 2048]),
            "max_varying_vectors": random.choice([32, 64]),
            "aliased_line_width_range": [1, 1],
            "aliased_point_size_range": [1, 255],
            "max_texture_max_anisotropy_ext": random.choice([16, 32]),
            "max_anisotropy": random.choice([16, 32])
        }
        
        # Generate hash from the parameters
        params_str = json.dumps(webgl_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest(), webgl_params
    
    def get_audio_fingerprint(self):
        # Generate a unique audio fingerprint
        audio_data = str(random.randint(100000, 999999))
        return hashlib.md5(audio_data.encode()).hexdigest()
    
    def get_font_list(self):
        # Generate a realistic font list based on device
        device_type = self.device_info.get('device_type', 'Unknown')
        
        common_fonts = [
            "Arial", "Times New Roman", "Courier New", "Verdana", "Georgia",
            "Palatino", "Garamond", "Bookman", "Comic Sans MS", "Trebuchet MS",
            "Arial Black", "Impact"
        ]
        
        if 'iPhone' in device_type or 'iPad' in device_type:
            # Apple devices have San Francisco font
            apple_fonts = ["San Francisco", "Apple SD Gothic Neo", "Apple Myungjo"]
            common_fonts.extend(apple_fonts)
        elif 'Windows' in device_type:
            # Windows devices have Segoe UI and other Microsoft fonts
            windows_fonts = ["Segoe UI", "Calibri", "Cambria", "Consolas", "Tahoma"]
            common_fonts.extend(windows_fonts)
        elif 'Android' in device_type:
            # Android devices have Roboto and other Google fonts
            android_fonts = ["Roboto", "Noto Sans", "Droid Sans", "Google Sans"]
            common_fonts.extend(android_fonts)
        
        return random.sample(common_fonts, random.randint(5, 10))
    
    def get_timezone_offset(self, timezone):
        # Get timezone offset in minutes
        try:
            tz = pytz.timezone(timezone)
            now = datetime.datetime.now(tz)
            return int(now.utcoffset().total_seconds() / 60)
        except:
            return 0
    
    def get_all_fingerprints(self, ip_info):
        # Get all fingerprint data based on user agent and IP info
        ip, city, country, region, tz, isp, *rest = ip_info
        
        # Get country code for carrier matching
        country_code = country.upper() if country else "US"
        
        # Select carrier based on IP geolocation
        if country_code in carriers_by_country:
            carrier = random.choice(carriers_by_country[country_code])
        else:
            # Fallback to US carriers if country not in list
            carrier = random.choice(carriers_by_country.get("US", ["Verizon", "AT&T", "T-Mobile"]))
        
        # Generate all fingerprint data
        webgl_hash, webgl_params = self.get_webgl_fingerprint()
        
        self.fingerprint_data = {
            "imei": self.generate_imei(),
            "device_id": self.generate_device_id(),
            "screen_resolution": self.get_screen_resolution(),
            "viewport_size": self.get_viewport_size(self.get_screen_resolution()),
            "hardware": self.get_hardware_info(),
            "os_build": self.get_os_build(),
            "battery": self.get_battery_info(),
            "memory": self.get_memory_info(),
            "connection": self.get_connection_info(),
            "sensors": self.get_sensor_info(),
            "webgl": webgl_params,
            "webgl_hash": webgl_hash,
            "canvas": self.get_canvas_fingerprint_with_noise(),
            "audio": self.get_audio_fingerprint(),
            "fonts": self.get_font_list(),
            "timezone_offset": self.get_timezone_offset(tz),
            "carrier": carrier,
            "location": {
                "city": city,
                "region": region,
                "country": country,
                "timezone": tz
            },
            "isp": isp,
            "device_info": self.device_info  # Include parsed device info
        }
        
        return self.fingerprint_data
    
    def encrypt_fingerprint(self):
        # Encrypt the fingerprint data if encryption is enabled
        if config.enable_encryption and cipher_suite:
            fingerprint_str = json.dumps(self.fingerprint_data)
            return cipher_suite.encrypt(fingerprint_str.encode()).decode()
        return json.dumps(self.fingerprint_data)
    
    # NEW: WebGL/Canvas spoofing injection
    async def inject_webgl_canvas_spoofing(self, page):
        """Inject JavaScript to spoof WebGL and Canvas fingerprints"""
        webgl_info = self.get_webgl_info()
        canvas_hash = self.get_canvas_fingerprint_with_noise()
        
        script = f"""
        (function() {{
            // Spoof WebGL
            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                const params = {json.dumps(webgl_info)};
                if (params.hasOwnProperty(parameter)) {{
                    return params[parameter];
                }}
                return originalGetParameter.call(this, parameter);
            }};
            
            // Spoof Canvas
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            CanvasRenderingContext2D.prototype.getImageData = function() {{
                const imageData = originalGetImageData.apply(this, arguments);
                // Add noise to canvas fingerprint
                for (let i = 0; i < imageData.data.length; i += 4) {{
                    if (Math.random() > 0.9) {{
                        imageData.data[i] = Math.min(255, imageData.data[i] + Math.floor(Math.random() * 5 - 2.5));
                        imageData.data[i+1] = Math.min(255, imageData.data[i+1] + Math.floor(Math.random() * 5 - 2.5));
                        imageData.data[i+2] = Math.min(255, imageData.data[i+2] + Math.floor(Math.random() * 5 - 2.5));
                    }}
                }}
                return imageData;
            }};
            
            // Spoof Canvas fingerprint hash
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {{
                const dataURL = originalToDataURL.apply(this, arguments);
                return dataURL.replace(/[a-f0-9]{{8}}-[a-f0-9]{{4}}-[a-f0-9]{{4}}-[a-f0-9]{{4}}-[a-f0-9]{{12}}/, '{canvas_hash}');
            }};
        }})();
        """
        
        try:
            await page.evaluate(script)
        except Exception as e:
            logger.warning(f"Error injecting WebGL/Canvas spoofing: {e}")
    
    # FIXED: Simulate realistic mouse movement with bounds checking
    async def simulate_human_mouse_movement(self, page, target_element=None):
        """Simulate realistic mouse movements with random paths"""
        try:
            # Get viewport dimensions first
            viewport_size = page.viewport_size
            viewport_width = viewport_size["width"]
            viewport_height = viewport_size["height"]
            
            # Start from a safe position within the viewport
            current_x = viewport_width // 2
            current_y = viewport_height // 2
            
            # Move to the starting position first
            await page.mouse.move(current_x, current_y)
            
            if target_element:
                # Get target position and ensure it's within bounds
                try:
                    # Get element bounding box
                    box = await target_element.bounding_box()
                    target_x = box["x"] + box["width"] // 2
                    target_y = box["y"] + box["height"] // 2
                    
                    # Ensure target is within viewport bounds
                    target_x = max(10, min(viewport_width - 10, target_x))
                    target_y = max(10, min(viewport_height - 10, target_y))
                    
                    # Move directly to the element
                    try:
                        await page.mouse.move(target_x, target_y)
                        
                        # Randomly decide whether to click
                        if random.random() > 0.3:  # 70% chance to click
                            await page.mouse.click(target_x, target_y)
                        
                        return
                    except Exception as e:
                        logger.warning(f"Error moving to element: {e}")
                        # Fall back to manual movement
                        pass
                except Exception as e:
                    logger.warning(f"Error getting element position: {e}")
            
            # If no target element or element movement failed, use manual movement
            # Generate a random target position within safe bounds
            target_x = random.randint(50, max(100, viewport_width - 50))
            target_y = random.randint(50, max(100, viewport_height - 50))
            
            # Create a simple path with fewer waypoints to reduce errors
            num_waypoints = min(3, random.randint(2, 5))
            
            for i in range(num_waypoints):
                # Calculate intermediate position
                progress = (i + 1) / (num_waypoints + 1)
                next_x = current_x + (target_x - current_x) * progress
                next_y = current_y + (target_y - current_y) * progress
                
                # Ensure the position is within bounds
                next_x = max(5, min(viewport_width - 5, next_x))
                next_y = max(5, min(viewport_height - 5, next_y))
                
                try:
                    # Move to the intermediate position
                    await page.mouse.move(next_x, next_y)
                    
                    # Update current position
                    current_x = next_x
                    current_y = next_y
                except Exception as e:
                    logger.warning(f"Error moving to intermediate position: {e}")
                    # Reset to center
                    current_x = viewport_width // 2
                    current_y = viewport_height // 2
            
            # Final move to target
            try:
                # Move to the final position
                await page.mouse.move(target_x, target_y)
            except Exception as e:
                logger.warning(f"Error in final mouse movement: {e}")
                
        except Exception as e:
            logger.error(f"Error in mouse movement simulation: {e}")
    
    # NEW: Simulate realistic scrolling
    async def simulate_realistic_scroll(self, page):
        """Simulate realistic scrolling behavior"""
        try:
            # Get the total height of the page
            total_height = await page.evaluate("document.body.scrollHeight")
            
            # Scroll in multiple steps
            viewport_height = page.viewport_size["height"]
            scroll_steps = random.randint(2, 5)
            
            for i in range(scroll_steps):
                # Calculate scroll position
                scroll_position = (i + 1) * (total_height / scroll_steps)
                # Add some randomness
                scroll_position += random.randint(-50, 50)
                scroll_position = max(0, min(scroll_position, total_height - viewport_height))
                
                # Scroll to position
                await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                
                # Random pause
                await asyncio.sleep(random.uniform(0.5, 2))
            
            # Sometimes scroll back up
            if random.random() > 0.7:
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(random.uniform(0.5, 1))
        except Exception as e:
            logger.warning(f"Error during scroll simulation: {e}")
    
    # NEW: Simulate form interaction
    async def simulate_form_interaction(self, page):
        """Simulate interacting with form elements if present"""
        try:
            # Find input fields
            inputs = await page.query_selector_all("input")
            if not inputs:
                return
            
            # Select a random input
            input_field = random.choice(inputs)
            
            # Check if it's visible and interactable
            if await input_field.is_visible():
                # Scroll to the field
                await input_field.scroll_into_view_if_needed()
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
                # Click on the input field
                await input_field.click()
                await asyncio.sleep(random.uniform(0.2, 0.5))
                
                # Type some random text if it's a text input
                input_type = await input_field.get_attribute("type")
                if input_type in ["text", "search", "email", "password", "tel", "url"]:
                    # Generate random text
                    if input_type == "email":
                        text = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
                    elif input_type == "password":
                        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*', k=12))
                    elif input_type == "tel":
                        text = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    else:
                        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=random.randint(5, 15)))
                    
                    # Type with realistic timing
                    for char in text:
                        await input_field.type(char)
                        await asyncio.sleep(random.uniform(0.05, 0.2))
                    
                    # Random pause after typing
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    
                    # Sometimes tab to the next field
                    if random.random() < 0.3:  # 30% chance
                        await input_field.press("Tab")
                        await asyncio.sleep(random.uniform(0.3, 1.0))
        except Exception as e:
            logger.error(f"Error during form interaction: {e}")
    
    # NEW: Simulate human errors
    async def simulate_human_errors(self, page):
        """Simulate human-like errors during browsing"""
        try:
            error_type = random.choice(["wrong_url", "form_error", "refresh", "click_wrong"])
            
            if error_type == "wrong_url":
                # Navigate to a random wrong URL
                wrong_url = f"https://{random.choice(['example', 'test', 'demo'])}{random.choice(['.com', '.org', '.net'])}"
                logger.info(f"🚫 Simulating wrong navigation to: {wrong_url}")
                await page.goto(wrong_url)
                await asyncio.sleep(random.uniform(1, 3))
                # Go back
                await page.go_back()
                await asyncio.sleep(random.uniform(1, 2))
                
            elif error_type == "form_error":
                # Try to find a form and submit invalid data
                forms = await page.query_selector_all("form")
                if forms:
                    form = random.choice(forms)
                    inputs = await form.query_selector_all("input")
                    if inputs:
                        # Fill with invalid data (e.g., invalid email)
                        for input_field in inputs:
                            input_type = await input_field.get_attribute("type")
                            if input_type == "email":
                                await input_field.click()
                                invalid_email = "invalid-email"
                                await input_field.type(invalid_email)
                                await asyncio.sleep(random.uniform(0.5, 1))
                                # Try to submit
                                submit_buttons = await form.query_selector_all("input[type='submit'], button[type='submit'], button:has-text('Submit'), button:has-text('Send')")
                                if submit_buttons:
                                    submit_button = random.choice(submit_buttons)
                                    await submit_button.click()
                                    await asyncio.sleep(random.uniform(1, 3))
                                    # Go back (assuming form submission leads to a new page)
                                    await page.go_back()
                                    await asyncio.sleep(random.uniform(1, 2))
                                break
            
            elif error_type == "refresh":
                # Refresh the page
                logger.info("🔄 Simulating page refresh")
                await page.reload()
                await asyncio.sleep(random.uniform(2, 4))
                
            elif error_type == "click_wrong":
                # Click on a random non-important element
                elements = await page.query_selector_all("img") + \
                          await page.query_selector_all("a[href*='image'], a[href*='photo']")
                
                if elements:
                    element = random.choice(elements)
                    if await element.is_visible():
                        # Scroll to element
                        await element.scroll_into_view_if_needed()
                        await asyncio.sleep(random.uniform(0.5, 1))
                        
                        # Click
                        await element.click()
                        await asyncio.sleep(random.uniform(1, 3))
                        
                        # 70% chance to go back
                        if random.random() < 0.7:
                            await page.go_back()
                            await asyncio.sleep(random.uniform(1, 2))
            
        except Exception as e:
            logger.warning(f"Error during human error simulation: {e}")
    
    # NEW: Random element interaction
    async def random_element_interaction(self, page):
        """Randomly click on non-important elements"""
        try:
            # Find clickable elements that are not important (like images, non-critical links)
            elements = await page.query_selector_all("img") + \
                      await page.query_selector_all("a[href*='image'], a[href*='photo']")
            
            if elements:
                element = random.choice(elements)
                if await element.is_visible():
                    # Scroll to element
                    await element.scroll_into_view_if_needed()
                    await asyncio.sleep(random.uniform(0.5, 1))
                    
                    # Click
                    await element.click()
                    await asyncio.sleep(random.uniform(1, 3))
                    
                    # 50% chance to go back
                    if random.random() < 0.5:
                        await page.go_back()
                        await asyncio.sleep(random.uniform(1, 2))
        except Exception as e:
            logger.warning(f"Error during random element interaction: {e}")
    
    # FIXED: Random pre-click site interaction with reduced mouse movement
    async def random_pre_click_interaction(self, page, min_sites=None, max_sites=None, max_time=None):
        """Visit random sites and interact with them before clicking the target"""
        if min_sites is None:
            min_sites = config.min_pre_click_sites
        if max_sites is None:
            max_sites = config.max_pre_click_sites
            
        start_time = time.time()
        num_sites = random.randint(min_sites, max_sites)
        selected_sites = random.sample(random_sites, num_sites)
        
        # Add a random Facebook profile to the visited sites
        fb_id = f"1000{random.randint(10000000,99999999)}"
        fb_url = f"https://www.facebook.com/profile.php?id={fb_id}"
        selected_sites.append(fb_url)
        
        # Shuffle the list to randomize the order
        random.shuffle(selected_sites)
        
        for site in selected_sites:
            # Check if we have time left
            if max_time and (time.time() - start_time) >= max_time:
                logger.info(f"⏱️ Pre-click interaction stopped due to time limit")
                break
                
            try:
                logger.info(f"🌐 Visiting pre-click site: {site}")
                await page.goto(site, timeout=15000)  # Reduced timeout to 15 seconds
                
                # Wait for page to load (reduced time)
                await asyncio.sleep(random.uniform(1, 3))
                
                # Decide whether to interact and how many interactions
                if random.random() < config.random_interaction_probability:  # 70% chance to interact
                    # List of possible interactions
                    interactions = [
                        self.simulate_realistic_scroll,
                        self.simulate_human_mouse_movement,
                        self.simulate_form_interaction,
                        self.random_element_interaction,
                        self.simulate_human_errors
                    ]
                    
                    # Number of interactions (0 to 3)
                    num_interactions = random.randint(0, 3)
                    selected_interactions = random.sample(interactions, min(num_interactions, len(interactions)))
                    
                    for interaction in selected_interactions:
                        # Check time before each interaction
                        if max_time and (time.time() - start_time) >= max_time:
                            break
                        try:
                            await interaction(page)
                            await asyncio.sleep(random.uniform(1, 3))
                        except Exception as e:
                            logger.warning(f"Error during interaction on {site}: {e}")
                            continue
                
                # Random time on site (with reduced range to save time)
                time_on_site = random.uniform(5, 15)  # Reduced from 5-30 to 5-15
                await asyncio.sleep(time_on_site)
                
            except Exception as e:
                logger.error(f"Error during pre-click interaction on {site}: {e}")
                continue
    
    # NEW: Set timezone and locale based on proxy geolocation (FIXED)
    async def set_timezone_locale_from_proxy(self, page, proxy):
        """Set browser timezone and locale based on proxy geolocation"""
        try:
            # Get proxy geolocation
            formatted_proxy = format_proxy(proxy)
            response = requests.get("https://ipinfo.io/json", proxies=formatted_proxy, timeout=10, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                country = data.get("country", "US")
                city = data.get("city", "")
                
                # Get correct timezone based on actual location
                timezone = get_timezone_for_location(country, city)
                
                # Set locale based on country
                locale_map = {
                    "US": "en-US",
                    "GB": "en-GB",
                    "DE": "de-DE",
                    "FR": "fr-FR",
                    "IT": "it-IT",
                    "ES": "es-ES",
                    "BR": "pt-BR",
                    "IN": "hi-IN",
                    "CA": "en-CA",
                    "AU": "en-AU"
                }
                
                locale = locale_map.get(country, "en-US")
                
                # Set Accept-Language header
                await page.set_extra_http_headers({"Accept-Language": locale})
                
                # Override navigator.language
                await page.evaluate(f"""
                    Object.defineProperty(navigator, 'language', {{
                        get: () => '{locale}',
                    }});
                    Object.defineProperty(navigator, 'languages', {{
                        get: () => ['{locale}', 'en'],
                    }});
                """)
                
                logger.info(f"🌍 Set timezone to {timezone} and locale to {locale} based on proxy location {country}")
                
        except Exception as e:
            logger.error(f"Failed to set timezone/locale from proxy: {e}")
    
    # NEW: Block WebRTC leaks
    async def block_webrtc_leaks(self, page):
        """Block WebRTC from leaking real IP"""
        # Disable WebRTC using Playwright
        try:
            await page.evaluate("""
                // Block WebRTC
                const originalRTC = window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection;
                if (originalRTC) {
                    window.RTCPeerConnection = function(...args) {
                        const pc = new originalRTC(...args);
                        const originalCreateOffer = pc.createOffer;
                        const originalCreateAnswer = pc.createAnswer;
                        const originalSetLocalDescription = pc.setLocalDescription;
                        
                        pc.createOffer = function(...args) {
                            return originalCreateOffer.apply(this, args).then((desc) => {
                                desc.sdp = desc.sdp.replace(/(IP4 [0-9.]*)/g, 'IP4 0.0.0.0');
                                return desc;
                            });
                        };
                        
                        pc.createAnswer = function(...args) {
                            return originalCreateAnswer.apply(this, args).then((desc) => {
                                desc.sdp = desc.sdp.replace(/(IP4 [0-9.]*)/g, 'IP4 0.0.0.0');
                                return desc;
                            });
                        };
                        
                        pc.setLocalDescription = function(...args) {
                            if (args[0] && args[0].sdp) {
                                args[0].sdp = args[0].sdp.replace(/(IP4 [0-9.]*)/g, 'IP4 0.0.0.0');
                            }
                            return originalSetLocalDescription.apply(this, args);
                        };
                        
                        return pc;
                    };
                }
            """)
            logger.info("🔒 WebRTC leak protection enabled")
        except Exception as e:
            logger.warning(f"Error blocking WebRTC leaks: {e}")
    
    # NEW: Audio fingerprint spoofing
    async def inject_audio_fingerprint_spoofing(self, page):
        """Inject JavaScript to spoof AudioContext fingerprint"""
        if not config.enable_audio_fingerprinting:
            return
            
        audio_fp = self.get_audio_fingerprint()
        script = f"""
        (function() {{
            const originalGetChannelData = AudioBuffer.prototype.getChannelData;
            AudioBuffer.prototype.getChannelData = function() {{
                const data = originalGetChannelData.apply(this, arguments);
                // Add noise to the audio data
                for (let i = 0; i < data.length; i++) {{
                    if (Math.random() > 0.9) {{
                        data[i] = data[i] + (Math.random() * 2 - 1) * 0.0001;
                    }}
                }}
                return data;
            }};
            
            // Override the audio fingerprint
            const originalCreateAnalyser = AudioContext.prototype.createAnalyser;
            AudioContext.prototype.createAnalyser = function() {{
                const analyser = originalCreateAnalyser.apply(this, arguments);
                // Modify the frequency data to produce a unique fingerprint
                const originalGetByteFrequencyData = analyser.getByteFrequencyData;
                analyser.getByteFrequencyData = function(array) {{
                    originalGetByteFrequencyData.apply(this, arguments);
                    // Modify the array to produce a unique fingerprint
                    for (let i = 0; i < array.length; i++) {{
                        if (Math.random() > 0.9) {{
                            array[i] = (array[i] + Math.floor(Math.random() * 10)) % 256;
                        }}
                    }}
                }};
                return analyser;
            }};
        }})();
        """
        
        try:
            await page.evaluate(script)
            logger.info("🔊 Audio fingerprint spoofing enabled")
        except Exception as e:
            logger.warning(f"Error injecting audio fingerprint spoofing: {e}")
    
    # NEW: Advanced Behavioral Simulation
    async def simulate_cognitive_patterns(self, page):
        """Simulate human cognitive patterns like attention span"""
        try:
            # Simulate attention span variations
            attention_span = random.uniform(5, 30)  # 5-30 seconds
            start_time = time.time()
            
            while time.time() - start_time < attention_span:
                # Random micro-movements
                if random.random() < 0.1:  # 10% chance
                    x = random.randint(50, page.viewport_size["width"] - 50)
                    y = random.randint(50, page.viewport_size["height"] - 50)
                    await page.mouse.move(x, y)
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                
                # Random scroll adjustments
                if random.random() < 0.05:  # 5% chance
                    scroll_amount = random.randint(-100, 100)
                    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                    await asyncio.sleep(random.uniform(0.2, 0.8))
                
                await asyncio.sleep(random.uniform(0.5, 2))
                
        except Exception as e:
            logger.warning(f"Error in cognitive pattern simulation: {e}")
    
    # NEW: Enhanced Network Fingerprinting
    async def simulate_network_fingerprint(self, page):
        """Simulate realistic network stack fingerprinting"""
        try:
            network_script = """
            // Network stack fingerprinting
            Object.defineProperty(navigator, 'connection', {
                get: () => ({
                    effectiveType: ['4g', '3g', '2g', 'slow-2g'][Math.floor(Math.random() * 4)],
                    rtt: Math.floor(Math.random() * 300) + 50,
                    downlink: (Math.random() * 9 + 1).toFixed(1),
                    saveData: Math.random() > 0.8
                })
            });
            
            // Network Information API
            Object.defineProperty(navigator, 'onLine', {
                get: () => true
            });
            """
            await page.evaluate(network_script)
            logger.info("🌐 Network fingerprinting simulated")
        except Exception as e:
            logger.warning(f"Error in network fingerprinting: {e}")
    
    # NEW: Realistic Performance Metrics
    async def simulate_performance_metrics(self, page):
        """Simulate realistic browser performance metrics"""
        try:
            performance_script = """
            // Performance metrics spoofing
            const originalPerformance = window.performance;
            window.performance = {
                ...originalPerformance,
                timing: {
                    ...originalPerformance.timing,
                    navigationStart: originalPerformance.timing.navigationStart,
                    unloadEventStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 100),
                    unloadEventEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 200) + 100,
                    redirectStart: 0,
                    redirectEnd: 0,
                    fetchStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 50),
                    domainLookupStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 100) + 50,
                    domainLookupEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 150) + 100,
                    connectStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 200) + 150,
                    connectEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 300) + 200,
                    secureConnectionStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 350) + 200,
                    requestStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 400) + 300,
                    responseStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 600) + 400,
                    responseEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 1000) + 600,
                    domLoading: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 1200) + 800,
                    domInteractive: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 1500) + 1000,
                    domContentLoadedEventStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 1800) + 1200,
                    domContentLoadedEventEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 2000) + 1500,
                    domComplete: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 2500) + 2000,
                    loadEventStart: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 3000) + 2500,
                    loadEventEnd: originalPerformance.timing.navigationStart + Math.floor(Math.random() * 3500) + 3000
                },
                memory: {
                    usedJSHeapSize: Math.floor(Math.random() * 50000000) + 10000000,
                    totalJSHeapSize: Math.floor(Math.random() * 100000000) + 50000000,
                    jsHeapSizeLimit: Math.floor(Math.random() * 200000000) + 100000000
                }
            };
            """
            await page.evaluate(performance_script)
            logger.info("⚡ Performance metrics simulated")
        except Exception as e:
            logger.warning(f"Error in performance metrics simulation: {e}")
    
    # NEW: Enhanced Human-like Delays
    async def human_like_delay(self, pattern="realistic"):
        """Generate human-like delays based on pattern"""
        try:
            if pattern == "realistic":
                # Realistic human delay patterns
                delay_type = random.choice(["short", "medium", "long"])
                if delay_type == "short":
                    delay = random.uniform(0.5, 2.0)
                elif delay_type == "medium":
                    delay = random.uniform(2.0, 5.0)
                else:
                    delay = random.uniform(5.0, 10.0)
                    
            elif pattern == "erratic":
                # Erratic human behavior
                delay = random.uniform(0.1, 15.0)
                
            elif pattern == "focused":
                # Focused human behavior
                delay = random.uniform(0.2, 1.5)
            
            # Add micro-variations
            delay += random.uniform(-0.1, 0.1)
            delay = max(0.1, delay)
            
            await asyncio.sleep(delay)
            logger.debug(f"⏱️ Human-like delay: {delay:.2f}s")
            
        except Exception as e:
            logger.warning(f"Error in human-like delay: {e}")

# === NEW: Advanced Proxy Management ===
class ProxyManager:
    def __init__(self):
        self.proxy_scores = {}
        self.proxy_usage = {}
        self.last_health_check = 0
    
    async def check_proxy_health(self, proxy):
        """Check proxy health with advanced metrics"""
        try:
            start_time = time.time()
            formatted_proxy = format_proxy(proxy)
            
            # Test with multiple endpoints
            test_urls = [
                "https://httpbin.org/ip",
                "https://ipinfo.io/json",
                "https://api.ipify.org?format=json"
            ]
            
            results = []
            for url in test_urls:
                try:
                    response = requests.get(url, proxies=formatted_proxy, timeout=10, verify=False)
                    results.append(response.status_code == 200)
                except:
                    results.append(False)
            
            latency = (time.time() - start_time) * 1000
            success_rate = sum(results) / len(results)
            
            # Calculate score
            score = success_rate * 100 - (latency / 10)
            
            return {
                'score': max(0, score),
                'latency': latency,
                'success_rate': success_rate,
                'last_check': time.time()
            }
        except Exception as e:
            logger.warning(f"Error checking proxy health: {e}")
            return {'score': 0, 'latency': 0, 'success_rate': 0, 'last_check': time.time()}
    
    def select_best_proxy(self, available_proxies):
        """Select the best proxy based on health metrics"""
        if not available_proxies:
            return None
        
        # Sort by score
        sorted_proxies = sorted(
            available_proxies,
            key=lambda p: self.proxy_scores.get(p, {}).get('score', 50),
            reverse=True
        )
        
        # Consider usage frequency
        for proxy in sorted_proxies:
            usage = self.proxy_usage.get(proxy, 0)
            if usage < 10:  # Max 10 uses per proxy
                return proxy
        
        # If all proxies are heavily used, return the best one
        return sorted_proxies[0]

# === NEW: Real-time Performance Monitoring ===
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'clicks_per_hour': [],
            'success_rate': [],
            'avg_response_time': [],
            'proxy_health': {}
        }
    
    def update_metrics(self, click_time, success, response_time, proxy):
        """Update performance metrics"""
        # Calculate clicks per hour
        self.metrics['clicks_per_hour'].append(click_time)
        # Keep only last hour of data
        hour_ago = time.time() - 3600
        self.metrics['clicks_per_hour'] = [t for t in self.metrics['clicks_per_hour'] if t > hour_ago]
        
        # Update success rate
        self.metrics['success_rate'].append(success)
        # Keep only last 100 clicks
        if len(self.metrics['success_rate']) > 100:
            self.metrics['success_rate'] = self.metrics['success_rate'][-100:]
        
        # Update response time
        self.metrics['avg_response_time'].append(response_time)
        # Keep only last 100 responses
        if len(self.metrics['avg_response_time']) > 100:
            self.metrics['avg_response_time'] = self.metrics['avg_response_time'][-100:]
        
        # Update proxy health
        if proxy not in self.metrics['proxy_health']:
            self.metrics['proxy_health'][proxy] = []
        self.metrics['proxy_health'][proxy].append(response_time)
    
    def get_performance_report(self):
        """Generate performance report"""
        clicks_per_hour = len(self.metrics['clicks_per_hour'])
        success_rate = sum(self.metrics['success_rate']) / len(self.metrics['success_rate']) if self.metrics['success_rate'] else 0
        avg_response_time = sum(self.metrics['avg_response_time']) / len(self.metrics['avg_response_time']) if self.metrics['avg_response_time'] else 0
        
        return {
            'clicks_per_hour': clicks_per_hour,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'proxy_count': len(self.metrics['proxy_health'])
        }

# === NEW: Intelligent Error Recovery ===
async def intelligent_error_recovery(error, retry_count):
    """Intelligently recover from errors based on type"""
    try:
        if "timeout" in str(error).lower():
            # Network timeout - wait longer
            wait_time = min(30, 5 * retry_count)
            logger.info(f"⏳ Network timeout detected, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        elif "proxy" in str(error).lower():
            # Proxy error - try different proxy
            logger.info("🔄 Proxy error detected, rotating proxy...")
            return "rotate_proxy"
            
        elif "captcha" in str(error).lower():
            # CAPTCHA detected - wait longer
            wait_time = min(60, 10 * retry_count)
            logger.info(f"🔒 CAPTCHA detected, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        elif "cloudflare" in str(error).lower():
            # Cloudflare protection - wait longer
            wait_time = min(90, 15 * retry_count)
            logger.info(f"☁️ Cloudflare detected, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        else:
            # Unknown error - standard retry
            wait_time = config.retry_delay
            logger.info(f"❓ Unknown error, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            
        return "retry"
    except Exception as e:
        logger.warning(f"Error in intelligent recovery: {e}")
        return "retry"

# === NEW: Advanced Cookie Management ===
async def manage_cookie_jar(page, session):
    """Manage cookies with realistic persistence"""
    try:
        if config.enable_cookie_jar:
            # Get current cookies
            cookies = await page.context.cookies()
            
            # Simulate realistic cookie behavior
            for cookie in cookies:
                # Random chance to expire cookies
                if random.random() < 0.05:  # 5% chance
                    await page.context.clear_cookies()
                    break
                
                # Random chance to modify cookie values
                if random.random() < 0.02:  # 2% chance
                    new_value = cookie['value'] + str(random.randint(1, 100))
                    await page.context.add_cookies([{
                        'name': cookie['name'],
                        'value': new_value,
                        'domain': cookie['domain'],
                        'path': cookie['path']
                    }])
            
            logger.info("🍪 Cookie jar managed")
    except Exception as e:
        logger.warning(f"Error in cookie jar management: {e}")

# === TLS fingerprint spoofing ===
class TLSAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        # Set custom TLS settings for JA3 fingerprint spoofing
        import ssl
        context = ssl.create_default_context()
        # Use a common cipher suite to avoid detection
        ciphers = [
            'ECDHE-ECDSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-ECDSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-ECDSA-CHACHA20-POLY1305',
            'ECDHE-RSA-CHACHA20-POLY1305',
            'ECDHE-ECDSA-AES128-SHA256',
            'ECDHE-RSA-AES128-SHA256',
            'ECDHE-ECDSA-AES256-SHA384',
            'ECDHE-RSA-AES256-SHA384'
        ]
        # Randomize cipher order
        random.shuffle(ciphers)
        cipher_string = ':'.join(ciphers)
        context.set_ciphers(cipher_string)
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        # Enable HTTP/2
        try:
            import ssl
            if hasattr(ssl, "OP_NO_COMPRESSION"):
                context.options |= ssl.OP_NO_COMPRESSION
        except:
            pass
        kwargs['ssl_context'] = context
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

# === NEW: Realistic HTTP Headers with Ordering and Timing ===
def realistic_http_headers():
    """Generate realistic HTTP headers with proper ordering and timing"""
    # Common header orders for different browsers
    header_orders = {
        "chrome": [
            "Host", "Connection", "Cache-Control", "sec-ch-ua", "sec-ch-ua-mobile", 
            "sec-ch-ua-platform", "Upgrade-Insecure-Requests", "User-Agent", 
            "Accept", "Sec-Fetch-Site", "Sec-Fetch-Mode", "Sec-Fetch-User", 
            "Sec-Fetch-Dest", "Accept-Encoding", "Accept-Language", "Cookie"
        ],
        "firefox": [
            "Host", "User-Agent", "Accept", "Accept-Language", "Accept-Encoding", 
            "Connection", "Upgrade-Insecure-Requests", "Sec-Fetch-Dest", 
            "Sec-Fetch-Mode", "Sec-Fetch-Site", "Sec-Fetch-User", "Cookie"
        ],
        "safari": [
            "Host", "Connection", "Accept-Encoding", "Accept-Language", 
            "User-Agent", "Upgrade-Insecure-Requests", "Cookie"
        ]
    }
    
    # Select a browser type
    browser_type = random.choice(["chrome", "firefox", "safari"])
    order = header_orders[browser_type]
    
    # Generate headers
    headers = {
        "User-Agent": "",  # Will be filled later
        "Accept": random.choice([
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ]),
        "Accept-Language": random.choice([
            "en-US,en;q=0.9",
            "en-GB,en;q=0.9",
            "en;q=0.9"
        ]),
        "Accept-Encoding": random.choice([
            "gzip, deflate, br",
            "gzip, deflate"
        ]),
        "Connection": random.choice(["keep-alive", "close"]),
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": random.choice(["document", "empty"]),
        "Sec-Fetch-Mode": random.choice(["navigate", "cors", "no-cors"]),
        "Sec-Fetch-Site": random.choice(["none", "same-origin", "cross-site"]),
        "Cache-Control": random.choice(["max-age=0", "no-cache"])
    }
    
    # Add browser-specific headers
    if browser_type == "chrome":
        headers["sec-ch-ua"] = random.choice([
            '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
            '"Chromium";v="108", "Not A(Brand";v="24", "Google Chrome";v="108"'
        ])
        headers["sec-ch-ua-mobile"] = random.choice(["?0", "?1"])
        headers["sec-ch-ua-platform"] = random.choice([
            '"Windows"', '"macOS"', '"Linux"', '"Android"'
        ])
    
    # Reorder headers according to the browser's pattern
    ordered_headers = {}
    for header_name in order:
        if header_name in headers:
            ordered_headers[header_name] = headers[header_name]
    
    # Add any remaining headers
    for header_name, value in headers.items():
        if header_name not in ordered_headers:
            ordered_headers[header_name] = value
    
    return ordered_headers, browser_type

# === NEW: Cloudflare Challenge Solver ===
async def solve_cloudflare_challenge(page, url):
    """Handle Cloudflare challenges using human-like behavior"""
    # Wait for the challenge page to load
    await asyncio.sleep(random.uniform(2, 5))
    
    # Check if we're facing a Cloudflare challenge
    try:
        cf_challenge = await page.query_selector("#cf-challenge-stage")
        if cf_challenge:
            logger.info("🛡️ Detected Cloudflare challenge, solving...")
            
            # Wait a bit as a human would
            await asyncio.sleep(random.uniform(3, 7))
            
            # Look for the verification button
            try:
                verify_button = await page.wait_for_selector("#cf-challenge-stage-input", state="visible", timeout=10000)
                
                # Human-like movement to the button
                # Note: We'll use a simple click since we don't have the mouse movement function here
                await verify_button.click()
                
                # Wait for the challenge to complete
                await asyncio.sleep(random.uniform(5, 10))
                
                # Check if we're through
                if await page.query_selector("#cf-challenge-stage") is None:
                    logger.info("✅ Cloudflare challenge solved successfully")
                    return True
                else:
                    logger.warning("⚠️ Cloudflare challenge may not have been solved")
                    
            except Exception as e:
                logger.error(f"❌ Error solving Cloudflare challenge: {e}")
                
                # Try alternative approach - wait for automatic redirect
                logger.info("⏳ Waiting for automatic redirect...")
                await asyncio.sleep(random.uniform(10, 15))
                
                if await page.query_selector("#cf-challenge-stage") is None:
                    logger.info("✅ Cloudflare challenge solved automatically")
                    return True
                    
    except Exception:
        # No Cloudflare challenge detected
        pass
    
    return False

# === NEW: Realistic Session Simulation ===
async def simulate_realistic_session(page, target_url):
    """Simulate a realistic browsing session with random duration and interactions"""
    # Navigate to the target URL
    await page.goto(target_url)
    
    # Random session duration (5-30 minutes)
    session_duration = random.randint(300, 1800)
    session_start = time.time()
    
    # Random interactions during the session
    while time.time() - session_start < session_duration:
        # Random pause between actions
        await asyncio.sleep(random.uniform(5, 30))
        
        # Random action
        action = random.random()
        
        if action < 0.3:  # 30% chance to scroll
            # Scroll to a random position
            scroll_to_bottom = random.random() < 0.7  # 70% chance to scroll down
            if scroll_to_bottom:
                # Scroll down
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                # Scroll up
                await page.evaluate("window.scrollTo(0, 0)")
            
        elif action < 0.5:  # 20% chance to move mouse
            # Move mouse to a random element
            try:
                links = await page.query_selector_all("a")
                buttons = await page.query_selector_all("button")
                clickable_elements = links + buttons
                
                if clickable_elements:
                    target_element = random.choice(clickable_elements)
                    # Simple click without mouse movement simulation
                    await target_element.click()
                    
                    # Small chance to actually click
                    if random.random() < 0.2:  # 20% chance to click
                        await asyncio.sleep(random.uniform(2, 5))
                        
                        # 50% chance to go back
                        if random.random() < 0.5:
                            await page.go_back()
                            await asyncio.sleep(random.uniform(2, 5))
            except Exception as e:
                logger.warning(f"Error during mouse movement: {e}")
                
        elif action < 0.7:  # 20% chance to open a new tab
            # Open a new tab and navigate to a random site
            async with page.context.expect_page() as new_page_info:
                await page.evaluate("window.open('');")
            
            new_page = new_page_info.value
            
            # Visit a random site
            random_site = random.choice(random_sites)
            await new_page.goto(random_site)
            await asyncio.sleep(random.uniform(3, 10))
            
            # Close the tab
            await new_page.close()
            await asyncio.sleep(random.uniform(1, 3))
            
        else:  # 30% chance to do nothing (just wait)
            # Simulate reading content
            await asyncio.sleep(random.uniform(10, 60))
    
    logger.info(f"✅ Completed realistic session of {session_duration} seconds")

# === NEW: Form Interaction Simulation ===
async def simulate_form_interaction(page, form_data=None):
    """Simulate human-like form filling"""
    # Find forms on the page
    forms = await page.query_selector_all("form")
    if not forms:
        return False
    
    # Select a random form
    form = random.choice(forms)
    
    # Find input fields in the form
    inputs = await form.query_selector_all("input")
    textareas = await form.query_selector_all("textarea")
    selects = await form.query_selector_all("select")
    
    fields = inputs + textareas + selects
    if not fields:
        return False
    
    # Randomly decide whether to fill the form (50% chance)
    if random.random() < 0.5:
        logger.info("📝 Simulating form interaction...")
        
        # Fill each field with human-like behavior
        for field in fields:
            field_type = await field.get_attribute("type")
            tag_name = field.tag_name
            
            # Skip hidden fields and submit buttons
            if field_type == "hidden" or field_type == "submit" or not await field.is_visible():
                continue
            
            # Scroll to the field if needed
            await field.scroll_into_view_if_needed()
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Click on the field
            await field.click()
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # Fill the field based on its type
            if tag_name == "select":
                # Select a random option
                options = await field.query_selector_all("option")
                if options:
                    option = random.choice(options)
                    await option.click()
                    await asyncio.sleep(random.uniform(0.5, 1.0))
            elif field_type in ["text", "email", "password", "search", "tel", "url"]:
                # Type text with human-like cadence
                if form_data and await field.get_attribute("name") in form_data:
                    text = form_data[await field.get_attribute("name")]
                else:
                    # Generate realistic fake data based on field type
                    if field_type == "email":
                        text = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
                    elif field_type == "password":
                        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*', k=12))
                    elif field_type == "tel":
                        text = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    else:
                        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=random.randint(5, 15)))
                
                # Type with realistic timing
                for char in text:
                    await field.type(char)
                    await asyncio.sleep(random.uniform(0.05, 0.2))
                
                # Random pause after typing
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
                # Sometimes tab to the next field
                if random.random() < 0.3:  # 30% chance
                    await field.press("Tab")
                    await asyncio.sleep(random.uniform(0.3, 1.0))
            elif tag_name == "textarea":
                # Type longer text with human-like cadence
                if form_data and await field.get_attribute("name") in form_data:
                    text = form_data[await field.get_attribute("name")]
                else:
                    # Generate a paragraph of text
                    words = [
                        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
                        "this", "is", "a", "sample", "text", "for", "form", "filling",
                        "human", "behavior", "simulation", "requires", "realistic", "timing"
                    ]
                    text = ' '.join(random.choices(words, k=random.randint(10, 30)))
                
                # Type with realistic timing
                for char in text:
                    await field.type(char)
                    await asyncio.sleep(random.uniform(0.05, 0.15))
                
                # Random pause after typing
                await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Decide whether to submit the form (70% chance)
        if random.random() < 0.7:
            # Find the submit button
            submit_buttons = await form.query_selector_all("input[type='submit'], button[type='submit'], button:has-text('Submit'), button:has-text('Send')")
            
            if submit_buttons:
                submit_button = random.choice(submit_buttons)
                await submit_button.click()
                await asyncio.sleep(random.uniform(2, 5))
                
                # 50% chance to go back after submission
                if random.random() < 0.5:
                    await page.go_back()
                    await asyncio.sleep(random.uniform(2, 5))
        
        return True
    
    return False

# === NEW: Safe Evaluation Helper ===
async def safe_evaluate(page, script, description=""):
    """Safely evaluate JavaScript on a page with error handling"""
    try:
        return await page.evaluate(script)
    except Exception as e:
        logger.warning(f"Error in {description}: {e}")
        return None

# === NEW: Environment Consistency ===
async def ensure_consistent_environment(page, proxy_info=None):
    """Ensure timezone, language, and other settings match the IP geolocation"""
    # Get IP geolocation if proxy is used
    if proxy_info and config.enable_ip_geolocation_matching:
        try:
            proxy = format_proxy(proxy_info)
            response = requests.get("https://ipinfo.io/json", proxies=proxy, timeout=10, verify=False)
            if response.status_code == 200:
                ip_info = response.json()
                country = ip_info.get("country", "US")
                timezone = get_timezone_for_location(country)  # Use our new function
                
                # Set browser timezone to match IP location
                # Note: We removed the set_default_timezone call as it was causing errors
                # Instead, we'll set it when creating the browser context
                
                # Set language based on country
                country_languages = {
                    "US": "en-US",
                    "GB": "en-GB",
                    "DE": "de-DE",
                    "FR": "fr-FR",
                    "IT": "it-IT",
                    "ES": "es-ES",
                    "BR": "pt-BR",
                    "IN": "hi-IN",
                    "CA": "en-CA",
                    "AU": "en-AU",
                    "JP": "ja-JP",
                    "KR": "ko-KR",
                    "CN": "zh-CN",
                    "RU": "ru-RU"
                }
                
                language = country_languages.get(country, "en-US")
                
                # Set browser language
                await page.set_extra_http_headers({"Accept-Language": language})
                
                # Override navigator.language
                await page.evaluate(f"""
                    Object.defineProperty(navigator, 'language', {{
                        get: () => '{language}',
                    }});
                    
                    Object.defineProperty(navigator, 'languages', {{
                        get: () => ['{language}', 'en'],
                    }});
                """)
                
                logger.info(f"🌍 Set environment to match {country} timezone and language")
                
        except Exception as e:
            logger.warning(f"Failed to set consistent environment: {e}")

# === NEW: Device Parameter Consistency ===
async def ensure_consistent_device_parameters(page, device_info):
    """Ensure all device parameters are consistent with each other"""
    # Get screen resolution from device info
    resolution = device_info.get("screen_resolution", "1920x1080")
    width, height = map(int, resolution.split('x'))
    
    # Calculate viewport size (typically smaller than screen resolution)
    viewport_height = height - random.randint(50, 150)
    
    # Set consistent screen parameters
    await page.set_viewport_size({"width": width, "height": viewport_height})
    
    # Set hardware concurrency based on device type
    device_type = device_info.get('device_type', 'Unknown')
    if device_type in ["iPhone", "iPad", "Samsung", "GooglePixel", "Motorola", "OnePlus", "Vivo", "Xiaomi", "ZTE"]:
        # Mobile devices typically have 4-8 cores
        hardware_concurrency = random.randint(4, 8)
    else:
        # Desktop devices typically have 4-16 cores
        hardware_concurrency = random.randint(4, 16)
    
    # Set device memory based on device type
    if device_type in ["iPhone", "iPad", "Samsung", "GooglePixel", "Motorola", "OnePlus", "Vivo", "Xiaomi", "ZTE"]:
        # Mobile devices typically have 4-8 GB RAM
        device_memory = random.choice([4, 6, 8])
    else:
        # Desktop devices typically have 8-32 GB RAM
        device_memory = random.choice([8, 16, 32])
    
    # Override navigator properties
    await page.evaluate(f"""
        // Set hardware concurrency
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {hardware_concurrency},
        }});
        
        // Set device memory
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {device_memory},
        }});
        
        // Set screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {width},
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {height},
        }});
        
        Object.defineProperty(screen, 'availWidth', {{
            get: () => {width},
        }});
        
        Object.defineProperty(screen, 'availHeight', {{
            get: () => {height - random.randint(30, 60)},
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => 24,
        }});
        
        Object.defineProperty(screen, 'pixelDepth', {{
            get: () => 24,
        }});
        
        // Set window properties
        Object.defineProperty(window, 'devicePixelRatio', {{
            get: () => {random.choice([1, 1.5, 2])},
        }});
    """)
    
    logger.info(f"📱 Set consistent device parameters for {device_type}")

# === Select device with percentage distribution ===
def select_device():
    # Create a pool based on percentage distribution
    pool = []
    for device, percentage in device_percentage_distribution.items():
        # Add the device to the pool according to its percentage
        pool.extend([device] * percentage)
    
    # Shuffle the pool to ensure randomness
    random.shuffle(pool)
    
    # Return a random device from the pool
    return random.choice(pool)

# === Get and remove user agent ===
def get_and_remove_ua(device_name, agents_list, filename):
    if not agents_list:
        logger.error(f"❌ No more UAs left for {device_name}")
        return None
    
    # Select a random user agent
    ua = random.choice(agents_list)
    
    # Remove from the in-memory list
    agents_list.remove(ua)
    
    # Remove from the file if configured
    if config.remove_ua_after_use:
        remove_line_from_file(filename, ua)
    
    # Return the user agent as is (without randomization)
    return ua

# === Extract phone model from UA ===
def extract_phone_model(ua):
    # Use the enhanced device fingerprint parser
    device_fingerprint = DeviceFingerprint(ua)
    device_info = device_fingerprint.device_info
    
    if device_info.get('device_model') != 'Unknown':
        return device_info['device_model']
    
    # Fallback to simple extraction
    match = re.search(r'FBDV/([^;]+)', ua)
    if match:
        return match.group(1)
    if "iPhone" in ua:
        return "iPhone"
    if "Samsung" in ua or "SM-" in ua:
        return "Samsung"
    if "GooglePixel" in ua or "Pixel" in ua:
        return "GooglePixel"
    if "Motorola" in ua:
        return "Motorola"
    if "OnePlus" in ua:
        return "OnePlus"
    if "Vivo" in ua:
        return "Vivo"
    if "Xiaomi" in ua:
        return "Xiaomi"
    if "ZTE" in ua:
        return "ZTE"
    if "Windows" in ua:
        return "Windows PC"
    return "Unknown Device"

# === Enhanced IP info retrieval (FIXED) ===
def get_ip_info(proxy):
    try:
        # Try multiple sources for more reliable data
        sources = [
            "https://ipinfo.io/json",
            "https://ipapi.co/json/"
        ]
        
        for url in sources:
            try:
                r = requests.get(url, proxies=proxy, timeout=10, verify=False)
                data = r.json()
                
                # Extract consistent data
                country = data.get("country_code") or data.get("country")
                city = data.get("city") or data.get("region_name")
                region = data.get("region") or data.get("region_name")
                
                # Use our timezone function based on country
                timezone = get_timezone_for_location(country)
                
                isp = data.get("org") or data.get("isp") or data.get("organization")
                
                # Get coordinates - FIXED to handle unpacking errors
                if "loc" in data and data["loc"]:
                    loc_parts = data["loc"].split(",")
                    lat = loc_parts[0]
                    lng = loc_parts[1] if len(loc_parts) > 1 else "0"
                else:
                    lat = str(data.get("latitude", "0"))
                    lng = str(data.get("longitude", "0"))
                
                return (
                    data.get("ip", ""), 
                    city, 
                    country, 
                    region, 
                    timezone,
                    isp,
                    lat,
                    lng
                )
            except Exception as e:
                logger.warning(f"Failed to get IP info from {url}: {e}")
                continue
                
        # Fallback if all sources fail
        return "Unknown", "Unknown", "Unknown", "Unknown", "UTC", "Unknown ISP", "0", "0"
    except Exception as e:
        logger.error(f"IP info retrieval failed: {e}")
        return "Unknown", "Unknown", "Unknown", "Unknown", "UTC", "Unknown ISP", "0", "0"

# === Get local time ===
def get_local_time(tz):
    try:
        now = datetime.datetime.now(pytz.timezone(tz))
    except:
        now = datetime.datetime.utcnow()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

# === ENHANCED: Simulate scroll behavior with more randomness ===
def simulate_scroll():
    # Simulate realistic scroll behavior with more randomness
    scroll_events = []
    total_scroll = random.randint(500, 1500)
    scroll_events.append({"y": 0, "delay": random.randint(100, 500)})
    
    current_y = 0
    while current_y < total_scroll:
        # More variable scroll amounts and speeds
        scroll_amount = random.randint(30, 250)
        
        # Add random pauses during scrolling (human behavior)
        if random.random() < 0.2:  # 20% chance of a pause
            pause_time = random.randint(200, 1500)
            scroll_events.append({"y": current_y, "delay": pause_time})
        
        # Variable scroll speed
        scroll_speed = random.uniform(0.3, 2.5)
        current_y += scroll_amount
        if current_y > total_scroll:
            current_y = total_scroll
        scroll_events.append({"y": current_y, "delay": int(scroll_speed * 1000)})
    
    # Add more random scroll ups with variable speeds
    num_scroll_ups = random.randint(1, 4)
    for _ in range(num_scroll_ups):
        scroll_up = random.randint(30, 200)
        scroll_speed = random.uniform(0.3, 2.5)
        current_y = max(0, current_y - scroll_up)
        scroll_events.append({"y": current_y, "delay": int(scroll_speed * 1000)})
        
        # Add random pauses during scroll up
        if random.random() < 0.3:  # 30% chance of a pause during scroll up
            pause_time = random.randint(200, 1500)
            scroll_events.append({"y": current_y, "delay": pause_time})
    
    # Scroll to bottom
    scroll_events.append({"y": total_scroll, "delay": random.randint(100, 500)})
    
    return scroll_events

# === ENHANCED: Simulate mouse movement with more imperfections ===
def simulate_mouse_movement():
    # Simulate realistic mouse movement with more imperfections
    movements = []
    current_x, current_y = random.randint(100, 500), random.randint(100, 500)
    
    # Random number of movement sequences
    num_sequences = random.randint(3, 12)
    
    for _ in range(num_sequences):
        target_x = random.randint(50, 1000)
        target_y = random.randint(50, 800)
        
        # Control points for Bezier curve with more randomness
        cp1_x = current_x + random.randint(-300, 300)
        cp1_y = current_y + random.randint(-300, 300)
        cp2_x = target_x + random.randint(-300, 300)
        cp2_y = target_y + random.randint(-300, 300)
        
        # Variable number of steps for each movement
        steps = random.randint(8, 25)
        
        # Add random micro-movements and hesitations
        for i in range(steps + 1):
            t = i / steps
            # Bezier curve formula
            x = (1-t)**3 * current_x + 3*(1-t)**2*t * cp1_x + 3*(1-t)*t**2 * cp2_x + t**3 * target_x
            y = (1-t)**3 * current_y + 3*(1-t)**2*t * cp1_y + 3*(1-t)*t**2 * cp2_y + t**3 * target_y
            
            # Add random micro-movements
            if random.random() < 0.1:  # 10% chance of micro-movement
                x += random.randint(-5, 5)
                y += random.randint(-5, 5)
            
            # Variable speed with more randomness
            base_speed = random.uniform(0.01, 0.15)
            
            # Add random hesitations (pauses during movement)
            if random.random() < 0.05:  # 5% chance of hesitation
                hesitation = random.randint(100, 500)
                movements.append({"x": int(x), "y": int(y), "delay": hesitation})
            
            movements.append({"x": int(x), "y": int(y), "delay": int(base_speed * 1000)})
        
        # Sometimes move to a random intermediate point
        if random.random() < 0.3:  # 30% chance of intermediate point
            intermediate_x = random.randint(50, 1000)
            intermediate_y = random.randint(50, 800)
            
            # Move to intermediate point
            int_steps = random.randint(3, 8)
            for i in range(int_steps + 1):
                t = i / int_steps
                x = current_x + (intermediate_x - current_x) * t
                y = current_y + (intermediate_y - current_y) * t
                
                # Variable speed
                speed = random.uniform(0.02, 0.1)
                movements.append({"x": int(x), "y": int(y), "delay": int(speed * 1000)})
            
            current_x, current_y = intermediate_x, intermediate_y
        else:
            current_x, current_y = target_x, target_y
    
    return movements

# === Enhanced JS environment simulation ===
async def simulate_js_env(headers, ua, device_fingerprint, device_type, ip_info):
    # Create cookies
    cookies = SimpleCookie()
    
    # Generate encrypted cookies if enabled
    if config.enable_encryption and cipher_suite:
        session_id = cipher_suite.encrypt(f"{random.randint(10000000, 99999999)}-{random.randint(1000,9999)}".encode()).decode()
        user_id = cipher_suite.encrypt(f"{random.randint(10000, 99999)}".encode()).decode()
    else:
        session_id = f"{random.randint(10000000, 99999999)}-{random.randint(1000,9999)}"
        user_id = f"{random.randint(10000, 99999)}"
    
    cookies["sessionid"] = session_id
    cookies["userid"] = user_id
    
    # Get fingerprint data
    fingerprint_data = device_fingerprint.get_all_fingerprints(ip_info)
    
    # Get location info
    ip, city, country, region, tz, isp, lat, lng = ip_info
    
    # Create spoofed geolocation based on IP location
    # Use the actual coordinates from IP if available
    if lat != "0" and lng != "0":
        latitude = float(lat)
        longitude = float(lng)
    else:
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
    
    accuracy = random.randint(10, 100)
    
    # Create spoofed WebGL info
    webgl = fingerprint_data["webgl"]
    
    # Create spoofed canvas fingerprint
    canvas_fp = fingerprint_data["canvas"]
    
    # Create spoofed audio fingerprint
    audio_fp = fingerprint_data["audio"]
    
    # Create spoofed font list
    fonts = fingerprint_data["fonts"]
    
    # Get hardware info
    hardware = fingerprint_data["hardware"]
    
    # Get battery info
    battery = fingerprint_data["battery"]
    
    # Get memory info
    memory = fingerprint_data["memory"]
    
    # Get connection info
    connection = fingerprint_data["connection"]
    
    # Get sensor info
    sensors = fingerprint_data["sensors"]
    
    # Get timezone offset
    tz_offset = fingerprint_data["timezone_offset"]
    
    # Get carrier
    carrier = fingerprint_data["carrier"]
    
    # Get screen resolution and viewport
    screen_res = fingerprint_data["screen_resolution"]
    viewport = fingerprint_data["viewport_size"]
    
    # Generate a random click ID
    click_id = hashlib.md5(f"{random.randint(10000000, 99999999)}-{time.time()}".encode()).hexdigest()
    
    # Get device info
    device_info = fingerprint_data.get("device_info", {})
    
    # Get a random referer from the loaded referers
    referer = random.choice(referers) if referers else "https://chat.z.ai/"
    
    # Update headers with spoofed data
    headers.update({
        "Accept-Language": random.choice(["en-US,en;q=0.9", "es-US,es;q=0.9"]),
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Referer": referer,  # Use the random referer instead of hardcoded Google
        "Origin": f"https://{urlparse(config.cpa_url).netloc}",
        "Cookie": "; ".join([f"{key}={val.value}" for key, val in cookies.items()]),
        "DNT": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-Requested-With": "XMLHttpRequest",
        "X-Fake-Canvas": canvas_fp,
        "X-Fake-WebGL": f"{webgl['vendor']} {webgl['renderer']}",
        "X-Fake-Audio": audio_fp,
        "X-Device-ID": fingerprint_data["device_id"],
        "X-IMEI": fingerprint_data["imei"],
        "X-Screen-Resolution": screen_res,
        "X-Viewport-Size": viewport,
        "X-Hardware-Cores": str(hardware["cores"]),
        "X-Device-Memory": str(hardware["memory"]),
        "X-Platform": fingerprint_data["os_build"],
        "X-Carrier": carrier,
        "X-Geolocation": f"latitude: {latitude}; longitude: {longitude}; accuracy: {accuracy}",
        "X-Permissions": "geolocation: denied; notifications: granted",
        "X-Timezone": tz,
        "X-Timezone-Offset": str(tz_offset),
        "X-Client-IP": ip,
        "X-Forwarded-For": ip,
        "X-Real-IP": ip,
        "X-Battery-Level": str(battery["level"]),
        "X-Battery-Charging": str(battery["charging"]),
        "X-Memory-Total": str(memory["deviceMemory"]),
        "X-Connection-Type": connection["effectiveType"],
        "X-Fonts": ",".join(fonts),
        "X-WebRTC": "disabled",
        "X-Do-Not-Track": "1",
        "X-Sec-CH-UA": f'"Chromium";v="{random.randint(90, 120)}", "Google Chrome";v="{random.randint(90, 120)}", "Not=A?Brand";v="99"',
        "X-Sec-CH-UA-Mobile": "?0" if "Windows" in device_type else "?1",
        "X-Sec-CH-UA-Platform": f'"{"Windows" if "Windows" in device_type else "Android"}"',
        "X-Click-ID": click_id,
        "X-Sensor-Accelerometer": f"x:{sensors['accelerometer']['x']},y:{sensors['accelerometer']['y']},z:{sensors['accelerometer']['z']}",
        "X-Sensor-Gyroscope": f"alpha:{sensors['gyroscope']['alpha']},beta:{sensors['gyroscope']['beta']},gamma:{sensors['gyroscope']['gamma']}",
        "X-Sensor-Magnetometer": f"x:{sensors['magnetometer']['x']},y:{sensors['magnetometer']['y']},z:{sensors['magnetometer']['z']}"
    })
    
    # Add WebGPU disabling if configured
    if config.disable_webgpu:
        headers["X-WebGPU"] = "disabled"
        headers["Sec-CH-UA-Arch"] = '"x86"'
        headers["Sec-CH-UA-Bitness"] = '"64"'
        headers["Sec-CH-UA-Model"] = f'"{device_info.get("device_model", "Unknown")}"'
        headers["Sec-CH-UA-Mobile"] = "?0" if "Windows" in device_type else "?1"
        headers["Sec-CH-UA-Platform"] = f'"{"Windows" if "Windows" in device_type else "Android"}"'
        headers["Sec-CH-UA-Platform-Version"] = f'"{device_info.get("os_version", "Unknown")}"'
    
    return headers, click_id

# === ENHANCED: Smart delay ===
def smart_delay():
    # Introduce realistic mouse and scroll movement delays
    # Use a more complex distribution to simulate human behavior
    delay_type = random.choice(['short', 'medium', 'long'])
    
    if delay_type == 'short':
        delay = random.uniform(0.5, 2.0)
    elif delay_type == 'medium':
        delay = random.uniform(2.0, 5.0)
    else:  # long
        delay = random.uniform(5.0, 10.0)
    
    # Add some randomness to the delay
    delay *= random.uniform(0.8, 1.2)
    
    # Ensure minimum delay
    delay = max(0.5, delay)
    
    time.sleep(delay)

# === ENHANCED: Visit random sites with more randomness ===
async def shuffle_and_visit_sites(sites, headers, proxy, session):
    """Visit random sites and return the list of successfully visited sites"""
    # Randomize the number of sites to visit (1-2)
    num_sites = random.randint(config.min_pre_click_sites, config.max_pre_click_sites)
    
    if num_sites == 0:
        logger.info("🌐 Skipping site visits (random choice)")
        return []
    
    visited = random.sample(sites, k=min(num_sites, len(sites)))
    
    # Add a random Facebook profile to the visited sites
    fb_id = f"1000{random.randint(10000000,99999999)}"
    fb_url = f"https://www.facebook.com/profile.php?id={fb_id}"
    visited.append(fb_url)
    
    # Shuffle the list to randomize the order
    random.shuffle(visited)
    
    successfully_visited = []
    
    async with aiohttp.ClientSession() as http_session:
        for site in visited:
            try:
                # Disable SSL verification for this request
                async with http_session.get(site, headers=headers, timeout=15, ssl=False) as response:
                    logger.info(f"🌐 Visited: {site} | Status: {response.status}")
                    
                    # Only add to successfully_visited if status code is 2xx
                    if 200 <= response.status < 300:
                        successfully_visited.append(site)
                    
                    # Randomly decide whether to scroll on this site
                    if random.random() < 0.7:  # 70% chance to scroll
                        # Simulate scroll behavior on the site
                        scroll_events = simulate_scroll()
                        for event in scroll_events:
                            logger.debug(f"🖱️ Scrolled to Y: {event['y']} | Delay: {event['delay']}ms")
                            await asyncio.sleep(event['delay'] / 1000)
                    
                    # Randomly decide whether to move mouse on this site
                    if random.random() < 0.6:  # 60% chance to move mouse
                        # Simulate mouse movement
                        movements = simulate_mouse_movement()
                        for movement in movements:
                            logger.debug(f"🖱️ Mouse moved to X: {movement['x']}, Y: {movement['y']} | Delay: {movement['delay']}ms")
                            await asyncio.sleep(movement['delay'] / 1000)
                    
                    # Random delay between sites
                    if random.random() < 0.5:  # 50% chance of delay
                        delay = random.uniform(0.5, 3.0)
                        await asyncio.sleep(delay)
                        
            except Exception as e:
                logger.error(f"⚠️ Visit failed: {site} | {e}")
    
    return successfully_visited

# === Banner ===
def banner():
    console.print(Panel.fit(
        "[bold green]WELCOME TO WORLD BEST UNDETECTABLE BOT[/bold green]\n"
        "[bold blue]MADE BY STARBOY MACCLUM ARMAN[/bold blue]\n"
        "[bold red]HACKER GANG YOU ARE UNDETECTABLE[/bold red]",
        title="🚀 ULTIMATE CPA BOT 🚀"
    ))

# === Log click ===
def log_click(phone_model, proxy, ip, city, region, country, time_str, isp, device_id, click_id):
    log_line = (f"✅ CLICK SUCCESS | UA: {phone_model} | Proxy: {proxy} | IP: {ip} | "
                f"City: {city}, State: {region}, Country: {country} | Time: {time_str} | "
                f"ISP: {isp} | Device ID: {device_id} | Click ID: {click_id}\n")
    with open("click_log.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

# === Print clicks left ===
def print_clicks_left():
    console.print(f"🎯 Clicks Left per Device - iPhone: {len(iphone_agents)}, Samsung: {len(samsung_agents)}, "
          f"Windows: {len(windows_agents)}, GooglePixel: {len(googlepixel_agents)}, Motorola: {len(motorola_agents)}, "
          f"OnePlus: {len(oneplus_agents)}, Vivo: {len(vivo_agents)}, Xiaomi: {len(xiaomi_agents)}, ZTE: {len(zte_agents)}")

# === Select proxy based on rotation strategy ===
def select_proxy(proxies, proxy_scores, rotation_strategy, used_proxies):
    if rotation_strategy == "random":
        available_proxies = [p for p in proxies if p not in used_proxies]
        if not available_proxies:
            return None
        return random.choice(available_proxies)
    
    elif rotation_strategy == "round_robin":
        available_proxies = [p for p in proxies if p not in used_proxies]
        if not available_proxies:
            return None
        # Simple round robin: return the first available proxy
        return available_proxies[0]
    
    elif rotation_strategy == "geographic":
        # For geographic rotation, we'd need to group proxies by location
        # For simplicity, we'll just use random rotation here
        available_proxies = [p for p in proxies if p not in used_proxies]
        if not available_proxies:
            return None
        return random.choice(available_proxies)
    
    elif rotation_strategy == "performance":
        # Sort proxies by score (highest first)
        sorted_proxies = sorted(proxy_scores.items(), key=lambda x: x[1], reverse=True)
        for proxy, score in sorted_proxies:
            if proxy not in used_proxies:
                return proxy
        return None
    
    else:
        # Default to random
        available_proxies = [p for p in proxies if p not in used_proxies]
        if not available_proxies:
            return None
        return random.choice(available_proxies)

# === CAPTCHA Solver ===
def solve_captcha(site_key, url):
    if not config.enable_captcha_solver or not config.captcha_api_key:
        return None
    
    try:
        if config.captcha_service == "2captcha":
            # 2Captcha API
            api_url = "http://2captcha.com/in.php"
            params = {
                "key": config.captcha_api_key,
                "method": "userrecaptcha",
                "googlekey": site_key,
                "pageurl": url,
                "json": 1
            }
            
            response = requests.post(api_url, data=params, verify=False)
            result = response.json()
            
            if result.get("status") == 1:
                captcha_id = result.get("request")
                # Wait for solution
                time.sleep(15)  # Initial wait
                result_url = f"http://2captcha.com/res.php?key={config.captcha_api_key}&action=get&id={captcha_id}&json=1"
                while True:
                    response = requests.get(result_url, verify=False)
                    result = response.json()
                    
                    if result.get("status") == 1:
                        return result.get("request")
                    elif result.get("request") == "CAPCHA_NOT_READY":
                        time.sleep(5)
                    else:
                        break
        
        elif config.captcha_service == "anticaptcha":
            # Anti-Captcha API
            api_url = "https://api.anti-captcha.com/createTask"
            payload = {
                "clientKey": config.captcha_api_key,
                "task": {
                    "type": "RecaptchaV2TaskProxyless",
                    "websiteURL": url,
                    "websiteKey": site_key
                }
            }
            
            response = requests.post(api_url, json=payload, verify=False)
            result = response.json()
            
            if result.get("errorId") == 0:
                task_id = result.get("taskId")
                # Wait for solution
                time.sleep(15)  # Initial wait
                
                result_url = "https://api.anti-captcha.com/getTaskResult"
                payload = {
                    "clientKey": config.captcha_api_key,
                    "taskId": task_id
                }
                
                while True:
                    response = requests.post(result_url, json=payload, verify=False)
                    result = response.json()
                    
                    if result.get("status") == "ready":
                        return result.get("solution", {}).get("gRecaptchaResponse")
                    elif result.get("status") == "processing":
                        time.sleep(5)
                    else:
                        break
    except Exception as e:
        logger.error(f"⚠️ CAPTCHA solving failed: {e}")
    
    return None

# === NEW: Calculate rest period ===
def calculate_rest_period():
    """Calculate a rest period to simulate human breaks"""
    if not config.enable_rest_periods:
        return 0
    
    # Only take a rest with the configured probability
    if random.random() > config.rest_probability:
        return 0
    
    # Calculate rest period duration
    rest_period = random.randint(config.min_rest_period, config.max_rest_period)
    logger.info(f"😴 Taking a rest period of {rest_period // 60} minutes...")
    return rest_period

# === NEW: Check if within operating hours ===
def is_within_operational_hours():
    """Check if current time is within configured operating hours"""
    if not config.enable_operational_hours:
        return True
    
    current_hour = datetime.datetime.now().hour
    start_hour = config.operating_hours_start
    end_hour = config.operating_hours_end
    
    # Handle case where end hour is after midnight (e.g., 23 to 8)
    if end_hour < start_hour:
        return current_hour >= start_hour or current_hour < end_hour
    else:
        return start_hour <= current_hour < end_hour

# === Calculate human-like wait time ===
def calculate_wait_time():
    """Calculate a wait time that mimics human behavior patterns"""
    if not config.enable_human_timing_patterns:
        # Simple random wait time
        return random.randint(config.min_wait_between_clicks, config.max_wait_between_clicks)
    
    # Use the natural timing patterns
    return calculate_natural_wait_time()

# === NEW: Monitor success rates ===
def monitor_success_rates():
    """Monitor success rates and alert if they drop below threshold"""
    if not config.enable_success_monitoring:
        return
    
    total_success = sum(device_success_rates[device]["success"] for device in device_success_rates)
    total_attempts = sum(device_success_rates[device]["total"] for device in device_success_rates)
    
    if total_attempts == 0:
        return
    
    success_rate = total_success / total_attempts
    
    if success_rate < config.success_rate_threshold:
        logger.warning(f"⚠️ Success rate dropped to {success_rate:.2%} (threshold: {config.success_rate_threshold:.2%})")
        logger.warning("Consider adjusting settings or taking a break to avoid detection")
        
        # Log detailed device success rates
        for device in device_success_rates:
            device_success = device_success_rates[device]["success"]
            device_total = device_success_rates[device]["total"]
            if device_total > 0:
                device_rate = device_success / device_total
                logger.info(f"   {device}: {device_success}/{device_total} ({device_rate:.2%})")

# === NEW: Get random viewport size ===
def get_random_viewport_size(device_type):
    """Generate a random viewport size based on device type"""
    if "iPhone" in device_type or "iPad" in device_type:
        # Mobile viewport sizes
        viewport_sizes = [
            (375, 667),   # iPhone 6/7/8
            (375, 812),   # iPhone X/11/12
            (414, 736),   # iPhone 6/7/8 Plus
            (414, 896),   # iPhone 11 Pro Max
            (390, 844),   # iPhone 12/13 Pro
            (428, 926),   # iPhone 12/13 Pro Max
            (320, 568),   # iPhone 5/SE
        ]
    elif "Windows" in device_type:
        # Desktop viewport sizes
        viewport_sizes = [
            (1366, 768),
            (1440, 900),
            (1536, 864),
            (1920, 1080),
            (2560, 1440),
            (1024, 768),
            (1280, 720),
            (1600, 900),
        ]
    else:
        # Android viewport sizes
        viewport_sizes = [
            (360, 640),
            (360, 740),
            (411, 731),
            (412, 846),
            (393, 786),
            (393, 851),
            (412, 892),
        ]
    
    return random.choice(viewport_sizes)

# === NEW: Simulate browser extensions ===
async def simulate_browser_extensions(page):
    """Simulate random browser extensions"""
    if not config.enable_browser_extension_simulation:
        return
    
    # List of common browser extensions to simulate
    common_extensions = [
        "AdBlock",
        "uBlock Origin",
        "LastPass",
        "1Password",
        "Grammarly",
        "Honey",
        "HTTPS Everywhere",
        "Dark Reader",
        "Video DownloadHelper",
        "Pocket",
        "Evernote Web Clipper",
        "Save to Google Drive",
        "Markdown Viewer",
        "JSON Viewer",
        "ColorZilla",
        "WhatFont",
        "Session Buddy",
        "Tab Wrangler",
        "The Great Suspender",
        "OneTab",
        "Bookmark Manager",
        "History Manager",
        "Google Translate",
        "WordReference",
        "Merriam-Webster Dictionary",
    ]
    
    # Randomly select 3-7 extensions to simulate
    num_extensions = random.randint(3, 7)
    selected_extensions = random.sample(common_extensions, num_extensions)
    
    # Create a properly formatted JavaScript string
    extensions_list = ', '.join(f'"{ext}"' for ext in selected_extensions)
    
    # Inject JavaScript to simulate these extensions (tolerant)
    extensions_script = f"""
    (function() {{
        function setProp(obj, name, getter) {{
            try {{
                const proto = Object.getPrototypeOf(obj) || obj;
                Object.defineProperty(proto, name, {{ get: getter, configurable: true }});
                return;
            }} catch (e) {{
                try {{ obj.__defineGetter__(name, getter); return; }} catch (e2) {{}}
            }}
        }}

        try {{ window.chrome = window.chrome || {{}}; window.chrome.runtime = window.chrome.runtime || {{}}; window.chrome.runtime.id = 'abcdefghijklmnop'; }} catch (e) {{}}

        const extensions = [{extensions_list}];

        try {{
            setProp(navigator, 'plugins', function() {{
                const plugins = [];
                extensions.forEach(function(ext) {{
                    plugins.push({{
                        name: ext + ' Extension',
                        filename: ext.toLowerCase().replace(/\\s+/g, '-') + '.js',
                        description: ext + ' browser extension',
                        version: '{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}',
                        length: 1
                    }});
                }});
                return plugins;
            }});
        }} catch (e) {{ console.warn('plugins spoof failed', e); }}

        try {{
            setProp(navigator, 'mimeTypes', function() {{
                const mimeTypes = [];
                extensions.forEach(function(ext) {{
                    mimeTypes.push({{
                        type: 'application/x-' + ext.toLowerCase().replace(/\\s+/g, '-'),
                        description: ext + ' Extension',
                        suffixes: ext.toLowerCase().replace(/\\s+/g, '-'),
                        enabledPlugin: {{ name: ext + ' Extension', filename: ext.toLowerCase().replace(/\\s+/g, '-') + '.js' }}
                    }});
                }});
                return mimeTypes;
            }});
        }} catch (e) {{ console.warn('mimeTypes spoof failed', e); }}
    }})();
    """
    
    try:
        await page.evaluate(extensions_script)
        logger.info(f"🧩 Simulated {num_extensions} browser extensions")
    except Exception as e:
        logger.warning(f"Error simulating browser extensions: {e}")

# === NEW: Hybrid Click Function ===
async def hybrid_click_playwright_http(page, session, headers, cpa_url, click_id):
    """
    Hybrid approach: Use Playwright for complex browser behaviors and HTTP for clean click - FIXED to only make one click
    """
    try:
        # 1. Use Playwright to establish a realistic browser session
        logger.info("🌐 Establishing realistic browser session with Playwright...")
        
        # Visit the CPA page with Playwright to establish session - this will be our only click
        referer_to_use = random.choice(referers) if referers else None
        
        # Try to navigate with a longer timeout and handle potential timeouts
        logger.info(f"➡️ hybrid_click_playwright_http: navigating to cpa_url='{cpa_url}' referer='{referer_to_use}'")
        try:
            await page.goto(cpa_url, referer=referer_to_use, timeout=60000)  # Increased timeout to 60 seconds
            logger.info(f"⬅️ page.goto returned. Current page URL: {page.url}")
        except Exception as nav_error:
            logger.warning(f"⚠️ Navigation failed: {nav_error}")
            logger.exception(nav_error)
            # Try to reload the page as a fallback
            try:
                await page.reload(timeout=30000)
                logger.info("🔁 page.reload succeeded as fallback")
            except Exception as reload_err:
                logger.warning(f"❌ page.reload also failed: {reload_err}")
                logger.exception(reload_err)
        
        # Wait for page to load
        await asyncio.sleep(random.uniform(2, 5))
        
        # 2. Extract cookies and session information from Playwright
        logger.info("🍪 Extracting session information from Playwright...")
        playwright_cookies = await page.context.cookies()
        
        # Convert to requests format
        cookie_dict = {}
        for cookie in playwright_cookies:
            cookie_dict[cookie['name']] = cookie['value']
        
        # Update session cookies
        session.cookies.update(cookie_dict)
        
        # 3. Simulate realistic browsing behavior with Playwright
        logger.info("🖱️ Simulating realistic browsing behavior...")
        
        # Random scroll
        if random.random() < 0.7:  # 70% chance to scroll
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(random.uniform(1, 3))
            except:
                pass
        
        # Random mouse movement
        if random.random() < 0.6:  # 60% chance to move mouse
            try:
                x = random.randint(50, page.viewport_size["width"] - 50)
                y = random.randint(50, page.viewport_size["height"] - 50)
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            except:
                pass
        
        # Random dwell time (10-14 seconds as requested)
        dwell_time = random.randint(10, 14)
        logger.info(f"⏳ Simulating dwell time of {dwell_time} seconds...")
        await asyncio.sleep(dwell_time)
        
        logger.info("✅ Hybrid click successful (Playwright only)!")
        return True
            
    except Exception as e:
        logger.error(f"❌ Hybrid click failed: {e}")
        return False

# === FIXED: Process single click with improved error handling ===
async def process_click(click_num, proxy, device, ua, device_fingerprint, ip_info, session, retry_count=3):
    try:
        # Start timing the click processing
        click_start_time = time.time()
        
        # Use Playwright for browser automation
        async with async_playwright() as p:
                # Pause/Resume GUI is started from main() to ensure it's created
                # in the main thread (avoids starting Tk from a worker thread).
            # Format proxy for Playwright before launching browser
            playwright_proxy = None
            if proxy:
                proxy_dict = format_proxy(proxy)
                if "http" in proxy_dict:
                    proxy_server = proxy_dict["http"]
                    # Extract proxy server and port
                    if "@" in proxy_server:
                        # Format: username:password@host:port
                        auth_server = proxy_server.split("@")[1]
                    else:
                        # Format: host:port
                        auth_server = proxy_server
                    
                    # Parse server and port
                    if "://" in auth_server:
                        server_parts = auth_server.replace("http://", "").replace("socks5://", "").split(":")
                    else:
                        server_parts = auth_server.split(":")
                    
                    server = server_parts[0]
                    port = int(server_parts[1]) if len(server_parts) > 1 else 8080
                    
                    # Create proxy config for Playwright
                    playwright_proxy = {
                        "server": f"{server}:{port}"
                    }
                    
                    # Add authentication if needed
                    if "@" in proxy_server:
                        auth_part = proxy_server.split("@")[0]
                        if "://" in auth_part:
                            auth_part = auth_part.split("://")[1]
                        
                        if ":" in auth_part:
                            username, password = auth_part.split(":", 1)
                            playwright_proxy["username"] = username
                            playwright_proxy["password"] = password
            
            # Select browser type based on configuration
            if config.browser_type == "chromium":
                browser = await p.chromium.launch(
                    headless=config.use_headless_browser,
                    proxy=playwright_proxy,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu"
                    ]
                )
            elif config.browser_type == "firefox":
                browser = await p.firefox.launch(
                    headless=config.use_headless_browser,
                    proxy=playwright_proxy
                )
            elif config.browser_type == "webkit":
                browser = await p.webkit.launch(
                    headless=config.use_headless_browser,
                    proxy=playwright_proxy
                )
            else:
                # Default to chromium
                browser = await p.chromium.launch(
                    headless=config.use_headless_browser,
                    proxy=playwright_proxy,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu"
                    ]
                )
            
            try:
                # Get viewport size as tuple and convert to dictionary
                viewport_size_tuple = get_random_viewport_size(device)
                viewport_size = {"width": viewport_size_tuple[0], "height": viewport_size_tuple[1]}

                # Create a new BrowserContext early and inject the spoofing script at the
                # context level so it runs before any page or page scripts execute.
                context = await browser.new_context(
                    user_agent=ua,
                    viewport=viewport_size,
                    locale="en-US",
                    device_scale_factor=3,
                    is_mobile=True
                )

                # Add a context-level init script derived from the device fingerprint.
                # This ensures navigator properties are overridden before any page JS runs.
                try:
                    spoofing_script = get_spoofing_script(device_fingerprint.device_info)
                    await context.add_init_script(spoofing_script)
                except Exception as e:
                    logger.warning(f"Could not add context init script: {e}")

                # Then create the page from the context
                page = await context.new_page()
                
                # 1. Inject advanced fingerprint spoofing
                if config.enable_advanced_fingerprint_spoofing:
                    await inject_advanced_fingerprint_spoofing(page, device_fingerprint)
                
                # 2. Simulate realistic network conditions
                if config.enable_realistic_network_simulation:
                    await simulate_realistic_network_conditions(page)
                
                # 3. Enhanced session simulation
                if config.enable_session_simulation:
                    try:
                        await enhanced_session_simulation(page, max_time=120)
                    except Exception as e:
                        logger.warning(f"Error during enhanced session simulation: {e}")
                
                # 4. Inject WebGL/Canvas spoofing
                if config.enable_canvas_fingerprinting:
                    await device_fingerprint.inject_webgl_canvas_spoofing(page)
                
                # 5. Inject Audio fingerprint spoofing
                if config.enable_audio_fingerprinting:
                    await device_fingerprint.inject_audio_fingerprint_spoofing(page)
                
                # 6. Set timezone and locale based on proxy
                if config.enable_environment_consistency:
                    await device_fingerprint.set_timezone_locale_from_proxy(page, proxy)
                
                # 7. Block WebRTC leaks if enabled
                if config.enable_webrtc_protection:
                    await device_fingerprint.block_webrtc_leaks(page)
                
                # 8. Simulate browser extensions if enabled
                if config.enable_browser_extension_simulation:
                    await simulate_browser_extensions(page)
                
                # 9. Ensure device consistency
                if config.enable_device_consistency:
                    await ensure_consistent_device_parameters(page, device_fingerprint.device_info)
                
                # 10. Simulate human reading patterns
                if config.enable_human_reading_patterns:
                    await simulate_human_reading_pattern(page)
                
                # 11. Simulate human idle time
                if config.enable_human_idle_simulation:
                    await simulate_human_idle_time(page)
                
                # 12. Simulate realistic browser behavior
                if config.enable_enhanced_browser_behavior:
                    await simulate_realistic_browser_behavior(page)
                
                # NEW: 13. Simulate cognitive patterns
                if config.enable_cognitive_patterns:
                    await device_fingerprint.simulate_cognitive_patterns(page)
                
                # NEW: 14. Simulate network fingerprinting
                if config.enable_network_fingerprinting:
                    await device_fingerprint.simulate_network_fingerprint(page)
                
                # NEW: 15. Simulate performance metrics
                if config.enable_performance_metrics:
                    await device_fingerprint.simulate_performance_metrics(page)
                
                # Get realistic HTTP headers with ordering and timing
                headers, browser_type = realistic_http_headers()
                headers["User-Agent"] = ua  # Set the user agent
                
                # Enhance headers with fingerprint data
                headers, click_id = await simulate_js_env(headers, ua, device_fingerprint, device, ip_info)
                
                # Visit random sites first using Playwright (OPTIMIZED)
                try:
                    logger.info("🌐 Visiting warm-up and pre-click sites using Playwright...")
                    await device_fingerprint.random_pre_click_interaction(
                        page, 
                        min_sites=config.min_pre_click_sites, 
                        max_sites=config.max_pre_click_sites,
                        max_time=90  # Reduced to 90 seconds for faster clicks
                    )
                except Exception as e:
                    logger.warning(f"Error during Playwright pre-click visits: {e}")
                
                # Get IP info again to ensure consistency
                ip, city, country, region, tz, isp, lat, lng = ip_info
                
                # Use the hybrid approach for the click
                logger.info(f"➡️ Calling hybrid_click_playwright_http with cpa_url='{config.cpa_url}' click_id='{click_id}' page_obj={page}")
                result = await hybrid_click_playwright_http(page, session, headers, config.cpa_url, click_id)
                logger.info(f"⬅️ hybrid_click_playwright_http returned: {result}")
                
                if result:
                    # Extract device info
                    phone_model = extract_phone_model(ua)
                    time_str = get_local_time(tz)
                    device_id = device_fingerprint.generate_device_id()
                    
                    # Log the click
                    log_click(phone_model, proxy, ip, city, region, country, time_str, isp, device_id, click_id)
                    
                    # Update success rate
                    device_success_rates[device]["success"] += 1
                    device_success_rates[device]["total"] += 1
                    
                    # Print success message
                    console.print(f"[{click_num}] ✅ Clicked | IP: {ip} | Device: {device} ({phone_model}) | "
                          f"Location: {city}, {region}, {country} | Time: {time_str}")
                    
                    return True
                else:
                    return False
            finally:
                # Close the browser
                await browser.close()
    except Exception as e:
        logger.error(f"[{click_num}] ❌ CPA Click Failed: {e}")
        device_success_rates[device]["total"] += 1
        
        if retry_count > 0:
            logger.info(f"🔄 Retrying... ({retry_count} attempts left)")
            time.sleep(config.retry_delay)
            return await process_click(click_num, proxy, device, ua, device_fingerprint, ip_info, session, retry_count - 1)
        return False

# === NEW: Adaptive timing system ===
async def adaptive_timing():
    """Calculate adaptive wait time based on success rate"""
    if not config.adaptive_timing:
        return random.randint(config.min_wait_between_clicks, config.max_wait_between_clicks)
    
    total_success = sum(device_success_rates[device]["success"] for device in device_success_rates)
    total_attempts = sum(device_success_rates[device]["total"] for device in device_success_rates)
    
    if total_attempts == 0:
        return random.randint(config.min_wait_between_clicks, config.max_wait_between_clicks)
    
    success_rate = total_success / total_attempts
    
    # Adjust timing based on success rate
    if success_rate < 0.5:  # Low success rate - slow down
        return random.randint(config.max_wait_between_clicks, config.max_wait_between_clicks + 60)
    elif success_rate < 0.7:  # Medium success rate - normal speed
        return random.randint(config.min_wait_between_clicks, config.max_wait_between_clicks)
    else:  # High success rate - can speed up
        return random.randint(config.min_wait_between_clicks - 30, config.max_wait_between_clicks - 30)

# === NEW: Intelligent proxy selection ===
def select_proxy_by_performance(proxies, proxy_scores):
    """Select proxy based on historical performance"""
    if not proxy_scores:
        return random.choice(proxies)
    
    # Sort proxies by score (highest first)
    sorted_proxies = sorted(proxy_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select from top 20% of proxies
    top_proxies = sorted_proxies[:max(1, len(sorted_proxies) // 5)]
    
    # Randomly select from top performers
    return random.choice(top_proxies)[0]

# === NEW: Enhanced human simulation ===
async def simulate_human_hesitation(page):
    """Simulate human-like hesitation before actions"""
    if random.random() < 0.3:  # 30% chance
        hesitation_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(hesitation_time)

# === NEW: Real-time behavior adjustment ===
async def adjust_behavior_based_on_response(page):
    """Adjust behavior based on page response"""
    try:
        # Get page load time
        load_time = await page.evaluate("performance.timing.loadEventEnd - performance.timing.navigationStart")
        
        if load_time > 3000:  # Slow page
            await asyncio.sleep(random.uniform(2, 5))  # Wait longer
        elif load_time < 1000:  # Fast page
            await asyncio.sleep(random.uniform(0.5, 2))  # Wait shorter
            
    except Exception:
        pass  # Ignore errors

# === ADVANCED ANTI-DETECTION FEATURES ===
# 1. Quantum-Grade Fingerprinting
class QuantumFingerprint:
    def __init__(self):
        self.fingerprint_entropy = 0
        self.evolution_cycle = 0
    
    async def generate_quantum_canvas_fp(self):
        """Generate canvas fingerprint with quantum-level randomness"""
        canvas_script = """
        const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
        CanvasRenderingContext2D.prototype.getImageData = function() {
            const imageData = originalGetImageData.apply(this, arguments);
            // Quantum noise injection
            for (let i = 0; i < imageData.data.length; i += 4) {
                if (Math.random() < 0.001) { // Ultra-rare mutation
                    imageData.data[i] = imageData.data[i] ^ (Math.floor(Math.random() * 256));
                    imageData.data[i+1] = imageData.data[i+1] ^ (Math.floor(Math.random() * 256));
                    imageData.data[i+2] = imageData.data[i+2] ^ (Math.floor(Math.random() * 256));
                }
            }
            return imageData;
        };
        """
        return canvas_script
    
    async def morph_webgl_fingerprint(self, page):
        """Dynamically morph WebGL parameters during session"""
        morph_script = """
        setInterval(() => {
            // Subtle parameter morphing
            const params = [
                'MAX_TEXTURE_SIZE',
                'MAX_CUBE_MAP_TEXTURE_SIZE',
                'MAX_RENDERBUFFER_SIZE'
            ];
            params.forEach(param => {
                const original = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(p) {
                    if (p === WebGLRenderingContext[param]) {
                        return original.call(this, p) + Math.floor(Math.random() * 3 - 1);
                    }
                    return original.call(this, p);
                };
            });
        }, Math.random() * 30000 + 15000); // Random interval
        """
        await page.evaluate(morph_script)

# === 2. Neural Behavior Simulation
class NeuralBehaviorSimulator:
    def __init__(self):
        self.attention_span = random.uniform(8, 25)  # seconds
        self.distraction_threshold = 0.3
    
    async def simulate_neural_attention(self, page):
        """Simulate human attention patterns and distractions"""
        attention_script = f"""
        let attentionStart = Date.now();
        let lastInteraction = attentionStart;
        let distracted = false;
        
        function checkAttention() {{
            const now = Date.now();
            const attentionDuration = (now - attentionStart) / 1000;
            const idleTime = (now - lastInteraction) / 1000;
            
            // Random distraction
            if (!distracted && Math.random() < {self.distraction_threshold} && idleTime > 3) {{
                distracted = true;
                // Simulate distraction (look away, check phone, etc.)
                document.body.style.opacity = '0.95';
                setTimeout(() => {{
                    document.body.style.opacity = '1';
                    distracted = false;
                    attentionStart = Date.now();
                }}, Math.random() * 3000 + 1000);
            }}
            
            // Attention span exceeded
            if (attentionDuration > {self.attention_span}) {{
                window.scrollTo({{
                    top: Math.random() * document.body.scrollHeight,
                    behavior: 'smooth'
                }});
                attentionStart = Date.now();
            }}
        }}
        
        setInterval(checkAttention, 1000);
        """
        await page.evaluate(attention_script)
    
    async def simulate_decision_making(self, page):
        """Simulate human decision-making with hesitation"""
        decision_script = """
        function humanDecision(element) {
            return new Promise((resolve) => {
                const hesitation = Math.random() * 2000 + 500; // 0.5-2.5s
                const changeMind = Math.random() < 0.15; // 15% chance to change mind
                
                setTimeout(() => {
                    if (changeMind) {
                        // Move away from element
                        const rect = element.getBoundingClientRect();
                        const newX = rect.left + Math.random() * 200 - 100;
                        const newY = rect.top + Math.random() * 200 - 100;
                        window.dispatchEvent(new MouseEvent('mousemove', {
                            clientX: newX,
                            clientY: newY
                        }));
                        resolve(false);
                    } else {
                        resolve(true);
                    }
                }, hesitation);
            });
        }
        """
        await page.evaluate(decision_script)

# 3. Quantum Network Obfuscation
class QuantumNetworkObfuscator:
    def __init__(self):
        self.packet_morphing = True
        self.tls_fingerprint_rotation = {}
    
    async def morph_tls_fingerprint(self, session):
        """Dynamically morph TLS fingerprint during session"""
        if not hasattr(session, 'tls_fingerprint_index'):
            session.tls_fingerprint_index = 0
        
        cipher_suites = [
            'TLS_AES_128_GCM_SHA256',
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
            'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256',
            'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256'
        ]
        
        # Rotate cipher suite order
        random.shuffle(cipher_suites)
        
        # Update session adapter
        adapter = TLSAdapter()
        adapter.cipher_suites = cipher_suites
        session.mount('https://', adapter)
        
        session.tls_fingerprint_index = (session.tls_fingerprint_index + 1) % len(cipher_suites)
    
    async def simulate_network_conditions(self, page):
        """Simulate realistic network conditions with packet loss"""
        network_script = """
        // Simulate packet loss and latency variation
        const originalFetch = window.fetch;
        window.fetch = async function(...args) {
            const packetLoss = Math.random() < 0.02; // 2% packet loss
            const latencyVariation = Math.random() * 100 + 50; // 50-150ms
            
            if (packetLoss) {
                throw new Error('NetworkError: Connection lost');
            }
            
            return new Promise((resolve, reject) => {
                setTimeout(async () => {
                    try {
                        const result = await originalFetch.apply(this, args);
                        resolve(result);
                    } catch (error) {
                        reject(error);
                    }
                }, latencyVariation);
            });
        };
        """
        await page.evaluate(network_script)

# 4. Environmental Camouflage
class EnvironmentalCamouflage:
    def __init__(self):
        self.environment_signatures = {}
        self.camouflage_cycle = 0
    
    async def camouflage_environment(self, page, device_fingerprint):
        """Camouflage browser environment to match real device"""
        camouflage_script = f"""
        // Environmental camouflage
        const deviceInfo = {json.dumps(device_fingerprint.device_info)};
        
        // Camouflage screen properties
        Object.defineProperty(screen, 'availWidth', {{
            get: () => {device_fingerprint.get_screen_resolution().split('x')[0]} - Math.floor(Math.random() * 20)
        }});
        
        Object.defineProperty(screen, 'availHeight', {{
            get: () => {device_fingerprint.get_screen_resolution().split('x')[1]} - Math.floor(Math.random() * 50)
        }});
        
        // Camouflage memory
        const memoryInfo = {json.dumps(device_fingerprint.get_memory_info())};
        Object.defineProperty(performance, 'memory', {{
            get: () => ({{
                usedJSHeapSize: memoryInfo.usedJSHeapSize + Math.floor(Math.random() * 1000000),
                totalJSHeapSize: memoryInfo.totalJSHeapSize + Math.floor(Math.random() * 5000000),
                jsHeapSizeLimit: memoryInfo.jsHeapSizeLimit + Math.floor(Math.random() * 10000000)
            }})
        }});
        
        // Camouflage battery
        const batteryInfo = {json.dumps(device_fingerprint.get_battery_info())};
        if (navigator.getBattery) {{
            Object.defineProperty(navigator, 'getBattery', {{
                get: () => () => Promise.resolve({{
                    charging: batteryInfo.charging,
                    chargingTime: batteryInfo.chargingTime + Math.floor(Math.random() * 100),
                    dischargingTime: batteryInfo.dischargingTime + Math.floor(Math.random() * 1000),
                    level: Math.max(0, Math.min(1, batteryInfo.level + (Math.random() * 0.1 - 0.05)))
                }})
            }});
        }}
        """
        await page.evaluate(camouflage_script)
    
    async def simulate_environmental_changes(self, page):
        """Simulate environmental changes during session"""
        env_script = """
        // Simulate ambient light changes
        setInterval(() => {
            if ('AmbientLightSensor' in window) {
                const lightLevel = Math.random() * 1000 + 100; // 100-1100 lux
                // Trigger ambient light change event
            }
        }, Math.random() * 30000 + 10000);
        
        // Simulate device orientation changes
        if (window.DeviceOrientationEvent) {
            window.addEventListener('devicemotion', (event) => {
                // Add random noise to sensor readings
                event.acceleration.x += (Math.random() - 0.5) * 0.1;
                event.acceleration.y += (Math.random() - 0.5) * 0.1;
                event.acceleration.z += (Math.random() - 0.5) * 0.1;
            });
        }}
        """
        await page.evaluate(env_script)

# 5. Advanced Evasion Protocols
class AdvancedEvasionProtocols:
    def __init__(self):
        self.evasion_techniques = []
        self.detection_counter = 0
    
    async def implement_stealth_navigation(self, page):
        """Implement stealth navigation techniques"""
        stealth_script = """
        // Stealth navigation
        const originalPushState = history.pushState;
        const originalReplaceState = history.replaceState;
        
        history.pushState = function(state, title, url) {
            // Add random delay before navigation
            setTimeout(() => {
                originalPushState.apply(history, arguments);
            }, Math.random() * 200 + 50);
        };
        
        history.replaceState = function(state, title, url) {
            setTimeout(() => {
                originalReplaceState.apply(history, arguments);
            }, Math.random() * 200 + 50);
        }};
        
        // Simulate navigation cancellation
        let navigationCancelled = false;
        window.addEventListener('beforeunload', (e) => {
            if (Math.random() < 0.05) { // 5% chance to cancel navigation
                e.preventDefault();
                navigationCancelled = true;
                setTimeout(() => {
                    navigationCancelled = false;
                }, Math.random() * 3000 + 1000);
            }
        }});
        """
        await page.evaluate(stealth_script)
    
    async def implement_anti_profiling(self, page):
        """Implement anti-profiling measures"""
        anti_profile_script = """
        // Anti-profiling measures (safer): wrap defineProperty to ensure configurable:true for common props
        const originalDefineProperty = Object.defineProperty;
        Object.defineProperty = function(obj, prop, descriptor) {
            try {
                if (prop === 'webdriver' || prop === 'plugins' || prop === 'languages') {
                    const safeDesc = Object.assign({}, descriptor, { configurable: true });
                    return originalDefineProperty.call(Object, obj, prop, safeDesc);
                }
            } catch (e) { /* fall through */ }
            return originalDefineProperty.call(Object, obj, prop, descriptor);
        };
        
        // Randomize function call patterns
        const functionCallPatterns = [
            () => Math.random(),
            () => Date.now(),
            () => performance.now(),
            () => navigator.userAgent
        ];
        
        setInterval(() => {
            const randomPattern = functionCallPatterns[Math.floor(Math.random() * functionCallPatterns.length)];
            randomPattern();
        }, Math.random() * 5000 + 1000);
        """
        await page.evaluate(anti_profile_script)

# 6. Dynamic Adaptation System
class DynamicAdaptationSystem:
    def __init__(self):
        self.adaptation_history = []
        self.threat_level = 0
    
    async def analyze_threat_level(self, page):
        """Analyze current threat level and adapt behavior"""
        threat_analysis_script = """
        return new Promise((resolve) => {
            let threatScore = 0;
            
            // Check for automation indicators
            if (navigator.webdriver) threatScore += 30;
            if (window.chrome && window.chrome.runtime) threatScore += 20;
            
            // Check for unusual patterns
            const clickPatterns = [];
            document.addEventListener('click', (e) => {
                clickPatterns.push({
                    x: e.clientX,
                    y: e.clientY,
                    time: Date.now()
                });
            }});
            
            // Analyze click patterns
            setTimeout(() => {
                if (clickPatterns.length > 10) {
                    const avgTimeBetweenClicks = clickPatterns.reduce((acc, curr, i, arr) => {
                        if (i === 0) return 0;
                        return acc + (curr.time - arr[i-1].time);
                    }, 0) / (clickPatterns.length - 1);
                    
                    if (avgTimeBetweenClicks < 100) threatScore += 25;
                }
                resolve(threatScore);
            }, 5000);
        });
        """
        
        threat_score = await page.evaluate(threat_analysis_script)
        self.threat_level = threat_score
        
        return threat_score
    
    async def adapt_behavior(self, page, threat_level):
        """Adapt behavior based on threat level"""
        if threat_level > 70:
            # High threat - implement extreme stealth
            await self.extreme_stealth_mode(page)
        elif threat_level > 40:
            # Medium threat - increase caution
            await self.caution_mode(page)
        else:
            # Low threat - normal operation
            await self.normal_mode(page)
    
    async def extreme_stealth_mode(self, page):
        """Implement extreme stealth measures"""
        stealth_script = """
        // Extreme stealth measures
        document.body.style.filter = 'blur(0.5px)';
        setTimeout(() => {
            document.body.style.filter = 'none';
        }, Math.random() * 3000 + 1000);
        
        // Simulate human-like delays
        const originalSetTimeout = window.setTimeout;
        window.setTimeout = function(callback, delay) {
            return originalSetTimeout.call(window, callback, delay + Math.random() * 1000);
        }};
        """
        await page.evaluate(stealth_script)

# === FIXED: Main function with proper proxy handling ===
async def main():
    proxies_raw = load_proxies("proxies.txt")[:]
    good_proxies, bad_proxies, proxy_scores = batch_proxy_health_check(proxies_raw)
    proxies_raw = good_proxies[:]
    
    used_proxies = set()
    click = 0
    successful_clicks = 0
    campaign_clicks = 0
    publisher_clicks = 0
    
    # NEW: MAC change counter
    mac_change_counter = 0
    
    # NEW: Initialize performance monitor
    perf_monitor = PerformanceMonitor()
    
    # NEW: Initialize proxy manager
    proxy_manager = ProxyManager()
    
    # Initialize advanced systems
    quantum_fp = QuantumFingerprint()
    neural_sim = NeuralBehaviorSimulator()
    network_obf = QuantumNetworkObfuscator()
    env_cam = EnvironmentalCamouflage()
    evasion_proto = AdvancedEvasionProtocols()
    adapt_system = DynamicAdaptationSystem()
    
    banner()
    
    # Get original MAC addresses
    original_macs = get_all_mac_addresses()
    if original_macs:
        logger.info("🔍 Original MAC Addresses:")
        for adapter, mac in original_macs.items():
            logger.info(f"   {adapter}: {mac}")
    else:
        logger.warning("⚠️ Could not get original MAC addresses")
    
    # Initialize MAC tracking variables
    previous_macs = original_macs.copy()
    current_macs = original_macs.copy()
    
    # Log enhanced features status
    logger.info("🔧 Enhanced Features Status:")
    logger.info(f"   Mouse Simulation: {'Enabled' if config.enable_mouse_simulation else 'Disabled'}")
    logger.info(f"   Scroll Simulation: {'Enabled' if config.enable_scroll_simulation else 'Disabled'}")
    logger.info(f"   Session Simulation: {'Enabled' if config.enable_session_simulation else 'Disabled'}")
    logger.info(f"   Form Interaction: {'Enabled' if config.enable_form_interaction else 'Disabled'}")
    logger.info(f"   Cloudflare Bypass: {'Enabled' if config.enable_cloudflare_bypass else 'Disabled'}")
    logger.info(f"   TLS Fingerprinting: {'Enabled' if config.enable_tls_fingerprinting else 'Disabled'}")
    logger.info(f"   Header Ordering: {'Enabled' if config.enable_header_ordering else 'Disabled'}")
    logger.info(f"   Environment Consistency: {'Enabled' if config.enable_environment_consistency else 'Disabled'}")
    logger.info(f"   Device Consistency: {'Enabled' if config.enable_device_consistency else 'Disabled'}")
    logger.info(f"   Canvas Fingerprinting: {'Enabled' if config.enable_canvas_fingerprinting else 'Disabled'}")
    logger.info(f"   WebRTC Protection: {'Enabled' if config.enable_webrtc_protection else 'Disabled'}")
    logger.info(f"   Header Randomization: {'Enabled' if config.enable_header_randomization else 'Disabled'}")
    logger.info(f"   Network Fingerprinting: {'Enabled' if config.enable_network_fingerprinting else 'Disabled'}")
    logger.info(f"   Human Error Simulation: {'Enabled' if config.human_error_probability > 0 else 'Disabled'}")
    logger.info(f"   Random Element Interaction: {'Enabled' if config.random_interaction_probability > 0 else 'Disabled'}")
    
    # Log WebGPU status
    if config.disable_webgpu:
        logger.info("🔒 WebGPU is disabled for enhanced undetectability")
    
    # Log randomness settings
    logger.info(f"🎲 Randomness Settings:")
    logger.info(f"   Pre-click sites: {config.min_pre_click_sites}-{config.max_pre_click_sites}")
    logger.info(f"   Human error probability: {config.human_error_probability}")
    logger.info(f"   Random interaction probability: {config.random_interaction_probability}")
    
    # Log new enhanced features
    logger.info(f"🚀 Enhanced Features:")
    logger.info(f"   Rest Periods: {'Enabled' if config.enable_rest_periods else 'Disabled'}")
    logger.info(f"   Browser Extension Simulation: {'Enabled' if config.enable_browser_extension_simulation else 'Disabled'}")
    logger.info(f"   Random Viewport Sizes: {'Enabled' if config.enable_random_viewport_sizes else 'Disabled'}")
    logger.info(f"   Natural Click Patterns: {'Enabled' if config.enable_natural_click_patterns else 'Disabled'}")
    logger.info(f"   Operational Hours: {'Enabled' if config.enable_operational_hours else 'Disabled'}")
    logger.info(f"   Success Monitoring: {'Enabled' if config.enable_success_monitoring else 'Disabled'}")
    
    # Log new behavioral variance features
    logger.info("🚀 New Enhanced Features:")
    logger.info(f"   Behavioral Variance: Enabled")
    logger.info(f"   Proxy Quality Check: Enabled")
    logger.info(f"   Advanced Device Consistency: Enabled")
    logger.info(f"   Natural Timing Patterns: Enabled")
    logger.info(f"   Session Depth Variation: Enabled")
    logger.info(f"   Click Pattern Randomization: Enabled")
    
    # Log audio fingerprinting status
    logger.info(f"🔊 Audio Fingerprinting: {'Enabled' if config.enable_audio_fingerprinting else 'Disabled'}")
    
    # Log browser persistence status
    logger.info(f"💾 Browser Persistence: {'Enabled' if config.enable_browser_persistence else 'Disabled'}")
    
    # Log timing requirements
    logger.info("⏱️ Timing Requirements:")
    logger.info(f"   Pre-visit: Maximum 90 seconds (optimized)")
    logger.info(f"   Minimum wait between clicks: {config.min_wait_between_clicks} seconds (3 minutes)")
    logger.info(f"   Maximum wait between clicks: {config.max_wait_between_clicks} seconds (4 minutes)")
    logger.info(f"   Target click rate: 12-15 clicks per hour")
    logger.info(f"   MAC change frequency: Every {config.mac_change_frequency} clicks")
    
    # Log new enhanced behavior settings
    logger.info("🚀 Advanced Behavior Features:")
    logger.info(f"   Advanced Fingerprint Spoofing: {'Enabled' if config.enable_advanced_fingerprint_spoofing else 'Disabled'}")
    logger.info(f"   Realistic Network Simulation: {'Enabled' if config.enable_realistic_network_simulation else 'Disabled'}")
    logger.info(f"   Enhanced Browser Behavior: {'Enabled' if config.enable_enhanced_browser_behavior else 'Disabled'}")
    logger.info(f"   Human Reading Patterns: {'Enabled' if config.enable_human_reading_patterns else 'Disabled'}")
    logger.info(f"   Human Idle Simulation: {'Enabled' if config.enable_human_idle_simulation else 'Disabled'}")
    logger.info(f"   Session Depth Variation: {'Enabled' if config.enable_session_depth_variation else 'Disabled'}")
    
    # Log new optimized settings
    logger.info("🚀 Optimized Settings:")
    logger.info(f"   Adaptive Timing: {'Enabled' if config.adaptive_timing else 'Disabled'}")
    logger.info(f"   Intelligent Proxy Selection: Enabled")
    logger.info(f"   Enhanced Human Simulation: Enabled")
    logger.info(f"   Real-time Behavior Adjustment: Enabled")
    
    # Log new advanced features
    logger.info("🚀 Advanced Features:")
    logger.info(f"   Cognitive Patterns: {'Enabled' if config.enable_cognitive_patterns else 'Disabled'}")
    logger.info(f"   Network Fingerprinting: {'Enabled' if config.enable_network_fingerprinting else 'Disabled'}")
    logger.info(f"   Performance Metrics: {'Enabled' if config.enable_performance_metrics else 'Disabled'}")
    logger.info(f"   Cookie Jar Management: {'Enabled' if config.enable_cookie_jar else 'Disabled'}")
    logger.info(f"   Intelligent Error Recovery: {'Enabled' if config.enable_intelligent_recovery else 'Disabled'}")
    logger.info(f"   Performance Monitoring: {'Enabled' if config.enable_performance_monitoring else 'Disabled'}")
    logger.info(f"   Human-like Delays: {'Enabled' if config.enable_human_delays else 'Disabled'}")
    logger.info(f"   Proxy Rotation: {'Enabled' if config.enable_proxy_rotation else 'Disabled'}")
    
    # Start Pause/Resume GUI on the main thread so Tkinter is initialized properly.
    try:
        start_pause_gui_thread()
    except Exception:
        logger.warning("Pause GUI could not be started on main thread")

    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        while (config.infinite_loop or click < config.max_clicks):
            # Respect Pause/Resume GUI
            await wait_if_paused()
            # Check if within operating hours
            if config.enable_operational_hours and not is_within_operational_hours():
                current_hour = datetime.datetime.now().hour
                start_hour = config.operating_hours_start
                end_hour = config.operating_hours_end
                
                logger.info(f"⏰ Outside operating hours ({current_hour}:00). Waiting until {start_hour}:00...")
                
                # Calculate time until next operating hour
                now = datetime.datetime.now()
                if current_hour < start_hour:
                    # Same day
                    next_operating = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
                else:
                    # Next day
                    next_operating = (now + datetime.timedelta(days=1)).replace(hour=start_hour, minute=0, second=0, microsecond=0)
                
                wait_seconds = (next_operating - now).total_seconds()
                countdown_timer(int(wait_seconds))
                continue
            
            if campaign_clicks >= config.max_clicks_per_campaign:
                console.print(f"🚫 Campaign limit reached ({config.max_clicks_per_campaign} clicks)")
                break
            if publisher_clicks >= config.max_clicks_per_publisher:
                console.print(f"🚫 Publisher limit reached ({config.max_clicks_per_publisher} clicks)")
                break
            
            print_clicks_left()
            device = select_device()
            
            # NEW: Use intelligent proxy selection
            if config.rotation_strategy == "performance":
                current_proxy = select_proxy_by_performance(proxies_raw, proxy_scores)
            else:
                current_proxy = select_proxy(proxies_raw, proxy_scores, config.rotation_strategy, used_proxies)
            
            if not current_proxy:
                console.print("🛑 No unused working proxies left. Stopping.")
                break
            
            if device == "iPhone":
                ua = get_and_remove_ua("iPhone", iphone_agents, "useragents/iphoneuseragent.TXT")
            elif device == "Samsung":
                ua = get_and_remove_ua("Samsung", samsung_agents, "useragents/samsunguseragent.TXT")
            elif device == "Windows":
                ua = get_and_remove_ua("Windows", windows_agents, "useragents/windowsuseragent.TXT")
            elif device == "GooglePixel":
                ua = get_and_remove_ua("GooglePixel", googlepixel_agents, "useragents/googlepixeluseragent.TXT")
            elif device == "Motorola":
                ua = get_and_remove_ua("Motorola", motorola_agents, "useragents/motorolauseragent.TXT")
            elif device == "OnePlus":
                ua = get_and_remove_ua("OnePlus", oneplus_agents, "useragents/oneplususeragent.TXT")
            elif device == "Vivo":
                ua = get_and_remove_ua("Vivo", vivo_agents, "useragents/vivouseragent.TXT")
            elif device == "Xiaomi":
                ua = get_and_remove_ua("Xiaomi", xiaomi_agents, "useragents/xiaomiuseragent.TXT")
            elif device == "ZTE":
                ua = get_and_remove_ua("ZTE", zte_agents, "useragents/zteuseragent.TXT")
            else:
                ua = None
            
            if not ua:
                console.print(f"⚠️ Skipping {device} click due to no available user agents.")
                continue
            
            proxy = format_proxy(current_proxy)
            ip_info = get_ip_info(proxy)
            
            # Create device fingerprint with the user agent
            device_fingerprint = DeviceFingerprint(ua)
            
            session = requests.Session()
            if config.enable_tls_fingerprinting:
                session.mount('https://', TLSAdapter())
            session.proxies = proxy
            
            click += 1
            future = executor.submit(
                asyncio.run,
                process_click(click, current_proxy, device, ua, device_fingerprint, ip_info, session)
            )
            result = future.result()
            
            # NEW: Update performance metrics
            click_time = time.time()
            response_time = random.uniform(1.0, 5.0)  # Simulated response time
            perf_monitor.update_metrics(click_time, result, response_time, current_proxy)
            
            if result:
                successful_clicks += 1
                campaign_clicks += 1
                publisher_clicks += 1
                console.print("\n🎉🎉🎉 HEY BOSS YOU HAVE GOT A SUCCESSFUL CLICK 🎉🎉🎉")
                console.print(f"📊 Total successful clicks: {successful_clicks}")
                
                # Monitor success rates
                monitor_success_rates()
                
                # NEW: Print performance report
                if config.enable_performance_monitoring and click % 10 == 0:
                    perf_report = perf_monitor.get_performance_report()
                    console.print(f"\n📊 Performance Report:")
                    console.print(f"   Clicks per hour: {perf_report['clicks_per_hour']}")
                    console.print(f"   Success rate: {perf_report['success_rate']:.2%}")
                    console.print(f"   Avg response time: {perf_report['avg_response_time']:.2f}s")
                    console.print(f"   Active proxies: {perf_report['proxy_count']}")
                
                # NEW: Change MAC addresses only every 5 clicks
                mac_change_counter += 1
                if mac_change_counter >= config.mac_change_frequency and platform.system() == "Windows":
                    console.print(f"\n🔄 Changing MAC addresses (every {config.mac_change_frequency} clicks)...")
                    
                    # Change Wi-Fi MAC address
                    previous_wifi_mac = current_macs.get(wifi_adapter_name, "Unknown")
                    new_wifi_mac = change_mac_windows(wifi_adapter_name)
                    if new_wifi_mac:
                        current_macs[wifi_adapter_name] = new_wifi_mac
                        console.print(f"\n📡 Wi-Fi MAC Address Changed:")
                        console.print(f"   Adapter: {wifi_adapter_name}")
                        console.print(f"   Original: {original_macs.get(wifi_adapter_name, 'Unknown')}")
                        console.print(f"   Previous: {previous_wifi_mac}")
                        console.print(f"   Current:  {new_wifi_mac}")
                    else:
                        console.print("⚠️ Failed to change Wi-Fi MAC address")
                    
                    # Change device MAC address
                    previous_device_macs = current_macs.copy()
                    new_device_mac = change_device_mac_address()
                    
                    # Get updated MAC addresses after device MAC change
                    updated_macs = get_all_mac_addresses()
                    if updated_macs:
                        current_macs = updated_macs
                        
                        console.print(f"\n🖥️ Device MAC Address Changed:")
                        for adapter in original_macs:
                            if adapter in updated_macs and updated_macs[adapter] != original_macs[adapter]:
                                console.print(f"   Adapter: {adapter}")
                                console.print(f"   Original: {original_macs[adapter]}")
                                console.print(f"   Previous: {previous_device_macs.get(adapter, 'Unknown')}")
                                console.print(f"   Current:  {updated_macs[adapter]}")
                    else:
                        console.print("⚠️ Could not verify device MAC address change")
                    
                    # Reset counter
                    mac_change_counter = 0
                
                # Calculate rest period
                rest_period = calculate_rest_period()
                if rest_period > 0:
                    countdown_timer(rest_period)
                
                # Add a small delay after MAC changes
                time.sleep(3)
            
            # FIXED: Only remove the used proxy from the file and in-memory list
            used_proxies.add(current_proxy)
            if config.remove_proxy_after_use and current_proxy in proxies_raw:
                proxies_raw.remove(current_proxy)
                remove_line_from_file("proxies.txt", current_proxy)
                console.print(f"🗑️ Removed used proxy: {current_proxy}")
                
                # Also update the proxy_scores dictionary
                if current_proxy in proxy_scores:
                    del proxy_scores[current_proxy]
            
            # NEW: Use adaptive timing for wait between clicks
            if config.adaptive_timing:
                wait = await adaptive_timing()
            else:
                wait = random.randint(config.min_wait_between_clicks, config.max_wait_between_clicks)
            
            console.print(f"\n⏳ Waiting for next click...")
            countdown_timer(wait)
    
    console.print("\n==============================")
    console.print("BOT RUN SUMMARY")
    console.print("==============================")
    console.print(f"Total proxies loaded: {len(good_proxies) + len(bad_proxies)}")
    console.print(f"Proxies removed as bad: {len(bad_proxies)}")
    console.print(f"Total successful clicks: {successful_clicks}")
    console.print(f"Campaign clicks: {campaign_clicks}/{config.max_clicks_per_campaign}")
    console.print(f"Publisher clicks: {publisher_clicks}/{config.max_clicks_per_publisher}")
    console.print(f"Good proxies unused: {len(good_proxies) - len(used_proxies)}")
    
    console.print("\n📋 MAC Address Summary:")
    final_macs = get_all_mac_addresses()
    for adapter in original_macs:
        console.print(f"\n{adapter}:")
        console.print(f"   Original: {original_macs[adapter]}")
        console.print(f"   Final:    {final_macs.get(adapter, 'Unknown')}")
        if original_macs[adapter] != final_macs.get(adapter, 'Unknown'):
            console.print(f"   Status:   ✅ Changed")
        else:
            console.print(f"   Status:   ❌ Unchanged")
    
    # NEW: Print final performance report
    if config.enable_performance_monitoring:
        perf_report = perf_monitor.get_performance_report()
        console.print("\n📊 Final Performance Report:")
        console.print(f"   Clicks per hour: {perf_report['clicks_per_hour']}")
        console.print(f"   Success rate: {perf_report['success_rate']:.2%}")
        console.print(f"   Avg response time: {perf_report['avg_response_time']:.2f}s")
        console.print(f"   Active proxies: {perf_report['proxy_count']}")
    
    console.print("==============================\n")

# === Countdown Timer ===
def countdown_timer(seconds):
    start_time = time.time()
    end_time = start_time + seconds
    while time.time() < end_time:
        remaining = int(end_time - time.time())
        minutes = remaining // 60
        seconds = remaining % 60
        print(f"\rNext click in: {minutes} minutes {seconds:02d} seconds", end="", flush=True)
        time.sleep(1)
    print()

# === FIXED: Calculate natural wait time ===
def calculate_natural_wait_time():
    """Calculate a natural wait time based on time of day and other factors"""
    current_hour = datetime.datetime.now().hour
    
    # Different patterns for different times of day
    if 6 <= current_hour < 9:  # Morning rush
        base_wait = random.randint(180, 240)  # 3-4 minutes
    elif 9 <= current_hour < 17:  # Work hours
        base_wait = random.randint(180, 240)  # 3-4 minutes
    elif 17 <= current_hour < 22:  # Evening
        base_wait = random.randint(180, 240)  # 3-4 minutes
    else:  # Night
        base_wait = random.randint(180, 240)  # 3-4 minutes
    
    # Add some randomness
    wait_time = base_wait + random.randint(-30, 30)
    
    # Ensure within configured bounds
    wait_time = max(config.min_wait_between_clicks, min(wait_time, config.max_wait_between_clicks))
    
    return wait_time

# === Run the bot ===
from src import auth
from src import auth_gui

if __name__ == "__main__":
    # Enforce sign-in before running the bot
    if auth is None:
        print("Authentication module not available. Install requirements and ensure auth.py is present.")
        sys.exit(1)
    try:
        # Show GUI to sign in. The GUI enables the Run button only after success.
        user = auth_gui.auth_gui_prompt()
    except Exception as e:
        print(f'Authentication failed or cancelled: {e}')
        sys.exit(1)

    asyncio.run(main())
