### Title: "Compiler Explorer: Features You Never Knew Existed"

### Abstract

Remember when Compiler Explorer just showed assembly code? Those innocent days are long gone. What started as a simple "what does my compiler do?" tool (in bash!) has morphed into 100,000 lines of TypeScript powering 100 million compilations a year. It runs CUDA on actual GPUs, tests your code against 10 compilers simultaneously, and maintains 3,000+ compiler versions that refuse to retire. Join me as we excavate CE's archaeological layers of features, from ones you probably know about ("it can run code!"), through the mysterious Conformance Viewer and IDE mode, to the fun and interesting visualisations of your compiler's output. 

We'll uncover the features hiding in plain sight: arbitrary subdomain URLs (badger.godbolt.org works!), keyboard shortcuts that make you look like a wizard, and the optimization pipeline viewer that shows exactly how -O3 turns your careful algorithms into three assembly instructions. I'll reveal how every short link lasts forever, how we safely run arbitrary code from the internet, and how we squeeze 80+ languages into $3,000/month of infrastructure. Whether you're a CE power user or someone who just discovered the Executor view, you'll learn about some pretty cool features you never knew existed! (Honestly neither did I until I started preparing this talk!)

---

Broad plan:
- 5m Intro, me and CE
- 40m UI overview - JUICY MEAT THIS IS WHAT PEOPLE ARE HERE FOR
  - 10m Overview, all the main parts
  - 5-10m Execution, settings therein. CUDA & ARM
  - 5m Analysis tooling
  - 5m IDE mode  
  - 5m Misc trivia
- 5m Behind the scenes (WATCH TIME)
  - Overview
  - goo.gl
- 10m QUESTIONS and buffer (lots of live demos)

---

### Claude's suggestions

(mostly ignored but kept for posterity)

## Major Features to Demo (35 minutes)

### 1. The Evolution Story (5 minutes)
Start with the dramatic statistics:
- **2020**: ~5 languages, assembly-only → **2025**: 81+ languages, full execution
- **Scale**: 92 million compilations/year (1.8M/week)
- **Infrastructure**: 3,000+ compiler versions that never retire
- **The verb**: How "Godbolt it" entered developer vocabulary

### 2. "Wait, It Can Run Code?!" (7 minutes)
**The #1 surprise feature** - demonstrate the execution capabilities:
- Show simple program execution with output
- Demo debugging with sanitizers
- Run multi-threaded code (up to 3 threads)
- **CUDA on real GPUs**: Show actual GPU kernel execution
- Quick Android/Kotlin optimization demo

### 3. Hidden Analysis Tools (8 minutes)
**"X-ray vision into the compiler"** features:
- **Optimization Pipeline Viewer**: Watch -O0 transform to -O3 step-by-step
- **Control Flow Graphs**: Visual program flow most users never find
- **Multiple IR views**: LLVM IR, GCC Tree/RTL, Rust MIR
- **Stack Usage Analysis**: Critical for embedded developers
- **Machine Code Analyzer (LLVM-MCA)**: Theoretical performance analysis

### 4. The Conformance Viewer Mystery (5 minutes)
**The most underutilized power feature**:
- Demo testing code against 10 compilers simultaneously
- Show cross-platform compatibility checking
- Demonstrate compiler bug detection across versions
- Quick demonstration of how this replaced "10 windows open"

### 5. Multi-File Projects & Libraries (5 minutes)
**Breaking the single-file myth**:
- CMake project support demonstration
- Show Conan.io library integration (1,000+ libraries)
- Upload multiple files for real project testing
- Include files from URLs with `#include`

### 6. URL Superpowers & Productivity (5 minutes)
**Features hiding in plain sight**:
- **Subdomain magic**: `https://myproject.godbolt.org/` for isolated state
- **Language shortcuts**: `rust.compiler-explorer.com`
- **Eternal links**: The 12,000 legacy links preservation story
- **Keyboard shortcuts**: Ctrl+S, Ctrl+D, Ctrl+Shift+Enter
- **Right-click assembly docs**: Instant instruction documentation

## Backend Deep Dive (10 minutes)

### Infrastructure Evolution
1. **SquashFS Revolution**: How mounting compilers "over the top" solved NFS latency
2. **Security with nsjail**: Google's sandboxing enabling safe execution
3. **Daily Compiler Builds**: Automated GitHub Actions maintaining trunk versions
4. **Cost Efficiency**: Running 92M compilations on $3,000/month

### Technical Architecture
- Multi-region AWS with auto-scaling
- Three-level caching: Browser → Instance LRU → S3
- CloudFront CDN distribution
- Custom URL shortener replacing Google's deprecated service
- Grafana monitoring and public stats dashboard

### Scaling Challenges Solved
- Windows compiler support via dedicated instances
- ARM64 native compilation
- GPU instance management for CUDA
- 1.8 million weekly compilations handling

## Demo Script Suggestions

### Opening Hook
Start with a simple C++ function, then progressively reveal features:
1. Basic assembly view (what everyone knows)
2. Click "Execution" - **surprise #1**: it runs!
3. Add Optimization Pipeline - **surprise #2**: see inside the compiler
4. Open Conformance View - **surprise #3**: test 10 compilers at once

### Crowd Participation
Ask audience: "Who knew CE could..."
- Run code? (expect ~30% hands)
- Show optimization passes? (expect ~5%)
- Handle CMake projects? (expect ~2%)
- Run CUDA on GPUs? (expect ~1%)

### Live Discoveries
Show features as if discovering them:
- "Let me right-click this assembly instruction... oh look, documentation!"
- "What if I use a random subdomain... independent workspace!"
- "Let's see what happens with Ctrl+D... advanced selection!"

## Key Statistics to Highlight
- **81+ languages** supported (up from ~5 in 2020)
- **92 million** annual compilations
- **3,000+** compiler versions maintained forever
- **1,000+** libraries available via Conan/vcpkg
- **12,000** legacy short links preserved
- **$3,000/month** infrastructure cost

## Closing Message
"Compiler Explorer started as a simple assembly viewer. Today, it's a comprehensive development platform that continues to surprise even its most dedicated users. The best feature of CE might just be the one you discover next. So go explore, experiment, and remember - every link you create will last forever, just like the compilers themselves."

## Alternative Title Options
1. "Compiler Explorer's Hidden Treasures: A 2025 Feature Safari"
2. "Beyond Assembly: The Compiler Explorer Features You're Missing"
3. "92 Million Compilations Later: What's New in Compiler Explorer"
4. "The Compiler Explorer Iceberg: What Lies Beneath"

## Talk Timing
- Introduction & Evolution: 5 minutes
- Feature Demonstrations: 35 minutes
- Backend Deep Dive: 10 minutes
- Q&A Buffer: 5 minutes
- **Total**: 45-minute slot + questions
