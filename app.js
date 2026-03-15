// === Deep Learning - Core Application ===
window.CHAPTERS = window.CHAPTERS || [];
window.EXTRA_VIZ = window.EXTRA_VIZ || {};

const App = {
    chapters: [],
    manifest: [],
    currentChapter: null,
    currentSectionIndex: 0,
    progress: {},
    _loadedChapters: {},
    _activeVizEngines: [],
    _loading: false,

    init() {
        this.manifest = (window.CHAPTER_MANIFEST || []).sort((a, b) => a.number - b.number);
        this.chapters = window.CHAPTERS.sort((a, b) => a.number - b.number);
        this.loadProgress();
        this.renderSidebar();
        this.bindEvents();
        this.checkHash();
    },

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = src;
            s.onload = resolve;
            s.onerror = reject;
            document.body.appendChild(s);
        });
    },

    async ensureChapterLoaded(chId) {
        if (this._loadedChapters[chId]) return;
        const entry = this.manifest.find(m => m.id === chId);
        if (!entry) return;
        const base = 'chapters/' + entry.file;
        await this.loadScript(base + '.js');
        try { await this.loadScript(base + '-viz.js'); } catch (e) { /* viz file optional */ }
        this._loadedChapters[chId] = true;
        this.chapters = window.CHAPTERS.sort((a, b) => a.number - b.number);
    },

    loadProgress() {
        try { this.progress = JSON.parse(localStorage.getItem('deep-learning-progress') || '{}'); }
        catch (e) { this.progress = {}; }
    },
    saveProgress() {
        try { localStorage.setItem('deep-learning-progress', JSON.stringify(this.progress)); } catch (e) {}
    },
    markSectionComplete(chId, idx) {
        if (!this.progress[chId]) this.progress[chId] = [];
        if (!this.progress[chId].includes(idx)) this.progress[chId].push(idx);
        this.saveProgress();
        this.updateProgressUI();
    },
    isSectionComplete(chId, idx) { return (this.progress[chId] || []).includes(idx); },
    getChapterProgress(ch) {
        const id = ch.id;
        const done = (this.progress[id] || []).length;
        const loaded = this.chapters.find(c => c.id === id);
        const total = loaded ? loaded.sections.length : (done || 5);
        return { done, total };
    },
    getTotalProgress() {
        let done = 0, total = 0;
        for (const m of this.manifest) {
            const { done: d, total: t } = this.getChapterProgress(m);
            done += d; total += t;
        }
        return { done, total };
    },
    resetProgress() { this.progress = {}; this.saveProgress(); this.updateProgressUI(); this.renderSidebar(); },

    renderSidebar() {
        const nav = document.getElementById('chapter-nav');
        nav.innerHTML = '';
        for (const m of this.manifest) {
            const item = document.createElement('div');
            item.className = 'chapter-item';
            if (this.currentChapter?.id === m.id) item.classList.add('active');
            const { done, total } = this.getChapterProgress(m);
            if (done === total && total > 0) item.classList.add('completed');
            item.innerHTML = '<div class="ch-num">Chapter ' + m.number + '</div><div class="ch-name">' + m.title + '</div><div class="ch-prog">' + done + '/' + total + ' sections</div>';
            item.addEventListener('click', () => this.loadChapter(m.id));
            nav.appendChild(item);
        }
        this.updateProgressUI();
    },

    updateProgressUI() {
        const { done, total } = this.getTotalProgress();
        const pct = total > 0 ? Math.round(done / total * 100) : 0;
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('progress-text').textContent = pct + '%';
        document.querySelectorAll('.section-tab').forEach((tab, i) => {
            if (this.currentChapter && this.isSectionComplete(this.currentChapter.id, i)) tab.classList.add('completed');
        });
        document.querySelectorAll('.chapter-item').forEach((item, i) => {
            if (i < this.manifest.length) {
                const m = this.manifest[i];
                const { done, total } = this.getChapterProgress(m);
                const p = item.querySelector('.ch-prog');
                if (p) p.textContent = done + '/' + total + ' sections';
                item.classList.toggle('completed', done === total && total > 0);
            }
        });
    },

    async loadChapter(chId) {
        if (this._loading) return;
        this._loading = true;
        document.getElementById('welcome-screen').style.display = 'none';
        document.getElementById('chapter-content').style.display = 'block';
        document.getElementById('section-content').innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:60px 0;">Loading chapter...</p>';
        document.getElementById('section-tabs').innerHTML = '';

        try {
            await this.ensureChapterLoaded(chId);
        } catch (e) {
            document.getElementById('section-content').innerHTML = '<p style="color:var(--accent-red);text-align:center;padding:60px 0;">Failed to load chapter: ' + e.message + '</p>';
            this._loading = false;
            return;
        }

        const ch = this.chapters.find(c => c.id === chId);
        if (!ch) {
            document.getElementById('section-content').innerHTML = '<p style="color:var(--accent-red);text-align:center;padding:60px 0;">Chapter data not found.</p>';
            this._loading = false;
            return;
        }

        this.currentChapter = ch;
        this.currentSectionIndex = 0;
        window.location.hash = chId;
        document.querySelectorAll('.chapter-item').forEach((item, i) => {
            item.classList.toggle('active', this.manifest[i]?.id === chId);
        });
        document.getElementById('chapter-number-badge').textContent = 'CHAPTER ' + ch.number;
        document.getElementById('chapter-title').textContent = ch.title;
        document.getElementById('chapter-subtitle').textContent = ch.subtitle || '';
        this.renderSectionTabs();
        this.loadSection(0);
        document.getElementById('main-content').scrollTop = 0;
        document.getElementById('sidebar').classList.remove('open');
        this._loading = false;
        this.renderSidebar();
    },

    renderSectionTabs() {
        const c = document.getElementById('section-tabs');
        c.innerHTML = '';
        if (!this.currentChapter) return;
        this.currentChapter.sections.forEach((sec, i) => {
            const tab = document.createElement('button');
            tab.className = 'section-tab';
            if (i === this.currentSectionIndex) tab.classList.add('active');
            if (this.isSectionComplete(this.currentChapter.id, i)) tab.classList.add('completed');
            tab.textContent = sec.title;
            tab.addEventListener('click', () => this.loadSection(i));
            c.appendChild(tab);
        });
    },

    escapeMathHtml(html) {
        if (!html) return '';
        return html
            .replace(/\\\([\s\S]*?\\\)/g, m => m.replace(/</g, '&lt;').replace(/>/g, '&gt;'))
            .replace(/\\\[[\s\S]*?\\\]/g, m => m.replace(/</g, '&lt;').replace(/>/g, '&gt;'));
    },

    loadSection(idx) {
        if (!this.currentChapter) return;
        if (idx < 0 || idx >= this.currentChapter.sections.length) return;
        if (this._activeVizEngines) {
            this._activeVizEngines.forEach(v => v.stopAnimation());
        }
        this._activeVizEngines = [];

        this.currentSectionIndex = idx;
        const sec = this.currentChapter.sections[idx];
        document.querySelectorAll('.section-tab').forEach((t, i) => t.classList.toggle('active', i === idx));

        const el = document.getElementById('section-content');
        el.innerHTML = this.escapeMathHtml(sec.content);
        el.className = 'fade-in';

        this.renderExercises(sec.exercises || []);
        this.renderMath(el);
        this.initVisualizations(sec.visualizations || []);

        const extraViz = window.EXTRA_VIZ?.[this.currentChapter.id]?.[sec.id];
        if (extraViz && extraViz.length > 0) {
            for (const viz of extraViz) {
                const ph = document.createElement('div');
                ph.className = 'viz-placeholder';
                ph.setAttribute('data-viz', viz.id);
                el.appendChild(ph);
            }
            this.initVisualizations(extraViz);
        }

        this.updateNavButtons();
        document.getElementById('section-indicator').textContent = (idx + 1) + ' / ' + this.currentChapter.sections.length;
        this.markSectionComplete(this.currentChapter.id, idx);
        document.getElementById('main-content').scrollTop = 0;
    },

    renderMath(el) {
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(el, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
                trust: true
            });
        }
    },

    initVisualizations(vizList) {
        document.querySelectorAll('.viz-placeholder').forEach(ph => {
            const vizId = ph.getAttribute('data-viz');
            const viz = vizList.find(v => v.id === vizId);
            if (!viz) return;
            const container = document.createElement('div');
            container.className = 'viz-container';
            const header = document.createElement('div');
            header.className = 'viz-header';
            const title = document.createElement('span');
            title.className = 'viz-title';
            title.innerHTML = viz.title || 'Interactive Visualization';
            if (window.renderMathInElement) renderMathInElement(title, {delimiters:[{left:'\\(',right:'\\)',display:false},{left:'\\[',right:'\\]',display:true}],throwOnError:false});
            header.appendChild(title);
            const controls = document.createElement('div');
            controls.className = 'viz-controls';
            header.appendChild(controls);
            const body = document.createElement('div');
            body.className = 'viz-body';
            container.appendChild(header);
            container.appendChild(body);
            if (viz.description) {
                const desc = document.createElement('div');
                desc.className = 'viz-description';
                desc.innerHTML = viz.description;
                if (window.renderMathInElement) renderMathInElement(desc, {delimiters:[{left:'\\(',right:'\\)',display:false},{left:'\\[',right:'\\]',display:true}],throwOnError:false});
                container.appendChild(desc);
            }
            ph.replaceWith(container);
            try {
                if (typeof viz.setup === 'function') {
                    const engine = viz.setup(body, controls);
                    if (engine && engine.stopAnimation) this._activeVizEngines.push(engine);
                }
            } catch (e) {
                console.error('Viz error:', vizId, e);
                body.innerHTML = '<p style="color:#f85149;">Visualization failed to load: ' + e.message + '</p>';
            }
        });
    },

    renderExercises(exercises) {
        const panel = document.getElementById('exercises-panel');
        const list = document.getElementById('exercises-list');
        if (!exercises || exercises.length === 0) { panel.style.display = 'none'; return; }
        panel.style.display = 'block';
        list.innerHTML = '';
        const freshList = list.cloneNode(false);
        list.parentNode.replaceChild(freshList, list);
        exercises.forEach((ex, i) => {
            const item = document.createElement('div');
            item.className = 'exercise-item';
            const q = document.createElement('div');
            q.className = 'exercise-question';
            q.innerHTML = '<div class="ex-num">Exercise ' + (i + 1) + '</div><div class="ex-text">' + this.escapeMathHtml(ex.question || '') + '</div>';
            item.appendChild(q);
            const toggle = document.createElement('div');
            toggle.className = 'exercise-toggle';
            if (ex.hint) {
                const hBtn = document.createElement('button');
                hBtn.textContent = 'Show Hint';
                hBtn.setAttribute('data-action', 'hint');
                hBtn.setAttribute('data-idx', i);
                toggle.appendChild(hBtn);
            }
            const sBtn = document.createElement('button');
            sBtn.textContent = 'Show Solution';
            sBtn.setAttribute('data-action', 'solution');
            sBtn.setAttribute('data-idx', i);
            toggle.appendChild(sBtn);
            item.appendChild(toggle);
            if (ex.hint) {
                const hDiv = document.createElement('div');
                hDiv.className = 'exercise-hint';
                hDiv.id = 'exhint-' + i;
                hDiv.innerHTML = this.escapeMathHtml(ex.hint);
                item.appendChild(hDiv);
            }
            const solDiv = document.createElement('div');
            solDiv.className = 'exercise-solution';
            solDiv.id = 'exsol-' + i;
            solDiv.innerHTML = this.escapeMathHtml(ex.solution || '');
            item.appendChild(solDiv);
            freshList.appendChild(item);
        });
        freshList.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-action]');
            if (!btn) return;
            const action = btn.getAttribute('data-action');
            const idx = btn.getAttribute('data-idx');
            const prefix = action === 'hint' ? 'exhint-' : 'exsol-';
            const el = document.getElementById(prefix + idx);
            if (el) {
                el.classList.toggle('show');
                const label = action.charAt(0).toUpperCase() + action.slice(1);
                btn.textContent = el.classList.contains('show') ? 'Hide ' + label : 'Show ' + label;
                if (el.classList.contains('show')) this.renderMath(el);
            }
        });
        this.renderMath(freshList);
    },

    updateNavButtons() {
        const prev = document.getElementById('prev-btn');
        const next = document.getElementById('next-btn');
        prev.disabled = this.currentSectionIndex === 0;
        const isLast = this.currentSectionIndex >= this.currentChapter.sections.length - 1;
        const nextM = this.manifest.find(m => m.number === this.currentChapter.number + 1);
        if (isLast && nextM) { next.textContent = 'Next: ' + nextM.title + ' \u2192'; next.disabled = false; }
        else if (isLast) { next.textContent = 'Course Complete!'; next.disabled = true; }
        else { next.textContent = 'Next \u2192'; next.disabled = false; }
    },
    nextSection() {
        if (!this.currentChapter) return;
        if (this.currentSectionIndex < this.currentChapter.sections.length - 1) this.loadSection(this.currentSectionIndex + 1);
        else { const nm = this.manifest.find(m => m.number === this.currentChapter.number + 1); if (nm) this.loadChapter(nm.id); }
    },
    prevSection() {
        if (!this.currentChapter || this.currentSectionIndex <= 0) return;
        this.loadSection(this.currentSectionIndex - 1);
    },

    bindEvents() {
        document.getElementById('prev-btn').addEventListener('click', () => this.prevSection());
        document.getElementById('next-btn').addEventListener('click', () => this.nextSection());
        document.getElementById('reset-progress').addEventListener('click', () => this.resetProgress());
        document.getElementById('sidebar-toggle').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
        document.getElementById('search-input').addEventListener('input', (e) => this.filterChapters(e.target.value));
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowRight' || e.key === 'n') this.nextSection();
            if (e.key === 'ArrowLeft' || e.key === 'p') this.prevSection();
        });
    },
    filterChapters(q) {
        q = q.toLowerCase();
        document.querySelectorAll('.chapter-item').forEach((item, i) => {
            if (i < this.manifest.length) {
                const m = this.manifest[i];
                const match = !q || m.title.toLowerCase().includes(q);
                item.style.display = match ? '' : 'none';
            }
        });
    },
    checkHash() {
        const h = window.location.hash.slice(1);
        if (h) this.loadChapter(h);
    }
};

document.addEventListener('DOMContentLoaded', () => App.init());
