// === Chapter 12: Recurrent Neural Networks ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch12',
    number: 12,
    title: 'Recurrent Neural Networks',
    subtitle: 'Sequence modeling, hidden state dynamics, backpropagation through time, and bidirectional architectures',
    sections: [
        // ========== SECTION 1: Sequence Modeling ==========
        {
            id: 'sec01-sequence-modeling',
            title: 'Sequence Modeling',
            content: `
<h2>12.1 Sequence Modeling</h2>

<div class="env-block intuition">
<strong>Why Sequences Need Special Architectures.</strong> In Chapters 4 and 9 we built feedforward networks (MLPs) and convolutional networks (CNNs). Both assume that each input is independent of every other: a feedforward net classifying an image has no notion that yesterday's image might influence today's prediction. But many real-world problems involve <em>sequential</em> data, where order matters and the past influences the future. Language, speech, music, stock prices, sensor readings, DNA sequences: in all of these, the meaning of each element depends on what came before.
</div>

<p>A feedforward network receiving a sentence one word at a time would treat each word identically, with no memory of previous words. The word "bank" means something very different after "river" than after "investment." To capture such dependencies, we need an architecture that maintains a <em>state</em> across time steps, accumulating information from past inputs and using it to interpret the present.</p>

<h3>Types of Sequential Data</h3>

<p>Sequential data appears throughout machine learning:</p>

<ul>
<li><strong>Natural language.</strong> A sentence is a sequence of tokens. The probability of each word depends on all preceding words: \\(P(w_t \\mid w_1, w_2, \\ldots, w_{t-1})\\).</li>
<li><strong>Time series.</strong> Stock prices, temperature readings, and heart-rate monitors produce observations indexed by time: \\(x_1, x_2, \\ldots, x_T\\).</li>
<li><strong>Audio and speech.</strong> A speech waveform sampled at 16 kHz produces 16,000 values per second, each depending on the phonetic and linguistic context established by preceding samples.</li>
<li><strong>Genomics.</strong> DNA is a sequence of nucleotides (A, C, G, T) where long-range patterns encode biological function.</li>
</ul>

<h3>The Core Challenge: Variable-Length Dependencies</h3>

<p>The fundamental difficulty is that sequences can be arbitrarily long, and relevant information can appear at any distance in the past. A naive approach, concatenating all past inputs into a single fixed-size vector, fails for two reasons:</p>

<ol>
<li><strong>Variable length.</strong> Sentences have different numbers of words; time series have different durations. A fixed-dimension input layer cannot accommodate all of them.</li>
<li><strong>Combinatorial explosion.</strong> If each position can take \\(V\\) values and the sequence has length \\(T\\), the input space has \\(V^T\\) possibilities, growing exponentially.</li>
</ol>

<div class="env-block definition">
<strong>Definition 12.1.1 (Sequence-to-Sequence Mapping).</strong> A sequence model is a function \\(f\\) that maps an input sequence \\((x_1, x_2, \\ldots, x_T)\\) to an output sequence \\((y_1, y_2, \\ldots, y_{T'})\\), where \\(T\\) and \\(T'\\) may differ. Special cases include:
<ul>
<li><strong>Many-to-one:</strong> \\(T' = 1\\) (e.g., sentiment classification of a sentence).</li>
<li><strong>One-to-many:</strong> \\(T = 1\\) (e.g., image captioning).</li>
<li><strong>Many-to-many:</strong> \\(T' = T\\) (e.g., part-of-speech tagging) or \\(T' \\neq T\\) (e.g., machine translation).</li>
</ul>
</div>

<h3>The Hidden State Idea</h3>

<p>The key insight behind recurrent neural networks is to compress all relevant information from the past into a fixed-dimensional <em>hidden state</em> vector \\(\\mathbf{h}_t \\in \\mathbb{R}^d\\). At each time step \\(t\\), the network reads the current input \\(\\mathbf{x}_t\\) and the previous hidden state \\(\\mathbf{h}_{t-1}\\), then produces an updated state:</p>

\\[
\\mathbf{h}_t = \\phi(\\mathbf{h}_{t-1}, \\mathbf{x}_t)
\\]

<p>for some learned function \\(\\phi\\). The hidden state acts as the network's "memory," summarizing all inputs seen so far. We initialize with \\(\\mathbf{h}_0 = \\mathbf{0}\\) (or a learned initial state). The output at time \\(t\\) is then a function of \\(\\mathbf{h}_t\\).</p>

<div class="env-block remark">
<strong>Parameter Sharing.</strong> A crucial property is that the same function \\(\\phi\\) (with the same parameters) is applied at every time step. This <em>parameter sharing</em> across time is analogous to how a CNN shares convolutional filters across spatial positions. It allows the model to generalize to sequences of any length and reduces the number of parameters compared to having a separate transformation for each time step.
</div>

<div class="viz-placeholder" data-viz="viz-sequence-processing"></div>

<p>The visualization above demonstrates the core mechanism: as each word in a sentence is processed, the hidden state vector updates to incorporate new information. Early dimensions might encode syntactic structure, while others track semantic content. By the final time step, the hidden state encodes a summary of the entire sequence.</p>
`,
            visualizations: [
                {
                    id: 'viz-sequence-processing',
                    title: 'Sequence Processing: Hidden State Evolution',
                    description: 'Watch a sentence being processed word by word. The bar chart shows the hidden state vector updating at each step. Press Play to animate, or use the slider to step manually.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 700, height: 420 });
                        const ctx = viz.ctx;

                        const words = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
                        const T = words.length;
                        const hiddenDim = 8;

                        // Pre-compute hidden states (simulated with pseudo-random deterministic values)
                        const hiddenStates = [new Array(hiddenDim).fill(0)];
                        // Simulated weight matrices for realistic evolution
                        const seed = 42;
                        function seededRand(s) { let x = Math.sin(s) * 10000; return x - Math.floor(x); }
                        for (let t = 0; t < T; t++) {
                            const prev = hiddenStates[t];
                            const h = [];
                            for (let d = 0; d < hiddenDim; d++) {
                                const r1 = seededRand(seed + t * hiddenDim + d) * 2 - 1;
                                const r2 = seededRand(seed + 100 + t * hiddenDim + d) * 0.6;
                                const val = Math.tanh(prev[d] * 0.5 + r1 * 0.8 + r2);
                                h.push(val);
                            }
                            hiddenStates.push(h);
                        }

                        let currentStep = 0;
                        let animFrame = 0;
                        let playing = false;
                        let animInterval = null;

                        const playBtn = VizEngine.createButton(controls, 'Play', () => {
                            if (playing) {
                                playing = false;
                                playBtn.textContent = 'Play';
                                if (animInterval) { clearInterval(animInterval); animInterval = null; }
                            } else {
                                playing = true;
                                playBtn.textContent = 'Pause';
                                currentStep = 0;
                                animFrame = 0;
                                animInterval = setInterval(() => {
                                    if (currentStep < T) {
                                        currentStep++;
                                        draw();
                                    } else {
                                        playing = false;
                                        playBtn.textContent = 'Play';
                                        clearInterval(animInterval);
                                        animInterval = null;
                                    }
                                }, 800);
                            }
                        });

                        VizEngine.createButton(controls, 'Reset', () => {
                            playing = false;
                            playBtn.textContent = 'Play';
                            if (animInterval) { clearInterval(animInterval); animInterval = null; }
                            currentStep = 0;
                            draw();
                        });

                        const stepSlider = VizEngine.createSlider(controls, 'Step', 0, T, 0, 1, val => {
                            currentStep = Math.round(val);
                            playing = false;
                            playBtn.textContent = 'Play';
                            if (animInterval) { clearInterval(animInterval); animInterval = null; }
                            draw();
                        });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            stepSlider.value = currentStep;
                            stepSlider.nextElementSibling.textContent = currentStep;

                            const leftMargin = 50;
                            const wordY = 60;
                            const wordSpacing = 90;
                            const barAreaTop = 140;
                            const barAreaBottom = 380;
                            const barMaxH = (barAreaBottom - barAreaTop) / 2;
                            const barCenterY = (barAreaTop + barAreaBottom) / 2;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Input Sequence', viz.width / 2, 20);

                            // Draw words
                            ctx.font = '14px -apple-system,sans-serif';
                            for (let t = 0; t < T; t++) {
                                const wx = leftMargin + t * wordSpacing + wordSpacing / 2;

                                // Word box
                                const isProcessed = t < currentStep;
                                const isCurrent = t === currentStep - 1;
                                ctx.fillStyle = isCurrent ? viz.colors.blue + '44' : (isProcessed ? viz.colors.teal + '22' : viz.colors.grid);
                                ctx.strokeStyle = isCurrent ? viz.colors.blue : (isProcessed ? viz.colors.teal : viz.colors.axis);
                                ctx.lineWidth = isCurrent ? 2 : 1;
                                const bw = 70, bh = 30;
                                ctx.beginPath();
                                ctx.roundRect(wx - bw / 2, wordY - bh / 2, bw, bh, 6);
                                ctx.fill();
                                ctx.stroke();

                                ctx.fillStyle = isProcessed ? viz.colors.white : viz.colors.text;
                                ctx.textAlign = 'center';
                                ctx.fillText(words[t], wx, wordY);

                                // Arrow down from current word
                                if (isCurrent) {
                                    ctx.strokeStyle = viz.colors.blue;
                                    ctx.lineWidth = 2;
                                    ctx.beginPath();
                                    ctx.moveTo(wx, wordY + bh / 2 + 2);
                                    ctx.lineTo(wx, barAreaTop - 20);
                                    ctx.stroke();
                                    // arrowhead
                                    ctx.fillStyle = viz.colors.blue;
                                    ctx.beginPath();
                                    ctx.moveTo(wx, barAreaTop - 12);
                                    ctx.lineTo(wx - 5, barAreaTop - 22);
                                    ctx.lineTo(wx + 5, barAreaTop - 22);
                                    ctx.closePath();
                                    ctx.fill();
                                }
                            }

                            // Draw hidden state bar chart
                            const h = hiddenStates[currentStep];
                            const barAreaLeft = 80;
                            const barAreaRight = viz.width - 40;
                            const totalBarWidth = barAreaRight - barAreaLeft;
                            const barWidth = totalBarWidth / hiddenDim - 4;
                            const gap = 4;

                            // Label
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Hidden State h' + (currentStep > 0 ? '\u2080'.replace('0', String.fromCharCode(8320 + currentStep)) : '\u2080') + '  (dim=' + hiddenDim + ')', viz.width / 2, barAreaTop - 5);

                            // Zero line
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(barAreaLeft - 10, barCenterY);
                            ctx.lineTo(barAreaRight + 10, barCenterY);
                            ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.fillText('+1', barAreaLeft - 14, barAreaTop + 10);
                            ctx.fillText('0', barAreaLeft - 14, barCenterY);
                            ctx.fillText('-1', barAreaLeft - 14, barAreaBottom - 5);

                            // Draw bars
                            for (let d = 0; d < hiddenDim; d++) {
                                const bx = barAreaLeft + d * (barWidth + gap);
                                const val = h[d];
                                const barH = val * barMaxH;

                                // Color: positive = teal, negative = orange
                                const color = val >= 0 ? viz.colors.teal : viz.colors.orange;
                                ctx.fillStyle = color + 'cc';
                                if (barH >= 0) {
                                    ctx.fillRect(bx, barCenterY - barH, barWidth, barH);
                                } else {
                                    ctx.fillRect(bx, barCenterY, barWidth, -barH);
                                }

                                // Border
                                ctx.strokeStyle = color;
                                ctx.lineWidth = 1;
                                if (barH >= 0) {
                                    ctx.strokeRect(bx, barCenterY - barH, barWidth, barH);
                                } else {
                                    ctx.strokeRect(bx, barCenterY, barWidth, -barH);
                                }

                                // Dimension label
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('d' + d, bx + barWidth / 2, barAreaBottom + 12);

                                // Value label
                                ctx.fillStyle = color;
                                ctx.font = '9px -apple-system,sans-serif';
                                const labelY = val >= 0 ? barCenterY - barH - 8 : barCenterY - barH + 10;
                                ctx.fillText(val.toFixed(2), bx + barWidth / 2, labelY);
                            }

                            // Status text
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            if (currentStep === 0) {
                                ctx.fillText('Initial state h\u2080 = 0. Press Play or advance the Step slider.', viz.width / 2, viz.height - 10);
                            } else {
                                ctx.fillText('After processing "' + words[currentStep - 1] + '" (step ' + currentStep + '/' + T + ')', viz.width / 2, viz.height - 10);
                            }
                        }

                        draw();
                        return { stopAnimation() { if (animInterval) clearInterval(animInterval); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Explain why a standard feedforward network (MLP) cannot naturally handle variable-length input sequences. What architectural constraint prevents it?',
                    hint: 'Think about the input layer of an MLP. What happens if you train on sentences of length 10 and then encounter a sentence of length 15?',
                    solution: 'An MLP has a fixed-size input layer with a predetermined number of neurons. This means the input dimension is fixed at architecture design time. If trained on sequences of length \\(T\\), it cannot accept sequences of length \\(T\' \\neq T\\) without padding, truncation, or redesign. Furthermore, even with padding to a maximum length, the MLP treats each position independently and cannot share learned patterns across positions (unlike an RNN, which applies the same transformation at every step). This fixed-dimensionality constraint makes MLPs unsuitable for naturally variable-length sequential data.'
                },
                {
                    question: 'Consider a vocabulary of size \\(V = 10{,}000\\) and a maximum sequence length \\(T = 50\\). If we tried to enumerate all possible sequences, how many would there be? Why does this make a lookup-table approach infeasible?',
                    hint: 'Each of the \\(T\\) positions can independently take any of \\(V\\) values.',
                    solution: 'The number of possible sequences is \\(V^T = 10{,}000^{50} = 10^{200}\\). This is astronomically larger than the number of atoms in the observable universe (approximately \\(10^{80}\\)). A lookup table storing one output per sequence would require \\(10^{200}\\) entries, which is physically impossible. This combinatorial explosion is precisely why we need models that <em>generalize</em> from a tractable number of training examples through parameter sharing and compositional computation, rather than memorizing input-output pairs.'
                },
                {
                    question: 'Suppose the hidden state has dimension \\(d = 256\\). At each time step, the RNN computes \\(\\mathbf{h}_t = \\tanh(\\mathbf{W}_h \\mathbf{h}_{t-1} + \\mathbf{W}_x \\mathbf{x}_t + \\mathbf{b})\\) where \\(\\mathbf{x}_t \\in \\mathbb{R}^{128}\\). How many parameters does this recurrence use (excluding any output layer)?',
                    hint: 'Count the dimensions of \\(\\mathbf{W}_h\\), \\(\\mathbf{W}_x\\), and \\(\\mathbf{b}\\).',
                    solution: '\\(\\mathbf{W}_h \\in \\mathbb{R}^{256 \\times 256}\\) has \\(256^2 = 65{,}536\\) parameters. \\(\\mathbf{W}_x \\in \\mathbb{R}^{256 \\times 128}\\) has \\(256 \\times 128 = 32{,}768\\) parameters. \\(\\mathbf{b} \\in \\mathbb{R}^{256}\\) has \\(256\\) parameters. Total: \\(65{,}536 + 32{,}768 + 256 = 98{,}560\\) parameters. Crucially, this count is <em>independent</em> of the sequence length \\(T\\), because the same weights are shared across all time steps.'
                },
                {
                    question: 'Name two real-world sequence modeling tasks that are (a) many-to-one and (b) many-to-many with \\(T\' \\neq T\\). For each, describe the input sequence and the output.',
                    hint: 'Think about NLP tasks: what takes a full sequence and produces a single label? What takes a sequence in one language and produces a sequence in another?',
                    solution: '(a) <strong>Many-to-one: Sentiment analysis.</strong> The input is a sequence of word embeddings representing a product review (variable length). The output is a single label: positive, negative, or neutral. The RNN reads the entire review and the final hidden state \\(\\mathbf{h}_T\\) is passed to a classifier.<br>(b) <strong>Many-to-many with \\(T\' \\neq T\\): Machine translation.</strong> The input is a sentence in English (say, 8 words) and the output is its French translation (perhaps 11 words). An encoder RNN reads the source sentence into a context vector, and a decoder RNN generates the target sentence token by token. The input and output lengths generally differ.'
                }
            ]
        },

        // ========== SECTION 2: RNN Architecture ==========
        {
            id: 'sec02-rnn-architecture',
            title: 'RNN Architecture',
            content: `
<h2>12.2 RNN Architecture</h2>

<p>We now formalize the recurrent neural network. The central idea is a single computational cell that is applied repeatedly at each time step, using the same parameters. We can visualize this in two equivalent ways: as a <em>folded</em> diagram with a self-loop, or as an <em>unrolled</em> chain across time.</p>

<h3>The Elman RNN</h3>

<p>The simplest and most common RNN architecture is the <strong>Elman network</strong> (1990). Given an input sequence \\(\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x}_T\\) with \\(\\mathbf{x}_t \\in \\mathbb{R}^n\\), the hidden state \\(\\mathbf{h}_t \\in \\mathbb{R}^d\\) and output \\(\\mathbf{y}_t \\in \\mathbb{R}^m\\) are computed as:</p>

<div class="env-block definition">
<strong>Definition 12.2.1 (Elman RNN).</strong> The recurrence relation is
\\[
\\mathbf{h}_t = \\tanh\\!\\left(\\mathbf{W}_h \\mathbf{h}_{t-1} + \\mathbf{W}_x \\mathbf{x}_t + \\mathbf{b}_h\\right), \\qquad t = 1, 2, \\ldots, T,
\\]
with initial state \\(\\mathbf{h}_0 = \\mathbf{0}\\) (or a learned vector). The output at each step is
\\[
\\mathbf{y}_t = \\mathbf{W}_y \\mathbf{h}_t + \\mathbf{b}_y,
\\]
where \\(\\mathbf{W}_h \\in \\mathbb{R}^{d \\times d}\\), \\(\\mathbf{W}_x \\in \\mathbb{R}^{d \\times n}\\), \\(\\mathbf{b}_h \\in \\mathbb{R}^d\\), \\(\\mathbf{W}_y \\in \\mathbb{R}^{m \\times d}\\), \\(\\mathbf{b}_y \\in \\mathbb{R}^m\\).
</div>

<div class="env-block remark">
<strong>Why \\(\\tanh\\)?</strong> The hyperbolic tangent activation squashes the hidden state into \\((-1, 1)^d\\), preventing unbounded growth. Historically, \\(\\tanh\\) was preferred over the logistic sigmoid \\(\\sigma\\) because its output is centered at zero, leading to better gradient flow. However, as we will see in Section 12.3, even \\(\\tanh\\) does not fully solve the vanishing gradient problem.
</div>

<h3>Folded vs. Unrolled Views</h3>

<p>The <strong>folded</strong> view draws a single cell with a self-loop arrow representing the recurrence \\(\\mathbf{h}_{t-1} \\to \\mathbf{h}_t\\). This is compact but hides the temporal structure.</p>

<p>The <strong>unrolled</strong> (or "unfolded") view replicates the cell for each time step, making the temporal computation graph explicit. Unrolling is essential for understanding backpropagation through time (Section 12.3), because it reveals that an RNN over \\(T\\) steps is equivalent to a very deep feedforward network with \\(T\\) layers, all sharing the same weights.</p>

<div class="env-block theorem">
<strong>Proposition 12.2.2 (Depth-Width Equivalence).</strong> An RNN unrolled for \\(T\\) time steps is computationally equivalent to a feedforward network of depth \\(T\\) where:
<ul>
<li>Each layer \\(t\\) has \\(d\\) hidden units.</li>
<li>All layers share the same weight matrices \\(\\mathbf{W}_h\\), \\(\\mathbf{W}_x\\), and bias \\(\\mathbf{b}_h\\).</li>
<li>Layer \\(t\\) receives an additional "skip input" \\(\\mathbf{x}_t\\) from the input sequence.</li>
</ul>
This weight-sharing constraint is what distinguishes an RNN from a generic deep network. It reduces the parameter count from \\(\\mathcal{O}(T \\cdot d^2)\\) to \\(\\mathcal{O}(d^2)\\), independent of sequence length.
</div>

<h3>Compact Notation</h3>

<p>It is often convenient to combine the hidden-to-hidden and input-to-hidden transformations. Define the concatenation \\(\\mathbf{z}_t = [\\mathbf{h}_{t-1}; \\mathbf{x}_t] \\in \\mathbb{R}^{d+n}\\) and the combined weight matrix \\(\\mathbf{W} = [\\mathbf{W}_h \\mid \\mathbf{W}_x] \\in \\mathbb{R}^{d \\times (d+n)}\\). Then</p>
\\[
\\mathbf{h}_t = \\tanh(\\mathbf{W} \\mathbf{z}_t + \\mathbf{b}_h).
\\]
<p>This formulation makes the RNN cell look like a single affine transformation followed by a nonlinearity, simplifying both implementation and analysis.</p>

<div class="viz-placeholder" data-viz="viz-rnn-unrolling"></div>

<h3>Stacking RNN Layers</h3>

<p>A single-layer RNN may lack the representational capacity for complex tasks. We can stack multiple RNN layers by feeding the hidden state sequence of one layer as the input sequence to the next:</p>
\\[
\\mathbf{h}_t^{(\\ell)} = \\tanh\\!\\left(\\mathbf{W}_h^{(\\ell)} \\mathbf{h}_{t-1}^{(\\ell)} + \\mathbf{W}_x^{(\\ell)} \\mathbf{h}_t^{(\\ell-1)} + \\mathbf{b}^{(\\ell)}\\right),
\\]
<p>where \\(\\mathbf{h}_t^{(0)} = \\mathbf{x}_t\\). Each layer has its own set of weights. Stacking \\(L\\) layers yields a network that is deep in both time (\\(T\\) steps) and space (\\(L\\) layers).</p>
`,
            visualizations: [
                {
                    id: 'viz-rnn-unrolling',
                    title: 'RNN Unrolling: From Folded to Unfolded',
                    description: 'Click "Unroll" to animate the transition from the compact folded RNN (with self-loop) to the unrolled computation graph across time steps. Use the slider to control the number of time steps.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 750, height: 400 });
                        const ctx = viz.ctx;

                        let unrolled = false;
                        let animProgress = 0; // 0 = folded, 1 = fully unrolled
                        let numSteps = 4;
                        let animating = false;

                        const unrollBtn = VizEngine.createButton(controls, 'Unroll', () => {
                            if (animating) return;
                            unrolled = !unrolled;
                            unrollBtn.textContent = unrolled ? 'Fold' : 'Unroll';
                            animating = true;
                        });

                        VizEngine.createSlider(controls, 'Steps', 2, 6, numSteps, 1, val => {
                            numSteps = Math.round(val);
                            if (!animating) draw();
                        });

                        function drawRoundedRect(x, y, w, h, r) {
                            ctx.beginPath();
                            ctx.roundRect(x, y, w, h, r);
                        }

                        function drawArrow(x1, y1, x2, y2, color, lw) {
                            const dx = x2 - x1, dy = y2 - y1;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            if (len < 1) return;
                            const angle = Math.atan2(dy, dx);
                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 2;
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2 - Math.cos(angle) * 8, y2 - Math.sin(angle) * 8);
                            ctx.stroke();
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 10 * Math.cos(angle - Math.PI / 6), y2 - 10 * Math.sin(angle - Math.PI / 6));
                            ctx.lineTo(x2 - 10 * Math.cos(angle + Math.PI / 6), y2 - 10 * Math.sin(angle + Math.PI / 6));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function drawCurvedArrow(cx, cy, r, startAngle, endAngle, color) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.arc(cx, cy, r, startAngle, endAngle);
                            ctx.stroke();
                            // arrowhead at end
                            const ax = cx + r * Math.cos(endAngle);
                            const ay = cy + r * Math.sin(endAngle);
                            const tangent = endAngle + Math.PI / 2;
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(ax, ay);
                            ctx.lineTo(ax - 8 * Math.cos(tangent - Math.PI / 5), ay - 8 * Math.sin(tangent - Math.PI / 5));
                            ctx.lineTo(ax - 8 * Math.cos(tangent + Math.PI / 5), ay - 8 * Math.sin(tangent + Math.PI / 5));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            const t = animProgress;
                            const cellW = 70, cellH = 50;
                            const centerY = 200;

                            if (t < 0.05) {
                                // ---- FOLDED VIEW ----
                                const cx = viz.width / 2, cy = centerY;

                                // Self-loop
                                drawCurvedArrow(cx, cy - cellH / 2 - 25, 25, Math.PI * 0.8, Math.PI * 0.1, viz.colors.purple);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('W_h', cx, cy - cellH / 2 - 55);

                                // Cell
                                ctx.fillStyle = viz.colors.blue + '33';
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 2;
                                drawRoundedRect(cx - cellW / 2, cy - cellH / 2, cellW, cellH, 8);
                                ctx.fill(); ctx.stroke();
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 14px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('tanh', cx, cy);

                                // Input arrow from below
                                drawArrow(cx, cy + cellH / 2 + 60, cx, cy + cellH / 2 + 2, viz.colors.teal, 2);
                                ctx.fillStyle = viz.colors.teal;
                                ctx.font = '13px -apple-system,sans-serif';
                                ctx.fillText('x_t', cx, cy + cellH / 2 + 75);

                                // Output arrow upward
                                drawArrow(cx, cy - cellH / 2 - 2, cx, cy - cellH / 2 - 60, viz.colors.orange, 2);
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('y_t', cx, cy - cellH / 2 - 75);

                                // Label
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.fillText('Folded RNN Cell', cx, viz.height - 25);
                                ctx.fillText('h_{t-1} \u2192 h_t', cx + 55, cy - cellH / 2 - 25);

                            } else {
                                // ---- UNROLLED VIEW (or transitioning) ----
                                const spacing = Math.min(140, (viz.width - 80) / numSteps);
                                const startX = (viz.width - (numSteps - 1) * spacing) / 2;

                                for (let i = 0; i < numSteps; i++) {
                                    const cx = startX + i * spacing;
                                    const cy = centerY;

                                    // Cell
                                    const alpha = Math.min(1, t * 2);
                                    ctx.fillStyle = viz.colors.blue + '33';
                                    ctx.strokeStyle = viz.colors.blue;
                                    ctx.lineWidth = 2;
                                    drawRoundedRect(cx - cellW / 2, cy - cellH / 2, cellW, cellH, 8);
                                    ctx.fill(); ctx.stroke();
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = 'bold 13px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText('tanh', cx, cy);

                                    // Input arrow from below
                                    drawArrow(cx, cy + cellH / 2 + 55, cx, cy + cellH / 2 + 2, viz.colors.teal, 2);
                                    ctx.fillStyle = viz.colors.teal;
                                    ctx.font = '12px -apple-system,sans-serif';
                                    ctx.fillText('x\u2080'.slice(0, 1) + String.fromCharCode(8321 + i), cx, cy + cellH / 2 + 70);

                                    // Output arrow upward
                                    drawArrow(cx, cy - cellH / 2 - 2, cx, cy - cellH / 2 - 50, viz.colors.orange, 2);
                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.fillText('y\u2080'.slice(0, 1) + String.fromCharCode(8321 + i), cx, cy - cellH / 2 - 62);

                                    // Hidden state arrow to next cell
                                    if (i < numSteps - 1) {
                                        const nextCx = startX + (i + 1) * spacing;
                                        drawArrow(cx + cellW / 2 + 2, cy, nextCx - cellW / 2 - 2, cy, viz.colors.purple, 2);

                                        // Label h_t
                                        ctx.fillStyle = viz.colors.purple;
                                        ctx.font = '11px -apple-system,sans-serif';
                                        ctx.fillText('h' + String.fromCharCode(8321 + i), (cx + nextCx) / 2, cy - 14);
                                    }

                                    // Initial h_0 arrow
                                    if (i === 0) {
                                        drawArrow(cx - cellW / 2 - 40, cy, cx - cellW / 2 - 2, cy, viz.colors.purple, 2);
                                        ctx.fillStyle = viz.colors.purple;
                                        ctx.font = '11px -apple-system,sans-serif';
                                        ctx.textAlign = 'right';
                                        ctx.fillText('h\u2080', cx - cellW / 2 - 44, cy - 2);
                                    }
                                }

                                // Weight sharing annotation
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Unrolled RNN: same W_h, W_x, b shared at every step', viz.width / 2, viz.height - 25);

                                // Dashed bracket under cells
                                ctx.strokeStyle = viz.colors.yellow + '66';
                                ctx.lineWidth = 1;
                                ctx.setLineDash([4, 3]);
                                const bracketY = centerY + cellH / 2 + 4;
                                ctx.beginPath();
                                ctx.moveTo(startX - cellW / 2, bracketY);
                                ctx.lineTo(startX + (numSteps - 1) * spacing + cellW / 2, bracketY);
                                ctx.stroke();
                                ctx.setLineDash([]);

                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('shared parameters', viz.width / 2, bracketY + 12);
                            }
                        }

                        viz.animate((time) => {
                            if (animating) {
                                if (unrolled) {
                                    animProgress = Math.min(1, animProgress + 0.04);
                                    if (animProgress >= 1) animating = false;
                                } else {
                                    animProgress = Math.max(0, animProgress - 0.04);
                                    if (animProgress <= 0) animating = false;
                                }
                            }
                            draw();
                        });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Write out the full computation for an Elman RNN processing a sequence of length \\(T = 3\\), with \\(\\mathbf{h}_0 = \\mathbf{0}\\). Express \\(\\mathbf{h}_3\\) as a nested composition of transformations applied to the inputs \\(\\mathbf{x}_1, \\mathbf{x}_2, \\mathbf{x}_3\\).',
                    hint: 'Start with \\(\\mathbf{h}_1 = \\tanh(\\mathbf{W}_h \\cdot \\mathbf{0} + \\mathbf{W}_x \\mathbf{x}_1 + \\mathbf{b})\\) and substitute recursively.',
                    solution: 'Step by step: \\(\\mathbf{h}_1 = \\tanh(\\mathbf{W}_x \\mathbf{x}_1 + \\mathbf{b})\\), since \\(\\mathbf{W}_h \\mathbf{0} = \\mathbf{0}\\). Then \\(\\mathbf{h}_2 = \\tanh(\\mathbf{W}_h \\tanh(\\mathbf{W}_x \\mathbf{x}_1 + \\mathbf{b}) + \\mathbf{W}_x \\mathbf{x}_2 + \\mathbf{b})\\). Finally: \\[\\mathbf{h}_3 = \\tanh\\!\\Big(\\mathbf{W}_h \\tanh\\!\\big(\\mathbf{W}_h \\tanh(\\mathbf{W}_x \\mathbf{x}_1 + \\mathbf{b}) + \\mathbf{W}_x \\mathbf{x}_2 + \\mathbf{b}\\big) + \\mathbf{W}_x \\mathbf{x}_3 + \\mathbf{b}\\Big).\\] This nested structure shows that \\(\\mathbf{h}_3\\) depends on all three inputs, with earlier inputs passing through more layers of nonlinear transformation.'
                },
                {
                    question: 'Consider a stacked RNN with \\(L = 2\\) layers, hidden dimension \\(d = 128\\), and input dimension \\(n = 64\\). How many parameters are in the recurrence (excluding output layer)? Compare to a single-layer RNN with \\(d = 256\\).',
                    hint: 'Layer 1 has input dim \\(n\\) and hidden dim \\(d\\). Layer 2 has input dim \\(d\\) (the hidden states of layer 1) and hidden dim \\(d\\).',
                    solution: '<strong>Stacked 2-layer RNN (\\(d=128\\)):</strong> Layer 1: \\(\\mathbf{W}_h^{(1)} \\in \\mathbb{R}^{128 \\times 128}\\) (16,384) + \\(\\mathbf{W}_x^{(1)} \\in \\mathbb{R}^{128 \\times 64}\\) (8,192) + \\(\\mathbf{b}^{(1)}\\) (128) = 24,704. Layer 2: \\(\\mathbf{W}_h^{(2)} \\in \\mathbb{R}^{128 \\times 128}\\) (16,384) + \\(\\mathbf{W}_x^{(2)} \\in \\mathbb{R}^{128 \\times 128}\\) (16,384) + \\(\\mathbf{b}^{(2)}\\) (128) = 32,896. Total: 57,600.<br><strong>Single-layer RNN (\\(d=256\\)):</strong> \\(256^2 + 256 \\times 64 + 256 = 65,536 + 16,384 + 256 = 82,176\\).<br>The stacked RNN uses fewer parameters (57,600 vs. 82,176) but introduces hierarchical feature extraction, which can be more expressive per parameter.'
                },
                {
                    question: 'Show that the compact notation \\(\\mathbf{h}_t = \\tanh(\\mathbf{W}[\\mathbf{h}_{t-1}; \\mathbf{x}_t] + \\mathbf{b})\\) with \\(\\mathbf{W} = [\\mathbf{W}_h \\mid \\mathbf{W}_x]\\) is mathematically equivalent to the standard formulation \\(\\mathbf{h}_t = \\tanh(\\mathbf{W}_h \\mathbf{h}_{t-1} + \\mathbf{W}_x \\mathbf{x}_t + \\mathbf{b})\\).',
                    hint: 'Write out the matrix-vector product \\(\\mathbf{W}[\\mathbf{h}_{t-1}; \\mathbf{x}_t]\\) using the block structure of \\(\\mathbf{W}\\).',
                    solution: 'Let \\(\\mathbf{W} = [\\mathbf{W}_h \\mid \\mathbf{W}_x] \\in \\mathbb{R}^{d \\times (d+n)}\\) and \\(\\mathbf{z}_t = [\\mathbf{h}_{t-1}; \\mathbf{x}_t] \\in \\mathbb{R}^{d+n}\\) (vertical concatenation). Then \\(\\mathbf{W} \\mathbf{z}_t = [\\mathbf{W}_h \\mid \\mathbf{W}_x] \\begin{bmatrix} \\mathbf{h}_{t-1} \\\\ \\mathbf{x}_t \\end{bmatrix} = \\mathbf{W}_h \\mathbf{h}_{t-1} + \\mathbf{W}_x \\mathbf{x}_t\\). Adding the bias and applying \\(\\tanh\\) yields exactly the standard form. The equivalence follows from the definition of block matrix multiplication. \\(\\square\\)'
                }
            ]
        },

        // ========== SECTION 3: Training RNNs ==========
        {
            id: 'sec03-training-rnns',
            title: 'Training RNNs',
            content: `
<h2>12.3 Training RNNs</h2>

<p>Training an RNN requires computing gradients with respect to the shared parameters \\(\\mathbf{W}_h\\), \\(\\mathbf{W}_x\\), and \\(\\mathbf{b}\\). Because the unrolled computation graph forms a deep chain, we must propagate gradients backward through all time steps. This procedure, called <strong>backpropagation through time (BPTT)</strong>, reveals both the power and the fundamental weakness of vanilla RNNs.</p>

<h3>Backpropagation Through Time (BPTT)</h3>

<p>Consider a loss function that decomposes across time steps:</p>
\\[
\\mathcal{L} = \\sum_{t=1}^{T} \\mathcal{L}_t(\\mathbf{y}_t, \\hat{\\mathbf{y}}_t),
\\]
<p>where \\(\\hat{\\mathbf{y}}_t\\) is the target at step \\(t\\). To compute \\(\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}_h}\\), we sum contributions from every time step:</p>
\\[
\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}_h} = \\sum_{t=1}^{T} \\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{W}_h}.
\\]

<p>Each term \\(\\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{W}_h}\\) requires the chain rule through the recurrence. Since \\(\\mathbf{h}_t\\) depends on \\(\\mathbf{h}_{t-1}\\), which depends on \\(\\mathbf{h}_{t-2}\\), and so on, we get:</p>
\\[
\\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{W}_h} = \\sum_{k=1}^{t} \\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{h}_t} \\left(\\prod_{j=k+1}^{t} \\frac{\\partial \\mathbf{h}_j}{\\partial \\mathbf{h}_{j-1}}\\right) \\frac{\\partial \\mathbf{h}_k}{\\partial \\mathbf{W}_h}.
\\]

<div class="env-block definition">
<strong>Definition 12.3.1 (Jacobian of the Recurrence).</strong> The Jacobian of the hidden state transition is
\\[
\\frac{\\partial \\mathbf{h}_t}{\\partial \\mathbf{h}_{t-1}} = \\text{diag}\\!\\left(\\tanh'(\\mathbf{W}_h \\mathbf{h}_{t-1} + \\mathbf{W}_x \\mathbf{x}_t + \\mathbf{b})\\right) \\cdot \\mathbf{W}_h,
\\]
where \\(\\tanh'(z) = 1 - \\tanh^2(z) \\in (0, 1]\\).
</div>

<h3>The Vanishing and Exploding Gradient Problem</h3>

<p>The product of Jacobians \\(\\prod_{j=k+1}^{t} \\frac{\\partial \\mathbf{h}_j}{\\partial \\mathbf{h}_{j-1}}\\) involves repeated multiplication by matrices of the form \\(\\text{diag}(\\mathbf{d}_j) \\mathbf{W}_h\\). Over many time steps, two regimes emerge:</p>

<div class="env-block theorem">
<strong>Theorem 12.3.2 (Gradient Decay/Explosion, Bengio et al. 1994).</strong> Let \\(\\lambda_1\\) be the largest singular value of \\(\\mathbf{W}_h\\), and let \\(\\gamma = \\max_j \\|\\tanh'(\\cdot)\\|_\\infty \\le 1\\). Then:
<ul>
<li>If \\(\\gamma \\lambda_1 &lt; 1\\), the gradient norm \\(\\left\\|\\prod_{j=k+1}^{t} \\frac{\\partial \\mathbf{h}_j}{\\partial \\mathbf{h}_{j-1}}\\right\\|\\) decays exponentially as \\((t-k) \\to \\infty\\). Gradients <strong>vanish</strong>.</li>
<li>If \\(\\gamma \\lambda_1 &gt; 1\\), the gradient norm can grow exponentially. Gradients <strong>explode</strong>.</li>
</ul>
</div>

<div class="env-block remark">
<strong>Practical Consequence.</strong> Vanishing gradients mean that the loss at time step \\(t\\) has negligible influence on weights through early hidden states \\(\\mathbf{h}_k\\) for \\(k \\ll t\\). The network effectively "forgets" inputs from the distant past. Exploding gradients cause NaN values and training instability. The former is harder to detect (training simply plateaus), while the latter is immediately visible.
</div>

<h3>Gradient Clipping</h3>

<p>Exploding gradients can be mitigated by <strong>gradient clipping</strong>: if the gradient norm exceeds a threshold \\(\\tau\\), we rescale it:</p>
\\[
\\mathbf{g} \\leftarrow \\frac{\\tau}{\\|\\mathbf{g}\\|} \\mathbf{g} \\quad \\text{if } \\|\\mathbf{g}\\| &gt; \\tau.
\\]
<p>This preserves the gradient direction while capping its magnitude. Typical values are \\(\\tau \\in [1, 5]\\).</p>

<h3>Truncated BPTT</h3>

<p>For very long sequences, full BPTT is computationally expensive (\\(\\mathcal{O}(T)\\) memory and time). <strong>Truncated BPTT</strong> divides the sequence into chunks of length \\(K\\) and only backpropagates gradients within each chunk:</p>

<div class="env-block definition">
<strong>Definition 12.3.3 (Truncated BPTT).</strong> Partition the sequence into segments of length \\(K\\). For time step \\(t\\), only backpropagate through \\(\\mathbf{h}_{t-K+1}, \\ldots, \\mathbf{h}_t\\), treating \\(\\mathbf{h}_{t-K}\\) as a constant (detaching it from the computation graph). The forward pass still uses the full hidden state chain.
</div>

<div class="env-block remark">
<strong>Trade-off.</strong> Truncated BPTT introduces a bias: gradients for dependencies longer than \\(K\\) steps are zeroed out. However, since vanilla RNN gradients vanish exponentially, the bias is small in practice. The main benefit is reduced memory consumption from \\(\\mathcal{O}(T)\\) to \\(\\mathcal{O}(K)\\).
</div>

<div class="viz-placeholder" data-viz="viz-bptt-gradient"></div>
`,
            visualizations: [
                {
                    id: 'viz-bptt-gradient',
                    title: 'BPTT Gradient Flow Through Time',
                    description: 'Visualize how gradients flow backward from the loss at the final time step. The color intensity shows gradient magnitude, which fades with distance due to vanishing gradients. Adjust the spectral radius to see the transition between vanishing and exploding regimes.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 750, height: 420 });
                        const ctx = viz.ctx;

                        let numSteps = 6;
                        let spectralRadius = 0.7; // gamma * lambda_1
                        let time = 0;

                        VizEngine.createSlider(controls, 'Steps', 3, 8, numSteps, 1, val => { numSteps = Math.round(val); });
                        VizEngine.createSlider(controls, '\u03b3\u03bb\u2081', 0.2, 1.8, spectralRadius, 0.05, val => { spectralRadius = val; });

                        function draw(timestamp) {
                            time = timestamp / 1000;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            const cellW = 60, cellH = 44;
                            const spacing = Math.min(95, (viz.width - 100) / numSteps);
                            const startX = (viz.width - (numSteps - 1) * spacing) / 2;
                            const cellY = 180;

                            // Determine gradient magnitudes (from last step backward)
                            const gradMag = [];
                            for (let i = 0; i < numSteps; i++) {
                                const dist = numSteps - 1 - i;
                                gradMag.push(Math.pow(spectralRadius, dist));
                            }

                            // Draw cells and connections
                            for (let i = 0; i < numSteps; i++) {
                                const cx = startX + i * spacing;

                                // Gradient flow arrow (backward, from right to left)
                                if (i < numSteps - 1) {
                                    const nextCx = startX + (i + 1) * spacing;

                                    // Forward arrow (bottom, gray)
                                    ctx.strokeStyle = viz.colors.axis;
                                    ctx.lineWidth = 1.5;
                                    ctx.beginPath();
                                    ctx.moveTo(cx + cellW / 2 + 2, cellY + 8);
                                    ctx.lineTo(nextCx - cellW / 2 - 2, cellY + 8);
                                    ctx.stroke();
                                    // arrowhead
                                    ctx.fillStyle = viz.colors.axis;
                                    ctx.beginPath();
                                    ctx.moveTo(nextCx - cellW / 2 - 2, cellY + 8);
                                    ctx.lineTo(nextCx - cellW / 2 - 10, cellY + 3);
                                    ctx.lineTo(nextCx - cellW / 2 - 10, cellY + 13);
                                    ctx.closePath();
                                    ctx.fill();
                                }

                                // Backward gradient arrows (top, colored by magnitude)
                                if (i > 0) {
                                    const prevCx = startX + (i - 1) * spacing;
                                    const mag = gradMag[i - 1]; // gradient arriving at step i-1
                                    const cappedMag = Math.min(mag, 3);
                                    const alpha = Math.min(1, cappedMag);

                                    // Pulse effect
                                    const pulseDelay = (numSteps - 1 - i) * 0.3;
                                    const pulse = Math.max(0, Math.sin((time - pulseDelay) * 2));
                                    const finalAlpha = alpha * (0.5 + 0.5 * pulse);

                                    let arrowColor;
                                    if (spectralRadius > 1.0) {
                                        // Red for exploding
                                        arrowColor = `rgba(248, 81, 73, ${finalAlpha})`;
                                    } else {
                                        // Green-teal for vanishing
                                        arrowColor = `rgba(63, 185, 160, ${finalAlpha})`;
                                    }

                                    const lw = Math.min(4, 1 + cappedMag * 2);
                                    ctx.strokeStyle = arrowColor;
                                    ctx.lineWidth = lw;
                                    ctx.beginPath();
                                    ctx.moveTo(cx - cellW / 2 - 2, cellY - 10);
                                    ctx.lineTo(prevCx + cellW / 2 + 2, cellY - 10);
                                    ctx.stroke();
                                    // arrowhead
                                    ctx.fillStyle = arrowColor;
                                    ctx.beginPath();
                                    ctx.moveTo(prevCx + cellW / 2 + 2, cellY - 10);
                                    ctx.lineTo(prevCx + cellW / 2 + 10, cellY - 15);
                                    ctx.lineTo(prevCx + cellW / 2 + 10, cellY - 5);
                                    ctx.closePath();
                                    ctx.fill();
                                }

                                // Cell rectangle
                                const mag = gradMag[i];
                                const cappedMag = Math.min(mag, 3);
                                let cellFill;
                                if (spectralRadius > 1.0) {
                                    const a = Math.min(0.6, cappedMag * 0.2);
                                    cellFill = `rgba(248, 81, 73, ${a})`;
                                } else {
                                    const a = Math.min(0.6, cappedMag * 0.4);
                                    cellFill = `rgba(63, 185, 160, ${a})`;
                                }

                                ctx.fillStyle = cellFill;
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.roundRect(cx - cellW / 2, cellY - cellH / 2, cellW, cellH, 6);
                                ctx.fill();
                                ctx.stroke();

                                // Cell label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('h' + String.fromCharCode(8321 + i), cx, cellY);

                                // Input
                                ctx.strokeStyle = viz.colors.teal + '88';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(cx, cellY + cellH / 2 + 2);
                                ctx.lineTo(cx, cellY + cellH / 2 + 30);
                                ctx.stroke();
                                ctx.fillStyle = viz.colors.teal;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.fillText('x' + String.fromCharCode(8321 + i), cx, cellY + cellH / 2 + 42);

                                // Gradient magnitude label
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('|g|=' + mag.toFixed(2), cx, cellY + cellH / 2 + 60);
                            }

                            // Loss at the final step
                            const lastCx = startX + (numSteps - 1) * spacing;
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(lastCx, cellY - cellH / 2 - 2);
                            ctx.lineTo(lastCx, cellY - cellH / 2 - 40);
                            ctx.stroke();
                            ctx.fillStyle = viz.colors.red;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillText('\u2112(y_T)', lastCx, cellY - cellH / 2 - 52);

                            // Legend
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.axis;
                            ctx.fillText('\u2192 Forward pass', 20, viz.height - 50);
                            const gradColor = spectralRadius > 1.0 ? viz.colors.red : viz.colors.teal;
                            ctx.fillStyle = gradColor;
                            ctx.fillText('\u2190 Backward gradients (BPTT)', 20, viz.height - 32);

                            // Status
                            ctx.textAlign = 'center';
                            ctx.font = '13px -apple-system,sans-serif';
                            if (spectralRadius < 1.0) {
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText('\u03b3\u03bb\u2081 = ' + spectralRadius.toFixed(2) + ' < 1 \u2192 Gradients VANISH exponentially', viz.width / 2, 30);
                            } else if (spectralRadius > 1.0) {
                                ctx.fillStyle = viz.colors.red;
                                ctx.fillText('\u03b3\u03bb\u2081 = ' + spectralRadius.toFixed(2) + ' > 1 \u2192 Gradients EXPLODE exponentially', viz.width / 2, 30);
                            } else {
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.fillText('\u03b3\u03bb\u2081 = 1.00 \u2192 Gradients remain stable (critical boundary)', viz.width / 2, 30);
                            }

                            // Formula
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Gradient at step k: |g_k| \u2248 (\u03b3\u03bb\u2081)^(T-k) = ' + spectralRadius.toFixed(2) + '^(T-k)', viz.width / 2, 52);
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'For a simple scalar RNN \\(h_t = \\tanh(w_h h_{t-1} + w_x x_t)\\), derive the exact expression for \\(\\frac{\\partial h_t}{\\partial h_k}\\) for \\(k &lt; t\\).',
                    hint: 'Apply the chain rule: \\(\\frac{\\partial h_t}{\\partial h_k} = \\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}}\\). For the scalar case, each factor is \\(\\tanh\'(\\cdot) \\cdot w_h\\).',
                    solution: 'By the chain rule: \\[\\frac{\\partial h_t}{\\partial h_k} = \\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}} = \\prod_{j=k+1}^{t} \\tanh\'(w_h h_{j-1} + w_x x_j) \\cdot w_h = w_h^{t-k} \\prod_{j=k+1}^{t} (1 - h_j^2).\\] Since \\(0 &lt; 1 - h_j^2 \\le 1\\) and \\(|w_h| &lt; 1\\) is typical after tanh saturation, the product \\(|w_h|^{t-k} \\prod(1-h_j^2) \\to 0\\) exponentially as \\(t - k \\to \\infty\\). This is the vanishing gradient in its simplest form.'
                },
                {
                    question: 'Suppose \\(\\mathbf{W}_h\\) has singular values \\(\\sigma_1 = 1.2, \\sigma_2 = 0.8\\). After \\(T = 20\\) time steps of gradient backpropagation (ignoring the \\(\\tanh\'\\) terms), what are the magnification factors along each singular direction?',
                    hint: 'The gradient product involves \\(\\mathbf{W}_h^T\\) raised to the \\(T\\)-th power. The effect along each singular direction scales as \\(\\sigma_i^T\\).',
                    solution: 'Along the first singular direction: \\(\\sigma_1^{20} = 1.2^{20} \\approx 38.34\\). Along the second: \\(\\sigma_2^{20} = 0.8^{20} \\approx 0.0115\\). This means gradients along the first direction are amplified by a factor of \\(\\approx 38\\) (exploding), while gradients along the second direction are attenuated by a factor of \\(\\approx 87\\) (vanishing). Even within a single weight matrix, different singular directions can simultaneously exhibit exploding and vanishing behavior, making training unstable.'
                },
                {
                    question: 'Explain why gradient clipping addresses exploding gradients but not vanishing gradients. What architectural solution (explored in Ch. 13) addresses vanishing gradients?',
                    hint: 'Gradient clipping rescales large gradients but cannot amplify small ones without introducing noise. Think about what LSTM gates do to the gradient pathway.',
                    solution: 'Gradient clipping caps the gradient norm at a threshold \\(\\tau\\), preventing explosion: \\(\\mathbf{g} \\leftarrow \\frac{\\tau}{\\|\\mathbf{g}\\|} \\mathbf{g}\\). But if \\(\\|\\mathbf{g}\\| \\approx 0\\) due to vanishing, clipping does nothing (the condition \\(\\|\\mathbf{g}\\| &gt; \\tau\\) is never triggered). One cannot simply scale up small gradients, because near zero, the signal-to-noise ratio is poor, and amplification would magnify noise.<br><br>The architectural solution is the <strong>LSTM</strong> (Long Short-Term Memory), which introduces a cell state \\(\\mathbf{c}_t\\) with an additive update: \\(\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t\\). The forget gate \\(\\mathbf{f}_t\\) can be close to 1, creating a near-identity mapping for the gradient: \\(\\frac{\\partial \\mathbf{c}_t}{\\partial \\mathbf{c}_{t-1}} \\approx \\mathbf{I}\\), which prevents exponential decay.'
                },
                {
                    question: 'In truncated BPTT with chunk size \\(K = 20\\), the network processes a sequence of length \\(T = 200\\). (a) How many gradient computation chunks are there? (b) Can the model still learn dependencies longer than 20 steps? Explain.',
                    hint: 'The forward pass is untruncated; only the backward pass is truncated. Think about what information the hidden state carries forward.',
                    solution: '(a) There are \\(T / K = 200 / 20 = 10\\) gradient computation chunks. Each chunk backpropagates through 20 time steps.<br>(b) The model <em>can</em> represent dependencies longer than 20 steps because the <strong>forward pass is untruncated</strong>: \\(\\mathbf{h}_t\\) depends on all inputs \\(\\mathbf{x}_1, \\ldots, \\mathbf{x}_t\\). The hidden state carries information across chunk boundaries. However, the model cannot <em>learn</em> such dependencies through gradient-based optimization, because gradients are zeroed at chunk boundaries. So \\(\\frac{\\partial \\mathcal{L}_{t}}{\\partial \\mathbf{W}_h}\\) does not include contributions from steps more than \\(K\\) in the past. In practice, for vanilla RNNs, this is rarely a problem because gradients vanish exponentially anyway, meaning the bias from truncation is small compared to the inherent limitation.'
                }
            ]
        },

        // ========== SECTION 4: Bidirectional RNN ==========
        {
            id: 'sec04-bidirectional-rnn',
            title: 'Bidirectional RNN',
            content: `
<h2>12.4 Bidirectional RNN</h2>

<p>A standard (unidirectional) RNN processes the sequence from left to right. At time step \\(t\\), the hidden state \\(\\overrightarrow{\\mathbf{h}}_t\\) contains information only about past and present inputs \\(\\mathbf{x}_1, \\ldots, \\mathbf{x}_t\\). But in many tasks, the interpretation of a token depends on <em>future</em> context as well. Consider the word "bank" in these sentences:</p>

<ul>
<li>"I went to the bank to deposit money." (financial institution)</li>
<li>"I sat on the river bank and watched the sunset." (river edge)</li>
</ul>

<p>A forward-only RNN processing "bank" has not yet seen "deposit" or "river," so it cannot disambiguate. A <strong>bidirectional RNN</strong> (Schuster and Paliwal, 1997) solves this by processing the sequence in both directions.</p>

<h3>Architecture</h3>

<div class="env-block definition">
<strong>Definition 12.4.1 (Bidirectional RNN).</strong> A bidirectional RNN consists of two independent RNNs:
<ul>
<li>A <strong>forward RNN</strong> computing \\(\\overrightarrow{\\mathbf{h}}_t = \\tanh(\\overrightarrow{\\mathbf{W}}_h \\overrightarrow{\\mathbf{h}}_{t-1} + \\overrightarrow{\\mathbf{W}}_x \\mathbf{x}_t + \\overrightarrow{\\mathbf{b}})\\) for \\(t = 1, 2, \\ldots, T\\).</li>
<li>A <strong>backward RNN</strong> computing \\(\\overleftarrow{\\mathbf{h}}_t = \\tanh(\\overleftarrow{\\mathbf{W}}_h \\overleftarrow{\\mathbf{h}}_{t+1} + \\overleftarrow{\\mathbf{W}}_x \\mathbf{x}_t + \\overleftarrow{\\mathbf{b}})\\) for \\(t = T, T-1, \\ldots, 1\\).</li>
</ul>
The combined representation at time \\(t\\) is the concatenation:
\\[
\\mathbf{h}_t = [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t] \\in \\mathbb{R}^{2d}.
\\]
</div>

<p>The output layer operates on this concatenated representation:</p>
\\[
\\mathbf{y}_t = \\mathbf{W}_y [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t] + \\mathbf{b}_y,
\\]
<p>where \\(\\mathbf{W}_y \\in \\mathbb{R}^{m \\times 2d}\\).</p>

<div class="env-block remark">
<strong>Independence of the Two Chains.</strong> The forward and backward RNNs have completely separate parameters. They share only the input sequence \\(\\mathbf{x}_1, \\ldots, \\mathbf{x}_T\\). This means:
<ul>
<li>\\(\\overrightarrow{\\mathbf{h}}_t\\) encodes context from \\(\\mathbf{x}_1, \\ldots, \\mathbf{x}_t\\) (past and present).</li>
<li>\\(\\overleftarrow{\\mathbf{h}}_t\\) encodes context from \\(\\mathbf{x}_t, \\ldots, \\mathbf{x}_T\\) (present and future).</li>
<li>The concatenation \\(\\mathbf{h}_t\\) captures the <em>full</em> sequence context around position \\(t\\).</li>
</ul>
</div>

<h3>When to Use Bidirectional RNNs</h3>

<p>Bidirectional RNNs are appropriate when the <em>entire</em> input sequence is available at inference time. This includes:</p>
<ul>
<li><strong>Named entity recognition (NER):</strong> labeling each word in a sentence.</li>
<li><strong>Part-of-speech tagging:</strong> determining the grammatical role of each word.</li>
<li><strong>Speech recognition:</strong> the full utterance is recorded before processing.</li>
<li><strong>Encoding in seq2seq models:</strong> the encoder reads the entire source sentence.</li>
</ul>

<p>Bidirectional RNNs are <em>not</em> suitable for autoregressive generation (e.g., language modeling), where we predict the next token given only past tokens. In that setting, using future context would constitute "cheating," as the future tokens are the very things we are trying to predict.</p>

<div class="env-block theorem">
<strong>Proposition 12.4.2 (Parameter Count).</strong> A bidirectional RNN with forward and backward hidden dimensions \\(d\\) and input dimension \\(n\\) has, in its recurrence:
\\[
2 \\times (d^2 + d \\cdot n + d) = 2d^2 + 2dn + 2d
\\]
parameters. The output layer has \\(m \\times 2d + m\\) parameters. Compared to a unidirectional RNN with hidden dimension \\(2d\\) (to match the concatenated dimension), the bidirectional model has \\(2d^2 + 2dn + 2d\\) vs. \\(4d^2 + 2dn + 2d\\) recurrence parameters. Thus, a bidirectional RNN with dimension \\(d\\) per direction is more parameter-efficient than a unidirectional RNN with dimension \\(2d\\), though it requires the full sequence upfront.
</div>

<div class="viz-placeholder" data-viz="viz-bidirectional-rnn"></div>

<h3>Alternatives and Extensions</h3>

<p>The concatenation \\([\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t]\\) is the most common merging strategy, but alternatives include:</p>
<ul>
<li><strong>Summation:</strong> \\(\\mathbf{h}_t = \\overrightarrow{\\mathbf{h}}_t + \\overleftarrow{\\mathbf{h}}_t\\), which keeps the dimension at \\(d\\) but loses some information.</li>
<li><strong>Element-wise product:</strong> \\(\\mathbf{h}_t = \\overrightarrow{\\mathbf{h}}_t \\odot \\overleftarrow{\\mathbf{h}}_t\\), which captures interactions but can be unstable.</li>
<li><strong>Learned combination:</strong> \\(\\mathbf{h}_t = \\mathbf{W}_f \\overrightarrow{\\mathbf{h}}_t + \\mathbf{W}_b \\overleftarrow{\\mathbf{h}}_t\\), with additional parameters.</li>
</ul>

<p>In Chapter 13 we will combine bidirectionality with gated architectures (LSTM, GRU), which address the vanishing gradient problem while retaining the benefit of full-context representations.</p>
`,
            visualizations: [
                {
                    id: 'viz-bidirectional-rnn',
                    title: 'Bidirectional RNN Architecture',
                    description: 'Two parallel RNN chains process the sequence in opposite directions. Blue arrows show the forward pass (left to right), orange arrows show the backward pass (right to left). The concatenated output at each step combines both contexts.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 750, height: 440 });
                        const ctx = viz.ctx;

                        let numSteps = 5;
                        let highlightStep = 2; // 0-indexed
                        let time = 0;

                        VizEngine.createSlider(controls, 'Steps', 3, 7, numSteps, 1, val => { numSteps = Math.round(val); highlightStep = Math.min(highlightStep, numSteps - 1); });
                        VizEngine.createSlider(controls, 'Focus', 0, numSteps - 1, highlightStep, 1, val => { highlightStep = Math.min(Math.round(val), numSteps - 1); });

                        function drawArrow(x1, y1, x2, y2, color, lw) {
                            const dx = x2 - x1, dy = y2 - y1;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            if (len < 1) return;
                            const angle = Math.atan2(dy, dx);
                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 2;
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2 - Math.cos(angle) * 7, y2 - Math.sin(angle) * 7);
                            ctx.stroke();
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 9 * Math.cos(angle - Math.PI / 6), y2 - 9 * Math.sin(angle - Math.PI / 6));
                            ctx.lineTo(x2 - 9 * Math.cos(angle + Math.PI / 6), y2 - 9 * Math.sin(angle + Math.PI / 6));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function draw(timestamp) {
                            time = timestamp / 1000;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            const cellW = 56, cellH = 36;
                            const spacing = Math.min(110, (viz.width - 120) / numSteps);
                            const startX = (viz.width - (numSteps - 1) * spacing) / 2;
                            const fwdY = 150; // forward chain y
                            const bwdY = 260; // backward chain y
                            const inputY = 380; // input row
                            const outputY = 50; // output row
                            const concatY = 205; // between fwd and bwd

                            // Title labels
                            ctx.fillStyle = viz.colors.blue;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Forward', startX - cellW / 2 - 10, fwdY);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('Backward', startX - cellW / 2 - 10, bwdY);

                            for (let i = 0; i < numSteps; i++) {
                                const cx = startX + i * spacing;
                                const isHighlight = (i === highlightStep);
                                const pulse = isHighlight ? 0.3 + 0.2 * Math.sin(time * 3) : 0;

                                // ---- INPUT ----
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '13px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('x' + String.fromCharCode(8321 + i), cx, inputY);

                                // Arrow from input to forward cell
                                ctx.strokeStyle = viz.colors.teal + '66';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(cx, inputY - 12);
                                ctx.lineTo(cx, bwdY + cellH / 2 + 2);
                                ctx.stroke();

                                // Arrow from input to backward cell
                                ctx.beginPath();
                                ctx.moveTo(cx - 8, inputY - 12);
                                ctx.lineTo(cx - 8, bwdY + cellH / 2 + 2);
                                ctx.stroke();

                                ctx.beginPath();
                                ctx.moveTo(cx + 8, inputY - 12);
                                ctx.lineTo(cx + 8, fwdY + cellH / 2 + 2);
                                ctx.stroke();

                                // ---- FORWARD CELL ----
                                const fwdAlpha = isHighlight ? (0.4 + pulse) : 0.2;
                                ctx.fillStyle = `rgba(88, 166, 255, ${fwdAlpha})`;
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = isHighlight ? 2.5 : 1.5;
                                ctx.beginPath();
                                ctx.roundRect(cx - cellW / 2, fwdY - cellH / 2, cellW, cellH, 6);
                                ctx.fill(); ctx.stroke();

                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('\u2192h' + String.fromCharCode(8321 + i), cx, fwdY);

                                // Forward connections (left to right)
                                if (i < numSteps - 1) {
                                    const nextCx = startX + (i + 1) * spacing;
                                    drawArrow(cx + cellW / 2 + 2, fwdY, nextCx - cellW / 2 - 2, fwdY, viz.colors.blue + 'cc', 2);
                                }

                                // ---- BACKWARD CELL ----
                                const bwdAlpha = isHighlight ? (0.4 + pulse) : 0.2;
                                ctx.fillStyle = `rgba(240, 136, 62, ${bwdAlpha})`;
                                ctx.strokeStyle = viz.colors.orange;
                                ctx.lineWidth = isHighlight ? 2.5 : 1.5;
                                ctx.beginPath();
                                ctx.roundRect(cx - cellW / 2, bwdY - cellH / 2, cellW, cellH, 6);
                                ctx.fill(); ctx.stroke();

                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('\u2190h' + String.fromCharCode(8321 + i), cx, bwdY);

                                // Backward connections (right to left)
                                if (i > 0) {
                                    const prevCx = startX + (i - 1) * spacing;
                                    drawArrow(cx - cellW / 2 - 2, bwdY, prevCx + cellW / 2 + 2, bwdY, viz.colors.orange + 'cc', 2);
                                }

                                // ---- CONCATENATION / OUTPUT ----
                                // Merge arrows from fwd and bwd to output
                                if (isHighlight) {
                                    // Forward to output
                                    drawArrow(cx, fwdY - cellH / 2 - 2, cx - 10, outputY + 18, viz.colors.blue, 2);
                                    // Backward to output
                                    drawArrow(cx, bwdY - cellH / 2 - 2, cx + 4, fwdY + cellH / 2 + 2, viz.colors.orange + '66', 1);

                                    // Output box
                                    const outW = 80, outH = 28;
                                    ctx.fillStyle = viz.colors.green + '33';
                                    ctx.strokeStyle = viz.colors.green;
                                    ctx.lineWidth = 2;
                                    ctx.beginPath();
                                    ctx.roundRect(cx - outW / 2, outputY - outH / 2, outW, outH, 5);
                                    ctx.fill(); ctx.stroke();

                                    ctx.fillStyle = viz.colors.green;
                                    ctx.font = 'bold 11px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.fillText('[\u2192h;\u2190h]' + String.fromCharCode(8321 + i), cx, outputY);

                                    // Highlight vertical lines
                                    ctx.strokeStyle = viz.colors.yellow + '33';
                                    ctx.lineWidth = 1;
                                    ctx.setLineDash([4, 4]);
                                    ctx.beginPath();
                                    ctx.moveTo(cx, outputY + outH / 2);
                                    ctx.lineTo(cx, inputY - 14);
                                    ctx.stroke();
                                    ctx.setLineDash([]);
                                } else {
                                    // Thin lines from both cells to a merge point
                                    ctx.strokeStyle = viz.colors.text + '44';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.moveTo(cx, fwdY - cellH / 2 - 2);
                                    ctx.lineTo(cx, outputY + 14);
                                    ctx.stroke();
                                }

                                // Small output circle
                                if (!isHighlight) {
                                    ctx.fillStyle = viz.colors.green + '66';
                                    ctx.beginPath();
                                    ctx.arc(cx, outputY, 6, 0, Math.PI * 2);
                                    ctx.fill();
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.fillText('y' + String.fromCharCode(8321 + i), cx, outputY - 14);
                                }
                            }

                            // Info text at bottom
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Adjust Focus slider to highlight a time step. The output combines past (\u2192) and future (\u2190) context.', viz.width / 2, viz.height - 10);

                            // Context range annotation for highlighted step
                            ctx.font = '11px -apple-system,sans-serif';
                            const hx = startX + highlightStep * spacing;
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText('\u2192 sees x\u2081..x' + String.fromCharCode(8321 + highlightStep), hx, fwdY + cellH / 2 + 16);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('\u2190 sees x' + String.fromCharCode(8321 + highlightStep) + '..x' + String.fromCharCode(8321 + numSteps - 1), hx, bwdY + cellH / 2 + 16);
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'A bidirectional RNN with \\(d = 128\\) per direction produces concatenated states of dimension \\(2d = 256\\). A unidirectional RNN with \\(d = 256\\) also produces states of dimension 256. Compare their recurrence parameter counts (assuming input dimension \\(n = 100\\)).',
                    hint: 'For the bidirectional model, each direction has \\(d^2 + dn + d\\) parameters. For the unidirectional model, compute \\(d^2 + dn + d\\) with \\(d = 256\\).',
                    solution: '<strong>Bidirectional (\\(d=128\\)):</strong> Each direction: \\(128^2 + 128 \\times 100 + 128 = 16{,}384 + 12{,}800 + 128 = 29{,}312\\). Two directions: \\(2 \\times 29{,}312 = 58{,}624\\).<br><strong>Unidirectional (\\(d=256\\)):</strong> \\(256^2 + 256 \\times 100 + 256 = 65{,}536 + 25{,}600 + 256 = 91{,}392\\).<br>The bidirectional model uses \\(58{,}624 / 91{,}392 \\approx 64\\%\\) of the unidirectional model\'s parameters while producing the same output dimension. The savings come from the quadratic scaling of \\(\\mathbf{W}_h\\): two \\(128 \\times 128\\) matrices total \\(2 \\times 16{,}384 = 32{,}768\\) entries, while one \\(256 \\times 256\\) matrix has \\(65{,}536\\) entries.'
                },
                {
                    question: 'Explain why a bidirectional RNN is unsuitable for autoregressive language modeling (predicting the next word), but appropriate for masked language modeling (predicting a masked word given its context).',
                    hint: 'In autoregressive modeling, the model predicts \\(w_t\\) given \\(w_1, \\ldots, w_{t-1}\\). What information does the backward pass use?',
                    solution: 'In <strong>autoregressive language modeling</strong>, the goal is to compute \\(P(w_t \\mid w_1, \\ldots, w_{t-1})\\). If we use a bidirectional RNN, the backward pass would read \\(w_t, w_{t+1}, \\ldots, w_T\\), meaning the model has access to the very token \\(w_t\\) it is trying to predict (and all future tokens). This is information leakage: the model "sees the answer." It would learn to simply copy \\(w_t\\) from the backward hidden state rather than learning to predict.<br><br>In <strong>masked language modeling</strong> (e.g., BERT), the token at position \\(t\\) is replaced by a [MASK] token, so the backward RNN reads the mask symbol, not the true word. Both forward and backward passes provide surrounding context without revealing the answer. The bidirectional architecture is then appropriate and desirable, as it leverages the full context for prediction.'
                },
                {
                    question: 'Derive the gradient \\(\\frac{\\partial \\mathcal{L}}{\\partial \\overrightarrow{\\mathbf{W}}_h}\\) for a bidirectional RNN. Does the backward chain affect this gradient?',
                    hint: 'Trace the computation graph. Does the loss at step \\(t\\) flow through the backward chain to reach \\(\\overrightarrow{\\mathbf{W}}_h\\)?',
                    solution: 'The loss at step \\(t\\) is \\(\\mathcal{L}_t = \\ell(\\mathbf{y}_t, \\hat{\\mathbf{y}}_t)\\) where \\(\\mathbf{y}_t = \\mathbf{W}_y [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t] + \\mathbf{b}_y\\). The gradient flows: \\(\\mathcal{L}_t \\to \\mathbf{y}_t \\to [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t]\\). From the concatenation, the gradient splits into two independent paths: one to \\(\\overrightarrow{\\mathbf{h}}_t\\) and one to \\(\\overleftarrow{\\mathbf{h}}_t\\). The path to \\(\\overrightarrow{\\mathbf{h}}_t\\) depends only on the forward chain: \\(\\overrightarrow{\\mathbf{h}}_t \\to \\overrightarrow{\\mathbf{h}}_{t-1} \\to \\cdots\\), all of which involve \\(\\overrightarrow{\\mathbf{W}}_h\\). The backward chain \\(\\overleftarrow{\\mathbf{h}}_t\\) uses \\(\\overleftarrow{\\mathbf{W}}_h\\), a completely separate parameter. Therefore: \\[\\frac{\\partial \\mathcal{L}}{\\partial \\overrightarrow{\\mathbf{W}}_h} = \\sum_{t=1}^T \\frac{\\partial \\mathcal{L}_t}{\\partial \\overrightarrow{\\mathbf{h}}_t} \\sum_{k=1}^t \\left(\\prod_{j=k+1}^t \\frac{\\partial \\overrightarrow{\\mathbf{h}}_j}{\\partial \\overrightarrow{\\mathbf{h}}_{j-1}}\\right) \\frac{\\partial \\overrightarrow{\\mathbf{h}}_k}{\\partial \\overrightarrow{\\mathbf{W}}_h}.\\] The backward chain does <em>not</em> affect this gradient. The two chains are trained independently through their respective gradient paths.'
                },
                {
                    question: 'Suppose we use element-wise summation \\(\\mathbf{h}_t = \\overrightarrow{\\mathbf{h}}_t + \\overleftarrow{\\mathbf{h}}_t\\) instead of concatenation. What is the advantage and disadvantage compared to concatenation?',
                    hint: 'Consider the output dimension and the ability to distinguish forward-derived features from backward-derived features.',
                    solution: '<strong>Advantages of summation:</strong> (1) The output dimension remains \\(d\\) instead of \\(2d\\), reducing the parameter count of subsequent layers (e.g., the output projection \\(\\mathbf{W}_y \\in \\mathbb{R}^{m \\times d}\\) vs. \\(\\mathbb{R}^{m \\times 2d}\\)). (2) If pre-trained embeddings are involved, maintaining consistent dimensions can simplify architecture design. (3) Summation acts as an implicit regularizer by forcing both directions to operate in the same feature space.<br><br><strong>Disadvantages:</strong> (1) Information loss: the downstream layers cannot distinguish whether a feature came from the forward or backward pass. For example, if \\(\\overrightarrow{h}_{t,j} = 0.5\\) (past context suggests noun) and \\(\\overleftarrow{h}_{t,j} = -0.3\\) (future context suggests verb), the sum \\(0.2\\) loses this conflict. With concatenation, both values are preserved and the output layer can learn to resolve conflicts. (2) Destructive interference: if the two directions produce features of opposite sign for different reasons, summation can cancel meaningful signals. In practice, concatenation is almost always preferred because the additional parameters in \\(\\mathbf{W}_y\\) are a small cost compared to the information preserved.'
                }
            ]
        }
    ]
});
