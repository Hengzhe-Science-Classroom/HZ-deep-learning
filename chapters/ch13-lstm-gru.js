// === Chapter 13: LSTM & Gated Units ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch13',
    number: 13,
    title: 'LSTM & Gated Units',
    subtitle: 'Long Short-Term Memory, Gated Recurrent Units, and the art of learning what to remember',
    sections: [
        // ======================== Section 1 ========================
        {
            id: 'vanishing-gradient',
            title: 'The Vanishing Gradient Problem in RNNs',
            content: `
<h2>The Vanishing Gradient Problem in RNNs</h2>

<div class="env-block env-intuition">
<div class="env-header">Why Vanilla RNNs Forget</div>
<div class="env-body">
<p>In Chapter 12 we saw that a recurrent neural network maintains a hidden state \\(h_t\\) that is updated at each time step. In principle, this gives the network access to the entire history of the sequence. In practice, vanilla RNNs are notoriously bad at capturing long-range dependencies. The culprit is the <strong>vanishing gradient problem</strong>: during backpropagation through time (BPTT), gradients are multiplied by the same weight matrix at every step, causing them to shrink (or explode) exponentially with the number of time steps.</p>
</div>
</div>

<p>Recall the vanilla RNN update:</p>
\\[h_t = \\tanh(W_h h_{t-1} + W_x x_t + b).\\]

<p>When we compute gradients of a loss \\(\\mathcal{L}\\) at time \\(T\\) with respect to the hidden state at an earlier time \\(t\\), the chain rule gives us a product of Jacobians:</p>
\\[\\frac{\\partial \\mathcal{L}}{\\partial h_t} = \\frac{\\partial \\mathcal{L}}{\\partial h_T} \\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}}.\\]

<p>Each factor in this product is:</p>
\\[\\frac{\\partial h_k}{\\partial h_{k-1}} = \\text{diag}(1 - h_k^2) \\cdot W_h,\\]
<p>where \\(\\text{diag}(1 - h_k^2)\\) comes from the derivative of \\(\\tanh\\). Since \\(|\\tanh'(z)| \\leq 1\\) and the entries of \\(W_h\\) are typically small, the spectral norm of each factor is usually less than 1. After multiplying \\(T - t\\) such matrices together, the result shrinks exponentially.</p>

<div class="env-block env-definition">
<div class="env-header">Definition 13.1 &mdash; Gradient Decay Bound</div>
<div class="env-body">
<p>Let \\(\\gamma = \\|\\text{diag}(1 - h_k^2) \\cdot W_h\\|\\) be the spectral norm of a single Jacobian factor (assumed roughly constant across steps). Then:</p>
\\[\\left\\|\\frac{\\partial h_T}{\\partial h_t}\\right\\| \\leq \\gamma^{T-t}.\\]
<p>If \\(\\gamma &lt; 1\\), the gradient decays exponentially (vanishing). If \\(\\gamma &gt; 1\\), it grows exponentially (exploding).</p>
</div>
</div>

<div class="env-block env-example">
<div class="env-header">Example 13.1 &mdash; Exponential Decay in Practice</div>
<div class="env-body">
<p>Suppose \\(\\gamma = 0.9\\). After 10 steps, the gradient is attenuated by \\(0.9^{10} \\approx 0.35\\). After 50 steps, it becomes \\(0.9^{50} \\approx 0.005\\), essentially zero for optimization purposes. With \\(\\gamma = 0.7\\), after just 20 steps the gradient is \\(0.7^{20} \\approx 0.0008\\).</p>
</div>
</div>

<p>This has a devastating practical consequence: <strong>a vanilla RNN cannot learn dependencies that span more than roughly 10 to 20 time steps</strong>. If the subject of a sentence appears 30 words before its verb, a vanilla RNN will not learn the agreement pattern, because the gradient signal from the verb's loss cannot propagate back to the subject's representation.</p>

<div class="env-block env-warning">
<div class="env-header">Exploding Gradients</div>
<div class="env-body">
<p>The opposite problem, exploding gradients (\\(\\gamma &gt; 1\\)), is easier to handle: we simply clip gradients to a maximum norm. But vanishing gradients cannot be fixed by clipping. When the gradient is near zero, there is no information to amplify. This asymmetry motivates architectural solutions rather than optimization tricks.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="gradient-decay-viz"></div>

<p>The visualization above reveals the core problem. The gradient magnitude drops off precipitously with distance, meaning that early inputs have almost no influence on the loss. The key insight that led to LSTMs is this: if the gradient flows through a <em>multiplicative</em> bottleneck at every step, we need an architecture where the gradient can flow along an <em>additive</em> path. This is exactly what the cell state in an LSTM provides.</p>

<div class="env-block env-intuition">
<div class="env-header">The Highway Metaphor</div>
<div class="env-body">
<p>Think of gradient flow in a vanilla RNN as passing through a series of tunnels, each of which absorbs some of the signal. By the time the signal reaches the entrance, it has been absorbed almost entirely. The LSTM introduces a highway that bypasses these tunnels: the <strong>cell state</strong>. Gradients can travel along this highway with minimal attenuation, preserving long-range dependencies.</p>
</div>
</div>
`,
            visualizations: [
                {
                    id: 'gradient-decay-viz',
                    title: 'Gradient Magnitude vs. Time Step Distance',
                    description: 'Adjust \\(\\gamma\\) (spectral norm per step) to see how the gradient decays exponentially for vanilla RNNs. The blue shaded region shows where gradients are too small to drive learning.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 60, originY: 20 });
                        var W = viz.width;
                        var H = viz.height;
                        var ctx = viz.ctx;
                        var gamma = { value: 0.85 };
                        var maxSteps = 50;

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var padL = 70, padR = 30, padT = 40, padB = 50;
                            var plotW = W - padL - padR;
                            var plotH = H - padT - padB;

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.moveTo(padL, padT);
                            ctx.lineTo(padL, padT + plotH);
                            ctx.lineTo(padL + plotW, padT + plotH);
                            ctx.stroke();

                            // Y-axis label
                            ctx.save();
                            ctx.translate(18, padT + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Gradient Magnitude', 0, 0);
                            ctx.restore();

                            // X-axis label
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Time Steps Back (T - t)', padL + plotW / 2, padT + plotH + 30);

                            // Y-axis ticks
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            ctx.font = '11px -apple-system,sans-serif';
                            for (var i = 0; i <= 5; i++) {
                                var yVal = i / 5;
                                var yPx = padT + plotH - yVal * plotH;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(yVal.toFixed(1), padL - 8, yPx);
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.moveTo(padL, yPx);
                                ctx.lineTo(padL + plotW, yPx);
                                ctx.stroke();
                            }

                            // X-axis ticks
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (var i = 0; i <= maxSteps; i += 10) {
                                var xPx = padL + (i / maxSteps) * plotW;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(i, xPx, padT + plotH + 5);
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.moveTo(xPx, padT);
                                ctx.lineTo(xPx, padT + plotH);
                                ctx.stroke();
                            }

                            // Dead zone (gradient too small to learn)
                            var threshold = 0.01;
                            ctx.fillStyle = 'rgba(88,166,255,0.08)';
                            var threshY = padT + plotH - threshold * plotH;
                            ctx.fillRect(padL, threshY, plotW, padT + plotH - threshY);

                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath();
                            ctx.moveTo(padL, threshY);
                            ctx.lineTo(padL + plotW, threshY);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            ctx.fillStyle = viz.colors.blue;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText('Vanishing threshold', padL + 5, threshY - 3);

                            // Vanilla RNN curve
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var s = 0; s <= maxSteps; s++) {
                                var grad = Math.pow(gamma.value, s);
                                var xPx = padL + (s / maxSteps) * plotW;
                                var yPx = padT + plotH - grad * plotH;
                                if (s === 0) ctx.moveTo(xPx, yPx);
                                else ctx.lineTo(xPx, yPx);
                            }
                            ctx.stroke();

                            // LSTM ideal curve (much slower decay)
                            var gammaLSTM = 0.98;
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 3]);
                            ctx.beginPath();
                            for (var s = 0; s <= maxSteps; s++) {
                                var grad = Math.pow(gammaLSTM, s);
                                var xPx = padL + (s / maxSteps) * plotW;
                                var yPx = padT + plotH - grad * plotH;
                                if (s === 0) ctx.moveTo(xPx, yPx);
                                else ctx.lineTo(xPx, yPx);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Legend
                            var legX = padL + plotW - 180, legY = padT + 15;
                            ctx.fillStyle = 'rgba(12,12,32,0.85)';
                            ctx.fillRect(legX - 10, legY - 10, 195, 60);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(legX - 10, legY - 10, 195, 60);

                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([]);
                            ctx.beginPath(); ctx.moveTo(legX, legY + 4); ctx.lineTo(legX + 25, legY + 4); ctx.stroke();
                            ctx.fillStyle = viz.colors.red;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Vanilla RNN (\u03b3 = ' + gamma.value.toFixed(2) + ')', legX + 30, legY + 4);

                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 3]);
                            ctx.beginPath(); ctx.moveTo(legX, legY + 28); ctx.lineTo(legX + 25, legY + 28); ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('LSTM cell state (\u03b3 \u2248 0.98)', legX + 30, legY + 28);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('\u2225\u2202L/\u2202h_t\u2225 = \u03b3^(T\u2212t)', padL + plotW / 2, padT - 15);
                        }

                        draw();
                        VizEngine.createSlider(controls, '\u03b3', 0.5, 0.99, gamma.value, 0.01, function(v) {
                            gamma.value = v;
                            draw();
                        });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'For a vanilla RNN with \\(\\gamma = 0.8\\), how many time steps back can the gradient propagate before it falls below 1% of its original magnitude? Express your answer as \\(T - t\\).',
                    hint: 'Solve \\(0.8^{n} &lt; 0.01\\), i.e., \\(n &gt; \\frac{\\ln 0.01}{\\ln 0.8}\\).',
                    solution: 'We need \\(0.8^n &lt; 0.01\\). Taking logarithms: \\(n &gt; \\frac{\\ln(0.01)}{\\ln(0.8)} = \\frac{-4.605}{-0.2231} \\approx 20.6\\). So after about <strong>21 steps</strong>, the gradient has decayed below 1%.'
                },
                {
                    question: 'Explain why gradient clipping helps with exploding gradients but not with vanishing gradients.',
                    hint: 'Think about what happens when you clip a large vector vs. when you try to recover a near-zero vector.',
                    solution: 'Gradient clipping rescales the gradient when its norm exceeds a threshold: \\(g \\leftarrow g \\cdot \\frac{\\theta}{\\|g\\|}\\) if \\(\\|g\\| &gt; \\theta\\). This prevents explosions by capping the magnitude while preserving the direction. However, when gradients vanish (\\(\\|g\\| \\approx 0\\)), clipping does nothing, because there is no meaningful directional information to preserve. You cannot amplify a zero signal. Hence clipping is a one-sided fix: it handles the \\(\\gamma &gt; 1\\) regime but not the \\(\\gamma &lt; 1\\) regime.'
                },
                {
                    question: 'In the Jacobian \\(\\frac{\\partial h_k}{\\partial h_{k-1}} = \\text{diag}(1 - h_k^2) \\cdot W_h\\), what is the maximum possible value of any diagonal entry in \\(\\text{diag}(1 - h_k^2)\\)? Why does this contribute to vanishing gradients?',
                    hint: 'Recall that \\(h_k = \\tanh(\\cdot)\\), so \\(h_k \\in (-1, 1)\\). What is the range of \\(1 - h_k^2\\)?',
                    solution: 'Since \\(h_k = \\tanh(z)\\) for some \\(z\\), we have \\(h_k \\in (-1, 1)\\), so \\(h_k^2 \\in [0, 1)\\) and \\(1 - h_k^2 \\in (0, 1]\\). The maximum value of \\(1 - h_k^2\\) is 1, achieved when \\(h_k = 0\\) (i.e., \\(z = 0\\)). In practice, for most activations \\(h_k\\) is far from zero, so the diagonal entries are significantly less than 1. This means every Jacobian factor has spectral norm bounded well below \\(\\|W_h\\|\\), and in many practical settings the product \\(\\|(1 - h_k^2)\\| \\cdot \\|W_h\\| &lt; 1\\), causing exponential decay.'
                }
            ]
        },

        // ======================== Section 2 ========================
        {
            id: 'lstm-cell-state-gates',
            title: 'LSTM: Cell State & Gates',
            content: `
<h2>LSTM: Cell State &amp; Gates</h2>

<div class="env-block env-intuition">
<div class="env-header">The Key Innovation</div>
<div class="env-body">
<p>The Long Short-Term Memory (LSTM) network, introduced by Hochreiter &amp; Schmidhuber (1997), solves the vanishing gradient problem with a deceptively simple idea: maintain a <strong>cell state</strong> \\(C_t\\) that is updated via <em>addition</em> rather than multiplication. The cell state acts as a conveyor belt running through the entire sequence. Information can be placed onto the belt, read from the belt, or removed from the belt, all controlled by learned <strong>gates</strong>.</p>
</div>
</div>

<p>An LSTM cell has four components at each time step:</p>
<ol>
<li><strong>Forget gate</strong> \\(f_t\\): decides what to erase from the cell state</li>
<li><strong>Input gate</strong> \\(i_t\\): decides what new information to write</li>
<li><strong>Candidate cell value</strong> \\(\\tilde{C}_t\\): proposes new content</li>
<li><strong>Output gate</strong> \\(o_t\\): decides what to expose as the hidden state</li>
</ol>

<div class="env-block env-definition">
<div class="env-header">Definition 13.2 &mdash; LSTM Equations</div>
<div class="env-body">
<p>Given input \\(x_t\\), previous hidden state \\(h_{t-1}\\), and previous cell state \\(C_{t-1}\\):</p>
\\[f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f) \\quad \\text{(forget gate)}\\]
\\[i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i) \\quad \\text{(input gate)}\\]
\\[\\tilde{C}_t = \\tanh(W_C [h_{t-1}, x_t] + b_C) \\quad \\text{(candidate)}\\]
\\[C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t \\quad \\text{(cell update)}\\]
\\[o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o) \\quad \\text{(output gate)}\\]
\\[h_t = o_t \\odot \\tanh(C_t) \\quad \\text{(hidden state)}\\]
<p>where \\(\\sigma\\) is the sigmoid function, \\(\\odot\\) denotes element-wise multiplication, and \\([h_{t-1}, x_t]\\) is the concatenation of \\(h_{t-1}\\) and \\(x_t\\).</p>
</div>
</div>

<h3>The Forget Gate</h3>
<p>The forget gate \\(f_t \\in (0, 1)^d\\) looks at the current input \\(x_t\\) and previous hidden state \\(h_{t-1}\\) and decides, for each dimension of the cell state, how much to retain. A value close to 1 means "keep this information"; a value close to 0 means "discard it."</p>

<div class="env-block env-example">
<div class="env-header">Example 13.2 &mdash; Language Modeling</div>
<div class="env-body">
<p>Consider a language model processing the sentence: "The cat, which was sitting on the mat, <strong>was</strong> happy." When the model encounters "which," the forget gate might keep the subject "cat" (singular noun) in the cell state. When it reaches "was happy," the cell state still remembers the subject is singular, enabling correct verb agreement across the relative clause.</p>
</div>
</div>

<h3>The Input Gate and Candidate</h3>
<p>The input gate \\(i_t\\) determines which dimensions of the cell state get updated. The candidate \\(\\tilde{C}_t\\) provides the new values. Their element-wise product \\(i_t \\odot \\tilde{C}_t\\) is the actual update added to the cell state. Notice the <strong>additive</strong> structure:</p>
\\[C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t.\\]
<p>This is the reason gradients do not vanish. During backpropagation, \\(\\frac{\\partial C_t}{\\partial C_{t-1}} = \\text{diag}(f_t)\\), which can be close to the identity when the forget gate is open (\\(f_t \\approx 1\\)). There is no mandatory nonlinearity compressing the gradient at every step.</p>

<h3>The Output Gate</h3>
<p>The output gate \\(o_t\\) controls what portion of the cell state is exposed as the hidden state \\(h_t\\). This separation is crucial: the cell state can store information that is not immediately relevant to the current output but will be needed later.</p>

<div class="env-block env-intuition">
<div class="env-header">Why the Cell State Solves Vanishing Gradients</div>
<div class="env-body">
<p>In a vanilla RNN, \\(\\frac{\\partial h_t}{\\partial h_{t-1}}\\) always involves multiplying by \\(W_h\\) and squashing through \\(\\tanh'\\). In an LSTM, the gradient through the cell state is:</p>
\\[\\frac{\\partial C_T}{\\partial C_t} = \\prod_{k=t+1}^{T} f_k.\\]
<p>If the forget gates are close to 1 (which the network learns when long-range memory is beneficial), this product stays close to 1 even over many steps. The gradient has a <em>linear shortcut</em> through the cell state, analogous to residual connections in deep feedforward networks.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="lstm-cell-viz"></div>

<div class="env-block env-warning">
<div class="env-header">Forget Gate Bias Initialization</div>
<div class="env-body">
<p>A practical detail that makes a large difference: the forget gate bias \\(b_f\\) should be initialized to a positive value (typically 1 or 2). This ensures that at the start of training, the forget gates are open (\\(f_t \\approx \\sigma(1) \\approx 0.73\\)), allowing gradients to flow. Without this, an LSTM can behave almost as poorly as a vanilla RNN during early training. This trick was highlighted by Jozefowicz et al. (2015).</p>
</div>
</div>
`,
            visualizations: [
                {
                    id: 'lstm-cell-viz',
                    title: 'LSTM Cell: Step-by-Step Data Flow',
                    description: 'Watch data flow through the LSTM cell. The animation cycles through: forget gate (red), input gate (green), cell state update, and output gate (blue). Use the slider to step through manually.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 0, originY: 0 });
                        var W = viz.width;
                        var H = viz.height;
                        var ctx = viz.ctx;
                        var phase = { value: 0 };
                        var animTime = 0;
                        var autoPlay = true;

                        var phaseNames = ['Forget Gate', 'Input Gate + Candidate', 'Cell State Update', 'Output Gate + Hidden State'];
                        var phaseColors = ['#f85149', '#3fb950', '#d29922', '#58a6ff'];

                        function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

                        // Layout constants
                        var cellX = W * 0.5;
                        var cellY = H * 0.45;
                        var cellW = W * 0.55;
                        var cellH = H * 0.55;
                        var gateR = 18;

                        function drawRoundedRect(x, y, w, h, r, fill, stroke) {
                            ctx.beginPath();
                            ctx.moveTo(x + r, y);
                            ctx.lineTo(x + w - r, y);
                            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                            ctx.lineTo(x + w, y + h - r);
                            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                            ctx.lineTo(x + r, y + h);
                            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                            ctx.lineTo(x, y + r);
                            ctx.quadraticCurveTo(x, y, x + r, y);
                            ctx.closePath();
                            if (fill) { ctx.fillStyle = fill; ctx.fill(); }
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
                        }

                        function drawGate(cx, cy, label, color, active) {
                            ctx.beginPath();
                            ctx.arc(cx, cy, gateR, 0, Math.PI * 2);
                            ctx.fillStyle = active ? color + 'cc' : color + '33';
                            ctx.fill();
                            ctx.strokeStyle = active ? color : color + '66';
                            ctx.lineWidth = active ? 2.5 : 1.5;
                            ctx.stroke();
                            ctx.fillStyle = active ? '#fff' : '#aaa';
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(label, cx, cy);
                        }

                        function drawArrow(x1, y1, x2, y2, color, lw, pulse) {
                            var dx = x2 - x1, dy = y2 - y1;
                            var len = Math.sqrt(dx * dx + dy * dy);
                            if (len < 1) return;
                            var ux = dx / len, uy = dy / len;

                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 2;
                            if (pulse) {
                                ctx.shadowColor = color;
                                ctx.shadowBlur = 8;
                            }
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2 - ux * 8, y2 - uy * 8);
                            ctx.stroke();

                            ctx.fillStyle = color;
                            ctx.beginPath();
                            var angle = Math.atan2(dy, dx);
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 10 * Math.cos(angle - Math.PI / 7), y2 - 10 * Math.sin(angle - Math.PI / 7));
                            ctx.lineTo(x2 - 10 * Math.cos(angle + Math.PI / 7), y2 - 10 * Math.sin(angle + Math.PI / 7));
                            ctx.closePath();
                            ctx.fill();

                            ctx.shadowColor = 'transparent';
                            ctx.shadowBlur = 0;
                        }

                        function drawOp(cx, cy, symbol, color, active) {
                            ctx.beginPath();
                            ctx.arc(cx, cy, 14, 0, Math.PI * 2);
                            ctx.fillStyle = active ? color + '55' : '#1a1a40';
                            ctx.fill();
                            ctx.strokeStyle = active ? color : '#4a4a7a';
                            ctx.lineWidth = active ? 2 : 1;
                            ctx.stroke();
                            ctx.fillStyle = active ? color : '#8b949e';
                            ctx.font = 'bold 16px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(symbol, cx, cy);
                        }

                        function draw() {
                            var p = Math.floor(phase.value);
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Cell boundary
                            var bx = cellX - cellW / 2, by = cellY - cellH / 2;
                            drawRoundedRect(bx, by, cellW, cellH, 16, '#14142e', '#30363d');

                            // Cell state line (the highway)
                            var csY = by + 40;
                            var csX1 = bx - 30, csX2 = bx + cellW + 30;
                            ctx.strokeStyle = p === 2 ? '#d29922' : '#d2992266';
                            ctx.lineWidth = p === 2 ? 3.5 : 2.5;
                            ctx.beginPath();
                            ctx.moveTo(csX1, csY);
                            ctx.lineTo(csX2, csY);
                            ctx.stroke();

                            // Labels on cell state line
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('C(t-1)', csX1 - 5, csY);
                            ctx.textAlign = 'left';
                            ctx.fillText('C(t)', csX2 + 5, csY);

                            // Hidden state line
                            var hsY = by + cellH - 40;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('h(t-1)', bx - 35, hsY);
                            ctx.textAlign = 'left';
                            ctx.fillText('h(t)', bx + cellW + 35, hsY);

                            // Gate positions
                            var fgX = bx + cellW * 0.2;
                            var igX = bx + cellW * 0.42;
                            var candX = bx + cellW * 0.42;
                            var ogX = bx + cellW * 0.75;
                            var gateY = by + cellH * 0.6;

                            // Forget gate multiply op
                            var fMultX = fgX, fMultY = csY;
                            drawOp(fMultX, fMultY, '\u00d7', '#f85149', p === 0 || p === 2);

                            // Input gate add op
                            var addX = igX, addY = csY;
                            drawOp(addX, addY, '+', '#3fb950', p === 1 || p === 2);

                            // Output tanh op
                            var tanhX = ogX, tanhY = csY + 35;

                            // Gates
                            drawGate(fgX, gateY, 'f', '#f85149', p === 0);
                            drawGate(igX - 30, gateY, 'i', '#3fb950', p === 1);
                            drawGate(candX + 30, gateY + 35, '\u0108', '#3fb9a0', p === 1);
                            drawGate(ogX, gateY, 'o', '#58a6ff', p === 3);

                            // Tanh on cell state for output
                            drawOp(tanhX, tanhY, 'th', '#58a6ff', p === 3);

                            // Connections: forget gate to multiply
                            drawArrow(fgX, gateY - gateR, fgX, csY + 16, '#f85149', p === 0 ? 2.5 : 1.2, p === 0);

                            // Connections: input gate to add
                            drawArrow(igX - 30, gateY - gateR, addX - 8, csY + 16, '#3fb950', p === 1 ? 2.5 : 1.2, p === 1);

                            // Connections: candidate to add
                            drawArrow(candX + 30, gateY + 35 - gateR, addX + 8, csY + 16, '#3fb9a0', p === 1 ? 2.5 : 1.2, p === 1);

                            // Connections: output gate to hidden
                            drawArrow(ogX, gateY - gateR, ogX, tanhY + 16, '#58a6ff', p === 3 ? 2.5 : 1.2, p === 3);

                            // Hidden state output line
                            var hOutY = hsY;
                            drawArrow(ogX, tanhY + 16, ogX, hOutY, '#58a6ff', p === 3 ? 2 : 1, p === 3);
                            drawArrow(ogX, hOutY, bx + cellW + 30, hOutY, '#58a6ff', p === 3 ? 2 : 1, p === 3);

                            // Input arrows from h(t-1) and x(t)
                            var inputY = by + cellH + 20;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('x(t)', cellX, inputY + 15);

                            drawArrow(cellX - 40, inputY, fgX, gateY + gateR, '#f8514966', 1, false);
                            drawArrow(cellX - 15, inputY, igX - 30, gateY + gateR, '#3fb95066', 1, false);
                            drawArrow(cellX + 15, inputY, candX + 30, gateY + 35 + gateR, '#3fb9a066', 1, false);
                            drawArrow(cellX + 40, inputY, ogX, gateY + gateR, '#58a6ff66', 1, false);

                            // Phase indicator
                            ctx.fillStyle = '#0c0c20cc';
                            ctx.fillRect(10, H - 55, W * 0.6, 45);
                            ctx.fillStyle = phaseColors[p];
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Step ' + (p + 1) + ': ' + phaseNames[p], 20, H - 32);

                            // Phase description
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            var desc = [
                                'f(t) = \u03c3(W_f[h(t-1), x(t)] + b_f) decides what to erase from C(t-1)',
                                'i(t) and \u0108(t) propose new information to store',
                                'C(t) = f(t)\u2299C(t-1) + i(t)\u2299\u0108(t) : additive update!',
                                'h(t) = o(t)\u2299tanh(C(t)) : filtered output'
                            ];
                            ctx.fillText(desc[p], 20, H - 14);

                            // Title label
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('LSTM Cell', cellX, by - 12);
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Step', 0, 3, 0, 1, function(v) {
                            phase.value = v;
                            autoPlay = false;
                            draw();
                        });

                        var playBtn = VizEngine.createButton(controls, 'Auto-Play', function() {
                            autoPlay = !autoPlay;
                            playBtn.textContent = autoPlay ? 'Pause' : 'Auto-Play';
                        });

                        viz.animate(function(t) {
                            if (autoPlay) {
                                animTime += 0.008;
                                phase.value = Math.floor(animTime % 4);
                            }
                            draw();
                        });

                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Write out the dimensions of all weight matrices in an LSTM cell if the input dimension is \\(d_x\\) and the hidden dimension is \\(d_h\\). How many parameters does a single LSTM layer have in total?',
                    hint: 'Each gate (f, i, o) and the candidate each have a weight matrix of shape \\(d_h \\times (d_h + d_x)\\) plus a bias of shape \\(d_h\\). There are 4 such sets.',
                    solution: 'Each of the four components (forget gate, input gate, candidate, output gate) has:<br>Weight matrix: \\(W \\in \\mathbb{R}^{d_h \\times (d_h + d_x)}\\), bias: \\(b \\in \\mathbb{R}^{d_h}\\).<br>Total parameters: \\(4 \\times [d_h(d_h + d_x) + d_h] = 4d_h(d_h + d_x + 1)\\).<br>For \\(d_h = 256\\) and \\(d_x = 100\\), that is \\(4 \\times 256 \\times (256 + 100 + 1) = 365{,}568\\) parameters.'
                },
                {
                    question: 'Why is the cell state update \\(C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t\\) better than \\(C_t = \\tanh(W[C_{t-1}, x_t])\\) for preserving gradients?',
                    hint: 'Compute \\(\\frac{\\partial C_t}{\\partial C_{t-1}}\\) for both update rules and compare.',
                    solution: 'With the LSTM update: \\(\\frac{\\partial C_t}{\\partial C_{t-1}} = \\text{diag}(f_t)\\). When \\(f_t \\approx 1\\), this is close to the identity matrix, so the gradient passes through almost unchanged. The product over many steps \\(\\prod_k f_k\\) can stay close to 1.<br><br>With the vanilla-style update: \\(\\frac{\\partial C_t}{\\partial C_{t-1}} = \\text{diag}(1 - C_t^2) \\cdot W\\), which always has spectral norm \\(&lt; \\|W\\|\\), causing exponential decay or explosion. The additive structure of the LSTM creates a linear gradient highway that avoids mandatory compression through nonlinearities.'
                },
                {
                    question: 'Suppose at a certain time step, the forget gate outputs \\(f_t = [0.99, 0.01, 0.95]\\) and the input gate outputs \\(i_t = [0.02, 0.98, 0.10]\\). Interpret what the LSTM is doing with each of the three dimensions of the cell state.',
                    hint: 'A forget gate value near 1 means "keep." An input gate value near 1 means "write new information."',
                    solution: '<strong>Dimension 1</strong> (\\(f=0.99, i=0.02\\)): The LSTM keeps the existing cell state almost perfectly and writes almost nothing new. This dimension stores long-term memory that is still relevant.<br><br><strong>Dimension 2</strong> (\\(f=0.01, i=0.98\\)): The LSTM erases the old content almost completely and writes new information. This dimension is being "reset" to store something new from the current context.<br><br><strong>Dimension 3</strong> (\\(f=0.95, i=0.10\\)): The LSTM mostly retains old information and adds a small amount of new content. This represents a gradual, incremental update.'
                }
            ]
        },

        // ======================== Section 3 ========================
        {
            id: 'lstm-gate-dynamics',
            title: 'LSTM Gate Dynamics',
            content: `
<h2>LSTM Gate Dynamics</h2>

<div class="env-block env-intuition">
<div class="env-header">Gates as Learned Attention Over Time</div>
<div class="env-body">
<p>In practice, LSTM gates develop interpretable behavior. The forget gate learns to stay open for dimensions that track global properties (like the subject of a sentence or the key of a musical piece) and to close when context changes. The input gate learns to activate when genuinely new information arrives. The output gate learns to suppress irrelevant internal state from influencing the current prediction. Observing gate activations across a sequence is one of the most informative ways to understand what an LSTM has learned.</p>
</div>
</div>

<h3>Gate Activation Patterns</h3>

<p>Consider processing a sentence through a trained LSTM language model. At each time step, each gate produces a vector of activations in \\((0, 1)\\). Plotting these activations as a heatmap reveals clear structure:</p>

<ul>
<li><strong>Forget gate</strong>: Tends to stay high (\\(\\approx 1\\)) for most dimensions, with sharp drops to \\(\\approx 0\\) at sentence boundaries, clause breaks, or topic changes. This is the gate doing its job: maintaining memory by default, erasing selectively.</li>
<li><strong>Input gate</strong>: Shows bursts of activity when content words (nouns, verbs) arrive, and stays low during function words (the, a, is). This makes sense: content words carry new information to record.</li>
<li><strong>Output gate</strong>: Tends to be more uniformly active but with interpretable suppression, for example, it may reduce activation for dimensions storing long-range information that is irrelevant to the current token's prediction.</li>
</ul>

<div class="env-block env-example">
<div class="env-header">Example 13.3 &mdash; Bracket Counting</div>
<div class="env-body">
<p>A classic demonstration of LSTM memory: train an LSTM to predict closing brackets that match opening ones in nested sequences like <code>((()))</code>. A specific cell state dimension acts as a counter. When an open bracket appears, the input gate activates for that dimension and the forget gate stays open. When a close bracket appears, the forget gate slightly reduces the count. The output gate activates that dimension only when predicting the next bracket type. One can extract a near-perfect counting mechanism from the learned weights.</p>
</div>
</div>

<h3>The Peephole Connection Variant</h3>
<p>Gers &amp; Schmidhuber (2000) proposed <strong>peephole connections</strong>, which allow gates to look at the cell state directly:</p>
\\[f_t = \\sigma(W_f [h_{t-1}, x_t] + w_f \\odot C_{t-1} + b_f),\\]
\\[i_t = \\sigma(W_i [h_{t-1}, x_t] + w_i \\odot C_{t-1} + b_i),\\]
\\[o_t = \\sigma(W_o [h_{t-1}, x_t] + w_o \\odot C_t + b_o),\\]
<p>where \\(w_f, w_i, w_o \\in \\mathbb{R}^{d_h}\\) are diagonal peephole weight vectors. In practice, peepholes provide marginal improvements on some tasks but add complexity, so they are not universally used.</p>

<div class="env-block env-intuition">
<div class="env-header">Coupled Forget-Input Gates</div>
<div class="env-body">
<p>Some LSTM variants constrain \\(i_t = 1 - f_t\\), so the gate cannot simultaneously keep old information and write new information in the same dimension. This reduces parameters and enforces a tradeoff: you can either remember or rewrite, but not both at full capacity. This <em>coupled gate</em> design is one step toward the GRU architecture we will study next.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="gate-heatmap-viz"></div>

<h3>Gradient Flow Through Gates</h3>
<p>During backpropagation, the gradient through the cell state from step \\(T\\) back to step \\(t\\) is:</p>
\\[\\frac{\\partial C_T}{\\partial C_t} = \\prod_{k=t+1}^{T} \\text{diag}(f_k).\\]
<p>If all forget gates are exactly 1, this product is the identity and the gradient is perfectly preserved. In practice, forget gates are slightly less than 1, creating a <em>soft exponential decay</em> with a much longer time constant than a vanilla RNN. The effective "memory half-life" of an LSTM is:</p>
\\[\\tau_{1/2} = \\frac{\\ln 2}{-\\ln \\bar{f}},\\]
<p>where \\(\\bar{f}\\) is the average forget gate value. For \\(\\bar{f} = 0.99\\), the half-life is about 69 steps; for \\(\\bar{f} = 0.95\\), it is about 14 steps.</p>

<div class="env-block env-definition">
<div class="env-header">Definition 13.3 &mdash; Effective Memory Half-Life</div>
<div class="env-body">
<p>For an LSTM dimension with average forget gate activation \\(\\bar{f} \\in (0, 1)\\), the number of steps after which the gradient through the cell state drops to half its original magnitude is:</p>
\\[\\tau_{1/2} = \\frac{\\ln 2}{|\\ln \\bar{f}|} \\approx \\frac{0.693}{1 - \\bar{f}} \\quad \\text{when } \\bar{f} \\approx 1.\\]
</div>
</div>
`,
            visualizations: [
                {
                    id: 'gate-heatmap-viz',
                    title: 'Gate Activations Over a Text Sequence',
                    description: 'Simulated gate activations for a trained LSTM processing a sentence. Each row is a gate dimension; each column is a token. Brighter colors indicate higher activation.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 0, originY: 0 });
                        var W = viz.width;
                        var H = viz.height;
                        var ctx = viz.ctx;

                        var tokens = ['The', 'cat', ',', 'which', 'was', 'on', 'the', 'mat', ',', 'slept', '.'];
                        var nDims = 8;
                        var nTokens = tokens.length;

                        // Seed a pseudo-random generator for reproducibility
                        function seededRandom(seed) {
                            var x = Math.sin(seed) * 10000;
                            return x - Math.floor(x);
                        }

                        // Generate realistic gate patterns
                        function generateForgetGate() {
                            var data = [];
                            for (var d = 0; d < nDims; d++) {
                                var row = [];
                                for (var t = 0; t < nTokens; t++) {
                                    var base = 0.85 + seededRandom(d * 100 + t * 7 + 1) * 0.12;
                                    // Drop at commas / periods (sentence boundaries)
                                    if (tokens[t] === ',' || tokens[t] === '.') {
                                        base = 0.3 + seededRandom(d * 50 + t) * 0.3;
                                    }
                                    row.push(Math.min(1, Math.max(0, base)));
                                }
                                data.push(row);
                            }
                            return data;
                        }

                        function generateInputGate() {
                            var data = [];
                            var contentWords = { 'cat': true, 'mat': true, 'slept': true };
                            for (var d = 0; d < nDims; d++) {
                                var row = [];
                                for (var t = 0; t < nTokens; t++) {
                                    var base = 0.15 + seededRandom(d * 80 + t * 11 + 3) * 0.2;
                                    if (contentWords[tokens[t]]) {
                                        base = 0.65 + seededRandom(d * 30 + t) * 0.3;
                                    }
                                    row.push(Math.min(1, Math.max(0, base)));
                                }
                                data.push(row);
                            }
                            return data;
                        }

                        function generateOutputGate() {
                            var data = [];
                            for (var d = 0; d < nDims; d++) {
                                var row = [];
                                for (var t = 0; t < nTokens; t++) {
                                    var base = 0.4 + seededRandom(d * 60 + t * 13 + 5) * 0.4;
                                    // Higher at content words and verb
                                    if (tokens[t] === 'slept' || tokens[t] === 'cat') {
                                        base = 0.7 + seededRandom(d * 20 + t) * 0.25;
                                    }
                                    row.push(Math.min(1, Math.max(0, base)));
                                }
                                data.push(row);
                            }
                            return data;
                        }

                        var forgetData = generateForgetGate();
                        var inputData = generateInputGate();
                        var outputData = generateOutputGate();

                        var gateType = { value: 0 };
                        var gateNames = ['Forget Gate (f)', 'Input Gate (i)', 'Output Gate (o)'];
                        var gateColorSchemes = [
                            function(v) { return 'rgb(' + Math.round(248 * v) + ',' + Math.round(81 * v * 0.4 + 30) + ',' + Math.round(73 * v * 0.3 + 20) + ')'; },
                            function(v) { return 'rgb(' + Math.round(63 * v * 0.4 + 20) + ',' + Math.round(185 * v) + ',' + Math.round(80 * v * 0.5 + 20) + ')'; },
                            function(v) { return 'rgb(' + Math.round(88 * v * 0.5 + 20) + ',' + Math.round(166 * v) + ',' + Math.round(255 * v) + ')'; }
                        ];
                        var gateData = [forgetData, inputData, outputData];

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var g = Math.floor(gateType.value);
                            var data = gateData[g];
                            var colorFn = gateColorSchemes[g];

                            var padL = 80, padR = 60, padT = 50, padB = 70;
                            var plotW = W - padL - padR;
                            var plotH = H - padT - padB;

                            var cellW = plotW / nTokens;
                            var cellH = plotH / nDims;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(gateNames[g] + ' Activations', W / 2, padT - 25);

                            // Draw heatmap
                            for (var d = 0; d < nDims; d++) {
                                for (var t = 0; t < nTokens; t++) {
                                    var val = data[d][t];
                                    var x = padL + t * cellW;
                                    var y = padT + d * cellH;

                                    ctx.fillStyle = colorFn(val);
                                    ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

                                    // Value text
                                    ctx.fillStyle = val > 0.55 ? '#000' : '#ddd';
                                    ctx.font = '10px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(val.toFixed(2), x + cellW / 2, y + cellH / 2);
                                }
                            }

                            // Row labels (dimensions)
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var d = 0; d < nDims; d++) {
                                ctx.fillText('dim ' + d, padL - 8, padT + d * cellH + cellH / 2);
                            }

                            // Column labels (tokens)
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.font = '12px -apple-system,sans-serif';
                            for (var t = 0; t < nTokens; t++) {
                                ctx.fillStyle = viz.colors.white;
                                ctx.save();
                                ctx.translate(padL + t * cellW + cellW / 2, padT + plotH + 8);
                                ctx.rotate(Math.PI / 6);
                                ctx.fillText(tokens[t], 0, 0);
                                ctx.restore();
                            }

                            // Color bar
                            var barX = W - padR + 15, barY = padT, barW = 15, barH = plotH;
                            for (var i = 0; i < barH; i++) {
                                var val = 1 - i / barH;
                                ctx.fillStyle = colorFn(val);
                                ctx.fillRect(barX, barY + i, barW, 1);
                            }
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(barX, barY, barW, barH);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('1.0', barX + barW + 4, barY + 4);
                            ctx.fillText('0.0', barX + barW + 4, barY + barH);
                        }

                        draw();

                        VizEngine.createButton(controls, 'Forget', function() { gateType.value = 0; draw(); });
                        VizEngine.createButton(controls, 'Input', function() { gateType.value = 1; draw(); });
                        VizEngine.createButton(controls, 'Output', function() { gateType.value = 2; draw(); });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute the memory half-life \\(\\tau_{1/2}\\) for an LSTM dimension with average forget gate \\(\\bar{f} = 0.97\\). How does this compare to a vanilla RNN with \\(\\gamma = 0.97\\)?',
                    hint: 'Use \\(\\tau_{1/2} = \\frac{\\ln 2}{|\\ln \\bar{f}|}\\). For the vanilla RNN, the gradient decays as \\(\\gamma^n\\), so the half-life formula is the same: \\(\\frac{\\ln 2}{|\\ln \\gamma|}\\).',
                    solution: 'For \\(\\bar{f} = 0.97\\): \\(\\tau_{1/2} = \\frac{\\ln 2}{|\\ln 0.97|} = \\frac{0.693}{0.0305} \\approx 22.7\\) steps. For the vanilla RNN with \\(\\gamma = 0.97\\), the formula gives the same number: 22.7 steps. But the crucial difference is that in the LSTM, the forget gate \\(\\bar{f}\\) can be learned to be much closer to 1 (e.g., 0.995 gives \\(\\tau_{1/2} \\approx 138\\) steps), while in a vanilla RNN, \\(\\gamma\\) is constrained by the spectral norm of \\(W_h\\) times the tanh derivative, and pushing \\(\\gamma\\) close to 1 risks explosion. The LSTM decouples memory retention from the spectral properties of the weight matrix.'
                },
                {
                    question: 'In the coupled-gate variant where \\(i_t = 1 - f_t\\), the cell update becomes \\(C_t = f_t \\odot C_{t-1} + (1 - f_t) \\odot \\tilde{C}_t\\). Show that this is an exponential moving average. What is the smoothing parameter?',
                    hint: 'Compare with the standard EMA formula: \\(\\bar{x}_t = \\alpha \\bar{x}_{t-1} + (1 - \\alpha) x_t\\).',
                    solution: 'The update \\(C_t = f_t \\odot C_{t-1} + (1 - f_t) \\odot \\tilde{C}_t\\) is exactly an element-wise exponential moving average with smoothing parameter \\(\\alpha = f_t\\) (where \\(f_t\\) varies per dimension and per step). When \\(f_t\\) is high, the EMA is "slow" (long memory). When \\(f_t\\) is low, the EMA is "fast" (quickly adapts to new input). The coupled-gate LSTM can therefore be interpreted as an adaptive-rate exponential smoother that learns, at each step and for each dimension, how quickly to integrate new information.'
                },
                {
                    question: 'Why might a peephole connection (where gates observe \\(C_{t-1}\\) directly) be useful for tasks that require precise timing, like rhythm detection?',
                    hint: 'Without peepholes, the forget gate decides based only on \\(h_{t-1}\\) and \\(x_t\\). What if the decision depends on the actual magnitude of the cell state?',
                    solution: 'Without peepholes, the gates see \\(h_{t-1} = o_{t-1} \\odot \\tanh(C_{t-1})\\), which is a filtered version of the cell state. The output gate may suppress dimensions that are relevant for gate decisions. With peepholes, the gates access \\(C_{t-1}\\) directly, so they can trigger on the precise value or magnitude stored in the cell. For timing tasks, a cell dimension might act as a counter that accumulates linearly. The gate needs to fire when the count reaches a threshold. Without peepholes, this threshold information is obscured by \\(\\tanh\\) and the output gate. With peepholes, the gate can observe the raw count and fire precisely when it crosses the boundary.'
                }
            ]
        },

        // ======================== Section 4 ========================
        {
            id: 'gru-simplified-gating',
            title: 'GRU: Simplified Gating',
            content: `
<h2>GRU: Simplified Gating</h2>

<div class="env-block env-intuition">
<div class="env-header">From Four Components to Two Gates</div>
<div class="env-body">
<p>The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014), asks: can we get the benefits of gating with a simpler architecture? The GRU merges the forget and input gates into a single <strong>update gate</strong>, eliminates the separate cell state, and uses a <strong>reset gate</strong> to control how much of the previous hidden state contributes to the candidate. The result has fewer parameters and is often faster to train, while matching LSTM performance on many benchmarks.</p>
</div>
</div>

<div class="env-block env-definition">
<div class="env-header">Definition 13.4 &mdash; GRU Equations</div>
<div class="env-body">
<p>Given input \\(x_t\\) and previous hidden state \\(h_{t-1}\\):</p>
\\[z_t = \\sigma(W_z [h_{t-1}, x_t] + b_z) \\quad \\text{(update gate)}\\]
\\[r_t = \\sigma(W_r [h_{t-1}, x_t] + b_r) \\quad \\text{(reset gate)}\\]
\\[\\tilde{h}_t = \\tanh(W_h [r_t \\odot h_{t-1}, x_t] + b_h) \\quad \\text{(candidate)}\\]
\\[h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t \\quad \\text{(hidden state update)}\\]
<p>where \\(\\sigma\\) is the sigmoid function and \\(\\odot\\) denotes element-wise multiplication.</p>
</div>
</div>

<h3>The Update Gate</h3>
<p>The update gate \\(z_t\\) plays the combined role of the LSTM's forget and input gates. Examine the hidden state update:</p>
\\[h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t.\\]
<p>When \\(z_t \\approx 0\\), the GRU copies \\(h_{t-1}\\) forward unchanged (pure memory). When \\(z_t \\approx 1\\), the GRU replaces \\(h_{t-1}\\) with the candidate \\(\\tilde{h}_t\\) (full update). This is exactly the coupled-gate LSTM variant, but built into the core design.</p>

<div class="env-block env-intuition">
<div class="env-header">Update Gate as Interpolation</div>
<div class="env-body">
<p>The GRU update is a convex combination (element-wise linear interpolation) between the old state and the candidate. This ensures that the hidden state always lies "between" the old and new proposals. No explicit cell state is needed, because the hidden state itself carries long-term memory through the \\((1 - z_t)\\) path.</p>
</div>
</div>

<h3>The Reset Gate</h3>
<p>The reset gate \\(r_t\\) controls how much of \\(h_{t-1}\\) is visible when computing the candidate \\(\\tilde{h}_t\\). When \\(r_t \\approx 0\\), the candidate ignores the previous state entirely and acts like a feedforward network on \\(x_t\\) alone. When \\(r_t \\approx 1\\), the candidate has full access to \\(h_{t-1}\\).</p>

<p>This mechanism allows the GRU to "forget" in a more targeted way: rather than erasing stored information (as the LSTM forget gate does), the GRU can choose to <em>not look at</em> certain past information when forming its proposal. The actual erasure happens implicitly through the update gate.</p>

<div class="env-block env-example">
<div class="env-header">Example 13.4 &mdash; LSTM vs. GRU Correspondence</div>
<div class="env-body">
<p>The following table maps LSTM concepts to their GRU counterparts:</p>
<table style="width:100%;border-collapse:collapse;margin:10px 0;">
<thead>
<tr style="border-bottom:2px solid #30363d;">
<th style="text-align:left;padding:8px;color:#58a6ff;">LSTM</th>
<th style="text-align:left;padding:8px;color:#3fb950;">GRU</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;">Separate cell state \\(C_t\\) and hidden state \\(h_t\\)</td>
<td style="padding:8px;">Single hidden state \\(h_t\\)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;">Forget gate \\(f_t\\) + Input gate \\(i_t\\)</td>
<td style="padding:8px;">Update gate \\(z_t\\) (with \\(1 - z_t\\) for forgetting)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;">Output gate \\(o_t\\)</td>
<td style="padding:8px;">No output gate (hidden state fully exposed)</td>
</tr>
<tr>
<td style="padding:8px;">4 parameter sets</td>
<td style="padding:8px;">3 parameter sets</td>
</tr>
</tbody>
</table>
</div>
</div>

<h3>Gradient Flow in the GRU</h3>
<p>The gradient through the GRU hidden state is:</p>
\\[\\frac{\\partial h_t}{\\partial h_{t-1}} = \\text{diag}(1 - z_t) + \\text{diag}(z_t) \\cdot \\frac{\\partial \\tilde{h}_t}{\\partial h_{t-1}}.\\]
<p>The first term, \\(\\text{diag}(1 - z_t)\\), provides the same kind of linear gradient shortcut as the LSTM's cell state path. When \\(z_t \\approx 0\\), the gradient flows directly through without attenuation. This is why the GRU also handles long-range dependencies effectively.</p>

<div class="viz-placeholder" data-viz="gru-cell-viz"></div>

<div class="env-block env-warning">
<div class="env-header">GRU Limitation: No Separate Memory</div>
<div class="env-body">
<p>Because the GRU exposes its entire hidden state as output, it cannot maintain "private" memory that is stored but not expressed. In an LSTM, the cell state can hold information that the output gate suppresses. This distinction matters in tasks where the network must store information for many steps without it influencing intermediate predictions. Empirically, this advantage is modest, but for very long-range tasks, LSTMs sometimes outperform GRUs for this reason.</p>
</div>
</div>
`,
            visualizations: [
                {
                    id: 'gru-cell-viz',
                    title: 'GRU Cell vs. LSTM Cell',
                    description: 'Side-by-side comparison. Left: GRU with update gate (yellow) and reset gate (purple). Right: LSTM with all four gates. Watch the animated data flow in both architectures.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 0, originY: 0 });
                        var W = viz.width;
                        var H = viz.height;
                        var ctx = viz.ctx;
                        var animTime = 0;

                        var halfW = W / 2 - 15;

                        function drawRoundedRect(x, y, w, h, r, fill, stroke) {
                            ctx.beginPath();
                            ctx.moveTo(x + r, y);
                            ctx.lineTo(x + w - r, y);
                            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                            ctx.lineTo(x + w, y + h - r);
                            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                            ctx.lineTo(x + r, y + h);
                            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                            ctx.lineTo(x, y + r);
                            ctx.quadraticCurveTo(x, y, x + r, y);
                            ctx.closePath();
                            if (fill) { ctx.fillStyle = fill; ctx.fill(); }
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
                        }

                        function drawGateBox(cx, cy, w, h, label, color, active) {
                            drawRoundedRect(cx - w / 2, cy - h / 2, w, h, 6,
                                active ? color + '44' : '#1a1a40',
                                active ? color : color + '55');
                            ctx.fillStyle = active ? color : color + '88';
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(label, cx, cy);
                        }

                        function drawArrow(x1, y1, x2, y2, color, lw) {
                            var dx = x2 - x1, dy = y2 - y1;
                            var len = Math.sqrt(dx * dx + dy * dy);
                            if (len < 1) return;
                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 1.5;
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2 - dx / len * 6, y2 - dy / len * 6);
                            ctx.stroke();
                            var angle = Math.atan2(dy, dx);
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 8 * Math.cos(angle - Math.PI / 7), y2 - 8 * Math.sin(angle - Math.PI / 7));
                            ctx.lineTo(x2 - 8 * Math.cos(angle + Math.PI / 7), y2 - 8 * Math.sin(angle + Math.PI / 7));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function drawOp(cx, cy, symbol, color, active) {
                            ctx.beginPath();
                            ctx.arc(cx, cy, 11, 0, Math.PI * 2);
                            ctx.fillStyle = active ? color + '44' : '#1a1a40';
                            ctx.fill();
                            ctx.strokeStyle = active ? color : '#4a4a7a';
                            ctx.lineWidth = active ? 2 : 1;
                            ctx.stroke();
                            ctx.fillStyle = active ? color : '#8b949e';
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(symbol, cx, cy);
                        }

                        function draw() {
                            var pulse = (Math.sin(animTime * 2) + 1) / 2;
                            var step = Math.floor(animTime * 0.5) % 3;

                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Divider
                            ctx.strokeStyle = '#30363d';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath();
                            ctx.moveTo(W / 2, 20);
                            ctx.lineTo(W / 2, H - 20);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // === GRU (left) ===
                            var gx = halfW / 2 + 5;
                            var gy = H * 0.5;
                            drawRoundedRect(gx - halfW * 0.42, gy - H * 0.35, halfW * 0.84, H * 0.6, 12, '#14142e', '#30363d');

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('GRU', gx, gy - H * 0.35 - 12);

                            // GRU gates
                            var gzX = gx - halfW * 0.15;
                            var grX = gx + halfW * 0.15;
                            var gGateY = gy + H * 0.07;

                            drawGateBox(gzX, gGateY, 50, 28, 'z (upd)', '#d29922', step === 0);
                            drawGateBox(grX, gGateY, 50, 28, 'r (rst)', '#bc8cff', step === 1);

                            // Candidate
                            drawGateBox(gx, gGateY - 55, 52, 28, '\u0068\u0303 (cand)', '#3fb9a0', step === 1 || step === 2);

                            // Interpolation op
                            drawOp(gx, gy - H * 0.22, 'mix', '#d29922', step === 2);

                            // h(t-1) and h(t) labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('h(t-1)', gx - halfW * 0.32, gy - H * 0.22);
                            ctx.fillText('h(t)', gx + halfW * 0.32, gy - H * 0.22);
                            ctx.fillText('x(t)', gx, gy + H * 0.22);

                            // Arrows
                            drawArrow(gx - halfW * 0.25, gy - H * 0.22, gx - 14, gy - H * 0.22, '#8b949e', 1.5);
                            drawArrow(gx + 14, gy - H * 0.22, gx + halfW * 0.25, gy - H * 0.22, '#d29922', step === 2 ? 2.5 : 1.5);
                            drawArrow(gzX, gGateY - 16, gx - 8, gy - H * 0.22 + 13, '#d29922', step === 0 ? 2 : 1);
                            drawArrow(grX, gGateY - 16, gx + 8, gGateY - 55 + 16, '#bc8cff', step === 1 ? 2 : 1);
                            drawArrow(gx, gGateY - 55 + 16, gx + 5, gy - H * 0.22 + 13, '#3fb9a0', step === 2 ? 2 : 1);

                            // Param count
                            ctx.fillStyle = '#3fb9a0';
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('3 gate matrices', gx, gy + H * 0.28);
                            ctx.fillText('3d_h(d_h + d_x + 1) params', gx, gy + H * 0.33);

                            // === LSTM (right) ===
                            var lx = W / 2 + halfW / 2 + 10;
                            var ly = gy;
                            drawRoundedRect(lx - halfW * 0.42, ly - H * 0.35, halfW * 0.84, H * 0.6, 12, '#14142e', '#30363d');

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('LSTM', lx, ly - H * 0.35 - 12);

                            // LSTM gates
                            var lfX = lx - halfW * 0.25;
                            var liX = lx - halfW * 0.08;
                            var loX = lx + halfW * 0.25;
                            var lcX = lx + halfW * 0.08;
                            var lGateY = ly + H * 0.07;

                            drawGateBox(lfX, lGateY, 40, 26, 'f', '#f85149', step === 0);
                            drawGateBox(liX, lGateY, 40, 26, 'i', '#3fb950', step === 1);
                            drawGateBox(lcX, lGateY + 35, 40, 26, '\u0108', '#3fb9a0', step === 1);
                            drawGateBox(loX, lGateY, 40, 26, 'o', '#58a6ff', step === 2);

                            // Cell state line
                            var csY = ly - H * 0.25;
                            ctx.strokeStyle = '#d2992288';
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            ctx.moveTo(lx - halfW * 0.35, csY);
                            ctx.lineTo(lx + halfW * 0.35, csY);
                            ctx.stroke();

                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('C(t-1) \u2192 C(t)', lx, csY - 12);

                            // Hidden state line
                            var hsY = ly - H * 0.12;
                            ctx.fillText('h(t-1) \u2192 h(t)', lx, hsY - 10);

                            ctx.fillText('x(t)', lx, ly + H * 0.22);

                            // Param count
                            ctx.fillStyle = '#58a6ff';
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('4 gate matrices', lx, ly + H * 0.28);
                            ctx.fillText('4d_h(d_h + d_x + 1) params', lx, ly + H * 0.33);

                            // Step indicator
                            var stepNames = ['Update/Forget gates compute', 'Reset/Input + candidate', 'State interpolation/Output'];
                            ctx.fillStyle = '#0c0c20cc';
                            ctx.fillRect(10, H - 35, W - 20, 28);
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Phase ' + (step + 1) + '/3: ' + stepNames[step], W / 2, H - 20);
                        }

                        viz.animate(function(t) {
                            animTime += 0.016;
                            draw();
                        });

                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'How many parameters does a single GRU layer have, given input dimension \\(d_x\\) and hidden dimension \\(d_h\\)? Compare with the LSTM.',
                    hint: 'The GRU has 3 parameter sets (z, r, candidate) vs. the LSTM\'s 4 (f, i, candidate, o). Each set has a weight matrix of shape \\(d_h \\times (d_h + d_x)\\) plus a bias.',
                    solution: 'GRU: \\(3 \\times [d_h(d_h + d_x) + d_h] = 3d_h(d_h + d_x + 1)\\).<br>LSTM: \\(4 \\times [d_h(d_h + d_x) + d_h] = 4d_h(d_h + d_x + 1)\\).<br>The GRU uses exactly 75% of the LSTM\'s parameters. For \\(d_h = 256, d_x = 100\\): GRU has 274,176 params; LSTM has 365,568 params; a savings of 91,392 parameters. This reduction also means fewer matrix multiplications per step, making the GRU faster.'
                },
                {
                    question: 'In the GRU, what happens if \\(r_t = 0\\) for all dimensions? Write out the simplified candidate and update equations.',
                    hint: 'Substitute \\(r_t = 0\\) into \\(\\tilde{h}_t = \\tanh(W_h [r_t \\odot h_{t-1}, x_t] + b_h)\\).',
                    solution: 'When \\(r_t = 0\\):<br>\\(\\tilde{h}_t = \\tanh(W_h [\\mathbf{0}, x_t] + b_h)\\), which means the candidate depends only on \\(x_t\\) and not on any past hidden state. The candidate is computed as if the sequence started fresh at this step.<br>The update becomes \\(h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_{hx} x_t + b_h)\\), where \\(W_{hx}\\) is the portion of \\(W_h\\) multiplying \\(x_t\\).<br>This means the GRU decides how much to "rewrite" the state with information from \\(x_t\\) alone, ignoring history in the candidate but still preserving history through the \\((1 - z_t)\\) term. This is useful when the new input represents a complete context shift.'
                },
                {
                    question: 'Show that if a GRU sets \\(z_t = 0\\) for all time steps, the hidden state is constant: \\(h_t = h_0\\) for all \\(t\\). What does this imply about the GRU\'s ability to act as a pure "copy" mechanism?',
                    hint: 'Substitute \\(z_t = 0\\) into \\(h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t\\).',
                    solution: 'With \\(z_t = 0\\): \\(h_t = 1 \\odot h_{t-1} + 0 \\odot \\tilde{h}_t = h_{t-1}\\). By induction, \\(h_t = h_{t-1} = \\cdots = h_0\\) for all \\(t\\).<br><br>This shows that the GRU can perfectly copy information over an arbitrary number of steps by learning to keep the update gate at zero. The gradient in this regime is \\(\\frac{\\partial h_t}{\\partial h_{t-1}} = I\\) (the identity), so there is zero gradient decay. This is the GRU analog of the LSTM setting \\(f_t = 1, i_t = 0\\). The difference is that in the GRU, the update gate enforces a strict tradeoff: dimensions with \\(z_t \\approx 0\\) cannot incorporate new information at all.'
                }
            ]
        },

        // ======================== Section 5 ========================
        {
            id: 'lstm-vs-gru-practice',
            title: 'LSTM vs GRU in Practice',
            content: `
<h2>LSTM vs GRU in Practice</h2>

<div class="env-block env-intuition">
<div class="env-header">No Universal Winner</div>
<div class="env-body">
<p>The question "should I use an LSTM or a GRU?" has no universal answer. Extensive empirical comparisons (Chung et al. 2014, Jozefowicz et al. 2015, Greff et al. 2017) consistently find that <strong>neither architecture dominates across all tasks</strong>. The choice depends on the task, the sequence length, the dataset size, and computational constraints. This section synthesizes the practical wisdom from the literature.</p>
</div>
</div>

<h3>Parameter Efficiency</h3>
<p>As derived in the previous section, the GRU has 75% of the LSTM's parameters:</p>
<table style="width:100%;border-collapse:collapse;margin:10px 0;">
<thead>
<tr style="border-bottom:2px solid #30363d;">
<th style="text-align:left;padding:8px;color:#f0f6fc;">Architecture</th>
<th style="text-align:center;padding:8px;color:#f0f6fc;">Parameters per layer</th>
<th style="text-align:center;padding:8px;color:#f0f6fc;">Relative</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;">Vanilla RNN</td>
<td style="text-align:center;padding:8px;">\\(d_h(d_h + d_x + 1)\\)</td>
<td style="text-align:center;padding:8px;">1x</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;">GRU</td>
<td style="text-align:center;padding:8px;">\\(3d_h(d_h + d_x + 1)\\)</td>
<td style="text-align:center;padding:8px;">3x</td>
</tr>
<tr>
<td style="padding:8px;">LSTM</td>
<td style="text-align:center;padding:8px;">\\(4d_h(d_h + d_x + 1)\\)</td>
<td style="text-align:center;padding:8px;">4x</td>
</tr>
</tbody>
</table>

<p>With fewer parameters, the GRU trains faster per step and may generalize better on small datasets. Conversely, the LSTM has more capacity, which can be advantageous when data is plentiful.</p>

<h3>Empirical Guidelines</h3>

<div class="env-block env-example">
<div class="env-header">When to Prefer GRU</div>
<div class="env-body">
<ul>
<li><strong>Small datasets</strong>: Fewer parameters mean less risk of overfitting.</li>
<li><strong>Real-time applications</strong>: The GRU is computationally cheaper per step.</li>
<li><strong>Moderate sequence lengths</strong> (up to ~200 tokens): GRU and LSTM perform comparably.</li>
<li><strong>Speech and audio</strong>: GRUs often match LSTMs with faster training on speech recognition tasks.</li>
</ul>
</div>
</div>

<div class="env-block env-example">
<div class="env-header">When to Prefer LSTM</div>
<div class="env-body">
<ul>
<li><strong>Very long sequences</strong> (500+ tokens): The separate cell state gives the LSTM additional capacity for long-range memory.</li>
<li><strong>Language modeling</strong>: On large-scale benchmarks like Penn Treebank and WikiText, LSTMs with careful tuning tend to edge out GRUs.</li>
<li><strong>Tasks requiring internal memory not reflected in output</strong>: The output gate allows the LSTM to store "private" information.</li>
<li><strong>When stacking many layers</strong>: The LSTM's extra gating gives more control over information flow in deep architectures.</li>
</ul>
</div>
</div>

<h3>The Greff et al. (2017) Study</h3>
<p>In one of the most comprehensive LSTM ablation studies, Greff et al. tested eight LSTM variants across multiple tasks and found:</p>
<ol>
<li><strong>The forget gate and the output activation function are the most critical components.</strong> Removing either causes significant degradation.</li>
<li><strong>The coupled input-forget gate</strong> (\\(i_t = 1 - f_t\\)) performs nearly as well as the standard LSTM.</li>
<li><strong>Peephole connections</strong> and <strong>full gate recurrence</strong> do not significantly improve performance.</li>
<li><strong>The GRU is comparable to the LSTM</strong> on the tasks studied, with neither consistently outperforming the other.</li>
</ol>

<h3>Modern Context: Transformers</h3>
<p>Since the introduction of the Transformer (Vaswani et al. 2017), attention-based architectures have largely replaced LSTMs and GRUs for most NLP tasks. However, recurrent architectures remain relevant in several settings:</p>
<ul>
<li><strong>Streaming and online processing</strong>: RNNs process one token at a time with constant memory, while Transformers need the full sequence.</li>
<li><strong>Edge devices</strong>: LSTMs and GRUs are compact and well-optimized for inference on mobile hardware.</li>
<li><strong>Time series forecasting</strong>: Recurrent models remain competitive and are widely used.</li>
<li><strong>State-space models</strong>: Modern architectures like S4, Mamba, and RWKV draw on ideas from recurrence and gating, showing that the principles behind LSTM/GRU are far from obsolete.</li>
</ul>

<div class="env-block env-intuition">
<div class="env-header">The Legacy of Gating</div>
<div class="env-body">
<p>Even though pure LSTMs and GRUs are no longer state-of-the-art for most tasks, the gating mechanism they introduced has become a fundamental building block. Highway networks, residual connections, gated convolutions, and even the gating in modern state-space models all descend from the same core idea: let the network learn, at each layer or step, how much information to pass through unchanged. This is arguably the single most important architectural idea in deep learning after backpropagation itself.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="training-curves-viz"></div>
`,
            visualizations: [
                {
                    id: 'training-curves-viz',
                    title: 'Training Loss: LSTM vs GRU vs Vanilla RNN',
                    description: 'Simulated training curves on sequence modeling tasks of different lengths. Short sequences (~20 tokens): all three converge. Medium (~50): RNN diverges. Long (~100+): LSTM has a slight edge over GRU. Adjust sequence length to see.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 0, originY: 0 });
                        var W = viz.width;
                        var H = viz.height;
                        var ctx = viz.ctx;
                        var seqLen = { value: 50 };
                        var maxEpochs = 100;

                        function seededRandom(seed) {
                            var x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
                            return x - Math.floor(x);
                        }

                        function generateCurve(finalLoss, convergenceRate, noiseLevel, divergeAfter, seed) {
                            var curve = [];
                            for (var e = 0; e < maxEpochs; e++) {
                                var base;
                                if (divergeAfter > 0 && e > divergeAfter) {
                                    // Divergence / plateau
                                    base = finalLoss + 0.3 + (e - divergeAfter) * 0.005;
                                    base = Math.min(base, 2.5);
                                } else {
                                    base = finalLoss + (2.0 - finalLoss) * Math.exp(-convergenceRate * e);
                                }
                                var noise = (seededRandom(seed + e * 7.31) - 0.5) * noiseLevel;
                                curve.push(Math.max(0.05, base + noise));
                            }
                            return curve;
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var sl = seqLen.value;
                            var padL = 70, padR = 40, padT = 45, padB = 50;
                            var plotW = W - padL - padR;
                            var plotH = H - padT - padB;

                            // Compute curves based on sequence length
                            var rnnDiverge = sl > 30 ? Math.max(10, 50 - sl * 0.5) : -1;
                            var rnnFinal = sl > 30 ? 1.5 : 0.5 + sl * 0.01;
                            var gruFinal = 0.25 + sl * 0.003;
                            var lstmFinal = 0.20 + sl * 0.002;
                            var gruRate = 0.06 - sl * 0.0002;
                            var lstmRate = 0.055 - sl * 0.00015;
                            var rnnRate = 0.08 - sl * 0.0005;

                            var rnnCurve = generateCurve(rnnFinal, Math.max(0.01, rnnRate), 0.06, rnnDiverge, 42);
                            var gruCurve = generateCurve(gruFinal, Math.max(0.02, gruRate), 0.03, -1, 137);
                            var lstmCurve = generateCurve(lstmFinal, Math.max(0.02, lstmRate), 0.025, -1, 256);

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.moveTo(padL, padT);
                            ctx.lineTo(padL, padT + plotH);
                            ctx.lineTo(padL + plotW, padT + plotH);
                            ctx.stroke();

                            // Labels
                            ctx.save();
                            ctx.translate(18, padT + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Training Loss', 0, 0);
                            ctx.restore();

                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Epoch', padL + plotW / 2, padT + plotH + 35);

                            // Y-axis ticks (loss 0 to 2.5)
                            var maxLoss = 2.5;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var i = 0; i <= 5; i++) {
                                var yVal = i * 0.5;
                                var yPx = padT + plotH - (yVal / maxLoss) * plotH;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(yVal.toFixed(1), padL - 8, yPx);
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.moveTo(padL, yPx);
                                ctx.lineTo(padL + plotW, yPx);
                                ctx.stroke();
                            }

                            // X-axis ticks
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (var i = 0; i <= maxEpochs; i += 20) {
                                var xPx = padL + (i / maxEpochs) * plotW;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(i, xPx, padT + plotH + 5);
                            }

                            // Draw curves
                            function drawCurve(data, color) {
                                ctx.strokeStyle = color;
                                ctx.lineWidth = 2.5;
                                ctx.beginPath();
                                for (var e = 0; e < data.length; e++) {
                                    var xPx = padL + (e / maxEpochs) * plotW;
                                    var yPx = padT + plotH - (Math.min(data[e], maxLoss) / maxLoss) * plotH;
                                    if (e === 0) ctx.moveTo(xPx, yPx);
                                    else ctx.lineTo(xPx, yPx);
                                }
                                ctx.stroke();
                            }

                            drawCurve(rnnCurve, viz.colors.red);
                            drawCurve(gruCurve, viz.colors.green);
                            drawCurve(lstmCurve, viz.colors.blue);

                            // Legend
                            var legX = padL + plotW - 170, legY = padT + 15;
                            ctx.fillStyle = 'rgba(12,12,32,0.9)';
                            ctx.fillRect(legX - 10, legY - 10, 185, 80);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(legX - 10, legY - 10, 185, 80);

                            var labels = [
                                { name: 'Vanilla RNN', color: viz.colors.red },
                                { name: 'GRU', color: viz.colors.green },
                                { name: 'LSTM', color: viz.colors.blue }
                            ];
                            for (var i = 0; i < labels.length; i++) {
                                ctx.strokeStyle = labels[i].color;
                                ctx.lineWidth = 2.5;
                                ctx.beginPath();
                                ctx.moveTo(legX, legY + i * 22 + 5);
                                ctx.lineTo(legX + 25, legY + i * 22 + 5);
                                ctx.stroke();
                                ctx.fillStyle = labels[i].color;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(labels[i].name, legX + 30, legY + i * 22 + 5);
                            }

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Training Curves (Sequence Length = ' + sl + ')', padL + plotW / 2, padT - 18);

                            // Final loss annotations
                            ctx.font = '10px monospace';
                            ctx.textAlign = 'left';
                            var finalX = padL + plotW + 5;
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText(rnnCurve[maxEpochs - 1].toFixed(2), finalX, padT + plotH - (Math.min(rnnCurve[maxEpochs - 1], maxLoss) / maxLoss) * plotH);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText(gruCurve[maxEpochs - 1].toFixed(2), finalX, padT + plotH - (gruCurve[maxEpochs - 1] / maxLoss) * plotH);
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText(lstmCurve[maxEpochs - 1].toFixed(2), finalX, padT + plotH - (lstmCurve[maxEpochs - 1] / maxLoss) * plotH);
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Seq Length', 10, 150, seqLen.value, 5, function(v) {
                            seqLen.value = v;
                            draw();
                        });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'You are building a real-time speech recognition system for a mobile device with limited memory. The input sequences are about 100 frames long. Would you choose an LSTM or a GRU? Justify your answer with parameter and computation considerations.',
                    hint: 'Consider: (1) GRU has 75% of LSTM parameters. (2) GRU computes 3 matrix multiplications per step vs. 4 for LSTM. (3) At sequence length 100, both perform similarly. (4) Mobile devices have tight memory and latency budgets.',
                    solution: 'For this application, the <strong>GRU is the better choice</strong>:<br><br>1. <strong>Parameters</strong>: 25% fewer parameters means less memory usage on the mobile device.<br>2. <strong>Computation</strong>: 3 matrix multiplications per step instead of 4, giving roughly a 25% speedup per step. Over 100 frames, this adds up significantly for real-time processing.<br>3. <strong>Performance</strong>: At sequence length 100, empirical studies show GRUs and LSTMs perform comparably on speech tasks (Chung et al. 2014).<br>4. <strong>Latency</strong>: The GRU\'s lower per-step cost directly translates to lower latency for streaming inference.<br><br>The LSTM\'s advantage with very long sequences is not relevant here since 100 frames is well within both architectures\' effective range.'
                },
                {
                    question: 'In the Greff et al. (2017) ablation study, removing the forget gate caused the largest performance drop. Explain intuitively why the forget gate is more critical than the input gate.',
                    hint: 'What happens to the cell state if it can never forget? What happens if it can never write new information (but can still forget)?',
                    solution: 'Without a forget gate (\\(f_t = 1\\) always), the cell state <em>accumulates</em> indefinitely: \\(C_t = C_{t-1} + i_t \\odot \\tilde{C}_t\\). Over a long sequence, values grow without bound, eventually saturating the \\(\\tanh\\) in \\(h_t = o_t \\odot \\tanh(C_t)\\) and destroying the gradient (since \\(\\tanh\'(x) \\to 0\\) for large \\(|x|\\)). This is catastrophic.<br><br>Without an input gate (\\(i_t = 1\\) always), the cell state always gets the full candidate added: \\(C_t = f_t \\odot C_{t-1} + \\tilde{C}_t\\). This is noisier, because irrelevant information is written at every step. But the forget gate can still erase this noise at subsequent steps, so the damage is contained. The forget gate thus plays a more structurally essential role: it prevents unbounded accumulation and gives the network the ability to "reset" when context changes.'
                },
                {
                    question: 'Modern state-space models like Mamba use a "selective" mechanism that is conceptually similar to LSTM gating. Describe the analogy: what in Mamba corresponds to the forget gate, and what corresponds to the input gate?',
                    hint: 'Mamba\'s core equation is \\(h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t\\), where \\(\\bar{A}_t\\) and \\(\\bar{B}_t\\) are input-dependent (selective). Compare with \\(C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t\\).',
                    solution: 'The analogy is direct:<br><br>Mamba: \\(h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t\\)<br>LSTM cell: \\(C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t\\)<br><br><strong>\\(\\bar{A}_t\\) corresponds to the forget gate \\(f_t\\)</strong>: it controls how much of the previous state is retained. In non-selective (linear time-invariant) SSMs, \\(\\bar{A}\\) is constant, analogous to a fixed forget rate. Mamba makes \\(\\bar{A}_t\\) input-dependent, exactly like the LSTM forget gate.<br><br><strong>\\(\\bar{B}_t\\) corresponds to the input gate \\(i_t\\)</strong>: it controls how much of the current input is integrated into the state. Making it selective means the model can choose to ignore irrelevant inputs, just as the LSTM input gate does.<br><br>The key insight is that "selectivity" in Mamba is essentially the same principle as "gating" in LSTMs: making the state dynamics input-dependent so the model can learn what to remember and what to ignore. Mamba achieves this while maintaining the efficient parallel scan computation of SSMs, combining the best of both worlds.'
                }
            ]
        }
    ]
});
