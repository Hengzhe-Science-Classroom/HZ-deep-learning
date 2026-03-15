// === Chapter 15: Transformer Architecture ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch15',
    number: 15,
    title: 'Transformer Architecture',
    subtitle: 'Self-attention, multi-head mechanisms, positional encoding, and the architecture that replaced recurrence',
    sections: [
        // ======================== Section 1 ========================
        {
            id: 'self-attention',
            title: 'Self-Attention',
            content: `
<h2>Self-Attention</h2>

<div class="env-block intuition">
<div class="env-title">From Cross-Attention to Self-Attention</div>
<div class="env-body">
<p>In Chapter 14, we saw how attention allows a decoder to selectively focus on encoder hidden states. The <strong>self-attention</strong> mechanism takes this idea one step further: instead of attending from one sequence to another, every position in a <em>single</em> sequence attends to every other position in that same sequence. This lets each token gather contextual information from all other tokens, regardless of distance, in a single computation step.</p>
</div>
</div>

<p>Self-attention was introduced as a core building block of the Transformer by Vaswani et al. (2017) in "Attention Is All You Need." The key insight is that recurrence is unnecessary: a purely attention-based architecture can capture all pairwise relationships in a sequence simultaneously, with \\(O(1)\\) path length between any two positions.</p>

<h3>Query, Key, Value</h3>

<p>Self-attention is built on three linear projections of the input. Given a sequence of \\(n\\) token embeddings \\(x_1, x_2, \\ldots, x_n \\in \\mathbb{R}^{d_{\\text{model}}}\\), we compute:</p>

<div class="env-block definition">
<div class="env-title">Definition 15.1 &mdash; Query, Key, Value Projections</div>
<div class="env-body">
<p>For each token embedding \\(x_i\\), we compute three vectors:</p>
\\[q_i = W_Q x_i, \\quad k_i = W_K x_i, \\quad v_i = W_V x_i\\]
<p>where \\(W_Q, W_K \\in \\mathbb{R}^{d_k \\times d_{\\text{model}}}\\) and \\(W_V \\in \\mathbb{R}^{d_v \\times d_{\\text{model}}}\\) are learnable weight matrices. In matrix form, stacking all tokens row-wise into \\(X \\in \\mathbb{R}^{n \\times d_{\\text{model}}}\\):</p>
\\[Q = X W_Q^\\top, \\quad K = X W_K^\\top, \\quad V = X W_V^\\top\\]
<p>where \\(Q \\in \\mathbb{R}^{n \\times d_k}\\), \\(K \\in \\mathbb{R}^{n \\times d_k}\\), \\(V \\in \\mathbb{R}^{n \\times d_v}\\).</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">The Library Analogy</div>
<div class="env-body">
<p>Think of self-attention as a library lookup system. Each token generates a <strong>query</strong> ("What information am I looking for?"), a <strong>key</strong> ("What information do I contain?"), and a <strong>value</strong> ("What information do I provide if selected?"). The attention score between tokens \\(i\\) and \\(j\\) is the dot product \\(q_i \\cdot k_j\\): how well token \\(i\\)'s query matches token \\(j\\)'s key. High-scoring keys have their values weighted more heavily in the output.</p>
</div>
</div>

<h3>The Self-Attention Computation</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.2 &mdash; Self-Attention Output</div>
<div class="env-body">
<p>The self-attention output for position \\(i\\) is the weighted sum of all value vectors:</p>
\\[\\text{Attn}(q_i, K, V) = \\sum_{j=1}^{n} \\alpha_{ij} \\, v_j\\]
<p>where the attention weights are obtained by softmax over scaled dot-product scores:</p>
\\[\\alpha_{ij} = \\frac{\\exp(q_i \\cdot k_j / \\sqrt{d_k})}{\\sum_{l=1}^{n} \\exp(q_i \\cdot k_l / \\sqrt{d_k})}\\]
<p>In matrix form:</p>
\\[\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V\\]
</div>
</div>

<p>The computation proceeds in four stages: (1) compute the score matrix \\(S = QK^\\top\\), (2) scale by \\(1/\\sqrt{d_k}\\), (3) apply softmax row-wise, (4) multiply by \\(V\\). Each stage is a simple matrix operation, making self-attention highly parallelizable on modern hardware.</p>

<div class="env-block example">
<div class="env-title">Example 15.1 &mdash; Self-Attention on Three Tokens</div>
<div class="env-body">
<p>Consider the sentence "The cat sat" with \\(d_k = d_v = 2\\). Suppose after projection:</p>
\\[Q = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \\end{pmatrix}, \\quad K = \\begin{pmatrix} 1 & 1 \\\\ 0 & 1 \\\\ 1 & 0 \\end{pmatrix}, \\quad V = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \\end{pmatrix}\\]
<p>Step 1: \\(QK^\\top = \\begin{pmatrix} 1 & 0 & 1 \\\\ 1 & 1 & 0 \\\\ 2 & 1 & 1 \\end{pmatrix}\\). Step 2: Scale by \\(1/\\sqrt{2} \\approx 0.707\\): \\(\\begin{pmatrix} 0.71 & 0 & 0.71 \\\\ 0.71 & 0.71 & 0 \\\\ 1.41 & 0.71 & 0.71 \\end{pmatrix}\\). Step 3: Softmax per row gives the attention weight matrix. Step 4: Multiply by \\(V\\) to get the output.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Complexity</div>
<div class="env-body">
<p>Self-attention has time complexity \\(O(n^2 d)\\) and memory complexity \\(O(n^2 + nd)\\), where \\(n\\) is the sequence length and \\(d\\) is the model dimension. The \\(n^2\\) term arises from computing all pairwise attention scores. This quadratic scaling is the primary bottleneck for long sequences, motivating efficient variants like FlashAttention, sparse attention, and linear attention.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="self-attention-viz"></div>

<p>The visualization above animates the four stages of self-attention step by step. Observe how the raw score matrix \\(QK^\\top\\) is scaled, then softmax sharpens the distribution, and finally the value matrix is weighted to produce the output.</p>
`,
            visualizations: [
                {
                    id: 'self-attention-viz',
                    title: 'Self-Attention Step by Step',
                    description: 'Watch the four stages of self-attention: Q*K^T, scaling, softmax, and weighting by V. Use the step slider to advance through each stage.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 430, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var step = { value: 0 };
                        var tokens = ['The', 'cat', 'sat'];
                        var n = tokens.length;
                        // Fixed Q, K, V
                        var Q = [[1,0],[0,1],[1,1]];
                        var K = [[1,1],[0,1],[1,0]];
                        var V = [[1,0],[0,1],[1,1]];
                        var dk = 2;

                        function matMul(A, B_T) {
                            var r = A.length, c = B_T.length;
                            var out = [];
                            for (var i = 0; i < r; i++) {
                                out[i] = [];
                                for (var j = 0; j < c; j++) {
                                    var s = 0;
                                    for (var k = 0; k < A[i].length; k++) s += A[i][k] * B_T[j][k];
                                    out[i][j] = s;
                                }
                            }
                            return out;
                        }
                        function softmaxRows(M) {
                            return M.map(function(row) {
                                var mx = Math.max.apply(null, row);
                                var exps = row.map(function(v) { return Math.exp(v - mx); });
                                var s = exps.reduce(function(a,b){return a+b;}, 0);
                                return exps.map(function(e) { return e / s; });
                            });
                        }
                        var raw = matMul(Q, K);
                        var scaled = raw.map(function(row) { return row.map(function(v) { return v / Math.sqrt(dk); }); });
                        var attnW = softmaxRows(scaled);
                        var output = [];
                        for (var i = 0; i < n; i++) {
                            output[i] = [0, 0];
                            for (var j = 0; j < n; j++) {
                                output[i][0] += attnW[i][j] * V[j][0];
                                output[i][1] += attnW[i][j] * V[j][1];
                            }
                        }

                        function drawMatrix(mat, x, y, cellW, cellH, title, highlight, fmt) {
                            var rows = mat.length, cols = mat[0].length;
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText(title, x + cols * cellW / 2, y - 6);
                            for (var i = 0; i < rows; i++) {
                                for (var j = 0; j < cols; j++) {
                                    var val = mat[i][j];
                                    var cx = x + j * cellW, cy = y + i * cellH;
                                    if (highlight) {
                                        var alpha = Math.min(1, Math.abs(val) / 2);
                                        ctx.fillStyle = val >= 0 ? 'rgba(88,166,255,' + (0.15 + alpha * 0.6) + ')' : 'rgba(248,81,73,' + (0.15 + alpha * 0.6) + ')';
                                    } else {
                                        ctx.fillStyle = '#1a1a40';
                                    }
                                    ctx.fillRect(cx, cy, cellW - 2, cellH - 2);
                                    ctx.strokeStyle = '#30363d';
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(cx, cy, cellW - 2, cellH - 2);
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = '11px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    var display = fmt ? val.toFixed(fmt) : val.toFixed(2);
                                    ctx.fillText(display, cx + cellW / 2 - 1, cy + cellH / 2 - 1);
                                }
                            }
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);
                            var s = Math.round(step.value);
                            var labels = ['Step 1: Q K\u1d40 (raw scores)', 'Step 2: Scale by 1/\u221Ad\u2096', 'Step 3: Softmax (row-wise)', 'Step 4: Attention \u00d7 V = Output'];
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText(labels[s], viz.width / 2, 10);

                            // Token labels on left
                            var matX = 220, matY = 70, cW = 60, cH = 34;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var i = 0; i < n; i++) {
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText(tokens[i], matX - 10, matY + i * cH + cH / 2 - 1);
                            }

                            if (s === 0) {
                                // Show Q, K^T, and result
                                drawMatrix(Q, 20, 200, 50, 34, 'Q', false, 1);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '16px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('\u00d7', 140, 230);
                                drawMatrix(K.map(function(r,i){return K.map(function(row){return row[i];});}), 160, 200, 50, 34, 'K\u1d40', false, 1);
                                ctx.fillText('=', 320, 230);
                                drawMatrix(raw, matX + 120, 200, cW, cH, 'QK\u1d40', true);
                                // Main display
                                drawMatrix(raw, matX, matY, cW, cH, 'Score Matrix (QK\u1d40)', true);
                            } else if (s === 1) {
                                drawMatrix(scaled, matX, matY, cW, cH, 'Scaled Scores (QK\u1d40 / \u221Ad\u2096)', true);
                                // Annotation
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('d\u2096 = ' + dk + ',  1/\u221Ad\u2096 = ' + (1/Math.sqrt(dk)).toFixed(3), matX, matY + n * cH + 20);
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText('Scaling prevents dot products from growing too large', matX, matY + n * cH + 40);
                            } else if (s === 2) {
                                drawMatrix(attnW, matX, matY, cW, cH, 'Attention Weights (softmax)', true);
                                // Show row sums
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                for (var i = 0; i < n; i++) {
                                    var rowSum = attnW[i].reduce(function(a,b){return a+b;},0);
                                    ctx.fillText('\u2211 = ' + rowSum.toFixed(2), matX + n * cW + 10, matY + i * cH + cH / 2);
                                }
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('Each row sums to 1 (valid probability distribution)', matX, matY + n * cH + 20);
                            } else {
                                drawMatrix(attnW, matX - 130, matY + 80, 52, 30, 'Attn Weights', true);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '16px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('\u00d7', matX - 130 + 3 * 52 + 15, matY + 80 + 45);
                                drawMatrix(V, matX - 130 + 3 * 52 + 30, matY + 80, 50, 30, 'V', false, 1);
                                ctx.fillText('=', matX - 130 + 3 * 52 + 30 + 2 * 50 + 15, matY + 80 + 45);
                                drawMatrix(output, matX - 130 + 3 * 52 + 30 + 2 * 50 + 30, matY + 80, 55, 30, 'Output', true);
                                // Token flow viz at top
                                var flowY = 50;
                                var flowSpacing = 120;
                                var flowStartX = (viz.width - (n - 1) * flowSpacing) / 2;
                                for (var i = 0; i < n; i++) {
                                    var fx = flowStartX + i * flowSpacing;
                                    ctx.fillStyle = viz.colors.blue + '44';
                                    ctx.beginPath(); ctx.arc(fx, flowY, 20, 0, Math.PI * 2); ctx.fill();
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = '12px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(tokens[i], fx, flowY);
                                }
                            }

                            // Draw token column headers for score matrices
                            if (s < 3) {
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';
                                for (var j = 0; j < n; j++) {
                                    ctx.fillStyle = viz.colors.blue;
                                    ctx.fillText(tokens[j], matX + j * cW + cW / 2, matY - 18);
                                }
                            }

                            // Stage indicators at bottom
                            var indY = viz.height - 30;
                            for (var i = 0; i < 4; i++) {
                                var ix = viz.width / 2 - 90 + i * 60;
                                ctx.beginPath();
                                ctx.arc(ix, indY, 8, 0, Math.PI * 2);
                                ctx.fillStyle = i === s ? viz.colors.blue : '#30363d';
                                ctx.fill();
                                ctx.fillStyle = i === s ? viz.colors.white : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(i + 1, ix, indY);
                            }
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Stage', 0, 3, 0, 1, function(v) { step.value = v; draw(); });
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'In self-attention, each token produces a query, key, and value. If \\(d_{\\text{model}} = 512\\) and \\(d_k = d_v = 64\\), how many parameters are in the three projection matrices \\(W_Q, W_K, W_V\\) combined (excluding biases)?',
                    hint: 'Each projection matrix maps from \\(d_{\\text{model}}\\) to \\(d_k\\) or \\(d_v\\).',
                    solution: 'Each of \\(W_Q\\) and \\(W_K\\) has shape \\(d_k \\times d_{\\text{model}} = 64 \\times 512 = 32{,}768\\) parameters. \\(W_V\\) has \\(d_v \\times d_{\\text{model}} = 64 \\times 512 = 32{,}768\\) parameters. Total: \\(3 \\times 32{,}768 = 98{,}304\\) parameters.'
                },
                {
                    question: 'Why does self-attention have \\(O(n^2)\\) complexity while an RNN has \\(O(n)\\) sequential steps? What is the tradeoff?',
                    hint: 'Think about the attention score matrix dimensions and whether the operations can be parallelized.',
                    solution: '<p>Self-attention computes a score between every pair of positions, producing an \\(n \\times n\\) attention matrix. This requires \\(O(n^2 d)\\) multiply-add operations. An RNN processes tokens one at a time, with \\(O(nd^2)\\) work per step (matrix-vector multiply), for \\(O(n d^2)\\) total.</p><p>However, self-attention has \\(O(1)\\) <em>sequential</em> depth (all pairs are computed in parallel), while an RNN has \\(O(n)\\) sequential depth (each step depends on the previous). The tradeoff: self-attention is more parallelizable and captures long-range dependencies in one step, but uses more memory (\\(O(n^2)\\)) and becomes expensive for very long sequences. RNNs are memory-efficient (\\(O(n)\\)) but suffer from vanishing gradients and cannot parallelize across time steps.</p>'
                },
                {
                    question: 'Suppose we remove the value projection entirely and set \\(V = X\\) (the raw embeddings). Does the model lose expressiveness? Explain.',
                    hint: 'Consider what role \\(W_V\\) plays in determining what information flows through attention.',
                    solution: '<p>Yes, expressiveness is reduced. The value projection \\(W_V\\) controls <em>what information is transmitted</em> when a token is attended to. Without it, the attention output is a weighted average of raw input embeddings. With \\(W_V\\), the model can learn to expose different aspects of the representation to different attention patterns. For instance, \\(W_V\\) might project out the syntactic features of a token even if its embedding also encodes semantic features. Removing \\(W_V\\) forces all attention heads (in multi-head attention) to work with the same value representation, eliminating the ability to extract diverse information from the same positions.</p>'
                }
            ]
        },

        // ======================== Section 2 ========================
        {
            id: 'scaled-dot-product',
            title: 'Scaled Dot-Product Attention',
            content: `
<h2>Scaled Dot-Product Attention</h2>

<div class="env-block intuition">
<div class="env-title">Why Scale?</div>
<div class="env-body">
<p>The raw dot product \\(q_i \\cdot k_j\\) grows in magnitude with the dimension \\(d_k\\). When \\(d_k\\) is large, some dot products become very large in absolute value, pushing the softmax into regions where its gradient is extremely small. Scaling by \\(1/\\sqrt{d_k}\\) counteracts this growth, keeping the softmax in a well-behaved regime.</p>
</div>
</div>

<h3>The Variance Argument</h3>

<p>To understand the scaling factor precisely, consider two random vectors \\(q, k \\in \\mathbb{R}^{d_k}\\) with entries drawn independently from a distribution with mean 0 and variance 1. The dot product is:</p>
\\[q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i\\]

<div class="env-block definition">
<div class="env-title">Definition 15.3 &mdash; Variance of Dot Products</div>
<div class="env-body">
<p>If \\(q_i\\) and \\(k_i\\) are independent with \\(\\mathbb{E}[q_i] = \\mathbb{E}[k_i] = 0\\) and \\(\\text{Var}(q_i) = \\text{Var}(k_i) = 1\\), then each product \\(q_i k_i\\) has mean 0 and variance 1 (since \\(\\text{Var}(q_i k_i) = \\mathbb{E}[q_i^2 k_i^2] - (\\mathbb{E}[q_i k_i])^2 = 1 \\cdot 1 - 0 = 1\\)). By independence across components:</p>
\\[\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = d_k\\]
<p>Thus \\(q \\cdot k\\) has standard deviation \\(\\sqrt{d_k}\\). Dividing by \\(\\sqrt{d_k}\\) normalizes the variance to 1, regardless of \\(d_k\\).</p>
</div>
</div>

<h3>Softmax Saturation</h3>

<p>The softmax function \\(\\sigma(z)_i = \\exp(z_i) / \\sum_j \\exp(z_j)\\) is sensitive to the scale of its inputs. When the input values are large in magnitude, the softmax output approaches a one-hot vector (all mass on the maximum element), and the gradients become vanishingly small.</p>

<div class="env-block definition">
<div class="env-title">Definition 15.4 &mdash; Softmax Temperature</div>
<div class="env-body">
<p>The <strong>temperature-scaled softmax</strong> is defined as:</p>
\\[\\sigma(z / \\tau)_i = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}\\]
<p>As \\(\\tau \\to 0\\), the distribution becomes a hard argmax. As \\(\\tau \\to \\infty\\), the distribution becomes uniform. The Transformer's \\(1/\\sqrt{d_k}\\) scaling is equivalent to using temperature \\(\\tau = \\sqrt{d_k}\\).</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 15.2 &mdash; Saturation in Practice</div>
<div class="env-body">
<p>Suppose \\(d_k = 512\\) and \\(q \\cdot k\\) happens to be \\(+2\\sqrt{d_k} \\approx 45.3\\) for one pair and 0 for others. Without scaling, \\(\\text{softmax}(45.3, 0, 0, 0) \\approx (1.0, 0, 0, 0)\\): a nearly hard assignment with gradient close to zero. After scaling: \\(\\text{softmax}(2, 0, 0, 0) \\approx (0.58, 0.14, 0.14, 0.14)\\): a soft distribution with healthy gradients.</p>
</div>
</div>

<div class="env-block warning">
<div class="env-title">Additive Attention Does Not Need Scaling</div>
<div class="env-body">
<p>Bahdanau's additive attention computes scores via a learned network \\(v^\\top \\tanh(W_a s + U_a h)\\). The \\(\\tanh\\) nonlinearity naturally bounds the scores to \\([-d_a, d_a]\\) regardless of the input dimensionality. Dot-product attention lacks this built-in bound, which is why explicit scaling is necessary. The tradeoff: dot-product attention is faster (matrix multiply vs. feedforward network) but requires the scaling fix.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Connection to Kernel Methods</div>
<div class="env-body">
<p>The softmax attention can be viewed through the lens of kernel methods. Define the kernel \\(\\kappa(q, k) = \\exp(q \\cdot k / \\sqrt{d_k})\\). Then the attention output is a <em>Nadaraya-Watson kernel regression estimate</em>: a normalized weighted average of values, where weights come from the kernel evaluated at query-key pairs. This connection has inspired linear attention variants that replace the exponential kernel with other feature maps, reducing complexity from \\(O(n^2)\\) to \\(O(n)\\).</p>
</div>
</div>

<div class="viz-placeholder" data-viz="scaling-viz"></div>

<p>The visualization above shows the attention score distribution with and without scaling. For large \\(d_k\\), the unscaled distribution collapses to near-one-hot (softmax saturation), while the scaled version maintains a well-spread distribution with healthy gradients.</p>
`,
            visualizations: [
                {
                    id: 'scaling-viz',
                    title: 'Effect of Scaling on Softmax',
                    description: 'Adjust \\(d_k\\) to see how the attention distribution changes with and without scaling. Higher dimensions cause the unscaled softmax to saturate.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var dimK = { value: 64 };

                        function softmax(arr) {
                            var mx = Math.max.apply(null, arr);
                            var exps = arr.map(function(v) { return Math.exp(v - mx); });
                            var s = exps.reduce(function(a, b) { return a + b; }, 0);
                            return exps.map(function(e) { return e / s; });
                        }

                        // Seeded pseudo-random
                        function seededRand(seed) {
                            var x = Math.sin(seed) * 10000;
                            return x - Math.floor(x);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var dk = Math.round(dimK.value);
                            var nTokens = 8;
                            // Generate deterministic "dot products" that scale with sqrt(dk)
                            var rawScores = [];
                            for (var i = 0; i < nTokens; i++) {
                                rawScores.push((seededRand(i * 7 + 3) - 0.3) * Math.sqrt(dk) * 1.5);
                            }

                            var unscaled = softmax(rawScores);
                            var scaledScores = rawScores.map(function(v) { return v / Math.sqrt(dk); });
                            var scaledSm = softmax(scaledScores);

                            var padL = 60, padR = 30, padT = 50, padB = 60;
                            var halfW = (viz.width - padL - padR - 40) / 2;
                            var plotH = viz.height - padT - padB;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Without Scaling', padL + halfW / 2, 22);
                            ctx.fillText('With Scaling (\u00f71/\u221Ad\u2096)', padL + halfW + 40 + halfW / 2, 22);

                            // Entropy computation
                            function entropy(p) {
                                var h = 0;
                                for (var i = 0; i < p.length; i++) {
                                    if (p[i] > 1e-12) h -= p[i] * Math.log(p[i]);
                                }
                                return h;
                            }
                            var H_unscaled = entropy(unscaled);
                            var H_scaled = entropy(scaledSm);
                            var H_max = Math.log(nTokens);

                            // Draw bars function
                            function drawBars(probs, startX, color, label) {
                                var barW = halfW / nTokens * 0.7;
                                var gap = halfW / nTokens;
                                ctx.strokeStyle = '#30363d';
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.moveTo(startX, padT);
                                ctx.lineTo(startX, padT + plotH);
                                ctx.lineTo(startX + halfW, padT + plotH);
                                ctx.stroke();

                                // Y ticks
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.textBaseline = 'middle';
                                ctx.fillStyle = viz.colors.text;
                                for (var t = 0; t <= 4; t++) {
                                    var yv = t / 4;
                                    var yp = padT + plotH - yv * plotH;
                                    ctx.fillText(yv.toFixed(2), startX - 5, yp);
                                    ctx.strokeStyle = viz.colors.grid;
                                    ctx.beginPath();
                                    ctx.moveTo(startX, yp);
                                    ctx.lineTo(startX + halfW, yp);
                                    ctx.stroke();
                                }

                                for (var i = 0; i < nTokens; i++) {
                                    var bx = startX + i * gap + (gap - barW) / 2;
                                    var bh = probs[i] * plotH;
                                    var by = padT + plotH - bh;
                                    ctx.fillStyle = color;
                                    ctx.fillRect(bx, by, barW, bh);
                                    ctx.strokeStyle = color;
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(bx, by, barW, bh);
                                    // Token label
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'top';
                                    ctx.fillText('t' + (i+1), bx + barW / 2, padT + plotH + 4);
                                    // Value on top
                                    if (probs[i] > 0.02) {
                                        ctx.fillStyle = viz.colors.white;
                                        ctx.font = '9px -apple-system,sans-serif';
                                        ctx.textBaseline = 'bottom';
                                        ctx.fillText(probs[i].toFixed(2), bx + barW / 2, by - 2);
                                    }
                                }
                            }

                            drawBars(unscaled, padL, viz.colors.red, 'Unscaled');
                            drawBars(scaledSm, padL + halfW + 40, viz.colors.blue, 'Scaled');

                            // Entropy labels
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText('H = ' + H_unscaled.toFixed(2) + ' / ' + H_max.toFixed(2) + ' nats', padL + halfW / 2, viz.height - 18);
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText('H = ' + H_scaled.toFixed(2) + ' / ' + H_max.toFixed(2) + ' nats', padL + halfW + 40 + halfW / 2, viz.height - 18);

                            // d_k label
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('d\u2096 = ' + dk, viz.width / 2, 40);
                        }

                        draw();
                        VizEngine.createSlider(controls, 'd_k', 2, 512, dimK.value, 2, function(v) { dimK.value = v; draw(); });
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that if \\(q_i, k_i \\sim \\mathcal{N}(0,1)\\) i.i.d., then \\(\\text{Var}(q \\cdot k) = d_k\\). What happens to the standard deviation of the dot product when \\(d_k = 512\\)?',
                    hint: 'Use the fact that for independent zero-mean unit-variance random variables, \\(\\text{Var}(q_i k_i) = \\mathbb{E}[q_i^2]\\mathbb{E}[k_i^2] = 1\\).',
                    solution: '<p>Each term \\(q_i k_i\\) has \\(\\mathbb{E}[q_i k_i] = \\mathbb{E}[q_i]\\mathbb{E}[k_i] = 0\\) and \\(\\text{Var}(q_i k_i) = \\mathbb{E}[q_i^2 k_i^2] - (\\mathbb{E}[q_i k_i])^2 = \\mathbb{E}[q_i^2]\\mathbb{E}[k_i^2] - 0 = 1\\). Since the \\(d_k\\) terms are independent, \\(\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} 1 = d_k\\). The standard deviation is \\(\\sqrt{d_k}\\). For \\(d_k = 512\\), the standard deviation is \\(\\sqrt{512} \\approx 22.6\\). Dot products will routinely take values in the range \\([-45, 45]\\), causing severe softmax saturation. Dividing by \\(\\sqrt{512} \\approx 22.6\\) brings the standard deviation back to 1.</p>'
                },
                {
                    question: 'Suppose we scale by \\(1/d_k\\) instead of \\(1/\\sqrt{d_k}\\). How would this affect the attention distribution for large \\(d_k\\)?',
                    hint: 'After scaling by \\(1/d_k\\), the variance of the scores becomes \\(1/d_k\\) instead of \\(1\\).',
                    solution: '<p>Scaling by \\(1/d_k\\) gives \\(\\text{Var}(q \\cdot k / d_k) = d_k / d_k^2 = 1/d_k\\), so the scores have standard deviation \\(1/\\sqrt{d_k}\\). For large \\(d_k\\) (e.g., 512), this means standard deviation \\(\\approx 0.044\\), concentrating scores very close to zero. The softmax of near-zero inputs is approximately uniform: \\(\\sigma(\\mathbf{0}) = (1/n, \\ldots, 1/n)\\). The model would attend to all positions roughly equally, losing its ability to focus. This is the opposite extreme from no scaling: over-scaling makes attention too diffuse, while under-scaling makes it too peaked. The \\(1/\\sqrt{d_k}\\) choice is the Goldilocks solution that preserves unit variance.</p>'
                },
                {
                    question: 'The Jacobian of softmax \\(\\sigma(z)\\) at input \\(z\\) is \\(\\text{diag}(\\sigma) - \\sigma \\sigma^\\top\\). Show that when \\(\\sigma\\) is near a one-hot vector, the Frobenius norm of this Jacobian approaches zero.',
                    hint: 'If \\(\\sigma \\approx e_j\\) (the \\(j\\)-th standard basis vector), what are the diagonal and outer product terms?',
                    solution: '<p>Let \\(\\sigma \\approx e_j\\), so \\(\\sigma_j \\approx 1\\) and \\(\\sigma_i \\approx 0\\) for \\(i \\neq j\\). Then \\(\\text{diag}(\\sigma) \\approx e_j e_j^\\top\\) (a matrix with a single 1 on the diagonal) and \\(\\sigma \\sigma^\\top \\approx e_j e_j^\\top\\). The Jacobian is \\(J = \\text{diag}(\\sigma) - \\sigma \\sigma^\\top \\approx e_j e_j^\\top - e_j e_j^\\top = 0\\). More precisely, the \\((j,j)\\) entry is \\(\\sigma_j(1-\\sigma_j) \\approx 0\\), and off-diagonal entries are \\(-\\sigma_i \\sigma_j \\approx 0\\). The Frobenius norm \\(\\|J\\|_F^2 = \\sum_{ij} J_{ij}^2 \\approx 0\\). This confirms that saturated softmax has vanishing gradients, making learning extremely slow.</p>'
                }
            ]
        },


        // ======================== Section 3 ========================
        {
            id: 'multi-head-attention',
            title: 'Multi-Head Attention',
            content: `
<h2>Multi-Head Attention</h2>

<div class="env-block intuition">
<div class="env-title">Why Multiple Heads?</div>
<div class="env-body">
<p>A single attention head computes one set of attention weights, capturing one type of relationship between tokens. But language is rich with multiple simultaneous relationships: syntactic (subject-verb agreement), semantic (coreference), positional (adjacent words), and more. <strong>Multi-head attention</strong> runs several attention operations in parallel, each with its own learned projections, allowing the model to attend to information from different representation subspaces at different positions simultaneously.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 15.5 &mdash; Multi-Head Attention</div>
<div class="env-body">
<p>Given \\(h\\) attention heads, the multi-head attention mechanism is:</p>
\\[\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h) \\, W_O\\]
<p>where each head is an independent scaled dot-product attention:</p>
\\[\\text{head}_i = \\text{Attention}(X W_{Q_i}^\\top, \\, X W_{K_i}^\\top, \\, X W_{V_i}^\\top)\\]
<p>with \\(W_{Q_i}, W_{K_i} \\in \\mathbb{R}^{d_k \\times d_{\\text{model}}}\\), \\(W_{V_i} \\in \\mathbb{R}^{d_v \\times d_{\\text{model}}}\\), and the output projection \\(W_O \\in \\mathbb{R}^{d_{\\text{model}} \\times h d_v}\\).</p>
<p>Typically \\(d_k = d_v = d_{\\text{model}} / h\\), so the total parameter count is comparable to a single head with full dimensionality.</p>
</div>
</div>

<p>The original Transformer uses \\(h = 8\\) heads with \\(d_{\\text{model}} = 512\\), giving \\(d_k = d_v = 64\\) per head. Each head operates on a 64-dimensional subspace, and the concatenated output is projected back to 512 dimensions.</p>

<div class="env-block example">
<div class="env-title">Example 15.3 &mdash; What Different Heads Learn</div>
<div class="env-body">
<p>In a trained English language model, researchers (Clark et al., 2019; Voita et al., 2019) have observed that different heads specialize in different linguistic phenomena:</p>
<ul>
<li><strong>Head A (positional):</strong> Attends primarily to the immediately preceding or following token, capturing local context similar to a bigram model.</li>
<li><strong>Head B (syntactic):</strong> Attends from a verb to its subject, regardless of distance, capturing long-range syntactic dependencies.</li>
<li><strong>Head C (coreference):</strong> Attends from a pronoun to its antecedent, resolving "it" or "they" to the correct noun.</li>
<li><strong>Head D (separator):</strong> Attends to delimiter tokens like periods or commas, using them as segment boundaries.</li>
</ul>
<p>This specialization emerges from training, not from explicit supervision.</p>
</div>
</div>

<h3>Parameter Efficiency</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.6 &mdash; Multi-Head Parameter Count</div>
<div class="env-body">
<p>The total parameters for multi-head attention are:</p>
\\[\\underbrace{h \\cdot (d_k \\cdot d_{\\text{model}} + d_k \\cdot d_{\\text{model}} + d_v \\cdot d_{\\text{model}})}_{\\text{all heads: } Q, K, V} + \\underbrace{h \\cdot d_v \\cdot d_{\\text{model}}}_{\\text{output projection}} = h(3 d_k + d_v) \\cdot d_{\\text{model}}\\]
<p>When \\(d_k = d_v = d_{\\text{model}}/h\\), this simplifies to \\(4 d_{\\text{model}}^2\\), exactly the same as a single head with \\(d_k = d_v = d_{\\text{model}}\\) plus an output projection. Multi-head attention does not increase the parameter count; it restructures the same capacity into parallel subspaces.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Grouped-Query and Multi-Query Attention</div>
<div class="env-body">
<p>Modern LLMs use variants that reduce the key-value cache during inference. <strong>Multi-Query Attention</strong> (MQA, Shazeer 2019) shares a single K and V across all heads, reducing KV-cache memory by a factor of \\(h\\). <strong>Grouped-Query Attention</strong> (GQA, Ainslie et al. 2023) uses \\(g\\) groups of KV heads (where \\(1 \\leq g \\leq h\\)), interpolating between MQA (\\(g=1\\)) and standard MHA (\\(g=h\\)). LLaMA 2 70B uses GQA with \\(g=8\\) groups for its 64 query heads.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="multi-head-viz"></div>

<p>The visualization above shows four attention heads operating simultaneously on the same input sentence. Each head learns different attention patterns, color-coded for clarity. Observe how some heads focus on local relationships while others capture long-distance dependencies.</p>
`,
            visualizations: [
                {
                    id: 'multi-head-viz',
                    title: 'Multi-Head Attention Patterns',
                    description: 'Four attention heads attending to different relationships in a sentence. Line thickness and opacity represent attention weight. Select a query token to see where each head attends.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var selectedToken = { value: 0 };
                        var tokens = ['The', 'cat', 'that', 'I', 'saw', 'sat'];
                        var n = tokens.length;
                        var headColors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.purple];
                        var headNames = ['Head 1: Local', 'Head 2: Syntactic', 'Head 3: Semantic', 'Head 4: Positional'];

                        // Pre-defined attention patterns for each head (row = query, col = key)
                        // Head 1: local / adjacent attention
                        var head1 = [
                            [0.40, 0.45, 0.08, 0.03, 0.02, 0.02],
                            [0.35, 0.35, 0.22, 0.03, 0.03, 0.02],
                            [0.05, 0.30, 0.35, 0.22, 0.05, 0.03],
                            [0.02, 0.03, 0.30, 0.40, 0.20, 0.05],
                            [0.02, 0.02, 0.05, 0.25, 0.40, 0.26],
                            [0.02, 0.02, 0.03, 0.05, 0.38, 0.50]
                        ];
                        // Head 2: syntactic (verb->subject)
                        var head2 = [
                            [0.60, 0.15, 0.08, 0.07, 0.05, 0.05],
                            [0.10, 0.65, 0.08, 0.07, 0.05, 0.05],
                            [0.12, 0.40, 0.30, 0.08, 0.05, 0.05],
                            [0.05, 0.05, 0.05, 0.70, 0.10, 0.05],
                            [0.05, 0.05, 0.05, 0.55, 0.25, 0.05],
                            [0.05, 0.55, 0.05, 0.05, 0.10, 0.20]
                        ];
                        // Head 3: semantic / coreference
                        var head3 = [
                            [0.50, 0.20, 0.10, 0.05, 0.05, 0.10],
                            [0.15, 0.40, 0.05, 0.05, 0.05, 0.30],
                            [0.08, 0.15, 0.50, 0.12, 0.10, 0.05],
                            [0.05, 0.05, 0.10, 0.55, 0.20, 0.05],
                            [0.05, 0.50, 0.05, 0.05, 0.30, 0.05],
                            [0.05, 0.60, 0.05, 0.03, 0.07, 0.20]
                        ];
                        // Head 4: broad / positional pattern
                        var head4 = [
                            [0.30, 0.15, 0.15, 0.15, 0.13, 0.12],
                            [0.18, 0.25, 0.15, 0.15, 0.14, 0.13],
                            [0.15, 0.15, 0.25, 0.17, 0.15, 0.13],
                            [0.12, 0.14, 0.17, 0.27, 0.17, 0.13],
                            [0.10, 0.12, 0.14, 0.18, 0.28, 0.18],
                            [0.10, 0.10, 0.12, 0.15, 0.18, 0.35]
                        ];
                        var heads = [head1, head2, head3, head4];

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var qi = Math.round(selectedToken.value);
                            var tokenY = 60;
                            var headH = 70;
                            var startY = 120;
                            var spacing = 110;
                            var tokenSpacing = spacing;
                            var tokStartX = (viz.width - (n - 1) * tokenSpacing) / 2;

                            // Draw query token row
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Query token: "' + tokens[qi] + '"', viz.width / 2, 22);

                            // Draw token circles at top
                            for (var i = 0; i < n; i++) {
                                var tx = tokStartX + i * tokenSpacing;
                                var isQuery = (i === qi);
                                ctx.beginPath();
                                ctx.arc(tx, tokenY, 20, 0, Math.PI * 2);
                                ctx.fillStyle = isQuery ? viz.colors.white + '33' : viz.colors.bg;
                                ctx.fill();
                                ctx.strokeStyle = isQuery ? viz.colors.white : viz.colors.text;
                                ctx.lineWidth = isQuery ? 2.5 : 1;
                                ctx.stroke();
                                ctx.fillStyle = isQuery ? viz.colors.white : viz.colors.text;
                                ctx.font = (isQuery ? 'bold ' : '') + '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(tokens[i], tx, tokenY);
                            }

                            // Draw each head's attention
                            for (var h = 0; h < 4; h++) {
                                var headY = startY + h * headH;
                                var color = headColors[h];
                                var attnRow = heads[h][qi];

                                // Head label
                                ctx.fillStyle = color;
                                ctx.font = 'bold 11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(headNames[h], 10, headY + 10);

                                // Draw attention arcs from query to each key
                                var qx = tokStartX + qi * tokenSpacing;
                                for (var j = 0; j < n; j++) {
                                    var kx = tokStartX + j * tokenSpacing;
                                    var w = attnRow[j];
                                    if (w < 0.02) continue;
                                    var arcY = headY + 10;
                                    // Draw line from query position down to arc, then to key
                                    ctx.strokeStyle = color;
                                    ctx.globalAlpha = 0.2 + w * 0.8;
                                    ctx.lineWidth = 1 + w * 6;
                                    ctx.beginPath();
                                    // Arc connecting query to key
                                    var midX = (qx + kx) / 2;
                                    var dist = Math.abs(kx - qx);
                                    var arcHeight = Math.max(8, dist * 0.15);
                                    if (qi === j) {
                                        // Self-attention: small loop
                                        ctx.beginPath();
                                        ctx.arc(qx, arcY - 8, 8, 0, Math.PI * 2);
                                        ctx.stroke();
                                    } else {
                                        ctx.beginPath();
                                        ctx.moveTo(qx, arcY);
                                        ctx.quadraticCurveTo(midX, arcY + arcHeight, kx, arcY);
                                        ctx.stroke();
                                    }
                                    // Weight label
                                    if (w > 0.10) {
                                        ctx.globalAlpha = 1;
                                        ctx.fillStyle = color;
                                        ctx.font = '9px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'top';
                                        var labelX = qi === j ? qx : midX;
                                        var labelY = qi === j ? arcY + 4 : arcY + arcHeight * 0.5 + 2;
                                        ctx.fillText(w.toFixed(2), labelX, labelY);
                                    }
                                    ctx.globalAlpha = 1;
                                }
                            }

                            // Legend
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Line thickness = attention weight', viz.width / 2, viz.height - 10);
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Query token', 0, n - 1, 0, 1, function(v) { selectedToken.value = v; draw(); });
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'In the original Transformer (\\(d_{\\text{model}} = 512\\), \\(h = 8\\), \\(d_k = d_v = 64\\)), compute the total parameter count for one multi-head attention layer including the output projection \\(W_O\\). How does this compare to a single-head attention with \\(d_k = d_v = 512\\)?',
                    hint: 'For each head: \\(W_Q, W_K, W_V\\) each have \\(d_k \\times d_{\\text{model}}\\) parameters. The output projection \\(W_O\\) maps \\(h d_v\\) back to \\(d_{\\text{model}}\\).',
                    solution: '<p>Per head: \\(3 \\times 64 \\times 512 = 98{,}304\\) parameters. For 8 heads: \\(8 \\times 98{,}304 = 786{,}432\\). Output projection: \\(W_O \\in \\mathbb{R}^{512 \\times 512}\\) has \\(262{,}144\\) parameters. Total: \\(786{,}432 + 262{,}144 = 1{,}048{,}576 = 4 \\times 512^2\\).</p><p>Single-head with \\(d_k = d_v = 512\\): \\(W_Q, W_K, W_V\\) each have \\(512 \\times 512 = 262{,}144\\) parameters, totaling \\(786{,}432\\). Adding an output projection gives the same \\(4 \\times 512^2 = 1{,}048{,}576\\). The parameter count is identical; multi-head attention simply partitions the capacity into parallel subspaces.</p>'
                },
                {
                    question: 'Why is concatenation followed by a linear projection (\\(\\text{Concat} \\cdot W_O\\)) preferred over, say, averaging the head outputs?',
                    hint: 'Consider what information is lost when averaging vs. when projecting.',
                    solution: '<p>Averaging the head outputs would force each head to produce representations in the same space, losing the benefit of learning in different subspaces. If head 1 captures syntax in its 64 dimensions and head 2 captures semantics in its 64 dimensions, averaging would destructively combine these unrelated representations.</p><p>Concatenation preserves all head outputs in separate dimensions (producing a 512-dimensional vector for 8 heads of size 64). The output projection \\(W_O\\) then learns how to <em>mix</em> information from all heads, potentially combining head 1\'s syntactic signal with head 2\'s semantic signal to produce the final representation. This is strictly more expressive: \\(W_O\\) can learn to implement averaging as a special case (by setting appropriate weights), but it can also learn far more complex combinations.</p>'
                },
                {
                    question: 'Voita et al. (2019) found that many attention heads can be pruned without significant performance loss. What does this suggest about the effective number of distinct attention patterns a Transformer learns?',
                    hint: 'Consider redundancy among heads and whether all heads are equally important.',
                    solution: '<p>This suggests substantial redundancy: the model over-parameterizes by using \\(h\\) heads but often needs far fewer distinct attention patterns. Voita et al. found that in a 6-layer Transformer with 8 heads per layer, only a few heads per layer (often 2 to 4) are "important" in the sense that removing them degrades performance. The remaining heads either learn redundant patterns or contribute marginally. This has practical implications: (1) not all heads are created equal, so head-level dropout or pruning can reduce compute at inference; (2) the optimal number of heads may be task-dependent, not a universal constant; (3) it motivates efficiency-oriented designs like multi-query attention, which reduces the number of unique KV heads.</p>'
                },
                {
                    question: 'In Grouped-Query Attention (GQA) with \\(h = 32\\) query heads and \\(g = 8\\) KV groups, how much KV-cache memory is saved compared to standard multi-head attention during autoregressive inference with sequence length \\(n\\)?',
                    hint: 'Standard MHA stores \\(h\\) sets of K and V vectors; GQA stores \\(g\\) sets.',
                    solution: '<p>In standard MHA, the KV cache stores \\(K\\) and \\(V\\) for all \\(h = 32\\) heads. Per token, this is \\(2 \\times 32 \\times d_k\\) values, and for a sequence of length \\(n\\): \\(2 \\times 32 \\times d_k \\times n\\). In GQA with \\(g = 8\\) groups, only 8 unique KV pairs are stored: \\(2 \\times 8 \\times d_k \\times n\\). The ratio is \\(8/32 = 1/4\\), so GQA uses <strong>75% less</strong> KV-cache memory. Each group of \\(32/8 = 4\\) query heads shares the same KV pair. This is critical for inference on long sequences where the KV cache is the primary memory bottleneck, and it is why models like LLaMA 2 70B adopted GQA.</p>'
                }
            ]
        },


        // ======================== Section 4 ========================
        {
            id: 'positional-encoding',
            title: 'Positional Encoding',
            content: `
<h2>Positional Encoding</h2>

<div class="env-block intuition">
<div class="env-title">The Permutation Invariance Problem</div>
<div class="env-body">
<p>Self-attention is inherently <strong>permutation-equivariant</strong>: if we shuffle the input tokens, the output values are shuffled in the same way (the attention weights adjust accordingly). This means that without additional information, a Transformer cannot distinguish "The cat sat on the mat" from "mat the on sat cat The." To inject positional information, we add <strong>positional encodings</strong> to the input embeddings.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 15.7 &mdash; Positional Encoding</div>
<div class="env-body">
<p>A <strong>positional encoding</strong> is a function \\(\\text{PE}: \\mathbb{Z}_{\\geq 0} \\to \\mathbb{R}^{d_{\\text{model}}}\\) that maps each position index \\(t\\) to a vector. The Transformer input is:</p>
\\[z_t = x_t + \\text{PE}(t)\\]
<p>where \\(x_t\\) is the token embedding and the addition is element-wise. This additive injection is simple but effective: the model learns to separate positional and content information through its subsequent layers.</p>
</div>
</div>

<h3>Sinusoidal Positional Encoding</h3>

<p>Vaswani et al. (2017) proposed a deterministic encoding using sinusoidal functions at different frequencies:</p>

<div class="env-block definition">
<div class="env-title">Definition 15.8 &mdash; Sinusoidal Positional Encoding</div>
<div class="env-body">
<p>For position \\(t\\) and dimension \\(i\\):</p>
\\[\\text{PE}(t, 2i) = \\sin\\!\\left(\\frac{t}{10000^{2i/d_{\\text{model}}}}\\right), \\quad \\text{PE}(t, 2i+1) = \\cos\\!\\left(\\frac{t}{10000^{2i/d_{\\text{model}}}}\\right)\\]
<p>where \\(i = 0, 1, \\ldots, d_{\\text{model}}/2 - 1\\). Each dimension pair \\((2i, 2i+1)\\) encodes position as a sinusoid with wavelength \\(\\lambda_i = 2\\pi \\cdot 10000^{2i/d_{\\text{model}}}\\), ranging from \\(2\\pi\\) (for \\(i=0\\)) to \\(2\\pi \\cdot 10000\\) (for \\(i = d_{\\text{model}}/2 - 1\\)).</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">The Binary Clock Analogy</div>
<div class="env-body">
<p>Think of sinusoidal encodings as a generalized binary clock. In a binary representation of position \\(t\\), the least significant bit oscillates every step, the next bit every 2 steps, and so on. The sinusoidal encoding does the same with continuous functions: the lowest-indexed dimensions oscillate rapidly (high frequency), and higher-indexed dimensions oscillate slowly (low frequency). Just as a binary number uniquely identifies a position, the combination of sinusoids at different frequencies uniquely identifies each position in the sequence.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Relative Position via Linear Transformation</div>
<div class="env-body">
<p>A key property of sinusoidal encoding is that the encoding of position \\(t + k\\) can be expressed as a linear function of the encoding at position \\(t\\). For each frequency \\(\\omega_i = 1/10000^{2i/d_{\\text{model}}}\\):</p>
\\[\\begin{pmatrix} \\sin(\\omega_i(t+k)) \\\\ \\cos(\\omega_i(t+k)) \\end{pmatrix} = \\begin{pmatrix} \\cos(\\omega_i k) & \\sin(\\omega_i k) \\\\ -\\sin(\\omega_i k) & \\cos(\\omega_i k) \\end{pmatrix} \\begin{pmatrix} \\sin(\\omega_i t) \\\\ \\cos(\\omega_i t) \\end{pmatrix}\\]
<p>This rotation matrix depends only on the offset \\(k\\), not on the absolute position \\(t\\). The model can therefore learn to attend to relative positions via linear operations on the positional encodings.</p>
</div>
</div>

<h3>Learned Positional Embeddings</h3>

<p>An alternative approach, used in BERT and GPT-2, is to learn the positional encodings as parameters:</p>
\\[\\text{PE}(t) = E_t, \\quad E \\in \\mathbb{R}^{T_{\\max} \\times d_{\\text{model}}}\\]
<p>where \\(T_{\\max}\\) is the maximum sequence length. This is more flexible but cannot extrapolate to positions beyond \\(T_{\\max}\\). Empirically, learned and sinusoidal encodings perform similarly (Vaswani et al., 2017).</p>

<h3>Rotary Position Embedding (RoPE)</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.9 &mdash; Rotary Position Embedding</div>
<div class="env-body">
<p>RoPE (Su et al., 2021) encodes position directly into the attention computation rather than adding it to embeddings. For a query or key vector \\(x \\in \\mathbb{R}^{d}\\), group the dimensions into pairs \\((x_{2i}, x_{2i+1})\\) and apply a rotation by angle \\(t \\theta_i\\):</p>
\\[f(x, t)_{2i} = x_{2i} \\cos(t\\theta_i) - x_{2i+1} \\sin(t\\theta_i)\\]
\\[f(x, t)_{2i+1} = x_{2i} \\sin(t\\theta_i) + x_{2i+1} \\cos(t\\theta_i)\\]
<p>where \\(\\theta_i = 10000^{-2i/d}\\). The key property is that \\(f(q, m)^\\top f(k, n)\\) depends only on \\(q, k\\), and the <em>relative</em> position \\(m - n\\), not the absolute positions. This naturally encodes relative position into the attention score.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">RoPE Advantages</div>
<div class="env-body">
<p>RoPE has become the dominant positional encoding in modern LLMs (LLaMA, Mistral, Qwen) because: (1) it directly encodes relative position in attention scores without extra parameters; (2) it has better length extrapolation than learned absolute embeddings; (3) the attention score naturally decays with distance due to the geometric properties of high-dimensional rotations; (4) it is compatible with linear attention and KV-cache.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="positional-encoding-viz"></div>

<p>The visualization above shows the sinusoidal positional encoding as a heatmap. Position runs along the x-axis and dimension along the y-axis. Lower dimensions oscillate rapidly (high frequency) while higher dimensions change slowly (low frequency). Each position has a unique fingerprint across all dimensions.</p>
`,
            visualizations: [
                {
                    id: 'positional-encoding-viz',
                    title: 'Sinusoidal Positional Encoding Heatmap',
                    description: 'The heatmap shows PE(t, d) for positions t = 0..63 and dimensions d = 0..63. Lower dimensions have higher frequency. Adjust the model dimension to see frequency patterns.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var maxPos = 64;
                        var dModel = { value: 64 };

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var d = Math.round(dModel.value);
                            var padL = 70, padT = 50, padR = 80, padB = 50;
                            var plotW = viz.width - padL - padR;
                            var plotH = viz.height - padT - padB;
                            var cellW = plotW / maxPos;
                            var cellH = plotH / d;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Sinusoidal Positional Encoding (d_model = ' + d + ')', viz.width / 2 - 20, 20);

                            // Compute and draw heatmap
                            for (var pos = 0; pos < maxPos; pos++) {
                                for (var dim = 0; dim < d; dim++) {
                                    var i = Math.floor(dim / 2);
                                    var freq = 1.0 / Math.pow(10000, 2 * i / d);
                                    var val;
                                    if (dim % 2 === 0) {
                                        val = Math.sin(pos * freq);
                                    } else {
                                        val = Math.cos(pos * freq);
                                    }
                                    // Map [-1,1] to color
                                    var cx = padL + pos * cellW;
                                    var cy = padT + dim * cellH;
                                    if (val > 0) {
                                        var intensity = Math.round(val * 200);
                                        ctx.fillStyle = 'rgb(' + (20 + intensity * 0.3) + ',' + (20 + intensity * 0.5) + ',' + (40 + intensity) + ')';
                                    } else {
                                        var intensity = Math.round(-val * 200);
                                        ctx.fillStyle = 'rgb(' + (40 + intensity) + ',' + (20 + intensity * 0.2) + ',' + (20 + intensity * 0.2) + ')';
                                    }
                                    ctx.fillRect(cx, cy, Math.ceil(cellW), Math.ceil(cellH));
                                }
                            }

                            // Axes labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Position (t)', padL + plotW / 2, padT + plotH + 8);

                            // X ticks
                            for (var t = 0; t < maxPos; t += 8) {
                                ctx.fillText(t, padL + t * cellW + cellW / 2, padT + plotH + 20);
                            }

                            // Y label
                            ctx.save();
                            ctx.translate(18, padT + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Dimension (d)', 0, 0);
                            ctx.restore();

                            // Y ticks
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            var yStep = Math.max(1, Math.floor(d / 8));
                            for (var dim = 0; dim < d; dim += yStep) {
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(dim, padL - 5, padT + dim * cellH + cellH / 2);
                            }

                            // Color legend
                            var legX = viz.width - 60, legY = padT;
                            var legH = plotH;
                            for (var ly = 0; ly < legH; ly++) {
                                var lv = 1 - 2 * ly / legH; // +1 to -1
                                if (lv > 0) {
                                    var li = Math.round(lv * 200);
                                    ctx.fillStyle = 'rgb(' + (20 + li * 0.3) + ',' + (20 + li * 0.5) + ',' + (40 + li) + ')';
                                } else {
                                    var li = Math.round(-lv * 200);
                                    ctx.fillStyle = 'rgb(' + (40 + li) + ',' + (20 + li * 0.2) + ',' + (20 + li * 0.2) + ')';
                                }
                                ctx.fillRect(legX, legY + ly, 15, 1);
                            }
                            ctx.strokeStyle = viz.colors.text;
                            ctx.lineWidth = 0.5;
                            ctx.strokeRect(legX, legY, 15, legH);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '9px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('+1', legX + 18, legY - 2);
                            ctx.textBaseline = 'middle';
                            ctx.fillText(' 0', legX + 18, legY + legH / 2);
                            ctx.textBaseline = 'bottom';
                            ctx.fillText('-1', legX + 18, legY + legH + 2);

                            // Frequency annotation
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('\u2190 high freq', padL + plotW + 5, padT + 5);
                            ctx.fillText('\u2190 low freq', padL + plotW + 5, padT + plotH - 5);
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Dimensions', 16, 128, dModel.value, 8, function(v) { dModel.value = v; draw(); });
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the sinusoidal positional encoding at position \\(t+k\\) can be written as a linear transformation of the encoding at position \\(t\\) for each frequency component. What is the transformation matrix?',
                    hint: 'Use the angle addition formulas: \\(\\sin(a+b) = \\sin a \\cos b + \\cos a \\sin b\\) and \\(\\cos(a+b) = \\cos a \\cos b - \\sin a \\sin b\\).',
                    solution: '<p>For frequency \\(\\omega_i\\), the encoding at positions \\(t\\) and \\(t+k\\) involves \\(\\sin(\\omega_i t)\\), \\(\\cos(\\omega_i t)\\) and \\(\\sin(\\omega_i(t+k))\\), \\(\\cos(\\omega_i(t+k))\\). By the angle addition formulas:</p><p>\\(\\sin(\\omega_i(t+k)) = \\sin(\\omega_i t)\\cos(\\omega_i k) + \\cos(\\omega_i t)\\sin(\\omega_i k)\\)</p><p>\\(\\cos(\\omega_i(t+k)) = \\cos(\\omega_i t)\\cos(\\omega_i k) - \\sin(\\omega_i t)\\sin(\\omega_i k)\\)</p><p>In matrix form: \\(\\begin{pmatrix} \\text{PE}(t+k, 2i) \\\\ \\text{PE}(t+k, 2i+1) \\end{pmatrix} = \\begin{pmatrix} \\cos(\\omega_i k) & \\sin(\\omega_i k) \\\\ -\\sin(\\omega_i k) & \\cos(\\omega_i k) \\end{pmatrix} \\begin{pmatrix} \\text{PE}(t, 2i) \\\\ \\text{PE}(t, 2i+1) \\end{pmatrix}\\)</p><p>This is a rotation matrix \\(R(\\omega_i k)\\) that depends only on the offset \\(k\\), not on \\(t\\). The full transformation across all dimensions is a block-diagonal matrix of \\(d_{\\text{model}}/2\\) rotation matrices. This allows the model to learn relative position attention via linear operations.</p>'
                },
                {
                    question: 'If \\(d_{\\text{model}} = 512\\), what are the shortest and longest wavelengths in the sinusoidal encoding? How many positions can be uniquely represented?',
                    hint: 'The wavelength for dimension pair \\(i\\) is \\(2\\pi \\cdot 10000^{2i/d_{\\text{model}}}\\). The shortest wavelength corresponds to \\(i=0\\) and the longest to \\(i = d_{\\text{model}}/2 - 1\\).',
                    solution: '<p>Shortest wavelength (\\(i=0\\)): \\(\\lambda_0 = 2\\pi \\cdot 10000^0 = 2\\pi \\approx 6.28\\). This oscillates with a period of about 6 positions. Longest wavelength (\\(i=255\\)): \\(\\lambda_{255} = 2\\pi \\cdot 10000^{510/512} \\approx 2\\pi \\cdot 9961 \\approx 62{,}578\\). This oscillates once every ~62,578 positions.</p><p>The number of uniquely representable positions is theoretically infinite (sinusoidal functions are aperiodic when wavelengths are incommensurate). In practice, the representation becomes numerically indistinguishable for positions beyond the longest wavelength, so about 10,000 positions can be practically distinguished. The original Transformer was trained on sequences up to 512 tokens, well within this range.</p>'
                },
                {
                    question: 'RoPE modifies the attention score \\(q^\\top k\\) to depend on relative position. Show that \\(f(q, m)^\\top f(k, n)\\) depends only on \\(q, k\\) and \\(m - n\\), not on \\(m\\) and \\(n\\) separately. (Consider a single dimension pair.)',
                    hint: 'Write out the rotation for a single pair \\((q_0, q_1)\\) at position \\(m\\) and \\((k_0, k_1)\\) at position \\(n\\), then compute the dot product.',
                    solution: '<p>For a single dimension pair with frequency \\(\\theta\\):</p><p>\\(f(q, m) = (q_0 \\cos m\\theta - q_1 \\sin m\\theta,\; q_0 \\sin m\\theta + q_1 \\cos m\\theta)\\)</p><p>\\(f(k, n) = (k_0 \\cos n\\theta - k_1 \\sin n\\theta,\; k_0 \\sin n\\theta + k_1 \\cos n\\theta)\\)</p><p>The dot product is:</p><p>\\(f(q,m)^\\top f(k,n) = (q_0 \\cos m\\theta - q_1 \\sin m\\theta)(k_0 \\cos n\\theta - k_1 \\sin n\\theta)\\)</p><p>\\(\\quad + (q_0 \\sin m\\theta + q_1 \\cos m\\theta)(k_0 \\sin n\\theta + k_1 \\cos n\\theta)\\)</p><p>Expanding and collecting terms using \\(\\cos^2 + \\sin^2 = 1\\) and \\(\\cos a \\cos b + \\sin a \\sin b = \\cos(a-b)\\):</p><p>\\(= q_0 k_0 \\cos((m-n)\\theta) + q_1 k_1 \\cos((m-n)\\theta) + q_0 k_1 \\sin((m-n)\\theta) - q_1 k_0 \\sin((m-n)\\theta)\\)</p><p>This depends on \\(m-n\\) only, not on \\(m\\) or \\(n\\) individually. Summing over all dimension pairs gives the full attention score as a function of \\(q, k\\) and the relative position \\(m-n\\).</p>'
                }
            ]
        },


        // ======================== Section 5 ========================
        {
            id: 'transformer-block',
            title: 'Transformer Block',
            content: `
<h2>Transformer Block</h2>

<div class="env-block intuition">
<div class="env-title">The Building Block of Modern Deep Learning</div>
<div class="env-body">
<p>The Transformer block is the fundamental repeating unit from which all Transformer models are built. It combines multi-head self-attention with a position-wise feedforward network, using residual connections and layer normalization to ensure stable gradient flow through deep stacks. Understanding this block is essential: GPT, BERT, ViT, and essentially all modern architectures are stacks of these blocks.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 15.10 &mdash; Transformer Block (Post-Norm)</div>
<div class="env-body">
<p>A <strong>post-norm Transformer block</strong> (as in the original Vaswani et al. paper) applies layer normalization <em>after</em> each sub-layer:</p>
\\[\\text{mid} = \\text{LayerNorm}(x + \\text{MultiHead}(x, x, x))\\]
\\[\\text{out} = \\text{LayerNorm}(\\text{mid} + \\text{FFN}(\\text{mid}))\\]
<p>where \\(x \\in \\mathbb{R}^{n \\times d_{\\text{model}}}\\) is the input sequence. Each sub-layer has a <strong>residual connection</strong> (the \\(+ x\\) or \\(+ \\text{mid}\\) terms) followed by layer normalization.</p>
</div>
</div>

<h3>Position-Wise Feedforward Network</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.11 &mdash; Position-Wise FFN</div>
<div class="env-body">
<p>The feedforward sub-layer applies the same two-layer MLP independently to each position:</p>
\\[\\text{FFN}(x) = W_2 \\, \\sigma(W_1 x + b_1) + b_2\\]
<p>where \\(W_1 \\in \\mathbb{R}^{d_{ff} \\times d_{\\text{model}}}\\), \\(W_2 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{ff}}\\), and \\(\\sigma\\) is an activation function. The original Transformer uses ReLU; modern models use GELU or SwiGLU. The inner dimension \\(d_{ff}\\) is typically \\(4 d_{\\text{model}}\\) (e.g., \\(d_{ff} = 2048\\) for \\(d_{\\text{model}} = 512\\)).</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Why Two Sub-Layers?</div>
<div class="env-body">
<p>The attention sub-layer and FFN sub-layer serve complementary roles. Attention enables <strong>inter-token communication</strong>: each position gathers information from other positions. The FFN enables <strong>per-token computation</strong>: it processes the gathered information through a nonlinear transformation. Without the FFN, the Transformer would be limited to linear combinations of input representations (attention output is a weighted sum plus linear projection). The FFN provides the nonlinearity needed for complex function approximation.</p>
</div>
</div>

<h3>Layer Normalization</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.12 &mdash; Layer Normalization</div>
<div class="env-body">
<p>For an input vector \\(x \\in \\mathbb{R}^d\\), layer normalization computes:</p>
\\[\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta\\]
<p>where \\(\\mu = \\frac{1}{d}\\sum_{i=1}^d x_i\\), \\(\\sigma^2 = \\frac{1}{d}\\sum_{i=1}^d (x_i - \\mu)^2\\), and \\(\\gamma, \\beta \\in \\mathbb{R}^d\\) are learnable scale and shift parameters. Unlike batch normalization (Chapter 6), layer normalization normalizes across the feature dimension for each token independently, making it suitable for variable-length sequences and autoregressive decoding.</p>
</div>
</div>

<h3>Pre-Norm vs. Post-Norm</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.13 &mdash; Pre-Norm Transformer Block</div>
<div class="env-body">
<p>The <strong>pre-norm</strong> variant (used in GPT-2 and most modern models) applies layer normalization <em>before</em> each sub-layer:</p>
\\[\\text{mid} = x + \\text{MultiHead}(\\text{LN}(x), \\text{LN}(x), \\text{LN}(x))\\]
\\[\\text{out} = \\text{mid} + \\text{FFN}(\\text{LN}(\\text{mid}))\\]
<p>This change has a profound effect on gradient flow: the residual path is now a <em>clean</em> identity from input to output, with sub-layers contributing additively. Pre-norm enables training very deep Transformers (100+ layers) without warm-up, because gradients flow through the residual stream unimpeded.</p>
</div>
</div>

<div class="env-block warning">
<div class="env-title">Post-Norm Training Instability</div>
<div class="env-body">
<p>Post-norm Transformers are harder to train, especially at large depth. The gradient through a post-norm block must pass through the layer norm, which can attenuate the signal. Without careful learning rate warm-up, post-norm training often diverges. Pre-norm avoids this issue because the gradient through the residual connection bypasses the layer norm entirely. However, some work (Xiong et al., 2020) suggests post-norm achieves slightly better final performance if training succeeds, creating a stability-performance tradeoff.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">RMSNorm</div>
<div class="env-body">
<p>Modern LLMs (LLaMA, Mistral) replace LayerNorm with <strong>RMSNorm</strong> (Zhang &amp; Sennrich, 2019), which omits the mean subtraction and learned bias:</p>
\\[\\text{RMSNorm}(x) = \\gamma \\odot \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2 + \\epsilon}}\\]
<p>This is simpler and faster (one fewer reduction operation), with negligible performance difference in practice.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="transformer-block-viz"></div>

<p>The visualization above shows data flowing through a Transformer block. The animated signal enters at the bottom, passes through multi-head attention with a residual connection, through layer normalization, then through the FFN with another residual connection and normalization. Toggle between pre-norm and post-norm to see how the architecture differs.</p>
`,
            visualizations: [
                {
                    id: 'transformer-block-viz',
                    title: 'Transformer Block Data Flow',
                    description: 'Animated data flow through a Transformer block. Toggle pre-norm vs. post-norm to see the architectural difference. The particle traces the signal path.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 480, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var preNorm = { value: 1 };
                        var animT = 0;

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

                        function drawArrow(x1, y1, x2, y2, color) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();
                            var angle = Math.atan2(y2 - y1, x2 - x1);
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 8 * Math.cos(angle - 0.4), y2 - 8 * Math.sin(angle - 0.4));
                            ctx.lineTo(x2 - 8 * Math.cos(angle + 0.4), y2 - 8 * Math.sin(angle + 0.4));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function drawAddCircle(x, y) {
                            ctx.beginPath();
                            ctx.arc(x, y, 12, 0, Math.PI * 2);
                            ctx.fillStyle = '#1a1a40';
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 1.5;
                            ctx.stroke();
                            ctx.fillStyle = viz.colors.green;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('+', x, y);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var isPreNorm = Math.round(preNorm.value) === 1;
                            var cx = viz.width / 2;
                            var boxW = 160, boxH = 34;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(isPreNorm ? 'Pre-Norm Transformer Block' : 'Post-Norm Transformer Block', cx, 18);

                            if (isPreNorm) {
                                // Pre-norm layout (bottom to top)
                                var inputY = 450, ln1Y = 400, attnY = 350, add1Y = 300, ln2Y = 250, ffnY = 200, add2Y = 150, outputY = 100;
                                // Input
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Input x', cx, inputY + 5);

                                // Arrow up to LN1
                                drawArrow(cx, inputY - 5, cx, ln1Y + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ln1Y, boxW, boxH, 6, '#2a1a3a', viz.colors.purple);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('LayerNorm', cx, ln1Y + boxH/2);

                                // Arrow up to MHA
                                drawArrow(cx, ln1Y - 2, cx, attnY + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, attnY, boxW, boxH, 6, '#1a2a3a', viz.colors.blue);
                                ctx.fillStyle = viz.colors.blue;
                                ctx.fillText('Multi-Head Attention', cx, attnY + boxH/2);

                                // Arrow up to Add1
                                drawArrow(cx, attnY - 2, cx, add1Y + 14, viz.colors.text);
                                drawAddCircle(cx, add1Y);

                                // Residual connection (skip from input to add1)
                                var resX = cx + boxW/2 + 30;
                                ctx.strokeStyle = viz.colors.green;
                                ctx.lineWidth = 1.5;
                                ctx.setLineDash([5, 3]);
                                ctx.beginPath();
                                ctx.moveTo(cx + 12, inputY - 8);
                                ctx.lineTo(resX, inputY - 8);
                                ctx.lineTo(resX, add1Y);
                                ctx.lineTo(cx + 14, add1Y);
                                ctx.stroke();
                                ctx.setLineDash([]);
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('residual', resX + 3, (inputY + add1Y) / 2);

                                // Arrow up to LN2
                                drawArrow(cx, add1Y - 14, cx, ln2Y + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ln2Y, boxW, boxH, 6, '#2a1a3a', viz.colors.purple);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('LayerNorm', cx, ln2Y + boxH/2);

                                // Arrow up to FFN
                                drawArrow(cx, ln2Y - 2, cx, ffnY + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ffnY, boxW, boxH, 6, '#1a3a2a', viz.colors.teal);
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText('Feed-Forward Network', cx, ffnY + boxH/2);

                                // Arrow up to Add2
                                drawArrow(cx, ffnY - 2, cx, add2Y + 14, viz.colors.text);
                                drawAddCircle(cx, add2Y);

                                // Residual connection (skip from add1 to add2)
                                var resX2 = cx - boxW/2 - 30;
                                ctx.strokeStyle = viz.colors.green;
                                ctx.setLineDash([5, 3]);
                                ctx.beginPath();
                                ctx.moveTo(cx - 14, add1Y);
                                ctx.lineTo(resX2, add1Y);
                                ctx.lineTo(resX2, add2Y);
                                ctx.lineTo(cx - 14, add2Y);
                                ctx.stroke();
                                ctx.setLineDash([]);
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.fillText('residual', resX2 - 3, (add1Y + add2Y) / 2);

                                // Output
                                drawArrow(cx, add2Y - 14, cx, outputY + 10, viz.colors.text);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Output', cx, outputY);

                                // Animated particle
                                var pathPoints = [
                                    [cx, inputY], [cx, ln1Y + boxH/2], [cx, attnY + boxH/2],
                                    [cx, add1Y], [cx, ln2Y + boxH/2], [cx, ffnY + boxH/2],
                                    [cx, add2Y], [cx, outputY]
                                ];
                                var segIdx = Math.floor(animT) % pathPoints.length;
                                var segFrac = animT % 1;
                                var nextIdx = (segIdx + 1) % pathPoints.length;
                                var px = pathPoints[segIdx][0] + (pathPoints[nextIdx][0] - pathPoints[segIdx][0]) * segFrac;
                                var py = pathPoints[segIdx][1] + (pathPoints[nextIdx][1] - pathPoints[segIdx][1]) * segFrac;
                                ctx.beginPath();
                                ctx.arc(px, py, 5, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.fill();
                                ctx.beginPath();
                                ctx.arc(px, py, 10, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.yellow + '33';
                                ctx.fill();

                            } else {
                                // Post-norm layout
                                var inputY = 450, attnY = 390, add1Y = 340, ln1Y = 290, ffnY = 230, add2Y = 180, ln2Y = 130, outputY = 85;

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Input x', cx, inputY + 5);

                                drawArrow(cx, inputY - 5, cx, attnY + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, attnY, boxW, boxH, 6, '#1a2a3a', viz.colors.blue);
                                ctx.fillStyle = viz.colors.blue;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('Multi-Head Attention', cx, attnY + boxH/2);

                                drawArrow(cx, attnY - 2, cx, add1Y + 14, viz.colors.text);
                                drawAddCircle(cx, add1Y);

                                var resX = cx + boxW/2 + 30;
                                ctx.strokeStyle = viz.colors.green;
                                ctx.setLineDash([5, 3]);
                                ctx.beginPath();
                                ctx.moveTo(cx + 12, inputY - 8);
                                ctx.lineTo(resX, inputY - 8);
                                ctx.lineTo(resX, add1Y);
                                ctx.lineTo(cx + 14, add1Y);
                                ctx.stroke();
                                ctx.setLineDash([]);
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('residual', resX + 3, (inputY + add1Y) / 2);

                                drawArrow(cx, add1Y - 14, cx, ln1Y + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ln1Y, boxW, boxH, 6, '#2a1a3a', viz.colors.purple);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('LayerNorm', cx, ln1Y + boxH/2);

                                drawArrow(cx, ln1Y - 2, cx, ffnY + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ffnY, boxW, boxH, 6, '#1a3a2a', viz.colors.teal);
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText('Feed-Forward Network', cx, ffnY + boxH/2);

                                drawArrow(cx, ffnY - 2, cx, add2Y + 14, viz.colors.text);
                                drawAddCircle(cx, add2Y);

                                var resX2 = cx - boxW/2 - 30;
                                ctx.strokeStyle = viz.colors.green;
                                ctx.setLineDash([5, 3]);
                                ctx.beginPath();
                                ctx.moveTo(cx - 14, add1Y - 14);
                                ctx.lineTo(resX2, add1Y - 14);
                                ctx.lineTo(resX2, add2Y);
                                ctx.lineTo(cx - 14, add2Y);
                                ctx.stroke();
                                ctx.setLineDash([]);
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.fillText('residual', resX2 - 3, (ln1Y + add2Y) / 2);

                                drawArrow(cx, add2Y - 14, cx, ln2Y + boxH + 4, viz.colors.text);
                                drawRoundedRect(cx - boxW/2, ln2Y, boxW, boxH, 6, '#2a1a3a', viz.colors.purple);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('LayerNorm', cx, ln2Y + boxH/2);

                                drawArrow(cx, ln2Y - 2, cx, outputY + 10, viz.colors.text);
                                ctx.fillStyle = viz.colors.white;
                                ctx.fillText('Output', cx, outputY);

                                var pathPoints = [
                                    [cx, inputY], [cx, attnY + boxH/2], [cx, add1Y],
                                    [cx, ln1Y + boxH/2], [cx, ffnY + boxH/2], [cx, add2Y],
                                    [cx, ln2Y + boxH/2], [cx, outputY]
                                ];
                                var segIdx = Math.floor(animT) % pathPoints.length;
                                var segFrac = animT % 1;
                                var nextIdx = (segIdx + 1) % pathPoints.length;
                                var px = pathPoints[segIdx][0] + (pathPoints[nextIdx][0] - pathPoints[segIdx][0]) * segFrac;
                                var py = pathPoints[segIdx][1] + (pathPoints[nextIdx][1] - pathPoints[segIdx][1]) * segFrac;
                                ctx.beginPath();
                                ctx.arc(px, py, 5, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.fill();
                                ctx.beginPath();
                                ctx.arc(px, py, 10, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.yellow + '33';
                                ctx.fill();
                            }
                        }

                        viz.animate(function(t) {
                            animT = t / 800;
                            draw();
                        });

                        VizEngine.createSlider(controls, 'Pre-norm (1) / Post-norm (0)', 0, 1, 1, 1, function(v) { preNorm.value = v; });

                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A Transformer block with \\(d_{\\text{model}} = 512\\) and \\(d_{ff} = 2048\\) has two sub-layers: multi-head attention and FFN. Compute the total parameter count for the FFN sub-layer (including biases). What fraction of the block\'s parameters does the FFN represent?',
                    hint: 'The FFN has two linear layers: \\(W_1 \\in \\mathbb{R}^{d_{ff} \\times d_{\\text{model}}}\\) and \\(W_2 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{ff}}\\), plus biases.',
                    solution: '<p>FFN parameters: \\(W_1\\) has \\(2048 \\times 512 = 1{,}048{,}576\\), \\(b_1\\) has \\(2048\\), \\(W_2\\) has \\(512 \\times 2048 = 1{,}048{,}576\\), \\(b_2\\) has \\(512\\). Total FFN: \\(2{,}097{,}152 + 2{,}560 = 2{,}099{,}712\\). The MHA sub-layer has \\(4 \\times 512^2 = 1{,}048{,}576\\) parameters (ignoring biases). The two LayerNorm layers add \\(2 \\times 2 \\times 512 = 2{,}048\\). Total block: approximately \\(3{,}150{,}336\\). The FFN represents about \\(2{,}099{,}712 / 3{,}150{,}336 \\approx 67\\%\\) of the block parameters. This ratio increases with \\(d_{ff}/d_{\\text{model}}\\): most of a Transformer block\'s parameters are in the FFN, not in attention.</p>'
                },
                {
                    question: 'Explain why pre-norm allows training without learning rate warm-up, while post-norm often requires it. Consider the gradient path through each architecture.',
                    hint: 'Trace the gradient from the output to the input through the residual connections and normalization layers.',
                    solution: '<p>In <strong>pre-norm</strong>, the residual connection directly adds the input to the sub-layer output: \\(\\text{out} = x + f(\\text{LN}(x))\\). The gradient \\(\\partial \\text{out}/\\partial x = I + \\partial f / \\partial x \\cdot \\partial \\text{LN}/\\partial x\\). The identity term \\(I\\) ensures that even if the sub-layer gradient is small or poorly conditioned early in training, the gradient flows cleanly through the residual path.</p><p>In <strong>post-norm</strong>, \\(\\text{out} = \\text{LN}(x + f(x))\\), and the gradient is \\(\\partial \\text{out}/\\partial x = \\partial \\text{LN}/\\partial z \\cdot (I + \\partial f/\\partial x)\\), where \\(z = x + f(x)\\). The gradient must pass <em>through</em> the LayerNorm Jacobian, which depends on the current activations and can amplify or suppress components. At initialization, when activations are random, this Jacobian may have poor conditioning, causing unstable gradients. Warm-up starts with a small learning rate, allowing the network to reach a region where the LayerNorm Jacobian is better conditioned before increasing the learning rate.</p>'
                },
                {
                    question: 'RMSNorm omits mean subtraction. Under what conditions would RMSNorm and LayerNorm produce significantly different outputs? When are they approximately equivalent?',
                    hint: 'Consider what happens when the mean of the input vector is far from zero vs. near zero.',
                    solution: '<p>RMSNorm divides by \\(\\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum x_i^2}\\), while LayerNorm first subtracts the mean and divides by the standard deviation. They differ when the input has a <strong>large non-zero mean</strong>. If \\(x = \\mu \\mathbf{1} + \\epsilon\\) where \\(\\mu\\) is large and \\(\\epsilon\\) is small, LayerNorm subtracts \\(\\mu\\) first, giving a scale of \\(\\text{std}(\\epsilon)\\), while RMSNorm gives a scale of \\(\\approx |\\mu|\\), effectively dividing the small fluctuations by a large number. They are approximately equivalent when the mean is near zero (\\(\\mu \\approx 0\\)), because then \\(\\text{RMS}(x) \\approx \\text{std}(x)\\). In practice, residual stream activations in Transformers tend to have small means relative to their norms (especially with pre-norm), so RMSNorm works well. The learned \\(\\gamma\\) parameter can also compensate for the missing mean subtraction.</p>'
                }
            ]
        },


        // ======================== Section 6 ========================
        {
            id: 'full-transformer',
            title: 'Full Transformer Architecture',
            content: `
<h2>Full Transformer Architecture</h2>

<div class="env-block intuition">
<div class="env-title">Putting It All Together</div>
<div class="env-body">
<p>The original Transformer (Vaswani et al., 2017) uses an <strong>encoder-decoder</strong> architecture for sequence-to-sequence tasks like machine translation. The encoder processes the source sequence in parallel using self-attention, and the decoder generates the target sequence autoregressively, attending both to its own previous outputs (masked self-attention) and to the encoder output (cross-attention). This section assembles all the components from the preceding sections into the complete architecture.</p>
</div>
</div>

<h3>Encoder</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.14 &mdash; Transformer Encoder</div>
<div class="env-body">
<p>The encoder consists of \\(N\\) identical layers (\\(N = 6\\) in the original paper). Each layer contains two sub-layers:</p>
<ol>
<li><strong>Multi-head self-attention</strong> over the source sequence.</li>
<li><strong>Position-wise FFN.</strong></li>
</ol>
<p>Both sub-layers employ residual connections and layer normalization. The encoder input is the sum of token embeddings and positional encodings. The encoder processes all positions in parallel (no masking), so each source token can attend to every other source token.</p>
</div>
</div>

<h3>Decoder</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.15 &mdash; Transformer Decoder</div>
<div class="env-body">
<p>The decoder also has \\(N\\) identical layers, but each layer has <em>three</em> sub-layers:</p>
<ol>
<li><strong>Masked multi-head self-attention</strong> over the target sequence (so far).</li>
<li><strong>Multi-head cross-attention</strong> with queries from the decoder and keys/values from the encoder output.</li>
<li><strong>Position-wise FFN.</strong></li>
</ol>
<p>Each sub-layer uses residual connections and layer normalization. The final decoder output is passed through a linear layer and softmax to produce the next-token probability distribution.</p>
</div>
</div>

<h3>Masked Self-Attention</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.16 &mdash; Causal Mask</div>
<div class="env-body">
<p>In autoregressive generation, position \\(t\\) must not attend to positions \\(t+1, t+2, \\ldots\\). This is enforced by adding a <strong>causal mask</strong> \\(M\\) to the attention scores before softmax:</p>
\\[\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\right) V\\]
<p>where \\(M_{ij} = 0\\) if \\(i \\geq j\\) (allowed) and \\(M_{ij} = -\\infty\\) if \\(i &lt; j\\) (blocked). After softmax, \\(e^{-\\infty} = 0\\), so future positions receive zero attention weight.</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 15.4 &mdash; Causal Mask for 4 Tokens</div>
<div class="env-body">
<p>For a sequence of 4 tokens, the mask matrix is:</p>
\\[M = \\begin{pmatrix} 0 & -\\infty & -\\infty & -\\infty \\\\ 0 & 0 & -\\infty & -\\infty \\\\ 0 & 0 & 0 & -\\infty \\\\ 0 & 0 & 0 & 0 \\end{pmatrix}\\]
<p>Position 1 sees only itself. Position 2 sees positions 1 and 2. Position 3 sees 1, 2, and 3. This lower-triangular structure ensures the autoregressive property: the prediction at each position depends only on previous positions.</p>
</div>
</div>

<h3>Cross-Attention</h3>

<div class="env-block definition">
<div class="env-title">Definition 15.17 &mdash; Cross-Attention</div>
<div class="env-body">
<p>In the encoder-decoder cross-attention sub-layer, the <strong>queries</strong> come from the decoder (the current target representation), while the <strong>keys and values</strong> come from the encoder output:</p>
\\[\\text{CrossAttn}(Q_{\\text{dec}}, K_{\\text{enc}}, V_{\\text{enc}}) = \\text{softmax}\\!\\left(\\frac{Q_{\\text{dec}} K_{\\text{enc}}^\\top}{\\sqrt{d_k}}\\right) V_{\\text{enc}}\\]
<p>This allows every decoder position to attend to every encoder position, implementing the "looking back at the source" behavior that Bahdanau attention introduced for RNNs, now within the purely attention-based framework.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Encoder-Only and Decoder-Only Variants</div>
<div class="env-body">
<p>The full encoder-decoder architecture is used for sequence-to-sequence tasks (translation, summarization). In practice, many modern models use simplified variants:</p>
<ul>
<li><strong>Encoder-only</strong> (BERT, RoBERTa): Uses only the encoder stack with bidirectional self-attention. Suitable for classification, NER, and other tasks where the full input is available. No causal mask.</li>
<li><strong>Decoder-only</strong> (GPT, LLaMA, Claude): Uses only the decoder stack with causal masking. The dominant architecture for language modeling and text generation. Cross-attention is removed; the model conditions purely on its own previous tokens.</li>
</ul>
<p>Decoder-only models have become the standard for large language models because of their simplicity, scalability, and the finding that they can match encoder-decoder performance on most tasks when scaled sufficiently.</p>
</div>
</div>

<div class="env-block warning">
<div class="env-title">The Quadratic Bottleneck</div>
<div class="env-body">
<p>The \\(O(n^2)\\) attention complexity limits the maximum sequence length. For \\(n = 8192\\) tokens with \\(d = 128\\) and 32 heads, the attention matrices alone require \\(32 \\times 8192^2 \\times 4\\) bytes \\(\\approx 8.6\\) GB per layer. Techniques to address this include:</p>
<ul>
<li><strong>FlashAttention</strong> (Dao et al., 2022): Fuses attention operations to avoid materializing the \\(n \\times n\\) matrix, reducing memory from \\(O(n^2)\\) to \\(O(n)\\) while maintaining exact computation.</li>
<li><strong>Sparse attention</strong> (Longformer, BigBird): Each token attends to only \\(O(\\sqrt{n})\\) or \\(O(n)\\) positions using local windows and global tokens.</li>
<li><strong>Ring attention</strong>: Distributes long sequences across devices, overlapping communication with computation.</li>
</ul>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 15.5 &mdash; Transformer-base Configuration</div>
<div class="env-body">
<p>The original "Transformer-base" model from Vaswani et al. (2017):</p>
<ul>
<li>\\(N = 6\\) encoder layers, \\(N = 6\\) decoder layers</li>
<li>\\(d_{\\text{model}} = 512\\), \\(d_{ff} = 2048\\), \\(h = 8\\), \\(d_k = d_v = 64\\)</li>
<li>Sinusoidal positional encoding</li>
<li>Total parameters: approximately 65 million</li>
<li>Trained on WMT 2014 English-German (4.5M sentence pairs)</li>
<li>Achieved 28.4 BLEU, surpassing all previous single models</li>
</ul>
<p>Training used 8 NVIDIA P100 GPUs for 3.5 days, a tiny fraction of the compute used for modern LLMs.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="full-transformer-viz"></div>

<p>The visualization above shows the full encoder-decoder Transformer architecture. Click on different layers to see their internal structure. The encoder (left) processes the source sequence with bidirectional self-attention, and the decoder (right) generates the target with masked self-attention and cross-attention to the encoder. The color highlighting shows the currently selected component and its data flow.</p>
`,
            visualizations: [
                {
                    id: 'full-transformer-viz',
                    title: 'Full Encoder-Decoder Transformer',
                    description: 'The complete Transformer architecture. Use the slider to highlight different components: embeddings, encoder self-attention, decoder masked attention, cross-attention, FFN, and output.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 500, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var highlight = { value: 0 };
                        var hlNames = ['Input Embeddings + PE', 'Encoder Self-Attention', 'Encoder FFN', 'Decoder Masked Self-Attn', 'Decoder Cross-Attention', 'Decoder FFN', 'Output Linear + Softmax'];

                        function drawRR(x, y, w, h, r, fill, stroke, lw) {
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
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var hl = Math.round(highlight.value);
                            var encX = 100, decX = 430;
                            var colW = 180, boxH = 28, gap = 6;

                            // Title + highlight label
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Highlighted: ' + hlNames[hl], viz.width / 2, 5);

                            // --- Encoder side ---
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Encoder', encX + colW / 2, 25);

                            // Input embeddings + PE
                            var embY = 460;
                            var embHL = (hl === 0);
                            drawRR(encX, embY, colW, boxH, 5, embHL ? '#2a3a1a' : '#1a1a2a', embHL ? viz.colors.green : '#30363d');
                            ctx.fillStyle = embHL ? viz.colors.green : viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Embeddings + PE', encX + colW / 2, embY + boxH / 2);

                            // Source label
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Source tokens', encX + colW / 2, embY + boxH + 12);

                            // Encoder layers (draw 3 to represent Nx)
                            var layerH = boxH * 2 + gap * 3 + 20;
                            var numEnc = 3;
                            for (var l = 0; l < numEnc; l++) {
                                var ly = embY - 40 - l * (layerH + 8);

                                // Background for layer
                                drawRR(encX - 8, ly - 4, colW + 16, layerH, 8, '#0f0f1a', '#30363d', 0.5);

                                if (l === 0) {
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.textAlign = 'right';
                                    ctx.fillText('x' + numEnc, encX - 12, ly + layerH / 2);
                                }

                                // Self-attention box
                                var saY = ly + 4;
                                var saHL = (hl === 1);
                                drawRR(encX + 10, saY, colW - 20, boxH, 4, saHL ? '#1a2a4a' : '#1a1a2a', saHL ? viz.colors.blue : '#30363d');
                                ctx.fillStyle = saHL ? viz.colors.blue : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('Multi-Head Self-Attention', encX + colW / 2, saY + boxH / 2);

                                // Add & Norm
                                drawRR(encX + 10, saY + boxH + gap, colW - 20, 16, 3, '#1a1a1a', '#30363d', 0.5);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.fillText('Add & Norm', encX + colW / 2, saY + boxH + gap + 8);

                                // FFN box
                                var ffY = saY + boxH + gap + 16 + gap;
                                var ffHL = (hl === 2);
                                drawRR(encX + 10, ffY, colW - 20, boxH, 4, ffHL ? '#1a3a2a' : '#1a1a2a', ffHL ? viz.colors.teal : '#30363d');
                                ctx.fillStyle = ffHL ? viz.colors.teal : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('Feed-Forward Network', encX + colW / 2, ffY + boxH / 2);

                                // Add & Norm
                                drawRR(encX + 10, ffY + boxH + gap, colW - 20, 16, 3, '#1a1a1a', '#30363d', 0.5);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.fillText('Add & Norm', encX + colW / 2, ffY + boxH + gap + 8);
                            }

                            // --- Decoder side ---
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Decoder', decX + colW / 2, 25);

                            // Decoder input embeddings + PE
                            drawRR(decX, embY, colW, boxH, 5, embHL ? '#2a3a1a' : '#1a1a2a', embHL ? viz.colors.green : '#30363d');
                            ctx.fillStyle = embHL ? viz.colors.green : viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Embeddings + PE', decX + colW / 2, embY + boxH / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Target tokens (shifted)', decX + colW / 2, embY + boxH + 12);

                            // Decoder layers (3 = Nx)
                            var dLayerH = boxH * 3 + gap * 5 + 36;
                            for (var l = 0; l < numEnc; l++) {
                                var ly = embY - 40 - l * (dLayerH + 8);
                                if (ly < 30) continue; // don't draw if too high

                                drawRR(decX - 8, ly - 4, colW + 16, dLayerH, 8, '#0f0f1a', '#30363d', 0.5);

                                if (l === 0) {
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.textAlign = 'right';
                                    ctx.fillText('x' + numEnc, decX - 12, ly + dLayerH / 2);
                                }

                                // Masked self-attention
                                var msaY = ly + 4;
                                var msaHL = (hl === 3);
                                drawRR(decX + 10, msaY, colW - 20, boxH, 4, msaHL ? '#3a2a1a' : '#1a1a2a', msaHL ? viz.colors.orange : '#30363d');
                                ctx.fillStyle = msaHL ? viz.colors.orange : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('Masked Self-Attention', decX + colW / 2, msaY + boxH / 2);

                                drawRR(decX + 10, msaY + boxH + gap, colW - 20, 16, 3, '#1a1a1a', '#30363d', 0.5);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.fillText('Add & Norm', decX + colW / 2, msaY + boxH + gap + 8);

                                // Cross-attention
                                var caY = msaY + boxH + gap + 16 + gap;
                                var caHL = (hl === 4);
                                drawRR(decX + 10, caY, colW - 20, boxH, 4, caHL ? '#2a1a3a' : '#1a1a2a', caHL ? viz.colors.pink : '#30363d');
                                ctx.fillStyle = caHL ? viz.colors.pink : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('Cross-Attention', decX + colW / 2, caY + boxH / 2);

                                // Arrow from encoder to cross-attention
                                if (l < numEnc) {
                                    ctx.strokeStyle = caHL ? viz.colors.pink : '#30363d';
                                    ctx.lineWidth = caHL ? 2 : 1;
                                    ctx.setLineDash(caHL ? [] : [3, 3]);
                                    ctx.beginPath();
                                    ctx.moveTo(encX + colW + 8, caY + boxH / 2);
                                    ctx.lineTo(decX + 10, caY + boxH / 2);
                                    ctx.stroke();
                                    // arrowhead
                                    ctx.fillStyle = caHL ? viz.colors.pink : '#30363d';
                                    ctx.beginPath();
                                    ctx.moveTo(decX + 10, caY + boxH / 2);
                                    ctx.lineTo(decX + 4, caY + boxH / 2 - 4);
                                    ctx.lineTo(decX + 4, caY + boxH / 2 + 4);
                                    ctx.closePath();
                                    ctx.fill();
                                    ctx.setLineDash([]);
                                    // K,V label
                                    if (caHL) {
                                        ctx.fillStyle = viz.colors.pink;
                                        ctx.font = '9px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.fillText('K, V from encoder', (encX + colW + decX + 10) / 2, caY + boxH / 2 - 10);
                                    }
                                }

                                drawRR(decX + 10, caY + boxH + gap, colW - 20, 16, 3, '#1a1a1a', '#30363d', 0.5);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Add & Norm', decX + colW / 2, caY + boxH + gap + 8);

                                // FFN
                                var dffY = caY + boxH + gap + 16 + gap;
                                var dffHL = (hl === 5);
                                drawRR(decX + 10, dffY, colW - 20, boxH, 4, dffHL ? '#1a3a2a' : '#1a1a2a', dffHL ? viz.colors.teal : '#30363d');
                                ctx.fillStyle = dffHL ? viz.colors.teal : viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('Feed-Forward Network', decX + colW / 2, dffY + boxH / 2);

                                drawRR(decX + 10, dffY + boxH + gap, colW - 20, 16, 3, '#1a1a1a', '#30363d', 0.5);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.fillText('Add & Norm', decX + colW / 2, dffY + boxH + gap + 8);
                            }

                            // Output head
                            var outY = 42;
                            var outHL = (hl === 6);
                            drawRR(decX, outY, colW, boxH, 5, outHL ? '#3a3a1a' : '#1a1a2a', outHL ? viz.colors.yellow : '#30363d');
                            ctx.fillStyle = outHL ? viz.colors.yellow : viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Linear + Softmax', decX + colW / 2, outY + boxH / 2);
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Component', 0, 6, 0, 1, function(v) { highlight.value = v; draw(); });
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Why must the decoder use a causal mask in self-attention during training but not during inference? (Hint: think about teacher forcing vs. autoregressive generation.)',
                    hint: 'During training, the entire target sequence is available. During inference, tokens are generated one at a time.',
                    solution: '<p>During <strong>training with teacher forcing</strong>, the entire target sequence \\(y_1, \\ldots, y_{T\'}\\) is fed to the decoder simultaneously. Without the causal mask, position \\(t\\) could attend to future positions \\(y_{t+1}, \\ldots, y_{T\'}\\), leaking information about the ground truth. The causal mask prevents this, ensuring the training condition matches inference. During <strong>inference</strong>, the mask is technically unnecessary because tokens are generated one at a time: when generating \\(y_t\\), the tokens \\(y_{t+1}, \\ldots\\) do not yet exist. However, implementations typically still apply the mask for two reasons: (1) KV-caching means all previous tokens are processed together, and the mask ensures consistency; (2) it simplifies the code to use the same masked attention in both modes.</p>'
                },
                {
                    question: 'In the encoder-decoder Transformer, the cross-attention uses queries from the decoder and keys/values from the encoder. What would happen if we reversed this (queries from encoder, keys/values from decoder)?',
                    hint: 'Think about the information flow direction: who needs to look at whom?',
                    solution: '<p>Reversing the cross-attention would be semantically wrong for generation tasks. The decoder needs to <em>query</em> the encoder to decide which source information is relevant for generating the next target token. If reversed, the encoder would be querying the decoder, asking "which target tokens are relevant for understanding this source token?" This is backward: the encoder processes the source <em>before</em> the decoder runs, so it cannot condition on decoder states that do not yet exist.</p><p>Interestingly, reversed cross-attention does appear in some architectures for different purposes. In Perceiver (Jaegle et al., 2021), a small set of latent vectors queries a large input, then the input queries the latents back, creating a bidirectional information exchange. But for standard sequence-to-sequence generation, the direction must be decoder-queries-encoder.</p>'
                },
                {
                    question: 'Compute the total parameter count for the Transformer-base model (\\(N=6\\), \\(d_{\\text{model}}=512\\), \\(d_{ff}=2048\\), \\(h=8\\), vocabulary size \\(V=37000\\), shared embeddings). Account for all components.',
                    hint: 'Components: token embeddings, positional encoding (no params for sinusoidal), encoder layers (MHA + FFN + LN per layer), decoder layers (masked MHA + cross-MHA + FFN + LN per layer), output linear.',
                    solution: '<p><strong>Token embeddings:</strong> \\(V \\times d_{\\text{model}} = 37{,}000 \\times 512 = 18{,}944{,}000\\). Shared between encoder input, decoder input, and output projection.</p><p><strong>Per encoder layer:</strong> MHA: \\(4 \\times 512^2 = 1{,}048{,}576\\). FFN: \\(2 \\times 512 \\times 2048 = 2{,}097{,}152\\). Two LayerNorms: \\(2 \\times 2 \\times 512 = 2{,}048\\). Total per layer: \\(3{,}147{,}776\\). Six layers: \\(18{,}886{,}656\\).</p><p><strong>Per decoder layer:</strong> Masked MHA + Cross-MHA + FFN = \\(2 \\times 1{,}048{,}576 + 2{,}097{,}152 = 4{,}194{,}304\\). Three LayerNorms: \\(3{,}072\\). Total per layer: \\(4{,}197{,}376\\). Six layers: \\(25{,}184{,}256\\).</p><p><strong>Output bias (optional):</strong> \\(V = 37{,}000\\).</p><p><strong>Grand total:</strong> \\(\\approx 18{,}944{,}000 + 18{,}886{,}656 + 25{,}184{,}256 \\approx 63{,}015{,}000 \\approx 63\\)M parameters. This matches the reported ~65M (the small difference comes from biases and the final layer norm).</p>'
                },
                {
                    question: 'Decoder-only models (GPT, LLaMA) have become dominant over encoder-decoder models for most NLP tasks. Give two reasons why, and one scenario where encoder-decoder remains advantageous.',
                    hint: 'Think about simplicity, scaling laws, and the nature of different tasks.',
                    solution: '<p><strong>Reason 1 (Simplicity and scaling):</strong> Decoder-only models have a single, uniform architecture: stacked Transformer blocks with causal masking. This simplicity makes them easier to scale (no need to balance encoder vs. decoder depth), easier to implement KV-caching for efficient inference, and easier to parallelize during training. Scaling laws (Kaplan et al., 2020) were developed for decoder-only models and show smooth power-law improvements.</p><p><strong>Reason 2 (Unified task formulation):</strong> Any NLP task can be cast as text generation: classification becomes "Generate the class label," translation becomes "Generate the translation," QA becomes "Generate the answer." This unification means a single decoder-only model can handle all tasks via prompting or fine-tuning, without architectural changes.</p><p><strong>Encoder-decoder advantage:</strong> Tasks with a clear input-output separation where the input should be processed bidirectionally benefit from encoder-decoder. Machine translation is the classic example: the encoder reads the full source sentence with bidirectional attention, capturing global context before the decoder begins generating. T5 and mBART show that encoder-decoder models can outperform similarly-sized decoder-only models on translation and summarization, especially at smaller scales.</p>'
                }
            ]
        }

    ]
});
