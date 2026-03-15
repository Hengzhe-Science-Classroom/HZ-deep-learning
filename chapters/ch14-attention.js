// === Chapter 14: Attention Mechanism ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch14',
    number: 14,
    title: 'Attention Mechanism',
    subtitle: 'Breaking the bottleneck: from fixed context vectors to dynamic, content-based addressing',
    sections: [
        // ======================== Section 1 ========================
        {
            id: 'seq2seq-bottleneck',
            title: 'Seq2Seq & The Bottleneck',
            content: `
<h2>Seq2Seq &amp; The Bottleneck</h2>

<div class="env-block intuition">
<div class="env-title">From Sequences to Sequences</div>
<div class="env-body">
<p>Many important problems in deep learning involve transforming one variable-length sequence into another: machine translation (English to French), speech recognition (audio frames to text), text summarization (long document to short summary). In Chapters 12 and 13, we studied RNNs and LSTMs that can process sequences of arbitrary length. But how do we produce an <em>output sequence</em> that may differ in length from the input? The <strong>encoder-decoder</strong> (Seq2Seq) architecture, introduced by Sutskever et al. (2014) and Cho et al. (2014), provides the foundational answer.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 14.1 &mdash; Encoder-Decoder Architecture</div>
<div class="env-body">
<p>An <strong>encoder-decoder</strong> model consists of two components:</p>
<ul>
<li><strong>Encoder:</strong> An RNN (or LSTM/GRU) that reads the source sequence \\(x_1, x_2, \\ldots, x_T\\) and produces hidden states \\(h_1, h_2, \\ldots, h_T\\). The final hidden state \\(h_T\\) (or a function of it) serves as the <strong>context vector</strong> \\(c\\).</li>
<li><strong>Decoder:</strong> A second RNN that generates the target sequence \\(y_1, y_2, \\ldots, y_{T'}\\) one token at a time, conditioned on the context vector \\(c\\) and its own previous outputs.</li>
</ul>
<p>The decoder hidden state at step \\(t\\) is:</p>
\\[s_t = f(s_{t-1},\\, y_{t-1},\\, c)\\]
<p>and the output distribution is:</p>
\\[P(y_t \\mid y_{&lt;t},\\, x) = g(s_t,\\, y_{t-1},\\, c)\\]
<p>where \\(f\\) and \\(g\\) are learned nonlinear functions (typically involving an LSTM cell and a softmax layer).</p>
</div>
</div>

<p>The elegance of this approach lies in its modularity: the encoder compresses the source into a fixed representation, and the decoder unfolds that representation into a target. Training uses teacher forcing, where the true previous token is fed to the decoder during training.</p>

<div class="env-block example">
<div class="env-title">Example 14.1 &mdash; Machine Translation</div>
<div class="env-body">
<p>Consider translating "The cat sat on the mat" (6 tokens) to "Le chat s'est assis sur le tapis" (7 tokens in French). The encoder reads all 6 English tokens and compresses them into a single context vector \\(c \\in \\mathbb{R}^d\\). The decoder then generates the 7 French tokens sequentially, each time conditioning on \\(c\\).</p>
</div>
</div>

<h3>The Information Bottleneck</h3>

<p>The critical weakness of the basic Seq2Seq model is that <em>the entire source sequence must be compressed into a single fixed-dimensional vector</em> \\(c\\). This creates an <strong>information bottleneck</strong>.</p>

<div class="env-block warning">
<div class="env-title">The Bottleneck Problem</div>
<div class="env-body">
<p>Consider the encoder's task for a 50-word sentence. All semantic relationships, word orderings, and nuances must be packed into one vector of dimension \\(d\\) (typically 256 to 1024). This is analogous to compressing an entire book into a single paragraph. The consequences are severe:</p>
<ul>
<li><strong>Long-range information loss:</strong> Early tokens in the source sequence are processed many time steps before the context is extracted, leading to vanishing gradient effects and information decay.</li>
<li><strong>Performance degradation with length:</strong> Cho et al. (2014) showed empirically that Seq2Seq BLEU scores drop sharply for sentences longer than about 20 tokens.</li>
<li><strong>Uniform compression:</strong> Every source token is treated equally in contributing to \\(c\\), regardless of its relevance to the current decoder step.</li>
</ul>
</div>
</div>

<div class="viz-placeholder" data-viz="seq2seq-bottleneck-viz"></div>

<p>The visualization above illustrates the problem concretely. As the source sequence grows longer, the fixed-size context vector must store proportionally more information per dimension. Information from earlier time steps is progressively overwritten or attenuated.</p>

<div class="env-block intuition">
<div class="env-title">Analogy</div>
<div class="env-body">
<p>Imagine a human translator who must listen to an entire lecture, then <em>close the textbook</em> and translate from memory alone. For a short sentence, this works fine. For a long paragraph, the translator inevitably forgets details. A better strategy: keep the textbook open and refer back to specific passages while translating each sentence. This is exactly what attention mechanisms do.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Bidirectional Encoders</div>
<div class="env-body">
<p>In practice, encoders are often bidirectional: a forward RNN reads \\(x_1, \\ldots, x_T\\) and a backward RNN reads \\(x_T, \\ldots, x_1\\). Each encoder hidden state is the concatenation \\(h_i = [\\overrightarrow{h}_i;\\, \\overleftarrow{h}_i]\\), capturing both past and future context around position \\(i\\). The context vector in vanilla Seq2Seq is then \\(c = h_T\\) (or some function of \\(h_T\\) from both directions). Even with bidirectional encoding, the bottleneck remains: all information must still flow through a single vector.</p>
</div>
</div>

<p>To quantify the bottleneck, consider that a single hidden state \\(h_T \\in \\mathbb{R}^d\\) stores \\(d\\) floating-point numbers. For a source sentence of length \\(T\\), the information per source token available to the decoder is at most \\(d/T\\) numbers. As \\(T\\) grows, the representation becomes increasingly lossy.</p>

<div class="env-block definition">
<div class="env-title">Definition 14.2 &mdash; Information Rate</div>
<div class="env-body">
<p>The <strong>effective information rate</strong> of a fixed-context encoder-decoder is the ratio \\(d / T\\), where \\(d\\) is the context vector dimension and \\(T\\) is the source sequence length. This rate decreases linearly with sequence length, implying that longer sequences suffer proportionally greater information loss.</p>
</div>
</div>

<p>The solution, as we will see in the next section, is to let the decoder <em>look back</em> at all encoder hidden states \\(h_1, \\ldots, h_T\\) at each decoding step, selectively focusing on the most relevant parts. This is the attention mechanism.</p>
`,
            visualizations: [
                {
                    id: 'seq2seq-bottleneck-viz',
                    title: 'Seq2Seq Information Bottleneck',
                    description: 'Use the slider to increase the source sequence length and observe how information must compress through the fixed-size context vector. Color intensity represents information density.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 700, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var seqLen = { value: 4 };

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var T = Math.round(seqLen.value);
                            var boxW = 50, boxH = 40;
                            var encY = 160, decY = 300;
                            var bottleneckX = viz.width / 2;
                            var bottleneckY = 230;

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Encoder', viz.width / 2, 30);

                            // Calculate encoder positions
                            var totalEncW = T * boxW + (T - 1) * 12;
                            var encStartX = (viz.width - totalEncW) / 2;

                            // Draw encoder boxes
                            for (var i = 0; i < T; i++) {
                                var ex = encStartX + i * (boxW + 12);
                                // Color intensity decreases for earlier tokens (information decay)
                                var decay = 0.3 + 0.7 * (i / (T - 1 || 1));
                                var r = Math.round(88 * decay);
                                var g = Math.round(166 * decay);
                                var b = Math.round(255 * decay);
                                ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                                ctx.fillRect(ex, encY, boxW, boxH);
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 1.5;
                                ctx.strokeRect(ex, encY, boxW, boxH);
                                // Token label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('h' + (i + 1), ex + boxW / 2, encY + boxH / 2);
                                // Input token
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('x' + (i + 1), ex + boxW / 2, encY - 14);

                                // Arrows between encoder boxes
                                if (i < T - 1) {
                                    ctx.strokeStyle = viz.colors.text;
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.moveTo(ex + boxW + 1, encY + boxH / 2);
                                    ctx.lineTo(ex + boxW + 11, encY + boxH / 2);
                                    ctx.stroke();
                                    // arrowhead
                                    ctx.beginPath();
                                    ctx.moveTo(ex + boxW + 11, encY + boxH / 2);
                                    ctx.lineTo(ex + boxW + 6, encY + boxH / 2 - 3);
                                    ctx.lineTo(ex + boxW + 6, encY + boxH / 2 + 3);
                                    ctx.closePath();
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.fill();
                                }

                                // Lines converging to bottleneck
                                ctx.strokeStyle = viz.colors.text + '66';
                                ctx.lineWidth = 1;
                                ctx.setLineDash([3, 3]);
                                ctx.beginPath();
                                ctx.moveTo(ex + boxW / 2, encY + boxH);
                                ctx.lineTo(bottleneckX, bottleneckY - 18);
                                ctx.stroke();
                                ctx.setLineDash([]);
                            }

                            // Bottleneck (context vector c)
                            var bWidth = 36;
                            var bHeight = 30;
                            // Color intensity based on compression
                            var compression = Math.min(1, 4 / T);
                            var rr = Math.round(248 * compression + 100 * (1 - compression));
                            var gg = Math.round(81 * (1 - compression) + 50 * compression);
                            var bb = Math.round(73 * (1 - compression) + 50 * compression);
                            ctx.fillStyle = 'rgb(' + rr + ',' + gg + ',' + bb + ')';
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            // Draw diamond shape for bottleneck
                            ctx.beginPath();
                            ctx.moveTo(bottleneckX, bottleneckY - bHeight / 2);
                            ctx.lineTo(bottleneckX + bWidth / 2, bottleneckY);
                            ctx.lineTo(bottleneckX, bottleneckY + bHeight / 2);
                            ctx.lineTo(bottleneckX - bWidth / 2, bottleneckY);
                            ctx.closePath();
                            ctx.fill();
                            ctx.stroke();
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('c', bottleneckX, bottleneckY);

                            // Info rate text
                            ctx.fillStyle = compression > 0.5 ? viz.colors.green : viz.colors.red;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('Info rate: d/' + T, bottleneckX, bottleneckY + bHeight / 2 + 16);

                            // Decoder
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.fillText('Decoder', viz.width / 2, decY - 30);

                            var decTokens = Math.max(2, Math.round(T * 0.8));
                            var totalDecW = decTokens * boxW + (decTokens - 1) * 12;
                            var decStartX = (viz.width - totalDecW) / 2;

                            // Line from bottleneck to decoder
                            ctx.strokeStyle = viz.colors.orange + '88';
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([4, 3]);
                            for (var j = 0; j < decTokens; j++) {
                                var dx = decStartX + j * (boxW + 12) + boxW / 2;
                                ctx.beginPath();
                                ctx.moveTo(bottleneckX, bottleneckY + bHeight / 2);
                                ctx.lineTo(dx, decY);
                                ctx.stroke();
                            }
                            ctx.setLineDash([]);

                            // Draw decoder boxes
                            for (var j = 0; j < decTokens; j++) {
                                var dx = decStartX + j * (boxW + 12);
                                ctx.fillStyle = viz.colors.teal + '55';
                                ctx.fillRect(dx, decY, boxW, boxH);
                                ctx.strokeStyle = viz.colors.teal;
                                ctx.lineWidth = 1.5;
                                ctx.strokeRect(dx, decY, boxW, boxH);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('s' + (j + 1), dx + boxW / 2, decY + boxH / 2);
                                // Output token
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('y' + (j + 1), dx + boxW / 2, decY + boxH + 14);

                                // Arrows between decoder boxes
                                if (j < decTokens - 1) {
                                    ctx.strokeStyle = viz.colors.text;
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.moveTo(dx + boxW + 1, decY + boxH / 2);
                                    ctx.lineTo(dx + boxW + 11, decY + boxH / 2);
                                    ctx.stroke();
                                    ctx.beginPath();
                                    ctx.moveTo(dx + boxW + 11, decY + boxH / 2);
                                    ctx.lineTo(dx + boxW + 6, decY + boxH / 2 - 3);
                                    ctx.lineTo(dx + boxW + 6, decY + boxH / 2 + 3);
                                    ctx.closePath();
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.fill();
                                }
                            }

                            // Compression warning bar at bottom
                            var barY = 385;
                            var barW = viz.width - 80;
                            var barH = 12;
                            var barX = 40;
                            ctx.fillStyle = '#1a1a40';
                            ctx.fillRect(barX, barY, barW, barH);
                            // Fill with gradient from green to red
                            var fillW = barW * Math.min(1, T / 12);
                            var grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
                            grad.addColorStop(0, viz.colors.green);
                            grad.addColorStop(0.5, viz.colors.yellow);
                            grad.addColorStop(1, viz.colors.red);
                            ctx.fillStyle = grad;
                            ctx.fillRect(barX, barY, fillW, barH);
                            ctx.strokeStyle = viz.colors.text;
                            ctx.lineWidth = 0.5;
                            ctx.strokeRect(barX, barY, barW, barH);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Low compression', barX, barY + barH + 12);
                            ctx.textAlign = 'right';
                            ctx.fillText('High compression (information loss)', barX + barW, barY + barH + 12);

                            // Source length indicator
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Source length T = ' + T, 20, 65);

                            // Info density label
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            if (T <= 4) {
                                ctx.fillText('Context vector has enough capacity', 20, 85);
                            } else if (T <= 8) {
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.fillText('Compression increasing, details may be lost', 20, 85);
                            } else {
                                ctx.fillStyle = viz.colors.red;
                                ctx.fillText('Severe bottleneck: early tokens are forgotten', 20, 85);
                            }
                        }

                        draw();
                        VizEngine.createSlider(controls, 'Sequence length', 2, 10, seqLen.value, 1, function(v) {
                            seqLen.value = v;
                            draw();
                        });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'In a standard Seq2Seq model, the encoder produces hidden states \\(h_1, \\ldots, h_T\\) and the decoder receives context \\(c = h_T\\). If the encoder is a unidirectional LSTM with hidden size 512, and the source sentence has 40 tokens, how many floating-point numbers does the decoder have to reconstruct the source from?',
                    hint: 'The context vector \\(c\\) has the same dimensionality as the LSTM hidden state.',
                    solution: 'The decoder receives \\(c = h_T \\in \\mathbb{R}^{512}\\). This is exactly 512 floating-point numbers, regardless of the source length. The effective information rate is \\(512/40 = 12.8\\) numbers per source token, a severe compression compared to the 512 numbers available for short sequences.'
                },
                {
                    question: 'Why does a bidirectional encoder not solve the bottleneck problem?',
                    hint: 'Think about what happens after the bidirectional encoding: how is the context vector formed?',
                    solution: 'A bidirectional encoder produces richer hidden states \\(h_i = [\\overrightarrow{h}_i;\\, \\overleftarrow{h}_i]\\) that capture both left and right context at each position. However, the vanilla Seq2Seq decoder still compresses these into a single context vector (e.g., \\(c = h_T\\) or \\(c = [\\overrightarrow{h}_T;\\, \\overleftarrow{h}_1]\\)). The bottleneck remains: one fixed-size vector must encode the entire sequence. The issue is not the quality of individual encoder states but the fact that the decoder only sees a single summary.'
                },
                {
                    question: 'Cho et al. (2014) observed that BLEU scores degrade for sentences longer than ~20 tokens. Propose two simple (non-attention) strategies to mitigate this and explain their limitations.',
                    hint: 'Think about reversing the input, increasing hidden size, or using multiple context vectors.',
                    solution: '<p><strong>Strategy 1: Reverse the source sequence.</strong> Sutskever et al. (2014) found that feeding the source in reverse order helps because the first source tokens (which map to the first target tokens) are closer to the context vector extraction point. Limitation: this helps the beginning of the sentence but hurts the end.</p><p><strong>Strategy 2: Increase hidden size \\(d\\).</strong> A larger context vector can store more information. Limitation: this increases parameters quadratically (LSTM has \\(O(d^2)\\) parameters), making training slower and risking overfitting. Moreover, the linear growth of capacity vs. the linear growth of information demand means the problem is merely delayed, not solved.</p>'
                },
                {
                    question: 'Formally, write the Seq2Seq training objective (conditional log-likelihood) for a source-target pair \\((x, y)\\) where \\(y = (y_1, \\ldots, y_{T\'})\\). Why does the bottleneck affect the gradient signal for early encoder time steps?',
                    hint: 'Use the chain rule: the gradient with respect to \\(h_1\\) must pass through \\(h_2, h_3, \\ldots, h_T\\) and then through \\(c\\).',
                    solution: '<p>The training objective is:</p><p>\\[\\mathcal{L}(\\theta) = \\sum_{t=1}^{T\'} \\log P(y_t \\mid y_1, \\ldots, y_{t-1},\\, c;\\, \\theta)\\]</p><p>where \\(c = h_T\\) and \\(h_t = f(h_{t-1}, x_t)\\). The gradient \\(\\frac{\\partial \\mathcal{L}}{\\partial h_1}\\) requires backpropagation through \\(T-1\\) recurrent steps: \\(\\frac{\\partial \\mathcal{L}}{\\partial h_1} = \\frac{\\partial \\mathcal{L}}{\\partial c} \\cdot \\frac{\\partial c}{\\partial h_T} \\cdot \\prod_{t=2}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}}\\). This product of Jacobians causes vanishing (or exploding) gradients, so early encoder states receive weak learning signals. The bottleneck compounds this: even if gradients flowed perfectly, \\(h_1\\)\'s contribution to \\(c\\) is diluted by subsequent overwrites.</p>'
                }
            ]
        },

        // ======================== Section 2 ========================
        {
            id: 'bahdanau-attention',
            title: 'Bahdanau Attention',
            content: `
<h2>Bahdanau (Additive) Attention</h2>

<div class="env-block intuition">
<div class="env-title">The Key Idea</div>
<div class="env-body">
<p>Instead of forcing the decoder to work with a single fixed context vector, let the decoder <em>look back</em> at all encoder hidden states at every time step. At each decoding step \\(t\\), the model computes a <strong>relevance score</strong> between the current decoder state and each encoder state, then forms a <strong>weighted sum</strong> of encoder states. This weighted sum, the <strong>context vector</strong> \\(c_t\\), changes at every decoding step, dynamically focusing on different parts of the source.</p>
</div>
</div>

<p>Bahdanau et al. (2015), in their landmark paper "Neural Machine Translation by Jointly Learning to Align and Translate," introduced this mechanism. The word "align" is deliberate: the attention weights learn a soft alignment between source and target positions.</p>

<div class="env-block definition">
<div class="env-title">Definition 14.3 &mdash; Bahdanau Attention</div>
<div class="env-body">
<p>Given encoder hidden states \\(h_1, \\ldots, h_T\\) and the decoder hidden state \\(s_{t-1}\\) at the previous step, the <strong>Bahdanau attention mechanism</strong> computes:</p>
<p><strong>Step 1. Alignment scores</strong> (also called energy):</p>
\\[e_{t,i} = v_a^\\top \\tanh(W_a \\, s_{t-1} + U_a \\, h_i)\\]
<p>where \\(W_a \\in \\mathbb{R}^{d_a \\times d_s}\\), \\(U_a \\in \\mathbb{R}^{d_a \\times d_h}\\), and \\(v_a \\in \\mathbb{R}^{d_a}\\) are learnable parameters, and \\(d_a\\) is the attention dimensionality.</p>
<p><strong>Step 2. Attention weights</strong> (via softmax normalization):</p>
\\[\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_{j=1}^{T} \\exp(e_{t,j})}\\]
<p><strong>Step 3. Context vector</strong> (weighted sum of encoder states):</p>
\\[c_t = \\sum_{i=1}^{T} \\alpha_{t,i} \\, h_i\\]
<p><strong>Step 4. Decoder update</strong>:</p>
\\[s_t = f(s_{t-1},\\, y_{t-1},\\, c_t)\\]
</div>
</div>

<p>The alignment score function \\(e_{t,i} = v_a^\\top \\tanh(W_a \\, s_{t-1} + U_a \\, h_i)\\) is called <strong>additive attention</strong> because the decoder and encoder representations are combined through addition inside the \\(\\tanh\\). This is a small feedforward network with one hidden layer.</p>

<div class="env-block intuition">
<div class="env-title">Interpreting the Alignment Score</div>
<div class="env-body">
<p>Think of the alignment score \\(e_{t,i}\\) as answering: "How relevant is source position \\(i\\) for generating the next target token at step \\(t\\)?" The function projects both the decoder state \\(s_{t-1}\\) and the encoder state \\(h_i\\) into a common space (via \\(W_a\\) and \\(U_a\\)), adds them, passes through \\(\\tanh\\), and produces a scalar score via \\(v_a\\). The softmax then converts these scores into a valid probability distribution \\(\\alpha_{t,\\cdot}\\) over source positions.</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 14.2 &mdash; Attention in Action</div>
<div class="env-body">
<p>Translating "The black cat" to "Le chat noir": when generating "chat" (cat), the attention should focus heavily on "cat" (\\(\\alpha_{2,3}\\) large). When generating "noir" (black), it should shift to "black" (\\(\\alpha_{3,2}\\) large). This demonstrates that attention learns a non-monotonic alignment; French adjective order differs from English.</p>
</div>
</div>

<h3>Why This Works</h3>

<p>The context vector \\(c_t\\) is now a <em>dynamic</em> weighted average of all encoder hidden states. This has several critical advantages over the fixed context:</p>
<ul>
<li><strong>No bottleneck:</strong> The decoder has access to all \\(T\\) encoder states, not just the final one. The effective information available is \\(T \\times d_h\\) numbers, not just \\(d_h\\).</li>
<li><strong>Gradient highways:</strong> The attention mechanism creates direct connections from each encoder state to the decoder, providing shorter gradient paths during backpropagation.</li>
<li><strong>Interpretability:</strong> The attention weights \\(\\alpha_{t,i}\\) form an alignment matrix that reveals which source words contribute to each target word.</li>
</ul>

<div class="env-block definition">
<div class="env-title">Definition 14.4 &mdash; Attention Alignment Matrix</div>
<div class="env-body">
<p>The <strong>alignment matrix</strong> \\(A \\in \\mathbb{R}^{T' \\times T}\\) has entry \\(A_{t,i} = \\alpha_{t,i}\\). Each row sums to 1 (since attention weights are a probability distribution). This matrix provides a soft, differentiable analog of the hard word-to-word alignments used in classical statistical machine translation.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="bahdanau-heatmap-viz"></div>

<p>The visualization above shows a simulated attention alignment matrix for English-to-French translation. Note how the attention concentrates on relevant source words for each target word, with the pattern deviating from the diagonal due to word reordering between languages.</p>

<div class="env-block remark">
<div class="env-title">Parameter Count</div>
<div class="env-body">
<p>The Bahdanau attention mechanism adds relatively few parameters: \\(W_a\\) contributes \\(d_a \\times d_s\\), \\(U_a\\) contributes \\(d_a \\times d_h\\), and \\(v_a\\) contributes \\(d_a\\). For typical values (\\(d_a = d_s = d_h = 512\\)), this is about 0.5M parameters, a small overhead compared to the encoder and decoder RNNs. The computational cost per decoding step is \\(O(T \\cdot d_a)\\) for computing all alignment scores, which is linear in the source length.</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 14.3 &mdash; Computing Attention Step by Step</div>
<div class="env-body">
<p>Let \\(d_a = 2\\), \\(d_s = d_h = 2\\), and suppose we have 3 encoder states. Let:</p>
\\[W_a = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}, \\quad U_a = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}, \\quad v_a = \\begin{pmatrix} 1 \\\\ 1 \\end{pmatrix}\\]
\\[s_0 = \\begin{pmatrix} 0.5 \\\\ 0.3 \\end{pmatrix}, \\quad h_1 = \\begin{pmatrix} 0.1 \\\\ 0.9 \\end{pmatrix}, \\quad h_2 = \\begin{pmatrix} 0.8 \\\\ 0.2 \\end{pmatrix}, \\quad h_3 = \\begin{pmatrix} 0.4 \\\\ 0.6 \\end{pmatrix}\\]
<p>Compute \\(e_{1,1} = v_a^\\top \\tanh(W_a s_0 + U_a h_1) = (1, 1) \\cdot \\tanh\\begin{pmatrix} 0.6 \\\\ 1.2 \\end{pmatrix} = \\tanh(0.6) + \\tanh(1.2) \\approx 0.537 + 0.834 = 1.371\\).</p>
<p>Similarly, \\(e_{1,2} \\approx \\tanh(1.3) + \\tanh(0.5) \\approx 0.862 + 0.462 = 1.324\\) and \\(e_{1,3} \\approx \\tanh(0.9) + \\tanh(0.9) \\approx 1.428\\).</p>
<p>After softmax: \\(\\alpha_{1,1} \\approx 0.335\\), \\(\\alpha_{1,2} \\approx 0.319\\), \\(\\alpha_{1,3} \\approx 0.346\\). The context vector is \\(c_1 = 0.335 \\cdot h_1 + 0.319 \\cdot h_2 + 0.346 \\cdot h_3 \\approx (0.427,\\, 0.573)\\).</p>
</div>
</div>
`,
            visualizations: [
                {
                    id: 'bahdanau-heatmap-viz',
                    title: 'Attention Alignment Heatmap',
                    description: 'Attention weights between source (English) and target (French) words. Brighter cells indicate higher attention weight. Click "Randomize" to generate new alignment patterns.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 650, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var srcWords = ['The', 'black', 'cat', 'sat', 'on', 'the', 'mat'];
                        var tgtWords = ['Le', 'chat', 'noir', 'est', 'assis', 'sur', 'le', 'tapis'];

                        // Precomputed "ideal" attention pattern (non-monotonic due to French word order)
                        var patterns = [
                            // Pattern 0: structured alignment
                            [
                                [0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  // Le -> The
                                [0.05, 0.10, 0.75, 0.02, 0.02, 0.03, 0.03],  // chat -> cat
                                [0.03, 0.78, 0.07, 0.03, 0.03, 0.03, 0.03],  // noir -> black
                                [0.05, 0.05, 0.05, 0.60, 0.10, 0.05, 0.10],  // est -> sat
                                [0.03, 0.03, 0.07, 0.65, 0.07, 0.05, 0.10],  // assis -> sat
                                [0.03, 0.03, 0.03, 0.05, 0.72, 0.07, 0.07],  // sur -> on
                                [0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.10],  // le -> the
                                [0.03, 0.03, 0.03, 0.03, 0.05, 0.08, 0.75]   // tapis -> mat
                            ],
                            // Pattern 1: more diffuse
                            [
                                [0.55, 0.10, 0.08, 0.07, 0.07, 0.07, 0.06],
                                [0.08, 0.15, 0.52, 0.08, 0.07, 0.05, 0.05],
                                [0.06, 0.55, 0.12, 0.07, 0.07, 0.06, 0.07],
                                [0.07, 0.08, 0.10, 0.45, 0.12, 0.08, 0.10],
                                [0.05, 0.05, 0.10, 0.50, 0.10, 0.10, 0.10],
                                [0.05, 0.05, 0.05, 0.08, 0.55, 0.12, 0.10],
                                [0.08, 0.06, 0.06, 0.06, 0.08, 0.50, 0.16],
                                [0.05, 0.05, 0.05, 0.05, 0.08, 0.12, 0.60]
                            ]
                        ];

                        var currentPattern = 0;
                        var attn = patterns[0];

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var cellW = 55, cellH = 36;
                            var marginLeft = 90, marginTop = 70;
                            var T = srcWords.length;
                            var Tp = tgtWords.length;

                            // Title
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Source (English)', marginLeft + T * cellW / 2, 18);

                            // Source words on top
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            for (var i = 0; i < T; i++) {
                                ctx.fillStyle = viz.colors.blue;
                                ctx.fillText(srcWords[i], marginLeft + i * cellW + cellW / 2, marginTop - 8);
                            }

                            // Target words on left
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var t = 0; t < Tp; t++) {
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText(tgtWords[t], marginLeft - 12, marginTop + t * cellH + cellH / 2);
                            }

                            // "Target (French)" label vertically
                            ctx.save();
                            ctx.translate(16, marginTop + Tp * cellH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Target (French)', 0, 0);
                            ctx.restore();

                            // Heatmap cells
                            for (var t = 0; t < Tp; t++) {
                                for (var i = 0; i < T; i++) {
                                    var w = attn[t][i];
                                    var cx = marginLeft + i * cellW;
                                    var cy = marginTop + t * cellH;

                                    // Color: interpolate from dark blue to bright blue/white
                                    var intensity = Math.min(1, w * 1.3);
                                    var r = Math.round(12 + intensity * 76);
                                    var g = Math.round(12 + intensity * 154);
                                    var b = Math.round(32 + intensity * 223);
                                    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                                    ctx.fillRect(cx + 1, cy + 1, cellW - 2, cellH - 2);

                                    // Border
                                    ctx.strokeStyle = '#1a1a40';
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(cx, cy, cellW, cellH);

                                    // Weight text
                                    ctx.fillStyle = intensity > 0.5 ? viz.colors.white : viz.colors.text;
                                    ctx.font = '10px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(w.toFixed(2), cx + cellW / 2, cy + cellH / 2);
                                }

                                // Row sum indicator
                                var rowSum = 0;
                                for (var i = 0; i < T; i++) rowSum += attn[t][i];
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px monospace';
                                ctx.textAlign = 'left';
                                ctx.fillText('\u03A3=' + rowSum.toFixed(2), marginLeft + T * cellW + 6, marginTop + t * cellH + cellH / 2);
                            }

                            // Legend
                            var legX = marginLeft, legY = marginTop + Tp * cellH + 20;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Weight:', legX, legY);
                            var legW = 200, legH = 12;
                            var grad = ctx.createLinearGradient(legX + 50, 0, legX + 50 + legW, 0);
                            grad.addColorStop(0, 'rgb(12,12,32)');
                            grad.addColorStop(1, 'rgb(88,166,255)');
                            ctx.fillStyle = grad;
                            ctx.fillRect(legX + 50, legY - 6, legW, legH);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '9px monospace';
                            ctx.textAlign = 'left';
                            ctx.fillText('0.0', legX + 50, legY + 16);
                            ctx.textAlign = 'right';
                            ctx.fillText('1.0', legX + 50 + legW, legY + 16);

                            // Note about non-monotonic alignment
                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Note: "chat" (row 2) attends to "cat" (col 3), "noir" (row 3) attends to "black" (col 2)', legX, legY + 36);
                            ctx.fillText('The alignment is non-monotonic because French adjective order differs from English.', legX, legY + 52);
                        }

                        draw();

                        VizEngine.createButton(controls, 'Randomize', function() {
                            // Generate a new noisy pattern based on structured alignment
                            var base = patterns[0];
                            attn = [];
                            for (var t = 0; t < tgtWords.length; t++) {
                                attn[t] = [];
                                var sum = 0;
                                for (var i = 0; i < srcWords.length; i++) {
                                    var noise = (Math.random() - 0.5) * 0.2;
                                    attn[t][i] = Math.max(0.01, base[t][i] + noise);
                                    sum += attn[t][i];
                                }
                                // Normalize to sum to 1
                                for (var i = 0; i < srcWords.length; i++) {
                                    attn[t][i] /= sum;
                                }
                            }
                            draw();
                        });

                        VizEngine.createButton(controls, 'Reset', function() {
                            attn = patterns[0];
                            draw();
                        });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'In Bahdanau attention, the alignment score is \\(e_{t,i} = v_a^\\top \\tanh(W_a s_{t-1} + U_a h_i)\\). How many scalar multiplications are needed to compute all \\(T\\) alignment scores for a single decoding step? Express your answer in terms of \\(T\\), \\(d_a\\), \\(d_s\\), and \\(d_h\\).',
                    hint: 'Break it down: \\(W_a s_{t-1}\\) is a matrix-vector product, \\(U_a h_i\\) is another, \\(\\tanh\\) is elementwise, and \\(v_a^\\top (\\cdot)\\) is a dot product.',
                    solution: '<p>Computing \\(W_a s_{t-1}\\): this is a \\(d_a \\times d_s\\) matrix times a \\(d_s\\)-vector, costing \\(d_a \\cdot d_s\\) multiplications. This is done once per decoding step.</p><p>Computing \\(U_a h_i\\): costs \\(d_a \\cdot d_h\\) multiplications per source position, so \\(T \\cdot d_a \\cdot d_h\\) total (though in practice \\(U_a h_i\\) can be precomputed for all \\(i\\)).</p><p>The addition and \\(\\tanh\\) are \\(O(T \\cdot d_a)\\) (no multiplications). The dot product \\(v_a^\\top (\\cdot)\\) costs \\(d_a\\) multiplications per position, so \\(T \\cdot d_a\\) total.</p><p>Grand total: \\(d_a \\cdot d_s + T \\cdot d_a \\cdot d_h + T \\cdot d_a = d_a(d_s + T(d_h + 1))\\). Since \\(U_a h_i\\) can be precomputed, the per-step cost is \\(d_a(d_s + T)\\).</p>'
                },
                {
                    question: 'Prove that the context vector \\(c_t = \\sum_{i=1}^T \\alpha_{t,i} h_i\\) lies in the convex hull of the encoder hidden states \\(\\{h_1, \\ldots, h_T\\}\\).',
                    hint: 'What are the two defining properties of a convex combination?',
                    solution: 'A point lies in the convex hull of a set of points if and only if it can be written as a convex combination: \\(c = \\sum_i \\lambda_i h_i\\) where \\(\\lambda_i \\geq 0\\) and \\(\\sum_i \\lambda_i = 1\\). The attention weights \\(\\alpha_{t,i}\\) are the output of a softmax, so: (1) \\(\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_j \\exp(e_{t,j})} &gt; 0\\) for all \\(i\\), and (2) \\(\\sum_{i=1}^T \\alpha_{t,i} = 1\\) by construction. Therefore \\(c_t\\) is a convex combination of \\(h_1, \\ldots, h_T\\) and lies in their convex hull. \\(\\square\\)'
                },
                {
                    question: 'The Bahdanau attention mechanism uses the <em>previous</em> decoder state \\(s_{t-1}\\) to compute attention for step \\(t\\). Why not use the <em>current</em> state \\(s_t\\)? What would be the computational difficulty?',
                    hint: 'Think about the dependency graph: to compute \\(s_t\\), you need \\(c_t\\), but to compute \\(c_t\\), you would need \\(s_t\\).',
                    solution: 'Using \\(s_t\\) to compute \\(c_t\\) creates a circular dependency: \\(s_t = f(s_{t-1}, y_{t-1}, c_t)\\) requires \\(c_t\\), and \\(c_t = \\sum_i \\alpha_{t,i} h_i\\) with \\(\\alpha_{t,i} \\propto \\exp(v_a^\\top \\tanh(W_a s_t + U_a h_i))\\) requires \\(s_t\\). This is an implicit equation that would require iterative solving (e.g., fixed-point iteration), making training significantly harder. Using \\(s_{t-1}\\) breaks this cycle, yielding a simple feedforward computation. Luong attention (next section) addresses this differently by computing attention after the decoder RNN step.'
                }
            ]
        },

        // ======================== Section 3 ========================
        {
            id: 'luong-attention',
            title: 'Luong Attention',
            content: `
<h2>Luong (Multiplicative) Attention</h2>

<div class="env-block intuition">
<div class="env-title">A Simpler, Faster Alternative</div>
<div class="env-body">
<p>Shortly after Bahdanau et al., Luong et al. (2015) proposed several alternative attention mechanisms in "Effective Approaches to Attention-based Neural Machine Translation." The key insight: the alignment score can be computed more efficiently using <em>dot products</em> rather than a feedforward network, and the attention can be applied <em>after</em> the decoder RNN step instead of before.</p>
</div>
</div>

<h3>Score Functions</h3>

<p>Luong et al. proposed three scoring functions to compute the alignment between the current decoder state \\(s_t\\) (note: current, not previous) and each encoder state \\(h_i\\):</p>

<div class="env-block definition">
<div class="env-title">Definition 14.5 &mdash; Luong Score Functions</div>
<div class="env-body">
<p>Given decoder hidden state \\(s_t \\in \\mathbb{R}^{d_s}\\) and encoder hidden state \\(h_i \\in \\mathbb{R}^{d_h}\\):</p>
<p><strong>Dot product</strong> (requires \\(d_s = d_h\\)):</p>
\\[\\text{score}(s_t, h_i) = s_t^\\top h_i\\]
<p><strong>General</strong> (allows \\(d_s \\neq d_h\\)):</p>
\\[\\text{score}(s_t, h_i) = s_t^\\top W_a \\, h_i\\]
<p>where \\(W_a \\in \\mathbb{R}^{d_s \\times d_h}\\) is a learnable weight matrix.</p>
<p><strong>Concat</strong> (similar to Bahdanau):</p>
\\[\\text{score}(s_t, h_i) = v_a^\\top \\tanh(W_a [s_t;\\, h_i])\\]
<p>where \\([s_t;\\, h_i]\\) denotes concatenation.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Dot Product vs. Additive</div>
<div class="env-body">
<p>The dot product score \\(s_t^\\top h_i\\) is computationally attractive because it requires no learnable parameters and can be computed as a single matrix multiplication across all source positions simultaneously: \\(e_t = H^\\top s_t\\) where \\(H = [h_1, \\ldots, h_T]\\). This is significantly faster than the additive score, especially on GPUs where matrix operations are highly optimized.</p>
</div>
</div>

<h3>The Luong Attention Pipeline</h3>

<p>A key architectural difference from Bahdanau: Luong attention computes the context vector <em>after</em> the decoder RNN step, not before. This avoids the circular dependency entirely.</p>

<div class="env-block definition">
<div class="env-title">Definition 14.6 &mdash; Luong Attention Mechanism</div>
<div class="env-body">
<p>The Luong attention pipeline at decoder step \\(t\\):</p>
<p><strong>Step 1.</strong> Compute decoder hidden state: \\(s_t = \\text{RNN}(s_{t-1}, y_{t-1})\\)</p>
<p><strong>Step 2.</strong> Compute alignment scores: \\(e_{t,i} = \\text{score}(s_t, h_i)\\) for all \\(i\\)</p>
<p><strong>Step 3.</strong> Compute attention weights: \\(\\alpha_{t,i} = \\text{softmax}(e_t)_i\\)</p>
<p><strong>Step 4.</strong> Compute context vector: \\(c_t = \\sum_{i=1}^T \\alpha_{t,i} h_i\\)</p>
<p><strong>Step 5.</strong> Compute attentional hidden state: \\(\\tilde{s}_t = \\tanh(W_c [c_t;\\, s_t])\\)</p>
<p><strong>Step 6.</strong> Predict: \\(P(y_t \\mid y_{&lt;t}, x) = \\text{softmax}(W_o \\tilde{s}_t)\\)</p>
</div>
</div>

<p>The attentional hidden state \\(\\tilde{s}_t\\) combines the context (what the attention selected from the source) with the decoder state (what the decoder has generated so far). The \\(\\tanh\\) nonlinearity allows the model to learn how to merge these two information streams.</p>

<div class="viz-placeholder" data-viz="luong-attention-flow-viz"></div>

<h3>Global vs. Local Attention</h3>

<p>Luong et al. also introduced <strong>local attention</strong>, which restricts the attention to a window of source positions around a predicted alignment point, rather than attending to all source positions (global attention).</p>

<div class="env-block definition">
<div class="env-title">Definition 14.7 &mdash; Local Attention</div>
<div class="env-body">
<p>In <strong>local attention</strong>, the model first predicts an alignment position \\(p_t\\) for each decoder step \\(t\\):</p>
\\[p_t = T \\cdot \\sigma(v_p^\\top \\tanh(W_p \\, s_t))\\]
<p>where \\(\\sigma\\) is the sigmoid function. Then attention weights are computed only within a window \\([p_t - D, p_t + D]\\), modulated by a Gaussian centered at \\(p_t\\):</p>
\\[\\alpha_{t,i} = \\text{softmax}(e_{t,i}) \\cdot \\exp\\!\\left(-\\frac{(i - p_t)^2}{2(D/2)^2}\\right)\\]
<p>This reduces the computational cost from \\(O(T)\\) to \\(O(D)\\) per decoding step.</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Comparing Bahdanau and Luong</div>
<div class="env-body">
<table style="width:100%;font-size:0.9rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #333;"><th style="text-align:left;padding:6px;">Aspect</th><th style="text-align:left;padding:6px;">Bahdanau</th><th style="text-align:left;padding:6px;">Luong</th></tr>
<tr><td style="padding:6px;">Score function</td><td style="padding:6px;">Additive (feedforward net)</td><td style="padding:6px;">Dot / General / Concat</td></tr>
<tr><td style="padding:6px;">Decoder state used</td><td style="padding:6px;">Previous \\(s_{t-1}\\)</td><td style="padding:6px;">Current \\(s_t\\)</td></tr>
<tr><td style="padding:6px;">When attention is applied</td><td style="padding:6px;">Before RNN step</td><td style="padding:6px;">After RNN step</td></tr>
<tr><td style="padding:6px;">Encoder</td><td style="padding:6px;">Bidirectional</td><td style="padding:6px;">Unidirectional (top layer)</td></tr>
<tr><td style="padding:6px;">Scope</td><td style="padding:6px;">Global only</td><td style="padding:6px;">Global and local</td></tr>
</table>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 14.4 &mdash; Dot Product Attention Computation</div>
<div class="env-body">
<p>Let \\(d = 3\\), \\(s_t = (1, 0, -1)^\\top\\), and encoder states:</p>
\\[h_1 = (2, 1, 0)^\\top, \\quad h_2 = (0, 0, 3)^\\top, \\quad h_3 = (1, 1, -1)^\\top\\]
<p>Dot product scores: \\(e_1 = s_t^\\top h_1 = 2\\), \\(e_2 = s_t^\\top h_2 = -3\\), \\(e_3 = s_t^\\top h_3 = 2\\).</p>
<p>Softmax: \\(\\alpha_1 = \\frac{e^2}{e^2 + e^{-3} + e^2} \\approx \\frac{7.389}{14.828} \\approx 0.498\\), \\(\\alpha_2 \\approx \\frac{0.050}{14.828} \\approx 0.003\\), \\(\\alpha_3 \\approx 0.498\\).</p>
<p>Context: \\(c_t \\approx 0.498(2,1,0)^\\top + 0.003(0,0,3)^\\top + 0.498(1,1,-1)^\\top \\approx (1.494,\\, 0.997,\\, -0.489)^\\top\\).</p>
<p>Notice how \\(h_2\\), which points in the opposite direction from \\(s_t\\), gets nearly zero weight. The dot product score naturally measures alignment in the geometric sense.</p>
</div>
</div>
`,
            visualizations: [
                {
                    id: 'luong-attention-flow-viz',
                    title: 'Dot-Product Attention Flow',
                    description: 'Animated computation flow: Query (decoder state) computes dot products with all Keys (encoder states), applies softmax, then weights the Values. Adjust the query vector to see how attention shifts.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 450, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var keys = [
                            [0.8, 0.3, -0.2],
                            [-0.1, 0.9, 0.4],
                            [0.5, -0.5, 0.7],
                            [0.2, 0.7, -0.6]
                        ];
                        var values = [
                            [1.0, 0.2],
                            [0.3, 0.8],
                            [0.7, 0.5],
                            [0.1, 0.9]
                        ];
                        var query = { v: [0.6, 0.4, -0.3] };
                        var animPhase = { value: 0 };
                        var animSpeed = 0.012;

                        function softmax(arr) {
                            var max = Math.max.apply(null, arr);
                            var exps = arr.map(function(v) { return Math.exp(v - max); });
                            var sum = exps.reduce(function(a, b) { return a + b; }, 0);
                            return exps.map(function(v) { return v / sum; });
                        }

                        function dot3(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

                        function draw(t) {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var phase = animPhase.value;
                            // Smooth phase for animation
                            var p = (Math.sin(t * animSpeed) + 1) / 2; // 0 to 1 oscillation

                            // Layout positions
                            var qX = 80, qY = 200;
                            var kX = 280, kStartY = 80;
                            var kSpacing = 80;
                            var scoreX = 420;
                            var smX = 520;
                            var vX = 280;
                            var outX = 620, outY = 200;

                            var T = keys.length;

                            // Compute scores
                            var scores = [];
                            for (var i = 0; i < T; i++) {
                                scores.push(dot3(query.v, keys[i]));
                            }
                            var weights = softmax(scores);

                            // Compute output
                            var output = [0, 0];
                            for (var i = 0; i < T; i++) {
                                output[0] += weights[i] * values[i][0];
                                output[1] += weights[i] * values[i][1];
                            }

                            // === Draw Query ===
                            ctx.fillStyle = viz.colors.orange;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Query (s_t)', qX, qY - 45);

                            ctx.fillStyle = viz.colors.orange + '33';
                            ctx.fillRect(qX - 40, qY - 30, 80, 60);
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(qX - 40, qY - 30, 80, 60);
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '11px monospace';
                            for (var d = 0; d < 3; d++) {
                                ctx.fillText(query.v[d].toFixed(1), qX, qY - 12 + d * 16);
                            }

                            // === Draw Keys ===
                            ctx.fillStyle = viz.colors.blue;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Keys (h_i)', kX, kStartY - 35);

                            for (var i = 0; i < T; i++) {
                                var ky = kStartY + i * kSpacing;
                                ctx.fillStyle = viz.colors.blue + '33';
                                ctx.fillRect(kX - 35, ky - 20, 70, 40);
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 1.5;
                                ctx.strokeRect(kX - 35, ky - 20, 70, 40);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '10px monospace';
                                var kStr = '[' + keys[i].map(function(v) { return v.toFixed(1); }).join(', ') + ']';
                                ctx.fillText(kStr, kX, ky);

                                // Arrow from query to key (dot product)
                                var arrowAlpha = Math.min(1, Math.max(0.2, weights[i] * 2));
                                ctx.strokeStyle = viz.colors.yellow + Math.round(arrowAlpha * 255).toString(16).padStart(2, '0');
                                ctx.lineWidth = 1 + weights[i] * 3;
                                ctx.beginPath();
                                ctx.moveTo(qX + 40, qY);
                                ctx.lineTo(kX - 35, ky);
                                ctx.stroke();

                                // === Score ===
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '11px monospace';
                                ctx.textAlign = 'center';
                                ctx.fillText(scores[i].toFixed(2), scoreX, ky);

                                // Arrow from score to softmax
                                ctx.strokeStyle = viz.colors.text + '66';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(scoreX + 25, ky);
                                ctx.lineTo(smX - 25, ky);
                                ctx.stroke();

                                // === Softmax weights ===
                                var barW = weights[i] * 80;
                                ctx.fillStyle = viz.colors.purple + '44';
                                ctx.fillRect(smX - 5, ky - 8, 80, 16);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.fillRect(smX - 5, ky - 8, barW, 16);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '10px monospace';
                                ctx.textAlign = 'left';
                                ctx.fillText(weights[i].toFixed(3), smX + 82, ky);

                                // Arrow from weight to output
                                ctx.strokeStyle = viz.colors.purple + Math.round(arrowAlpha * 200).toString(16).padStart(2, '0');
                                ctx.lineWidth = 1 + weights[i] * 4;
                                ctx.beginPath();
                                ctx.moveTo(smX + 75, ky);
                                ctx.lineTo(outX - 40, outY);
                                ctx.stroke();
                            }

                            // Score label
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Scores', scoreX, kStartY - 35);
                            ctx.fillText('(q\u00B7k)', scoreX, kStartY - 22);

                            // Softmax label
                            ctx.fillStyle = viz.colors.purple;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Softmax', smX + 35, kStartY - 35);
                            ctx.fillText('Weights \u03B1', smX + 35, kStartY - 22);

                            // === Output (context vector) ===
                            ctx.fillStyle = viz.colors.green;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Context c_t', outX, outY - 45);
                            ctx.fillStyle = viz.colors.green + '33';
                            ctx.fillRect(outX - 40, outY - 25, 80, 50);
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(outX - 40, outY - 25, 80, 50);
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '11px monospace';
                            ctx.fillText('[' + output[0].toFixed(3) + ',', outX, outY - 5);
                            ctx.fillText(' ' + output[1].toFixed(3) + ']', outX, outY + 12);

                            // Formula at bottom
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('c_t = \u03A3 \u03B1_i \u00B7 v_i   |   \u03B1_i = softmax(q \u00B7 k_i)', viz.width / 2, viz.height - 20);

                            // Highlight the max weight
                            var maxIdx = 0;
                            for (var i = 1; i < T; i++) {
                                if (weights[i] > weights[maxIdx]) maxIdx = i;
                            }
                            ctx.fillStyle = viz.colors.green;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Max attention on key ' + (maxIdx + 1) + ' (\u03B1 = ' + weights[maxIdx].toFixed(3) + ')', 20, viz.height - 40);
                        }

                        draw(0);

                        // Query adjustment sliders
                        var labels = ['q[0]', 'q[1]', 'q[2]'];
                        for (var d = 0; d < 3; d++) {
                            (function(dim) {
                                VizEngine.createSlider(controls, labels[dim], -1.0, 1.0, query.v[dim], 0.1, function(v) {
                                    query.v[dim] = v;
                                    draw(0);
                                });
                            })(d);
                        }

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'For the dot-product score \\(\\text{score}(s_t, h_i) = s_t^\\top h_i\\), show that the score equals \\(\\|s_t\\| \\|h_i\\| \\cos \\theta\\), where \\(\\theta\\) is the angle between the two vectors. What does this imply about what "high attention" means geometrically?',
                    hint: 'Use the definition of the dot product in terms of vector norms and the cosine of the included angle.',
                    solution: 'By the definition of the dot product, \\(s_t^\\top h_i = \\|s_t\\| \\|h_i\\| \\cos\\theta\\). This means the score is high when: (1) the vectors are large in magnitude, and (2) they point in similar directions (\\(\\cos\\theta \\approx 1\\)). High attention therefore means the decoder state and encoder state are "aligned" in representation space. Encoder states pointing in the opposite direction get negative scores, which the softmax maps to near-zero weights. This geometric interpretation explains why the dot product is a natural similarity measure.'
                },
                {
                    question: 'The dot-product score can become very large in magnitude when the dimensionality \\(d\\) is large. Vaswani et al. (2017) proposed <strong>scaled dot-product attention</strong> with score \\(s_t^\\top h_i / \\sqrt{d}\\). Why is this scaling necessary? What goes wrong without it?',
                    hint: 'Consider what happens to the softmax function when its inputs have large variance. If the elements of \\(s_t\\) and \\(h_i\\) are i.i.d. with zero mean and unit variance, what is the variance of their dot product?',
                    solution: '<p>If the components of \\(s_t\\) and \\(h_i\\) are i.i.d. with mean 0 and variance 1, then each product \\(s_t^{(k)} h_i^{(k)}\\) has mean 0 and variance 1. The dot product sums \\(d\\) such terms, so \\(\\text{Var}(s_t^\\top h_i) = d\\). For large \\(d\\) (e.g., 512), the dot products have standard deviation \\(\\sqrt{512} \\approx 22.6\\), producing scores in a wide range.</p><p>Large-magnitude softmax inputs push the softmax into saturated regions where its output is nearly one-hot (one entry close to 1, all others close to 0). In these regions, the gradients of the softmax are extremely small (near zero), causing vanishing gradients. Dividing by \\(\\sqrt{d}\\) normalizes the variance back to 1, keeping the softmax in a well-behaved regime.</p>'
                },
                {
                    question: 'Compare the parameter counts and computational costs of the three Luong score functions (dot, general, concat) for \\(d_s = d_h = d\\) and attention dimension \\(d_a = d\\).',
                    hint: 'Count learnable parameters and multiply-add operations per source position.',
                    solution: '<p><strong>Dot:</strong> 0 parameters. Cost: \\(d\\) multiplications per source position (one dot product). Total per step: \\(Td\\).</p><p><strong>General:</strong> \\(d^2\\) parameters (matrix \\(W_a\\)). Cost: \\(d^2\\) for \\(W_a h_i\\) plus \\(d\\) for the dot product, so \\(d^2 + d\\) per position. Total per step: \\(T(d^2 + d)\\). But \\(W_a H\\) can be precomputed in \\(O(Td^2)\\).</p><p><strong>Concat:</strong> \\(d \\cdot 2d + d = 2d^2 + d\\) parameters (\\(W_a \\in \\mathbb{R}^{d \\times 2d}\\), \\(v_a \\in \\mathbb{R}^d\\)). Cost: \\(2d^2\\) for the matrix-vector product, \\(d\\) for tanh, \\(d\\) for the dot with \\(v_a\\), totaling \\(2d^2 + 2d\\) per position.</p><p>The dot product is fastest with no parameters; general offers a good tradeoff; concat is most expressive but slowest.</p>'
                },
                {
                    question: 'In local attention, the Gaussian window modulates the softmax weights. If \\(D = 5\\) and the predicted position is \\(p_t = 10\\), what is the Gaussian multiplier for source position \\(i = 13\\)? How does this compare to position \\(i = 10\\)?',
                    hint: 'The Gaussian is \\(\\exp(-(i - p_t)^2 / (2\\sigma^2))\\) where \\(\\sigma = D/2\\).',
                    solution: 'With \\(D = 5\\) and \\(\\sigma = D/2 = 2.5\\): For \\(i = 13\\): \\(\\exp(-(13-10)^2 / (2 \\cdot 2.5^2)) = \\exp(-9/12.5) = \\exp(-0.72) \\approx 0.487\\). For \\(i = 10\\): \\(\\exp(-(10-10)^2 / (2 \\cdot 2.5^2)) = \\exp(0) = 1.0\\). So position 13 gets about half the weight of the center position, before softmax normalization. This means the Gaussian effectively down-weights positions far from the predicted alignment point, focusing computational resources on the most relevant region.'
                }
            ]
        },

        // ======================== Section 4 ========================
        {
            id: 'attention-interpretability',
            title: 'Attention Interpretability',
            content: `
<h2>Attention Interpretability</h2>

<div class="env-block intuition">
<div class="env-title">What Do Attention Weights Tell Us?</div>
<div class="env-body">
<p>One of the most appealing properties of attention mechanisms is their seeming interpretability. The attention weights \\(\\alpha_{t,i}\\) form a probability distribution over source positions, and visualizing these weights as heatmaps reveals intuitive patterns: when translating "cat" to "chat," the model attends heavily to the source word "cat." This has led to widespread use of attention weights as explanations for model behavior. But how reliable is this interpretation?</p>
</div>
</div>

<h3>Attention as Soft Alignment</h3>

<p>In machine translation, attention weights often closely resemble classical word alignments. For languages with similar word order (e.g., English-Spanish), the alignment matrix is roughly diagonal. For languages with different word order (e.g., English-Japanese), the pattern is more complex but still interpretable.</p>

<div class="env-block example">
<div class="env-title">Example 14.5 &mdash; Alignment Patterns</div>
<div class="env-body">
<p>Consider three translation scenarios:</p>
<ul>
<li><strong>Monotonic (EN-ES):</strong> "The cat eats fish" to "El gato come pescado." Alignment is nearly diagonal.</li>
<li><strong>Reordering (EN-DE):</strong> "I have eaten" to "Ich habe gegessen." The past participle "eaten" aligns to "gegessen" at the end, roughly preserving order.</li>
<li><strong>Long-range (EN-JA):</strong> In English-Japanese translation, the verb appears at the end in Japanese, creating a pattern where the last target tokens attend to early/middle source tokens.</li>
</ul>
</div>
</div>

<h3>Beyond Translation: Attention in Other Tasks</h3>

<p>Attention weights have been used to interpret models in many settings:</p>
<ul>
<li><strong>Sentiment analysis:</strong> Attention over input words reveals which words contribute most to the sentiment prediction. High weights on "excellent" and "terrible" match human intuition.</li>
<li><strong>Question answering:</strong> When the model attends to the passage, high-weight tokens often coincide with the answer span.</li>
<li><strong>Summarization:</strong> Attention patterns show which source sentences contribute to each summary sentence.</li>
</ul>

<div class="viz-placeholder" data-viz="attention-interactive-viz"></div>

<h3>Limitations and Caveats</h3>

<p>Despite their intuitive appeal, attention weights have significant limitations as explanations.</p>

<div class="env-block warning">
<div class="env-title">Attention Is Not Explanation</div>
<div class="env-body">
<p>Jain and Wallace (2019) demonstrated several important caveats:</p>
<ul>
<li><strong>Alternative weights, same prediction:</strong> There often exist very different attention distributions that produce nearly identical model outputs. If multiple explanations are equally valid, none is uniquely "the" explanation.</li>
<li><strong>Gradient disagreement:</strong> Attention weights frequently disagree with gradient-based feature importance measures. A word with high attention weight may have low gradient magnitude, meaning that changing it would barely affect the output.</li>
<li><strong>Uniform attention baselines:</strong> For some tasks, replacing learned attention with uniform attention (\\(\\alpha_i = 1/T\\)) barely affects performance, suggesting the model does not rely on selective attention.</li>
</ul>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 14.8 &mdash; Attention Entropy</div>
<div class="env-body">
<p>The <strong>attention entropy</strong> at decoder step \\(t\\) is:</p>
\\[H(\\alpha_t) = -\\sum_{i=1}^{T} \\alpha_{t,i} \\log \\alpha_{t,i}\\]
<p>Low entropy (close to 0) means the attention is sharply focused on one or a few source positions. High entropy (close to \\(\\log T\\)) means the attention is nearly uniform. Sharp attention is more interpretable, as it clearly indicates which source positions matter.</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 14.6 &mdash; Entropy of Attention Distributions</div>
<div class="env-body">
<p>For \\(T = 4\\) source positions:</p>
<ul>
<li><strong>Sharp:</strong> \\(\\alpha = (0.9, 0.05, 0.03, 0.02)\\). Entropy \\(H \\approx 0.47\\) nats.</li>
<li><strong>Diffuse:</strong> \\(\\alpha = (0.3, 0.3, 0.2, 0.2)\\). Entropy \\(H \\approx 1.36\\) nats.</li>
<li><strong>Uniform:</strong> \\(\\alpha = (0.25, 0.25, 0.25, 0.25)\\). Entropy \\(H = \\log 4 \\approx 1.39\\) nats (maximum).</li>
</ul>
<p>The sharp distribution clearly identifies one source position as most important. The diffuse distribution is harder to interpret, and the uniform distribution provides no information about relative importance.</p>
</div>
</div>

<h3>Faithful Interpretability</h3>

<p>To use attention weights responsibly, consider the following guidelines:</p>

<div class="env-block remark">
<div class="env-title">Best Practices for Interpreting Attention</div>
<div class="env-body">
<ol>
<li><strong>Cross-validate with gradients:</strong> Compare attention weights with gradient-based attribution (e.g., integrated gradients). High agreement increases confidence in the attention-based explanation.</li>
<li><strong>Check entropy:</strong> Only interpret attention when it is sharply peaked. Diffuse attention provides weak evidence.</li>
<li><strong>Perturbation tests:</strong> Verify that masking or modifying high-attention tokens actually changes the output. If it does not, the attention weight is misleading.</li>
<li><strong>Use attention rollout:</strong> In multi-layer models, attention at one layer does not account for information mixing at other layers. Attention rollout (Abnar and Zuidema, 2020) propagates attention through layers.</li>
</ol>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">The Deeper Lesson</div>
<div class="env-body">
<p>Attention was invented as a <em>performance</em> mechanism, not an interpretability tool. The fact that attention weights are sometimes interpretable is a happy coincidence, not a guarantee. The true purpose of attention is to provide the decoder with direct access to source information, bypassing the bottleneck. That it also provides a window into the model's behavior is a bonus, one that should be used with appropriate caution.</p>
</div>
</div>

<h3>Looking Ahead: Attention Without Recurrence</h3>

<p>The attention mechanisms we have studied operate alongside RNN encoder-decoders. In 2017, Vaswani et al. asked a revolutionary question: what if we remove the RNN entirely and use <em>only</em> attention? The result, the <strong>Transformer</strong> architecture, is the subject of Chapter 15. The key innovation is <strong>self-attention</strong>, where a sequence attends to itself, enabling parallel computation and capturing long-range dependencies without any recurrence.</p>
`,
            visualizations: [
                {
                    id: 'attention-interactive-viz',
                    title: 'Interactive Attention Explorer',
                    description: 'Click on different target words (left) to see which source words (top) receive the most attention. The bar chart shows the attention distribution for the selected target word.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 700, height: 460, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var srcWords = ['I', 'love', 'deep', 'learning', 'very', 'much'];
                        var tgtWords = ['Positive', 'sentiment', 'detected'];

                        // Attention weights: each target word's distribution over source words
                        var attnData = [
                            [0.05, 0.35, 0.08, 0.12, 0.15, 0.25],  // "Positive" -> "love", "much"
                            [0.03, 0.20, 0.25, 0.40, 0.07, 0.05],  // "sentiment" -> "deep learning"
                            [0.02, 0.55, 0.05, 0.08, 0.18, 0.12]   // "detected" -> "love", "very"
                        ];

                        var selectedTarget = { value: 0 };

                        // Define clickable regions for target words
                        var tgtStartY = 120;
                        var tgtSpacing = 55;
                        var tgtBoxX = 30;
                        var tgtBoxW = 110;
                        var tgtBoxH = 38;

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, viz.width, viz.height);

                            var sel = selectedTarget.value;
                            var T = srcWords.length;
                            var Tp = tgtWords.length;

                            // === Title ===
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Task: Sentiment Classification', viz.width / 2, 22);

                            // === Source words (top) ===
                            var srcStartX = 180;
                            var srcSpacing = 80;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Source words (input sentence)', srcStartX + T * srcSpacing / 2, 48);

                            for (var i = 0; i < T; i++) {
                                var sx = srcStartX + i * srcSpacing;
                                // Highlight based on attention weight
                                var w = attnData[sel][i];
                                var intensity = Math.min(1, w * 2);

                                // Source word box
                                var r = Math.round(63 + intensity * 192);
                                var g = Math.round(185 + intensity * 70);
                                var b = Math.round(160 + intensity * 95);
                                ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (0.15 + intensity * 0.5) + ')';
                                ctx.fillRect(sx - 30, 60, 60, 30);
                                ctx.strokeStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                                ctx.lineWidth = 1 + intensity * 2;
                                ctx.strokeRect(sx - 30, 60, 60, 30);

                                // Word text
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = (intensity > 0.4 ? 'bold ' : '') + '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(srcWords[i], sx, 75);
                            }

                            // === Target words (left) ===
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Target', tgtBoxX + tgtBoxW / 2, tgtStartY - 20);

                            for (var t = 0; t < Tp; t++) {
                                var ty = tgtStartY + t * tgtSpacing;
                                var isSelected = (t === sel);

                                // Target word box
                                ctx.fillStyle = isSelected ? (viz.colors.orange + '44') : (viz.colors.blue + '22');
                                ctx.fillRect(tgtBoxX, ty, tgtBoxW, tgtBoxH);
                                ctx.strokeStyle = isSelected ? viz.colors.orange : viz.colors.blue;
                                ctx.lineWidth = isSelected ? 2.5 : 1;
                                ctx.strokeRect(tgtBoxX, ty, tgtBoxW, tgtBoxH);

                                ctx.fillStyle = isSelected ? viz.colors.orange : viz.colors.blue;
                                ctx.font = (isSelected ? 'bold ' : '') + '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(tgtWords[t], tgtBoxX + tgtBoxW / 2, ty + tgtBoxH / 2);

                                // Click hint
                                if (!isSelected) {
                                    ctx.fillStyle = viz.colors.text + '88';
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.fillText('click', tgtBoxX + tgtBoxW / 2, ty + tgtBoxH + 8);
                                }

                                // Draw attention lines from selected target to source
                                if (isSelected) {
                                    for (var i = 0; i < T; i++) {
                                        var sx = srcStartX + i * srcSpacing;
                                        var w = attnData[t][i];
                                        var lineAlpha = Math.max(0.1, w);
                                        ctx.strokeStyle = viz.colors.orange;
                                        ctx.globalAlpha = lineAlpha;
                                        ctx.lineWidth = 1 + w * 5;
                                        ctx.beginPath();
                                        ctx.moveTo(tgtBoxX + tgtBoxW, ty + tgtBoxH / 2);
                                        ctx.lineTo(sx, 90);
                                        ctx.stroke();
                                        ctx.globalAlpha = 1.0;
                                    }
                                }
                            }

                            // === Bar chart of attention weights ===
                            var barChartX = 180;
                            var barChartY = 260;
                            var barChartW = viz.width - 220;
                            var barChartH = 140;

                            // Background
                            ctx.fillStyle = '#0f0f28';
                            ctx.fillRect(barChartX - 10, barChartY - 10, barChartW + 60, barChartH + 60);
                            ctx.strokeStyle = viz.colors.text + '44';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(barChartX - 10, barChartY - 10, barChartW + 60, barChartH + 60);

                            // Title
                            ctx.fillStyle = viz.colors.orange;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Attention distribution for "' + tgtWords[sel] + '"', barChartX + barChartW / 2, barChartY - 20);

                            // Grid lines
                            for (var g = 0; g <= 4; g++) {
                                var gv = g * 0.25;
                                var gy = barChartY + barChartH - gv * barChartH;
                                ctx.strokeStyle = viz.colors.text + '22';
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.moveTo(barChartX, gy);
                                ctx.lineTo(barChartX + barChartW, gy);
                                ctx.stroke();
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px monospace';
                                ctx.textAlign = 'right';
                                ctx.fillText(gv.toFixed(2), barChartX - 5, gy + 3);
                            }

                            // Bars
                            var barW = barChartW / T * 0.65;
                            var barGap = barChartW / T;
                            var maxW = Math.max.apply(null, attnData[sel]);

                            for (var i = 0; i < T; i++) {
                                var w = attnData[sel][i];
                                var bx = barChartX + i * barGap + (barGap - barW) / 2;
                                var bh = w * barChartH;
                                var by = barChartY + barChartH - bh;

                                // Bar color: brightest for highest weight
                                var isMax = (Math.abs(w - maxW) < 0.001);
                                if (isMax) {
                                    ctx.fillStyle = viz.colors.orange;
                                } else {
                                    ctx.fillStyle = viz.colors.blue;
                                }
                                ctx.fillRect(bx, by, barW, bh);

                                // Word label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(srcWords[i], bx + barW / 2, barChartY + barChartH + 14);

                                // Weight value on top of bar
                                ctx.fillStyle = isMax ? viz.colors.orange : viz.colors.text;
                                ctx.font = '10px monospace';
                                ctx.fillText(w.toFixed(2), bx + barW / 2, by - 6);
                            }

                            // Y-axis label
                            ctx.save();
                            ctx.translate(barChartX - 35, barChartY + barChartH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Attention weight \u03B1', 0, 0);
                            ctx.restore();

                            // Entropy display
                            var entropy = 0;
                            for (var i = 0; i < T; i++) {
                                var w = attnData[sel][i];
                                if (w > 0) entropy -= w * Math.log(w);
                            }
                            var maxEntropy = Math.log(T);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Entropy: ' + entropy.toFixed(3) + ' / ' + maxEntropy.toFixed(3) + ' nats', barChartX, barChartY + barChartH + 40);

                            var sharpness = 1 - entropy / maxEntropy;
                            ctx.fillStyle = sharpness > 0.5 ? viz.colors.green : viz.colors.yellow;
                            ctx.fillText('Sharpness: ' + (sharpness * 100).toFixed(1) + '%', barChartX + 260, barChartY + barChartH + 40);
                        }

                        draw();

                        // Click handler for target word selection
                        viz.canvas.addEventListener('click', function(e) {
                            var rect = viz.canvas.getBoundingClientRect();
                            var mx = e.clientX - rect.left;
                            var my = e.clientY - rect.top;

                            for (var t = 0; t < tgtWords.length; t++) {
                                var ty = tgtStartY + t * tgtSpacing;
                                if (mx >= tgtBoxX && mx <= tgtBoxX + tgtBoxW && my >= ty && my <= ty + tgtBoxH) {
                                    selectedTarget.value = t;
                                    draw();
                                    break;
                                }
                            }
                        });

                        // Also add buttons for each target word
                        for (var t = 0; t < tgtWords.length; t++) {
                            (function(idx) {
                                VizEngine.createButton(controls, tgtWords[idx], function() {
                                    selectedTarget.value = idx;
                                    draw();
                                });
                            })(t);
                        }

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute the attention entropy for the distribution \\(\\alpha = (0.7, 0.1, 0.1, 0.1)\\) over 4 source positions. Is this distribution "sharp" or "diffuse"? Compare to the maximum possible entropy.',
                    hint: 'Use \\(H = -\\sum_i \\alpha_i \\ln \\alpha_i\\). The maximum entropy for 4 positions is \\(\\ln 4\\).',
                    solution: '<p>\\(H = -(0.7 \\ln 0.7 + 0.1 \\ln 0.1 + 0.1 \\ln 0.1 + 0.1 \\ln 0.1)\\)</p><p>\\(= -(0.7 \\times (-0.357) + 3 \\times 0.1 \\times (-2.303))\\)</p><p>\\(= -(-0.250 + (-0.691)) = -(-0.941) = 0.941\\) nats.</p><p>Maximum entropy: \\(\\ln 4 = 1.386\\) nats. The ratio is \\(0.941/1.386 = 0.679\\), so the distribution uses about 68% of the maximum entropy. This is moderately sharp; the model concentrates on one position (weight 0.7) but maintains non-negligible weights on others. For interpretability, we would prefer even sharper distributions (e.g., \\(\\alpha = (0.95, 0.02, 0.02, 0.01)\\) with \\(H \\approx 0.27\\) nats).</p>'
                },
                {
                    question: 'Jain and Wallace (2019) show that adversarial attention distributions \\(\\tilde{\\alpha} \\neq \\alpha\\) can produce nearly the same output. Formally, if \\(c = \\sum_i \\alpha_i h_i\\) and \\(\\tilde{c} = \\sum_i \\tilde{\\alpha}_i h_i\\), under what geometric condition on \\(\\{h_i\\}\\) is it possible that \\(c \\approx \\tilde{c}\\) despite \\(\\alpha \\neq \\tilde{\\alpha}\\)?',
                    hint: 'Think about what happens when some encoder states are nearly collinear or lie in a low-dimensional subspace.',
                    solution: 'If the encoder states \\(h_1, \\ldots, h_T\\) are not in "general position" (i.e., they are approximately linearly dependent or lie in a subspace of dimension less than \\(T-1\\)), then the mapping from weights to context vectors, \\(\\alpha \\mapsto H\\alpha\\), has a nontrivial null space. Any \\(\\delta\\) in the null space of \\(H = [h_1, \\ldots, h_T]\\) satisfies \\(H\\delta = 0\\), so \\(\\alpha\\) and \\(\\alpha + \\delta\\) produce the same context vector. Concretely, if \\(h_3 \\approx 0.5 h_1 + 0.5 h_2\\), then shifting weight from \\(h_3\\) to a mixture of \\(h_1\\) and \\(h_2\\) barely changes \\(c\\). This is especially likely in practice because encoder states in nearby positions often encode similar information.'
                },
                {
                    question: 'Design a simple experiment to test whether attention weights are faithful explanations for a sentiment classifier. Describe the steps and what outcome would support (or undermine) the faithfulness of attention.',
                    hint: 'Consider what happens if you mask or permute the highest-attention tokens.',
                    solution: '<p><strong>Experiment design:</strong></p><ol><li>Train an attention-based sentiment classifier on a benchmark dataset (e.g., SST-2).</li><li>For each test example, identify the top-k words by attention weight.</li><li><strong>Masking test:</strong> Replace the top-k words with [UNK] or padding tokens. If attention is faithful, the model\'s prediction confidence should drop significantly. If the prediction barely changes, attention is not faithfully reflecting importance.</li><li><strong>Sufficiency test:</strong> Keep only the top-k words and mask everything else. If attention is faithful, the model should maintain its prediction from just these words.</li><li><strong>Comparison:</strong> Repeat steps 3-4 using gradient-based importance (e.g., input gradient norms) to select the top-k words. Compare the drop in accuracy.</li></ol><p><strong>Supportive outcome:</strong> Large accuracy drop when masking high-attention words, small drop for gradient-selected words, and vice versa for the sufficiency test, with both methods in agreement.</p><p><strong>Undermining outcome:</strong> Little accuracy change when masking high-attention words, or gradient-based selection significantly outperforms attention-based selection.</p>'
                },
                {
                    question: 'Attention mechanisms provide \\(O(1)\\) path length between any encoder and decoder position (compared to \\(O(T)\\) for vanilla RNNs). Explain why this matters for gradient flow and long-range dependency learning.',
                    hint: 'Think about the backpropagation path: how many multiplicative steps does the gradient pass through?',
                    solution: '<p>In a vanilla RNN encoder-decoder, the gradient from decoder step \\(t\\) to encoder step \\(i\\) must traverse \\(T - i\\) recurrent steps in the encoder plus the context vector bottleneck. Each recurrent step multiplies the gradient by the Jacobian \\(\\partial h_{k+1}/\\partial h_k\\), leading to exponential decay (vanishing gradients) or growth (exploding gradients) over \\(O(T)\\) steps.</p><p>With attention, the context vector \\(c_t = \\sum_i \\alpha_{t,i} h_i\\) creates a <em>direct</em> additive connection from each \\(h_i\\) to the decoder. The gradient \\(\\partial c_t / \\partial h_i = \\alpha_{t,i} I\\), which is a single multiplicative step regardless of how far apart \\(i\\) and \\(t\\) are. This means gradients can flow directly from the loss back to any encoder position without passing through intervening recurrent steps. The path length is \\(O(1)\\), dramatically improving the learning of long-range dependencies. This is the same principle that makes ResNet skip connections effective, and it is one reason why attention-based models consistently outperform vanilla RNNs on long sequences.</p>'
                }
            ]
        }
    ]
});
