// === Chapter 19: Pretrained Language Models ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch19',
    number: 19,
    title: 'Pretrained Language Models',
    subtitle: 'From next-token prediction to emergent intelligence: GPT, BERT, and the scaling revolution',
    sections: [
        // ======================== Section 1 ========================
        {
            id: 'sec19-1-language-modeling',
            title: 'Language Modeling',
            content: `
<h2>19.1 Language Modeling</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Predicting the Next Word</div><div class="env-body">
<p>A <strong>language model</strong> assigns probabilities to sequences of tokens. At its heart, a language model answers one question: given everything written so far, what comes next? This seemingly simple task, predicting the next word, turns out to require deep understanding of syntax, semantics, world knowledge, and reasoning. Language modeling is the foundational pretraining objective behind GPT, the task that launched the era of large language models.</p>
</div></div>

<h3>Autoregressive Factorization</h3>

<p>Any joint probability distribution over a sequence of tokens \\(w_1, w_2, \\ldots, w_T\\) can be decomposed using the chain rule of probability:</p>

\\[
P(w_1, w_2, \\ldots, w_T) = \\prod_{t=1}^{T} P(w_t \\mid w_1, \\ldots, w_{t-1}) = \\prod_{t=1}^{T} P(w_t \\mid w_{&lt;t}).
\\]

<p>This factorization is exact, requiring no assumptions. An <strong>autoregressive language model</strong> parameterizes each conditional \\(P_\\theta(w_t \\mid w_{&lt;t})\\) with a neural network and trains by maximizing the log-likelihood of observed text.</p>

<div class="env-block definition"><div class="env-title">Definition 19.1.1 &mdash; Autoregressive Language Model</div><div class="env-body">
<p>An autoregressive language model defines the probability of a sequence as</p>
\\[
P_\\theta(w_1, \\ldots, w_T) = \\prod_{t=1}^{T} P_\\theta(w_t \\mid w_{&lt;t}),
\\]
<p>where \\(P_\\theta(w_t \\mid w_{&lt;t}) = \\mathrm{softmax}(\\mathbf{W} \\mathbf{h}_t + \\mathbf{b})_{w_t}\\) and \\(\\mathbf{h}_t\\) is the hidden representation of the prefix \\(w_{&lt;t}\\) produced by a neural network (RNN, LSTM, or Transformer).</p>
</div></div>

<h3>Training Objective: Cross-Entropy</h3>

<p>Given a training corpus \\(\\mathcal{D} = \\{w_1^{(i)}, \\ldots, w_{T_i}^{(i)}\\}_{i=1}^N\\), the training objective is to minimize the negative log-likelihood:</p>
\\[
\\mathcal{L}(\\theta) = -\\frac{1}{|\\mathcal{D}|} \\sum_{i=1}^{N} \\sum_{t=1}^{T_i} \\log P_\\theta\\bigl(w_t^{(i)} \\mid w_{&lt;t}^{(i)}\\bigr).
\\]
<p>This is equivalent to the cross-entropy between the model's predicted distribution and the empirical one-hot distribution at each position.</p>

<div class="env-block definition"><div class="env-title">Definition 19.1.2 &mdash; Perplexity</div><div class="env-body">
<p>The <strong>perplexity</strong> of a language model on a test sequence \\(w_1, \\ldots, w_T\\) is</p>
\\[
\\mathrm{PPL} = \\exp\\!\\left(-\\frac{1}{T}\\sum_{t=1}^{T} \\log P_\\theta(w_t \\mid w_{&lt;t})\\right).
\\]
<p>Perplexity measures the effective number of equally likely next tokens the model considers at each step. A model with perplexity 30 is, on average, as uncertain as if it were choosing uniformly among 30 options. Lower perplexity indicates a better model.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Perplexity and Entropy</div><div class="env-body">
<p>Perplexity is the exponentiated cross-entropy: \\(\\mathrm{PPL} = 2^{H}\\) where \\(H\\) is the per-token cross-entropy in bits (or \\(e^{H}\\) when using natural log). English text typically has a perplexity of 20&ndash;100 for modern language models, reflecting the inherent redundancy and predictability of natural language.</p>
</div></div>

<h3>Tokenization: Byte-Pair Encoding (BPE)</h3>

<p>Neural language models do not operate on raw characters or whole words. Instead, they use <strong>subword tokenization</strong>, which splits text into variable-length pieces that balance vocabulary size with sequence length.</p>

<div class="env-block definition"><div class="env-title">Definition 19.1.3 &mdash; Byte-Pair Encoding (BPE)</div><div class="env-body">
<p>Byte-Pair Encoding (Sennrich et al., 2016) builds a subword vocabulary through a greedy iterative procedure:</p>
<ol>
<li>Initialize the vocabulary with all individual characters (or bytes).</li>
<li>Count all adjacent symbol pairs in the training corpus.</li>
<li>Merge the most frequent pair into a new symbol; add it to the vocabulary.</li>
<li>Repeat steps 2&ndash;3 until the vocabulary reaches a desired size \\(|V|\\).</li>
</ol>
<p>At inference, the same merge rules are applied deterministically to split any input text into subword tokens.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 19.1.4 &mdash; BPE in Action</div><div class="env-body">
<p>Consider the word "unbelievable" with initial character split: [u, n, b, e, l, i, e, v, a, b, l, e]. After learning merges, BPE might produce: [un, believ, able]. Common words like "the" become a single token, while rare words are split into recognizable pieces. This gives the model compositional understanding: "un-" as a prefix, "-able" as a suffix.</p>
</div></div>

<div class="env-block warning"><div class="env-title">Vocabulary Size Trade-off</div><div class="env-body">
<p>Larger vocabularies (e.g., 50k tokens) produce shorter sequences but require more embedding parameters. Smaller vocabularies produce longer sequences, increasing computational cost due to the \\(O(T^2)\\) self-attention complexity. GPT-2 uses \\(|V| = 50{,}257\\); LLaMA uses \\(|V| = 32{,}000\\). The choice is an engineering trade-off, not a fundamental limitation.</p>
</div></div>

<h3>Autoregressive Generation</h3>

<p>Once trained, a language model can <em>generate</em> text by sampling from its predicted distributions token by token:</p>
<ol>
<li>Start with a prompt \\(w_1, \\ldots, w_k\\).</li>
<li>Compute \\(P_\\theta(w_{k+1} \\mid w_{\\leq k})\\) and sample \\(w_{k+1}\\).</li>
<li>Append \\(w_{k+1}\\) to the sequence and repeat.</li>
</ol>

<p>The <strong>temperature</strong> parameter \\(\\tau\\) controls the sharpness of the sampling distribution:</p>
\\[
P_\\tau(w_t = v) = \\frac{\\exp(z_v / \\tau)}{\\sum_{v'} \\exp(z_{v'} / \\tau)},
\\]
<p>where \\(z_v\\) are the logits. At \\(\\tau \\to 0\\), the model always picks the most likely token (greedy decoding). At \\(\\tau = 1\\), we sample from the model's distribution. At \\(\\tau \\to \\infty\\), sampling approaches uniform randomness.</p>

<div class="viz-placeholder" data-viz="viz-autoregressive-gen"></div>
`,
            visualizations: [
                {
                    id: 'viz-autoregressive-gen',
                    title: 'Autoregressive Generation',
                    description: 'Watch token-by-token generation. At each step, the model produces a probability distribution over the vocabulary. The next token is sampled from this distribution. Adjust the temperature to control randomness.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        // Simulated vocabulary and bigram-ish probabilities
                        var vocab = ['The', 'cat', 'sat', 'on', 'a', 'mat', 'dog', 'ran', 'to', 'big', 'the', 'red', 'in', 'warm', 'house', '.'];
                        // Transition probabilities (simplified bigram)
                        var transitions = {
                            '_START_': { 'The': 0.45, 'A': 0.1, 'a': 0.15, 'the': 0.2, 'big': 0.05, 'red': 0.05 },
                            'The': { 'cat': 0.3, 'dog': 0.25, 'big': 0.15, 'red': 0.1, 'warm': 0.1, 'mat': 0.05, 'house': 0.05 },
                            'the': { 'cat': 0.2, 'dog': 0.2, 'big': 0.15, 'red': 0.15, 'warm': 0.1, 'mat': 0.1, 'house': 0.1 },
                            'cat': { 'sat': 0.4, 'ran': 0.3, 'on': 0.05, 'in': 0.05, '.': 0.2 },
                            'dog': { 'sat': 0.2, 'ran': 0.4, 'on': 0.05, 'in': 0.05, '.': 0.3 },
                            'sat': { 'on': 0.5, 'in': 0.2, '.': 0.15, 'a': 0.1, 'the': 0.05 },
                            'ran': { 'to': 0.4, 'on': 0.15, 'in': 0.2, '.': 0.15, 'a': 0.1 },
                            'on': { 'the': 0.35, 'a': 0.35, 'big': 0.1, 'red': 0.1, 'warm': 0.1 },
                            'to': { 'the': 0.35, 'a': 0.35, 'big': 0.1, 'red': 0.1, 'warm': 0.1 },
                            'in': { 'the': 0.35, 'a': 0.35, 'big': 0.1, 'red': 0.1, 'warm': 0.1 },
                            'a': { 'cat': 0.1, 'dog': 0.1, 'mat': 0.2, 'big': 0.2, 'red': 0.15, 'warm': 0.1, 'house': 0.15 },
                            'big': { 'cat': 0.2, 'dog': 0.2, 'mat': 0.15, 'red': 0.15, 'house': 0.15, 'warm': 0.15 },
                            'red': { 'cat': 0.15, 'dog': 0.15, 'mat': 0.25, 'house': 0.25, '.': 0.2 },
                            'warm': { 'cat': 0.1, 'dog': 0.1, 'mat': 0.2, 'house': 0.3, '.': 0.3 },
                            'mat': { '.': 0.7, 'in': 0.1, 'on': 0.1, 'sat': 0.1 },
                            'house': { '.': 0.7, 'in': 0.1, 'on': 0.1, 'sat': 0.1 },
                            '.': {}
                        };

                        var generated = [];
                        var currentProbs = {};
                        var temperature = 1.0;
                        var animStep = 0;
                        var animating = false;
                        var animId = null;

                        function getProbs(lastToken, temp) {
                            var raw = transitions[lastToken] || {};
                            var keys = Object.keys(raw);
                            if (keys.length === 0) return {};
                            // Apply temperature
                            var logits = {};
                            keys.forEach(function(k) { logits[k] = Math.log(raw[k] + 1e-10); });
                            var maxL = -Infinity;
                            keys.forEach(function(k) { if (logits[k] > maxL) maxL = logits[k]; });
                            var expSum = 0;
                            keys.forEach(function(k) { logits[k] = Math.exp((logits[k] - maxL) / temp); expSum += logits[k]; });
                            var result = {};
                            keys.forEach(function(k) { result[k] = logits[k] / expSum; });
                            return result;
                        }

                        function sampleFrom(probs) {
                            var keys = Object.keys(probs);
                            var r = Math.random();
                            var cum = 0;
                            for (var i = 0; i < keys.length; i++) {
                                cum += probs[keys[i]];
                                if (r <= cum) return keys[i];
                            }
                            return keys[keys.length - 1];
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Autoregressive Token Generation', 20, 15);

                            // Draw generated sequence
                            ctx.font = '15px monospace';
                            var seqX = 20;
                            var seqY = 55;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Generated:', seqX, seqY - 16);
                            ctx.font = '15px monospace';
                            var cursorX = seqX;
                            for (var i = 0; i < generated.length; i++) {
                                // Highlight last token
                                if (i === generated.length - 1) {
                                    var tw = ctx.measureText(generated[i]).width;
                                    ctx.fillStyle = viz.colors.blue + '33';
                                    ctx.fillRect(cursorX - 2, seqY - 2, tw + 4, 22);
                                    ctx.fillStyle = viz.colors.blue;
                                } else {
                                    ctx.fillStyle = viz.colors.white;
                                }
                                ctx.fillText(generated[i], cursorX, seqY);
                                cursorX += ctx.measureText(generated[i]).width + 8;
                            }
                            // Blinking cursor
                            if (generated.length < 10 && Object.keys(currentProbs).length > 0) {
                                var blink = Math.floor(Date.now() / 500) % 2;
                                if (blink) {
                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.fillRect(cursorX, seqY - 1, 2, 18);
                                }
                            }

                            // Draw probability distribution
                            var probKeys = Object.keys(currentProbs);
                            if (probKeys.length > 0) {
                                // Sort by probability descending
                                probKeys.sort(function(a, b) { return currentProbs[b] - currentProbs[a]; });

                                var barX = 40;
                                var barY = 110;
                                var barMaxW = W - 160;
                                var barH = 22;
                                var barGap = 6;

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('P(next token | context), temperature = ' + temperature.toFixed(1), barX, barY - 16);

                                var colors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.green, viz.colors.purple, viz.colors.red, viz.colors.yellow, viz.colors.pink];
                                for (var j = 0; j < probKeys.length && j < 10; j++) {
                                    var token = probKeys[j];
                                    var prob = currentProbs[token];
                                    var bw = prob * barMaxW;
                                    var by = barY + j * (barH + barGap);
                                    var col = colors[j % colors.length];

                                    // Bar
                                    ctx.fillStyle = col + '44';
                                    ctx.fillRect(barX + 60, by, bw, barH);
                                    ctx.fillStyle = col;
                                    ctx.fillRect(barX + 60, by, Math.max(bw, 2), barH);

                                    // Token label
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = '13px monospace';
                                    ctx.textAlign = 'right';
                                    ctx.fillText(token, barX + 52, by + barH / 2 + 4);

                                    // Probability value
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '11px -apple-system,sans-serif';
                                    ctx.textAlign = 'left';
                                    ctx.fillText((prob * 100).toFixed(1) + '%', barX + 64 + bw, by + barH / 2 + 4);
                                }

                                // Entropy and perplexity
                                var entropy = 0;
                                probKeys.forEach(function(k) {
                                    if (currentProbs[k] > 0) entropy -= currentProbs[k] * Math.log2(currentProbs[k]);
                                });
                                var ppl = Math.pow(2, entropy);
                                var infoY = barY + Math.min(probKeys.length, 10) * (barH + barGap) + 15;
                                ctx.fillStyle = viz.colors.teal;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('Entropy: ' + entropy.toFixed(2) + ' bits', barX, infoY);
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('Perplexity: ' + ppl.toFixed(1), barX + 180, infoY);
                            } else if (generated.length === 0) {
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '13px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Click "Generate" to start autoregressive generation', W / 2, H / 2);
                            }

                            // Temperature indicator
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.fillText('\u03C4 = ' + temperature.toFixed(1), W - 20, 15);
                            var tempLabel = temperature < 0.3 ? '(greedy)' : temperature < 0.8 ? '(focused)' : temperature < 1.5 ? '(balanced)' : '(creative)';
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText(tempLabel, W - 20, 32);
                        }

                        function stepGenerate() {
                            if (generated.length >= 10) { animating = false; return; }
                            var last = generated.length === 0 ? '_START_' : generated[generated.length - 1];
                            currentProbs = getProbs(last, temperature);
                            if (Object.keys(currentProbs).length === 0) { animating = false; return; }
                            var token = sampleFrom(currentProbs);
                            generated.push(token);
                            draw();
                            if (token === '.') { animating = false; return; }
                        }

                        function autoGenerate() {
                            generated = [];
                            currentProbs = {};
                            animating = true;
                            var stepCount = 0;
                            function tick() {
                                if (!animating || stepCount > 12) return;
                                stepGenerate();
                                stepCount++;
                                if (animating) animId = setTimeout(tick, 700);
                            }
                            tick();
                        }

                        VizEngine.createSlider(controls, 'Temperature', 0.1, 3.0, 1.0, 0.1, function(v) {
                            temperature = v;
                            if (generated.length > 0) {
                                var last = generated[generated.length - 1];
                                currentProbs = getProbs(last, temperature);
                            }
                            draw();
                        });
                        VizEngine.createButton(controls, 'Generate', function() { autoGenerate(); });
                        VizEngine.createButton(controls, 'Step', function() { stepGenerate(); });
                        VizEngine.createButton(controls, 'Reset', function() {
                            animating = false;
                            if (animId) clearTimeout(animId);
                            generated = [];
                            currentProbs = getProbs('_START_', temperature);
                            draw();
                        });

                        currentProbs = getProbs('_START_', temperature);
                        draw();

                        return { stopAnimation: function() { animating = false; if (animId) clearTimeout(animId); viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A language model assigns perplexity 25 to a test set. What is the average per-token cross-entropy in bits?',
                    hint: 'Use the relationship \\(\\mathrm{PPL} = 2^H\\), so \\(H = \\log_2(\\mathrm{PPL})\\).',
                    solution: '\\(H = \\log_2(25) \\approx 4.64\\) bits per token. This means the model\'s uncertainty at each position is equivalent to a uniform choice among roughly 25 options.'
                },
                {
                    question: 'Explain why BPE is preferred over character-level tokenization for Transformer language models. What is the main trade-off?',
                    hint: 'Think about sequence length and its effect on self-attention cost.',
                    solution: 'Character-level tokenization produces very long sequences (a 500-word passage might be 2,500 characters), and self-attention has \\(O(T^2)\\) cost. BPE reduces the sequence length dramatically (to perhaps 600 tokens for the same passage) by merging frequent character pairs into subwords, making training and inference far more efficient. The trade-off is that a larger vocabulary requires more embedding parameters, and the tokenization boundary can affect model behavior on rare or novel words.'
                },
                {
                    question: 'Show that as temperature \\(\\tau \\to 0\\), the softmax distribution \\(P_\\tau(v) = \\frac{\\exp(z_v/\\tau)}{\\sum_{v\'} \\exp(z_{v\'}/\\tau)}\\) converges to a point mass on \\(\\arg\\max_v z_v\\).',
                    hint: 'Factor out \\(\\exp(z_{\\max}/\\tau)\\) from numerator and denominator, then take the limit.',
                    solution: 'Let \\(v^* = \\arg\\max_v z_v\\). Then \\(P_\\tau(v) = \\frac{\\exp((z_v - z_{v^*})/\\tau)}{\\sum_{v\'} \\exp((z_{v\'} - z_{v^*})/\\tau)}\\). For \\(v \\neq v^*\\), \\(z_v - z_{v^*} < 0\\), so \\(\\exp((z_v - z_{v^*})/\\tau) \\to 0\\) as \\(\\tau \\to 0\\). The numerator for \\(v = v^*\\) is 1, and the denominator approaches 1 (since all other terms vanish). Thus \\(P_\\tau(v^*) \\to 1\\) and \\(P_\\tau(v) \\to 0\\) for \\(v \\neq v^*\\).'
                }
            ]
        },

        // ======================== Section 2 ========================
        {
            id: 'sec19-2-gpt',
            title: 'GPT: Autoregressive Pretraining',
            content: `
<h2>19.2 GPT: Autoregressive Pretraining</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Reading Left to Right</div><div class="env-body">
<p>GPT (Generative Pre-trained Transformer, Radford et al., 2018) combines the Transformer architecture from Chapter 14 with autoregressive language modeling. The key insight: a Transformer trained on massive text corpora to predict the next token learns rich internal representations that transfer to virtually any downstream NLP task. GPT reads text strictly left-to-right, like a human reading a page; each token can only attend to tokens before it.</p>
</div></div>

<h3>Causal (Unidirectional) Attention</h3>

<p>In the standard Transformer encoder (Chapter 14), each token attends to all other tokens. For autoregressive generation, this would be <em>cheating</em>: the model would see the very tokens it is supposed to predict. GPT uses <strong>causal masking</strong> (also called the autoregressive mask) to prevent information from flowing backward.</p>

<div class="env-block definition"><div class="env-title">Definition 19.2.1 &mdash; Causal Self-Attention</div><div class="env-body">
<p>Given a sequence of \\(T\\) tokens with queries \\(\\mathbf{Q}\\), keys \\(\\mathbf{K}\\), and values \\(\\mathbf{V}\\) (each in \\(\\mathbb{R}^{T \\times d_k}\\)), <strong>causal self-attention</strong> computes:</p>
\\[
\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\!\\left(\\frac{\\mathbf{Q}\\mathbf{K}^\\top}{\\sqrt{d_k}} + \\mathbf{M}\\right) \\mathbf{V},
\\]
<p>where \\(\\mathbf{M} \\in \\mathbb{R}^{T \\times T}\\) is the causal mask:</p>
\\[
M_{ij} = \\begin{cases} 0 & \\text{if } j \\leq i \\\\ -\\infty & \\text{if } j &gt; i \\end{cases}
\\]
<p>Setting \\(M_{ij} = -\\infty\\) for \\(j &gt; i\\) ensures that after softmax, position \\(i\\) assigns zero attention weight to all future positions \\(j &gt; i\\).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Why Masking Works</div><div class="env-body">
<p>Adding \\(-\\infty\\) before the softmax is mathematically equivalent to zeroing out the corresponding attention weights and renormalizing. If \\(z_j \\to -\\infty\\), then \\(\\exp(z_j) \\to 0\\), so the softmax output for that entry becomes zero. This is more numerically stable than explicitly zeroing and renormalizing, because it avoids division-by-zero issues when many entries are masked.</p>
</div></div>

<h3>GPT Architecture</h3>

<p>GPT stacks \\(L\\) Transformer <em>decoder</em> blocks (each with masked self-attention and a feed-forward network). The full architecture is:</p>
<ol>
<li><strong>Token + Position Embedding:</strong> \\(\\mathbf{h}_0 = \\mathbf{W}_e \\mathbf{x} + \\mathbf{W}_p\\), where \\(\\mathbf{W}_e \\in \\mathbb{R}^{|V| \\times d}\\) are token embeddings and \\(\\mathbf{W}_p \\in \\mathbb{R}^{T_{\\max} \\times d}\\) are learned positional embeddings.</li>
<li><strong>Transformer Blocks:</strong> For \\(\\ell = 1, \\ldots, L\\):
\\[
\\mathbf{a}_\\ell = \\text{LayerNorm}(\\mathbf{h}_{\\ell-1} + \\text{MaskedMHA}(\\mathbf{h}_{\\ell-1}))
\\]
\\[
\\mathbf{h}_\\ell = \\text{LayerNorm}(\\mathbf{a}_\\ell + \\text{FFN}(\\mathbf{a}_\\ell))
\\]
</li>
<li><strong>Language Model Head:</strong> \\(P(w_t \\mid w_{&lt;t}) = \\text{softmax}(\\mathbf{h}_L^{(t)} \\mathbf{W}_e^\\top)\\). Note: the output projection reuses the token embedding matrix (weight tying).</li>
</ol>

<div class="env-block example"><div class="env-title">Example 19.2.2 &mdash; GPT Model Sizes</div><div class="env-body">
<table style="width:100%; border-collapse: collapse; font-size: 0.9em;">
<tr style="border-bottom: 1px solid #30363d;"><th style="text-align:left; padding:4px;">Model</th><th>Parameters</th><th>Layers</th><th>\\(d_{\\text{model}}\\)</th><th>Heads</th><th>Context</th></tr>
<tr><td style="padding:4px;">GPT-1</td><td>117M</td><td>12</td><td>768</td><td>12</td><td>512</td></tr>
<tr><td style="padding:4px;">GPT-2</td><td>1.5B</td><td>48</td><td>1600</td><td>25</td><td>1024</td></tr>
<tr><td style="padding:4px;">GPT-3</td><td>175B</td><td>96</td><td>12288</td><td>96</td><td>2048</td></tr>
</table>
<p>The architecture remained essentially the same across all three versions; the difference is purely in scale.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 19.2.3 &mdash; Computational Cost of Causal Attention</div><div class="env-body">
<p>For a sequence of length \\(T\\) with model dimension \\(d\\), the time and memory cost of causal self-attention is \\(O(T^2 d)\\) for the attention computation and \\(O(T^2)\\) for storing the attention matrix. The causal mask does not reduce this asymptotic cost; it only changes the sparsity pattern of the attention weights, not the matrix dimensions.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; KV Cache for Efficient Inference</div><div class="env-body">
<p>During autoregressive generation, we generate one token at a time. Naively, each new token requires recomputing attention over the entire prefix. The <strong>KV cache</strong> stores the key and value projections of all previous tokens, so that generating token \\(t\\) only requires computing the query for position \\(t\\) and attending to the cached keys/values. This reduces per-token cost from \\(O(T \\cdot d)\\) to \\(O(d)\\) for the projection, though attention still requires \\(O(T)\\) per head to dot-product with all cached keys.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-causal-mask"></div>
`,
            visualizations: [
                {
                    id: 'viz-causal-mask',
                    title: 'Causal Attention Mask',
                    description: 'The causal mask is a lower-triangular matrix that determines which tokens can attend to which. Green cells indicate allowed attention (j <= i); dark cells indicate blocked attention (j > i). Hover over cells to see the attention pattern for each query position.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 720, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
                        var T = tokens.length;
                        var cellSize = 48;
                        var gridX = 180;
                        var gridY = 100;
                        var hoveredRow = -1;
                        var hoveredCol = -1;

                        // Simulated attention weights (lower triangular, each row sums to 1)
                        function getAttention(row) {
                            var weights = [];
                            var sum = 0;
                            for (var j = 0; j <= row; j++) {
                                // Roughly: more recent tokens get higher weight
                                var w = Math.exp(-0.3 * (row - j));
                                weights.push(w);
                                sum += w;
                            }
                            for (var j2 = 0; j2 <= row; j2++) weights[j2] /= sum;
                            return weights;
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Causal Self-Attention Mask', W / 2, 25);

                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Key (attends to)', gridX + T * cellSize / 2, gridY - 30);
                            ctx.save();
                            ctx.translate(gridX - 35, gridY + T * cellSize / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Query (current token)', 0, 0);
                            ctx.restore();

                            // Column headers (keys)
                            ctx.font = '12px monospace';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            for (var j = 0; j < T; j++) {
                                ctx.fillStyle = (hoveredCol === j) ? viz.colors.blue : viz.colors.text;
                                ctx.fillText(tokens[j], gridX + j * cellSize + cellSize / 2, gridY - 5);
                            }

                            // Row headers (queries)
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var i = 0; i < T; i++) {
                                ctx.fillStyle = (hoveredRow === i) ? viz.colors.orange : viz.colors.text;
                                ctx.fillText(tokens[i], gridX - 10, gridY + i * cellSize + cellSize / 2);
                            }

                            // Grid cells
                            var attnWeights = (hoveredRow >= 0) ? getAttention(hoveredRow) : null;

                            for (var i2 = 0; i2 < T; i2++) {
                                for (var j2 = 0; j2 < T; j2++) {
                                    var cx = gridX + j2 * cellSize;
                                    var cy = gridY + i2 * cellSize;
                                    var allowed = j2 <= i2;

                                    if (allowed) {
                                        if (hoveredRow === i2) {
                                            // Show attention weight intensity
                                            var aw = attnWeights[j2];
                                            var g = Math.round(100 + 155 * aw);
                                            ctx.fillStyle = 'rgba(63,' + g + ',80,' + (0.3 + 0.7 * aw) + ')';
                                        } else {
                                            ctx.fillStyle = '#1a4a2a';
                                        }
                                    } else {
                                        ctx.fillStyle = '#1a1a2a';
                                    }

                                    ctx.fillRect(cx + 1, cy + 1, cellSize - 2, cellSize - 2);

                                    // Border
                                    ctx.strokeStyle = allowed ? viz.colors.green + '44' : '#222';
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(cx + 1, cy + 1, cellSize - 2, cellSize - 2);

                                    // Show weight value when hovered
                                    if (hoveredRow === i2 && allowed) {
                                        ctx.fillStyle = viz.colors.white;
                                        ctx.font = '10px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText((attnWeights[j2] * 100).toFixed(0) + '%', cx + cellSize / 2, cy + cellSize / 2);
                                    } else if (!allowed) {
                                        ctx.fillStyle = viz.colors.red + '44';
                                        ctx.font = '14px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText('\u2716', cx + cellSize / 2, cy + cellSize / 2);
                                    } else {
                                        ctx.fillStyle = viz.colors.green + '66';
                                        ctx.font = '14px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText('\u2714', cx + cellSize / 2, cy + cellSize / 2);
                                    }
                                }
                            }

                            // Legend
                            var legX = gridX + T * cellSize + 30;
                            var legY = gridY + 20;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';

                            ctx.fillStyle = '#1a4a2a';
                            ctx.fillRect(legX, legY, 16, 16);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('Can attend (j \u2264 i)', legX + 22, legY + 8);

                            ctx.fillStyle = '#1a1a2a';
                            ctx.fillRect(legX, legY + 28, 16, 16);
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText('Masked (j > i)', legX + 22, legY + 36);

                            // Info text
                            if (hoveredRow >= 0) {
                                ctx.fillStyle = viz.colors.orange;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('"' + tokens[hoveredRow] + '" attends to: ' + tokens.slice(0, hoveredRow + 1).join(', '), 20, H - 20);
                            } else {
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Hover over a row to see attention weights for that query position', W / 2, H - 20);
                            }
                        }

                        viz.canvas.addEventListener('mousemove', function(e) {
                            var rect = viz.canvas.getBoundingClientRect();
                            var mx = e.clientX - rect.left;
                            var my = e.clientY - rect.top;
                            var col = Math.floor((mx - gridX) / cellSize);
                            var row = Math.floor((my - gridY) / cellSize);
                            hoveredRow = (row >= 0 && row < T) ? row : -1;
                            hoveredCol = (col >= 0 && col < T) ? col : -1;
                            draw();
                        });
                        viz.canvas.addEventListener('mouseleave', function() {
                            hoveredRow = -1;
                            hoveredCol = -1;
                            draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In a GPT model with context length \\(T = 2048\\) and \\(d_{\\text{model}} = 1024\\), how many floating-point numbers does the KV cache store per layer per sequence?',
                    hint: 'Each layer caches the key and value matrices, both of shape \\(T \\times d_{\\text{model}}\\).',
                    solution: 'The KV cache stores \\(\\mathbf{K} \\in \\mathbb{R}^{T \\times d}\\) and \\(\\mathbf{V} \\in \\mathbb{R}^{T \\times d}\\), so \\(2 \\times 2048 \\times 1024 = 4{,}194{,}304\\) floats per layer. For a 48-layer model (GPT-2 scale), this totals \\(48 \\times 4{,}194{,}304 \\approx 201\\)M floats, which at FP16 is about 402 MB per sequence.'
                },
                {
                    question: 'Explain why GPT uses weight tying between the input token embedding matrix and the output projection matrix. What benefit does this provide?',
                    hint: 'Consider the parameter count and the semantic relationship between input embeddings and output logits.',
                    solution: 'Weight tying sets \\(\\mathbf{W}_{\\text{output}} = \\mathbf{W}_e^\\top\\), saving \\(|V| \\times d\\) parameters (e.g., \\(50{,}257 \\times 1600 \\approx 80\\)M for GPT-2). Conceptually, it enforces that the model uses the same representation space for understanding input tokens and predicting output tokens: the logit for token \\(v\\) at position \\(t\\) is the dot product \\(\\mathbf{h}_t^\\top \\mathbf{e}_v\\), measuring how well the hidden state aligns with the embedding of \\(v\\). This acts as a regularizer and improves perplexity, especially for smaller models.'
                },
                {
                    question: 'Prove that the causal mask is necessary for correct training: without it, the cross-entropy loss at position \\(t\\) would be trivially minimized because the model can "see" the answer \\(w_t\\).',
                    hint: 'Consider what happens when position \\(t\\) can attend to position \\(t\\) (or later) in the input. Where does \\(w_t\\) appear in the input sequence?',
                    solution: 'In GPT, the input is the sequence \\([w_1, w_2, \\ldots, w_T]\\) and the target at position \\(t\\) is \\(w_{t+1}\\). If the attention at position \\(t\\) can attend to position \\(t+1\\), then \\(\\mathbf{h}_t\\) has direct access to the embedding of \\(w_{t+1}\\), the very token it must predict. The model can then learn to simply copy this information, achieving near-zero loss without learning any real language structure. This is called "information leakage." The causal mask prevents position \\(t\\) from attending to any position \\(j > t\\), forcing the model to genuinely predict rather than copy.'
                }
            ]
        },

        // ======================== Section 3 ========================
        {
            id: 'sec19-3-bert',
            title: 'BERT: Bidirectional Pretraining',
            content: `
<h2>19.3 BERT: Bidirectional Pretraining</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Fill in the Blank</div><div class="env-body">
<p>GPT reads text left-to-right, but many NLP tasks benefit from understanding context in <em>both</em> directions. Consider the sentence "The [???] chased the mouse." A left-to-right model sees only "The" before the blank, giving weak signal. But a model that also sees "chased the mouse" can confidently infer "cat." BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019) achieves this by replacing next-token prediction with a <strong>masked language model (MLM)</strong> objective: hide some tokens and predict them from surrounding context.</p>
</div></div>

<h3>Masked Language Modeling (MLM)</h3>

<div class="env-block definition"><div class="env-title">Definition 19.3.1 &mdash; Masked Language Model</div><div class="env-body">
<p>Given an input sequence \\(w_1, \\ldots, w_T\\), randomly select a subset \\(\\mathcal{M} \\subset \\{1, \\ldots, T\\}\\) of positions (typically 15% of tokens). For each position \\(i \\in \\mathcal{M}\\):</p>
<ul>
<li>With probability 0.8, replace \\(w_i\\) with a special [MASK] token.</li>
<li>With probability 0.1, replace \\(w_i\\) with a random token from the vocabulary.</li>
<li>With probability 0.1, keep \\(w_i\\) unchanged.</li>
</ul>
<p>The MLM objective trains the model to predict the original token at each masked position:</p>
\\[
\\mathcal{L}_{\\text{MLM}} = -\\sum_{i \\in \\mathcal{M}} \\log P_\\theta(w_i \\mid \\tilde{w}_1, \\ldots, \\tilde{w}_T),
\\]
<p>where \\(\\tilde{w}\\) denotes the corrupted sequence.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Why the 80/10/10 Split?</div><div class="env-body">
<p>If BERT only used [MASK] tokens during pretraining, there would be a distribution mismatch during fine-tuning (where no [MASK] tokens appear). The 10% random replacement forces the model to maintain good representations for all positions (not just masked ones), and the 10% unchanged case teaches it that even non-masked positions may need to be "predicted." This is a form of input noise regularization.</p>
</div></div>

<h3>BERT Architecture</h3>

<p>BERT uses a standard Transformer <em>encoder</em> (not decoder), with <strong>no causal mask</strong>. Every position can attend to every other position, enabling full bidirectional context.</p>

<div class="env-block definition"><div class="env-title">Definition 19.3.2 &mdash; BERT Encoder</div><div class="env-body">
<p>BERT encodes the input as:</p>
\\[
\\mathbf{h}_0^{(i)} = \\mathbf{E}_{\\text{tok}}(\\tilde{w}_i) + \\mathbf{E}_{\\text{pos}}(i) + \\mathbf{E}_{\\text{seg}}(s_i),
\\]
<p>where \\(\\mathbf{E}_{\\text{tok}}\\), \\(\\mathbf{E}_{\\text{pos}}\\), and \\(\\mathbf{E}_{\\text{seg}}\\) are the token, position, and segment embeddings respectively. The segment embedding \\(s_i \\in \\{A, B\\}\\) distinguishes the two sentences in a sentence pair. The input passes through \\(L\\) Transformer encoder layers with full (bidirectional) self-attention.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 19.3.3 &mdash; BERT Configurations</div><div class="env-body">
<table style="width:100%; border-collapse: collapse; font-size: 0.9em;">
<tr style="border-bottom: 1px solid #30363d;"><th style="text-align:left; padding:4px;">Model</th><th>Parameters</th><th>Layers</th><th>\\(d_{\\text{model}}\\)</th><th>Heads</th></tr>
<tr><td style="padding:4px;">BERT-Base</td><td>110M</td><td>12</td><td>768</td><td>12</td></tr>
<tr><td style="padding:4px;">BERT-Large</td><td>340M</td><td>24</td><td>1024</td><td>16</td></tr>
</table>
</div></div>

<h3>Next Sentence Prediction (NSP)</h3>

<p>BERT's second pretraining objective is <strong>Next Sentence Prediction</strong>: given two segments A and B, predict whether B is the actual next sentence after A in the corpus or a random sentence.</p>

<div class="env-block definition"><div class="env-title">Definition 19.3.4 &mdash; Next Sentence Prediction</div><div class="env-body">
<p>The input to BERT is formatted as <code>[CLS] A [SEP] B [SEP]</code>. The [CLS] token's final hidden state \\(\\mathbf{h}_L^{[\\text{CLS}]}\\) is fed to a binary classifier:</p>
\\[
P(\\text{IsNext} \\mid A, B) = \\sigma\\bigl(\\mathbf{w}_{\\text{nsp}}^\\top \\mathbf{h}_L^{[\\text{CLS}]} + b_{\\text{nsp}}\\bigr).
\\]
<p>During pretraining, 50% of sentence pairs are genuine consecutive sentences, and 50% are random pairs.</p>
</div></div>

<div class="env-block warning"><div class="env-title">NSP is Controversial</div><div class="env-body">
<p>Later work (RoBERTa, Liu et al., 2019) showed that removing the NSP objective and training only with MLM actually <em>improves</em> downstream performance. The NSP task is too easy and does not contribute meaningful learning signal. Most post-BERT models drop NSP entirely. ALBERT replaces it with sentence-order prediction (SOP), which asks whether A precedes B or B precedes A, forcing the model to learn discourse coherence rather than mere topic detection.</p>
</div></div>

<h3>GPT vs. BERT: A Fundamental Trade-off</h3>

<div class="env-block remark"><div class="env-title">Remark &mdash; Directionality Trade-off</div><div class="env-body">
<p>GPT and BERT represent two sides of a fundamental trade-off:</p>
<ul>
<li><strong>GPT (autoregressive):</strong> Models \\(P(w_t \\mid w_{&lt;t})\\). Can generate text naturally. But each position sees only left context, limiting representation quality for understanding tasks.</li>
<li><strong>BERT (masked LM):</strong> Models \\(P(w_t \\mid w_{\\setminus t})\\). Each position sees full bidirectional context, yielding richer representations. But cannot generate text autoregressively because it does not define a proper left-to-right joint distribution.</li>
</ul>
<p>In practice, GPT-style models dominate generation tasks, and BERT-style models (before the GPT-3 era) dominated understanding tasks like classification, NER, and question answering.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-bert-mlm"></div>
`,
            visualizations: [
                {
                    id: 'viz-bert-mlm',
                    title: 'BERT Masked Language Model',
                    description: 'A sentence with masked tokens and top-5 predictions for each [MASK] position. Click on different [MASK] tokens to see how bidirectional context informs predictions.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 430, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var sentences = [
                            {
                                display: ['The', '[MASK]', 'sat', 'on', 'the', '[MASK]', '.'],
                                original: ['The', 'cat', 'sat', 'on', 'the', 'mat', '.'],
                                predictions: {
                                    1: [
                                        { token: 'cat', prob: 0.35 },
                                        { token: 'dog', prob: 0.22 },
                                        { token: 'man', prob: 0.12 },
                                        { token: 'boy', prob: 0.08 },
                                        { token: 'bird', prob: 0.06 }
                                    ],
                                    5: [
                                        { token: 'mat', prob: 0.28 },
                                        { token: 'floor', prob: 0.18 },
                                        { token: 'table', prob: 0.14 },
                                        { token: 'bed', prob: 0.11 },
                                        { token: 'ground', prob: 0.09 }
                                    ]
                                }
                            },
                            {
                                display: ['Paris', 'is', 'the', '[MASK]', 'of', '[MASK]', '.'],
                                original: ['Paris', 'is', 'the', 'capital', 'of', 'France', '.'],
                                predictions: {
                                    3: [
                                        { token: 'capital', prob: 0.62 },
                                        { token: 'center', prob: 0.10 },
                                        { token: 'heart', prob: 0.07 },
                                        { token: 'city', prob: 0.05 },
                                        { token: 'hub', prob: 0.03 }
                                    ],
                                    5: [
                                        { token: 'France', prob: 0.71 },
                                        { token: 'Europe', prob: 0.08 },
                                        { token: 'love', prob: 0.04 },
                                        { token: 'fashion', prob: 0.03 },
                                        { token: 'art', prob: 0.02 }
                                    ]
                                }
                            },
                            {
                                display: ['She', '[MASK]', 'to', 'the', 'store', 'to', '[MASK]', 'groceries', '.'],
                                original: ['She', 'went', 'to', 'the', 'store', 'to', 'buy', 'groceries', '.'],
                                predictions: {
                                    1: [
                                        { token: 'went', prob: 0.40 },
                                        { token: 'drove', prob: 0.15 },
                                        { token: 'walked', prob: 0.13 },
                                        { token: 'ran', prob: 0.08 },
                                        { token: 'rushed', prob: 0.05 }
                                    ],
                                    6: [
                                        { token: 'buy', prob: 0.52 },
                                        { token: 'get', prob: 0.20 },
                                        { token: 'purchase', prob: 0.08 },
                                        { token: 'pick', prob: 0.05 },
                                        { token: 'grab', prob: 0.04 }
                                    ]
                                }
                            }
                        ];

                        var currentSent = 0;
                        var selectedMask = -1;

                        function getMaskPositions(sent) {
                            var pos = [];
                            for (var i = 0; i < sent.display.length; i++) {
                                if (sent.display[i] === '[MASK]') pos.push(i);
                            }
                            return pos;
                        }

                        var tokenPositions = []; // store screen positions for click detection

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var sent = sentences[currentSent];
                            var maskPos = getMaskPositions(sent);
                            if (selectedMask < 0 && maskPos.length > 0) selectedMask = maskPos[0];

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('BERT Masked Language Model Predictions', W / 2, 25);

                            // Draw sentence with tokens
                            tokenPositions = [];
                            ctx.font = '16px monospace';
                            var totalWidth = 0;
                            var gaps = [];
                            for (var i = 0; i < sent.display.length; i++) {
                                var tw = ctx.measureText(sent.display[i]).width + 16;
                                gaps.push(tw);
                                totalWidth += tw;
                            }
                            var startX = (W - totalWidth) / 2;
                            var tokY = 70;

                            for (var i2 = 0; i2 < sent.display.length; i2++) {
                                var tok = sent.display[i2];
                                var tw2 = gaps[i2];
                                var isMask = tok === '[MASK]';
                                var isSelected = isMask && (i2 === selectedMask);

                                // Background
                                if (isSelected) {
                                    ctx.fillStyle = viz.colors.orange + '55';
                                    ctx.fillRect(startX - 4, tokY - 6, tw2 + 2, 30);
                                    ctx.strokeStyle = viz.colors.orange;
                                    ctx.lineWidth = 2;
                                    ctx.strokeRect(startX - 4, tokY - 6, tw2 + 2, 30);
                                } else if (isMask) {
                                    ctx.fillStyle = viz.colors.purple + '33';
                                    ctx.fillRect(startX - 4, tokY - 6, tw2 + 2, 30);
                                    ctx.strokeStyle = viz.colors.purple;
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(startX - 4, tokY - 6, tw2 + 2, 30);
                                }

                                ctx.fillStyle = isMask ? viz.colors.orange : viz.colors.white;
                                ctx.font = isMask ? 'bold 15px monospace' : '16px monospace';
                                ctx.textAlign = 'left';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(tok, startX, tokY + 8);

                                tokenPositions.push({ x: startX - 4, y: tokY - 6, w: tw2 + 2, h: 30, idx: i2 });
                                startX += tw2;
                            }

                            // Draw bidirectional attention arrows
                            if (selectedMask >= 0) {
                                var selTp = tokenPositions[selectedMask];
                                var selCx = selTp.x + selTp.w / 2;
                                var arrowY = tokY + 36;

                                for (var a = 0; a < sent.display.length; a++) {
                                    if (a === selectedMask) continue;
                                    var tp = tokenPositions[a];
                                    var aCx = tp.x + tp.w / 2;

                                    // curved arrow from each token to selected mask
                                    ctx.strokeStyle = viz.colors.teal + '55';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    var cpY = arrowY + 12 + Math.abs(a - selectedMask) * 4;
                                    ctx.moveTo(aCx, tokY + 24);
                                    ctx.quadraticCurveTo((aCx + selCx) / 2, cpY, selCx, tokY + 24);
                                    ctx.stroke();
                                }

                                ctx.fillStyle = viz.colors.teal;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('bidirectional context', W / 2, arrowY + 25);
                            }

                            // Draw predictions for selected mask
                            var preds = sent.predictions[selectedMask];
                            if (preds) {
                                var predY = 170;
                                var barMaxW = 300;
                                var barH = 28;
                                var barGap = 8;
                                var barX = W / 2 - 100;

                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 13px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('Top-5 Predictions for position ' + selectedMask + ':', barX - 60, predY - 8);

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.fillText('(ground truth: "' + sent.original[selectedMask] + '")', barX + 200, predY - 8);

                                var predColors = [viz.colors.blue, viz.colors.teal, viz.colors.green, viz.colors.purple, viz.colors.yellow];
                                for (var p = 0; p < preds.length; p++) {
                                    var py = predY + 12 + p * (barH + barGap);
                                    var pw = preds[p].prob * barMaxW;
                                    var pc = predColors[p];
                                    var isCorrect = preds[p].token === sent.original[selectedMask];

                                    // Bar background
                                    ctx.fillStyle = pc + '22';
                                    ctx.fillRect(barX + 70, py, barMaxW, barH);

                                    // Bar fill
                                    ctx.fillStyle = pc + (isCorrect ? 'cc' : '88');
                                    ctx.fillRect(barX + 70, py, pw, barH);

                                    if (isCorrect) {
                                        ctx.strokeStyle = viz.colors.green;
                                        ctx.lineWidth = 2;
                                        ctx.strokeRect(barX + 70, py, barMaxW, barH);
                                    }

                                    // Token label
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = '13px monospace';
                                    ctx.textAlign = 'right';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(preds[p].token, barX + 60, py + barH / 2);

                                    // Probability
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '11px -apple-system,sans-serif';
                                    ctx.textAlign = 'left';
                                    ctx.fillText((preds[p].prob * 100).toFixed(0) + '%', barX + 78 + pw, py + barH / 2);

                                    if (isCorrect) {
                                        ctx.fillStyle = viz.colors.green;
                                        ctx.fillText('\u2714 correct', barX + barMaxW + 85, py + barH / 2);
                                    }
                                }
                            }

                            // Sentence selector indicator
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Sentence ' + (currentSent + 1) + ' / ' + sentences.length, W / 2, H - 15);
                        }

                        viz.canvas.addEventListener('click', function(e) {
                            var rect = viz.canvas.getBoundingClientRect();
                            var mx = e.clientX - rect.left;
                            var my = e.clientY - rect.top;
                            for (var i = 0; i < tokenPositions.length; i++) {
                                var tp = tokenPositions[i];
                                if (mx >= tp.x && mx <= tp.x + tp.w && my >= tp.y && my <= tp.y + tp.h) {
                                    if (sentences[currentSent].display[tp.idx] === '[MASK]') {
                                        selectedMask = tp.idx;
                                        draw();
                                        return;
                                    }
                                }
                            }
                        });

                        VizEngine.createButton(controls, 'Prev Sentence', function() {
                            currentSent = (currentSent - 1 + sentences.length) % sentences.length;
                            selectedMask = -1;
                            draw();
                        });
                        VizEngine.createButton(controls, 'Next Sentence', function() {
                            currentSent = (currentSent + 1) % sentences.length;
                            selectedMask = -1;
                            draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'BERT masks 15% of tokens during pretraining. If the training sequence length is \\(T = 512\\), approximately how many tokens per sequence contribute to the MLM loss? How does this compare to GPT, where every token contributes?',
                    hint: 'In GPT, all \\(T\\) positions contribute to the loss. In BERT, only the masked positions do.',
                    solution: 'BERT: \\(0.15 \\times 512 \\approx 77\\) tokens per sequence contribute to the loss. GPT: all 512 tokens contribute. BERT is roughly \\(512/77 \\approx 6.7\\times\\) less sample-efficient per token of training data, since it only receives gradients from 15% of positions. This is one reason BERT requires more training data and compute to reach comparable performance. RoBERTa partially addresses this by training much longer.'
                },
                {
                    question: 'Explain why BERT cannot be used for autoregressive text generation. What property of a joint probability distribution does MLM fail to define?',
                    hint: 'Autoregressive generation requires \\(P(w_t \\mid w_{&lt;t})\\), i.e., a proper factorization of the joint distribution.',
                    solution: 'BERT models \\(P(w_t \\mid w_{\\setminus t})\\), the conditional probability of one token given all others. These conditionals do not, in general, define a consistent joint distribution \\(P(w_1, \\ldots, w_T)\\). To generate text autoregressively, we need a factorized joint: \\(P(w_1)P(w_2 \\mid w_1) \\cdots P(w_T \\mid w_{&lt;T})\\). BERT provides no mechanism for this ordered factorization. Attempting iterative unmasking (predict one mask, fill it in, predict the next) does not correspond to sampling from any well-defined distribution, because each step conditions on previously sampled (potentially incorrect) tokens, and the "joint" is order-dependent in an inconsistent way.'
                },
                {
                    question: 'Derive the expected gradient magnitude for a BERT model where 15% of tokens are masked, compared to a hypothetical model that masks 50%. Under what conditions does higher masking rate improve or hurt learning?',
                    hint: 'Consider the bias-variance trade-off of the gradient estimate. More masks per sample means more gradient signal, but also more corruption of context.',
                    solution: 'Let the masking rate be \\(p\\). The loss for a single sequence is \\(\\mathcal{L} = -\\frac{1}{pT} \\sum_{i \\in \\mathcal{M}} \\log P(w_i \\mid \\tilde{w})\\). With higher \\(p\\), more terms contribute to the gradient (lower variance per sequence). However, higher \\(p\\) also corrupts more of the input context, making each individual prediction harder and potentially biasing the learned representations. At \\(p = 1\\) (all tokens masked), the model has no context at all, and each prediction reduces to the unigram prior. The 15% rate is empirically chosen as a sweet spot: enough signal per sequence, while preserving sufficient context (85% of tokens) for meaningful bidirectional prediction.'
                }
            ]
        },

        // ======================== Section 4 ========================
        {
            id: 'sec19-4-pretraining-finetuning',
            title: 'Pretraining to Finetuning',
            content: `
<h2>19.4 Pretraining to Finetuning</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; The Two-Phase Paradigm</div><div class="env-body">
<p>The pretraining-finetuning paradigm is the NLP analogue of what ImageNet pretraining did for computer vision. In Phase 1 (pretraining), a large model learns general language understanding from massive unsupervised text. In Phase 2 (finetuning), the pretrained model is adapted to a specific downstream task using a small labeled dataset. The key insight: features learned during pretraining (syntax, semantics, world knowledge, reasoning patterns) transfer remarkably well, so finetuning requires far less data and compute than training from scratch.</p>
</div></div>

<h3>Transfer Learning in NLP</h3>

<p>Before BERT and GPT, NLP systems used task-specific architectures with word embeddings (Word2Vec, GloVe) as the only pretrained component. These embeddings capture word similarity but not contextual meaning ("bank" has the same embedding whether it refers to a river bank or a financial institution).</p>

<div class="env-block definition"><div class="env-title">Definition 19.4.1 &mdash; Contextual Embeddings</div><div class="env-body">
<p>A <strong>contextual embedding</strong> produces a representation \\(\\mathbf{h}^{(i)} = f_\\theta(w_i, w_{\\setminus i})\\) that depends on the entire input context, not just the identity of \\(w_i\\). In a pretrained Transformer, the output at position \\(i\\) of layer \\(\\ell\\) encodes increasingly abstract contextual features:</p>
<ul>
<li>Lower layers: syntax, part-of-speech, local dependencies.</li>
<li>Middle layers: semantic roles, coreference.</li>
<li>Upper layers: task-relevant features, long-range reasoning.</li>
</ul>
<p>This is the key advance over static embeddings: the same word receives different representations in different contexts.</p>
</div></div>

<h3>Task-Specific Heads</h3>

<p>To adapt a pretrained model to a downstream task, we add a lightweight <strong>task head</strong> on top of the pretrained representations:</p>

<div class="env-block definition"><div class="env-title">Definition 19.4.2 &mdash; Common Task Heads</div><div class="env-body">
<ul>
<li><strong>Sequence classification</strong> (sentiment, topic): Feed the [CLS] token representation to a linear classifier: \\(\\hat{y} = \\text{softmax}(\\mathbf{W}_c \\mathbf{h}^{[\\text{CLS}]} + \\mathbf{b}_c)\\).</li>
<li><strong>Token classification</strong> (NER, POS tagging): Apply a classifier to each token's representation: \\(\\hat{y}_i = \\text{softmax}(\\mathbf{W}_t \\mathbf{h}^{(i)} + \\mathbf{b}_t)\\).</li>
<li><strong>Span extraction</strong> (question answering): Predict start and end positions using two linear heads: \\(P(\\text{start} = i) = \\text{softmax}(\\mathbf{w}_s^\\top \\mathbf{h}^{(i)})\\), \\(P(\\text{end} = j) = \\text{softmax}(\\mathbf{w}_e^\\top \\mathbf{h}^{(j)})\\).</li>
<li><strong>Text generation</strong> (GPT-style): The language model head is already built in; finetuning adjusts the model's generation behavior toward the desired task.</li>
</ul>
</div></div>

<h3>Finetuning Strategies</h3>

<div class="env-block definition"><div class="env-title">Definition 19.4.3 &mdash; Finetuning vs. Feature Extraction</div><div class="env-body">
<p>Two approaches for using pretrained models:</p>
<ol>
<li><strong>Full finetuning:</strong> Update all parameters (pretrained backbone + task head) with a small learning rate (e.g., \\(2 \\times 10^{-5}\\)). Most effective but risks catastrophic forgetting if the task dataset is tiny.</li>
<li><strong>Feature extraction (frozen):</strong> Freeze the pretrained parameters and only train the task head. The pretrained model acts as a fixed feature extractor. Less expressive but avoids forgetting and requires fewer resources.</li>
</ol>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Learning Rate Matters</div><div class="env-body">
<p>Finetuning uses a much smaller learning rate than pretraining (typically 100x smaller). If the learning rate is too large, the model quickly "forgets" its pretrained knowledge (catastrophic forgetting). If too small, the model fails to adapt to the task. Common practice: use a linear warmup over the first 6&ndash;10% of finetuning steps, then decay linearly or with cosine scheduling.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 19.4.4 &mdash; BERT on GLUE Benchmark</div><div class="env-body">
<p>BERT-Large achieved an average GLUE score of 80.5 (vs. 66.5 for pre-BERT methods). On some individual tasks:</p>
<ul>
<li><strong>MNLI</strong> (natural language inference): 86.7% accuracy</li>
<li><strong>QQP</strong> (paraphrase detection): 72.1% F1</li>
<li><strong>SST-2</strong> (sentiment): 94.9% accuracy</li>
</ul>
<p>Finetuning BERT on SST-2 requires only 67k labeled examples and roughly 1 hour on a single GPU, compared to months of pretraining on 3.3B tokens across 64 TPUs.</p>
</div></div>

<div class="env-block warning"><div class="env-title">Parameter-Efficient Finetuning</div><div class="env-body">
<p>As models grow to billions of parameters, full finetuning becomes prohibitively expensive (one copy of all parameters per task). <strong>Parameter-efficient finetuning (PEFT)</strong> methods update only a small number of additional parameters:</p>
<ul>
<li><strong>LoRA</strong> (Low-Rank Adaptation): Inject low-rank updates \\(\\Delta \\mathbf{W} = \\mathbf{B}\\mathbf{A}\\) where \\(\\mathbf{A} \\in \\mathbb{R}^{d \\times r}\\), \\(\\mathbf{B} \\in \\mathbb{R}^{r \\times d}\\), \\(r \\ll d\\). Only \\(\\mathbf{A}\\) and \\(\\mathbf{B}\\) are trained.</li>
<li><strong>Adapters:</strong> Insert small bottleneck layers between Transformer blocks.</li>
<li><strong>Prefix tuning:</strong> Prepend learnable "virtual tokens" to the input.</li>
</ul>
<p>These methods achieve 90&ndash;99% of full finetuning performance while updating less than 1% of total parameters.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-finetune-pipeline"></div>
`,
            visualizations: [
                {
                    id: 'viz-finetune-pipeline',
                    title: 'Pretraining to Finetuning Pipeline',
                    description: 'The pretrained Transformer backbone (blue) provides contextual representations. A task-specific head (orange) is added on top. Toggle between "Full Finetuning" (all layers updated) and "Frozen" (only head updated) to see which parameters receive gradients.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var mode = 'full'; // 'full', 'frozen', 'lora'
                        var animTime = 0;
                        var animId = null;

                        function draw(t) {
                            animTime = t || 0;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Pretraining \u2192 Finetuning Pipeline', W / 2, 22);

                            var centerX = W / 2;
                            var layerW = 280;
                            var layerH = 34;
                            var gap = 8;
                            var numLayers = 6;
                            var startY = 340;

                            // Draw input tokens
                            var inputTokens = ['[CLS]', 'This', 'movie', 'is', 'great', '[SEP]'];
                            ctx.font = '11px monospace';
                            ctx.textAlign = 'center';
                            var tokSpacing = 46;
                            var tokStartX = centerX - (inputTokens.length - 1) * tokSpacing / 2;
                            for (var it = 0; it < inputTokens.length; it++) {
                                var tx = tokStartX + it * tokSpacing;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(inputTokens[it], tx, startY + 35);
                            }

                            // Input arrow
                            ctx.strokeStyle = viz.colors.text;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(centerX, startY + 22);
                            ctx.lineTo(centerX, startY + 6);
                            ctx.stroke();
                            ctx.beginPath();
                            ctx.moveTo(centerX, startY + 6);
                            ctx.lineTo(centerX - 4, startY + 12);
                            ctx.moveTo(centerX, startY + 6);
                            ctx.lineTo(centerX + 4, startY + 12);
                            ctx.stroke();

                            // Draw Transformer layers (backbone)
                            for (var l = 0; l < numLayers; l++) {
                                var ly = startY - (l + 1) * (layerH + gap);
                                var lx = centerX - layerW / 2;

                                var isUpdated = (mode === 'full') || (mode === 'lora');
                                var pulse = isUpdated ? Math.sin(animTime / 600 + l * 0.5) * 0.15 + 0.85 : 0.5;

                                // Layer box
                                if (mode === 'frozen') {
                                    ctx.fillStyle = viz.colors.blue + '44';
                                    ctx.strokeStyle = viz.colors.blue + '66';
                                } else if (mode === 'lora') {
                                    ctx.fillStyle = 'rgba(88,166,255,' + (0.2 + 0.1 * pulse) + ')';
                                    ctx.strokeStyle = viz.colors.blue;
                                } else {
                                    ctx.fillStyle = 'rgba(88,166,255,' + (0.15 + 0.2 * pulse) + ')';
                                    ctx.strokeStyle = viz.colors.blue;
                                }
                                ctx.lineWidth = isUpdated && mode === 'full' ? 2 : 1;
                                ctx.fillRect(lx, ly, layerW, layerH);
                                ctx.strokeRect(lx, ly, layerW, layerH);

                                // Layer label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('Transformer Block ' + (l + 1), centerX, ly + layerH / 2 + 4);

                                // LoRA indicator
                                if (mode === 'lora') {
                                    var loraX = lx + layerW + 8;
                                    ctx.fillStyle = viz.colors.purple + 'aa';
                                    ctx.fillRect(loraX, ly + 4, 52, layerH - 8);
                                    ctx.strokeStyle = viz.colors.purple;
                                    ctx.lineWidth = 1.5;
                                    ctx.strokeRect(loraX, ly + 4, 52, layerH - 8);
                                    ctx.fillStyle = viz.colors.white;
                                    ctx.font = '9px -apple-system,sans-serif';
                                    ctx.fillText('LoRA', loraX + 26, ly + layerH / 2 + 3);
                                }

                                // Gradient flow indicator
                                if ((mode === 'full' || mode === 'lora') && isUpdated) {
                                    var gradAlpha = (Math.sin(animTime / 400 - l * 0.8) + 1) / 2;
                                    ctx.fillStyle = 'rgba(63,185,80,' + (gradAlpha * 0.6) + ')';
                                    ctx.fillRect(lx + 2, ly + 2, 4, layerH - 4);
                                }

                                // Frozen lock icon
                                if (mode === 'frozen') {
                                    ctx.fillStyle = viz.colors.text;
                                    ctx.font = '12px -apple-system,sans-serif';
                                    ctx.textAlign = 'left';
                                    ctx.fillText('\uD83D\uDD12', lx + layerW + 8, ly + layerH / 2 + 4);
                                }

                                // Connection arrows
                                if (l < numLayers - 1) {
                                    ctx.strokeStyle = viz.colors.text + '66';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.moveTo(centerX, ly);
                                    ctx.lineTo(centerX, ly - gap);
                                    ctx.stroke();
                                }
                            }

                            // Task head
                            var headY = startY - (numLayers + 1) * (layerH + gap);
                            var headW = 200;
                            var headX = centerX - headW / 2;
                            var headPulse = Math.sin(animTime / 500) * 0.2 + 0.8;

                            ctx.fillStyle = 'rgba(240,136,62,' + (0.3 + 0.3 * headPulse) + ')';
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            ctx.fillRect(headX, headY, headW, layerH);
                            ctx.strokeRect(headX, headY, headW, layerH);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Task Head (classifier)', centerX, headY + layerH / 2 + 4);

                            // Arrow from last layer to head
                            var lastLayerY = startY - numLayers * (layerH + gap);
                            ctx.strokeStyle = viz.colors.text + '66';
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(centerX, lastLayerY);
                            ctx.lineTo(centerX, headY + layerH);
                            ctx.stroke();

                            // Output
                            ctx.fillStyle = viz.colors.green;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Positive \u2714', centerX, headY - 14);

                            // Legend
                            var legX = 18;
                            var legY = 50;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(legX, legY, 14, 14);
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText('Pretrained backbone', legX + 20, legY + 11);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(legX, legY + 24, 14, 14);
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText('Task-specific head', legX + 20, legY + 35);

                            if (mode === 'full') {
                                ctx.fillStyle = viz.colors.green;
                                ctx.fillRect(legX, legY + 48, 14, 14);
                                ctx.fillStyle = viz.colors.white;
                                ctx.fillText('Gradient flow (all params)', legX + 20, legY + 59);
                            } else if (mode === 'frozen') {
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText('\uD83D\uDD12 = frozen (no gradients)', legX, legY + 59);
                            } else if (mode === 'lora') {
                                ctx.fillStyle = viz.colors.purple;
                                ctx.fillRect(legX, legY + 48, 14, 14);
                                ctx.fillStyle = viz.colors.white;
                                ctx.fillText('LoRA adapters (trained)', legX + 20, legY + 59);
                            }

                            // Mode label
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            var modeLabels = { full: 'Full Finetuning', frozen: 'Feature Extraction (Frozen)', lora: 'LoRA (Parameter-Efficient)' };
                            ctx.fillText('Mode: ' + modeLabels[mode], W - 20, 50);
                        }

                        function animate(t) {
                            draw(t);
                            animId = requestAnimationFrame(animate);
                        }

                        VizEngine.createButton(controls, 'Full Finetuning', function() { mode = 'full'; });
                        VizEngine.createButton(controls, 'Frozen', function() { mode = 'frozen'; });
                        VizEngine.createButton(controls, 'LoRA', function() { mode = 'lora'; });

                        animId = requestAnimationFrame(animate);

                        return { stopAnimation: function() { if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'BERT-Base has 110M parameters. A classification task head for 3-class sentiment adds a linear layer \\(\\mathbf{W}_c \\in \\mathbb{R}^{768 \\times 3}\\) and bias \\(\\mathbf{b}_c \\in \\mathbb{R}^3\\). What fraction of total parameters does the task head contribute?',
                    hint: 'Count the parameters in \\(\\mathbf{W}_c\\) and \\(\\mathbf{b}_c\\), then divide by total.',
                    solution: 'Task head parameters: \\(768 \\times 3 + 3 = 2{,}307\\). Fraction: \\(2{,}307 / 110{,}000{,}000 \\approx 0.002\\%\\). The task head is vanishingly small relative to the backbone. This explains why finetuning is so data-efficient: the pretrained backbone already provides rich representations, and the task head merely learns a simple linear decision boundary in this high-dimensional space.'
                },
                {
                    question: 'In LoRA with rank \\(r = 8\\) and model dimension \\(d = 1024\\), how many trainable parameters per weight matrix? Compare to full finetuning.',
                    hint: 'LoRA decomposes the update as \\(\\Delta \\mathbf{W} = \\mathbf{B}\\mathbf{A}\\) where \\(\\mathbf{A} \\in \\mathbb{R}^{d \\times r}\\), \\(\\mathbf{B} \\in \\mathbb{R}^{r \\times d}\\).',
                    solution: 'LoRA trainable parameters per matrix: \\(d \\times r + r \\times d = 2 \\times 1024 \\times 8 = 16{,}384\\). Full weight matrix: \\(d \\times d = 1{,}048{,}576\\). Ratio: \\(16{,}384 / 1{,}048{,}576 = 1.56\\%\\). LoRA reduces trainable parameters by roughly 64x per matrix, while the low-rank structure constrains the update to lie in a low-dimensional subspace, acting as an implicit regularizer.'
                },
                {
                    question: 'Explain the concept of catastrophic forgetting in the context of finetuning. Why does a smaller learning rate mitigate this problem?',
                    hint: 'Think about how gradient updates modify the pretrained weights and the relationship between step size and the distance from the pretrained initialization.',
                    solution: 'Catastrophic forgetting occurs when finetuning updates overwrite the general knowledge stored in pretrained weights. With a large learning rate, each gradient step moves the parameters far from their pretrained values, rapidly destroying features that are useful for broad language understanding in favor of narrow task-specific patterns. A small learning rate constrains the parameter trajectory to stay near the pretrained initialization, preserving general knowledge while making incremental task-specific adjustments. Formally, if \\(\\theta_0\\) are pretrained weights and \\(\\theta_t\\) are weights after \\(t\\) finetuning steps, we want \\(\\|\\theta_t - \\theta_0\\|\\) to remain small, which is guaranteed when the learning rate \\(\\eta\\) is small since \\(\\|\\theta_{t+1} - \\theta_t\\| = \\eta \\|\\nabla \\mathcal{L}\\|\\).'
                }
            ]
        },

        // ======================== Section 5 ========================
        {
            id: 'sec19-5-scaling-laws',
            title: 'Scaling Laws & Emergent Abilities',
            content: `
<h2>19.5 Scaling Laws &amp; Emergent Abilities</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Bigger is Predictably Better</div><div class="env-body">
<p>One of the most remarkable empirical discoveries in deep learning is that language model performance follows precise mathematical relationships with scale. More parameters, more data, and more compute each improve loss in a predictable, power-law fashion. Even more striking: beyond certain scale thresholds, models develop capabilities that appear to emerge discontinuously, solving tasks they were never explicitly trained on.</p>
</div></div>

<h3>Neural Scaling Laws</h3>

<p>Kaplan et al. (2020) observed that the test loss of a Transformer language model follows a power law in the number of parameters \\(N\\), dataset size \\(D\\), and compute budget \\(C\\):</p>

<div class="env-block theorem"><div class="env-title">Theorem 19.5.1 &mdash; Scaling Laws (Kaplan et al., 2020)</div><div class="env-body">
<p>For Transformer language models, the cross-entropy loss \\(L\\) on held-out text follows:</p>
\\[
L(N) \\approx \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}, \\quad L(D) \\approx \\left(\\frac{D_c}{D}\\right)^{\\alpha_D}, \\quad L(C) \\approx \\left(\\frac{C_c}{C}\\right)^{\\alpha_C},
\\]
<p>where \\(\\alpha_N \\approx 0.076\\), \\(\\alpha_D \\approx 0.095\\), \\(\\alpha_C \\approx 0.050\\) and \\(N_c, D_c, C_c\\) are constants. These power laws hold over many orders of magnitude (from 10M to 10B+ parameters).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Log-Log Linearity</div><div class="env-body">
<p>A power law \\(L = a N^{-\\alpha}\\) becomes a linear relationship on a log-log plot: \\(\\log L = \\log a - \\alpha \\log N\\). The empirical observation of straight lines on log-log plots (across 7+ orders of magnitude of compute) is what gives these scaling laws their predictive power. You can estimate the performance of a 100B-parameter model from experiments with models 1000x smaller.</p>
</div></div>

<h3>Compute-Optimal Scaling: Chinchilla</h3>

<p>The original Kaplan scaling laws suggested that for a fixed compute budget, one should train the largest possible model and stop early. Hoffmann et al. (2022) challenged this with the <strong>Chinchilla</strong> result.</p>

<div class="env-block theorem"><div class="env-title">Theorem 19.5.2 &mdash; Chinchilla Optimal Scaling</div><div class="env-body">
<p>For a fixed compute budget \\(C \\propto 6ND\\) (where \\(N\\) is parameters and \\(D\\) is training tokens), the compute-optimal allocation scales both equally:</p>
\\[
N_{\\text{opt}} \\propto C^{0.5}, \\quad D_{\\text{opt}} \\propto C^{0.5}.
\\]
<p>In practice, this means the optimal number of training tokens is approximately \\(D \\approx 20N\\). A 70B-parameter model should be trained on about 1.4 trillion tokens.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 19.5.3 &mdash; Chinchilla vs. Gopher</div><div class="env-body">
<p>DeepMind's Gopher (280B parameters, 300B tokens) was trained with the same compute budget as Chinchilla (70B parameters, 1.4T tokens). Despite being 4x smaller, Chinchilla <em>outperformed</em> Gopher on nearly every benchmark. Gopher was severely "over-parameterized and under-trained" relative to the compute-optimal frontier. This result fundamentally changed how the field allocates compute.</p>
</div></div>

<div class="env-block warning"><div class="env-title">Inference Cost Caveat</div><div class="env-body">
<p>Chinchilla-optimal scaling minimizes training loss for a fixed training budget. But <em>inference</em> cost scales with \\(N\\), not \\(D\\). If the model will serve billions of queries, it may be more cost-effective to train a smaller model for longer (beyond the Chinchilla-optimal \\(D\\)), accepting higher training cost for cheaper inference. LLaMA (Touvron et al., 2023) exemplifies this: a 7B model trained on 1T tokens (\\(D/N \\approx 143\\), well above the Chinchilla ratio of 20).</p>
</div></div>

<h3>Emergent Abilities</h3>

<p>Perhaps the most scientifically provocative finding in the scaling era is the phenomenon of <strong>emergent abilities</strong> (Wei et al., 2022): capabilities that are absent in smaller models but appear at larger scales.</p>

<div class="env-block definition"><div class="env-title">Definition 19.5.4 &mdash; Emergent Ability</div><div class="env-body">
<p>An ability is <strong>emergent</strong> if it is not present in smaller models but appears in larger models. Formally, an ability is emergent if its performance is near random for models below some threshold size \\(N^*\\) and then rises sharply above \\(N^*\\). Examples include:</p>
<ul>
<li><strong>Multi-step arithmetic:</strong> 3-digit addition appears at around 10B parameters.</li>
<li><strong>Chain-of-thought reasoning:</strong> Effective only above approximately 100B parameters.</li>
<li><strong>Few-shot in-context learning:</strong> Qualitative improvement between GPT-2 (1.5B) and GPT-3 (175B).</li>
</ul>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Are Emergent Abilities a Mirage?</div><div class="env-body">
<p>Schaeffer et al. (2023) argue that the appearance of emergence depends on the metric used. When performance is measured with nonlinear or discontinuous metrics (e.g., exact-match accuracy), the transition looks sharp. When measured with continuous metrics (e.g., per-token log-probability), performance improves smoothly and predictably. This does not diminish the practical significance of emergence: a model that gets 0% of multi-digit additions exactly right is useless for that task, while one that gets 90% right is highly useful, even if the underlying per-token probabilities improved smoothly.</p>
</div></div>

<h3>In-Context Learning</h3>

<div class="env-block definition"><div class="env-title">Definition 19.5.5 &mdash; In-Context Learning (ICL)</div><div class="env-body">
<p><strong>In-context learning</strong> refers to a model's ability to perform new tasks at inference time by conditioning on a few input-output examples in the prompt, without any gradient updates:</p>
\\[
\\text{Prompt: } (x_1, y_1), (x_2, y_2), \\ldots, (x_k, y_k), (x_{\\text{test}}, \\;?) \\quad \\Rightarrow \\quad \\hat{y}_{\\text{test}}
\\]
<p>This is called \\(k\\)-shot learning. At \\(k=0\\) (zero-shot), the model relies solely on the task description.</p>
</div></div>

<p>In-context learning is remarkable because the model adapts to a new task <em>without weight updates</em>. The mechanism is still not fully understood, but recent theoretical work suggests that Transformer attention layers can implicitly implement gradient descent on in-context examples (Akyurek et al., 2023; von Oswald et al., 2023).</p>

<div class="viz-placeholder" data-viz="viz-scaling-laws"></div>
`,
            visualizations: [
                {
                    id: 'viz-scaling-laws',
                    title: 'Scaling Laws & Chinchilla Frontier',
                    description: 'Log-log plot showing how test loss decreases with model parameters. The Chinchilla-optimal frontier shows the best loss achievable for each compute budget. Drag the compute budget slider to see the optimal parameter/data allocation.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        // Plot region
                        var plotL = 90, plotR = W - 40, plotT = 60, plotB = H - 60;
                        var plotW = plotR - plotL, plotH = plotB - plotT;

                        // Axes: log10(N) from 7 to 12 (10M to 1T params)
                        var xMin = 7, xMax = 12;
                        // Loss from 1.5 to 4.5
                        var yMin = 1.5, yMax = 4.5;

                        var computeLog = 21; // log10(FLOPs), range 18 to 26

                        function toPlotX(logN) { return plotL + (logN - xMin) / (xMax - xMin) * plotW; }
                        function toPlotY(loss) { return plotB - (loss - yMin) / (yMax - yMin) * plotH; }

                        // Scaling law: L(N) = a * N^(-alpha) for fixed data
                        // L(N, D) = (Nc/N)^alphaN + (Dc/D)^alphaD
                        function loss(logN, logD) {
                            var alphaN = 0.076, alphaD = 0.095;
                            var logNc = 8.8, logDc = 10.7; // characteristic scales
                            var lN = Math.pow(10, (logNc - logN) * alphaN);
                            var lD = Math.pow(10, (logDc - logD) * alphaD);
                            return 1.5 + lN + lD;
                        }

                        // Chinchilla optimal: D = 20*N, C = 6*N*D
                        // log10(C) = log10(6) + log10(N) + log10(D) = log10(6) + log10(N) + log10(20N) = log10(120) + 2*log10(N)
                        // So log10(N_opt) = (log10(C) - log10(120)) / 2
                        function chinchillaN(logC) {
                            return (logC - Math.log10(120)) / 2;
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Neural Scaling Laws (log-log)', W / 2, 25);

                            // Plot background
                            ctx.fillStyle = '#0f0f28';
                            ctx.fillRect(plotL, plotT, plotW, plotH);

                            // Grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            for (var gx = xMin; gx <= xMax; gx++) {
                                var px = toPlotX(gx);
                                ctx.beginPath(); ctx.moveTo(px, plotT); ctx.lineTo(px, plotB); ctx.stroke();
                            }
                            for (var gy = yMin; gy <= yMax; gy += 0.5) {
                                var py = toPlotY(gy);
                                ctx.beginPath(); ctx.moveTo(plotL, py); ctx.lineTo(plotR, py); ctx.stroke();
                            }

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            var labels = ['10M', '100M', '1B', '10B', '100B', '1T'];
                            for (var lbl = 0; lbl < labels.length; lbl++) {
                                ctx.fillText(labels[lbl], toPlotX(xMin + lbl), plotB + 8);
                            }
                            ctx.fillText('Parameters (N)', plotL + plotW / 2, plotB + 28);

                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var ly = yMin; ly <= yMax; ly += 0.5) {
                                ctx.fillText(ly.toFixed(1), plotL - 8, toPlotY(ly));
                            }

                            ctx.save();
                            ctx.translate(16, plotT + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Test Loss (cross-entropy)', 0, 0);
                            ctx.restore();

                            // Draw scaling curves for different data amounts
                            var dataLevels = [
                                { logD: 9, label: '1B tokens', color: viz.colors.purple + '99' },
                                { logD: 10, label: '10B tokens', color: viz.colors.blue + '99' },
                                { logD: 11, label: '100B tokens', color: viz.colors.teal + '99' },
                                { logD: 12, label: '1T tokens', color: viz.colors.green + '99' }
                            ];

                            for (var dl = 0; dl < dataLevels.length; dl++) {
                                var dInfo = dataLevels[dl];
                                ctx.strokeStyle = dInfo.color;
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                var first = true;
                                for (var lnx = xMin; lnx <= xMax; lnx += 0.05) {
                                    var lval = loss(lnx, dInfo.logD);
                                    if (lval < yMin || lval > yMax) { first = true; continue; }
                                    var sx = toPlotX(lnx);
                                    var sy = toPlotY(lval);
                                    if (first) { ctx.moveTo(sx, sy); first = false; }
                                    else ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();

                                // Label at end
                                var endLoss = loss(xMax, dInfo.logD);
                                if (endLoss >= yMin && endLoss <= yMax) {
                                    ctx.fillStyle = dInfo.color;
                                    ctx.font = '10px -apple-system,sans-serif';
                                    ctx.textAlign = 'left';
                                    ctx.textBaseline = 'middle';
                                }
                            }

                            // Legend for data levels
                            var legX = plotL + 10;
                            var legY = plotT + 10;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            for (var d2 = 0; d2 < dataLevels.length; d2++) {
                                ctx.strokeStyle = dataLevels[d2].color;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.moveTo(legX, legY + d2 * 16 + 5);
                                ctx.lineTo(legX + 20, legY + d2 * 16 + 5);
                                ctx.stroke();
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(dataLevels[d2].label, legX + 26, legY + d2 * 16 + 8);
                            }

                            // Chinchilla optimal frontier
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            var cFirst = true;
                            for (var logC = 18; logC <= 26; logC += 0.1) {
                                var optLogN = chinchillaN(logC);
                                var optLogD = Math.log10(20) + optLogN;
                                if (optLogN < xMin || optLogN > xMax) continue;
                                var cLoss = loss(optLogN, optLogD);
                                if (cLoss < yMin || cLoss > yMax) continue;
                                var cx2 = toPlotX(optLogN);
                                var cy2 = toPlotY(cLoss);
                                if (cFirst) { ctx.moveTo(cx2, cy2); cFirst = false; }
                                else ctx.lineTo(cx2, cy2);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Chinchilla label
                            ctx.fillStyle = viz.colors.orange;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            var chiLabelN = chinchillaN(computeLog);
                            var chiLabelD = Math.log10(20) + chiLabelN;
                            if (chiLabelN >= xMin && chiLabelN <= xMax) {
                                var chiLoss = loss(chiLabelN, chiLabelD);
                                if (chiLoss >= yMin && chiLoss <= yMax) {
                                    ctx.fillText('Chinchilla', toPlotX(chiLabelN) + 8, toPlotY(chiLoss) - 10);
                                    ctx.fillText('optimal', toPlotX(chiLabelN) + 8, toPlotY(chiLoss) + 4);

                                    // Draw marker at optimal point
                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.beginPath();
                                    ctx.arc(toPlotX(chiLabelN), toPlotY(chiLoss), 6, 0, Math.PI * 2);
                                    ctx.fill();
                                    ctx.strokeStyle = viz.colors.white;
                                    ctx.lineWidth = 1.5;
                                    ctx.stroke();
                                }
                            }

                            // Notable models
                            var models = [
                                { name: 'GPT-2', logN: Math.log10(1.5e9), logD: Math.log10(40e9) },
                                { name: 'GPT-3', logN: Math.log10(175e9), logD: Math.log10(300e9) },
                                { name: 'Chinchilla', logN: Math.log10(70e9), logD: Math.log10(1.4e12) },
                                { name: 'LLaMA-7B', logN: Math.log10(7e9), logD: Math.log10(1e12) }
                            ];

                            for (var m = 0; m < models.length; m++) {
                                var mod = models[m];
                                var mLoss = loss(mod.logN, mod.logD);
                                if (mod.logN < xMin || mod.logN > xMax || mLoss < yMin || mLoss > yMax) continue;
                                var mx2 = toPlotX(mod.logN);
                                var my2 = toPlotY(mLoss);

                                ctx.fillStyle = viz.colors.yellow;
                                ctx.beginPath();
                                ctx.moveTo(mx2, my2 - 7);
                                ctx.lineTo(mx2 + 5, my2 + 3);
                                ctx.lineTo(mx2 - 5, my2 + 3);
                                ctx.closePath();
                                ctx.fill();

                                ctx.fillStyle = viz.colors.yellow;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(mod.name, mx2, my2 - 12);
                            }

                            // Compute budget indicator
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Compute budget: 10^' + computeLog.toFixed(0) + ' FLOPs', W / 2, H - 10);

                            var optN = chinchillaN(computeLog);
                            var optD = Math.log10(20) + optN;
                            if (optN >= xMin && optN <= xMax) {
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('Optimal: N=' + Math.pow(10, optN).toExponential(1) + ', D=' + Math.pow(10, optD).toExponential(1) + ' tokens', W / 2, H - 28);
                            }
                        }

                        VizEngine.createSlider(controls, 'log\u2081\u2080(Compute)', 18, 26, 21, 0.5, function(v) {
                            computeLog = v;
                            draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Using the Chinchilla scaling rule \\(D_{\\text{opt}} \\approx 20N\\) and the compute estimate \\(C \\approx 6ND\\), determine the optimal model size and dataset size for a compute budget of \\(10^{24}\\) FLOPs.',
                    hint: 'Substitute \\(D = 20N\\) into \\(C = 6ND\\) to get \\(C = 120N^2\\), then solve for \\(N\\).',
                    solution: '\\(C = 6N \\cdot 20N = 120N^2\\). So \\(N = \\sqrt{C/120} = \\sqrt{10^{24}/120} \\approx \\sqrt{8.33 \\times 10^{21}} \\approx 2.9 \\times 10^{10}\\), i.e., about 29B parameters. Then \\(D = 20N \\approx 5.8 \\times 10^{11}\\), i.e., about 580B tokens. This is roughly the scale of Chinchilla (70B params, 1.4T tokens had about \\(5.9 \\times 10^{23}\\) FLOPs).'
                },
                {
                    question: 'Why might training a model well beyond the Chinchilla-optimal data amount (e.g., LLaMA with \\(D/N \\approx 143\\) instead of 20) be a rational engineering decision?',
                    hint: 'Consider the total cost of operating a model, which includes both training and serving many inference queries.',
                    solution: 'Chinchilla-optimal scaling minimizes training loss per training FLOP, but ignores inference cost. Each inference query costs \\(O(N)\\) FLOPs. A model that serves billions of queries incurs total inference cost \\(\\gg\\) training cost. By training a smaller model on more data (even past the Chinchilla-optimal point), we accept suboptimal training efficiency but gain a permanently cheaper model at inference. If a model will serve \\(Q\\) queries, the total cost is \\(C_{\\text{train}} + Q \\cdot O(N)\\). For large \\(Q\\), reducing \\(N\\) dominates, justifying "over-training" on data.'
                },
                {
                    question: 'Schaeffer et al. (2023) argue that emergent abilities are a "mirage" caused by nonlinear metrics. Explain their argument using the example of exact-match accuracy vs. per-token log-probability on an arithmetic task.',
                    hint: 'A 5-digit addition has 6 output tokens. Consider what happens to exact-match accuracy when per-token accuracy goes from 80% to 99%.',
                    solution: 'For a 6-token output (e.g., answer to a 5-digit addition), exact-match accuracy requires all 6 tokens to be correct simultaneously. If per-token accuracy is \\(p\\), exact-match accuracy is \\(p^6\\). When \\(p = 0.80\\), exact-match is \\(0.80^6 \\approx 0.26\\). When \\(p = 0.90\\), exact-match is \\(0.90^6 \\approx 0.53\\). When \\(p = 0.99\\), exact-match is \\(0.99^6 \\approx 0.94\\). The per-token probability improves smoothly and predictably with scale. But the nonlinear transformation \\(p^6\\) creates a sharp transition in exact-match accuracy, making it appear "emergent." The underlying capability improves continuously; the metric introduces the discontinuity.'
                }
            ]
        }
    ]
});
