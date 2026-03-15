// === Chapter 20: Instruction Tuning & Alignment ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch20',
    number: 20,
    title: 'Instruction Tuning & Alignment',
    subtitle: 'From raw language models to helpful assistants: SFT, RLHF, and DPO',
    sections: [
        // ======================== Section 1 ========================
        {
            id: 'sec20-1-sft',
            title: 'Supervised Fine-Tuning',
            content: `
<h2>20.1 Supervised Fine-Tuning</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Teaching a Model to Follow Instructions</div><div class="env-body">
<p>A pretrained language model (Chapter 19) is a powerful next-token predictor, but it does not know how to <em>follow instructions</em>. Given "Summarize this article," it might continue with "in 500 words for your English class" rather than producing a summary. <strong>Supervised fine-tuning (SFT)</strong> bridges this gap by training the model on curated (instruction, response) pairs, teaching it the conversational turn-taking pattern that users expect.</p>
</div></div>

<h3>Chat Format and Templates</h3>

<p>SFT data is structured as multi-turn conversations. Each example contains a <strong>system prompt</strong>, one or more <strong>user messages</strong>, and corresponding <strong>assistant responses</strong>. These are concatenated with special delimiter tokens into a single sequence for training:</p>

\\[
x = [\\texttt{&lt;|system|&gt;}, s, \\texttt{&lt;|user|&gt;}, u_1, \\texttt{&lt;|assistant|&gt;}, a_1, \\ldots]
\\]

<p>The loss is computed <em>only on assistant tokens</em>, not on the user or system tokens. This prevents the model from learning to generate user messages.</p>

<div class="env-block definition"><div class="env-title">Definition 20.1.1 &mdash; SFT Objective</div><div class="env-body">
<p>Given a dataset \\(\\mathcal{D}_{\\text{SFT}} = \\{(x^{(i)}, y^{(i)})\\}_{i=1}^N\\) of instruction-response pairs, the SFT objective minimizes:</p>
\\[
\\mathcal{L}_{\\text{SFT}}(\\theta) = -\\frac{1}{N}\\sum_{i=1}^{N}\\sum_{t \\in \\mathcal{A}^{(i)}} \\log P_\\theta(y_t^{(i)} \\mid y_{&lt;t}^{(i)}, x^{(i)}),
\\]
<p>where \\(\\mathcal{A}^{(i)}\\) denotes the set of token positions belonging to assistant responses. Tokens in system and user turns are masked from the loss.</p>
</div></div>

<h3>Data Quality over Quantity</h3>

<p>A surprising finding from the LIMA paper (Zhou et al., 2023) is that SFT with as few as 1,000 carefully curated examples can produce a highly capable assistant, sometimes matching models trained on orders of magnitude more data. The key insight: the model already possesses the knowledge from pretraining; SFT merely teaches it the <em>format</em> of helpful responses.</p>

<div class="env-block remark"><div class="env-title">Remark &mdash; Superficial Alignment Hypothesis</div><div class="env-body">
<p>The "superficial alignment hypothesis" posits that alignment is primarily about learning style and format rather than new knowledge. The pretrained model already knows how to answer questions; SFT teaches it <em>when and how</em> to deploy that knowledge in a conversational context. This explains why small, high-quality datasets suffice.</p>
</div></div>

<div class="env-block warning"><div class="env-title">Data Contamination</div><div class="env-body">
<p>Low-quality SFT data (e.g., noisy web scrapes, incorrect answers) can <em>degrade</em> the model's capabilities. A single confident but wrong answer in the training set can override the model's correct prior knowledge. Quality filtering, deduplication, and human review are critical.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-sft-pipeline"></div>
`,
            visualizations: [
                {
                    id: 'viz-sft-pipeline',
                    title: 'SFT Pipeline',
                    description: 'The supervised fine-tuning pipeline takes a pretrained base model and instruction-response data to produce a fine-tuned assistant model.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 380, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var phase = 0;

                        function roundRect(x, y, w, h, r, fill, stroke) {
                            ctx.beginPath();
                            ctx.moveTo(x + r, y);
                            ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                            ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                            ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                            ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
                            ctx.closePath();
                            if (fill) { ctx.fillStyle = fill; ctx.fill(); }
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
                        }

                        function arrow(x1, y1, x2, y2, col) {
                            var dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy);
                            var ang = Math.atan2(dy, dx);
                            ctx.strokeStyle = col; ctx.lineWidth = 2;
                            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - Math.cos(ang) * 8, y2 - Math.sin(ang) * 8); ctx.stroke();
                            ctx.fillStyle = col; ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 10 * Math.cos(ang - 0.4), y2 - 10 * Math.sin(ang - 0.4));
                            ctx.lineTo(x2 - 10 * Math.cos(ang + 0.4), y2 - 10 * Math.sin(ang + 0.4));
                            ctx.closePath(); ctx.fill();
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var midY = H / 2;
                            var t = Math.min(phase, 1);

                            // Base model box
                            var bx = 30, bw = 140, bh = 80;
                            roundRect(bx, midY - bh / 2, bw, bh, 8, viz.colors.blue + '22', viz.colors.blue);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                            ctx.fillText('Base LM', bx + bw / 2, midY - 10);
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Pretrained', bx + bw / 2, midY + 10);

                            // Arrow to SFT
                            arrow(bx + bw + 10, midY, bx + bw + 60, midY, viz.colors.text);

                            // SFT box
                            var sx = bx + bw + 70, sw = 160;
                            roundRect(sx, midY - bh / 2, sw, bh, 8, viz.colors.orange + '22', viz.colors.orange);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillText('SFT Training', sx + sw / 2, midY - 10);
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Cross-entropy on', sx + sw / 2, midY + 6);
                            ctx.fillText('assistant tokens', sx + sw / 2, midY + 20);

                            // Instruction data feeding in from above
                            var dx = sx + sw / 2, dy = midY - bh / 2 - 70;
                            roundRect(dx - 70, dy, 140, 50, 6, viz.colors.teal + '22', viz.colors.teal);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillText('Instruction Data', dx, dy + 16);
                            ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('(instruction, response) pairs', dx, dy + 34);
                            arrow(dx, dy + 50 + 5, dx, midY - bh / 2 - 5, viz.colors.teal);

                            // Arrow to output
                            arrow(sx + sw + 10, midY, sx + sw + 60, midY, viz.colors.text);

                            // Fine-tuned model
                            var fx = sx + sw + 70, fw = 150;
                            var glow = 0.3 + 0.15 * Math.sin(phase * 2);
                            roundRect(fx, midY - bh / 2, fw, bh, 8, viz.colors.green + Math.round(glow * 255).toString(16).padStart(2, '0'), viz.colors.green);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillText('SFT Model', fx + fw / 2, midY - 10);
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Follows instructions', fx + fw / 2, midY + 10);

                            // Example data at bottom
                            var ey = midY + bh / 2 + 30;
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Example:', 30, ey);
                            ctx.fillStyle = viz.colors.blue; ctx.font = '11px monospace';
                            ctx.fillText('[User] Explain gravity in one sentence.', 30, ey + 18);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('[Asst] Gravity is the force by which objects with mass attract each other.', 30, ey + 36);
                            ctx.fillStyle = viz.colors.orange; ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('\u2190 loss computed only on these tokens', 30, ey + 54);
                        }

                        viz.animate(function(t) { phase = t * 0.001; draw(); });
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In SFT, the loss is masked so that only assistant tokens contribute. If a training example has 50 user tokens and 200 assistant tokens, what fraction of tokens contribute to the gradient? Why is this masking important?',
                    hint: 'Count the tokens that contribute to the loss vs. the total sequence length.',
                    solution: '200 / (50 + 200) = 80% of tokens contribute to the gradient. The masking is important because we want the model to learn to <em>generate</em> helpful responses, not to generate user queries. Without masking, the model would also learn to produce user-style text, which could cause it to role-play as the user or generate questions when it should be answering them.'
                },
                {
                    question: 'The LIMA paper showed that 1,000 high-quality SFT examples can match models trained on 50,000+ examples. Explain this in terms of the superficial alignment hypothesis.',
                    hint: 'What does the pretrained model already know? What does SFT actually need to teach?',
                    solution: 'Under the superficial alignment hypothesis, the pretrained model already has vast knowledge and capabilities from next-token prediction on trillions of tokens. SFT only needs to teach the model the <em>format</em> of helpful conversation: when to answer, how to structure responses, and the appropriate tone. This is a low-dimensional behavioral shift, not new knowledge acquisition, so a small number of high-quality demonstrations suffice to specify the target distribution of response styles.'
                },
                {
                    question: 'Compute the per-token SFT loss for an assistant response "Yes" (single token, vocabulary size \\(|V| = 32000\\)) if the model assigns probability 0.02 to "Yes." Compare to a model assigning probability 0.85.',
                    hint: 'Use \\(\\mathcal{L} = -\\log P_\\theta(y_t \\mid \\cdot)\\).',
                    solution: 'For \\(P = 0.02\\): \\(\\mathcal{L} = -\\ln(0.02) \\approx 3.91\\) nats. For \\(P = 0.85\\): \\(\\mathcal{L} = -\\ln(0.85) \\approx 0.16\\) nats. The weak model incurs about 24x higher loss. A uniform model would have loss \\(-\\ln(1/32000) \\approx 10.37\\) nats, so even the 0.02 probability is far better than random, reflecting the pretrained model\'s prior knowledge.'
                }
            ]
        },

        // ======================== Section 2 ========================
        {
            id: 'sec20-2-reward-modeling',
            title: 'Reward Modeling',
            content: `
<h2>20.2 Reward Modeling</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Learning What Humans Prefer</div><div class="env-body">
<p>SFT teaches a model to follow instructions, but there are many <em>ways</em> to follow an instruction. Some responses are more helpful, accurate, or safe than others. How do we teach a model which responses are better? We ask humans to <strong>compare</strong> pairs of responses and train a <strong>reward model</strong> to predict these preferences. This reward model then serves as a proxy for human judgment in the reinforcement learning phase.</p>
</div></div>

<h3>Preference Data Collection</h3>

<p>Given a prompt \\(x\\), we sample two (or more) responses \\(y_w\\) (preferred/winner) and \\(y_l\\) (dispreferred/loser) from the SFT model. Human annotators then label which response is better. The resulting dataset consists of triples \\((x, y_w, y_l)\\).</p>

<div class="env-block definition"><div class="env-title">Definition 20.2.1 &mdash; Bradley-Terry Model</div><div class="env-body">
<p>The <strong>Bradley-Terry model</strong> assumes that the probability of preferring response \\(y_1\\) over \\(y_2\\) is determined by their latent "quality" scores. Given a parameterized reward function \\(r_\\phi(x, y)\\):</p>
\\[
P(y_1 \\succ y_2 \\mid x) = \\sigma\\bigl(r_\\phi(x, y_1) - r_\\phi(x, y_2)\\bigr),
\\]
<p>where \\(\\sigma(z) = 1/(1 + e^{-z})\\) is the sigmoid function. The reward difference determines the preference probability.</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 20.2.2 &mdash; Reward Model Loss</div><div class="env-body">
<p>The reward model is trained to maximize the log-likelihood of observed preferences:</p>
\\[
\\mathcal{L}_{\\text{RM}}(\\phi) = -\\mathbb{E}_{(x,y_w,y_l) \\sim \\mathcal{D}} \\bigl[\\log \\sigma\\bigl(r_\\phi(x, y_w) - r_\\phi(x, y_l)\\bigr)\\bigr].
\\]
<p>This is equivalent to binary cross-entropy where the label is always "preferred response wins."</p>
</div></div>

<h3>Reward Model Architecture</h3>

<p>The reward model is typically initialized from the SFT model, with the language model head replaced by a scalar output head. The final hidden state of the last token is projected to a single number \\(r_\\phi(x, y) \\in \\mathbb{R}\\). This leverages the pretrained representations to understand response quality.</p>

<div class="env-block remark"><div class="env-title">Remark &mdash; Scale Invariance</div><div class="env-body">
<p>The Bradley-Terry model depends only on reward <em>differences</em>, not absolute values. Adding a constant \\(c\\) to all rewards does not change preferences: \\(\\sigma((r_1 + c) - (r_2 + c)) = \\sigma(r_1 - r_2)\\). This means the reward model has one degree of freedom that must be fixed, typically by normalizing rewards to have zero mean on a reference set.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-reward-model"></div>
`,
            visualizations: [
                {
                    id: 'viz-reward-model',
                    title: 'Reward Model: Preference Pairs',
                    description: 'Each pair shows a prompt with two responses. The reward model assigns a scalar score to each; the preferred response should receive a higher reward. Adjust the reward gap to see how the Bradley-Terry probability changes.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var gap = 2.0;

                        var pairs = [
                            { prompt: 'Explain photosynthesis.', win: 'Plants convert sunlight to energy via chlorophyll...', lose: 'Its when plants do stuff with sun idk' },
                            { prompt: 'Is 7 prime?', win: 'Yes. 7 is divisible only by 1 and itself.', lose: 'Yes it is a prime number because its odd.' },
                            { prompt: 'Write a haiku about rain.', win: 'Drops tap the window / puddles mirror gray-cloud sky / earth drinks and exhales', lose: 'Rain is wet and cold / I do not like rainy days / Rain rain go away' }
                        ];

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Reward Model: Preference Scoring', W / 2, 20);

                            var py = 50;
                            var barMaxW = 120;

                            for (var i = 0; i < pairs.length; i++) {
                                var p = pairs[i];
                                var rw = 1.0 + gap / 2 + (i * 0.3);
                                var rl = 1.0 - gap / 2 + (i * 0.3);
                                var prob = 1.0 / (1.0 + Math.exp(-(rw - rl)));

                                // Prompt
                                ctx.fillStyle = viz.colors.yellow; ctx.font = 'bold 11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('Prompt: ' + p.prompt, 20, py);

                                // Winner bar
                                var by = py + 16;
                                ctx.fillStyle = viz.colors.green + '33';
                                ctx.fillRect(20, by, barMaxW * (rw / 5), 20);
                                ctx.fillStyle = viz.colors.green;
                                ctx.fillRect(20, by, Math.max(barMaxW * (rw / 5), 2), 20);
                                ctx.fillStyle = viz.colors.white; ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('\u2713 ' + p.win.substring(0, 45) + '...', barMaxW * (rw / 5) + 30, by + 13);
                                ctx.fillStyle = viz.colors.green; ctx.font = 'bold 11px monospace';
                                ctx.textAlign = 'right';
                                ctx.fillText('r=' + rw.toFixed(2), W - 20, by + 13);

                                // Loser bar
                                by += 26;
                                ctx.textAlign = 'left';
                                ctx.fillStyle = viz.colors.red + '33';
                                ctx.fillRect(20, by, barMaxW * (rl / 5), 20);
                                ctx.fillStyle = viz.colors.red;
                                ctx.fillRect(20, by, Math.max(barMaxW * (rl / 5), 2), 20);
                                ctx.fillStyle = viz.colors.white; ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText('\u2717 ' + p.lose.substring(0, 45) + '...', barMaxW * (rl / 5) + 30, by + 13);
                                ctx.fillStyle = viz.colors.red; ctx.font = 'bold 11px monospace';
                                ctx.textAlign = 'right';
                                ctx.fillText('r=' + rl.toFixed(2), W - 20, by + 13);

                                // Bradley-Terry probability
                                by += 26;
                                ctx.fillStyle = viz.colors.teal; ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('P(win \u227B lose) = \u03C3(' + (rw - rl).toFixed(2) + ') = ' + prob.toFixed(3), 20, by + 4);

                                py = by + 24;
                            }

                            // Summary
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Reward gap \u0394r = ' + gap.toFixed(1) + ' \u2192 larger gap = stronger preference signal', W / 2, H - 15);
                        }

                        VizEngine.createSlider(controls, 'Reward Gap', 0.0, 5.0, 2.0, 0.1, function(v) {
                            gap = v; draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In the Bradley-Terry model, if \\(r_\\phi(x, y_w) = 3.0\\) and \\(r_\\phi(x, y_l) = 1.0\\), what is \\(P(y_w \\succ y_l)\\)? What reward difference is needed for 95% confidence?',
                    hint: 'Use \\(P = \\sigma(\\Delta r)\\) and solve \\(\\sigma(\\Delta r) = 0.95\\).',
                    solution: '\\(P(y_w \\succ y_l) = \\sigma(3.0 - 1.0) = \\sigma(2.0) = 1/(1 + e^{-2}) \\approx 0.881\\). For 95% confidence: \\(\\sigma(\\Delta r) = 0.95 \\Rightarrow \\Delta r = \\ln(0.95/0.05) = \\ln(19) \\approx 2.944\\). So a reward gap of about 3.0 corresponds to 95% preference probability.'
                },
                {
                    question: 'Derive the gradient of the reward model loss \\(\\mathcal{L}_{\\text{RM}}\\) with respect to the reward difference \\(\\Delta = r_\\phi(x, y_w) - r_\\phi(x, y_l)\\).',
                    hint: 'Use the identity \\(\\sigma\'(z) = \\sigma(z)(1 - \\sigma(z))\\) and the chain rule.',
                    solution: '\\(\\mathcal{L} = -\\log\\sigma(\\Delta)\\). Taking the derivative: \\(\\frac{\\partial \\mathcal{L}}{\\partial \\Delta} = -\\frac{\\sigma\'(\\Delta)}{\\sigma(\\Delta)} = -\\frac{\\sigma(\\Delta)(1-\\sigma(\\Delta))}{\\sigma(\\Delta)} = -(1-\\sigma(\\Delta)) = \\sigma(\\Delta) - 1\\). When \\(\\Delta\\) is large (correct ranking, high confidence), the gradient is near zero. When \\(\\Delta\\) is small or negative (wrong ranking), the gradient is large, pushing the model to increase the reward gap.'
                },
                {
                    question: 'Why is the reward model initialized from the SFT model rather than trained from scratch? What would go wrong with a randomly initialized reward model?',
                    hint: 'Think about what the reward model needs to understand in order to evaluate response quality.',
                    solution: 'Evaluating response quality requires understanding language, factual accuracy, coherence, and instruction-following, all of which the SFT model has already learned. A randomly initialized model would need to learn all of these from the relatively small preference dataset (typically 50k-500k comparisons), which is insufficient for learning language understanding from scratch. Initializing from SFT lets the reward model leverage pretrained representations and focus on learning the comparatively simpler task of scoring quality differences between already-competent responses.'
                }
            ]
        },

        // ======================== Section 3 ========================
        {
            id: 'sec20-3-rlhf',
            title: 'RLHF with PPO',
            content: `
<h2>20.3 RLHF with PPO</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Optimizing for Human Preferences</div><div class="env-body">
<p>With a reward model in hand, we can now optimize the language model to produce responses that score highly. This is <strong>reinforcement learning from human feedback (RLHF)</strong>. The language model is the "policy," generating text is the "action," and the reward model provides the "reward." We use Proximal Policy Optimization (PPO) to update the policy while preventing it from deviating too far from the original SFT model.</p>
</div></div>

<h3>The RLHF Objective</h3>

<div class="env-block definition"><div class="env-title">Definition 20.3.1 &mdash; RLHF Objective with KL Penalty</div><div class="env-body">
<p>The RLHF objective maximizes expected reward while penalizing divergence from the reference policy \\(\\pi_{\\text{ref}}\\) (the SFT model):</p>
\\[
\\max_{\\pi_\\theta} \\; \\mathbb{E}_{x \\sim \\mathcal{D},\\, y \\sim \\pi_\\theta(\\cdot|x)} \\bigl[r_\\phi(x, y)\\bigr] - \\beta\\, \\mathrm{KL}\\bigl(\\pi_\\theta(\\cdot|x) \\,\\|\\, \\pi_{\\text{ref}}(\\cdot|x)\\bigr),
\\]
<p>where \\(\\beta &gt; 0\\) controls the strength of the KL penalty. The KL divergence is computed at the token level and summed over the response.</p>
</div></div>

<p>The KL penalty serves two critical purposes:</p>
<ol>
<li><strong>Prevents reward hacking:</strong> Without it, the policy would find degenerate outputs that exploit quirks of the reward model (e.g., repeating certain phrases that receive artificially high scores).</li>
<li><strong>Preserves capabilities:</strong> The pretrained model's language abilities should not be destroyed during RL optimization.</li>
</ol>

<h3>PPO Algorithm for Language Models</h3>

<p>PPO (Schulman et al., 2017) is a policy gradient method that clips the objective to prevent excessively large updates. For language models, each token generation is an "action" and the full response receives a reward at the end.</p>

<div class="env-block definition"><div class="env-title">Definition 20.3.2 &mdash; PPO Clipped Objective</div><div class="env-body">
<p>At each token position \\(t\\), define the probability ratio \\(\\rho_t = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}\\). The PPO objective is:</p>
\\[
\\mathcal{L}_{\\text{PPO}} = \\mathbb{E}_t\\bigl[\\min\\bigl(\\rho_t \\hat{A}_t,\\; \\mathrm{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon)\\hat{A}_t\\bigr)\\bigr],
\\]
<p>where \\(\\hat{A}_t\\) is the advantage estimate and \\(\\epsilon \\approx 0.2\\) is the clipping range. The \\(\\min\\) ensures that the objective is a lower bound on the unclipped objective.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; The Full RLHF Pipeline</div><div class="env-body">
<p>RLHF requires maintaining <strong>four models</strong> simultaneously in memory: (1) the active policy \\(\\pi_\\theta\\), (2) the reference policy \\(\\pi_{\\text{ref}}\\) (frozen SFT model, for KL computation), (3) the reward model \\(r_\\phi\\), and (4) a value function (critic) for advantage estimation. For a 70B parameter model, this means roughly 280B parameters total, requiring enormous GPU memory.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-rlhf-pipeline"></div>
`,
            visualizations: [
                {
                    id: 'viz-rlhf-pipeline',
                    title: 'RLHF Pipeline',
                    description: 'The three-stage RLHF pipeline: SFT pretraining, reward model training from human preferences, and PPO optimization with a KL penalty against the reference policy.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var activeStage = 0;

                        function rr(x, y, w, h, r, fill, stroke, lw) {
                            ctx.beginPath();
                            ctx.moveTo(x + r, y);
                            ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                            ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                            ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                            ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
                            ctx.closePath();
                            if (fill) { ctx.fillStyle = fill; ctx.fill(); }
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 2; ctx.stroke(); }
                        }

                        function arr(x1, y1, x2, y2, col) {
                            var dx = x2 - x1, dy = y2 - y1, a = Math.atan2(dy, dx);
                            ctx.strokeStyle = col; ctx.lineWidth = 2;
                            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - Math.cos(a) * 7, y2 - Math.sin(a) * 7); ctx.stroke();
                            ctx.fillStyle = col; ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 9 * Math.cos(a - 0.4), y2 - 9 * Math.sin(a - 0.4));
                            ctx.lineTo(x2 - 9 * Math.cos(a + 0.4), y2 - 9 * Math.sin(a + 0.4));
                            ctx.closePath(); ctx.fill();
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var cols = [viz.colors.blue, viz.colors.orange, viz.colors.green];
                            var labels = ['Stage 1: SFT', 'Stage 2: Reward Model', 'Stage 3: PPO'];
                            var descs = [
                                ['Base LM', '+', 'Demo Data', '\u2192', 'SFT Model (\u03C0_ref)'],
                                ['SFT Model', '+', 'Human Prefs', '\u2192', 'Reward Model (r)'],
                                ['\u03C0_\u03B8 generates', '\u2192', 'r scores', '\u2192', '\u03C0_\u03B8 updated']
                            ];

                            // Three stage rows
                            for (var i = 0; i < 3; i++) {
                                var y = 50 + i * 115;
                                var alpha = (i === activeStage) ? 'ff' : '66';
                                var col = cols[i];

                                // Stage label
                                ctx.fillStyle = col; ctx.font = 'bold 13px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText(labels[i], 20, y + 12);

                                // Flow boxes
                                var d = descs[i];
                                var bx = 20;
                                for (var j = 0; j < d.length; j++) {
                                    if (d[j] === '\u2192' || d[j] === '+') {
                                        ctx.fillStyle = viz.colors.text + alpha;
                                        ctx.font = '16px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.fillText(d[j], bx + 20, y + 50);
                                        bx += 40;
                                    } else {
                                        var tw = ctx.measureText(d[j]).width + 24;
                                        tw = Math.max(tw, 100);
                                        rr(bx, y + 30, tw, 36, 6, col + '1a', col + alpha, i === activeStage ? 2 : 1);
                                        ctx.fillStyle = viz.colors.white + alpha;
                                        ctx.font = '12px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.fillText(d[j], bx + tw / 2, y + 48);
                                        bx += tw + 10;
                                    }
                                }
                            }

                            // KL penalty note for stage 3
                            if (activeStage === 2) {
                                ctx.fillStyle = viz.colors.purple;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText('KL penalty: \u03B2 \u00B7 KL(\u03C0_\u03B8 || \u03C0_ref) prevents reward hacking', 20, 380);
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText('4 models in memory: policy, reference, reward, critic', 20, 398);
                            }
                        }

                        VizEngine.createSlider(controls, 'Active Stage', 0, 2, 0, 1, function(v) {
                            activeStage = Math.round(v); draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'If \\(\\beta = 0\\) in the RLHF objective, what behavior would you expect from the optimized policy? Why is this problematic?',
                    hint: 'Without the KL penalty, the policy is free to deviate arbitrarily from the reference model.',
                    solution: 'With \\(\\beta = 0\\), the policy would maximize reward without constraint, leading to <strong>reward hacking</strong>: the policy finds degenerate outputs that score highly on the reward model but are not genuinely helpful. For example, it might produce repetitive flattery ("Great question! You are so smart!"), exploit formatting tricks, or produce outputs that are adversarial examples for the reward model. The KL penalty forces the policy to remain close to the SFT model, preserving linguistic coherence and general capability.'
                },
                {
                    question: 'RLHF requires 4 copies of the model in memory. For a 13B parameter model in FP16, estimate the total GPU memory required (parameters only, ignoring optimizer states and activations).',
                    hint: 'FP16 uses 2 bytes per parameter.',
                    solution: 'Each model copy: \\(13 \\times 10^9 \\times 2\\) bytes \\(= 26\\) GB. Four copies: \\(4 \\times 26 = 104\\) GB. Adding Adam optimizer states for the policy (2 additional copies of parameters for first and second moments, each 26 GB) brings the total to \\(104 + 52 = 156\\) GB minimum, requiring at least 2-3 A100 80GB GPUs just for parameters. Activation memory and gradient checkpointing add further requirements. This enormous memory footprint is a key motivation for methods like DPO that eliminate the need for separate reward and value models.'
                },
                {
                    question: 'The PPO clipping parameter \\(\\epsilon = 0.2\\) means the probability ratio \\(\\rho_t\\) is clipped to \\([0.8, 1.2]\\). Why is it important to clip from both sides? What happens if we only clip from above?',
                    hint: 'Consider both positive and negative advantages. What does the clipping prevent in each case?',
                    solution: 'For positive advantages (good actions), clipping from above (\\(\\rho_t \\leq 1.2\\)) prevents the policy from becoming too much more likely to take that action in a single update, ensuring stability. For negative advantages (bad actions), clipping from below (\\(\\rho_t \\geq 0.8\\)) prevents the policy from too aggressively reducing the probability. Without the lower clip, a single bad experience could cause the policy to assign near-zero probability to an action, which is difficult to reverse and can cause mode collapse. Both clips together ensure that each PPO step makes bounded changes to the policy.'
                }
            ]
        },

        // ======================== Section 4 ========================
        {
            id: 'sec20-4-dpo',
            title: 'Direct Preference Optimization',
            content: `
<h2>20.4 Direct Preference Optimization</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Skipping the Reward Model</div><div class="env-body">
<p>RLHF is effective but complex: it requires training a separate reward model and running an RL loop with four models in memory. <strong>Direct Preference Optimization (DPO)</strong> (Rafailov et al., 2023) shows that we can bypass the reward model entirely. The key insight is that the optimal policy under the KL-constrained RLHF objective has a <em>closed-form relationship</em> with the reward function, allowing us to reparameterize the reward model loss directly in terms of the policy.</p>
</div></div>

<h3>From RLHF to DPO</h3>

<div class="env-block theorem"><div class="env-title">Theorem 20.4.1 &mdash; Optimal Policy under KL Constraint</div><div class="env-body">
<p>The optimal solution to the KL-regularized RLHF objective</p>
\\[
\\max_{\\pi} \\; \\mathbb{E}_{y \\sim \\pi}[r(x,y)] - \\beta\\,\\mathrm{KL}(\\pi \\| \\pi_{\\text{ref}})
\\]
<p>is given by</p>
\\[
\\pi^*(y \\mid x) = \\frac{1}{Z(x)}\\pi_{\\text{ref}}(y \\mid x)\\exp\\!\\left(\\frac{r(x,y)}{\\beta}\\right),
\\]
<p>where \\(Z(x) = \\sum_y \\pi_{\\text{ref}}(y|x)\\exp(r(x,y)/\\beta)\\) is the partition function.</p>
</div></div>

<p>Rearranging the optimal policy expression, we can express the reward as:</p>
\\[
r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x).
\\]

<p>The partition function \\(Z(x)\\) is intractable, but it cancels when we substitute into the Bradley-Terry preference model, because it depends only on \\(x\\) and not on the particular response \\(y\\).</p>

<div class="env-block definition"><div class="env-title">Definition 20.4.2 &mdash; DPO Loss</div><div class="env-body">
<p>Substituting the implicit reward into the Bradley-Terry model yields the <strong>DPO loss</strong>:</p>
\\[
\\mathcal{L}_{\\text{DPO}}(\\theta) = -\\mathbb{E}_{(x,y_w,y_l)}\\left[\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right].
\\]
<p>This is a supervised loss on preference pairs that directly optimizes the policy without any RL loop or reward model.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Implicit Reward</div><div class="env-body">
<p>DPO defines an <strong>implicit reward</strong>: \\(r(x,y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)\\). The policy itself is the reward model. Increasing the relative probability of a response (compared to the reference) is equivalent to increasing its reward. This elegant reparameterization is what makes DPO work without a separate reward model.</p>
</div></div>

<div class="env-block warning"><div class="env-title">DPO vs. RLHF Trade-offs</div><div class="env-body">
<p>DPO is simpler to implement and more memory-efficient (2 models instead of 4), but it has limitations: (1) it requires on-policy or near-on-policy data, (2) it can overfit to the preference dataset more easily, and (3) the implicit reward cannot be reused for other purposes (e.g., best-of-N sampling). Recent work (IPO, KTO, ORPO) extends the DPO framework to address some of these issues.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-dpo-vs-rlhf"></div>
`,
            visualizations: [
                {
                    id: 'viz-dpo-vs-rlhf',
                    title: 'DPO vs. RLHF',
                    description: 'Side-by-side comparison of RLHF and DPO pipelines. RLHF requires a separate reward model and RL loop; DPO directly optimizes the policy on preference data.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        function rr(x, y, w, h, r, fill, stroke) {
                            ctx.beginPath();
                            ctx.moveTo(x + r, y);
                            ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                            ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                            ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                            ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
                            ctx.closePath();
                            if (fill) { ctx.fillStyle = fill; ctx.fill(); }
                            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
                        }

                        function arr(x1, y1, x2, y2, col) {
                            var a = Math.atan2(y2 - y1, x2 - x1);
                            ctx.strokeStyle = col; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - Math.cos(a) * 6, y2 - Math.sin(a) * 6); ctx.stroke();
                            ctx.fillStyle = col; ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 8 * Math.cos(a - 0.4), y2 - 8 * Math.sin(a - 0.4));
                            ctx.lineTo(x2 - 8 * Math.cos(a + 0.4), y2 - 8 * Math.sin(a + 0.4));
                            ctx.closePath(); ctx.fill();
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var midX = W / 2;

                            // Divider
                            ctx.strokeStyle = viz.colors.text + '44'; ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath(); ctx.moveTo(midX, 30); ctx.lineTo(midX, H - 10); ctx.stroke();
                            ctx.setLineDash([]);

                            // RLHF side (left)
                            ctx.fillStyle = viz.colors.orange; ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('RLHF (PPO)', midX / 2, 25);

                            var lx = 20, bw = 110, bh = 32, gap = 14;
                            var boxes = [
                                { t: 'SFT Model', c: viz.colors.blue, y: 50 },
                                { t: 'Reward Model', c: viz.colors.orange, y: 50 + bh + gap + 40 },
                                { t: 'PPO Loop', c: viz.colors.green, y: 50 + 2 * (bh + gap) + 80 },
                                { t: 'Aligned Model', c: viz.colors.teal, y: 50 + 3 * (bh + gap) + 120 }
                            ];

                            // Preference data
                            rr(lx + 140, 50, 100, bh, 5, viz.colors.yellow + '1a', viz.colors.yellow);
                            ctx.fillStyle = viz.colors.white; ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText('Pref Data', lx + 190, 50 + bh / 2 + 3);

                            // SFT box
                            rr(lx, boxes[0].y, bw, bh, 5, boxes[0].c + '1a', boxes[0].c);
                            ctx.fillStyle = viz.colors.white; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText(boxes[0].t, lx + bw / 2, boxes[0].y + bh / 2 + 3);

                            arr(lx + bw / 2, boxes[0].y + bh + 2, lx + bw / 2, boxes[1].y - 2, viz.colors.text);
                            arr(lx + 190, 50 + bh + 2, lx + bw / 2 + 30, boxes[1].y - 2, viz.colors.yellow);

                            // RM box
                            rr(lx, boxes[1].y, bw, bh, 5, boxes[1].c + '1a', boxes[1].c);
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText(boxes[1].t, lx + bw / 2, boxes[1].y + bh / 2 + 3);

                            arr(lx + bw / 2, boxes[1].y + bh + 2, lx + bw / 2, boxes[2].y - 2, viz.colors.text);

                            // PPO box
                            rr(lx, boxes[2].y, bw, bh, 5, boxes[2].c + '1a', boxes[2].c);
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText(boxes[2].t, lx + bw / 2, boxes[2].y + bh / 2 + 3);

                            // PPO side boxes
                            ctx.fillStyle = viz.colors.text; ctx.font = '9px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('4 models in GPU', lx + bw + 12, boxes[2].y + 10);
                            ctx.fillText('memory: policy,', lx + bw + 12, boxes[2].y + 22);
                            ctx.fillText('ref, reward, critic', lx + bw + 12, boxes[2].y + 34);

                            arr(lx + bw / 2, boxes[2].y + bh + 2, lx + bw / 2, boxes[3].y - 2, viz.colors.text);

                            // Output
                            rr(lx, boxes[3].y, bw, bh, 5, boxes[3].c + '1a', boxes[3].c);
                            ctx.fillStyle = viz.colors.white; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(boxes[3].t, lx + bw / 2, boxes[3].y + bh / 2 + 3);

                            // DPO side (right)
                            var rx = midX + 30;
                            ctx.fillStyle = viz.colors.purple; ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.fillText('DPO', midX + (W - midX) / 2, 25);

                            var dboxes = [
                                { t: 'SFT Model (\u03C0_ref)', c: viz.colors.blue, y: 50 },
                                { t: 'Pref Data', c: viz.colors.yellow, y: 50 + bh + gap + 40 },
                                { t: 'Supervised Loss', c: viz.colors.purple, y: 50 + 2 * (bh + gap) + 80 },
                                { t: 'Aligned Model', c: viz.colors.teal, y: 50 + 3 * (bh + gap) + 120 }
                            ];

                            for (var i = 0; i < dboxes.length; i++) {
                                rr(rx, dboxes[i].y, bw, bh, 5, dboxes[i].c + '1a', dboxes[i].c);
                                ctx.fillStyle = viz.colors.white; ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(dboxes[i].t, rx + bw / 2, dboxes[i].y + bh / 2 + 3);
                                if (i < dboxes.length - 1) {
                                    arr(rx + bw / 2, dboxes[i].y + bh + 2, rx + bw / 2, dboxes[i + 1].y - 2, viz.colors.text);
                                }
                            }

                            // DPO advantages
                            ctx.fillStyle = viz.colors.text; ctx.font = '9px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('2 models only:', rx + bw + 12, dboxes[2].y + 8);
                            ctx.fillText('policy + ref (frozen)', rx + bw + 12, dboxes[2].y + 20);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('No RL loop needed', rx + bw + 12, dboxes[2].y + 36);

                            // Bottom comparison
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Complex but flexible', midX / 2, H - 12);
                            ctx.fillText('Simple but less flexible', midX + (W - midX) / 2, H - 12);
                        }

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Starting from the RLHF objective \\(\\max_\\pi \\mathbb{E}_{y \\sim \\pi}[r(x,y)] - \\beta\\,\\mathrm{KL}(\\pi\\|\\pi_{\\text{ref}})\\), derive the optimal policy \\(\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x)\\exp(r(x,y)/\\beta)\\).',
                    hint: 'Write the KL divergence explicitly and take the functional derivative with respect to \\(\\pi(y|x)\\) subject to \\(\\sum_y \\pi(y|x) = 1\\).',
                    solution: 'The objective is \\(J = \\sum_y \\pi(y|x)[r(x,y) - \\beta\\log(\\pi(y|x)/\\pi_{\\text{ref}}(y|x))]\\). Taking the functional derivative and setting it to zero (using a Lagrange multiplier \\(\\lambda\\) for the normalization constraint): \\(\\frac{\\partial J}{\\partial \\pi(y|x)} = r(x,y) - \\beta\\log\\frac{\\pi(y|x)}{\\pi_{\\text{ref}}(y|x)} - \\beta - \\lambda = 0\\). Solving: \\(\\log\\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} = \\frac{r(x,y) - \\beta - \\lambda}{\\beta}\\), so \\(\\pi^*(y|x) = \\pi_{\\text{ref}}(y|x)\\exp\\bigl(\\frac{r(x,y)}{\\beta}\\bigr) \\cdot \\exp\\bigl(-1 - \\lambda/\\beta\\bigr)\\). The last factor is absorbed into the normalization constant \\(1/Z(x)\\).'
                },
                {
                    question: 'In DPO, show that the partition function \\(Z(x)\\) cancels when computing the preference probability. That is, show that \\(\\sigma(r(x,y_w) - r(x,y_l))\\) does not depend on \\(Z(x)\\).',
                    hint: 'Substitute the implicit reward \\(r(x,y) = \\beta\\log\\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta\\log Z(x)\\) into the difference.',
                    solution: '\\(r(x,y_w) - r(x,y_l) = \\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} + \\beta\\log Z(x) - \\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} - \\beta\\log Z(x) = \\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\). The \\(\\beta\\log Z(x)\\) terms cancel exactly because the partition function depends only on \\(x\\), not on the specific response. This cancellation is what makes DPO tractable.'
                },
                {
                    question: 'DPO requires computing \\(\\log \\pi_\\theta(y|x)\\) for full responses. For a response of \\(T = 200\\) tokens with vocabulary \\(|V| = 32000\\), estimate the computational cost of one DPO gradient step compared to one PPO step.',
                    hint: 'DPO needs forward passes for \\((y_w, y_l)\\) through both \\(\\pi_\\theta\\) and \\(\\pi_{\\text{ref}}\\). PPO needs forward passes through all four models plus sampling.',
                    solution: 'DPO: 4 forward passes (\\(\\pi_\\theta\\) and \\(\\pi_{\\text{ref}}\\) on both \\(y_w\\) and \\(y_l\\)) plus 1 backward pass through \\(\\pi_\\theta\\). Total: ~5 model forward-equivalent passes. PPO: generate response (~T forward passes through \\(\\pi_\\theta\\)), score with \\(r_\\phi\\) (1 forward), compute KL with \\(\\pi_{\\text{ref}}\\) (1 forward), estimate values with critic (1 forward), backward through \\(\\pi_\\theta\\) and critic. Total: ~T+4 forward-equivalent passes plus overhead. Per gradient step, DPO is significantly cheaper (no autoregressive generation), but it requires pre-generated preference data whereas PPO generates on-policy data.'
                }
            ]
        },

        // ======================== Section 5 ========================
        {
            id: 'sec20-5-scaling-alignment-tax',
            title: 'Scaling & Alignment Tax',
            content: `
<h2>20.5 Scaling &amp; Alignment Tax</h2>

<div class="env-block intuition"><div class="env-title">Intuition &mdash; Does Alignment Cost Capability?</div><div class="env-body">
<p>Alignment procedures (SFT, RLHF, DPO) modify the model's output distribution to be more helpful and safe. But does this come at a cost? The <strong>alignment tax</strong> refers to any reduction in raw capability (e.g., coding, math, factual knowledge) caused by the alignment process. Empirically, the tax is often small for well-calibrated alignment, and aligned models can even <em>outperform</em> base models on benchmarks by following instructions more reliably.</p>
</div></div>

<h3>Scaling Laws for Aligned Models</h3>

<p>The Chinchilla scaling laws (Chapter 19) predict that loss scales as a power law in compute. Does alignment change this relationship? Empirical evidence suggests that alignment introduces a roughly constant offset in capability rather than changing the scaling exponent:</p>

\\[
L_{\\text{aligned}}(C) \\approx L_{\\text{base}}(C) + \\Delta_{\\text{align}},
\\]

<p>where \\(\\Delta_{\\text{align}}\\) is small and can even be negative on tasks where instruction-following improves benchmark performance.</p>

<div class="env-block definition"><div class="env-title">Definition 20.5.1 &mdash; Alignment Tax</div><div class="env-body">
<p>The <strong>alignment tax</strong> on a benchmark \\(B\\) is the relative performance difference between the aligned and base models:</p>
\\[
\\text{Tax}(B) = \\frac{\\text{Score}_{\\text{base}}(B) - \\text{Score}_{\\text{aligned}}(B)}{\\text{Score}_{\\text{base}}(B)}.
\\]
<p>A positive tax means alignment hurts performance; a negative tax (alignment bonus) means alignment helps. Tasks requiring instruction-following typically show a bonus; tasks measuring raw knowledge may show a small tax.</p>
</div></div>

<h3>Overoptimization and Goodhart's Law</h3>

<p>A critical finding from Gao et al. (2022) is that optimizing too aggressively against the reward model leads to <strong>reward overoptimization</strong>: the proxy reward increases but true human preference decreases. This is an instance of Goodhart's Law ("when a measure becomes a target, it ceases to be a good measure").</p>

<div class="env-block theorem"><div class="env-title">Theorem 20.5.2 &mdash; Reward Overoptimization Scaling</div><div class="env-body">
<p>Gao et al. (2022) find that the gold reward (true human preference) under best-of-N sampling scales as</p>
\\[
R_{\\text{gold}}(N) \\approx \\alpha \\sqrt{\\mathrm{KL}(N)} - \\gamma\\, \\mathrm{KL}(N),
\\]
<p>where \\(\\mathrm{KL}(N) \\approx \\log N - (1 - 1/N)\\) is the KL divergence of the best-of-N policy from the base policy. The first term captures genuine improvement; the second captures overoptimization. There exists an optimal \\(N^*\\) beyond which further optimization hurts.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Iterated RLHF and Constitutional AI</div><div class="env-body">
<p>One approach to reducing alignment tax is <strong>iterative alignment</strong>: alternate between collecting new preference data from the improved model and retraining. Constitutional AI (Bai et al., 2022) takes this further by using the model itself to generate preference labels according to a set of principles ("constitution"), reducing reliance on human annotators while maintaining alignment quality.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-scaling-alignment"></div>
`,
            visualizations: [
                {
                    id: 'viz-scaling-alignment',
                    title: 'Scaling Laws: Base vs. Aligned Models',
                    description: 'Log-log plot of model capability vs. compute for base and aligned models. The alignment tax is a small offset that may be positive (capability loss) or negative (capability gain). Adjust the alignment tax to see how it shifts the curve.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var tax = 0.02;
                        var klBudget = 5.0;

                        var px = 80, py = 40, pw = W - 120, ph = H - 100;

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.lineTo(px + pw, py + ph); ctx.stroke();

                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('log\u2081\u2080(Compute FLOPs)', px + pw / 2, py + ph + 35);
                            ctx.save();
                            ctx.translate(20, py + ph / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Benchmark Score', 0, 0);
                            ctx.restore();

                            // X-axis ticks (log compute)
                            var xMin = 19, xMax = 26;
                            for (var x = xMin; x <= xMax; x++) {
                                var sx = px + (x - xMin) / (xMax - xMin) * pw;
                                ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('10^' + x, sx, py + ph + 16);
                                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(sx, py); ctx.lineTo(sx, py + ph); ctx.stroke();
                            }

                            // Y-axis ticks
                            for (var y = 0; y <= 100; y += 20) {
                                var sy = py + ph - (y / 100) * ph;
                                ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.fillText(y + '%', px - 8, sy + 3);
                                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(px, sy); ctx.lineTo(px + pw, sy); ctx.stroke();
                            }

                            // Base model curve (sigmoid-like in log-compute)
                            ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var i = 0; i <= 200; i++) {
                                var t = i / 200;
                                var lc = xMin + t * (xMax - xMin);
                                var score = 95 / (1 + Math.exp(-1.5 * (lc - 22.5)));
                                var sx2 = px + t * pw;
                                var sy2 = py + ph - (score / 100) * ph;
                                if (i === 0) ctx.moveTo(sx2, sy2); else ctx.lineTo(sx2, sy2);
                            }
                            ctx.stroke();

                            // Aligned model curve
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var j = 0; j <= 200; j++) {
                                var t2 = j / 200;
                                var lc2 = xMin + t2 * (xMax - xMin);
                                var baseScore = 95 / (1 + Math.exp(-1.5 * (lc2 - 22.5)));
                                var alignedScore = Math.min(98, Math.max(0, baseScore * (1 - tax) + (tax < 0 ? -tax * 10 : 0)));
                                var sx3 = px + t2 * pw;
                                var sy3 = py + ph - (alignedScore / 100) * ph;
                                if (j === 0) ctx.moveTo(sx3, sy3); else ctx.lineTo(sx3, sy3);
                            }
                            ctx.stroke();

                            // Overoptimization curve
                            ctx.strokeStyle = viz.colors.red; ctx.lineWidth = 1.5;
                            ctx.setLineDash([5, 3]);
                            ctx.beginPath();
                            var nPts = 100;
                            for (var k = 0; k < nPts; k++) {
                                var kl = (k / nPts) * 15;
                                var gold = 3.0 * Math.sqrt(kl) - 0.3 * kl;
                                var goldNorm = 50 + gold * 5;
                                var sx4 = px + pw * 0.6 + (kl / 15) * pw * 0.35;
                                var sy4 = py + ph - (goldNorm / 100) * ph;
                                if (k === 0) ctx.moveTo(sx4, sy4); else ctx.lineTo(sx4, sy4);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Legend
                            var lx = px + 15, ly = py + 15;
                            var items = [
                                { c: viz.colors.blue, t: 'Base model' },
                                { c: viz.colors.green, t: 'Aligned model (tax=' + (tax * 100).toFixed(0) + '%)' },
                                { c: viz.colors.red, t: 'Overoptimized (Goodhart)' }
                            ];
                            for (var m = 0; m < items.length; m++) {
                                ctx.fillStyle = items[m].c;
                                ctx.fillRect(lx, ly + m * 18, 16, 3);
                                ctx.fillStyle = viz.colors.white; ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText(items[m].t, lx + 22, ly + m * 18 + 4);
                            }

                            // Tax annotation
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            if (tax > 0) {
                                ctx.fillText('Alignment tax: ' + (tax * 100).toFixed(0) + '% capability reduction', W / 2, H - 12);
                            } else if (tax < 0) {
                                ctx.fillStyle = viz.colors.green;
                                ctx.fillText('Alignment bonus: ' + (-tax * 100).toFixed(0) + '% capability gain (instruction-following helps)', W / 2, H - 12);
                            } else {
                                ctx.fillText('Zero alignment tax: no capability change', W / 2, H - 12);
                            }
                        }

                        VizEngine.createSlider(controls, 'Alignment Tax', -0.10, 0.15, 0.02, 0.01, function(v) {
                            tax = v; draw();
                        });

                        draw();
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'An aligned model scores 82% on a reasoning benchmark while the base model scores 85%. Compute the alignment tax. If the aligned model scores 90% on an instruction-following benchmark where the base model scores 70%, compute the alignment tax there.',
                    hint: 'Use \\(\\text{Tax}(B) = (\\text{Score}_{\\text{base}} - \\text{Score}_{\\text{aligned}}) / \\text{Score}_{\\text{base}}\\).',
                    solution: 'Reasoning: Tax = (85 - 82)/85 = 3.5% (positive tax, alignment hurts). Instruction-following: Tax = (70 - 90)/70 = -28.6% (negative tax, i.e., a 28.6% alignment bonus). This illustrates that alignment can help on tasks that require following instructions while imposing a small cost on raw reasoning. The net effect depends on the evaluation distribution.'
                },
                {
                    question: 'Using the overoptimization formula \\(R_{\\text{gold}}(N) = \\alpha\\sqrt{\\mathrm{KL}} - \\gamma\\,\\mathrm{KL}\\), find the optimal KL divergence \\(\\mathrm{KL}^*\\) that maximizes gold reward. Express \\(N^*\\) using \\(\\mathrm{KL} \\approx \\log N\\).',
                    hint: 'Differentiate with respect to KL and set to zero.',
                    solution: 'Let \\(d = \\mathrm{KL}\\). Then \\(R\' = \\frac{\\alpha}{2\\sqrt{d}} - \\gamma = 0 \\Rightarrow \\sqrt{d} = \\frac{\\alpha}{2\\gamma} \\Rightarrow d^* = \\frac{\\alpha^2}{4\\gamma^2}\\). Using \\(d \\approx \\log N\\): \\(N^* \\approx \\exp\\bigl(\\frac{\\alpha^2}{4\\gamma^2}\\bigr)\\). The maximum gold reward is \\(R^* = \\alpha \\cdot \\frac{\\alpha}{2\\gamma} - \\gamma \\cdot \\frac{\\alpha^2}{4\\gamma^2} = \\frac{\\alpha^2}{4\\gamma}\\). Beyond \\(N^*\\), further optimization actually decreases the true reward, a concrete manifestation of Goodhart\'s Law.'
                },
                {
                    question: 'Constitutional AI uses the model itself to generate preference labels. What are the advantages and risks of this approach compared to human-labeled preferences?',
                    hint: 'Consider scalability, cost, consistency, and potential failure modes.',
                    solution: 'Advantages: (1) Vastly more scalable, as no human annotators are needed per example. (2) More consistent, since the model applies the same principles deterministically (human annotators often disagree). (3) Cheaper and faster iteration cycles. Risks: (1) The model may have systematic biases that are amplified through self-reinforcement. (2) The constitution may not capture all aspects of human preferences, leading to alignment gaps. (3) If the model is wrong about what is harmful or helpful, it will systematically train itself to be wrong. (4) The model cannot identify novel harms that were not anticipated in the constitution. Hybrid approaches (model labels + human spot-checks) can mitigate these risks.'
                }
            ]
        }
    ]
});
