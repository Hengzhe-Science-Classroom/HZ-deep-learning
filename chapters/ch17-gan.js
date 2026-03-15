// === Chapter 17: Generative Adversarial Networks ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch17',
    number: 17,
    title: '生成对抗网络 (GANs)',
    subtitle: 'Learning to generate data through an adversarial game between a generator and a discriminator',
    sections: [
        // ========== SECTION 1: The GAN Framework ==========
        {
            id: 'ch17-sec01',
            title: 'The GAN Framework',
            content: `
<h2>17.1 The GAN Framework</h2>

<div class="env-block intuition"><div class="env-title">Intuition: The Counterfeiter and the Detective</div><div class="env-body">
<p>Imagine a counterfeiter trying to produce fake banknotes and a detective trying to distinguish fakes from genuine bills. Initially the counterfeiter is terrible and the detective catches every fake easily. But the counterfeiter studies the detective's feedback and improves; the detective, in turn, becomes more discerning. This arms race continues until the counterfeiter produces notes so perfect that even the best detective cannot tell them apart. A Generative Adversarial Network (GAN) implements exactly this dynamic with two neural networks.</p>
</div></div>

<p>Introduced by Goodfellow et al. (2014), GANs are a framework for training generative models through an adversarial game. Unlike VAEs (Chapter 16), which maximize a variational lower bound on the log-likelihood, GANs sidestep density estimation entirely. The generator never sees the data directly; it only receives gradient signals from the discriminator.</p>

<div class="env-block definition"><div class="env-title">Definition 17.1.1 &mdash; Generator</div><div class="env-body">
<p>The <strong>generator</strong> \\(G_\\theta: \\mathbb{R}^d \\to \\mathbb{R}^D\\) is a neural network parameterized by \\(\\theta\\) that maps a latent vector \\(\\mathbf{z} \\sim p_z(\\mathbf{z})\\) (typically \\(\\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) or \\(\\text{Uniform}(-1,1)^d\\)) to a synthetic data sample \\(\\mathbf{x}_{\\text{fake}} = G_\\theta(\\mathbf{z})\\). The induced distribution over generated samples is denoted \\(p_g\\).</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 17.1.2 &mdash; Discriminator</div><div class="env-body">
<p>The <strong>discriminator</strong> \\(D_\\phi: \\mathbb{R}^D \\to [0,1]\\) is a neural network parameterized by \\(\\phi\\) that takes a data sample \\(\\mathbf{x}\\) and outputs the probability that \\(\\mathbf{x}\\) is real (drawn from the true data distribution \\(p_{\\text{data}}\\)) rather than fake (drawn from \\(p_g\\)).</p>
</div></div>

<h3>The Minimax Game</h3>

<p>Training a GAN is formulated as a two-player minimax game. The discriminator tries to maximize its classification accuracy, while the generator tries to minimize it (equivalently, to maximize the probability that the discriminator is fooled).</p>

<div class="env-block definition"><div class="env-title">Definition 17.1.3 &mdash; GAN Value Function</div><div class="env-body">
<p>The GAN objective is the <strong>minimax value function</strong>:</p>
\\[
\\min_\\theta \\max_\\phi \\; V(G_\\theta, D_\\phi) = \\mathbb{E}_{\\mathbf{x} \\sim p_{\\text{data}}}[\\log D_\\phi(\\mathbf{x})] + \\mathbb{E}_{\\mathbf{z} \\sim p_z}[\\log(1 - D_\\phi(G_\\theta(\\mathbf{z})))]
\\]
<p>The first term rewards the discriminator for assigning high probability to real data. The second term rewards the discriminator for assigning low probability to fake data, while the generator wants to maximize \\(D_\\phi(G_\\theta(\\mathbf{z}))\\), which is equivalent to minimizing \\(\\log(1 - D_\\phi(G_\\theta(\\mathbf{z})))\\).</p>
</div></div>

<h3>Optimal Discriminator</h3>

<div class="env-block theorem"><div class="env-title">Theorem 17.1.4 &mdash; Optimal Discriminator</div><div class="env-body">
<p>For a fixed generator \\(G_\\theta\\), the optimal discriminator is</p>
\\[
D^*_\\phi(\\mathbf{x}) = \\frac{p_{\\text{data}}(\\mathbf{x})}{p_{\\text{data}}(\\mathbf{x}) + p_g(\\mathbf{x})}.
\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>For any fixed \\(G\\), the value function can be written as an integral over \\(\\mathbf{x}\\):</p>
\\[
V = \\int \\bigl[ p_{\\text{data}}(\\mathbf{x}) \\log D(\\mathbf{x}) + p_g(\\mathbf{x}) \\log(1 - D(\\mathbf{x})) \\bigr] d\\mathbf{x}.
\\]
<p>For each \\(\\mathbf{x}\\), maximizing the integrand \\(a \\log y + b \\log(1-y)\\) with \\(a = p_{\\text{data}}(\\mathbf{x})\\), \\(b = p_g(\\mathbf{x})\\), and \\(y = D(\\mathbf{x}) \\in [0,1]\\) gives \\(y^* = a/(a+b)\\) by setting the derivative \\(a/y - b/(1-y) = 0\\).</p>
<div class="qed">&#8718;</div>
</div></div>

<h3>Global Optimality</h3>

<div class="env-block theorem"><div class="env-title">Theorem 17.1.5 &mdash; Global Optimum at \\(p_g = p_{\\text{data}}\\)</div><div class="env-body">
<p>Substituting \\(D^*\\) into \\(V\\), the generator's objective becomes</p>
\\[
C(G) = -\\log 4 + 2 \\cdot D_{\\text{JS}}(p_{\\text{data}} \\| p_g),
\\]
<p>where \\(D_{\\text{JS}}\\) is the Jensen-Shannon divergence. Since \\(D_{\\text{JS}} \\geq 0\\) with equality if and only if \\(p_{\\text{data}} = p_g\\), the global minimum of \\(C(G)\\) is achieved when the generator perfectly replicates the data distribution, at which point \\(D^*(\\mathbf{x}) = 1/2\\) everywhere.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; The Implicit Generative Model</div><div class="env-body">
<p>Unlike VAEs or autoregressive models, the generator \\(G_\\theta\\) defines \\(p_g\\) <em>implicitly</em>: we can sample from \\(p_g\\) (by feeding random \\(\\mathbf{z}\\) through \\(G_\\theta\\)), but we cannot evaluate \\(p_g(\\mathbf{x})\\) for a given \\(\\mathbf{x}\\). This makes GANs powerful generators but unsuitable for tasks requiring density evaluation (anomaly detection, compression).</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-gan-game"></div>
`,
            visualizations: [
                {
                    id: 'viz-gan-game',
                    title: 'GAN Game: Generator vs. Data Distribution',
                    description: 'The green curve shows the real data distribution (a mixture of two Gaussians). The blue curve is the generator distribution, which evolves over training iterations to match the real distribution. Press Play to watch the adversarial training unfold.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        // Real distribution: mixture of two Gaussians
                        function pReal(x) {
                            var s1 = 0.7, s2 = 0.8;
                            var g1 = Math.exp(-0.5 * Math.pow((x - (-2)) / s1, 2)) / (s1 * Math.sqrt(2 * Math.PI));
                            var g2 = Math.exp(-0.5 * Math.pow((x - 2.5) / s2, 2)) / (s2 * Math.sqrt(2 * Math.PI));
                            return 0.45 * g1 + 0.55 * g2;
                        }

                        // Generator distribution: single Gaussian that evolves
                        var genMu1 = 0.5, genSig1 = 2.0;
                        var genMu2 = 0.5, genSig2 = 2.0;
                        var genW = 0.5;
                        var iter = 0;
                        var maxIter = 200;
                        var playing = false;
                        var animId = null;

                        // Target params
                        var tgtMu1 = -2, tgtSig1 = 0.7, tgtMu2 = 2.5, tgtSig2 = 0.8, tgtW = 0.45;

                        function pGen(x) {
                            var g1 = Math.exp(-0.5 * Math.pow((x - genMu1) / genSig1, 2)) / (genSig1 * Math.sqrt(2 * Math.PI));
                            var g2 = Math.exp(-0.5 * Math.pow((x - genMu2) / genSig2, 2)) / (genSig2 * Math.sqrt(2 * Math.PI));
                            return genW * g1 + (1 - genW) * g2;
                        }

                        function updateParams() {
                            var t = Math.min(iter / maxIter, 1);
                            // Smooth easing
                            var e = t * t * (3 - 2 * t);
                            genMu1 = 0.5 + (tgtMu1 - 0.5) * e;
                            genSig1 = 2.0 + (tgtSig1 - 2.0) * e;
                            genMu2 = 0.5 + (tgtMu2 - 0.5) * e;
                            genSig2 = 2.0 + (tgtSig2 - 2.0) * e;
                            genW = 0.5 + (tgtW - 0.5) * e;
                        }

                        function xToScreen(x) { return 60 + (x + 5) / 10 * (W - 120); }
                        function yToScreen(y) { return H - 50 - y * (H - 100) / 0.7; }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            updateParams();

                            // x axis
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(60, H - 50);
                            ctx.lineTo(W - 60, H - 50);
                            ctx.stroke();

                            // Tick marks
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (var tx = -4; tx <= 4; tx += 2) {
                                var sx = xToScreen(tx);
                                ctx.beginPath(); ctx.moveTo(sx, H - 50); ctx.lineTo(sx, H - 45); ctx.stroke();
                                ctx.fillText(tx.toString(), sx, H - 42);
                            }

                            // Draw real distribution (green, filled)
                            ctx.beginPath();
                            ctx.moveTo(xToScreen(-5), yToScreen(0));
                            for (var x = -5; x <= 5; x += 0.05) {
                                ctx.lineTo(xToScreen(x), yToScreen(pReal(x)));
                            }
                            ctx.lineTo(xToScreen(5), yToScreen(0));
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.green + '22';
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var x = -5; x <= 5; x += 0.05) {
                                if (x === -5) ctx.moveTo(xToScreen(x), yToScreen(pReal(x)));
                                else ctx.lineTo(xToScreen(x), yToScreen(pReal(x)));
                            }
                            ctx.stroke();

                            // Draw generator distribution (blue, filled)
                            ctx.beginPath();
                            ctx.moveTo(xToScreen(-5), yToScreen(0));
                            for (var x = -5; x <= 5; x += 0.05) {
                                ctx.lineTo(xToScreen(x), yToScreen(pGen(x)));
                            }
                            ctx.lineTo(xToScreen(5), yToScreen(0));
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.blue + '18';
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var x = -5; x <= 5; x += 0.05) {
                                if (x === -5) ctx.moveTo(xToScreen(x), yToScreen(pGen(x)));
                                else ctx.lineTo(xToScreen(x), yToScreen(pGen(x)));
                            }
                            ctx.stroke();

                            // Discriminator output (orange dashed)
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([5, 4]);
                            ctx.beginPath();
                            for (var x = -5; x <= 5; x += 0.05) {
                                var pr = pReal(x);
                                var pg = pGen(x);
                                var dOpt = (pr + pg) > 1e-8 ? pr / (pr + pg) : 0.5;
                                var dScreen = H - 50 - dOpt * (H - 100) / 1.0;
                                if (x === -5) ctx.moveTo(xToScreen(x), dScreen);
                                else ctx.lineTo(xToScreen(x), dScreen);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Legend
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(W - 200, 18, 14, 3);
                            ctx.fillText('p_data (real)', W - 180, 20);

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(W - 200, 38, 14, 3);
                            ctx.fillText('p_g (generator)', W - 180, 40);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(W - 200, 58, 14, 3);
                            ctx.fillText('D*(x)', W - 180, 60);

                            // Iteration counter
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'left';
                            ctx.fillText('Iteration: ' + iter + ' / ' + maxIter, 14, 20);

                            // JSD estimate
                            var jsd = 0;
                            for (var x = -5; x <= 5; x += 0.05) {
                                var pr = pReal(x);
                                var pg = pGen(x);
                                var m = 0.5 * (pr + pg);
                                if (pr > 1e-10 && m > 1e-10) jsd += pr * Math.log(pr / m) * 0.05;
                                if (pg > 1e-10 && m > 1e-10) jsd += pg * Math.log(pg / m) * 0.05;
                            }
                            jsd *= 0.5;
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillText('JSD: ' + Math.max(0, jsd).toFixed(4), 14, 40);
                        }

                        draw();

                        function step() {
                            if (iter < maxIter) {
                                iter++;
                                draw();
                            }
                            if (iter >= maxIter) {
                                playing = false;
                                playBtn.textContent = 'Play';
                            }
                        }

                        var playBtn = VizEngine.createButton(controls, 'Play', function() {
                            if (playing) {
                                playing = false;
                                playBtn.textContent = 'Play';
                                return;
                            }
                            playing = true;
                            playBtn.textContent = 'Pause';
                            function tick() {
                                if (!playing) return;
                                step();
                                if (playing) animId = setTimeout(tick, 40);
                            }
                            tick();
                        });

                        VizEngine.createButton(controls, 'Reset', function() {
                            playing = false;
                            playBtn.textContent = 'Play';
                            iter = 0;
                            draw();
                        });

                        VizEngine.createSlider(controls, 'Iteration', 0, maxIter, 0, 1, function(v) {
                            iter = Math.round(v);
                            draw();
                        });

                        return { stopAnimation: function() { playing = false; if (animId) clearTimeout(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Derive the optimal discriminator \\(D^*(\\mathbf{x})\\) from scratch. Starting from the value function \\(V(G, D)\\), fix \\(G\\) and find the \\(D\\) that maximizes \\(V\\) pointwise.',
                    hint: 'For each \\(\\mathbf{x}\\), maximize \\(f(y) = a \\log y + b \\log(1-y)\\) over \\(y \\in (0,1)\\), where \\(a = p_{\\text{data}}(\\mathbf{x})\\) and \\(b = p_g(\\mathbf{x})\\).',
                    solution: 'Taking the derivative: \\(f\'(y) = a/y - b/(1-y) = 0\\) gives \\(a(1-y) = by\\), so \\(y^* = a/(a+b)\\). The second derivative \\(f\'\'(y) = -a/y^2 - b/(1-y)^2 < 0\\) confirms this is a maximum. Substituting \\(a = p_{\\text{data}}(\\mathbf{x})\\) and \\(b = p_g(\\mathbf{x})\\) yields \\(D^*(\\mathbf{x}) = p_{\\text{data}}(\\mathbf{x}) / (p_{\\text{data}}(\\mathbf{x}) + p_g(\\mathbf{x}))\\).'
                },
                {
                    question: 'Show that at the global optimum \\(p_g = p_{\\text{data}}\\), the value of the game is \\(V(G^*, D^*) = -\\log 4\\) and the optimal discriminator outputs \\(1/2\\) everywhere.',
                    hint: 'Substitute \\(p_g = p_{\\text{data}}\\) into \\(D^*\\) and then evaluate \\(V\\).',
                    solution: 'When \\(p_g = p_{\\text{data}}\\), the optimal discriminator becomes \\(D^*(\\mathbf{x}) = p_{\\text{data}} / (p_{\\text{data}} + p_{\\text{data}}) = 1/2\\) for all \\(\\mathbf{x}\\). Substituting into \\(V\\): \\(V = \\mathbb{E}[\\log(1/2)] + \\mathbb{E}[\\log(1/2)] = \\log(1/2) + \\log(1/2) = -2\\log 2 = -\\log 4\\).'
                },
                {
                    question: 'Explain why GANs are called "implicit" generative models. What can a GAN do that a VAE cannot, and what can a VAE do that a GAN cannot?',
                    hint: 'Think about the difference between being able to sample from a distribution vs. being able to evaluate its density.',
                    solution: 'A GAN is "implicit" because it defines a distribution \\(p_g\\) by specifying a sampling procedure (\\(\\mathbf{z} \\to G(\\mathbf{z})\\)) without providing a formula for \\(p_g(\\mathbf{x})\\). A GAN can generate sharp, high-quality samples (no blurriness from reconstruction loss), but it cannot evaluate the likelihood of a given sample. A VAE can evaluate \\(\\log p_\\theta(\\mathbf{x})\\) (via ELBO), enabling anomaly detection, compression, and model comparison via held-out likelihood. However, VAE samples tend to be blurrier because the Gaussian decoder assumption encourages averaging over modes.'
                }
            ]
        },

        // ========== SECTION 2: GAN Training ==========
        {
            id: 'ch17-sec02',
            title: 'GAN Training',
            content: `
<h2>17.2 GAN Training</h2>

<div class="env-block intuition"><div class="env-title">Intuition: Walking a Tightrope</div><div class="env-body">
<p>Training a GAN is notoriously difficult. The generator and discriminator must improve at roughly the same pace. If the discriminator becomes too strong, it provides perfect classification with zero gradient, and the generator learns nothing. If the generator improves too fast, it can exploit weaknesses in the discriminator without actually learning the data distribution. Stable GAN training is like walking a tightrope between these two failure modes.</p>
</div></div>

<h3>Alternating Gradient Descent</h3>

<p>In practice, training alternates between updating the discriminator and the generator. Each training iteration consists of:</p>
<ol>
<li><strong>Discriminator step</strong>: Sample a minibatch \\(\\{\\mathbf{x}^{(i)}\\}\\) from the data and \\(\\{\\mathbf{z}^{(i)}\\}\\) from the prior. Update \\(\\phi\\) by ascending:
\\[
\\nabla_\\phi \\frac{1}{m} \\sum_{i=1}^m \\bigl[ \\log D_\\phi(\\mathbf{x}^{(i)}) + \\log(1 - D_\\phi(G_\\theta(\\mathbf{z}^{(i)}))) \\bigr]
\\]</li>
<li><strong>Generator step</strong>: Sample a fresh \\(\\{\\mathbf{z}^{(i)}\\}\\). Update \\(\\theta\\) by descending:
\\[
\\nabla_\\theta \\frac{1}{m} \\sum_{i=1}^m \\log(1 - D_\\phi(G_\\theta(\\mathbf{z}^{(i)})))
\\]</li>
</ol>

<div class="env-block definition"><div class="env-title">Definition 17.2.1 &mdash; Non-Saturating Generator Loss</div><div class="env-body">
<p>In the original formulation, the generator minimizes \\(\\log(1 - D(G(\\mathbf{z})))\\). Early in training, when \\(D(G(\\mathbf{z})) \\approx 0\\), this function is flat (saturated), providing vanishingly small gradients. Goodfellow et al. (2014) proposed the <strong>non-saturating</strong> alternative: maximize \\(\\log D(G(\\mathbf{z}))\\) instead. The gradient of \\(-\\log(1-y)\\) near \\(y=0\\) is \\(1/(1-y) \\approx 1\\), while the gradient of \\(\\log(y)\\) near \\(y=0\\) is \\(1/y \\to \\infty\\), providing much stronger learning signal.</p>
</div></div>

<h3>Nash Equilibrium</h3>

<div class="env-block definition"><div class="env-title">Definition 17.2.2 &mdash; Nash Equilibrium in GANs</div><div class="env-body">
<p>A <strong>Nash equilibrium</strong> of the GAN game is a pair \\((\\theta^*, \\phi^*)\\) such that neither player can improve their objective by unilaterally changing their strategy:</p>
\\[
V(G_{\\theta^*}, D_{\\phi^*}) \\leq V(G_{\\theta^*}, D_\\phi) \\quad \\forall\\,\\phi, \\qquad V(G_{\\theta^*}, D_{\\phi^*}) \\leq V(G_\\theta, D_{\\phi^*}) \\quad \\forall\\,\\theta.
\\]
<p>The global Nash equilibrium corresponds to \\(p_g = p_{\\text{data}}\\) and \\(D^* = 1/2\\). However, gradient descent does not guarantee convergence to Nash equilibria in non-convex games.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Why SGD May Not Converge</div><div class="env-body">
<p>Consider the simple game \\(\\min_x \\max_y \\; xy\\). Simultaneous gradient descent gives \\(\\dot{x} = -y\\) and \\(\\dot{y} = x\\), which traces circles around the equilibrium \\((0,0)\\) instead of converging to it. This oscillatory behavior is intrinsic to minimax games and is a core reason why GAN training is unstable. The dynamics can exhibit limit cycles, divergence, or chaotic behavior even in simple settings.</p>
</div></div>

<h3>Practical Training Tricks</h3>

<p>Over the years, practitioners have accumulated a catalog of heuristics for stabilizing GAN training:</p>
<ul>
<li><strong>Train D more than G</strong>: Typically \\(k = 1\\) to \\(5\\) discriminator updates per generator update, keeping the discriminator strong enough to provide useful gradients.</li>
<li><strong>Label smoothing</strong>: Replace the "real" label 1 with 0.9 and "fake" label 0 with 0.1, preventing the discriminator from becoming overconfident.</li>
<li><strong>Spectral normalization</strong> (Miyato et al., 2018): Normalize each weight matrix by its spectral norm (largest singular value) to control the Lipschitz constant of D.</li>
<li><strong>Two time-scale learning rates</strong>: Use a smaller learning rate for G than for D (or vice versa) to balance the adversarial dynamics.</li>
</ul>

<div class="env-block warning"><div class="env-title">Warning &mdash; The Vanishing Gradient Problem</div><div class="env-body">
<p>If the discriminator is too good (\\(D(G(\\mathbf{z})) \\approx 0\\) for all generated samples), the generator receives near-zero gradients under the saturating loss \\(\\log(1 - D(G(\\mathbf{z})))\\). Even with the non-saturating loss \\(-\\log D(G(\\mathbf{z}))\\), an overly confident discriminator can still cause instability because the JS divergence saturates when \\(p_{\\text{data}}\\) and \\(p_g\\) have disjoint supports (which is common in high dimensions).</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-gan-loss"></div>
`,
            visualizations: [
                {
                    id: 'viz-gan-loss',
                    title: 'GAN Training Dynamics: Loss Curves',
                    description: 'Simulated discriminator and generator loss curves during GAN training. Observe the characteristic oscillations and the phases of training: initial discriminator dominance, generator catch-up, and eventual (idealized) convergence.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var maxEpoch = 300;
                        var epoch = 0;
                        var playing = false;
                        var animId = null;

                        // Pre-generate realistic loss curves with oscillation
                        var dLosses = [];
                        var gLosses = [];
                        var seed = 12345;
                        function seededRandom() {
                            seed = (seed * 16807 + 0) % 2147483647;
                            return (seed - 1) / 2147483646;
                        }
                        // Generate loss curves
                        var dL = 0.7, gL = 2.5;
                        for (var i = 0; i <= maxEpoch; i++) {
                            var t = i / maxEpoch;
                            // D loss: starts high, drops, then oscillates near log(4) ~ 1.386
                            var dTarget = 1.386;
                            dL = dL + 0.03 * (dTarget - dL) + (seededRandom() - 0.5) * 0.08;
                            dL = Math.max(0.3, Math.min(2.5, dL));
                            dLosses.push(dL);

                            // G loss: starts very high, decreases with oscillation
                            var gTarget = 1.386;
                            gL = gL + 0.025 * (gTarget - gL) + (seededRandom() - 0.5) * 0.12;
                            gL = Math.max(0.3, Math.min(4.0, gL));
                            gLosses.push(gL);
                        }

                        function epochToX(e) { return 70 + (e / maxEpoch) * (W - 100); }
                        function lossToY(l) { return H - 50 - (l / 4.5) * (H - 90); }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(70, 30); ctx.lineTo(70, H - 50); ctx.lineTo(W - 30, H - 50); ctx.stroke();

                            // Y axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var yl = 0; yl <= 4; yl++) {
                                var yy = lossToY(yl);
                                ctx.fillText(yl.toFixed(0), 62, yy);
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(70, yy); ctx.lineTo(W - 30, yy); ctx.stroke();
                            }

                            // X axis labels
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (var xl = 0; xl <= maxEpoch; xl += 50) {
                                ctx.fillText(xl.toString(), epochToX(xl), H - 42);
                            }

                            // Axis titles
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'center';
                            ctx.fillText('Epoch', W / 2, H - 18);

                            ctx.save();
                            ctx.translate(18, H / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Loss', 0, 0);
                            ctx.restore();

                            // Equilibrium line at log(4)
                            ctx.strokeStyle = viz.colors.yellow + '66';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            var eqY = lossToY(1.386);
                            ctx.beginPath(); ctx.moveTo(70, eqY); ctx.lineTo(W - 30, eqY); ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('-log(4) equilibrium', W - 170, eqY - 8);

                            // Phase annotations
                            var phases = [
                                { start: 0, end: 0.15, label: 'D dominates', color: viz.colors.red + '33' },
                                { start: 0.15, end: 0.5, label: 'G catches up', color: viz.colors.blue + '22' },
                                { start: 0.5, end: 1.0, label: 'Oscillating equilibrium', color: viz.colors.green + '18' }
                            ];
                            for (var pi = 0; pi < phases.length; pi++) {
                                var p = phases[pi];
                                if (epoch / maxEpoch >= p.start) {
                                    var pEnd = Math.min(p.end, epoch / maxEpoch);
                                    ctx.fillStyle = p.color;
                                    ctx.fillRect(epochToX(p.start * maxEpoch), 30, epochToX(pEnd * maxEpoch) - epochToX(p.start * maxEpoch), H - 80);
                                    if (epoch / maxEpoch > (p.start + p.end) / 2) {
                                        ctx.fillStyle = viz.colors.text;
                                        ctx.font = '10px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.fillText(p.label, epochToX((p.start + Math.min(p.end, epoch / maxEpoch)) / 2 * maxEpoch), 40);
                                    }
                                }
                            }

                            // D loss curve (orange)
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i <= epoch; i++) {
                                var x = epochToX(i);
                                var y = lossToY(dLosses[i]);
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            }
                            ctx.stroke();

                            // G loss curve (blue)
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (var i = 0; i <= epoch; i++) {
                                var x = epochToX(i);
                                var y = lossToY(gLosses[i]);
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            }
                            ctx.stroke();

                            // Legend
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';

                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 3;
                            ctx.beginPath(); ctx.moveTo(W - 200, 20); ctx.lineTo(W - 175, 20); ctx.stroke();
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('D loss', W - 170, 20);

                            ctx.strokeStyle = viz.colors.blue;
                            ctx.beginPath(); ctx.moveTo(W - 200, 38); ctx.lineTo(W - 175, 38); ctx.stroke();
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText('G loss', W - 170, 38);

                            // Epoch counter
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'left';
                            ctx.fillText('Epoch: ' + epoch, 80, 20);
                        }

                        draw();

                        var playBtn = VizEngine.createButton(controls, 'Play', function() {
                            if (playing) { playing = false; playBtn.textContent = 'Play'; return; }
                            playing = true;
                            playBtn.textContent = 'Pause';
                            function tick() {
                                if (!playing) return;
                                if (epoch < maxEpoch) { epoch++; draw(); animId = setTimeout(tick, 30); }
                                else { playing = false; playBtn.textContent = 'Play'; }
                            }
                            tick();
                        });

                        VizEngine.createButton(controls, 'Reset', function() {
                            playing = false; playBtn.textContent = 'Play'; epoch = 0; draw();
                        });

                        VizEngine.createSlider(controls, 'Epoch', 0, maxEpoch, 0, 1, function(v) {
                            epoch = Math.round(v); draw();
                        });

                        return { stopAnimation: function() { playing = false; if (animId) clearTimeout(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the non-saturating generator loss \\(-\\mathbb{E}[\\log D(G(\\mathbf{z}))]\\) and the original \\(\\mathbb{E}[\\log(1 - D(G(\\mathbf{z})))]\\) have the same fixed point but different gradient dynamics. Compute the gradient of each with respect to the generator output \\(D(G(\\mathbf{z}))\\) when \\(D(G(\\mathbf{z})) \\to 0\\).',
                    hint: 'Let \\(p = D(G(\\mathbf{z}))\\). Compare \\(d/dp [\\log(1-p)]\\) and \\(d/dp [-\\log p]\\) as \\(p \\to 0\\).',
                    solution: 'Both losses have the same optimum: \\(D(G(\\mathbf{z})) = 1\\) (discriminator completely fooled). For the saturating loss: \\(d/dp \\log(1-p) = -1/(1-p)\\). At \\(p \\to 0\\), this is \\(-1\\), a weak gradient. For the non-saturating loss: \\(d/dp [-\\log p] = -1/p\\). At \\(p \\to 0\\), this diverges to \\(-\\infty\\), providing a very strong gradient. Early in training when the generator is poor (\\(p \\approx 0\\)), the non-saturating loss gives much larger gradients, enabling faster learning.'
                },
                {
                    question: 'Consider the simple bilinear game \\(\\min_x \\max_y \\; xy\\). Show that simultaneous gradient descent with learning rate \\(\\eta\\) leads to oscillations. What is the trajectory?',
                    hint: 'Write the update equations \\(x_{t+1} = x_t - \\eta y_t\\), \\(y_{t+1} = y_t + \\eta x_t\\). Write this as a matrix iteration and find the eigenvalues.',
                    solution: 'The update is \\(\\begin{pmatrix} x_{t+1} \\\\ y_{t+1} \\end{pmatrix} = \\begin{pmatrix} 1 & -\\eta \\\\ \\eta & 1 \\end{pmatrix} \\begin{pmatrix} x_t \\\\ y_t \\end{pmatrix}\\). The eigenvalues are \\(1 \\pm i\\eta\\), with magnitude \\(\\sqrt{1 + \\eta^2} > 1\\). So the iterates spiral outward with exponentially growing radius. The system not only fails to converge to the Nash equilibrium \\((0,0)\\) but actually diverges. This demonstrates why naive simultaneous gradient descent is insufficient for minimax games.'
                },
                {
                    question: 'A practitioner observes that the discriminator loss quickly drops to near zero and stays there, while the generator loss remains high. Diagnose the problem and propose two solutions.',
                    hint: 'What happens to the generator gradient when the discriminator is perfect?',
                    solution: 'The discriminator has become too strong: it can perfectly distinguish real from fake, so \\(D(G(\\mathbf{z})) \\approx 0\\) for all fakes. The generator gradients vanish (with the saturating loss) or become very noisy (even with the non-saturating loss, the log-probability landscape is extremely steep). Solutions: (1) Reduce the discriminator learning rate or train it fewer steps per generator step, allowing the generator to keep pace. (2) Apply spectral normalization or gradient penalty to the discriminator to limit its capacity. (3) Use label smoothing (replace 1 with 0.9 for real labels) to prevent the discriminator from becoming overconfident.'
                }
            ]
        },

        // ========== SECTION 3: Mode Collapse ==========
        {
            id: 'ch17-sec03',
            title: 'Mode Collapse',
            content: `
<h2>17.3 Mode Collapse</h2>

<div class="env-block intuition"><div class="env-title">Intuition: The Lazy Counterfeiter</div><div class="env-body">
<p>Suppose the counterfeiter discovers that a certain type of fake banknote consistently fools the detective. Rather than learning to forge all denominations, the counterfeiter keeps producing only that one type. The detective eventually catches on and rejects it, so the counterfeiter switches to another type, and the cycle repeats. This is <strong>mode collapse</strong>: the generator learns to produce only a small subset of the data distribution's modes, cycling between them without ever covering them all simultaneously.</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 17.3.1 &mdash; Mode Collapse</div><div class="env-body">
<p><strong>Mode collapse</strong> (also called the <em>Helvetica scenario</em>) occurs when the generator maps many different latent codes \\(\\mathbf{z}\\) to the same or very similar outputs, thereby covering only a few modes of the true data distribution \\(p_{\\text{data}}\\). Formally, if \\(p_{\\text{data}}\\) has \\(K\\) well-separated modes, mode collapse means \\(p_g\\) places most of its mass on \\(k \\ll K\\) of them.</p>
</div></div>

<h3>Why Mode Collapse Happens</h3>

<p>Mode collapse is a consequence of the minimax game dynamics. Consider a generator that finds a single point \\(\\mathbf{x}^*\\) that maximally fools the current discriminator. The generator has a strong incentive to map <em>all</em> latent codes to \\(\\mathbf{x}^*\\), because this minimizes the generator loss regardless of what the discriminator does to other points.</p>

<div class="env-block theorem"><div class="env-title">Theorem 17.3.2 &mdash; Mode Collapse as Max-Min vs. Min-Max</div><div class="env-body">
<p>The theoretical GAN objective is \\(\\min_G \\max_D V(G, D)\\). If instead we solve \\(\\max_D \\min_G V(G, D)\\) (reversing the order), the optimal generator for a fixed \\(D\\) concentrates all its mass on \\(\\arg\\max_\\mathbf{x} D(\\mathbf{x})\\). In practice, alternating optimization with a fixed discriminator approximates the inner loop \\(\\min_G\\) more aggressively than the outer \\(\\max_D\\), leading the generator toward this degenerate solution.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Partial vs. Full Mode Collapse</div><div class="env-body">
<p><strong>Full mode collapse</strong> means the generator outputs (nearly) the same sample for all \\(\\mathbf{z}\\). <strong>Partial mode collapse</strong> means the generator covers some modes but not others. Partial mode collapse is far more common in practice and can be difficult to detect without careful evaluation metrics (e.g., Frechet Inception Distance, precision/recall).</p>
</div></div>

<h3>Mitigating Mode Collapse</h3>

<p>Several approaches have been proposed to combat mode collapse:</p>
<ul>
<li><strong>Minibatch discrimination</strong> (Salimans et al., 2016): Provide the discriminator with information about the <em>batch</em> of generated samples, not just individual samples. If many generated samples are identical, the discriminator can detect this.</li>
<li><strong>Unrolled GANs</strong> (Metz et al., 2017): When computing the generator gradient, "unroll" \\(k\\) steps of discriminator optimization. This gives the generator a better estimate of the fully-trained discriminator's response, discouraging short-sighted mode-dropping.</li>
<li><strong>Feature matching</strong>: Train the generator to match statistics (mean activations) of intermediate discriminator features on real vs. fake data, rather than trying to maximize the discriminator's output directly.</li>
<li><strong>Wasserstein distance</strong> (Section 17.4): The Wasserstein distance does not saturate when supports are disjoint, providing more stable gradients that reduce mode collapse.</li>
</ul>

<div class="env-block example"><div class="env-title">Example 17.3.3 &mdash; Diagnosing Mode Collapse</div><div class="env-body">
<p>Suppose we train a GAN on MNIST (10 digit classes). After training, we generate 10,000 samples and classify them with a pre-trained classifier. If the generated distribution assigns 98% of mass to digits 1 and 7 and essentially zero to other digits, the GAN has suffered partial mode collapse onto 2 of 10 modes. A well-trained GAN should produce roughly uniform coverage of all 10 classes.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-mode-collapse"></div>
`,
            visualizations: [
                {
                    id: 'viz-mode-collapse',
                    title: 'Mode Collapse: Generator Fails to Cover All Modes',
                    description: 'The green curve is the real data distribution with three modes. The blue curve shows the generator. Use the slider to transition from healthy training (all modes covered) to full mode collapse (only one mode). Watch how the generator mass concentrates.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        // Real distribution: 3 modes
                        var modes = [
                            { mu: -3, sig: 0.5, w: 0.33 },
                            { mu: 0, sig: 0.6, w: 0.34 },
                            { mu: 3, sig: 0.5, w: 0.33 }
                        ];

                        function gaussian(x, mu, sig) {
                            return Math.exp(-0.5 * Math.pow((x - mu) / sig, 2)) / (sig * Math.sqrt(2 * Math.PI));
                        }

                        function pReal(x) {
                            var s = 0;
                            for (var i = 0; i < modes.length; i++) {
                                s += modes[i].w * gaussian(x, modes[i].mu, modes[i].sig);
                            }
                            return s;
                        }

                        var collapseLevel = 0; // 0 = healthy, 1 = full collapse

                        function pGen(x) {
                            var t = collapseLevel;
                            // Interpolate: at t=0, 3 modes; at t=0.5, 2 modes; at t=1, 1 mode
                            var w1 = 0.33 * (1 - t) + 0.0 * t;
                            var w2 = 0.34 * (1 - t) + 0.0 * t;
                            var w3 = 0.33 * (1 - t) + 1.0 * t;
                            var s1 = 0.5 * (1 - t) + 0.3 * t;
                            var s2 = 0.6 * (1 - t) + 0.3 * t;
                            var s3 = 0.5 * (1 - t) + 0.35 * t;
                            return w1 * gaussian(x, -3, s1) + w2 * gaussian(x, 0, s2) + w3 * gaussian(x, 3, s3);
                        }

                        function xToScreen(x) { return 70 + (x + 6) / 12 * (W - 120); }
                        function yToScreen(y) { return H - 55 - y * (H - 100) / 0.65; }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Axis
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(70, H - 55);
                            ctx.lineTo(W - 50, H - 55);
                            ctx.stroke();

                            // Ticks
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (var tx = -5; tx <= 5; tx += 1) {
                                var sx = xToScreen(tx);
                                ctx.fillText(tx.toString(), sx, H - 48);
                            }

                            // Real distribution
                            ctx.beginPath();
                            ctx.moveTo(xToScreen(-6), yToScreen(0));
                            for (var x = -6; x <= 6; x += 0.05) {
                                ctx.lineTo(xToScreen(x), yToScreen(pReal(x)));
                            }
                            ctx.lineTo(xToScreen(6), yToScreen(0));
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.green + '20';
                            ctx.fill();

                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var x = -6; x <= 6; x += 0.05) {
                                if (x === -6) ctx.moveTo(xToScreen(x), yToScreen(pReal(x)));
                                else ctx.lineTo(xToScreen(x), yToScreen(pReal(x)));
                            }
                            ctx.stroke();

                            // Generator distribution
                            ctx.beginPath();
                            ctx.moveTo(xToScreen(-6), yToScreen(0));
                            for (var x = -6; x <= 6; x += 0.05) {
                                ctx.lineTo(xToScreen(x), yToScreen(pGen(x)));
                            }
                            ctx.lineTo(xToScreen(6), yToScreen(0));
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.blue + '18';
                            ctx.fill();

                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var x = -6; x <= 6; x += 0.05) {
                                if (x === -6) ctx.moveTo(xToScreen(x), yToScreen(pGen(x)));
                                else ctx.lineTo(xToScreen(x), yToScreen(pGen(x)));
                            }
                            ctx.stroke();

                            // Highlight dropped modes with X markers
                            if (collapseLevel > 0.3) {
                                var droppedModes = [];
                                if (collapseLevel > 0.3) droppedModes.push(modes[0]);
                                if (collapseLevel > 0.6) droppedModes.push(modes[1]);

                                for (var di = 0; di < droppedModes.length; di++) {
                                    var dm = droppedModes[di];
                                    var opacity = Math.min(1, (collapseLevel - (di === 0 ? 0.3 : 0.6)) / 0.3);
                                    var cx = xToScreen(dm.mu);
                                    var cy = yToScreen(pReal(dm.mu)) - 20;
                                    ctx.strokeStyle = viz.colors.red;
                                    ctx.globalAlpha = opacity;
                                    ctx.lineWidth = 3;
                                    ctx.beginPath(); ctx.moveTo(cx - 8, cy - 8); ctx.lineTo(cx + 8, cy + 8); ctx.stroke();
                                    ctx.beginPath(); ctx.moveTo(cx + 8, cy - 8); ctx.lineTo(cx - 8, cy + 8); ctx.stroke();
                                    ctx.globalAlpha = 1;
                                }
                            }

                            // Labels
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(W - 210, 18, 14, 3);
                            ctx.fillText('p_data (3 modes)', W - 190, 20);

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(W - 210, 38, 14, 3);
                            ctx.fillText('p_g (generator)', W - 190, 40);

                            // Status
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'left';
                            var status = 'Healthy: all modes covered';
                            if (collapseLevel > 0.3 && collapseLevel <= 0.65) status = 'Partial collapse: mode 1 dropped';
                            else if (collapseLevel > 0.65) status = 'Severe collapse: only mode 3 remains';
                            ctx.fillText(status, 14, 20);

                            // KL estimate
                            var kl = 0;
                            for (var x = -6; x <= 6; x += 0.05) {
                                var pr = pReal(x);
                                var pg = pGen(x);
                                if (pr > 1e-10 && pg > 1e-10) {
                                    kl += pr * Math.log(pr / pg) * 0.05;
                                }
                            }
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillText('KL(p_data || p_g): ' + Math.max(0, kl).toFixed(3), 14, 40);
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Collapse Level', 0, 1, 0, 0.01, function(v) {
                            collapseLevel = v;
                            draw();
                        });

                        VizEngine.createButton(controls, 'Healthy', function() { collapseLevel = 0; draw(); });
                        VizEngine.createButton(controls, 'Partial', function() { collapseLevel = 0.5; draw(); });
                        VizEngine.createButton(controls, 'Full Collapse', function() { collapseLevel = 1.0; draw(); });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Explain the connection between mode collapse and the order of min/max in the GAN objective. What happens when we solve \\(\\max_D \\min_G V(G, D)\\) instead of \\(\\min_G \\max_D V(G, D)\\)?',
                    hint: 'For a fixed \\(D\\), what is the generator that minimizes \\(\\mathbb{E}_{\\mathbf{z}}[\\log(1 - D(G(\\mathbf{z})))]\\)?',
                    solution: 'For a fixed \\(D\\), the generator minimizes \\(\\mathbb{E}[\\log(1-D(G(\\mathbf{z})))]\\) by mapping every \\(\\mathbf{z}\\) to \\(\\mathbf{x}^* = \\arg\\max_{\\mathbf{x}} D(\\mathbf{x})\\), since \\(\\log(1-D(\\cdot))\\) is minimized where \\(D(\\cdot)\\) is maximized. This gives a degenerate generator whose output distribution is a delta function \\(\\delta_{\\mathbf{x}^*}\\). In \\(\\min_G \\max_D\\), the max over \\(D\\) sees the full generator distribution and must detect all modes. But in \\(\\max_D \\min_G\\), the inner minimization over \\(G\\) collapses to a point for each fixed \\(D\\). Since alternating optimization approximates the inner loop more than the outer loop, practical GANs lean toward the max-min solution, causing mode collapse.'
                },
                {
                    question: 'Describe minibatch discrimination as a solution to mode collapse. Why does giving the discriminator access to batch-level statistics help?',
                    hint: 'Think about what information is lost when the discriminator sees only one sample at a time vs. an entire batch.',
                    solution: 'Without minibatch discrimination, the discriminator evaluates each sample independently. If the generator produces a single high-quality sample and replicates it, the discriminator sees identical "good" samples and cannot detect the lack of diversity. With minibatch discrimination, the discriminator computes pairwise distances (or other statistics) across samples in a minibatch. A batch of identical or near-identical fake samples will have very different batch statistics than a batch of diverse real samples. This gives the discriminator a new signal to penalize the generator for low diversity, even when individual samples look realistic.'
                },
                {
                    question: 'Suppose \\(p_{\\text{data}}\\) is a uniform mixture of 10 Gaussians, \\(p_{\\text{data}} = \\frac{1}{10}\\sum_{k=1}^{10} \\mathcal{N}(\\mu_k, \\sigma^2 I)\\), and the generator has collapsed to a single Gaussian \\(p_g = \\mathcal{N}(\\mu_1, \\sigma^2 I)\\). Compute the Jensen-Shannon divergence \\(D_{\\text{JS}}(p_{\\text{data}} \\| p_g)\\) approximately, assuming the modes are well-separated (negligible overlap).',
                    hint: 'When modes are well-separated, \\(p_{\\text{data}}\\) near mode \\(k\\) is \\(\\approx \\frac{1}{10}\\mathcal{N}(\\mu_k, \\sigma^2 I)\\) and \\(p_g\\) near mode 1 is \\(\\approx \\mathcal{N}(\\mu_1, \\sigma^2 I)\\), while \\(p_g \\approx 0\\) near modes \\(2,\\ldots,10\\).',
                    solution: 'Let \\(m = (p_{\\text{data}} + p_g)/2\\). Near mode 1: \\(p_{\\text{data}} \\approx p_1/10\\), \\(p_g \\approx p_1\\), so \\(m \\approx 11p_1/20\\). The KL contribution from \\(p_{\\text{data}}\\) near mode 1 is \\(\\frac{1}{10}\\ln\\frac{p_1/10}{11p_1/20} = \\frac{1}{10}\\ln\\frac{2}{11}\\). Near modes 2-10: \\(p_{\\text{data}} \\approx p_k/10\\), \\(p_g \\approx 0\\), \\(m \\approx p_k/20\\), giving \\(\\frac{9}{10}\\ln\\frac{p_k/10}{p_k/20} = \\frac{9}{10}\\ln 2\\). Similarly for \\(D_{\\text{KL}}(p_g \\| m)\\): near mode 1, \\(1 \\cdot \\ln\\frac{p_1}{11p_1/20} = \\ln\\frac{20}{11}\\). Therefore \\(D_{\\text{JS}} = \\frac{1}{2}[\\frac{1}{10}\\ln\\frac{2}{11} + \\frac{9}{10}\\ln 2 + \\ln\\frac{20}{11}] \\approx \\frac{1}{2}[-0.170 + 0.624 + 0.598] \\approx 0.526\\). For reference, \\(D_{\\text{JS}} \\in [0, \\ln 2] \\approx [0, 0.693]\\), so the collapse is severe.'
                }
            ]
        },

        // ========== SECTION 4: WGAN & Training Stability ==========
        {
            id: 'ch17-sec04',
            title: 'WGAN & Training Stability',
            content: `
<h2>17.4 WGAN &amp; Training Stability</h2>

<div class="env-block intuition"><div class="env-title">Intuition: A Better Measuring Stick</div><div class="env-body">
<p>The original GAN minimizes the Jensen-Shannon divergence (JSD) between the real and generated distributions. But JSD has a fundamental flaw: when the two distributions have non-overlapping supports (which is common in high dimensions, as two manifolds generically do not intersect), JSD saturates at \\(\\log 2\\) and its gradient vanishes. The Wasserstein distance does not have this problem; it measures the "cost of transporting" mass from one distribution to the other, and it provides meaningful gradients even when supports are disjoint.</p>
</div></div>

<h3>The Wasserstein Distance</h3>

<div class="env-block definition"><div class="env-title">Definition 17.4.1 &mdash; Wasserstein-1 Distance (Earth Mover's Distance)</div><div class="env-body">
<p>The <strong>Wasserstein-1 distance</strong> between two distributions \\(\\mathbb{P}_r\\) and \\(\\mathbb{P}_g\\) is</p>
\\[
W_1(\\mathbb{P}_r, \\mathbb{P}_g) = \\inf_{\\gamma \\in \\Pi(\\mathbb{P}_r, \\mathbb{P}_g)} \\mathbb{E}_{(\\mathbf{x}, \\mathbf{y}) \\sim \\gamma}[\\|\\mathbf{x} - \\mathbf{y}\\|],
\\]
<p>where \\(\\Pi(\\mathbb{P}_r, \\mathbb{P}_g)\\) is the set of all joint distributions (couplings) whose marginals are \\(\\mathbb{P}_r\\) and \\(\\mathbb{P}_g\\). Intuitively, \\(\\gamma(\\mathbf{x}, \\mathbf{y})\\) describes how much mass is transported from \\(\\mathbf{x}\\) to \\(\\mathbf{y}\\), and \\(W_1\\) is the minimum total transportation cost.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 17.4.2 &mdash; Kantorovich-Rubinstein Duality</div><div class="env-body">
<p>The Wasserstein-1 distance admits the dual representation</p>
\\[
W_1(\\mathbb{P}_r, \\mathbb{P}_g) = \\sup_{\\|f\\|_L \\leq 1} \\left[ \\mathbb{E}_{\\mathbf{x} \\sim \\mathbb{P}_r}[f(\\mathbf{x})] - \\mathbb{E}_{\\mathbf{x} \\sim \\mathbb{P}_g}[f(\\mathbf{x})] \\right],
\\]
<p>where the supremum is over all 1-Lipschitz functions \\(f: \\mathbb{R}^D \\to \\mathbb{R}\\), i.e., \\(|f(\\mathbf{x}) - f(\\mathbf{y})| \\leq \\|\\mathbf{x} - \\mathbf{y}\\|\\) for all \\(\\mathbf{x}, \\mathbf{y}\\).</p>
</div></div>

<h3>The WGAN Objective</h3>

<div class="env-block definition"><div class="env-title">Definition 17.4.3 &mdash; WGAN Objective</div><div class="env-body">
<p>The <strong>Wasserstein GAN</strong> (Arjovsky et al., 2017) replaces the discriminator with a <strong>critic</strong> \\(f_\\phi\\) (not passed through sigmoid) and optimizes:</p>
\\[
\\min_\\theta \\max_{\\|f_\\phi\\|_L \\leq 1} \\left[ \\mathbb{E}_{\\mathbf{x} \\sim p_{\\text{data}}}[f_\\phi(\\mathbf{x})] - \\mathbb{E}_{\\mathbf{z} \\sim p_z}[f_\\phi(G_\\theta(\\mathbf{z}))] \\right].
\\]
<p>Note that the critic output is unbounded (no sigmoid), and the Lipschitz constraint replaces the implicit constraint in standard GANs. The critic no longer outputs a probability but a "realness score."</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Why Wasserstein is Better</div><div class="env-body">
<p>Consider learning \\(p_\\theta = \\delta_{\\theta}\\) to match \\(p_{\\text{data}} = \\delta_0\\) in 1D. The JSD is \\(\\log 2\\) for all \\(\\theta \\neq 0\\) and 0 at \\(\\theta = 0\\), providing zero gradient everywhere except at the optimum. The Wasserstein distance is \\(W_1 = |\\theta|\\), which is continuous, differentiable (except at 0), and has gradient \\(\\text{sign}(\\theta)\\) everywhere. This gradient always points toward the optimum.</p>
</div></div>

<h3>Enforcing the Lipschitz Constraint</h3>

<p>The key challenge in WGAN is enforcing the Lipschitz constraint on the critic. Three approaches have been proposed:</p>

<div class="env-block definition"><div class="env-title">Definition 17.4.4 &mdash; Weight Clipping</div><div class="env-body">
<p>The original WGAN enforces the Lipschitz constraint by <strong>weight clipping</strong>: after each gradient update, clamp all critic weights to \\([-c, c]\\) for some small constant \\(c\\) (e.g., 0.01). This is crude but simple. However, it biases the critic toward very simple functions and can cause vanishing or exploding gradients.</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 17.4.5 &mdash; Gradient Penalty (WGAN-GP)</div><div class="env-body">
<p>Gulrajani et al. (2017) proposed a <strong>gradient penalty</strong>: instead of clipping weights, add a penalty term that encourages the critic's gradient norm to be close to 1:</p>
\\[
\\mathcal{L}_{\\text{GP}} = \\lambda \\, \\mathbb{E}_{\\hat{\\mathbf{x}}}\\left[\\left(\\|\\nabla_{\\hat{\\mathbf{x}}} f_\\phi(\\hat{\\mathbf{x}})\\|_2 - 1\\right)^2\\right],
\\]
<p>where \\(\\hat{\\mathbf{x}} = \\epsilon \\mathbf{x} + (1 - \\epsilon) G(\\mathbf{z})\\) with \\(\\epsilon \\sim \\text{Uniform}(0,1)\\) is a random interpolation between real and fake samples. The coefficient \\(\\lambda\\) is typically set to 10.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 17.4.6 &mdash; Optimal Critic Has Unit Gradient</div><div class="env-body">
<p>Under mild assumptions, the optimal critic \\(f^*\\) for the Wasserstein objective has \\(\\|\\nabla f^*(\\mathbf{x})\\| = 1\\) almost everywhere along the straight lines connecting points from \\(\\mathbb{P}_r\\) and \\(\\mathbb{P}_g\\). This motivates penalizing deviations from unit gradient norm along these interpolated points.</p>
</div></div>

<h3>Advantages of WGAN</h3>

<ul>
<li><strong>Meaningful loss</strong>: The critic loss approximates \\(W_1\\) and correlates with sample quality, unlike the standard GAN discriminator loss.</li>
<li><strong>Stable gradients</strong>: No vanishing gradients when supports are disjoint.</li>
<li><strong>No mode collapse</strong>: The Wasserstein distance penalizes missing modes because transporting mass from the missed mode to the generated mass incurs a cost proportional to the distance.</li>
<li><strong>No careful balancing</strong>: The critic can be trained to optimality without harming the generator (unlike standard GAN where a too-strong discriminator kills gradients).</li>
</ul>

<div class="viz-placeholder" data-viz="viz-wgan-compare"></div>
`,
            visualizations: [
                {
                    id: 'viz-wgan-compare',
                    title: 'Standard GAN vs. WGAN Training Stability',
                    description: 'Left panel: Standard GAN loss and generated distribution. Right panel: WGAN with gradient penalty. Notice how the WGAN critic loss correlates with distribution quality, while the standard GAN discriminator loss oscillates unpredictably.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 450, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var halfW = Math.floor(W / 2) - 10;

                        var iter = 0;
                        var maxIter = 200;
                        var playing = false;
                        var animId = null;

                        // Real distribution
                        function gaussian(x, mu, sig) {
                            return Math.exp(-0.5 * Math.pow((x - mu) / sig, 2)) / (sig * Math.sqrt(2 * Math.PI));
                        }
                        function pReal(x) {
                            return 0.5 * gaussian(x, -1.5, 0.6) + 0.5 * gaussian(x, 2, 0.7);
                        }

                        // Standard GAN generator evolution (with oscillation and mode dropping)
                        var seed1 = 42;
                        function seeded1() { seed1 = (seed1 * 16807) % 2147483647; return (seed1 - 1) / 2147483646; }
                        var ganDLosses = [], ganGLosses = [];
                        var dL = 0.7, gL = 2.2;
                        for (var i = 0; i <= maxIter; i++) {
                            var t = i / maxIter;
                            dL += 0.02 * (0.5 - dL) + (seeded1() - 0.5) * 0.15;
                            dL = Math.max(0.05, Math.min(2.5, dL));
                            ganDLosses.push(dL);
                            gL += 0.015 * (1.5 - gL) + (seeded1() - 0.5) * 0.2 + 0.3 * Math.sin(t * 20);
                            gL = Math.max(0.2, Math.min(4.5, gL));
                            ganGLosses.push(gL);
                        }

                        // WGAN critic loss (smoother, monotonically improving)
                        var seed2 = 99;
                        function seeded2() { seed2 = (seed2 * 16807) % 2147483647; return (seed2 - 1) / 2147483646; }
                        var wganCLosses = [];
                        var cL = 3.0;
                        for (var i = 0; i <= maxIter; i++) {
                            var t = i / maxIter;
                            cL += 0.025 * (0.1 - cL) + (seeded2() - 0.5) * 0.04;
                            cL = Math.max(0.0, Math.min(4.0, cL));
                            wganCLosses.push(cL);
                        }

                        function pGanGen(x, t) {
                            // GAN: partial mode collapse at intermediate t
                            var e = t * t * (3 - 2 * t);
                            // oscillation causes mode dropping
                            var w1 = 0.5 * (1 - 0.7 * Math.sin(t * 8) * Math.max(0, 1 - t * 2));
                            w1 = Math.max(0.05, Math.min(0.95, 0.5 + (0.5 - w1) * e));
                            var mu1 = -1.5 * e + 0 * (1 - e);
                            var mu2 = 2 * e + 0 * (1 - e);
                            var sig = 2.0 * (1 - e) + 0.65 * e;
                            return w1 * gaussian(x, mu1, sig) + (1 - w1) * gaussian(x, mu2, sig);
                        }

                        function pWganGen(x, t) {
                            // WGAN: smooth convergence, no mode collapse
                            var e = t * t * (3 - 2 * t);
                            var mu1 = 0 * (1 - e) + (-1.5) * e;
                            var mu2 = 0 * (1 - e) + 2 * e;
                            var sig = 2.0 * (1 - e) + 0.65 * e;
                            return 0.5 * gaussian(x, mu1, sig) + 0.5 * gaussian(x, mu2, sig);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var t = iter / maxIter;
                            var distH = 180;
                            var lossH = 160;
                            var topY = 40;
                            var lossTopY = topY + distH + 30;

                            // Divider
                            ctx.strokeStyle = viz.colors.axis + '44';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath(); ctx.moveTo(W / 2, 10); ctx.lineTo(W / 2, H - 10); ctx.stroke();
                            ctx.setLineDash([]);

                            // Panel titles
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('Standard GAN', halfW / 2 + 5, 18);
                            ctx.fillStyle = viz.colors.teal;
                            ctx.fillText('WGAN-GP', W / 2 + halfW / 2 + 5, 18);

                            // ---- Draw distributions ----
                            function drawDist(offsetX, pGenFunc) {
                                var xToS = function(x) { return offsetX + 20 + (x + 5) / 10 * (halfW - 40); };
                                var yToS = function(y) { return topY + distH - y * distH / 0.7; };

                                // Axis
                                ctx.strokeStyle = viz.colors.axis;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(offsetX + 20, topY + distH); ctx.lineTo(offsetX + halfW - 20, topY + distH); ctx.stroke();

                                // Real
                                ctx.strokeStyle = viz.colors.green;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                for (var x = -5; x <= 5; x += 0.1) {
                                    var sx = xToS(x), sy = yToS(pReal(x));
                                    if (x === -5) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();

                                // Gen
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                for (var x = -5; x <= 5; x += 0.1) {
                                    var pg = pGenFunc(x, t);
                                    var sx = xToS(x), sy = yToS(pg);
                                    if (x === -5) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();
                            }

                            drawDist(5, pGanGen);
                            drawDist(W / 2 + 5, pWganGen);

                            // ---- Draw loss curves ----
                            function drawLoss(offsetX, losses1, losses2, label1, label2, color1, color2) {
                                var eToX = function(e) { return offsetX + 20 + (e / maxIter) * (halfW - 40); };
                                var lToY = function(l) { return lossTopY + lossH - (l / 5) * lossH; };

                                // Axis
                                ctx.strokeStyle = viz.colors.axis;
                                ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(offsetX + 20, lossTopY + lossH); ctx.lineTo(offsetX + halfW - 20, lossTopY + lossH); ctx.stroke();
                                ctx.beginPath(); ctx.moveTo(offsetX + 20, lossTopY); ctx.lineTo(offsetX + 20, lossTopY + lossH); ctx.stroke();

                                // Loss 1
                                ctx.strokeStyle = color1;
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                for (var i = 0; i <= iter; i++) {
                                    var x = eToX(i), y = lToY(losses1[i]);
                                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                                }
                                ctx.stroke();

                                // Loss 2
                                if (losses2) {
                                    ctx.strokeStyle = color2;
                                    ctx.lineWidth = 1.5;
                                    ctx.beginPath();
                                    for (var i = 0; i <= iter; i++) {
                                        var x = eToX(i), y = lToY(losses2[i]);
                                        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                                    }
                                    ctx.stroke();
                                }

                                // Legend
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillStyle = color1;
                                ctx.fillText(label1, offsetX + 30, lossTopY + 12);
                                if (label2) {
                                    ctx.fillStyle = color2;
                                    ctx.fillText(label2, offsetX + 30, lossTopY + 24);
                                }
                            }

                            drawLoss(5, ganDLosses, ganGLosses, 'D loss', 'G loss', viz.colors.orange, viz.colors.blue);
                            drawLoss(W / 2 + 5, wganCLosses, null, 'Critic loss (W\u2081)', null, viz.colors.teal, null);

                            // Dist legend
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('p_data', 25, topY + 8);
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText('p_g', 80, topY + 8);

                            // Epoch
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'center';
                            ctx.fillText('Iteration: ' + iter + ' / ' + maxIter, W / 2, H - 8);
                        }

                        draw();

                        var playBtn = VizEngine.createButton(controls, 'Play', function() {
                            if (playing) { playing = false; playBtn.textContent = 'Play'; return; }
                            playing = true;
                            playBtn.textContent = 'Pause';
                            function tick() {
                                if (!playing) return;
                                if (iter < maxIter) { iter++; draw(); animId = setTimeout(tick, 40); }
                                else { playing = false; playBtn.textContent = 'Play'; }
                            }
                            tick();
                        });

                        VizEngine.createButton(controls, 'Reset', function() {
                            playing = false; playBtn.textContent = 'Play'; iter = 0; draw();
                        });

                        VizEngine.createSlider(controls, 'Iteration', 0, maxIter, 0, 1, function(v) {
                            iter = Math.round(v); draw();
                        });

                        return { stopAnimation: function() { playing = false; if (animId) clearTimeout(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the Jensen-Shannon divergence between \\(\\mathbb{P}_r = \\delta_0\\) and \\(\\mathbb{P}_g = \\delta_\\theta\\) is \\(\\log 2\\) for all \\(\\theta \\neq 0\\), while \\(W_1(\\mathbb{P}_r, \\mathbb{P}_g) = |\\theta|\\). Explain why this makes WGAN more suitable for training.',
                    hint: 'For disjoint point masses, compute the JSD directly. For \\(W_1\\), the only coupling is the product measure.',
                    solution: 'When \\(\\theta \\neq 0\\), the supports of \\(\\delta_0\\) and \\(\\delta_\\theta\\) are disjoint. The mixture \\(m = (\\delta_0 + \\delta_\\theta)/2\\) has mass 1/2 at each point. \\(D_{\\text{KL}}(\\delta_0 \\| m) = \\log(1 / (1/2)) = \\log 2\\), and similarly for \\(D_{\\text{KL}}(\\delta_\\theta \\| m)\\). So \\(D_{\\text{JS}} = \\frac{1}{2}(\\log 2 + \\log 2) = \\log 2\\), a constant with zero gradient. For \\(W_1\\): the only coupling between \\(\\delta_0\\) and \\(\\delta_\\theta\\) transports all mass from 0 to \\(\\theta\\), costing \\(|\\theta|\\). This has gradient \\(\\text{sign}(\\theta)\\), always pointing toward the optimum. WGAN is better because it provides informative gradients even when supports do not overlap.'
                },
                {
                    question: 'In WGAN-GP, the gradient penalty samples interpolated points \\(\\hat{\\mathbf{x}} = \\epsilon \\mathbf{x} + (1-\\epsilon) G(\\mathbf{z})\\). Why is this preferable to penalizing the gradient at arbitrary points in the data space?',
                    hint: 'Think about where the optimal critic needs to have unit gradient (Theorem 17.4.6) and where computational effort should be concentrated.',
                    solution: 'By Theorem 17.4.6, the optimal critic has unit gradient norm along the straight lines connecting points in the supports of \\(\\mathbb{P}_r\\) and \\(\\mathbb{P}_g\\). Penalizing the gradient at random points in \\(\\mathbb{R}^D\\) would waste computation on regions far from both distributions where the gradient constraint does not matter much. The interpolated points \\(\\hat{\\mathbf{x}}\\) lie on the "transport rays" between real and fake samples, exactly where the Lipschitz constraint is most important. This targeted enforcement is both computationally efficient and theoretically motivated.'
                },
                {
                    question: 'The original WGAN used weight clipping \\(w \\leftarrow \\text{clip}(w, -c, c)\\) to enforce the Lipschitz constraint. Explain two problems with this approach that WGAN-GP resolves.',
                    hint: 'Think about what happens to the critic capacity and gradient flow as the clipping constant \\(c\\) varies.',
                    solution: 'Problem 1: <strong>Capacity underuse.</strong> Weight clipping restricts the critic to the set of functions expressible with weights in \\([-c, c]\\). If \\(c\\) is too small, the critic can only represent very simple functions (nearly linear), failing to provide useful gradients. If \\(c\\) is too large, the clipping is ineffective and the Lipschitz constraint is violated. Problem 2: <strong>Pathological gradients.</strong> Weight clipping biases the critic toward functions that saturate the clipping bounds (weights at \\(\\pm c\\)), producing either vanishing gradients (weights near 0) or exploding gradients (all weights at \\(\\pm c\\)). WGAN-GP resolves both by directly penalizing the gradient norm, allowing the weights to take any value while softly enforcing the Lipschitz constraint where it matters most.'
                }
            ]
        },

        // ========== SECTION 5: DCGAN & StyleGAN ==========
        {
            id: 'ch17-sec05',
            title: 'DCGAN & StyleGAN',
            content: `
<h2>17.5 DCGAN &amp; StyleGAN</h2>

<div class="env-block intuition"><div class="env-title">Intuition: From Pixels to Architecture</div><div class="env-body">
<p>The theoretical framework of Sections 17.1-17.4 says nothing about network architecture; G and D can be any differentiable functions. In practice, architecture matters enormously. The leap from fully-connected GANs (which produce blurry 32x32 images) to photorealistic 1024x1024 face synthesis was driven primarily by architectural innovations: convolutional generators (DCGAN), progressive growing (ProGAN), and style-based generation (StyleGAN).</p>
</div></div>

<h3>DCGAN: Deep Convolutional GAN</h3>

<div class="env-block definition"><div class="env-title">Definition 17.5.1 &mdash; DCGAN Architecture</div><div class="env-body">
<p>Radford et al. (2016) introduced <strong>DCGAN</strong>, a set of architectural guidelines for stable convolutional GANs:</p>
<ol>
<li><strong>Replace pooling with strided convolutions</strong>: Use fractionally-strided (transposed) convolutions in the generator and strided convolutions in the discriminator.</li>
<li><strong>Use batch normalization</strong> in both G and D (except the output layer of G and the input layer of D).</li>
<li><strong>Remove fully connected layers</strong>: The generator maps the latent vector directly to a 3D tensor via reshape, then upsamples through transposed convolutions.</li>
<li><strong>Use ReLU in the generator</strong> (except the output, which uses Tanh) and <strong>LeakyReLU in the discriminator</strong>.</li>
</ol>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; Transposed Convolutions</div><div class="env-body">
<p>A standard convolution with stride 2 halves the spatial resolution. A <strong>transposed convolution</strong> (sometimes misleadingly called "deconvolution") with stride 2 doubles it. In the generator, each transposed convolution layer upsamples the feature maps by a factor of 2 while reducing the number of channels, progressively constructing a higher-resolution image from a low-dimensional representation. The generator thus looks like a reversed CNN: \\(\\mathbf{z} \\in \\mathbb{R}^{100} \\to 4 \\times 4 \\times 512 \\to 8 \\times 8 \\times 256 \\to 16 \\times 16 \\times 128 \\to 32 \\times 32 \\times 64 \\to 64 \\times 64 \\times 3\\).</p>
</div></div>

<h3>Progressive Growing (ProGAN)</h3>

<p>Karras et al. (2018) proposed <strong>progressive growing</strong>: start training G and D at 4x4 resolution, then gradually add higher-resolution layers during training (8x8, 16x16, ..., up to 1024x1024). Each new layer is "faded in" using a learnable blending weight \\(\\alpha\\). This curriculum approach stabilizes training at high resolutions by first learning coarse structure and then refining details.</p>

<h3>StyleGAN</h3>

<p>Karras et al. (2019) redesigned the generator architecture with a <strong>style-based</strong> approach inspired by neural style transfer. The key insight is to separate the control of high-level attributes (pose, identity) from stochastic variation (freckles, hair strands).</p>

<div class="env-block definition"><div class="env-title">Definition 17.5.2 &mdash; StyleGAN Generator Architecture</div><div class="env-body">
<p>The StyleGAN generator consists of three components:</p>
<ol>
<li><strong>Mapping network</strong> \\(f: \\mathbb{R}^{512} \\to \\mathcal{W}\\): An 8-layer MLP that maps the latent code \\(\\mathbf{z}\\) to an intermediate latent space \\(\\mathbf{w} = f(\\mathbf{z})\\). The space \\(\\mathcal{W}\\) is less entangled than the input space \\(\\mathcal{Z}\\).</li>
<li><strong>Synthesis network</strong>: A sequence of blocks at increasing resolution (4x4, 8x8, ..., 1024x1024). Each block applies two convolution layers, each preceded by an <strong>Adaptive Instance Normalization</strong> (AdaIN) layer that injects style information.</li>
<li><strong>Noise injection</strong>: Per-pixel Gaussian noise is added after each convolution, controlling stochastic details (hair placement, skin texture) independently of the global style.</li>
</ol>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 17.5.3 &mdash; Adaptive Instance Normalization (AdaIN)</div><div class="env-body">
<p>Given a feature map \\(\\mathbf{x}_i\\) (the \\(i\\)-th channel) and style parameters \\((y_{s,i}, y_{b,i})\\) derived from \\(\\mathbf{w}\\) via learned affine transforms:</p>
\\[
\\text{AdaIN}(\\mathbf{x}_i, \\mathbf{w}) = y_{s,i}(\\mathbf{w}) \\cdot \\frac{\\mathbf{x}_i - \\mu(\\mathbf{x}_i)}{\\sigma(\\mathbf{x}_i)} + y_{b,i}(\\mathbf{w}),
\\]
<p>where \\(\\mu(\\mathbf{x}_i)\\) and \\(\\sigma(\\mathbf{x}_i)\\) are the spatial mean and standard deviation of channel \\(i\\). The style \\(\\mathbf{w}\\) controls <em>what</em> content is generated (via the channel-wise scales and biases), while the convolution weights control <em>how</em> it is spatially arranged.</p>
</div></div>

<h3>Style Mixing and Hierarchical Control</h3>

<p>A key property of StyleGAN is that different layers control different levels of detail:</p>
<ul>
<li><strong>Coarse layers</strong> (4x4 to 8x8): Control high-level attributes like pose, face shape, and eyeglasses.</li>
<li><strong>Middle layers</strong> (16x16 to 32x32): Control facial features, hairstyle, and eyes open/closed.</li>
<li><strong>Fine layers</strong> (64x64 to 1024x1024): Control color scheme, microstructure, and background details.</li>
</ul>

<div class="env-block definition"><div class="env-title">Definition 17.5.4 &mdash; Style Mixing Regularization</div><div class="env-body">
<p>During training, <strong>style mixing</strong> (or <em>mixing regularization</em>) uses two different latent codes \\(\\mathbf{z}_1\\) and \\(\\mathbf{z}_2\\), producing \\(\\mathbf{w}_1 = f(\\mathbf{z}_1)\\) and \\(\\mathbf{w}_2 = f(\\mathbf{z}_2)\\). A random crossover point is chosen in the synthesis network: layers before the crossover use \\(\\mathbf{w}_1\\), and layers after use \\(\\mathbf{w}_2\\). This encourages each layer to learn independent style control, improving disentanglement.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark &mdash; The \\(\\mathcal{W}\\) Space</div><div class="env-body">
<p>The intermediate latent space \\(\\mathcal{W}\\) is crucial. Direct sampling \\(\\mathbf{z} \\sim \\mathcal{N}(0, I)\\) forces the latent space to follow a fixed (Gaussian) distribution, which may not align well with the manifold of meaningful face attributes. The mapping network \\(f\\) can warp this distribution into a space where linear interpolation corresponds to semantically meaningful changes (e.g., linearly varying the age or adding a smile). Perceptual path length (PPL) metrics show that \\(\\mathcal{W}\\) is significantly less curved (more disentangled) than \\(\\mathcal{Z}\\).</p>
</div></div>

<h3>StyleGAN2 and Beyond</h3>

<p>StyleGAN2 (Karras et al., 2020) addressed several artifacts in StyleGAN:</p>
<ul>
<li><strong>Weight demodulation</strong>: Replaces AdaIN to eliminate "blob" artifacts caused by instance normalization destroying information about feature magnitudes.</li>
<li><strong>Path length regularization</strong>: Encourages the mapping from \\(\\mathcal{W}\\) to images to have a constant Jacobian norm, improving smoothness of the latent space.</li>
<li><strong>No progressive growing</strong>: Achieves high-resolution synthesis with a single fixed architecture using residual connections and skip connections.</li>
</ul>

<p>StyleGAN3 (Karras et al., 2021) further addressed aliasing artifacts by redesigning the generator with continuous signal processing principles, ensuring translation and rotation equivariance.</p>

<div class="viz-placeholder" data-viz="viz-stylegan-arch"></div>
`,
            visualizations: [
                {
                    id: 'viz-stylegan-arch',
                    title: 'StyleGAN Architecture: Style Injection at Multiple Scales',
                    description: 'This diagram shows the StyleGAN generator architecture. The mapping network transforms z into w, which is then injected at each resolution level via AdaIN. Coarse layers control pose and shape; middle layers control features; fine layers control texture and color. Click the style level buttons to highlight different layers.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 480, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var highlight = 'all'; // 'coarse', 'medium', 'fine', 'all'

                        // Architecture layout
                        var layers = [
                            { res: '4x4', level: 'coarse', y: 400, blockW: 28, blockH: 28 },
                            { res: '8x8', level: 'coarse', y: 350, blockW: 36, blockH: 36 },
                            { res: '16x16', level: 'medium', y: 295, blockW: 44, blockH: 44 },
                            { res: '32x32', level: 'medium', y: 232, blockW: 52, blockH: 52 },
                            { res: '64x64', level: 'fine', y: 165, blockW: 60, blockH: 60 },
                            { res: '128x128', level: 'fine', y: 92, blockW: 68, blockH: 68 },
                            { res: '1024x1024', level: 'fine', y: 20, blockW: 78, blockH: 72 }
                        ];

                        var synthX = 380;
                        var mappingX = 100;

                        function getLevelColor(level) {
                            if (level === 'coarse') return viz.colors.orange;
                            if (level === 'medium') return viz.colors.blue;
                            return viz.colors.teal;
                        }

                        function isHighlighted(level) {
                            return highlight === 'all' || highlight === level;
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // ---- Mapping Network ----
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillStyle = viz.colors.purple;
                            ctx.fillText('Mapping Network f', mappingX, 30);

                            // z input
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('z ~ N(0,I)', mappingX, 55);

                            // MLP layers for mapping network
                            var mlpY = 70;
                            for (var m = 0; m < 8; m++) {
                                var my = mlpY + m * 28;
                                var alpha = 0.5 + 0.5 * (m / 7);
                                ctx.fillStyle = viz.colors.purple;
                                ctx.globalAlpha = alpha;
                                ctx.fillRect(mappingX - 30, my, 60, 18);
                                ctx.globalAlpha = 1;
                                ctx.strokeStyle = viz.colors.purple;
                                ctx.lineWidth = 1;
                                ctx.strokeRect(mappingX - 30, my, 60, 18);
                                if (m < 7) {
                                    ctx.strokeStyle = viz.colors.purple + '66';
                                    ctx.beginPath();
                                    ctx.moveTo(mappingX, my + 18);
                                    ctx.lineTo(mappingX, my + 28);
                                    ctx.stroke();
                                }
                            }
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText('FC + ReLU', mappingX, mlpY + 3.5 * 28 + 9);

                            // w output
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('w \u2208 W', mappingX, mlpY + 8 * 28 + 10);

                            // ---- Synthesis Network ----
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'center';
                            ctx.fillText('Synthesis Network', synthX, 15);

                            // Draw each layer block
                            for (var i = 0; i < layers.length; i++) {
                                var l = layers[i];
                                var color = getLevelColor(l.level);
                                var highlighted = isHighlighted(l.level);
                                var alpha = highlighted ? 1.0 : 0.2;

                                var bx = synthX - l.blockW / 2;
                                var by = l.y;

                                // Block
                                ctx.globalAlpha = alpha;
                                ctx.fillStyle = color + '44';
                                ctx.fillRect(bx, by, l.blockW, l.blockH);
                                ctx.strokeStyle = color;
                                ctx.lineWidth = highlighted ? 2 : 1;
                                ctx.strokeRect(bx, by, l.blockW, l.blockH);

                                // Resolution label
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.fillStyle = viz.colors.white;
                                ctx.textAlign = 'center';
                                ctx.fillText(l.res, synthX, by + l.blockH / 2 + 3);

                                // Connection to next layer
                                if (i < layers.length - 1) {
                                    var nl = layers[i + 1];
                                    ctx.strokeStyle = viz.colors.axis + '66';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.moveTo(synthX, by);
                                    ctx.lineTo(synthX, nl.y + nl.blockH);
                                    ctx.stroke();
                                }

                                // Style injection arrow from mapping network
                                var arrowStartX = mappingX + 40;
                                var arrowEndX = bx - 5;
                                var arrowY = by + l.blockH / 2;

                                // w to A (affine) to AdaIN
                                ctx.strokeStyle = color;
                                ctx.lineWidth = highlighted ? 1.5 : 0.5;
                                ctx.setLineDash([3, 3]);
                                ctx.beginPath();
                                ctx.moveTo(arrowStartX, arrowY);
                                ctx.lineTo(arrowEndX, arrowY);
                                ctx.stroke();
                                ctx.setLineDash([]);

                                // A box (affine transform)
                                var aX = (arrowStartX + arrowEndX) / 2;
                                ctx.fillStyle = color;
                                ctx.fillRect(aX - 8, arrowY - 8, 16, 16);
                                ctx.fillStyle = viz.colors.bg;
                                ctx.font = 'bold 9px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('A', aX, arrowY);

                                // AdaIN label
                                ctx.fillStyle = color;
                                ctx.font = '8px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('AdaIN', arrowEndX - 2, arrowY - 10);

                                ctx.globalAlpha = 1;
                                ctx.textBaseline = 'alphabetic';
                            }

                            // Noise injection arrows on right side
                            ctx.globalAlpha = 0.6;
                            for (var i = 0; i < layers.length; i++) {
                                var l = layers[i];
                                var nx = synthX + l.blockW / 2 + 5;
                                var ny = l.y + l.blockH / 2;

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '9px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('noise', nx + 18, ny);

                                // Arrow
                                ctx.strokeStyle = viz.colors.text;
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(nx + 15, ny);
                                ctx.lineTo(nx + 2, ny);
                                ctx.stroke();
                                // Arrowhead
                                ctx.beginPath();
                                ctx.moveTo(nx + 2, ny);
                                ctx.lineTo(nx + 6, ny - 3);
                                ctx.lineTo(nx + 6, ny + 3);
                                ctx.closePath();
                                ctx.fill();
                            }
                            ctx.globalAlpha = 1;
                            ctx.textBaseline = 'alphabetic';

                            // Level legend on right
                            var legendX = W - 140;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';

                            var legendItems = [
                                { label: 'Coarse (pose, shape)', color: viz.colors.orange, level: 'coarse', y: 30 },
                                { label: 'Medium (features)', color: viz.colors.blue, level: 'medium', y: 50 },
                                { label: 'Fine (texture, color)', color: viz.colors.teal, level: 'fine', y: 70 }
                            ];

                            for (var li = 0; li < legendItems.length; li++) {
                                var item = legendItems[li];
                                var ihl = isHighlighted(item.level);
                                ctx.globalAlpha = ihl ? 1 : 0.3;
                                ctx.fillStyle = item.color;
                                ctx.fillRect(legendX, item.y - 5, 12, 12);
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(item.label, legendX + 18, item.y + 5);
                            }
                            ctx.globalAlpha = 1;

                            // Output arrow at top
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            var topL = layers[layers.length - 1];
                            ctx.fillText('Output Image', synthX, topL.y - 8);
                        }

                        draw();

                        VizEngine.createButton(controls, 'All Layers', function() { highlight = 'all'; draw(); });
                        VizEngine.createButton(controls, 'Coarse', function() { highlight = 'coarse'; draw(); });
                        VizEngine.createButton(controls, 'Medium', function() { highlight = 'medium'; draw(); });
                        VizEngine.createButton(controls, 'Fine', function() { highlight = 'fine'; draw(); });

                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Explain why DCGAN uses batch normalization in both the generator and discriminator, but omits it from the generator output layer and discriminator input layer.',
                    hint: 'Think about what batch normalization does to the statistics of its input and why this might be harmful at the boundaries of the network.',
                    solution: 'Batch normalization standardizes activations to zero mean and unit variance, which stabilizes training by preventing internal covariate shift. However, in the <em>generator output layer</em>, we want the network to produce pixel values in a specific range (e.g., [-1, 1] with Tanh); normalizing would destroy this carefully calibrated output range. In the <em>discriminator input layer</em>, the raw pixel values carry meaningful information about the data distribution (e.g., overall brightness, color statistics); normalizing them would discard this signal. In intermediate layers, normalization is beneficial because it prevents gradient issues and mode collapse.'
                },
                {
                    question: 'StyleGAN\'s mapping network transforms \\(\\mathbf{z} \\in \\mathcal{Z}\\) to \\(\\mathbf{w} \\in \\mathcal{W}\\) via an 8-layer MLP. Why does this improve the quality and disentanglement of generated images compared to directly injecting \\(\\mathbf{z}\\)?',
                    hint: 'Consider the constraint that \\(\\mathbf{z} \\sim \\mathcal{N}(0, I)\\) imposes on the geometry of \\(\\mathcal{Z}\\). Does the distribution of face attributes necessarily follow a Gaussian?',
                    solution: 'The latent space \\(\\mathcal{Z}\\) is forced to follow a fixed prior \\(\\mathcal{N}(0, I)\\), which imposes a spherical geometry. If the true factors of variation (age, gender, pose, expression) have a different joint distribution (e.g., some combinations like "baby with wrinkles" are impossible), the generator must warp the Gaussian into this non-Gaussian manifold, creating curved paths in \\(\\mathcal{Z}\\) where small perturbations change multiple attributes simultaneously (entanglement). The mapping network \\(f\\) learns this warping explicitly, so \\(\\mathcal{W}\\) can have a distribution that naturally matches the factor space. In \\(\\mathcal{W}\\), linear interpolation corresponds to semantically meaningful changes, and the Perceptual Path Length (PPL) metric confirms that paths in \\(\\mathcal{W}\\) are straighter (more disentangled) than in \\(\\mathcal{Z}\\).'
                },
                {
                    question: 'In StyleGAN, noise is injected after each convolution layer independently. How does this differ from the latent code injection, and what would happen if noise were removed entirely?',
                    hint: 'Think about what aspects of a face image are deterministic given high-level attributes vs. what aspects are genuinely random.',
                    solution: 'The style code \\(\\mathbf{w}\\) controls <em>deterministic</em> attributes (identity, pose, expression, gender): given a style, these should be fixed. Noise controls <em>stochastic</em> details (exact hair strand placement, skin pore locations, background texture): these vary even among images of the same person with the same pose. If noise were removed, the generator would have to encode stochastic variation in the style code or in the learned constant input, leading to deterministic outputs for each \\(\\mathbf{w}\\). The resulting images would either look unnaturally smooth (lacking high-frequency detail) or the style space would become entangled with stochastic variation. By providing an independent noise source, the network can allocate its style capacity entirely to meaningful attributes while noise handles irrelevant variation, improving overall image quality and disentanglement.'
                }
            ]
        }
    ]
});
