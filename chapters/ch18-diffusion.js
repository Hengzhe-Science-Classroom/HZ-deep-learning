// === Chapter 18: Diffusion Models ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch18',
    number: 18,
    title: 'Diffusion Models',
    subtitle: 'Generating data by learning to reverse the gradual destruction of structure',
    sections: [
        // ========== SECTION 1: Forward Diffusion Process ==========
        {
            id: 'sec18-1-forward-diffusion',
            title: 'Forward Diffusion Process',
            content: `
<h2>18.1 Forward Diffusion Process</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Destroying Information, One Step at a Time</div><div class="env-body">
Imagine dropping ink into still water. The ink begins as a concentrated blob with rich structure, but gradually diffuses until the water is uniformly tinted. This physical process of diffusion progressively destroys information about where the ink started. Diffusion models in deep learning exploit exactly this idea: if we can learn to <em>reverse</em> the diffusion (recovering the ink blob from the uniform solution), then we have a generative model that can conjure structured data out of pure noise.
</div></div>

<p>The forward diffusion process defines a Markov chain that gradually adds Gaussian noise to data over \\(T\\) time steps. Starting from a data sample \\(\\mathbf{x}_0 \\sim q(\\mathbf{x}_0)\\), each step injects a small amount of noise controlled by a <em>variance schedule</em> \\(\\beta_1, \\beta_2, \\ldots, \\beta_T\\).</p>

<div class="env-block definition"><div class="env-title">Definition 18.1.1 — Forward Diffusion Kernel</div><div class="env-body">
The forward process is defined by the transition kernel
\\[
q(\\mathbf{x}_t \\mid \\mathbf{x}_{t-1}) = \\mathcal{N}\\bigl(\\mathbf{x}_t;\\, \\sqrt{1 - \\beta_t}\\,\\mathbf{x}_{t-1},\\; \\beta_t \\mathbf{I}\\bigr),
\\]
where \\(\\beta_t \\in (0, 1)\\) is the noise variance at step \\(t\\). Equivalently,
\\[
\\mathbf{x}_t = \\sqrt{1 - \\beta_t}\\,\\mathbf{x}_{t-1} + \\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}_t, \\quad \\boldsymbol{\\epsilon}_t \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}).
\\]
</div></div>

<p>The coefficient \\(\\sqrt{1 - \\beta_t}\\) on \\(\\mathbf{x}_{t-1}\\) ensures that the variance does not blow up. If \\(\\mathbf{x}_{t-1}\\) had unit variance, then \\(\\mathbf{x}_t\\) also has unit variance: \\((1 - \\beta_t) \\cdot 1 + \\beta_t = 1\\). This <em>variance-preserving</em> property is critical for numerical stability.</p>

<h3>The Noise Schedule</h3>

<p>The sequence \\(\\{\\beta_t\\}_{t=1}^T\\) is called the <em>noise schedule</em>. Common choices include:</p>
<ul>
<li><strong>Linear schedule</strong> (Ho et al., 2020): \\(\\beta_t\\) increases linearly from \\(\\beta_1 = 10^{-4}\\) to \\(\\beta_T = 0.02\\).</li>
<li><strong>Cosine schedule</strong> (Nichol &amp; Dhariwal, 2021): designed so that \\(\\bar{\\alpha}_t\\) follows a cosine curve, preventing the signal from being destroyed too quickly at the start.</li>
</ul>

<div class="env-block remark"><div class="env-title">Remark — Why Not Destroy Everything in One Step?</div><div class="env-body">
We <em>could</em> define a single-step corruption \\(q(\\mathbf{x}_T \\mid \\mathbf{x}_0) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\), but the key insight is that the <em>reverse</em> of a single giant step is intractably complex, whereas each small reverse step is approximately Gaussian and therefore easy to parameterize with a neural network.
</div></div>

<h3>Closed-Form Marginals</h3>

<p>A remarkable property of the forward process is that we can sample \\(\\mathbf{x}_t\\) at any arbitrary time step <em>directly</em> from \\(\\mathbf{x}_0\\), without iterating through intermediate steps. Define</p>
\\[
\\alpha_t = 1 - \\beta_t, \\qquad \\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s.
\\]

<div class="env-block theorem"><div class="env-title">Theorem 18.1.2 — Direct Sampling at Time \\(t\\)</div><div class="env-body">
The marginal distribution of \\(\\mathbf{x}_t\\) given \\(\\mathbf{x}_0\\) is
\\[
q(\\mathbf{x}_t \\mid \\mathbf{x}_0) = \\mathcal{N}\\bigl(\\mathbf{x}_t;\\, \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0,\\; (1 - \\bar{\\alpha}_t)\\mathbf{I}\\bigr).
\\]
Equivalently,
\\[
\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}).
\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof (by induction)</div><div class="env-body">
<p><strong>Base case</strong> (\\(t=1\\)): \\(\\mathbf{x}_1 = \\sqrt{\\alpha_1}\\,\\mathbf{x}_0 + \\sqrt{1 - \\alpha_1}\\,\\boldsymbol{\\epsilon}_1\\), so \\(q(\\mathbf{x}_1 \\mid \\mathbf{x}_0) = \\mathcal{N}(\\sqrt{\\alpha_1}\\,\\mathbf{x}_0, (1 - \\alpha_1)\\mathbf{I})\\). Since \\(\\bar{\\alpha}_1 = \\alpha_1\\), this matches.</p>

<p><strong>Inductive step</strong>: Assume \\(\\mathbf{x}_{t-1} = \\sqrt{\\bar{\\alpha}_{t-1}}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_{t-1}}\\,\\boldsymbol{\\epsilon}'\\) where \\(\\boldsymbol{\\epsilon}' \\sim \\mathcal{N}(\\mathbf{0},\\mathbf{I})\\). Then</p>
\\[
\\mathbf{x}_t = \\sqrt{\\alpha_t}\\,\\mathbf{x}_{t-1} + \\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}_t = \\sqrt{\\alpha_t \\bar{\\alpha}_{t-1}}\\,\\mathbf{x}_0 + \\sqrt{\\alpha_t(1 - \\bar{\\alpha}_{t-1})}\\,\\boldsymbol{\\epsilon}' + \\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}_t.
\\]
<p>The sum of two independent Gaussians has variance \\(\\alpha_t(1 - \\bar{\\alpha}_{t-1}) + \\beta_t = 1 - \\alpha_t\\bar{\\alpha}_{t-1} = 1 - \\bar{\\alpha}_t\\), completing the induction.</p>
<div class="qed">&#8718;</div>
</div></div>

<p>This closed-form result is computationally essential: during training, we can jump directly to any noise level \\(t\\) without simulating the chain, reducing the cost from \\(O(T)\\) to \\(O(1)\\) per training sample.</p>

<h3>Asymptotic Behavior</h3>

<p>As \\(T \\to \\infty\\) (or equivalently \\(\\bar{\\alpha}_T \\to 0\\)), the signal term \\(\\sqrt{\\bar{\\alpha}_T}\\,\\mathbf{x}_0 \\to \\mathbf{0}\\) vanishes and the noise term dominates. Thus \\(q(\\mathbf{x}_T) \\approx \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\), regardless of the original data distribution. All information about \\(\\mathbf{x}_0\\) has been erased. The forward process has mapped every data distribution to the <em>same</em> simple distribution.</p>

<div class="viz-placeholder" data-viz="viz-forward-diffusion"></div>

<div class="env-block example"><div class="env-title">Example 18.1.3 — Signal-to-Noise Ratio</div><div class="env-body">
We can define the signal-to-noise ratio (SNR) at time \\(t\\) as
\\[
\\mathrm{SNR}(t) = \\frac{\\bar{\\alpha}_t}{1 - \\bar{\\alpha}_t}.
\\]
At \\(t = 0\\), \\(\\bar{\\alpha}_0 = 1\\) so \\(\\mathrm{SNR} = \\infty\\) (pure signal). At \\(t = T\\), \\(\\bar{\\alpha}_T \\approx 0\\) so \\(\\mathrm{SNR} \\approx 0\\) (pure noise). The noise schedule determines how quickly the SNR decays, which profoundly affects sample quality.
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-forward-diffusion',
                    title: 'Forward Diffusion: From Data to Noise',
                    description: 'Watch how a structured 2D distribution (three clusters) is progressively corrupted into isotropic Gaussian noise. Use the slider to control the diffusion timestep \\(t\\). The signal (colored points near cluster centers) fades as noise takes over.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50 });
                        const ctx = viz.ctx;

                        // Generate initial data: 3 clusters
                        const N = 300;
                        const clusters = [
                            { cx: -2, cy: 1.5, color: viz.colors.blue },
                            { cx: 1.5, cy: 1.8, color: viz.colors.teal },
                            { cx: 0.5, cy: -1.5, color: viz.colors.orange }
                        ];
                        const points = [];
                        const rng = () => {
                            let u = 0, v = 0;
                            while (u === 0) u = Math.random();
                            while (v === 0) v = Math.random();
                            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
                        };
                        for (let i = 0; i < N; i++) {
                            const cl = clusters[i % 3];
                            points.push({
                                x0: cl.cx + rng() * 0.3,
                                y0: cl.cy + rng() * 0.3,
                                ex: rng(),
                                ey: rng(),
                                color: cl.color
                            });
                        }

                        let tVal = 0;

                        // Noise schedule: linear beta from 0.0001 to 0.02
                        const T = 1000;
                        function getAlphaBar(t) {
                            // Continuous approximation
                            const frac = t / T;
                            const betaStart = 0.0001, betaEnd = 0.02;
                            // Integral of linear beta schedule
                            const integral = frac * betaStart + 0.5 * frac * frac * (betaEnd - betaStart);
                            return Math.exp(-integral * T);
                        }

                        const slider = VizEngine.createSlider(controls, 't/T', 0, 1, 0, 0.01, v => { tVal = v; draw(); });

                        function draw() {
                            viz.clear();

                            // Draw subtle grid
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.3;
                            for (let gx = -6; gx <= 6; gx++) {
                                const [sx] = viz.toScreen(gx, 0);
                                ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, viz.height); ctx.stroke();
                            }
                            for (let gy = -4; gy <= 4; gy++) {
                                const [, sy] = viz.toScreen(0, gy);
                                ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(viz.width, sy); ctx.stroke();
                            }

                            const t = tVal * T;
                            const abar = getAlphaBar(t);
                            const sqrtAbar = Math.sqrt(abar);
                            const sqrtOneMinusAbar = Math.sqrt(1 - abar);

                            // Draw each point at x_t = sqrt(abar)*x_0 + sqrt(1-abar)*eps
                            for (const p of points) {
                                const xt = sqrtAbar * p.x0 + sqrtOneMinusAbar * p.ex;
                                const yt = sqrtAbar * p.y0 + sqrtOneMinusAbar * p.ey;
                                const [sx, sy] = viz.toScreen(xt, yt);
                                // Fade color as noise increases
                                const alpha = 0.3 + 0.5 * abar;
                                ctx.globalAlpha = alpha;
                                ctx.fillStyle = p.color;
                                ctx.beginPath();
                                ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
                                ctx.fill();
                            }
                            ctx.globalAlpha = 1;

                            // Info text
                            const snr = abar / (1 - abar + 1e-10);
                            viz.screenText('t = ' + Math.round(t) + ' / ' + T, 14, 20, viz.colors.white, 13, 'left');
                            viz.screenText('\u03B1\u0305_t = ' + abar.toFixed(4), 14, 38, viz.colors.teal, 12, 'left');
                            viz.screenText('SNR = ' + (snr > 100 ? '\u221E' : snr.toFixed(2)), 14, 54, viz.colors.orange, 12, 'left');

                            // Phase label
                            let phase = 'Pure Data';
                            if (tVal > 0.02 && tVal <= 0.3) phase = 'Slight Noise';
                            else if (tVal > 0.3 && tVal <= 0.7) phase = 'Mixed Signal + Noise';
                            else if (tVal > 0.7 && tVal < 0.95) phase = 'Mostly Noise';
                            else if (tVal >= 0.95) phase = 'Nearly Pure Noise';
                            viz.screenText(phase, viz.width - 14, 20, viz.colors.yellow, 13, 'right');
                        }

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the forward process is variance-preserving: if \\(\\mathbf{x}_{t-1}\\) has identity covariance, then \\(\\mathbf{x}_t\\) also has identity covariance.',
                    hint: 'Use the fact that \\(\\operatorname{Var}(aX + bZ) = a^2 \\operatorname{Var}(X) + b^2 \\operatorname{Var}(Z)\\) when \\(X\\) and \\(Z\\) are independent.',
                    solution: 'We have \\(\\mathbf{x}_t = \\sqrt{1 - \\beta_t}\\,\\mathbf{x}_{t-1} + \\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}_t\\). Since \\(\\boldsymbol{\\epsilon}_t\\) is independent of \\(\\mathbf{x}_{t-1}\\) and both have identity covariance: \\(\\operatorname{Cov}(\\mathbf{x}_t) = (1 - \\beta_t)\\mathbf{I} + \\beta_t \\mathbf{I} = \\mathbf{I}\\).'
                },
                {
                    question: 'For a linear noise schedule with \\(\\beta_1 = 10^{-4}\\) and \\(\\beta_T = 0.02\\) and \\(T = 1000\\), compute \\(\\bar{\\alpha}_T\\). Is the final distribution close to \\(\\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\)?',
                    hint: 'Use \\(\\ln \\bar{\\alpha}_T = \\sum_{t=1}^T \\ln(1 - \\beta_t) \\approx -\\sum_{t=1}^T \\beta_t\\) for small \\(\\beta_t\\). The sum of a linear sequence is \\(T \\cdot (\\beta_1 + \\beta_T)/2\\).',
                    solution: '\\(\\sum_{t=1}^{1000} \\beta_t \\approx 1000 \\times (10^{-4} + 0.02)/2 = 1000 \\times 0.01005 = 10.05\\). So \\(\\bar{\\alpha}_T \\approx e^{-10.05} \\approx 4.3 \\times 10^{-5}\\). Since \\(\\sqrt{\\bar{\\alpha}_T} \\approx 0.0066\\), the signal coefficient is negligible and \\(q(\\mathbf{x}_T) \\approx \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\).'
                },
                {
                    question: 'Derive the posterior \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0)\\) and show it is Gaussian. Express its mean and variance in terms of \\(\\mathbf{x}_t\\), \\(\\mathbf{x}_0\\), \\(\\alpha_t\\), and \\(\\bar{\\alpha}_t\\).',
                    hint: 'Use Bayes\' rule: \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0) \\propto q(\\mathbf{x}_t \\mid \\mathbf{x}_{t-1}) \\, q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_0)\\). Since both factors are Gaussian, their product (in \\(\\mathbf{x}_{t-1}\\)) is Gaussian. Complete the square.',
                    solution: 'Both \\(q(\\mathbf{x}_t \\mid \\mathbf{x}_{t-1})\\) and \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_0)\\) are Gaussian, so the posterior is Gaussian with mean \\(\\tilde{\\boldsymbol{\\mu}}_t = \\frac{\\sqrt{\\alpha_t}(1 - \\bar{\\alpha}_{t-1})}{1 - \\bar{\\alpha}_t}\\mathbf{x}_t + \\frac{\\sqrt{\\bar{\\alpha}_{t-1}}\\,\\beta_t}{1 - \\bar{\\alpha}_t}\\mathbf{x}_0\\) and variance \\(\\tilde{\\beta}_t = \\frac{(1 - \\bar{\\alpha}_{t-1})\\,\\beta_t}{1 - \\bar{\\alpha}_t}\\). This follows from completing the square in the exponent of the product of the two Gaussian densities.'
                }
            ]
        },

        // ========== SECTION 2: Reverse Diffusion ==========
        {
            id: 'sec18-2-reverse-diffusion',
            title: 'Reverse Diffusion',
            content: `
<h2>18.2 Reverse Diffusion</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Running the Film Backwards</div><div class="env-body">
The forward process is a film of structured data dissolving into noise. If we could run this film backwards, we would see noise spontaneously organizing into meaningful data. The reverse diffusion process is exactly this: a learned, step-by-step denoising procedure that transforms Gaussian noise back into data. Each reverse step removes a small amount of noise, and a neural network provides the instructions for how to denoise at each step.
</div></div>

<p>The forward process \\(q(\\mathbf{x}_t \\mid \\mathbf{x}_{t-1})\\) destroys structure. To generate data, we need the <em>reverse</em> process \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\). Unfortunately, computing this requires knowledge of the entire data distribution \\(q(\\mathbf{x}_0)\\). So we approximate it with a learned model.</p>

<h3>The Reverse Process is Also Gaussian</h3>

<p>A foundational result (Feller, 1949; Sohl-Dickstein et al., 2015) states that when each forward step adds a sufficiently small amount of noise (i.e., \\(\\beta_t\\) is small), the reverse conditional \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) is also approximately Gaussian.</p>

<div class="env-block theorem"><div class="env-title">Theorem 18.2.1 — Gaussian Reverse Transitions</div><div class="env-body">
For small \\(\\beta_t\\), the reverse transition \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) is well-approximated by a Gaussian. This motivates the parameterization
\\[
p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t) = \\mathcal{N}\\bigl(\\mathbf{x}_{t-1};\\, \\boldsymbol{\\mu}_\\theta(\\mathbf{x}_t, t),\\; \\sigma_t^2 \\mathbf{I}\\bigr),
\\]
where \\(\\boldsymbol{\\mu}_\\theta\\) is a neural network predicting the denoised mean and \\(\\sigma_t^2\\) is a fixed or learned variance.
</div></div>

<h3>Tractable Posterior with Known \\(\\mathbf{x}_0\\)</h3>

<p>Although \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) is intractable, the posterior <em>conditioned on the clean data</em> \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0)\\) is tractable and Gaussian (as we derived in Exercise 3 of Section 18.1):</p>
\\[
q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0) = \\mathcal{N}\\bigl(\\mathbf{x}_{t-1};\\, \\tilde{\\boldsymbol{\\mu}}_t(\\mathbf{x}_t, \\mathbf{x}_0),\\; \\tilde{\\beta}_t \\mathbf{I}\\bigr),
\\]
<p>where</p>
\\[
\\tilde{\\boldsymbol{\\mu}}_t(\\mathbf{x}_t, \\mathbf{x}_0) = \\frac{\\sqrt{\\alpha_t}(1 - \\bar{\\alpha}_{t-1})}{1 - \\bar{\\alpha}_t}\\mathbf{x}_t + \\frac{\\sqrt{\\bar{\\alpha}_{t-1}}\\,\\beta_t}{1 - \\bar{\\alpha}_t}\\mathbf{x}_0, \\qquad \\tilde{\\beta}_t = \\frac{(1 - \\bar{\\alpha}_{t-1})\\,\\beta_t}{1 - \\bar{\\alpha}_t}.
\\]

<div class="env-block remark"><div class="env-title">Remark — The Training Signal</div><div class="env-body">
The fact that \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0)\\) is a known Gaussian is what makes DDPM training possible. We train \\(p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) to match this tractable posterior, using the KL divergence between two Gaussians (which has a closed-form expression).
</div></div>

<h3>Sampling Procedure</h3>

<p>Generation proceeds by ancestral sampling through the reverse chain:</p>
<ol>
<li>Sample \\(\\mathbf{x}_T \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\).</li>
<li>For \\(t = T, T-1, \\ldots, 1\\): sample \\(\\mathbf{x}_{t-1} \\sim p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\).</li>
<li>Return \\(\\mathbf{x}_0\\).</li>
</ol>

<p>Each step applies the learned denoiser, gradually sharpening the sample from noise into data. This requires \\(T\\) sequential neural network evaluations, which is why diffusion models are slower at sampling than GANs or VAEs. Techniques like DDIM (Song et al., 2020) and distillation reduce the number of required steps.</p>

<div class="env-block definition"><div class="env-title">Definition 18.2.2 — DDPM Sampling Step</div><div class="env-body">
Given the noise prediction \\(\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\), the reverse sampling step is
\\[
\\mathbf{x}_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\left(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}\\,\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\right) + \\sigma_t \\mathbf{z}, \\quad \\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}),
\\]
where \\(\\sigma_t = \\sqrt{\\tilde{\\beta}_t}\\) (or \\(\\sigma_t = \\sqrt{\\beta_t}\\) as an alternative).
</div></div>

<div class="viz-placeholder" data-viz="viz-reverse-diffusion"></div>

<div class="env-block warning"><div class="env-title">Warning — The Noise at Step \\(t=1\\)</div><div class="env-body">
At the final reverse step (\\(t = 1\\)), we do <strong>not</strong> add noise (\\(\\mathbf{z} = \\mathbf{0}\\)). Adding noise at the last step would degrade the output. This is a subtle but important implementation detail.
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-reverse-diffusion',
                    title: 'Reverse Diffusion: From Noise to Data',
                    description: 'Starting from pure Gaussian noise, watch the reverse diffusion process recover structured data step by step. Each frame applies a small denoising step. Click "Run Reverse" to animate the process.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50 });
                        const ctx = viz.ctx;

                        const N = 300;
                        const rng = () => {
                            let u = 0, v = 0;
                            while (u === 0) u = Math.random();
                            while (v === 0) v = Math.random();
                            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
                        };

                        // Target clusters
                        const clusters = [
                            { cx: -2, cy: 1.5, color: viz.colors.blue },
                            { cx: 1.5, cy: 1.8, color: viz.colors.teal },
                            { cx: 0.5, cy: -1.5, color: viz.colors.orange }
                        ];

                        // Each point has a target and current position
                        let points = [];
                        let animProgress = 1; // 1 = full noise, 0 = data
                        let animating = false;
                        let animId = null;

                        function initPoints() {
                            points = [];
                            for (let i = 0; i < N; i++) {
                                const cl = clusters[i % 3];
                                points.push({
                                    x0: cl.cx + rng() * 0.3,
                                    y0: cl.cy + rng() * 0.3,
                                    ex: rng(),
                                    ey: rng(),
                                    color: cl.color
                                });
                            }
                        }
                        initPoints();

                        const runBtn = VizEngine.createButton(controls, 'Run Reverse', () => {
                            if (animating) return;
                            initPoints();
                            animProgress = 1;
                            animating = true;
                            runBtn.textContent = 'Running...';
                            animate();
                        });

                        const resetBtn = VizEngine.createButton(controls, 'Reset', () => {
                            if (animId) cancelAnimationFrame(animId);
                            animating = false;
                            animProgress = 1;
                            initPoints();
                            runBtn.textContent = 'Run Reverse';
                            draw();
                        });

                        function getAlphaBar(frac) {
                            const betaStart = 0.0001, betaEnd = 0.02, T = 1000;
                            const integral = frac * betaStart + 0.5 * frac * frac * (betaEnd - betaStart);
                            return Math.exp(-integral * T);
                        }

                        function draw() {
                            viz.clear();

                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.3;
                            for (let gx = -6; gx <= 6; gx++) {
                                const [sx] = viz.toScreen(gx, 0);
                                ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, viz.height); ctx.stroke();
                            }
                            for (let gy = -4; gy <= 4; gy++) {
                                const [, sy] = viz.toScreen(0, gy);
                                ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(viz.width, sy); ctx.stroke();
                            }

                            const abar = getAlphaBar(animProgress);
                            const sqrtAbar = Math.sqrt(abar);
                            const sqrtOma = Math.sqrt(1 - abar);

                            for (const p of points) {
                                const xt = sqrtAbar * p.x0 + sqrtOma * p.ex;
                                const yt = sqrtAbar * p.y0 + sqrtOma * p.ey;
                                const [sx, sy] = viz.toScreen(xt, yt);
                                const alpha = 0.3 + 0.5 * (1 - animProgress);
                                ctx.globalAlpha = alpha;
                                ctx.fillStyle = p.color;
                                ctx.beginPath();
                                ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
                                ctx.fill();
                            }
                            ctx.globalAlpha = 1;

                            const step = Math.round((1 - animProgress) * 1000);
                            viz.screenText('Reverse step: ' + step + ' / 1000', 14, 20, viz.colors.white, 13, 'left');
                            viz.screenText('\u03B1\u0305_t = ' + abar.toFixed(4), 14, 38, viz.colors.teal, 12, 'left');

                            let phase = 'Pure Noise';
                            if (animProgress < 0.95 && animProgress >= 0.7) phase = 'Emerging Structure';
                            else if (animProgress < 0.7 && animProgress >= 0.3) phase = 'Clusters Forming';
                            else if (animProgress < 0.3 && animProgress > 0.02) phase = 'Refining Details';
                            else if (animProgress <= 0.02) phase = 'Clean Data';
                            viz.screenText(phase, viz.width - 14, 20, viz.colors.yellow, 13, 'right');
                        }

                        function animate() {
                            if (!animating) return;
                            animProgress -= 0.004;
                            if (animProgress <= 0) {
                                animProgress = 0;
                                animating = false;
                                runBtn.textContent = 'Run Reverse';
                            }
                            draw();
                            if (animating) {
                                animId = requestAnimationFrame(animate);
                            } else {
                                draw();
                            }
                        }

                        draw();
                        return { stopAnimation() { if (animId) cancelAnimationFrame(animId); animating = false; } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Why is the reverse of a single large noising step intractable, whereas many small reverse steps are each approximately Gaussian?',
                    hint: 'Think about what kind of distribution \\(q(\\mathbf{x}_0 \\mid \\mathbf{x}_T)\\) would need to be if \\(q(\\mathbf{x}_0)\\) is a complex multimodal distribution, versus what \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) looks like for small \\(\\beta_t\\).',
                    solution: 'If we try to reverse a single large step, \\(q(\\mathbf{x}_0 \\mid \\mathbf{x}_T)\\) must capture the <em>entire</em> complexity of the data distribution (all modes, correlations, etc.), since \\(\\mathbf{x}_T\\) is nearly pure noise and carries almost no information about which \\(\\mathbf{x}_0\\) produced it. In contrast, for small \\(\\beta_t\\), \\(\\mathbf{x}_t\\) and \\(\\mathbf{x}_{t-1}\\) are very similar, so the reverse step \\(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\) only needs to make a small correction. By a result of Feller (1949), this small correction is approximately Gaussian, regardless of the complexity of \\(q(\\mathbf{x}_0)\\).'
                },
                {
                    question: 'Write out the full ELBO (Evidence Lower Bound) for the diffusion model and identify the three types of terms that appear.',
                    hint: 'Start from \\(\\ln p(\\mathbf{x}_0) \\geq \\mathbb{E}_q\\bigl[\\ln \\frac{p_\\theta(\\mathbf{x}_{0:T})}{q(\\mathbf{x}_{1:T} \\mid \\mathbf{x}_0)}\\bigr]\\) and decompose using the chain rule.',
                    solution: 'The ELBO decomposes as: \\(-\\ln p(\\mathbf{x}_0) \\leq \\underbrace{D_{\\mathrm{KL}}(q(\\mathbf{x}_T \\mid \\mathbf{x}_0) \\| p(\\mathbf{x}_T))}_{L_T \\text{ (prior matching)}} + \\sum_{t=2}^{T} \\underbrace{D_{\\mathrm{KL}}(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0) \\| p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t))}_{L_{t-1} \\text{ (denoising matching)}} - \\underbrace{\\mathbb{E}_q[\\ln p_\\theta(\\mathbf{x}_0 \\mid \\mathbf{x}_1)]}_{L_0 \\text{ (reconstruction)}}\\). The three types are: (1) \\(L_T\\), which is constant with no learnable parameters; (2) \\(L_{t-1}\\) for \\(t = 2, \\ldots, T\\), which are KL divergences between two Gaussians; (3) \\(L_0\\), the reconstruction term.'
                },
                {
                    question: 'During sampling, we set \\(\\sigma_t = \\sqrt{\\tilde{\\beta}_t}\\). Show that \\(\\tilde{\\beta}_t \\to \\beta_t\\) as \\(\\bar{\\alpha}_{t-1} \\to 1\\) (i.e., early in the chain when little noise has been added).',
                    hint: 'Substitute \\(\\bar{\\alpha}_{t-1} = 1\\) into the formula for \\(\\tilde{\\beta}_t\\).',
                    solution: '\\(\\tilde{\\beta}_t = \\frac{(1 - \\bar{\\alpha}_{t-1})\\beta_t}{1 - \\bar{\\alpha}_t}\\). When \\(\\bar{\\alpha}_{t-1} \\to 1\\), the numerator \\((1 - \\bar{\\alpha}_{t-1})\\beta_t \\to 0\\) and the denominator \\(1 - \\bar{\\alpha}_t = 1 - \\alpha_t \\bar{\\alpha}_{t-1} \\to 1 - \\alpha_t = \\beta_t\\). So \\(\\tilde{\\beta}_t \\to 0/\\beta_t\\)... Actually, let us be more careful. When \\(\\bar{\\alpha}_{t-1} \\approx 1\\), \\(1 - \\bar{\\alpha}_{t-1} \\approx \\sum_{s=1}^{t-1} \\beta_s\\) is small, and \\(1 - \\bar{\\alpha}_t \\approx \\sum_{s=1}^{t} \\beta_s\\). For \\(t\\) small (few steps taken), \\(\\tilde{\\beta}_t \\approx \\frac{(\\sum_{s=1}^{t-1}\\beta_s) \\cdot \\beta_t}{\\sum_{s=1}^{t} \\beta_s}\\), which is approximately \\(\\beta_t (1 - \\beta_t / \\sum_{s=1}^t \\beta_s)\\), converging to \\(\\beta_t\\) as \\(t\\) grows (but still early in the chain). In the limit \\(\\bar{\\alpha}_{t-1} \\to 0\\) (late in the chain), \\(\\tilde{\\beta}_t \\to \\beta_t\\) as well, since both numerator and denominator approach \\(\\beta_t\\).'
                }
            ]
        },

        // ========== SECTION 3: DDPM Training ==========
        {
            id: 'sec18-3-ddpm-training',
            title: 'DDPM Training',
            content: `
<h2>18.3 DDPM Training</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Predicting the Noise</div><div class="env-body">
DDPM (Denoising Diffusion Probabilistic Models, Ho et al. 2020) introduced an elegant reparameterization: instead of predicting the denoised mean \\(\\tilde{\\boldsymbol{\\mu}}_t\\) directly, the network predicts the <em>noise</em> \\(\\boldsymbol{\\epsilon}\\) that was added. Since we know the noise used during training (we sampled it), we can directly supervise the prediction with a simple mean-squared error loss. This reformulation turned a complex variational objective into something as clean as training a denoiser.
</div></div>

<h3>From ELBO to Noise Prediction</h3>

<p>Recall the denoising matching term from the ELBO:</p>
\\[
L_{t-1} = D_{\\mathrm{KL}}\\bigl(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0) \\,\\|\\, p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t)\\bigr).
\\]
<p>Both distributions are Gaussian with the same variance \\(\\tilde{\\beta}_t\\). The KL divergence between two Gaussians with equal variance reduces to the squared difference of their means:</p>
\\[
L_{t-1} = \\frac{1}{2\\tilde{\\beta}_t} \\left\\| \\tilde{\\boldsymbol{\\mu}}_t(\\mathbf{x}_t, \\mathbf{x}_0) - \\boldsymbol{\\mu}_\\theta(\\mathbf{x}_t, t) \\right\\|^2 + C,
\\]
<p>where \\(C\\) is independent of \\(\\theta\\).</p>

<h3>The Noise Reparameterization</h3>

<p>Since \\(\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}\\), we can express \\(\\mathbf{x}_0\\) in terms of \\(\\mathbf{x}_t\\) and \\(\\boldsymbol{\\epsilon}\\):</p>
\\[
\\mathbf{x}_0 = \\frac{1}{\\sqrt{\\bar{\\alpha}_t}}\\bigl(\\mathbf{x}_t - \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}\\bigr).
\\]
<p>Substituting into the posterior mean \\(\\tilde{\\boldsymbol{\\mu}}_t\\):</p>
\\[
\\tilde{\\boldsymbol{\\mu}}_t = \\frac{1}{\\sqrt{\\alpha_t}}\\left(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}\\boldsymbol{\\epsilon}\\right).
\\]
<p>So we parameterize the model mean as:</p>
\\[
\\boldsymbol{\\mu}_\\theta(\\mathbf{x}_t, t) = \\frac{1}{\\sqrt{\\alpha_t}}\\left(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\right),
\\]
<p>where \\(\\boldsymbol{\\epsilon}_\\theta\\) is a neural network that predicts the noise.</p>

<div class="env-block theorem"><div class="env-title">Theorem 18.3.1 — Simplified DDPM Loss</div><div class="env-body">
After dropping the time-dependent weighting factor (which Ho et al. found empirically beneficial), the DDPM training objective becomes
\\[
L_{\\text{simple}} = \\mathbb{E}_{t \\sim \\mathcal{U}\\{1,\\ldots,T\\},\\, \\mathbf{x}_0 \\sim q,\\, \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0},\\mathbf{I})} \\left[ \\left\\| \\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta\\bigl(\\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon},\\; t\\bigr) \\right\\|^2 \\right].
\\]
</div></div>

<div class="env-block remark"><div class="env-title">Remark — Why Drop the Weighting?</div><div class="env-body">
The full ELBO weights each \\(L_{t-1}\\) by \\(\\frac{\\beta_t^2}{2\\tilde{\\beta}_t \\alpha_t (1 - \\bar{\\alpha}_t)}\\), which down-weights the loss at large \\(t\\) (where the denoising task is hardest). Dropping this weight gives equal importance to all timesteps. Ho et al. (2020) found that the unweighted version (\\(L_{\\text{simple}}\\)) produces better sample quality despite being a looser bound on the log-likelihood.
</div></div>

<h3>The Training Algorithm</h3>

<div class="env-block definition"><div class="env-title">Algorithm 18.3.2 — DDPM Training</div><div class="env-body">
<p><strong>repeat</strong> until convergence:</p>
<ol>
<li>Sample \\(\\mathbf{x}_0 \\sim q(\\mathbf{x}_0)\\) from the dataset.</li>
<li>Sample \\(t \\sim \\text{Uniform}\\{1, 2, \\ldots, T\\}\\).</li>
<li>Sample \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\).</li>
<li>Compute \\(\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}\\).</li>
<li>Take gradient step on \\(\\nabla_\\theta \\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\|^2\\).</li>
</ol>
</div></div>

<p>The simplicity of this algorithm is striking. There is no adversarial training, no posterior collapse, no mode-seeking vs. mode-covering trade-off. The network simply learns to predict noise, and the diffusion framework handles the rest.</p>

<h3>Architecture: The U-Net</h3>

<p>The noise prediction network \\(\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\) is typically a U-Net with:</p>
<ul>
<li><strong>Sinusoidal time embeddings</strong> (analogous to positional encodings in Transformers) to inform the network of the current noise level \\(t\\).</li>
<li><strong>Self-attention layers</strong> at low-resolution feature maps to capture global structure.</li>
<li><strong>Skip connections</strong> between the encoder and decoder to preserve fine-grained details.</li>
<li><strong>Group normalization</strong> for stable training.</li>
</ul>

<p>The input and output have the same spatial dimensions, since the network predicts noise of the same shape as \\(\\mathbf{x}_t\\).</p>

<div class="viz-placeholder" data-viz="viz-ddpm-training"></div>
`,
            visualizations: [
                {
                    id: 'viz-ddpm-training',
                    title: 'DDPM Training: Noise Prediction Pipeline',
                    description: 'Visualize a single DDPM training step. A clean sample \\(\\mathbf{x}_0\\) is corrupted to \\(\\mathbf{x}_t\\) by adding noise \\(\\boldsymbol{\\epsilon}\\) at a random timestep \\(t\\). The model predicts \\(\\hat{\\boldsymbol{\\epsilon}}\\), and the loss is \\(\\|\\boldsymbol{\\epsilon} - \\hat{\\boldsymbol{\\epsilon}}\\|^2\\). Click "New Sample" to see different training examples.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;

                        const rng = () => {
                            let u = 0, v = 0;
                            while (u === 0) u = Math.random();
                            while (v === 0) v = Math.random();
                            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
                        };

                        // Generate a 2D "image" patch as a small grid
                        const patchSize = 8;
                        let x0 = [], eps = [], epsHat = [];
                        let tStep = 0;
                        let loss = 0;

                        function generateSample() {
                            x0 = []; eps = []; epsHat = [];
                            tStep = Math.floor(Math.random() * 900) + 50;

                            // x0: a structured pattern (checkerboard or gradient)
                            const pattern = Math.floor(Math.random() * 3);
                            for (let i = 0; i < patchSize; i++) {
                                for (let j = 0; j < patchSize; j++) {
                                    let val;
                                    if (pattern === 0) val = ((i + j) % 2 === 0) ? 0.8 : -0.8; // checkerboard
                                    else if (pattern === 1) val = (i / patchSize - 0.5) * 2; // vertical gradient
                                    else val = Math.sin(i * 0.8) * Math.cos(j * 0.8); // sinusoidal
                                    x0.push(val);
                                    eps.push(rng());
                                }
                            }

                            // alpha_bar
                            const frac = tStep / 1000;
                            const betaStart = 0.0001, betaEnd = 0.02;
                            const integral = frac * betaStart + 0.5 * frac * frac * (betaEnd - betaStart);
                            const abar = Math.exp(-integral * 1000);
                            const sqrtAbar = Math.sqrt(abar);
                            const sqrtOma = Math.sqrt(1 - abar);

                            // eps_hat: simulated prediction (eps + small error)
                            loss = 0;
                            for (let k = 0; k < patchSize * patchSize; k++) {
                                const noise = rng() * 0.3; // prediction error
                                epsHat.push(eps[k] + noise);
                                loss += (eps[k] - epsHat[k]) ** 2;
                            }
                            loss /= (patchSize * patchSize);
                        }

                        generateSample();

                        VizEngine.createButton(controls, 'New Sample', () => { generateSample(); draw(); });

                        function drawPatch(data, cx, cy, cellSize, label) {
                            const pxW = patchSize * cellSize;
                            const startX = cx - pxW / 2;
                            const startY = cy - pxW / 2;

                            for (let i = 0; i < patchSize; i++) {
                                for (let j = 0; j < patchSize; j++) {
                                    const val = data[i * patchSize + j];
                                    // Map val to color: negative = blue, positive = orange
                                    const clamped = Math.max(-2, Math.min(2, val)) / 2;
                                    let r, g, b;
                                    if (clamped >= 0) {
                                        r = Math.round(88 + clamped * 152);
                                        g = Math.round(166 - clamped * 60);
                                        b = Math.round(255 - clamped * 155);
                                    } else {
                                        r = Math.round(88 + clamped * 48);
                                        g = Math.round(166 + clamped * 66);
                                        b = Math.round(255);
                                    }
                                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                                    ctx.fillRect(startX + j * cellSize, startY + i * cellSize, cellSize - 1, cellSize - 1);
                                }
                            }

                            // Border
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(startX, startY, pxW, pxW);

                            // Label
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText(label, cx, cy + pxW / 2 + 6);
                        }

                        function drawArrow(x1, y1, x2, y2, color, label) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();

                            const angle = Math.atan2(y2 - y1, x2 - x1);
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 10 * Math.cos(angle - 0.4), y2 - 10 * Math.sin(angle - 0.4));
                            ctx.lineTo(x2 - 10 * Math.cos(angle + 0.4), y2 - 10 * Math.sin(angle + 0.4));
                            ctx.closePath();
                            ctx.fill();

                            if (label) {
                                ctx.fillStyle = color;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';
                                ctx.fillText(label, (x1 + x2) / 2, Math.min(y1, y2) - 4);
                            }
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            const cellSize = Math.min(16, Math.floor((W - 200) / (5 * patchSize)));
                            const yCenter = H / 2;

                            // Compute x_t
                            const frac = tStep / 1000;
                            const betaStart = 0.0001, betaEnd = 0.02;
                            const integral = frac * betaStart + 0.5 * frac * frac * (betaEnd - betaStart);
                            const abar = Math.exp(-integral * 1000);
                            const sqrtAbar = Math.sqrt(abar);
                            const sqrtOma = Math.sqrt(1 - abar);

                            const xt = x0.map((v, k) => sqrtAbar * v + sqrtOma * eps[k]);

                            // Layout: x0 -> x_t -> eps_hat, with eps shown above
                            const spacing = W / 5;
                            const x0cx = spacing * 0.8;
                            const xtcx = spacing * 2;
                            const epscx = spacing * 3.2;
                            const epsHatcx = spacing * 4.2;

                            drawPatch(x0, x0cx, yCenter, cellSize, 'x\u2080');
                            drawPatch(xt, xtcx, yCenter, cellSize, 'x\u209C');
                            drawPatch(eps, epscx, yCenter - 70, cellSize, '\u03B5 (true)');
                            drawPatch(epsHat, epsHatcx, yCenter, cellSize, '\u03B5\u0302 (pred)');

                            // Arrows
                            const halfPatch = patchSize * cellSize / 2;
                            drawArrow(x0cx + halfPatch + 8, yCenter, xtcx - halfPatch - 8, yCenter, viz.colors.teal, 'Add noise');
                            drawArrow(xtcx + halfPatch + 8, yCenter, epsHatcx - halfPatch - 8, yCenter, viz.colors.purple, 'U-Net');

                            // Comparison arrow between eps and epsHat
                            drawArrow(epscx, yCenter - 70 + halfPatch + 24, epsHatcx, yCenter - halfPatch - 24, viz.colors.red, 'MSE Loss');

                            // Info
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('t = ' + tStep, 14, 14);
                            ctx.fillStyle = viz.colors.teal;
                            ctx.fillText('\u03B1\u0305_t = ' + abar.toFixed(4), 14, 32);
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText('Loss = ' + loss.toFixed(4), 14, 50);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('DDPM Training Step', W / 2, H - 14);
                        }

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Starting from the KL divergence \\(L_{t-1} = D_{\\mathrm{KL}}(q(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t, \\mathbf{x}_0) \\| p_\\theta(\\mathbf{x}_{t-1} \\mid \\mathbf{x}_t))\\), derive the simplified loss \\(L_{\\text{simple}} = \\mathbb{E}[\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\|^2]\\). Clearly state each substitution.',
                    hint: 'Use the formula for KL between two Gaussians with equal variance: \\(D_{\\mathrm{KL}} = \\frac{1}{2\\sigma^2}\\|\\mu_1 - \\mu_2\\|^2\\). Then substitute the noise reparameterization of \\(\\tilde{\\boldsymbol{\\mu}}_t\\) and \\(\\boldsymbol{\\mu}_\\theta\\).',
                    solution: 'Since both distributions are \\(\\mathcal{N}(\\cdot; \\mu, \\tilde{\\beta}_t I)\\), \\(L_{t-1} = \\frac{1}{2\\tilde{\\beta}_t}\\|\\tilde{\\boldsymbol{\\mu}}_t - \\boldsymbol{\\mu}_\\theta\\|^2\\). Substituting \\(\\tilde{\\boldsymbol{\\mu}}_t = \\frac{1}{\\sqrt{\\alpha_t}}(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\boldsymbol{\\epsilon})\\) and \\(\\boldsymbol{\\mu}_\\theta = \\frac{1}{\\sqrt{\\alpha_t}}(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\boldsymbol{\\epsilon}_\\theta)\\), the \\(\\mathbf{x}_t\\) terms cancel, yielding \\(L_{t-1} = \\frac{\\beta_t^2}{2\\tilde{\\beta}_t \\alpha_t (1-\\bar{\\alpha}_t)}\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta\\|^2\\). Dropping the time-dependent coefficient gives \\(L_{\\text{simple}}\\).'
                },
                {
                    question: 'Why does the noise prediction network need to know the timestep \\(t\\)? What would go wrong if we removed the time conditioning?',
                    hint: 'Consider what the optimal prediction \\(\\boldsymbol{\\epsilon}_\\theta^*(\\mathbf{x}_t)\\) would be without time information. How does the noise-to-signal ratio affect the denoising task?',
                    solution: 'Without \\(t\\), the network receives \\(\\mathbf{x}_t\\) but does not know the noise level. At early timesteps (small \\(t\\)), \\(\\mathbf{x}_t \\approx \\mathbf{x}_0\\) with barely any noise, so the optimal strategy is to predict near-zero noise. At late timesteps (large \\(t\\)), \\(\\mathbf{x}_t\\) is nearly pure noise, and the network must extract tiny residual signal. These are fundamentally different tasks. Without \\(t\\), the network would have to infer the noise level from \\(\\mathbf{x}_t\\) itself (possible but harder), and a single set of weights would need to handle all regimes simultaneously. Conditioning on \\(t\\) allows the network to specialize its behavior for each noise level.'
                },
                {
                    question: 'Show that the noise prediction objective is equivalent to a <em>score matching</em> objective (up to a weighting). That is, predicting \\(\\boldsymbol{\\epsilon}\\) is the same as estimating \\(\\nabla_{\\mathbf{x}_t} \\ln q(\\mathbf{x}_t \\mid \\mathbf{x}_0)\\).',
                    hint: 'Compute \\(\\nabla_{\\mathbf{x}_t} \\ln q(\\mathbf{x}_t \\mid \\mathbf{x}_0)\\) for the Gaussian \\(q(\\mathbf{x}_t \\mid \\mathbf{x}_0) = \\mathcal{N}(\\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0, (1-\\bar{\\alpha}_t)\\mathbf{I})\\).',
                    solution: 'For a Gaussian \\(q(\\mathbf{x}_t \\mid \\mathbf{x}_0) = \\mathcal{N}(\\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0, (1-\\bar{\\alpha}_t)\\mathbf{I})\\), the score is \\(\\nabla_{\\mathbf{x}_t} \\ln q(\\mathbf{x}_t \\mid \\mathbf{x}_0) = -\\frac{\\mathbf{x}_t - \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0}{1 - \\bar{\\alpha}_t} = -\\frac{\\boldsymbol{\\epsilon}}{\\sqrt{1 - \\bar{\\alpha}_t}}\\). Therefore \\(\\boldsymbol{\\epsilon} = -\\sqrt{1 - \\bar{\\alpha}_t}\\,\\nabla_{\\mathbf{x}_t} \\ln q(\\mathbf{x}_t \\mid \\mathbf{x}_0)\\), and predicting \\(\\boldsymbol{\\epsilon}\\) is equivalent to estimating the score scaled by \\(-\\sqrt{1-\\bar{\\alpha}_t}\\).'
                }
            ]
        },

        // ========== SECTION 4: Score-Based Models ==========
        {
            id: 'sec18-4-score-based',
            title: 'Score-Based Models',
            content: `
<h2>18.4 Score-Based Models</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Following the Gradient Uphill</div><div class="env-body">
Imagine a mountainous landscape where peaks correspond to high-probability data regions (images of cats, for instance) and valleys are low-probability regions (random noise). The <em>score function</em> \\(\\nabla_{\\mathbf{x}} \\ln p(\\mathbf{x})\\) is a vector field that points uphill at every location, toward the nearest peak. If we drop a particle anywhere and let it follow these arrows, it will climb to a high-probability point. Score-based generative modeling learns this vector field and uses it to guide samples from noise into data.
</div></div>

<h3>The Score Function</h3>

<div class="env-block definition"><div class="env-title">Definition 18.4.1 — Score Function</div><div class="env-body">
The <em>score function</em> (or Stein score) of a distribution \\(p(\\mathbf{x})\\) is the gradient of its log-density:
\\[
\\mathbf{s}(\\mathbf{x}) = \\nabla_{\\mathbf{x}} \\ln p(\\mathbf{x}).
\\]
Note that this is a vector-valued function with the same dimensionality as \\(\\mathbf{x}\\), and it does not depend on the normalizing constant of \\(p\\) (since \\(\\nabla \\ln Z = 0\\)).
</div></div>

<p>The score has a crucial advantage over the density itself: it can be estimated <em>without knowing the normalizing constant</em>. This sidesteps the fundamental difficulty of energy-based models.</p>

<h3>Score Matching</h3>

<p>Given data \\(\\{\\mathbf{x}_i\\}\\) from \\(p_{\\text{data}}(\\mathbf{x})\\), we train a neural network \\(\\mathbf{s}_\\theta(\\mathbf{x})\\) to approximate \\(\\nabla_{\\mathbf{x}} \\ln p_{\\text{data}}(\\mathbf{x})\\) by minimizing the <em>score matching</em> objective (Hyvarinen, 2005):</p>
\\[
\\mathcal{L}_{\\text{SM}} = \\mathbb{E}_{p_{\\text{data}}}\\left[\\frac{1}{2}\\|\\mathbf{s}_\\theta(\\mathbf{x}) - \\nabla_{\\mathbf{x}} \\ln p_{\\text{data}}(\\mathbf{x})\\|^2\\right].
\\]

<p>In practice, computing \\(\\nabla_{\\mathbf{x}} \\ln p_{\\text{data}}\\) is infeasible. Two workarounds exist:</p>
<ul>
<li><strong>Denoising score matching</strong> (Vincent, 2011): Perturb data with known noise \\(q(\\tilde{\\mathbf{x}} \\mid \\mathbf{x})\\), then match the <em>perturbed</em> score \\(\\nabla_{\\tilde{\\mathbf{x}}} \\ln q(\\tilde{\\mathbf{x}} \\mid \\mathbf{x})\\), which is analytically known for Gaussian noise.</li>
<li><strong>Sliced score matching</strong> (Song et al., 2020): Project onto random directions to avoid computing the trace of the Jacobian.</li>
</ul>

<div class="env-block theorem"><div class="env-title">Theorem 18.4.2 — Denoising Score Matching Equivalence</div><div class="env-body">
Let \\(q_\\sigma(\\tilde{\\mathbf{x}} \\mid \\mathbf{x}) = \\mathcal{N}(\\tilde{\\mathbf{x}};\\, \\mathbf{x},\\, \\sigma^2 \\mathbf{I})\\). Then
\\[
\\mathbb{E}_{q_\\sigma(\\tilde{\\mathbf{x}}, \\mathbf{x})}\\left[\\|\\mathbf{s}_\\theta(\\tilde{\\mathbf{x}}) - \\nabla_{\\tilde{\\mathbf{x}}} \\ln q_\\sigma(\\tilde{\\mathbf{x}} \\mid \\mathbf{x})\\|^2\\right]
\\]
equals \\(\\mathcal{L}_{\\text{SM}}\\) (applied to the perturbed distribution \\(q_\\sigma(\\tilde{\\mathbf{x}}) = \\int q_\\sigma(\\tilde{\\mathbf{x}} \\mid \\mathbf{x}) p_{\\text{data}}(\\mathbf{x})\\,d\\mathbf{x}\\)) plus a constant independent of \\(\\theta\\). Since \\(\\nabla_{\\tilde{\\mathbf{x}}} \\ln q_\\sigma(\\tilde{\\mathbf{x}} \\mid \\mathbf{x}) = -(\\tilde{\\mathbf{x}} - \\mathbf{x})/\\sigma^2\\), training reduces to predicting the noise direction.
</div></div>

<h3>Langevin Dynamics for Sampling</h3>

<p>Once we have the score, we generate samples using <em>Langevin dynamics</em>:</p>
\\[
\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\frac{\\eta}{2}\\,\\mathbf{s}_\\theta(\\mathbf{x}_k) + \\sqrt{\\eta}\\,\\mathbf{z}_k, \\quad \\mathbf{z}_k \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}),
\\]
<p>where \\(\\eta\\) is the step size. As \\(\\eta \\to 0\\) and the number of steps \\(\\to \\infty\\), the iterates converge to samples from \\(p(\\mathbf{x})\\) under mild regularity conditions.</p>

<div class="env-block remark"><div class="env-title">Remark — The Noise Level Challenge</div><div class="env-body">
Score estimation is accurate only in regions where data is dense. In low-density regions between modes, the estimated score is unreliable. The solution (Song &amp; Ermon, 2019) is to estimate scores at <em>multiple noise levels</em> \\(\\sigma_1 &gt; \\sigma_2 &gt; \\cdots &gt; \\sigma_L\\), training a single network \\(\\mathbf{s}_\\theta(\\mathbf{x}, \\sigma)\\). At high noise levels, the distribution is smooth and the score is easy to estimate everywhere. At low noise levels, it captures fine details near modes.
</div></div>

<h3>The SDE Framework</h3>

<p>Song et al. (2021) unified DDPM and score-based models under a continuous-time framework. The forward process is an SDE (stochastic differential equation):</p>
\\[
d\\mathbf{x} = \\mathbf{f}(\\mathbf{x}, t)\\,dt + g(t)\\,d\\mathbf{w},
\\]
<p>where \\(\\mathbf{f}\\) is the drift, \\(g\\) is the diffusion coefficient, and \\(\\mathbf{w}\\) is a Wiener process. Anderson (1982) showed that the <em>reverse-time SDE</em> is</p>
\\[
d\\mathbf{x} = \\bigl[\\mathbf{f}(\\mathbf{x}, t) - g(t)^2 \\nabla_{\\mathbf{x}} \\ln p_t(\\mathbf{x})\\bigr]\\,dt + g(t)\\,d\\bar{\\mathbf{w}},
\\]
<p>where \\(\\bar{\\mathbf{w}}\\) is a reverse Wiener process. The only unknown is the score \\(\\nabla_{\\mathbf{x}} \\ln p_t(\\mathbf{x})\\), which is learned. For DDPM, the forward SDE corresponds to \\(\\mathbf{f}(\\mathbf{x}, t) = -\\frac{1}{2}\\beta(t)\\mathbf{x}\\) and \\(g(t) = \\sqrt{\\beta(t)}\\) (the Variance Preserving or VP-SDE).</p>

<div class="viz-placeholder" data-viz="viz-score-field"></div>

<div class="env-block example"><div class="env-title">Example 18.4.3 — Score of a Gaussian Mixture</div><div class="env-body">
Consider \\(p(x) = \\frac{1}{2}\\mathcal{N}(x; -2, 1) + \\frac{1}{2}\\mathcal{N}(x; 2, 1)\\). The score is
\\[
\\nabla_x \\ln p(x) = \\frac{-\\frac{1}{2}(x+2)e^{-\\frac{(x+2)^2}{2}} - \\frac{1}{2}(x-2)e^{-\\frac{(x-2)^2}{2}}}{\\frac{1}{2}e^{-\\frac{(x+2)^2}{2}} + \\frac{1}{2}e^{-\\frac{(x-2)^2}{2}}}.
\\]
Near \\(x = -2\\), the score is approximately \\(-(x+2)\\) (pointing toward \\(-2\\)). Near \\(x = 2\\), it is approximately \\(-(x-2)\\) (pointing toward \\(2\\)). At \\(x = 0\\) (between the modes), the score is zero by symmetry.
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-score-field',
                    title: 'Score Field Visualization',
                    description: 'The arrows show \\(\\nabla_{\\mathbf{x}} \\ln p_\\sigma(\\mathbf{x})\\) for a Gaussian mixture at different noise levels \\(\\sigma\\). At high \\(\\sigma\\), the field is smooth and globally coherent. At low \\(\\sigma\\), arrows sharply point toward nearby cluster centers. Adjust \\(\\sigma\\) with the slider.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50 });
                        const ctx = viz.ctx;

                        // Gaussian mixture: 3 clusters in 2D
                        const modes = [
                            { mx: -2, my: 1.5, w: 0.33 },
                            { mx: 1.5, my: 1.8, w: 0.33 },
                            { mx: 0.5, my: -1.5, w: 0.34 }
                        ];

                        let sigma = 0.8;

                        VizEngine.createSlider(controls, '\u03C3', 0.1, 3.0, sigma, 0.1, v => { sigma = v; draw(); });

                        function gaussianPdf2D(x, y, mx, my, s) {
                            const dx = x - mx, dy = y - my;
                            return Math.exp(-(dx * dx + dy * dy) / (2 * s * s)) / (2 * Math.PI * s * s);
                        }

                        function scoreAt(x, y) {
                            let totalP = 0;
                            let gradX = 0, gradY = 0;
                            for (const m of modes) {
                                const s = sigma;
                                const p = m.w * gaussianPdf2D(x, y, m.mx, m.my, s);
                                totalP += p;
                                gradX += p * (-(x - m.mx) / (s * s));
                                gradY += p * (-(y - m.my) / (s * s));
                            }
                            if (totalP < 1e-30) return [0, 0];
                            return [gradX / totalP, gradY / totalP];
                        }

                        function draw() {
                            viz.clear();

                            // Draw density as background heatmap
                            const step = 4; // pixel step
                            for (let px = 0; px < viz.width; px += step) {
                                for (let py = 0; py < viz.height; py += step) {
                                    const [wx, wy] = viz.toMath(px, py);
                                    let density = 0;
                                    for (const m of modes) {
                                        density += m.w * gaussianPdf2D(wx, wy, m.mx, m.my, sigma);
                                    }
                                    const intensity = Math.min(1, density * 8);
                                    if (intensity > 0.01) {
                                        const r = Math.round(20 + intensity * 40);
                                        const g = Math.round(20 + intensity * 100);
                                        const b = Math.round(40 + intensity * 80);
                                        ctx.fillStyle = `rgb(${r},${g},${b})`;
                                        ctx.fillRect(px, py, step, step);
                                    }
                                }
                            }

                            // Draw score arrows on a grid
                            const gridStep = 0.6;
                            const xMin = -4, xMax = 4, yMin = -3.5, yMax = 3.5;
                            const maxArrowLen = 20;

                            for (let x = xMin; x <= xMax; x += gridStep) {
                                for (let y = yMin; y <= yMax; y += gridStep) {
                                    const [sx, sy] = scoreAt(x, y);
                                    const mag = Math.sqrt(sx * sx + sy * sy);
                                    if (mag < 0.01) continue;

                                    const scale = Math.min(maxArrowLen, mag * 12) / mag;
                                    const dx = sx * scale;
                                    const dy = sy * scale;

                                    const [px, py] = viz.toScreen(x, y);
                                    // Arrow color based on magnitude
                                    const colorIntensity = Math.min(1, mag / 3);
                                    const r = Math.round(88 + colorIntensity * 160);
                                    const g = Math.round(166 - colorIntensity * 80);
                                    const b = Math.round(255 - colorIntensity * 100);
                                    const color = `rgb(${r},${g},${b})`;

                                    ctx.strokeStyle = color;
                                    ctx.lineWidth = 1.2;
                                    ctx.beginPath();
                                    ctx.moveTo(px, py);
                                    ctx.lineTo(px + dx, py - dy);
                                    ctx.stroke();

                                    // Arrow head
                                    const angle = Math.atan2(-dy, dx);
                                    ctx.fillStyle = color;
                                    ctx.beginPath();
                                    ctx.moveTo(px + dx, py - dy);
                                    ctx.lineTo(px + dx - 5 * Math.cos(angle - 0.5), py - dy - 5 * Math.sin(angle - 0.5));
                                    ctx.lineTo(px + dx - 5 * Math.cos(angle + 0.5), py - dy - 5 * Math.sin(angle + 0.5));
                                    ctx.closePath();
                                    ctx.fill();
                                }
                            }

                            // Draw mode centers
                            for (const m of modes) {
                                viz.drawPoint(m.mx, m.my, viz.colors.yellow, '', 4);
                            }

                            viz.screenText('\u03C3 = ' + sigma.toFixed(1), 14, 20, viz.colors.white, 13, 'left');
                            viz.screenText('Score field: \u2207 ln p\u209B(x)', 14, 38, viz.colors.teal, 12, 'left');
                        }

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Verify that for a Gaussian \\(p(\\mathbf{x}) = \\mathcal{N}(\\mathbf{x}; \\boldsymbol{\\mu}, \\sigma^2 \\mathbf{I})\\), the score function is \\(\\nabla_{\\mathbf{x}} \\ln p(\\mathbf{x}) = -(\\mathbf{x} - \\boldsymbol{\\mu})/\\sigma^2\\). What is the geometric interpretation?',
                    hint: 'Write out \\(\\ln p(\\mathbf{x})\\) and differentiate with respect to \\(\\mathbf{x}\\).',
                    solution: '\\(\\ln p(\\mathbf{x}) = -\\frac{\\|\\mathbf{x} - \\boldsymbol{\\mu}\\|^2}{2\\sigma^2} - \\frac{d}{2}\\ln(2\\pi\\sigma^2)\\). Taking the gradient: \\(\\nabla_{\\mathbf{x}} \\ln p(\\mathbf{x}) = -\\frac{\\mathbf{x} - \\boldsymbol{\\mu}}{\\sigma^2}\\). Geometrically, the score always points from \\(\\mathbf{x}\\) toward the mean \\(\\boldsymbol{\\mu}\\), with magnitude inversely proportional to \\(\\sigma^2\\). Points far from the mean have larger scores, indicating a stronger "pull" toward the center.'
                },
                {
                    question: 'In Langevin dynamics, explain the role of the noise term \\(\\sqrt{\\eta}\\,\\mathbf{z}_k\\). Why not just do gradient ascent \\(\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\frac{\\eta}{2}\\mathbf{s}_\\theta(\\mathbf{x}_k)\\) without noise?',
                    hint: 'Consider a multimodal distribution. Where would pure gradient ascent converge? What guarantees that Langevin dynamics samples from \\(p(\\mathbf{x})\\) rather than just finding modes?',
                    solution: 'Without noise, the dynamics would converge to a local maximum of \\(\\ln p(\\mathbf{x})\\), always finding the nearest mode. This produces point estimates, not samples from the distribution. The noise term is essential for two reasons: (1) it allows the chain to explore multiple modes by occasionally jumping away from a mode, ensuring the stationary distribution is \\(p(\\mathbf{x})\\) rather than a delta at a mode; (2) it ensures proper sampling, with points visited proportionally to their probability density. The balance between the score (drift toward high-density regions) and noise (random exploration) is precisely what makes the stationary distribution equal to \\(p(\\mathbf{x})\\).'
                },
                {
                    question: 'Write down the VP-SDE (Variance Preserving SDE) corresponding to the DDPM forward process. Then write the corresponding reverse-time SDE using Anderson\'s result.',
                    hint: 'The DDPM forward process has \\(\\mathbf{x}_t = \\sqrt{\\alpha_t}\\,\\mathbf{x}_{t-1} + \\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}\\). In continuous time, this becomes \\(d\\mathbf{x} = -\\frac{1}{2}\\beta(t)\\mathbf{x}\\,dt + \\sqrt{\\beta(t)}\\,d\\mathbf{w}\\).',
                    solution: 'The VP-SDE is \\(d\\mathbf{x} = -\\frac{1}{2}\\beta(t)\\mathbf{x}\\,dt + \\sqrt{\\beta(t)}\\,d\\mathbf{w}\\), with drift \\(\\mathbf{f}(\\mathbf{x}, t) = -\\frac{1}{2}\\beta(t)\\mathbf{x}\\) and diffusion \\(g(t) = \\sqrt{\\beta(t)}\\). By Anderson\'s formula, the reverse-time SDE is \\(d\\mathbf{x} = \\bigl[-\\frac{1}{2}\\beta(t)\\mathbf{x} - \\beta(t)\\nabla_{\\mathbf{x}} \\ln p_t(\\mathbf{x})\\bigr]dt + \\sqrt{\\beta(t)}\\,d\\bar{\\mathbf{w}}\\). The learned score network \\(\\mathbf{s}_\\theta(\\mathbf{x}, t) \\approx \\nabla_{\\mathbf{x}} \\ln p_t(\\mathbf{x})\\) replaces the unknown score to make this reverse SDE simulable.'
                }
            ]
        },

        // ========== SECTION 5: Conditional Generation & Guidance ==========
        {
            id: 'sec18-5-guidance',
            title: 'Conditional Generation & Guidance',
            content: `
<h2>18.5 Conditional Generation &amp; Guidance</h2>

<div class="env-block intuition"><div class="env-title">Intuition — Steering the Reverse Process</div><div class="env-body">
An unconditional diffusion model generates samples from the full data distribution. But we often want <em>conditional</em> generation: produce an image of a "golden retriever playing in snow," not just any image. Guidance techniques modify the reverse diffusion process to steer samples toward regions of high conditional probability, amplifying the influence of the conditioning signal. Think of it as adjusting the direction of the score arrows so they point not just toward any data, but toward data that matches a desired condition.
</div></div>

<h3>Classifier Guidance</h3>

<p>Dhariwal &amp; Nichol (2021) proposed using a separately trained classifier \\(p_\\phi(y \\mid \\mathbf{x}_t)\\) to guide the diffusion model. By Bayes' rule:</p>
\\[
\\nabla_{\\mathbf{x}_t} \\ln p(\\mathbf{x}_t \\mid y) = \\underbrace{\\nabla_{\\mathbf{x}_t} \\ln p(\\mathbf{x}_t)}_{\\text{unconditional score}} + \\underbrace{\\nabla_{\\mathbf{x}_t} \\ln p_\\phi(y \\mid \\mathbf{x}_t)}_{\\text{classifier gradient}}.
\\]

<p>The modified sampling step becomes:</p>
\\[
\\hat{\\boldsymbol{\\epsilon}} = \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t) - \\sqrt{1 - \\bar{\\alpha}_t}\\,\\gamma\\,\\nabla_{\\mathbf{x}_t} \\ln p_\\phi(y \\mid \\mathbf{x}_t),
\\]
<p>where \\(\\gamma\\) is the <em>guidance scale</em>. Larger \\(\\gamma\\) produces samples more aligned with class \\(y\\) at the cost of reduced diversity.</p>

<div class="env-block remark"><div class="env-title">Remark — Limitations of Classifier Guidance</div><div class="env-body">
Classifier guidance requires training a separate classifier on noisy inputs \\(\\mathbf{x}_t\\) (not just clean images). This classifier must operate at every noise level, which is cumbersome. It also requires gradient computation through the classifier, adding computational overhead.
</div></div>

<h3>Classifier-Free Guidance (CFG)</h3>

<p>Ho &amp; Salimans (2022) eliminated the need for a separate classifier. During training, the conditioning signal \\(y\\) is randomly dropped (replaced with a null token \\(\\varnothing\\)) with some probability (e.g., 10%). This trains a single network that can operate in both conditional and unconditional modes.</p>

<div class="env-block definition"><div class="env-title">Definition 18.5.1 — Classifier-Free Guidance</div><div class="env-body">
At sampling time, the guided noise prediction is
\\[
\\hat{\\boldsymbol{\\epsilon}} = (1 + w)\\,\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, y) - w\\,\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\varnothing),
\\]
where \\(w \\geq 0\\) is the guidance scale (often written as \\(w = s - 1\\) where \\(s\\) is the "CFG scale").
</div></div>

<p>Rearranging, this is equivalent to:</p>
\\[
\\hat{\\boldsymbol{\\epsilon}} = \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\varnothing) + (1 + w)\\bigl[\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, y) - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\varnothing)\\bigr].
\\]
<p>We start from the unconditional prediction and move in the direction that the conditioning would push us, amplified by the factor \\((1 + w)\\).</p>

<div class="env-block theorem"><div class="env-title">Theorem 18.5.2 — CFG as Implicit Classifier</div><div class="env-body">
Classifier-free guidance with scale \\(w\\) implicitly applies classifier guidance with an implicit classifier whose log-probability gradient is proportional to
\\[
\\nabla_{\\mathbf{x}_t} \\ln p_{\\text{implicit}}(y \\mid \\mathbf{x}_t) \\propto \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\varnothing) - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, y).
\\]
This implicit classifier is more powerful than an external one because it operates in the same latent space as the diffusion model and is trained jointly.
</div></div>

<h3>The Effect of Guidance Scale</h3>

<p>The guidance scale \\(w\\) (or equivalently \\(s = 1 + w\\)) controls the <em>quality-diversity trade-off</em>:</p>

<ul>
<li>\\(w = 0\\) (\\(s = 1\\)): <strong>No guidance</strong>. The model samples from the learned conditional distribution. Maximum diversity but potentially lower quality.</li>
<li>\\(w \\in (0, 5)\\) (\\(s \\in (1, 6)\\)): <strong>Moderate guidance</strong>. Sharpened distribution, improved quality and fidelity to the prompt. This is the typical operating range.</li>
<li>\\(w \\gg 1\\): <strong>Strong guidance</strong>. Samples are highly aligned with the conditioning but may become overly saturated, repetitive, or artifacted. The effective distribution becomes very peaked.</li>
</ul>

<div class="env-block example"><div class="env-title">Example 18.5.3 — Stable Diffusion Defaults</div><div class="env-body">
In Stable Diffusion (Rombach et al., 2022), the default CFG scale is \\(s = 7.5\\) (i.e., \\(w = 6.5\\)). The text prompt is encoded via CLIP and injected into the U-Net through cross-attention layers. During each denoising step, both the conditional and unconditional noise predictions are computed, requiring two forward passes per step (though batching makes this efficient).
</div></div>

<div class="viz-placeholder" data-viz="viz-cfg-scale"></div>

<div class="env-block warning"><div class="env-title">Warning — Guidance Scale is Not Free</div><div class="env-body">
Increasing \\(w\\) beyond a certain point does not improve sample quality. Very high guidance scales cause:
<ul>
<li>Color saturation and contrast artifacts.</li>
<li>Loss of fine details and textures.</li>
<li>Reduced diversity (mode collapse to a few prototypical samples).</li>
</ul>
The optimal \\(w\\) depends on the task, model, and desired trade-off.
</div></div>

<h3>Beyond CFG: Recent Advances</h3>

<p>The diffusion model framework continues to evolve rapidly:</p>
<ul>
<li><strong>DDIM</strong> (Song et al., 2020): Deterministic sampling that reduces the number of steps from 1000 to 50 or fewer.</li>
<li><strong>Latent diffusion / Stable Diffusion</strong> (Rombach et al., 2022): Run diffusion in a compressed latent space (e.g., from a VAE encoder), dramatically reducing computational cost.</li>
<li><strong>Flow matching</strong> (Lipman et al., 2022): A simplified training objective based on optimal transport that avoids the noise schedule entirely.</li>
<li><strong>Consistency models</strong> (Song et al., 2023): Distill diffusion models into single-step generators while maintaining quality.</li>
<li><strong>Rectified flows</strong> (Liu et al., 2022): Straighten the probability flow ODE trajectories for faster sampling.</li>
</ul>
`,
            visualizations: [
                {
                    id: 'viz-cfg-scale',
                    title: 'Classifier-Free Guidance Scale Effect',
                    description: 'A 1D illustration of how CFG sharpens the conditional distribution. The blue curve is the unconditional distribution \\(p(x)\\), the green curve is the conditional \\(p(x \\mid y)\\), and the orange curve is the guided distribution \\(p_w(x \\mid y)\\). Increase the guidance scale \\(w\\) to see the distribution narrow around the conditional mode.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50, originY: null });
                        viz.originY = viz.height * 0.75;
                        const ctx = viz.ctx;

                        let wScale = 2;

                        VizEngine.createSlider(controls, 'w', 0, 10, wScale, 0.5, v => { wScale = v; draw(); });

                        // Unconditional: mixture of 3 Gaussians
                        function pUncond(x) {
                            return 0.3 * gauss(x, -2, 0.7) + 0.4 * gauss(x, 1, 0.6) + 0.3 * gauss(x, 3, 0.8);
                        }

                        // Conditional on y: emphasize the middle mode
                        function pCond(x) {
                            return 0.1 * gauss(x, -2, 0.7) + 0.7 * gauss(x, 1, 0.5) + 0.2 * gauss(x, 3, 0.8);
                        }

                        // Guided: p_w(x|y) ~ p(x|y)^(1+w) / p(x)^w
                        // In log space: log p_w = (1+w) log p(x|y) - w log p(x) + const
                        function pGuided(x) {
                            const pc = pCond(x);
                            const pu = pUncond(x);
                            if (pc < 1e-20 || pu < 1e-20) return 0;
                            return Math.exp((1 + wScale) * Math.log(pc + 1e-30) - wScale * Math.log(pu + 1e-30));
                        }

                        function gauss(x, mu, sig) {
                            return Math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
                        }

                        function draw() {
                            viz.clear();

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(0, viz.originY);
                            ctx.lineTo(viz.width, viz.originY);
                            ctx.stroke();

                            // Compute guided distribution and normalize for display
                            const xMin = -5, xMax = 6;
                            const nPts = 300;
                            const dx = (xMax - xMin) / nPts;

                            // Compute normalization for guided
                            let guidedSum = 0;
                            const guidedRaw = [];
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const v = pGuided(x);
                                guidedRaw.push(v);
                                guidedSum += v * dx;
                            }

                            const yScale = 180; // pixels per unit density

                            // Draw uncond
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const [px] = viz.toScreen(x, 0);
                                const py = viz.originY - pUncond(x) * yScale;
                                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.stroke();

                            // Fill uncond
                            ctx.fillStyle = viz.colors.blue + '15';
                            ctx.beginPath();
                            const [startPx] = viz.toScreen(xMin, 0);
                            ctx.moveTo(startPx, viz.originY);
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const [px] = viz.toScreen(x, 0);
                                const py = viz.originY - pUncond(x) * yScale;
                                ctx.lineTo(px, py);
                            }
                            const [endPx] = viz.toScreen(xMax, 0);
                            ctx.lineTo(endPx, viz.originY);
                            ctx.closePath();
                            ctx.fill();

                            // Draw cond
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const [px] = viz.toScreen(x, 0);
                                const py = viz.originY - pCond(x) * yScale;
                                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.stroke();

                            // Draw guided (normalized)
                            const guidedNorm = guidedSum > 1e-30 ? 1 / guidedSum : 1;
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const [px] = viz.toScreen(x, 0);
                                const val = guidedRaw[i] * guidedNorm;
                                const py = viz.originY - val * yScale;
                                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.stroke();

                            // Fill guided
                            ctx.fillStyle = viz.colors.orange + '20';
                            ctx.beginPath();
                            ctx.moveTo(startPx, viz.originY);
                            for (let i = 0; i <= nPts; i++) {
                                const x = xMin + i * dx;
                                const [px] = viz.toScreen(x, 0);
                                const val = guidedRaw[i] * guidedNorm;
                                const py = viz.originY - val * yScale;
                                ctx.lineTo(px, py);
                            }
                            ctx.lineTo(endPx, viz.originY);
                            ctx.closePath();
                            ctx.fill();

                            // Legend
                            const legendX = viz.width - 180;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(legendX, 16, 20, 3);
                            ctx.fillText('p(x) unconditional', legendX + 28, 22);

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(legendX, 34, 20, 3);
                            ctx.fillText('p(x|y) conditional', legendX + 28, 40);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(legendX, 52, 20, 3);
                            ctx.fillText('p_w(x|y) guided', legendX + 28, 58);

                            // Scale label
                            viz.screenText('w = ' + wScale.toFixed(1) + '  (CFG scale s = ' + (1 + wScale).toFixed(1) + ')', 14, 20, viz.colors.white, 13, 'left');

                            // x-axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (let x = -4; x <= 5; x += 2) {
                                const [px] = viz.toScreen(x, 0);
                                ctx.fillText(x.toString(), px, viz.originY + 14);
                            }
                        }

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that classifier-free guidance with scale \\(w\\) is equivalent to sampling from the distribution \\(p_w(\\mathbf{x} \\mid y) \\propto p(\\mathbf{x} \\mid y)^{1+w} / p(\\mathbf{x})^w\\). What happens when \\(w = 0\\)? When \\(w \\to \\infty\\)?',
                    hint: 'Take the log, apply the gradient, and use the relationship \\(\\nabla \\ln p(\\mathbf{x} \\mid y) = \\nabla \\ln p(\\mathbf{x}) + \\nabla \\ln p(y \\mid \\mathbf{x})\\).',
                    solution: 'The CFG score is \\(\\nabla \\ln p_w(\\mathbf{x} \\mid y) = (1+w) \\nabla \\ln p(\\mathbf{x} \\mid y) - w \\nabla \\ln p(\\mathbf{x})\\). Writing \\(\\ln p_w \\propto (1+w) \\ln p(\\mathbf{x} \\mid y) - w \\ln p(\\mathbf{x})\\) gives \\(p_w \\propto p(\\mathbf{x}\\mid y)^{1+w}/p(\\mathbf{x})^w\\). When \\(w=0\\): \\(p_w = p(\\mathbf{x} \\mid y)\\), the standard conditional. When \\(w \\to \\infty\\): \\(p_w \\propto [p(\\mathbf{x}\\mid y)/p(\\mathbf{x})]^w \\cdot p(\\mathbf{x}\\mid y) \\propto p(y \\mid \\mathbf{x})^w \\cdot p(\\mathbf{x}\\mid y)\\), which concentrates all mass on the \\(\\mathbf{x}\\) that maximizes \\(p(y \\mid \\mathbf{x})\\), i.e., the most class-typical samples.'
                },
                {
                    question: 'Classifier-free guidance requires two forward passes per denoising step (one conditional, one unconditional). Propose a method to reduce this computational cost, and discuss any trade-offs.',
                    hint: 'Think about batching, caching, or approximation strategies. Could you use a smaller model for one of the two passes?',
                    solution: 'Several strategies exist: (1) <strong>Batching</strong>: Compute both passes as a single batched forward pass (2x batch, 1 pass). This is the standard approach and has no quality cost but does not reduce FLOPs. (2) <strong>Negative prompt caching</strong>: The unconditional prediction \\(\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t, \\varnothing)\\) changes slowly across steps; it can be cached and reused every few steps with minimal quality loss. (3) <strong>Distillation</strong>: Train a student model that directly outputs the guided prediction in a single pass, eliminating the need for two evaluations entirely. (4) <strong>Approximate guidance</strong>: Use a smaller, distilled model for the unconditional pass and the full model for the conditional pass. Each approach trades some fidelity or training cost for inference speed.'
                },
                {
                    question: 'Latent diffusion models (Stable Diffusion) run the diffusion process in a latent space \\(\\mathbf{z} = \\mathcal{E}(\\mathbf{x})\\) obtained from a pretrained VAE encoder. Explain why this is more efficient than running diffusion in pixel space, and identify a potential drawback.',
                    hint: 'Consider the dimensionality reduction. A 512x512 RGB image has \\(512 \\times 512 \\times 3 \\approx 786{,}000\\) dimensions. A typical latent has shape \\(64 \\times 64 \\times 4 \\approx 16{,}384\\).',
                    solution: 'The latent space is roughly 48x smaller in dimensionality (786K vs. 16K), so each forward pass of the U-Net operates on a much smaller spatial resolution. Since the U-Net cost scales roughly quadratically with spatial dimension (due to attention layers), this provides a massive speedup (roughly \\(48^2\\approx 2300\\times\\) for attention). The VAE encoder/decoder are run only once (encode the dataset, decode the final sample), not at every diffusion step. <strong>Drawback</strong>: The reconstruction quality is limited by the VAE. Fine details that the VAE cannot faithfully encode/decode (e.g., small text, intricate textures, exact pixel-level fidelity) are lost and cannot be recovered by the diffusion model, no matter how powerful. This is why Stable Diffusion sometimes struggles with text rendering and very fine details.'
                }
            ]
        }
    ]
});
