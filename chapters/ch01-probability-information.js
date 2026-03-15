// === Chapter 1: Probability & Information Theory ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch01',
    number: 1,
    title: 'Probability & Information Theory',
    subtitle: 'Distributions, expectations, entropy, and maximum likelihood: the statistical backbone of deep learning',
    sections: [
        // ========== SECTION 1: Probability Distributions ==========
        {
            id: 'ch01-sec01',
            title: 'Probability Distributions',
            content: `
<h2>1.1 Probability Distributions</h2>

<div class="env-block intuition">
<strong>Why Probability?</strong> Deep learning is fundamentally about learning from uncertain, noisy data. We model data as samples from unknown distributions, design loss functions via probabilistic reasoning, and interpret network outputs as probabilities. A solid command of probability distributions is non-negotiable.
</div>

<h3>Discrete Distributions: PMF</h3>

<p>A <em>discrete random variable</em> \\(X\\) takes values in a countable set. Its <em>probability mass function</em> (PMF) assigns a probability to each outcome:</p>

\\[
P(X = x) = p(x), \\quad \\text{with } p(x) \\geq 0 \\text{ and } \\sum_x p(x) = 1.
\\]

<h3>Continuous Distributions: PDF and CDF</h3>

<p>A <em>continuous random variable</em> \\(X\\) is described by a <em>probability density function</em> (PDF) \\(f(x)\\) satisfying \\(f(x) \\geq 0\\) and \\(\\int_{-\\infty}^{\\infty} f(x)\\,dx = 1\\). The probability of \\(X\\) falling in an interval is</p>

\\[
P(a \\leq X \\leq b) = \\int_a^b f(x)\\,dx.
\\]

<p>Note that \\(f(x)\\) itself is <strong>not</strong> a probability; it can exceed 1. The <em>cumulative distribution function</em> (CDF) is \\(F(x) = P(X \\leq x) = \\int_{-\\infty}^{x} f(t)\\,dt\\), and satisfies \\(F'(x) = f(x)\\).</p>

<div class="env-block definition">
<strong>Definition 1.1.1 (Gaussian PDF).</strong> The Gaussian (normal) distribution with mean \\(\\mu\\) and variance \\(\\sigma^2\\) has density
\\[
f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right).
\\]
We write \\(X \\sim \\mathcal{N}(\\mu, \\sigma^2)\\). The standard normal has \\(\\mu=0, \\sigma=1\\).
</div>

<h3>Joint, Marginal, and Conditional Distributions</h3>

<p>For two random variables \\(X, Y\\), the <em>joint distribution</em> \\(p(x, y)\\) gives the probability of both events. The <em>marginal distribution</em> is obtained by summing (or integrating) out the other variable: \\(p(x) = \\sum_y p(x, y)\\). The <em>conditional distribution</em> is</p>

\\[
p(y \\mid x) = \\frac{p(x, y)}{p(x)}, \\quad \\text{provided } p(x) &gt; 0.
\\]

<div class="env-block theorem">
<strong>Theorem 1.1.2 (Bayes' Theorem).</strong>
\\[
p(\\theta \\mid \\mathcal{D}) = \\frac{p(\\mathcal{D} \\mid \\theta)\\, p(\\theta)}{p(\\mathcal{D})},
\\]
where \\(p(\\theta)\\) is the <em>prior</em>, \\(p(\\mathcal{D}\\mid\\theta)\\) the <em>likelihood</em>, \\(p(\\theta\\mid\\mathcal{D})\\) the <em>posterior</em>, and \\(p(\\mathcal{D}) = \\int p(\\mathcal{D}\\mid\\theta)p(\\theta)\\,d\\theta\\) the <em>evidence</em>.
</div>

<div class="env-block remark">
<strong>Deep Learning Connection.</strong> Bayes' theorem underpins regularization (MAP estimation), Bayesian neural networks, and variational inference (Chapter 16). The posterior \\(p(\\theta\\mid\\mathcal{D})\\) is typically intractable for neural networks, motivating approximate methods.
</div>

<div class="viz-placeholder" data-viz="viz-gaussian-pdf"></div>
`,
            visualizations: [
                {
                    id: 'viz-gaussian-pdf',
                    title: 'Gaussian PDF Explorer',
                    description: 'Adjust \\(\\mu\\) and \\(\\sigma\\) to see how the Gaussian density changes shape. The shaded area always integrates to 1.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 60, originX: null, originY: null });
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height * 0.75;
                        let mu = 0, sig = 1;
                        VizEngine.createSlider(controls, '\u03bc', -3, 3, 0, 0.1, v => { mu = v; });
                        VizEngine.createSlider(controls, '\u03c3', 0.2, 3, 1, 0.1, v => { sig = v; });

                        function gauss(x) {
                            return Math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
                        }
                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            const ctx = viz.ctx;
                            // Shaded area
                            ctx.beginPath();
                            const xMin = -6, xMax = 6, steps = 200;
                            let [sx0, sy0] = viz.toScreen(xMin, 0);
                            ctx.moveTo(sx0, sy0);
                            for (let i = 0; i <= steps; i++) {
                                const x = xMin + (xMax - xMin) * i / steps;
                                const [sx, sy] = viz.toScreen(x, gauss(x));
                                ctx.lineTo(sx, sy);
                            }
                            const [sxE, syE] = viz.toScreen(xMax, 0);
                            ctx.lineTo(sxE, syE); ctx.closePath();
                            ctx.fillStyle = viz.colors.blue + '25'; ctx.fill();
                            // Curve
                            ctx.beginPath();
                            for (let i = 0; i <= steps; i++) {
                                const x = xMin + (xMax - xMin) * i / steps;
                                const [sx, sy] = viz.toScreen(x, gauss(x));
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5; ctx.stroke();
                            // Mean line
                            const [smx, smy1] = viz.toScreen(mu, 0);
                            const [, smy2] = viz.toScreen(mu, gauss(mu));
                            ctx.setLineDash([4, 4]); ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(smx, smy1); ctx.lineTo(smx, smy2); ctx.stroke();
                            ctx.setLineDash([]);
                            // Labels
                            viz.screenText('\u03bc=' + mu.toFixed(1) + '  \u03c3=' + sig.toFixed(1) +
                                '  peak=' + gauss(mu).toFixed(3), viz.width / 2, 18, viz.colors.white, 13);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Let \\(X \\sim \\mathcal{N}(0, 1)\\). What is \\(P(|X| &lt; 1.96)\\)? Why is this number important in statistics?',
                    hint: 'Use the symmetry of the Gaussian and the fact that \\(\\Phi(1.96) \\approx 0.975\\).',
                    solution: '\\(P(|X| &lt; 1.96) = 2\\Phi(1.96) - 1 \\approx 0.95\\). This defines the 95% confidence interval, the most commonly used threshold in hypothesis testing.'
                },
                {
                    question: 'Given \\(p(x, y) = p(y \\mid x) p(x)\\), derive the marginal \\(p(y)\\) and show it equals \\(\\sum_x p(y \\mid x) p(x)\\).',
                    hint: 'Marginalize by summing the joint over all values of \\(x\\).',
                    solution: '\\(p(y) = \\sum_x p(x, y) = \\sum_x p(y \\mid x) p(x)\\). This is the <em>law of total probability</em>. It decomposes the marginal as a mixture weighted by \\(p(x)\\).'
                },
                {
                    question: 'A medical test has sensitivity \\(P(+|\\text{disease})=0.99\\) and specificity \\(P(-|\\text{healthy})=0.95\\). If disease prevalence is 0.1%, compute \\(P(\\text{disease}|+)\\) via Bayes\' theorem.',
                    hint: 'Compute \\(P(+) = P(+|D)P(D) + P(+|H)P(H)\\) first.',
                    solution: '\\(P(+) = 0.99 \\times 0.001 + 0.05 \\times 0.999 = 0.05094\\). So \\(P(D|+) = 0.00099 / 0.05094 \\approx 1.94\\%\\). Despite high sensitivity, the posterior is low because the disease is rare. This illustrates the base-rate fallacy.'
                }
            ]
        },

        // ========== SECTION 2: Expectation, Variance & Covariance ==========
        {
            id: 'ch01-sec02',
            title: 'Expectation, Variance & Covariance',
            content: `
<h2>1.2 Expectation, Variance & Covariance</h2>

<h3>Expectation</h3>

<p>The <em>expectation</em> (mean) of a function \\(g(X)\\) under distribution \\(p\\) is</p>

\\[
\\mathbb{E}[g(X)] = \\begin{cases} \\sum_x g(x)\\,p(x) & \\text{(discrete)} \\\\ \\int g(x)\\,f(x)\\,dx & \\text{(continuous)} \\end{cases}
\\]

<p>Key properties: linearity \\(\\mathbb{E}[aX + b] = a\\mathbb{E}[X] + b\\), and for independent \\(X, Y\\): \\(\\mathbb{E}[XY] = \\mathbb{E}[X]\\mathbb{E}[Y]\\).</p>

<h3>Variance</h3>

<div class="env-block definition">
<strong>Definition 1.2.1 (Variance).</strong> The variance measures spread around the mean:
\\[
\\text{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2] = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2.
\\]
The <em>standard deviation</em> is \\(\\sigma = \\sqrt{\\text{Var}(X)}\\). For a constant \\(a\\), \\(\\text{Var}(aX) = a^2 \\text{Var}(X)\\).
</div>

<h3>Covariance and Correlation</h3>

<div class="env-block definition">
<strong>Definition 1.2.2 (Covariance).</strong> For random variables \\(X, Y\\):
\\[
\\text{Cov}(X, Y) = \\mathbb{E}[(X - \\mathbb{E}[X])(Y - \\mathbb{E}[Y])] = \\mathbb{E}[XY] - \\mathbb{E}[X]\\mathbb{E}[Y].
\\]
The <em>correlation</em> normalizes covariance to \\([-1, 1]\\): \\(\\rho = \\text{Cov}(X,Y) / (\\sigma_X \\sigma_Y)\\).
</div>

<div class="env-block definition">
<strong>Definition 1.2.3 (Covariance Matrix).</strong> For a random vector \\(\\mathbf{X} \\in \\mathbb{R}^n\\), the covariance matrix is
\\[
\\boldsymbol{\\Sigma} = \\mathbb{E}[(\\mathbf{X} - \\boldsymbol{\\mu})(\\mathbf{X} - \\boldsymbol{\\mu})^\\top], \\quad \\Sigma_{ij} = \\text{Cov}(X_i, X_j).
\\]
This matrix is symmetric and positive semi-definite. Its eigenvalues give the variances along the principal axes.
</div>

<div class="env-block remark">
<strong>Deep Learning Connection.</strong> The covariance matrix appears in batch normalization (whitening inputs), in the reparameterization trick for VAEs (Chapter 16), and in natural gradient methods that use the Fisher information matrix.
</div>

<div class="viz-placeholder" data-viz="viz-2d-gaussian"></div>
`,
            visualizations: [
                {
                    id: 'viz-2d-gaussian',
                    title: '2D Gaussian Contour Ellipse',
                    description: 'Adjust the correlation \\(\\rho\\) to see how the covariance ellipse rotates and stretches. Samples cluster along the principal axis when \\(|\\rho|\\) is large.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 60 });
                        let rho = 0;
                        VizEngine.createSlider(controls, '\u03c1', -0.95, 0.95, 0, 0.05, v => { rho = v; });

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            const ctx = viz.ctx;
                            // Covariance: [[1, rho],[rho, 1]]. Eigendecomposition for ellipse.
                            const l1 = 1 + rho, l2 = 1 - rho;
                            const angle = Math.PI / 4; // eigenvectors are (1,1) and (1,-1)
                            // Draw contour ellipses at 1,2,3 sigma
                            [1, 2, 3].forEach((k, idx) => {
                                const rx = Math.sqrt(l1) * k;
                                const ry = Math.sqrt(l2) * k;
                                const alpha = ['44', '28', '18'][idx];
                                viz.drawEllipse(0, 0, rx, ry, angle, viz.colors.blue + alpha, viz.colors.blue + '88');
                            });
                            // Draw sample points
                            const seed = 42;
                            for (let i = 0; i < 120; i++) {
                                // Box-Muller with deterministic seed
                                const u1 = ((Math.sin(i * 127.1 + seed) * 43758.5453) % 1 + 1) % 1;
                                const u2 = ((Math.sin(i * 269.5 + seed * 3) * 76291.234) % 1 + 1) % 1;
                                const z1 = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
                                const z2 = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.sin(2 * Math.PI * u2);
                                // Transform: x = z1, y = rho*z1 + sqrt(1-rho^2)*z2
                                const x = z1;
                                const y = rho * z1 + Math.sqrt(Math.max(1 - rho * rho, 0)) * z2;
                                viz.drawPoint(x, y, viz.colors.teal + '99', null, 2.5);
                            }
                            // Axes labels
                            viz.screenText('\u03c1 = ' + rho.toFixed(2) +
                                '   \u03bb\u2081=' + l1.toFixed(2) + '  \u03bb\u2082=' + l2.toFixed(2),
                                viz.width / 2, 18, viz.colors.white, 13);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that \\(\\text{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2\\).',
                    hint: 'Expand \\(\\mathbb{E}[(X - \\mu)^2]\\) where \\(\\mu = \\mathbb{E}[X]\\).',
                    solution: '\\(\\text{Var}(X) = \\mathbb{E}[(X-\\mu)^2] = \\mathbb{E}[X^2 - 2\\mu X + \\mu^2] = \\mathbb{E}[X^2] - 2\\mu\\mathbb{E}[X] + \\mu^2 = \\mathbb{E}[X^2] - \\mu^2\\).'
                },
                {
                    question: 'Show that for independent \\(X, Y\\): \\(\\text{Var}(X + Y) = \\text{Var}(X) + \\text{Var}(Y)\\).',
                    hint: 'Use the fact that independence implies \\(\\text{Cov}(X,Y)=0\\).',
                    solution: '\\(\\text{Var}(X+Y) = \\text{Var}(X) + \\text{Var}(Y) + 2\\text{Cov}(X,Y)\\). Independence gives \\(\\text{Cov}(X,Y)=\\mathbb{E}[XY]-\\mathbb{E}[X]\\mathbb{E}[Y]=0\\), so the cross term vanishes.'
                },
                {
                    question: 'The covariance matrix \\(\\boldsymbol{\\Sigma} = \\begin{bmatrix}4 & 2\\\\2 & 1\\end{bmatrix}\\) has \\(\\det(\\boldsymbol{\\Sigma})=0\\). What does this mean about the joint distribution?',
                    hint: 'A singular covariance matrix means the distribution is degenerate.',
                    solution: 'The determinant is \\(4\\cdot1 - 2\\cdot2 = 0\\), so \\(\\boldsymbol{\\Sigma}\\) is singular. This means the two variables are perfectly linearly related: \\(X_2 = X_1/2 + c\\) with probability 1. The distribution is concentrated on a 1D subspace (a line), not the full 2D plane.'
                }
            ]
        },

        // ========== SECTION 3: Common Distribution Families ==========
        {
            id: 'ch01-sec03',
            title: 'Common Distribution Families',
            content: `
<h2>1.3 Common Distribution Families</h2>

<p>Deep learning uses a handful of parametric distributions repeatedly. Here we review the most important ones.</p>

<h3>Bernoulli and Categorical</h3>

<div class="env-block definition">
<strong>Definition 1.3.1 (Bernoulli).</strong> A binary random variable \\(X \\in \\{0, 1\\}\\) with \\(P(X=1) = p\\):
\\[
P(X = x) = p^x (1-p)^{1-x}, \\quad \\mathbb{E}[X] = p, \\quad \\text{Var}(X) = p(1-p).
\\]
</div>

<div class="env-block definition">
<strong>Definition 1.3.2 (Categorical).</strong> A generalization to \\(K\\) classes. \\(X \\in \\{1, \\ldots, K\\}\\) with \\(P(X=k) = \\pi_k\\), where \\(\\pi_k \\geq 0\\) and \\(\\sum_k \\pi_k = 1\\). The softmax output of a classifier parameterizes a categorical distribution.
</div>

<h3>Gaussian (Normal)</h3>

<p>The Gaussian is the most important distribution in deep learning. It arises from the Central Limit Theorem, serves as the default prior in Bayesian methods, and underlies the MSE loss function (Section 1.5).</p>

<div class="env-block definition">
<strong>Definition 1.3.3 (Multivariate Gaussian).</strong> \\(\\mathbf{X} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})\\) has density
\\[
f(\\mathbf{x}) = \\frac{1}{(2\\pi)^{d/2} |\\boldsymbol{\\Sigma}|^{1/2}} \\exp\\!\\left(-\\tfrac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^\\top \\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})\\right).
\\]
</div>

<h3>Beta Distribution</h3>

<div class="env-block definition">
<strong>Definition 1.3.4 (Beta).</strong> For \\(x \\in [0, 1]\\) with shape parameters \\(\\alpha, \\beta &gt; 0\\):
\\[
f(x; \\alpha, \\beta) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha, \\beta)}, \\quad \\mathbb{E}[X] = \\frac{\\alpha}{\\alpha+\\beta}.
\\]
The Beta is the conjugate prior for the Bernoulli parameter \\(p\\) and is useful for modeling probabilities.
</div>

<div class="viz-placeholder" data-viz="viz-dist-families"></div>
`,
            visualizations: [
                {
                    id: 'viz-dist-families',
                    title: 'Distribution Family Explorer',
                    description: 'Select a distribution and adjust its parameters to see the PDF/PMF.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 60, originX: null, originY: null });
                        viz.originX = viz.width * 0.12;
                        viz.originY = viz.height * 0.8;

                        let dist = 'gaussian', p1 = 0, p2 = 1;
                        const sliders = {};

                        // Distribution buttons
                        const btnDiv = document.createElement('div');
                        btnDiv.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;';
                        const dists = ['gaussian', 'bernoulli', 'beta'];
                        const btns = {};
                        dists.forEach(d => {
                            const b = VizEngine.createButton(btnDiv, d.charAt(0).toUpperCase() + d.slice(1), () => {
                                dist = d; updateSliders();
                                Object.values(btns).forEach(bb => bb.style.borderColor = '#30363d');
                                b.style.borderColor = '#58a6ff';
                            });
                            btns[d] = b;
                        });
                        btns['gaussian'].style.borderColor = '#58a6ff';
                        controls.appendChild(btnDiv);

                        const sliderDiv = document.createElement('div');
                        controls.appendChild(sliderDiv);

                        function updateSliders() {
                            sliderDiv.innerHTML = '';
                            if (dist === 'gaussian') {
                                p1 = 0; p2 = 1;
                                VizEngine.createSlider(sliderDiv, '\u03bc', -3, 3, 0, 0.1, v => { p1 = v; });
                                VizEngine.createSlider(sliderDiv, '\u03c3', 0.2, 3, 1, 0.1, v => { p2 = v; });
                            } else if (dist === 'bernoulli') {
                                p1 = 0.5;
                                VizEngine.createSlider(sliderDiv, 'p', 0, 1, 0.5, 0.01, v => { p1 = v; });
                            } else if (dist === 'beta') {
                                p1 = 2; p2 = 5;
                                VizEngine.createSlider(sliderDiv, '\u03b1', 0.1, 10, 2, 0.1, v => { p1 = v; });
                                VizEngine.createSlider(sliderDiv, '\u03b2', 0.1, 10, 5, 0.1, v => { p2 = v; });
                            }
                        }
                        updateSliders();

                        function logGamma(z) {
                            // Stirling approx for Beta function
                            const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
                                -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
                            let x = z, y = z, tmp = x + 5.5;
                            tmp -= (x + 0.5) * Math.log(tmp);
                            let ser = 1.000000000190015;
                            for (let j = 0; j < 6; j++) ser += c[j] / ++y;
                            return -tmp + Math.log(2.5066282746310005 * ser / x);
                        }
                        function betaFn(a, b) { return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b)); }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            // Background
                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, viz.width, viz.height);

                            if (dist === 'bernoulli') {
                                // Draw PMF as bars
                                const barW = 40, cx0 = viz.width * 0.35, cx1 = viz.width * 0.65;
                                const base = viz.originY, maxH = viz.height * 0.65;
                                // x=0 bar
                                const h0 = (1 - p1) * maxH, h1 = p1 * maxH;
                                ctx.fillStyle = viz.colors.blue + '66';
                                ctx.fillRect(cx0 - barW / 2, base - h0, barW, h0);
                                ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2;
                                ctx.strokeRect(cx0 - barW / 2, base - h0, barW, h0);
                                ctx.fillStyle = viz.colors.orange + '66';
                                ctx.fillRect(cx1 - barW / 2, base - h1, barW, h1);
                                ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2;
                                ctx.strokeRect(cx1 - barW / 2, base - h1, barW, h1);
                                // Labels
                                viz.screenText('x=0', cx0, base + 16, viz.colors.text, 13);
                                viz.screenText('x=1', cx1, base + 16, viz.colors.text, 13);
                                viz.screenText((1 - p1).toFixed(2), cx0, base - h0 - 12, viz.colors.blue, 12);
                                viz.screenText(p1.toFixed(2), cx1, base - h1 - 12, viz.colors.orange, 12);
                                // Baseline
                                ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                ctx.beginPath(); ctx.moveTo(viz.width * 0.15, base); ctx.lineTo(viz.width * 0.85, base); ctx.stroke();
                                viz.screenText('Bernoulli(p=' + p1.toFixed(2) + ')', viz.width / 2, 18, viz.colors.white, 13);
                            } else {
                                // Continuous: draw axes manually
                                const xMin = dist === 'beta' ? 0 : -6;
                                const xMax = dist === 'beta' ? 1 : 6;
                                const pxMin = viz.originX, pxMax = viz.width * 0.92;
                                const pyBase = viz.originY, pyTop = viz.height * 0.08;
                                const toSx = x => pxMin + (x - xMin) / (xMax - xMin) * (pxMax - pxMin);
                                const toSy = y => pyBase - y / (dist === 'beta' ? 4 : 0.8) * (pyBase - pyTop);

                                // Axes
                                ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
                                ctx.beginPath(); ctx.moveTo(pxMin, pyBase); ctx.lineTo(pxMax, pyBase); ctx.stroke();
                                ctx.beginPath(); ctx.moveTo(pxMin, pyBase); ctx.lineTo(pxMin, pyTop); ctx.stroke();

                                // Tick marks
                                ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                                const step = dist === 'beta' ? 0.2 : 2;
                                for (let x = xMin; x <= xMax + 0.001; x += step) {
                                    const sx = toSx(x);
                                    ctx.beginPath(); ctx.moveTo(sx, pyBase); ctx.lineTo(sx, pyBase + 4); ctx.stroke();
                                    ctx.fillText(x.toFixed(dist === 'beta' ? 1 : 0), sx, pyBase + 6);
                                }

                                // PDF curve
                                const N = 300;
                                const pdf = x => {
                                    if (dist === 'gaussian') return Math.exp(-0.5 * ((x - p1) / p2) ** 2) / (p2 * Math.sqrt(2 * Math.PI));
                                    if (dist === 'beta') {
                                        if (x <= 0.001 || x >= 0.999) return 0;
                                        return Math.pow(x, p1 - 1) * Math.pow(1 - x, p2 - 1) / betaFn(p1, p2);
                                    }
                                    return 0;
                                };

                                // Shaded area
                                ctx.beginPath();
                                ctx.moveTo(toSx(xMin), pyBase);
                                for (let i = 0; i <= N; i++) {
                                    const x = xMin + (xMax - xMin) * i / N;
                                    const y = Math.min(pdf(x), 20);
                                    ctx.lineTo(toSx(x), toSy(y));
                                }
                                ctx.lineTo(toSx(xMax), pyBase); ctx.closePath();
                                ctx.fillStyle = viz.colors.blue + '25'; ctx.fill();

                                // Line
                                ctx.beginPath();
                                for (let i = 0; i <= N; i++) {
                                    const x = xMin + (xMax - xMin) * i / N;
                                    const y = Math.min(pdf(x), 20);
                                    const sx = toSx(x), sy = toSy(y);
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5; ctx.stroke();

                                const label = dist === 'gaussian' ?
                                    'N(\u03bc=' + p1.toFixed(1) + ', \u03c3=' + p2.toFixed(1) + ')' :
                                    'Beta(\u03b1=' + p1.toFixed(1) + ', \u03b2=' + p2.toFixed(1) + ')';
                                viz.screenText(label, viz.width / 2, 18, viz.colors.white, 13);
                            }
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'If \\(X \\sim \\text{Bernoulli}(p)\\), show that the variance \\(p(1-p)\\) is maximized at \\(p=0.5\\).',
                    hint: 'Take the derivative of \\(p(1-p)\\) with respect to \\(p\\) and set it to zero.',
                    solution: '\\(d/dp[p(1-p)] = 1 - 2p = 0 \\Rightarrow p = 0.5\\). The second derivative is \\(-2 < 0\\), confirming a maximum. The maximum variance is \\(0.25\\). This means binary outcomes are most uncertain (highest entropy) when equally likely.'
                },
                {
                    question: 'Show that the Beta(1,1) distribution is the uniform distribution on \\([0,1]\\).',
                    hint: 'Substitute \\(\\alpha=\\beta=1\\) into the Beta PDF and simplify \\(B(1,1)\\).',
                    solution: '\\(f(x;1,1) = x^0(1-x)^0 / B(1,1) = 1/B(1,1)\\). Since \\(B(1,1) = \\Gamma(1)\\Gamma(1)/\\Gamma(2) = 1\\), we get \\(f(x) = 1\\) for \\(x \\in [0,1]\\), which is the uniform density.'
                },
                {
                    question: 'For \\(X \\sim \\mathcal{N}(\\mu, \\sigma^2)\\), derive the form of \\(\\log f(x)\\) and identify the sufficient statistics.',
                    hint: 'Take the log of the Gaussian PDF and collect terms involving \\(x\\).',
                    solution: '\\(\\log f(x) = -\\frac{1}{2}\\log(2\\pi\\sigma^2) - \\frac{(x-\\mu)^2}{2\\sigma^2} = -\\frac{1}{2}\\log(2\\pi\\sigma^2) - \\frac{x^2}{2\\sigma^2} + \\frac{\\mu x}{\\sigma^2} - \\frac{\\mu^2}{2\\sigma^2}\\). The sufficient statistics are \\(x\\) and \\(x^2\\), which is why sample mean and sample variance are sufficient for Gaussian parameters.'
                }
            ]
        },

        // ========== SECTION 4: Information Theory Basics ==========
        {
            id: 'ch01-sec04',
            title: 'Information Theory Basics',
            content: `
<h2>1.4 Information Theory Basics</h2>

<div class="env-block intuition">
<strong>Why Information Theory?</strong> Cross-entropy loss, the workhorse of classification, comes directly from information theory. KL divergence measures how far our model's distribution is from the true data distribution. Understanding these concepts reveals <em>why</em> we use the loss functions we do.
</div>

<h3>Shannon Entropy</h3>

<div class="env-block definition">
<strong>Definition 1.4.1 (Entropy).</strong> The Shannon entropy of a discrete distribution \\(P\\) is
\\[
H(P) = -\\sum_{x} p(x) \\log p(x).
\\]
Entropy measures the expected surprise (or uncertainty) of \\(P\\). Convention: \\(0 \\log 0 = 0\\). For a binary variable with \\(P(X=1)=p\\), the binary entropy is \\(H(p) = -p\\log p - (1-p)\\log(1-p)\\).
</div>

<div class="env-block remark">
<strong>Properties.</strong> (1) \\(H(P) \\geq 0\\). (2) \\(H(P) = 0\\) iff \\(P\\) is deterministic. (3) For \\(K\\) outcomes, \\(H(P) \\leq \\log K\\), with equality iff \\(P\\) is uniform. Entropy is maximized by the distribution with the least structure.
</div>

<h3>Cross-Entropy</h3>

<div class="env-block definition">
<strong>Definition 1.4.2 (Cross-Entropy).</strong> The cross-entropy between distributions \\(P\\) (true) and \\(Q\\) (model) is
\\[
H(P, Q) = -\\sum_x p(x) \\log q(x).
\\]
It measures the expected surprise under \\(Q\\) when the true distribution is \\(P\\). Always \\(H(P, Q) \\geq H(P)\\).
</div>

<h3>KL Divergence</h3>

<div class="env-block definition">
<strong>Definition 1.4.3 (KL Divergence).</strong> The Kullback-Leibler divergence from \\(P\\) to \\(Q\\) is
\\[
D_{\\text{KL}}(P \\| Q) = \\sum_x p(x) \\log \\frac{p(x)}{q(x)} = H(P, Q) - H(P).
\\]
</div>

<div class="env-block theorem">
<strong>Theorem 1.4.4 (Gibbs' Inequality).</strong> \\(D_{\\text{KL}}(P \\| Q) \\geq 0\\), with equality iff \\(P = Q\\).
</div>

<div class="env-block remark">
<strong>Asymmetry.</strong> KL divergence is <em>not</em> symmetric: \\(D_{\\text{KL}}(P\\|Q) \\neq D_{\\text{KL}}(Q\\|P)\\) in general. Minimizing \\(D_{\\text{KL}}(P\\|Q)\\) (forward KL) forces \\(Q\\) to cover all modes of \\(P\\); minimizing \\(D_{\\text{KL}}(Q\\|P)\\) (reverse KL) lets \\(Q\\) lock onto a single mode. This distinction is central to variational inference.
</div>

<div class="env-block remark">
<strong>Cross-Entropy Loss = KL + Constant.</strong> Since \\(H(P, Q) = D_{\\text{KL}}(P\\|Q) + H(P)\\), and \\(H(P)\\) does not depend on model parameters, minimizing cross-entropy loss is equivalent to minimizing \\(D_{\\text{KL}}(P\\|Q)\\). This is why cross-entropy is the canonical classification loss.
</div>

<div class="viz-placeholder" data-viz="viz-kl-divergence"></div>
`,
            visualizations: [
                {
                    id: 'viz-kl-divergence',
                    title: 'KL Divergence Between Two Distributions',
                    description: 'Adjust the bars of distribution \\(Q\\) to see how \\(D_{\\text{KL}}(P\\|Q)\\) changes. Notice the asymmetry and how KL diverges when \\(Q\\) assigns zero probability where \\(P\\) is nonzero.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 40 });
                        const K = 4;
                        const P = [0.4, 0.3, 0.2, 0.1];
                        const Q = [0.25, 0.25, 0.25, 0.25];
                        const labels = ['A', 'B', 'C', 'D'];

                        // Q sliders
                        const qSliders = [];
                        for (let i = 0; i < K; i++) {
                            qSliders.push(VizEngine.createSlider(controls, 'Q(' + labels[i] + ')', 0.01, 0.97, Q[i], 0.01, v => {
                                Q[i] = v; normalize();
                            }));
                        }
                        function normalize() {
                            const s = Q.reduce((a, b) => a + b, 0);
                            for (let i = 0; i < K; i++) Q[i] /= s;
                        }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, viz.width, viz.height);

                            const barW = 30, gap = 20, groupW = barW * 2 + gap;
                            const totalW = K * groupW + (K - 1) * 30;
                            const startX = (viz.width - totalW) / 2;
                            const base = viz.height * 0.78, maxH = viz.height * 0.55;

                            for (let i = 0; i < K; i++) {
                                const gx = startX + i * (groupW + 30);
                                // P bar
                                const hp = P[i] * maxH;
                                ctx.fillStyle = viz.colors.blue + '88';
                                ctx.fillRect(gx, base - hp, barW, hp);
                                ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 1.5;
                                ctx.strokeRect(gx, base - hp, barW, hp);
                                // Q bar
                                const hq = Q[i] * maxH;
                                ctx.fillStyle = viz.colors.orange + '88';
                                ctx.fillRect(gx + barW + gap, base - hq, barW, hq);
                                ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.5;
                                ctx.strokeRect(gx + barW + gap, base - hq, barW, hq);
                                // Labels
                                viz.screenText(labels[i], gx + groupW / 2, base + 16, viz.colors.text, 12);
                                viz.screenText(P[i].toFixed(2), gx + barW / 2, base - hp - 10, viz.colors.blue, 10);
                                viz.screenText(Q[i].toFixed(2), gx + barW + gap + barW / 2, base - hq - 10, viz.colors.orange, 10);
                            }

                            // Baseline
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(startX - 10, base); ctx.lineTo(startX + totalW + 10, base); ctx.stroke();

                            // Compute KL(P||Q) and H(P,Q)
                            let kl = 0, hp = 0, ce = 0;
                            for (let i = 0; i < K; i++) {
                                const pi = P[i], qi = Math.max(Q[i], 1e-10);
                                if (pi > 0) { kl += pi * Math.log(pi / qi); hp -= pi * Math.log2(pi); ce -= pi * Math.log(qi); }
                            }
                            // Legend and values
                            viz.screenText('P (true)', viz.width * 0.25, 20, viz.colors.blue, 12);
                            viz.screenText('Q (model)', viz.width * 0.50, 20, viz.colors.orange, 12);
                            viz.screenText('H(P)=' + (hp * Math.LN2 > 0 ? (hp * Math.LN2).toFixed(3) : '0.000') +
                                '   H(P,Q)=' + ce.toFixed(3) +
                                '   KL(P||Q)=' + kl.toFixed(3),
                                viz.width / 2, 42, viz.colors.white, 13);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute the entropy of a fair coin (\\(p=0.5\\)) and compare it to a biased coin (\\(p=0.9\\)) using natural log.',
                    hint: 'Apply \\(H(p) = -p\\ln p - (1-p)\\ln(1-p)\\) for each.',
                    solution: 'Fair: \\(H(0.5) = -2 \\times 0.5\\ln(0.5) = \\ln 2 \\approx 0.693\\) nats. Biased: \\(H(0.9) = -0.9\\ln(0.9) - 0.1\\ln(0.1) \\approx 0.325\\) nats. The fair coin has higher entropy (more uncertainty), confirming that entropy is maximized by the uniform distribution.'
                },
                {
                    question: 'Show that \\(D_{\\text{KL}}(P\\|Q)\\) can be infinite. Give a concrete example.',
                    hint: 'Consider what happens when \\(q(x) = 0\\) for some \\(x\\) where \\(p(x) > 0\\).',
                    solution: 'Let \\(P = (0.5, 0.5)\\) and \\(Q = (1, 0)\\). Then \\(D_{\\text{KL}}(P\\|Q) = 0.5\\log(0.5/1) + 0.5\\log(0.5/0) = -0.5\\log 2 + \\infty = \\infty\\). KL diverges whenever \\(Q\\) assigns zero probability to an event that \\(P\\) considers possible. This is why label smoothing and additive smoothing are important in practice.'
                },
                {
                    question: 'Prove Gibbs\' inequality: \\(D_{\\text{KL}}(P\\|Q) \\geq 0\\) using the inequality \\(\\ln x \\leq x - 1\\).',
                    hint: 'Write \\(-D_{\\text{KL}} = \\sum p(x) \\ln(q(x)/p(x))\\) and apply \\(\\ln(q/p) \\leq q/p - 1\\).',
                    solution: '\\(-D_{\\text{KL}}(P\\|Q) = \\sum_x p(x) \\ln\\frac{q(x)}{p(x)} \\leq \\sum_x p(x)\\left(\\frac{q(x)}{p(x)} - 1\\right) = \\sum_x q(x) - \\sum_x p(x) = 1 - 1 = 0\\). Therefore \\(D_{\\text{KL}}(P\\|Q) \\geq 0\\). Equality holds iff \\(\\ln(q(x)/p(x)) = q(x)/p(x) - 1\\) for all \\(x\\), which requires \\(q(x)/p(x) = 1\\), i.e., \\(P = Q\\).'
                }
            ]
        },

        // ========== SECTION 5: Maximum Likelihood Estimation ==========
        {
            id: 'ch01-sec05',
            title: 'Maximum Likelihood Estimation',
            content: `
<h2>1.5 Maximum Likelihood Estimation</h2>

<div class="env-block intuition">
<strong>The Central Question.</strong> Given observed data \\(\\mathcal{D} = \\{x_1, \\ldots, x_n\\}\\), how should we choose model parameters \\(\\theta\\)? Maximum likelihood estimation (MLE) answers: pick the \\(\\theta\\) that makes the observed data most probable. Nearly every loss function in deep learning can be derived from this principle.
</div>

<h3>The MLE Principle</h3>

<div class="env-block definition">
<strong>Definition 1.5.1 (Maximum Likelihood Estimator).</strong> Given i.i.d. data \\(\\{x_1, \\ldots, x_n\\}\\) and a parametric model \\(p(x; \\theta)\\), the MLE is
\\[
\\hat{\\theta}_{\\text{MLE}} = \\arg\\max_\\theta \\prod_{i=1}^n p(x_i; \\theta) = \\arg\\max_\\theta \\sum_{i=1}^n \\log p(x_i; \\theta).
\\]
Taking the log converts the product to a sum, which is numerically stable and analytically convenient.
</div>

<div class="env-block theorem">
<strong>Theorem 1.5.2 (MLE Minimizes KL Divergence).</strong> Let \\(p_{\\text{data}}\\) be the true data distribution. Then
\\[
\\hat{\\theta}_{\\text{MLE}} = \\arg\\min_\\theta D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta) = \\arg\\min_\\theta H(p_{\\text{data}}, p_\\theta).
\\]
That is, MLE is equivalent to minimizing the cross-entropy between the empirical data distribution and the model.
</div>

<h3>MLE for the Gaussian</h3>

<div class="env-block example">
<strong>Example 1.5.3.</strong> For \\(x_i \\sim \\mathcal{N}(\\mu, \\sigma^2)\\), the log-likelihood is
\\[
\\ell(\\mu, \\sigma^2) = -\\frac{n}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^n (x_i - \\mu)^2.
\\]
Setting derivatives to zero gives \\(\\hat{\\mu} = \\frac{1}{n}\\sum_i x_i\\) and \\(\\hat{\\sigma}^2 = \\frac{1}{n}\\sum_i (x_i - \\hat{\\mu})^2\\).
</div>

<div class="env-block remark">
<strong>MSE = Gaussian MLE.</strong> Minimizing mean squared error \\(\\frac{1}{n}\\sum_i(x_i - \\mu)^2\\) is equivalent to maximizing the Gaussian log-likelihood with respect to \\(\\mu\\). This is why MSE is the natural regression loss when errors are normally distributed.
</div>

<h3>MAP Estimation</h3>

<div class="env-block definition">
<strong>Definition 1.5.4 (MAP Estimator).</strong> Maximum a posteriori estimation adds a prior \\(p(\\theta)\\):
\\[
\\hat{\\theta}_{\\text{MAP}} = \\arg\\max_\\theta \\left[\\sum_{i=1}^n \\log p(x_i; \\theta) + \\log p(\\theta)\\right].
\\]
A Gaussian prior \\(\\theta \\sim \\mathcal{N}(0, \\lambda^{-1}I)\\) gives \\(\\log p(\\theta) = -\\frac{\\lambda}{2}\\|\\theta\\|^2 + \\text{const}\\), which is exactly \\(L_2\\) weight decay. A Laplace prior yields \\(L_1\\) regularization.
</div>

<div class="viz-placeholder" data-viz="viz-mle-gaussian"></div>
`,
            visualizations: [
                {
                    id: 'viz-mle-gaussian',
                    title: 'Gaussian MLE: Click to Add Data Points',
                    description: 'Click on the canvas to add data points. The fitted Gaussian updates in real time, showing how the MLE mean and variance track the sample statistics.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 60, originX: null, originY: null });
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height * 0.75;
                        const data = [];

                        VizEngine.createButton(controls, 'Clear Data', () => { data.length = 0; });
                        VizEngine.createButton(controls, 'Add 10 Random', () => {
                            for (let i = 0; i < 10; i++) {
                                // Box-Muller
                                const u = Math.random(), v = Math.random();
                                data.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 1.2 + 0.5);
                            }
                        });

                        viz.canvas.addEventListener('click', (e) => {
                            const r = viz.canvas.getBoundingClientRect();
                            const [mx] = viz.toMath(e.clientX - r.left, e.clientY - r.top);
                            data.push(mx);
                        });

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            const ctx = viz.ctx;

                            if (data.length > 0) {
                                // Compute MLE
                                const n = data.length;
                                const mu = data.reduce((a, b) => a + b, 0) / n;
                                const sig2 = data.reduce((a, x) => a + (x - mu) ** 2, 0) / n;
                                const sig = Math.sqrt(Math.max(sig2, 0.01));

                                // Draw PDF
                                const gauss = x => Math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
                                ctx.beginPath();
                                const xMin = -5, xMax = 5, steps = 200;
                                let [sx0, sy0] = viz.toScreen(xMin, 0);
                                ctx.moveTo(sx0, sy0);
                                for (let i = 0; i <= steps; i++) {
                                    const x = xMin + (xMax - xMin) * i / steps;
                                    const [sx, sy] = viz.toScreen(x, gauss(x));
                                    ctx.lineTo(sx, sy);
                                }
                                ctx.lineTo(...viz.toScreen(xMax, 0)); ctx.closePath();
                                ctx.fillStyle = viz.colors.blue + '20'; ctx.fill();
                                ctx.beginPath();
                                for (let i = 0; i <= steps; i++) {
                                    const x = xMin + (xMax - xMin) * i / steps;
                                    const [sx, sy] = viz.toScreen(x, gauss(x));
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5; ctx.stroke();

                                // Mean line
                                const [smx] = viz.toScreen(mu, 0);
                                const [, smy] = viz.toScreen(mu, gauss(mu));
                                ctx.setLineDash([4, 4]); ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.5;
                                ctx.beginPath(); ctx.moveTo(smx, viz.originY); ctx.lineTo(smx, smy); ctx.stroke();
                                ctx.setLineDash([]);

                                viz.screenText('n=' + n + '  \u03bc\u0302=' + mu.toFixed(2) + '  \u03c3\u0302=' + sig.toFixed(2),
                                    viz.width / 2, 18, viz.colors.white, 13);
                            } else {
                                viz.screenText('Click canvas to add data points', viz.width / 2, 18, viz.colors.text, 13);
                            }

                            // Draw data points on x-axis
                            data.forEach(x => {
                                const [sx, sy] = viz.toScreen(x, 0);
                                ctx.fillStyle = viz.colors.teal;
                                ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
                            });
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Derive the MLE for the parameter \\(p\\) of a Bernoulli distribution given \\(n\\) observations with \\(k\\) successes.',
                    hint: 'Write the log-likelihood \\(k\\log p + (n-k)\\log(1-p)\\) and differentiate.',
                    solution: '\\(\\ell(p) = k\\log p + (n-k)\\log(1-p)\\). Setting \\(d\\ell/dp = k/p - (n-k)/(1-p) = 0\\) gives \\(k(1-p) = (n-k)p\\), so \\(k = np\\) and \\(\\hat{p} = k/n\\). The MLE is simply the sample proportion.'
                },
                {
                    question: 'Show that minimizing cross-entropy loss for a classifier is equivalent to maximizing the log-likelihood of a categorical model.',
                    hint: 'Write the cross-entropy for one-hot labels and compare with the categorical log-likelihood.',
                    solution: 'For a one-hot label \\(y\\) (class \\(c\\)), cross-entropy is \\(-\\sum_k y_k \\log q_k = -\\log q_c\\). Summing over \\(n\\) samples: \\(-\\sum_{i=1}^n \\log q_{c_i}\\), which is the negative log-likelihood of the categorical model \\(P(Y=c_i) = q_{c_i}\\). Minimizing cross-entropy = maximizing log-likelihood.'
                },
                {
                    question: 'Explain why a Gaussian prior \\(\\theta \\sim \\mathcal{N}(0, \\lambda^{-1}I)\\) in MAP estimation is equivalent to \\(L_2\\) regularization. What is the corresponding regularization strength?',
                    hint: 'Write out \\(\\log p(\\theta)\\) for the Gaussian prior.',
                    solution: '\\(\\log p(\\theta) = -\\frac{\\lambda}{2}\\|\\theta\\|^2 - \\frac{d}{2}\\log(2\\pi/\\lambda)\\). The second term is constant w.r.t. \\(\\theta\\), so MAP becomes \\(\\arg\\max_\\theta[\\ell(\\theta) - \\frac{\\lambda}{2}\\|\\theta\\|^2]\\), or equivalently \\(\\arg\\min_\\theta[-\\ell(\\theta) + \\frac{\\lambda}{2}\\|\\theta\\|^2]\\). This is the loss plus \\(L_2\\) penalty with strength \\(\\lambda/2\\). Larger \\(\\lambda\\) (tighter prior) means stronger regularization.'
                }
            ]
        }
    ]
});
