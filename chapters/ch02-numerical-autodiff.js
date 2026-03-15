window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch02',
    number: 2,
    title: 'Numerical Computation & AutoDiff',
    subtitle: 'Numerical stability, gradient descent, computation graphs, and automatic differentiation',
    sections: [
        // ===================== Section 1: Numerical Stability =====================
        {
            id: 'ch02-sec01',
            title: 'Numerical Stability',
            content: `<h2>Numerical Stability</h2>

                <div class="env-block intuition">
                    <div class="env-title">Why Numerical Stability Matters</div>
                    <div class="env-body"><p>Deep learning models operate on floating-point numbers, which have finite precision. A 32-bit float can represent numbers roughly in the range \\(\\pm 3.4 \\times 10^{38}\\), but with only about 7 decimal digits of precision. A 64-bit double extends this to about 15 digits and a range of \\(\\pm 1.8 \\times 10^{308}\\). When computations push numbers outside these ranges, or when we subtract nearly equal quantities, catastrophic errors arise. Understanding these failure modes, and knowing the standard tricks that prevent them, is essential before building any neural network.</p></div>
                </div>

                <h3>Overflow and Underflow</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Overflow and Underflow)</div>
                    <div class="env-body"><p><strong>Overflow</strong> occurs when a number's magnitude exceeds the largest representable finite value. The result is typically stored as \\(\\pm\\infty\\). <strong>Underflow</strong> occurs when a number's magnitude is smaller than the smallest representable positive normal number. The result is either flushed to zero or stored as a <em>denormalized</em> number with reduced precision.</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Dangerous Exponentiation)</div>
                    <div class="env-body"><p>Consider computing \\(e^{1000}\\) in 32-bit floating point. Since \\(e^{1000} \\approx 10^{434}\\), this far exceeds \\(3.4 \\times 10^{38}\\), producing <code>inf</code>. Conversely, \\(e^{-1000}\\) underflows to <code>0.0</code>. Both of these arise routinely in the softmax function when input values are large.</p></div>
                </div>

                <h3>The Softmax Function</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Softmax)</div>
                    <div class="env-body"><p>Given a vector \\(\\mathbf{z} = (z_1, z_2, \\ldots, z_K) \\in \\mathbb{R}^K\\), the <strong>softmax function</strong> maps it to a probability distribution:</p>
                    <p>\\[\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}, \\quad i = 1, \\ldots, K\\]</p>
                    <p>Each output satisfies \\(0 &lt; \\text{softmax}(z_i) &lt; 1\\) and \\(\\sum_i \\text{softmax}(z_i) = 1\\).</p></div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">Overflow in Naive Softmax</div>
                    <div class="env-body"><p>If any \\(z_i\\) is large (say 1000), then \\(e^{z_i}\\) overflows to <code>inf</code>, and \\(\\text{inf}/\\text{inf} = \\text{NaN}\\). The entire computation becomes garbage. Even if only one component overflows, the denominator becomes <code>inf</code>, corrupting all outputs. This happens in practice whenever logits (the raw neural network outputs before softmax) have large magnitudes.</p></div>
                </div>

                <h3>The Softmax Stabilization Trick</h3>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Translation Invariance of Softmax)</div>
                    <div class="env-body"><p>For any constant \\(c \\in \\mathbb{R}\\),
                    \\[\\text{softmax}(z_i - c) = \\text{softmax}(z_i)\\]
                    That is, the softmax function is invariant under constant shifts of its input.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>
                    \\[\\frac{e^{z_i - c}}{\\sum_j e^{z_j - c}} = \\frac{e^{z_i} \\cdot e^{-c}}{\\sum_j e^{z_j} \\cdot e^{-c}} = \\frac{e^{z_i}}{\\sum_j e^{z_j}} = \\text{softmax}(z_i)\\]
                    </p><div class="qed">&#8718;</div></div>
                </div>

                <p>The stabilization trick exploits this invariance: set \\(c = \\max_j z_j\\). Then every exponent \\(z_i - c \\leq 0\\), so \\(e^{z_i - c} \\leq 1\\), which eliminates overflow. The largest exponent becomes \\(e^0 = 1\\), guaranteeing the denominator is at least 1.</p>

                <div class="env-block definition">
                    <div class="env-title">Algorithm (Stabilized Softmax)</div>
                    <div class="env-body">
                    <ol>
                        <li>Compute \\(c = \\max(z_1, z_2, \\ldots, z_K)\\).</li>
                        <li>Compute \\(\\tilde{z}_i = z_i - c\\) for each \\(i\\).</li>
                        <li>Return \\(\\text{softmax}(\\tilde{z}_i) = e^{\\tilde{z}_i} / \\sum_j e^{\\tilde{z}_j}\\).</li>
                    </ol>
                    </div>
                </div>

                <h3>The Log-Sum-Exp Trick</h3>

                <p>In many applications (cross-entropy loss, log-likelihood computation), we need \\(\\log \\sum_j e^{z_j}\\) rather than the softmax itself. Computing this naively is doubly dangerous: the exponentials can overflow, and if the sum underflows, the logarithm produces \\(-\\infty\\).</p>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Log-Sum-Exp Identity)</div>
                    <div class="env-body"><p>
                    \\[\\log \\sum_{j=1}^K e^{z_j} = c + \\log \\sum_{j=1}^K e^{z_j - c}\\]
                    where \\(c = \\max_j z_j\\). The right-hand side is numerically stable: all exponents are non-positive, and at least one equals 1.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>
                    \\[\\log \\sum_j e^{z_j} = \\log \\sum_j e^{z_j - c + c} = \\log \\left(e^c \\sum_j e^{z_j - c}\\right) = c + \\log \\sum_j e^{z_j - c}\\]
                    </p><div class="qed">&#8718;</div></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Stable Log-Softmax)</div>
                    <div class="env-body"><p>Combining the two tricks, we can compute the log-softmax stably:
                    \\[\\log \\text{softmax}(z_i) = z_i - \\log \\sum_j e^{z_j} = (z_i - c) - \\log \\sum_j e^{z_j - c}\\]
                    This is the formula used internally by <code>torch.nn.functional.log_softmax</code> and <code>tf.nn.log_softmax</code>. Never compute <code>log(softmax(z))</code> in two separate steps; use the fused log-softmax instead.</p></div>
                </div>

                <h3>Catastrophic Cancellation</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Catastrophic Cancellation)</div>
                    <div class="env-body"><p><strong>Catastrophic cancellation</strong> occurs when subtracting two nearly equal floating-point numbers, causing the leading significant digits to cancel and leaving only rounding errors in the remaining digits. For example, if \\(a = 1.000000\\) and \\(b = 0.999999\\) are each stored with 7 digits of precision, then \\(a - b = 0.000001\\) retains only 1 significant digit.</p></div>
                </div>

                <p>Catastrophic cancellation explains why computing variance as \\(\\mathbb{E}[X^2] - (\\mathbb{E}[X])^2\\) can produce negative results when the variance is small relative to the mean. The numerically stable alternative is the two-pass formula or Welford's online algorithm.</p>

                <div class="viz-placeholder" data-viz="ch02-softmax-stability"></div>`,

            visualizations: [
                {
                    id: 'ch02-softmax-stability',
                    title: 'Softmax Overflow: Raw vs. Stabilized',
                    description: 'Adjust the input magnitude slider to see how raw softmax overflows while the stabilized version remains correct. The bar chart shows the softmax output probabilities for a 4-element vector.',
                    setup: function(container, controls) {
                        const viz = new VizEngine(container, {width: 600, height: 400, scale: 1, originX: 0, originY: 400});
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        const state = { magnitude: 10 };
                        VizEngine.createSlider(controls, 'Input magnitude', 1, 1000, 10, 1, v => { state.magnitude = v; draw(); });

                        function softmaxRaw(z) {
                            const exps = z.map(v => Math.exp(v));
                            const sum = exps.reduce((a, b) => a + b, 0);
                            return exps.map(v => v / sum);
                        }

                        function softmaxStable(z) {
                            const c = Math.max(...z);
                            const exps = z.map(v => Math.exp(v - c));
                            const sum = exps.reduce((a, b) => a + b, 0);
                            return exps.map(v => v / sum);
                        }

                        function draw() {
                            const m = state.magnitude;
                            const z = [1.0 * m, 0.5 * m, -0.3 * m, 0.8 * m];

                            const raw = softmaxRaw(z);
                            const stable = softmaxStable(z);

                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Input z = [' + z.map(v => v.toFixed(1)).join(', ') + ']', W / 2, 10);

                            const labels = ['z\u2081', 'z\u2082', 'z\u2083', 'z\u2084'];
                            const barW = 40;
                            const gap = 20;
                            const groupGap = 60;
                            const maxBarH = 240;
                            const baseY = H - 60;

                            // --- Raw softmax (left group) ---
                            const rawGroupX = W / 2 - groupGap / 2 - (4 * barW + 3 * gap);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Raw Softmax', rawGroupX + (4 * barW + 3 * gap) / 2, 35);

                            const rawHasNaN = raw.some(v => !isFinite(v));

                            for (let i = 0; i < 4; i++) {
                                const x = rawGroupX + i * (barW + gap);
                                const val = raw[i];
                                const barH = isFinite(val) ? val * maxBarH : 0;

                                if (rawHasNaN) {
                                    // Show broken bars
                                    ctx.fillStyle = viz.colors.red + '44';
                                    ctx.fillRect(x, baseY - maxBarH * 0.5, barW, maxBarH * 0.5);
                                    ctx.strokeStyle = viz.colors.red;
                                    ctx.lineWidth = 2;
                                    ctx.setLineDash([4, 4]);
                                    ctx.strokeRect(x, baseY - maxBarH * 0.5, barW, maxBarH * 0.5);
                                    ctx.setLineDash([]);

                                    // NaN label
                                    ctx.fillStyle = viz.colors.red;
                                    ctx.font = 'bold 11px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.fillText('NaN', x + barW / 2, baseY - maxBarH * 0.5 - 8);
                                } else {
                                    ctx.fillStyle = viz.colors.orange + '88';
                                    ctx.fillRect(x, baseY - barH, barW, barH);
                                    ctx.strokeStyle = viz.colors.orange;
                                    ctx.lineWidth = 1.5;
                                    ctx.strokeRect(x, baseY - barH, barW, barH);

                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.font = '10px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.fillText(val.toFixed(4), x + barW / 2, baseY - barH - 8);
                                }

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(labels[i], x + barW / 2, baseY + 8);
                            }

                            if (rawHasNaN) {
                                ctx.fillStyle = viz.colors.red;
                                ctx.font = 'bold 12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('OVERFLOW!', rawGroupX + (4 * barW + 3 * gap) / 2, baseY - maxBarH - 15);
                            }

                            // --- Stabilized softmax (right group) ---
                            const stableGroupX = W / 2 + groupGap / 2;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Stabilized Softmax', stableGroupX + (4 * barW + 3 * gap) / 2, 35);

                            for (let i = 0; i < 4; i++) {
                                const x = stableGroupX + i * (barW + gap);
                                const val = stable[i];
                                const barH = val * maxBarH;

                                ctx.fillStyle = viz.colors.teal + '88';
                                ctx.fillRect(x, baseY - barH, barW, barH);
                                ctx.strokeStyle = viz.colors.teal;
                                ctx.lineWidth = 1.5;
                                ctx.strokeRect(x, baseY - barH, barW, barH);

                                ctx.fillStyle = viz.colors.teal;
                                ctx.font = '10px monospace';
                                ctx.textAlign = 'center';
                                ctx.fillText(val.toFixed(4), x + barW / 2, baseY - barH - 8);

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(labels[i], x + barW / 2, baseY + 8);
                            }

                            ctx.fillStyle = viz.colors.green;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('STABLE (sum = ' + stable.reduce((a, b) => a + b, 0).toFixed(6) + ')', stableGroupX + (4 * barW + 3 * gap) / 2, baseY - maxBarH - 15);

                            // Dividing line
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            ctx.moveTo(W / 2, 55);
                            ctx.lineTo(W / 2, baseY);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Bottom info
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('max(z) = ' + Math.max(...z).toFixed(1) + '   |   Subtracted before exp to prevent overflow', W / 2, H - 20);
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Compute \\(\\text{softmax}(100, 100, 100)\\) by hand. What happens if you try to evaluate \\(e^{100}\\) in 32-bit float? How does the stabilization trick help?',
                    hint: 'By symmetry, all three softmax outputs must be equal. After subtracting \\(c = 100\\), what are the shifted inputs?',
                    solution: 'By symmetry, \\(\\text{softmax}(100,100,100) = (1/3, 1/3, 1/3)\\). In 32-bit float, \\(e^{100} \\approx 2.69 \\times 10^{43}\\) is representable (it is below \\(3.4 \\times 10^{38}\\) only in single precision, so it actually overflows in float32). After stabilization: \\(c = 100\\), shifted inputs are \\((0, 0, 0)\\), so \\(e^0 / (3 \\cdot e^0) = 1/3\\). No overflow risk.'
                },
                {
                    question: 'Show that \\(\\log \\text{softmax}(z_i) = z_i - \\text{LogSumExp}(\\mathbf{z})\\). Why is computing \\(\\log(\\text{softmax}(z_i))\\) in two steps (first softmax, then log) numerically dangerous?',
                    hint: 'Write out the definition. For the danger: what if \\(\\text{softmax}(z_i)\\) is extremely small?',
                    solution: '\\(\\log \\text{softmax}(z_i) = \\log \\frac{e^{z_i}}{\\sum_j e^{z_j}} = z_i - \\log \\sum_j e^{z_j} = z_i - \\text{LSE}(\\mathbf{z})\\). Computing in two steps is dangerous because if \\(z_i \\ll \\max_j z_j\\), then \\(\\text{softmax}(z_i)\\) underflows to 0, and \\(\\log(0) = -\\infty\\). The fused formula avoids this: \\((z_i - c) - \\log \\sum_j e^{z_j - c}\\) never involves a near-zero intermediate.'
                },
                {
                    question: 'Explain why the naive variance formula \\(\\text{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2\\) can produce negative results in floating-point arithmetic. Give a concrete numerical example.',
                    hint: 'Consider a dataset where the mean is large but the variance is tiny, e.g., \\(X = \\{10^8, 10^8 + 1, 10^8 + 2\\}\\).',
                    solution: 'For \\(X = \\{10^8, 10^8+1, 10^8+2\\}\\): the true variance is \\(2/3 \\approx 0.667\\). But \\(\\mathbb{E}[X^2] \\approx 10^{16}\\) and \\((\\mathbb{E}[X])^2 \\approx 10^{16}\\). In float32 (7 significant digits), both quantities round to the same value, so their difference can be zero or even slightly negative due to rounding. The two-pass formula \\(\\text{Var} = \\frac{1}{n}\\sum(x_i - \\bar{x})^2\\) subtracts the mean first, keeping all differences small.'
                }
            ]
        },

        // ===================== Section 2: Gradient Descent =====================
        {
            id: 'ch02-sec02',
            title: 'Gradient Descent',
            content: `<h2>Gradient Descent</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Optimization Lens</div>
                    <div class="env-body"><p>Nearly all of deep learning reduces to optimization: find the parameters \\(\\boldsymbol{\\theta}\\) that minimize a loss function \\(L(\\boldsymbol{\\theta})\\). The loss surface is typically high-dimensional, non-convex, and impossible to minimize analytically. Gradient descent provides a simple, general-purpose iterative strategy: at each step, move \\(\\boldsymbol{\\theta}\\) in the direction that decreases \\(L\\) most rapidly. This direction is the negative gradient.</p></div>
                </div>

                <h3>The Gradient</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Gradient)</div>
                    <div class="env-body"><p>For a differentiable function \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\), the <strong>gradient</strong> at a point \\(\\mathbf{x}\\) is the vector of partial derivatives:
                    \\[\\nabla f(\\mathbf{x}) = \\begin{pmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\frac{\\partial f}{\\partial x_2} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{pmatrix}\\]
                    The gradient points in the direction of steepest ascent. Its magnitude \\(\\|\\nabla f(\\mathbf{x})\\|\\) gives the rate of steepest increase.</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Steepest Descent Direction)</div>
                    <div class="env-body"><p>Among all unit vectors \\(\\mathbf{u}\\) with \\(\\|\\mathbf{u}\\| = 1\\), the directional derivative \\(D_{\\mathbf{u}} f = \\nabla f \\cdot \\mathbf{u}\\) is minimized when \\(\\mathbf{u} = -\\nabla f / \\|\\nabla f\\|\\). That is, the direction of steepest descent is the negative gradient direction.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>By the Cauchy-Schwarz inequality, \\(\\nabla f \\cdot \\mathbf{u} \\geq -\\|\\nabla f\\|\\|\\mathbf{u}\\| = -\\|\\nabla f\\|\\), with equality when \\(\\mathbf{u} = -\\nabla f / \\|\\nabla f\\|\\).</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <h3>The Gradient Descent Algorithm</h3>

                <div class="env-block definition">
                    <div class="env-title">Algorithm (Gradient Descent)</div>
                    <div class="env-body">
                    <p>Given a differentiable function \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\), an initial point \\(\\mathbf{x}_0\\), and a <strong>learning rate</strong> (step size) \\(\\eta &gt; 0\\):</p>
                    <ol>
                        <li>For \\(t = 0, 1, 2, \\ldots\\):</li>
                        <li>&nbsp;&nbsp;&nbsp;&nbsp;Compute the gradient \\(\\nabla f(\\mathbf{x}_t)\\).</li>
                        <li>&nbsp;&nbsp;&nbsp;&nbsp;Update: \\(\\mathbf{x}_{t+1} = \\mathbf{x}_t - \\eta \\, \\nabla f(\\mathbf{x}_t)\\).</li>
                        <li>Stop when \\(\\|\\nabla f(\\mathbf{x}_t)\\| &lt; \\epsilon\\) or after a maximum number of iterations.</li>
                    </ol>
                    </div>
                </div>

                <h3>Learning Rate: The Critical Hyperparameter</h3>

                <p>The learning rate \\(\\eta\\) controls the step size. Its effect on convergence is dramatic:</p>
                <ul>
                    <li><strong>Too small</strong>: Convergence is painfully slow. The algorithm may require millions of steps.</li>
                    <li><strong>Too large</strong>: The iterates oscillate wildly or diverge. Each step overshoots the minimum.</li>
                    <li><strong>Just right</strong>: The algorithm converges steadily to a (local) minimum.</li>
                </ul>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Convergence for Convex Functions)</div>
                    <div class="env-body"><p>Suppose \\(f\\) is convex and \\(L\\)-smooth (i.e., \\(\\|\\nabla f(\\mathbf{x}) - \\nabla f(\\mathbf{y})\\| \\leq L \\|\\mathbf{x} - \\mathbf{y}\\|\\) for all \\(\\mathbf{x}, \\mathbf{y}\\)). Then gradient descent with \\(\\eta = 1/L\\) satisfies
                    \\[f(\\mathbf{x}_t) - f(\\mathbf{x}^*) \\leq \\frac{L \\|\\mathbf{x}_0 - \\mathbf{x}^*\\|^2}{2t}\\]
                    where \\(\\mathbf{x}^*\\) is the global minimizer. The convergence rate is \\(O(1/t)\\).</p></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Non-Convex Landscapes)</div>
                    <div class="env-body"><p>In deep learning, loss functions are almost never convex. Gradient descent can only guarantee convergence to a <em>stationary point</em> (where \\(\\nabla f = 0\\)), which could be a local minimum, saddle point, or even a local maximum. Nevertheless, empirical evidence suggests that in overparameterized networks, most local minima have loss values close to the global minimum, and saddle points are more problematic than local minima.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="ch02-gradient-descent"></div>`,

            visualizations: [
                {
                    id: 'ch02-gradient-descent',
                    title: 'Gradient Descent on a 2D Surface',
                    description: 'Watch gradient descent find the minimum. Adjust the learning rate to see convergence, oscillation, or divergence. Click "Reset" to restart from a random initial point.',
                    setup: function(container, controls) {
                        const viz = new VizEngine(container, {width: 600, height: 440, scale: 40, originX: 300, originY: 220});
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;

                        const state = { lr: 0.05, trajectory: [], running: false, step: 0 };

                        // The function: f(x,y) = 0.5*x^2 + 2.5*y^2 (an elongated bowl)
                        function f(x, y) { return 0.5 * x * x + 2.5 * y * y; }
                        function grad(x, y) { return [x, 5 * y]; }

                        function resetTrajectory() {
                            const x0 = (Math.random() - 0.5) * 10;
                            const y0 = (Math.random() - 0.5) * 4;
                            state.trajectory = [{x: x0, y: y0}];
                            state.step = 0;
                        }

                        resetTrajectory();

                        VizEngine.createSlider(controls, 'Learning rate', 0.001, 0.5, 0.05, 0.001, v => {
                            state.lr = v;
                            resetTrajectory();
                        });
                        VizEngine.createButton(controls, 'Reset', () => { resetTrajectory(); });

                        function drawContours() {
                            // Draw filled contours as heatmap
                            const imgData = ctx.createImageData(W, H);
                            for (let py = 0; py < H; py++) {
                                for (let px = 0; px < W; px++) {
                                    const [mx, my] = viz.toMath(px, py);
                                    const val = f(mx, my);
                                    const t = Math.min(val / 40, 1);
                                    const r = Math.floor(12 + t * 30);
                                    const g = Math.floor(12 + t * 15);
                                    const b = Math.floor(32 + t * 40);
                                    const idx = (py * W + px) * 4;
                                    imgData.data[idx] = r;
                                    imgData.data[idx + 1] = g;
                                    imgData.data[idx + 2] = b;
                                    imgData.data[idx + 3] = 255;
                                }
                            }
                            ctx.putImageData(imgData, 0, 0);

                            // Draw contour lines
                            const levels = [0.5, 1, 2, 4, 8, 16, 32];
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.8;
                            for (const level of levels) {
                                // For f = ax^2 + by^2 = level, this is an ellipse
                                const a = 0.5, b = 2.5;
                                const rx = Math.sqrt(level / a);
                                const ry = Math.sqrt(level / b);
                                const [cx, cy] = viz.toScreen(0, 0);
                                ctx.beginPath();
                                ctx.ellipse(cx, cy, rx * viz.scale, ry * viz.scale, 0, 0, Math.PI * 2);
                                ctx.stroke();
                            }
                        }

                        function draw() {
                            drawContours();

                            // Run GD steps
                            if (state.trajectory.length < 200) {
                                const last = state.trajectory[state.trajectory.length - 1];
                                if (Math.abs(last.x) < 50 && Math.abs(last.y) < 50) {
                                    const g = grad(last.x, last.y);
                                    const nx = last.x - state.lr * g[0];
                                    const ny = last.y - state.lr * g[1];
                                    state.trajectory.push({x: nx, y: ny});
                                }
                            }

                            // Draw trajectory
                            const traj = state.trajectory;
                            for (let i = 0; i < traj.length - 1; i++) {
                                const alpha = Math.max(0.2, 1 - i / traj.length);
                                const [sx1, sy1] = viz.toScreen(traj[i].x, traj[i].y);
                                const [sx2, sy2] = viz.toScreen(traj[i + 1].x, traj[i + 1].y);
                                ctx.strokeStyle = 'rgba(248, 81, 73, ' + alpha + ')';
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.moveTo(sx1, sy1);
                                ctx.lineTo(sx2, sy2);
                                ctx.stroke();
                            }

                            // Draw points
                            for (let i = 0; i < traj.length; i++) {
                                const [sx, sy] = viz.toScreen(traj[i].x, traj[i].y);
                                const alpha = Math.max(0.3, 1 - i / traj.length * 0.7);
                                const r = i === traj.length - 1 ? 6 : 3;
                                ctx.fillStyle = i === 0 ? viz.colors.yellow : (i === traj.length - 1 ? viz.colors.green : 'rgba(248, 81, 73, ' + alpha + ')');
                                ctx.beginPath();
                                ctx.arc(sx, sy, r, 0, Math.PI * 2);
                                ctx.fill();
                            }

                            // Draw gradient arrow at current point
                            const curr = traj[traj.length - 1];
                            const g = grad(curr.x, curr.y);
                            const gLen = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                            if (gLen > 0.01) {
                                const scale = Math.min(2, 1.5 / gLen);
                                viz.drawVector(curr.x, curr.y, curr.x - g[0] * scale, curr.y - g[1] * scale, viz.colors.blue, '', 2);
                            }

                            // Draw minimum marker
                            const [ox, oy] = viz.toScreen(0, 0);
                            ctx.strokeStyle = viz.colors.teal;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(ox - 6, oy - 6); ctx.lineTo(ox + 6, oy + 6);
                            ctx.moveTo(ox + 6, oy - 6); ctx.lineTo(ox - 6, oy + 6);
                            ctx.stroke();

                            // Info text
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '12px monospace';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            const last = traj[traj.length - 1];
                            ctx.fillText('Step ' + (traj.length - 1) + '  |  x=(' + last.x.toFixed(3) + ', ' + last.y.toFixed(3) + ')  |  f=' + f(last.x, last.y).toFixed(5), 10, 10);

                            // Legend
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillRect(10, H - 55, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Start', 24, H - 55);

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(10, H - 40, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Current', 24, H - 40);

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(10, H - 25, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('-\u2207f (neg gradient)', 24, H - 25);
                        }

                        viz.animate(function(t) {
                            draw();
                        });

                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Consider \\(f(x) = x^4\\). Write the gradient descent update rule. Starting from \\(x_0 = 2\\) with \\(\\eta = 0.01\\), compute \\(x_1\\) and \\(x_2\\).',
                    hint: '\\(f\'(x) = 4x^3\\). Apply the update \\(x_{t+1} = x_t - \\eta f\'(x_t)\\).',
                    solution: '\\(f\'(x) = 4x^3\\). Update: \\(x_{t+1} = x_t - 0.01 \\cdot 4x_t^3\\). Starting at \\(x_0 = 2\\): \\(x_1 = 2 - 0.01 \\cdot 4 \\cdot 8 = 2 - 0.32 = 1.68\\). Then \\(x_2 = 1.68 - 0.01 \\cdot 4 \\cdot (1.68)^3 = 1.68 - 0.01 \\cdot 4 \\cdot 4.7416 = 1.68 - 0.1897 = 1.4903\\).'
                },
                {
                    question: 'For the quadratic \\(f(x) = \\frac{1}{2}ax^2\\) with \\(a &gt; 0\\), show that gradient descent with learning rate \\(\\eta\\) converges if and only if \\(0 &lt; \\eta &lt; 2/a\\). What learning rate gives the fastest convergence?',
                    hint: 'The update is \\(x_{t+1} = x_t - \\eta a x_t = (1 - \\eta a) x_t\\). When does the geometric sequence converge?',
                    solution: '\\(x_{t+1} = (1 - \\eta a)x_t\\), so \\(x_t = (1 - \\eta a)^t x_0\\). This converges to 0 iff \\(|1 - \\eta a| &lt; 1\\), i.e., \\(-1 &lt; 1 - \\eta a &lt; 1\\), giving \\(0 &lt; \\eta &lt; 2/a\\). Fastest convergence when \\(|1 - \\eta a| = 0\\), i.e., \\(\\eta = 1/a\\), which gives \\(x_1 = 0\\) in one step. This matches the theorem: \\(L = a\\) is the smoothness constant, so \\(\\eta^* = 1/L = 1/a\\).'
                },
                {
                    question: 'Consider \\(f(x,y) = x^2 + 25y^2\\). The condition number is \\(\\kappa = 25\\). Explain geometrically why gradient descent oscillates on this function, and how the condition number relates to the difficulty of optimization.',
                    hint: 'Draw the elliptical contours. The gradient at most points is nearly perpendicular to the direction toward the minimum.',
                    solution: 'The contours are ellipses with semi-axes in ratio \\(1:5\\) (narrow in \\(y\\), wide in \\(x\\)). The gradient \\(\\nabla f = (2x, 50y)\\) points mostly in the \\(y\\)-direction when \\(y \\neq 0\\), even if the minimum is far away in \\(x\\). The large \\(y\\)-component forces small \\(\\eta\\) (to avoid divergence in \\(y\\)), but then progress in \\(x\\) is glacially slow. The condition number \\(\\kappa = \\lambda_{\\max}/\\lambda_{\\min} = 50/2 = 25\\) quantifies this elongation. Convergence rate scales as \\(O((\\kappa - 1)/(\\kappa + 1))^t\\), so high \\(\\kappa\\) means slow convergence.'
                }
            ]
        },

        // ===================== Section 3: Computation Graphs =====================
        {
            id: 'ch02-sec03',
            title: 'Computation Graphs',
            content: `<h2>Computation Graphs</h2>

                <div class="env-block intuition">
                    <div class="env-title">Programs as Graphs</div>
                    <div class="env-body"><p>Every mathematical expression can be decomposed into a sequence of elementary operations (addition, multiplication, exponentiation, etc.), and this decomposition can be represented as a <em>directed acyclic graph</em> (DAG). This graph structure is the foundation of automatic differentiation: by tracing how inputs flow through operations to produce outputs, we can mechanically compute derivatives by applying the chain rule at each node. Modern deep learning frameworks (PyTorch, TensorFlow, JAX) all build and manipulate computation graphs, either statically (before execution) or dynamically (during execution).</p></div>
                </div>

                <h3>Directed Acyclic Graphs</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Computation Graph)</div>
                    <div class="env-body"><p>A <strong>computation graph</strong> is a directed acyclic graph (DAG) where:</p>
                    <ul>
                        <li><strong>Leaf nodes</strong> (sources) represent input variables or constants.</li>
                        <li><strong>Interior nodes</strong> represent elementary operations (\\(+\\), \\(\\times\\), \\(\\sin\\), \\(\\exp\\), etc.).</li>
                        <li><strong>Directed edges</strong> indicate data flow: an edge from node \\(u\\) to node \\(v\\) means the output of \\(u\\) is an input to \\(v\\).</li>
                        <li>The <strong>root</strong> (sink) represents the final output.</li>
                    </ul>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Building a Computation Graph)</div>
                    <div class="env-body"><p>Consider \\(f(x, y) = (x + y) \\cdot \\sin(x)\\). We introduce intermediate variables:</p>
                    <ul>
                        <li>\\(v_1 = x\\) (input)</li>
                        <li>\\(v_2 = y\\) (input)</li>
                        <li>\\(v_3 = v_1 + v_2\\) (addition)</li>
                        <li>\\(v_4 = \\sin(v_1)\\) (sine)</li>
                        <li>\\(v_5 = v_3 \\cdot v_4\\) (multiplication, the output)</li>
                    </ul>
                    <p>The graph has two source nodes (\\(x, y\\)), three interior nodes (\\(+\\), \\(\\sin\\), \\(\\times\\)), and one root (\\(v_5\\)).</p></div>
                </div>

                <h3>Forward Evaluation (the Forward Pass)</h3>

                <p>To evaluate a computation graph, we process nodes in <strong>topological order</strong>: every node is evaluated only after all its inputs are available. This is the <strong>forward pass</strong>.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Topological Order)</div>
                    <div class="env-body"><p>A <strong>topological ordering</strong> of a DAG is a linear ordering of its nodes such that for every directed edge \\(u \\to v\\), node \\(u\\) appears before node \\(v\\). Every DAG has at least one topological ordering.</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Forward Evaluation)</div>
                    <div class="env-body"><p>Evaluating \\(f(x,y) = (x+y) \\cdot \\sin(x)\\) at \\(x = \\pi/4, y = 1\\):</p>
                    <ol>
                        <li>\\(v_1 = x = \\pi/4 \\approx 0.785\\)</li>
                        <li>\\(v_2 = y = 1\\)</li>
                        <li>\\(v_3 = v_1 + v_2 = 1.785\\)</li>
                        <li>\\(v_4 = \\sin(v_1) = \\sin(\\pi/4) \\approx 0.707\\)</li>
                        <li>\\(v_5 = v_3 \\cdot v_4 = 1.785 \\times 0.707 \\approx 1.262\\)</li>
                    </ol></div>
                </div>

                <h3>Why Computation Graphs?</h3>

                <p>Computation graphs provide three key benefits:</p>
                <ol>
                    <li><strong>Automatic differentiation</strong>: The chain rule can be applied systematically by walking the graph (forward or backward).</li>
                    <li><strong>Computational efficiency</strong>: Shared subexpressions are computed once. If \\(v_1 = x\\) feeds into both \\(v_3\\) and \\(v_4\\), we compute \\(x\\) only once.</li>
                    <li><strong>Parallelism</strong>: Independent branches of the graph can be computed simultaneously on GPUs or across machines.</li>
                </ol>

                <div class="env-block remark">
                    <div class="env-title">Remark (Static vs. Dynamic Graphs)</div>
                    <div class="env-body"><p><strong>Static graphs</strong> (TensorFlow 1.x) are built first, then executed. This allows global optimization (operator fusion, memory planning) but makes debugging harder. <strong>Dynamic graphs</strong> (PyTorch, TensorFlow Eager, JAX) are built on-the-fly during execution. Each forward pass constructs the graph anew, enabling Python control flow (if/else, loops) inside the model. The trend has shifted strongly toward dynamic graphs, with JIT compilation (e.g., <code>torch.compile</code>, <code>jax.jit</code>) recovering much of the performance advantage of static graphs.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="ch02-computation-graph"></div>`,

            visualizations: [
                {
                    id: 'ch02-computation-graph',
                    title: 'Computation Graph: f(x,y) = (x+y) \u00b7 sin(x)',
                    description: 'Adjust x and y to see values propagate through the computation graph in topological order. Each node shows its current value.',
                    setup: function(container, controls) {
                        const viz = new VizEngine(container, {width: 620, height: 420, scale: 1, originX: 0, originY: 0});
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;

                        const state = { x: 0.785, y: 1.0 };

                        VizEngine.createSlider(controls, 'x', -3.14, 3.14, 0.785, 0.01, v => { state.x = v; draw(); });
                        VizEngine.createSlider(controls, 'y', -3, 3, 1.0, 0.1, v => { state.y = v; draw(); });

                        // Node positions (px coords)
                        const nodes = {
                            x:    {px: 80,  py: 120, label: 'x',    op: 'input'},
                            y:    {px: 80,  py: 300, label: 'y',    op: 'input'},
                            add:  {px: 250, py: 210, label: '+',    op: 'add'},
                            sin:  {px: 250, py: 90,  label: 'sin',  op: 'sin'},
                            mul:  {px: 470, py: 160, label: '\u00d7',    op: 'mul'},
                            out:  {px: 570, py: 160, label: 'f',    op: 'output'}
                        };

                        const edges = [
                            ['x', 'add'], ['y', 'add'], ['x', 'sin'],
                            ['add', 'mul'], ['sin', 'mul'], ['mul', 'out']
                        ];

                        function computeValues() {
                            const v = {};
                            v.x = state.x;
                            v.y = state.y;
                            v.add = v.x + v.y;
                            v.sin = Math.sin(v.x);
                            v.mul = v.add * v.sin;
                            v.out = v.mul;
                            return v;
                        }

                        function drawArrow(x1, y1, x2, y2, color) {
                            const dx = x2 - x1, dy = y2 - y1;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            const ux = dx / len, uy = dy / len;
                            // Shorten by node radius
                            const r = 28;
                            const sx = x1 + ux * r, sy = y1 + uy * r;
                            const ex = x2 - ux * r, ey = y2 - uy * r;
                            const angle = Math.atan2(ey - sy, ex - sx);

                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(sx, sy);
                            ctx.lineTo(ex, ey);
                            ctx.stroke();

                            // Arrowhead
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(ex, ey);
                            ctx.lineTo(ex - 10 * Math.cos(angle - 0.4), ey - 10 * Math.sin(angle - 0.4));
                            ctx.lineTo(ex - 10 * Math.cos(angle + 0.4), ey - 10 * Math.sin(angle + 0.4));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function drawNode(key, val, highlight) {
                            const n = nodes[key];
                            const r = 26;

                            // Glow
                            if (highlight) {
                                ctx.fillStyle = viz.colors.teal + '22';
                                ctx.beginPath();
                                ctx.arc(n.px, n.py, r + 8, 0, Math.PI * 2);
                                ctx.fill();
                            }

                            // Circle
                            const isInput = n.op === 'input';
                            const isOutput = n.op === 'output';
                            ctx.fillStyle = isInput ? '#1a2a3a' : (isOutput ? '#2a1a2a' : '#1a1a30');
                            ctx.strokeStyle = isInput ? viz.colors.blue : (isOutput ? viz.colors.orange : viz.colors.teal);
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            ctx.arc(n.px, n.py, r, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.stroke();

                            // Op label
                            ctx.fillStyle = isInput ? viz.colors.blue : (isOutput ? viz.colors.orange : viz.colors.white);
                            ctx.font = 'bold 16px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(n.label, n.px, n.py);

                            // Value label
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px monospace';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            const valStr = isFinite(val) ? val.toFixed(3) : 'NaN';
                            ctx.fillText(valStr, n.px, n.py + r + 6);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            const vals = computeValues();

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('f(x, y) = (x + y) \u00b7 sin(x)', W / 2, 10);

                            // Draw edges first
                            for (const [from, to] of edges) {
                                drawArrow(nodes[from].px, nodes[from].py, nodes[to].px, nodes[to].py, viz.colors.grid + 'cc');
                            }

                            // Draw variable name labels near edges
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';

                            // Label for v3 = x+y
                            const addMidX = (nodes.add.px + nodes.mul.px) / 2;
                            const addMidY = (nodes.add.py + nodes.mul.py) / 2;
                            ctx.fillText('v\u2083=' + vals.add.toFixed(2), addMidX, addMidY + 16);

                            // Label for v4 = sin(x)
                            const sinMidX = (nodes.sin.px + nodes.mul.px) / 2;
                            const sinMidY = (nodes.sin.py + nodes.mul.py) / 2;
                            ctx.fillText('v\u2084=' + vals.sin.toFixed(3), sinMidX, sinMidY - 16);

                            // Draw nodes
                            for (const key of Object.keys(nodes)) {
                                drawNode(key, vals[key], false);
                            }

                            // Bottom info
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Forward pass: evaluate nodes in topological order (left to right)', W / 2, H - 15);
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Draw the computation graph for \\(f(x, y, z) = (x \\cdot y) + \\exp(y \\cdot z)\\). How many nodes and edges does it have? What is the topological order?',
                    hint: 'Introduce intermediate variables for each operation: \\(v_1 = xy\\), \\(v_2 = yz\\), \\(v_3 = \\exp(v_2)\\), \\(v_4 = v_1 + v_3\\).',
                    solution: 'The graph has 7 nodes: 3 inputs (\\(x, y, z\\)) and 4 operations (\\(\\times_1\\) for \\(xy\\); \\(\\times_2\\) for \\(yz\\); \\(\\exp\\); \\(+\\)). Edges: \\(x \\to \\times_1\\), \\(y \\to \\times_1\\), \\(y \\to \\times_2\\), \\(z \\to \\times_2\\), \\(\\times_1 \\to +\\), \\(\\times_2 \\to \\exp\\), \\(\\exp \\to +\\). That is 7 edges. One valid topological order: \\(x, y, z, \\times_1, \\times_2, \\exp, +\\). Note that \\(y\\) has two outgoing edges (it is used twice).'
                },
                {
                    question: 'Explain why a computation graph must be a DAG (no cycles). What would happen if we allowed a node to depend on its own output?',
                    hint: 'Think about whether you could evaluate such a graph in finite time.',
                    solution: 'If the graph had a cycle, then some node \\(v\\) would depend (directly or indirectly) on its own output. To compute \\(v\\), we would first need the value of \\(v\\), creating an infinite regress. There is no valid topological ordering for a graph with cycles. (Recurrent neural networks appear to have cycles, but when "unrolled" over time steps, the computation graph is acyclic: \\(h_t\\) depends on \\(h_{t-1}\\), not on \\(h_t\\) itself.)'
                },
                {
                    question: 'Consider \\(f(x) = \\sin(\\sin(\\sin(x)))\\). How many nodes does its computation graph have? Compare this to the computation graph for \\(g(x) = \\sin(x)^3\\) (three multiplications of \\(\\sin x\\) with itself). Which has more shared subexpressions?',
                    hint: 'For \\(f\\), each \\(\\sin\\) takes a different input. For \\(g\\), \\(\\sin(x)\\) is computed once and reused.',
                    solution: 'For \\(f(x) = \\sin(\\sin(\\sin(x)))\\): 4 nodes (\\(x, \\sin_1, \\sin_2, \\sin_3\\)), 3 edges, forming a chain. No shared subexpressions. For \\(g(x) = (\\sin x)^3 = \\sin(x) \\cdot \\sin(x) \\cdot \\sin(x)\\): 4 nodes if we reuse \\(\\sin(x)\\) (\\(x\\), \\(\\sin\\), \\(\\times_1 = \\sin x \\cdot \\sin x\\), \\(\\times_2 = \\times_1 \\cdot \\sin x\\)), 4 edges. \\(g\\) has the shared subexpression \\(\\sin(x)\\) feeding into two multiplication nodes. Without sharing, \\(g\\) would need 3 separate \\(\\sin\\) nodes.'
                }
            ]
        },

        // ===================== Section 4: Forward-Mode AD =====================
        {
            id: 'ch02-sec04',
            title: 'Forward-Mode AD',
            content: `<h2>Forward-Mode Automatic Differentiation</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Key Idea</div>
                    <div class="env-body"><p>Automatic differentiation (AD) is not symbolic differentiation (manipulating formulas) and not numerical differentiation (finite differences). It is <em>exact</em> (to machine precision) and <em>mechanical</em> (no human insight needed). The idea is simple: augment every intermediate value \\(v_i\\) with its derivative \\(\\dot{v}_i = \\partial v_i / \\partial x\\) (the "tangent"), and propagate both forward through the graph simultaneously. At each elementary operation, the chain rule gives the tangent of the output in terms of the tangents of the inputs.</p></div>
                </div>

                <h3>Dual Numbers</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Dual Numbers)</div>
                    <div class="env-body"><p>A <strong>dual number</strong> has the form \\(a + b\\varepsilon\\) where \\(\\varepsilon\\) is an abstract quantity satisfying \\(\\varepsilon^2 = 0\\) (but \\(\\varepsilon \\neq 0\\)). The real part \\(a\\) carries the <em>primal</em> (value) and the dual part \\(b\\) carries the <em>tangent</em> (derivative).</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Arithmetic of Dual Numbers)</div>
                    <div class="env-body"><p>For dual numbers \\(a + a'\\varepsilon\\) and \\(b + b'\\varepsilon\\):</p>
                    <ul>
                        <li>Addition: \\((a + a'\\varepsilon) + (b + b'\\varepsilon) = (a + b) + (a' + b')\\varepsilon\\)</li>
                        <li>Multiplication: \\((a + a'\\varepsilon)(b + b'\\varepsilon) = ab + (a'b + ab')\\varepsilon\\)</li>
                        <li>For any differentiable function: \\(g(a + a'\\varepsilon) = g(a) + g'(a) \\cdot a' \\cdot \\varepsilon\\)</li>
                    </ul>
                    <p>The multiplication rule recovers the product rule, and the function rule recovers the chain rule. This is because \\(\\varepsilon^2 = 0\\) kills all higher-order terms in the Taylor expansion.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof (Function Rule)</div>
                    <div class="env-body"><p>Taylor-expand \\(g\\) around \\(a\\):
                    \\[g(a + a'\\varepsilon) = g(a) + g'(a)(a'\\varepsilon) + \\frac{1}{2}g''(a)(a'\\varepsilon)^2 + \\cdots = g(a) + g'(a) \\cdot a' \\cdot \\varepsilon\\]
                    since \\(\\varepsilon^2 = 0\\) annihilates all terms of order \\(\\geq 2\\).</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <h3>Forward-Mode AD Algorithm</h3>

                <div class="env-block definition">
                    <div class="env-title">Algorithm (Forward-Mode AD)</div>
                    <div class="env-body">
                    <p>To compute \\(\\partial f / \\partial x\\) for \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\):</p>
                    <ol>
                        <li><strong>Seed</strong>: Set the tangent of the input variable \\(x\\) to 1 (\\(\\dot{x} = 1\\)) and all other input tangents to 0.</li>
                        <li><strong>Propagate</strong>: For each node \\(v_i\\) in topological order, compute both:
                            <ul>
                                <li>The primal value \\(v_i\\) (same as the forward pass).</li>
                                <li>The tangent \\(\\dot{v}_i = \\frac{\\partial v_i}{\\partial x}\\) using the chain rule for the elementary operation at node \\(v_i\\).</li>
                            </ul>
                        </li>
                        <li><strong>Read off</strong>: The tangent at the output node is \\(\\partial f / \\partial x\\).</li>
                    </ol>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Forward-Mode AD on \\(f(x,y) = (x+y) \\cdot \\sin(x)\\))</div>
                    <div class="env-body"><p>Computing \\(\\partial f / \\partial x\\) at \\(x = \\pi/4, y = 1\\). Seed: \\(\\dot{x} = 1, \\dot{y} = 0\\).</p>
                    <ol>
                        <li>\\(v_1 = x = 0.785\\), \\(\\dot{v}_1 = 1\\)</li>
                        <li>\\(v_2 = y = 1\\), \\(\\dot{v}_2 = 0\\)</li>
                        <li>\\(v_3 = v_1 + v_2 = 1.785\\), \\(\\dot{v}_3 = \\dot{v}_1 + \\dot{v}_2 = 1\\)</li>
                        <li>\\(v_4 = \\sin(v_1) = 0.707\\), \\(\\dot{v}_4 = \\cos(v_1) \\cdot \\dot{v}_1 = 0.707 \\cdot 1 = 0.707\\)</li>
                        <li>\\(v_5 = v_3 \\cdot v_4 = 1.262\\), \\(\\dot{v}_5 = \\dot{v}_3 \\cdot v_4 + v_3 \\cdot \\dot{v}_4 = 1 \\cdot 0.707 + 1.785 \\cdot 0.707 = 1.969\\)</li>
                    </ol>
                    <p>So \\(\\partial f / \\partial x \\approx 1.969\\). To get \\(\\partial f / \\partial y\\), we would need a <em>second</em> forward pass with \\(\\dot{x} = 0, \\dot{y} = 1\\).</p></div>
                </div>

                <h3>Cost Analysis</h3>

                <div class="env-block remark">
                    <div class="env-title">Remark (Forward-Mode Cost)</div>
                    <div class="env-body"><p>Each forward-mode pass computes one column of the Jacobian (the partial derivative with respect to one input). For a function \\(f: \\mathbb{R}^n \\to \\mathbb{R}^m\\), computing the full Jacobian requires \\(n\\) forward passes (one per input dimension). Forward-mode is efficient when \\(n\\) is small (few inputs, many outputs), but <strong>prohibitively expensive for deep learning</strong>, where \\(n\\) (the number of parameters) can be in the billions.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="ch02-forward-ad"></div>`,

            visualizations: [
                {
                    id: 'ch02-forward-ad',
                    title: 'Forward-Mode AD: Tangent Propagation',
                    description: 'Watch tangent values (derivatives) propagate forward through the graph alongside primal values. Toggle between computing \u2202f/\u2202x and \u2202f/\u2202y.',
                    setup: function(container, controls) {
                        const viz = new VizEngine(container, {width: 640, height: 460, scale: 1, originX: 0, originY: 0});
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;

                        const state = { x: 0.785, y: 1.0, wrt: 'x', animStep: 0, animTime: 0 };

                        VizEngine.createSlider(controls, 'x', -3.14, 3.14, 0.785, 0.01, v => { state.x = v; state.animStep = 0; state.animTime = 0; });
                        VizEngine.createSlider(controls, 'y', -3, 3, 1.0, 0.1, v => { state.y = v; state.animStep = 0; state.animTime = 0; });

                        const wrtBtn = VizEngine.createButton(controls, '\u2202f/\u2202x', () => {
                            state.wrt = state.wrt === 'x' ? 'y' : 'x';
                            wrtBtn.textContent = '\u2202f/\u2202' + state.wrt;
                            state.animStep = 0;
                            state.animTime = 0;
                        });

                        const resetBtn = VizEngine.createButton(controls, 'Replay', () => {
                            state.animStep = 0;
                            state.animTime = 0;
                        });

                        // Node layout
                        const nodeList = [
                            {key: 'x',   px: 70,  py: 120, label: 'x',    op: 'input'},
                            {key: 'y',   px: 70,  py: 310, label: 'y',    op: 'input'},
                            {key: 'add', px: 240, py: 220, label: '+',    op: 'add'},
                            {key: 'sin', px: 240, py: 100, label: 'sin',  op: 'sin'},
                            {key: 'mul', px: 430, py: 170, label: '\u00d7',    op: 'mul'},
                        ];

                        const edgeList = [
                            ['x', 'add'], ['y', 'add'], ['x', 'sin'],
                            ['add', 'mul'], ['sin', 'mul']
                        ];

                        function getNode(key) { return nodeList.find(n => n.key === key); }

                        function computeAll() {
                            const x = state.x, y = state.y;
                            const wrtX = state.wrt === 'x';
                            const vals = {}, dots = {};

                            vals.x = x;    dots.x = wrtX ? 1 : 0;
                            vals.y = y;    dots.y = wrtX ? 0 : 1;
                            vals.add = x + y;  dots.add = dots.x + dots.y;
                            vals.sin = Math.sin(x); dots.sin = Math.cos(x) * dots.x;
                            vals.mul = vals.add * vals.sin;
                            dots.mul = dots.add * vals.sin + vals.add * dots.sin;
                            return {vals, dots};
                        }

                        function drawArrow(x1, y1, x2, y2, color, lw) {
                            const dx = x2 - x1, dy = y2 - y1;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            const ux = dx / len, uy = dy / len;
                            const r = 30;
                            const sx = x1 + ux * r, sy = y1 + uy * r;
                            const ex = x2 - ux * r, ey = y2 - uy * r;
                            const angle = Math.atan2(ey - sy, ex - sx);

                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 2;
                            ctx.beginPath();
                            ctx.moveTo(sx, sy);
                            ctx.lineTo(ex, ey);
                            ctx.stroke();

                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(ex, ey);
                            ctx.lineTo(ex - 9 * Math.cos(angle - 0.4), ey - 9 * Math.sin(angle - 0.4));
                            ctx.lineTo(ex - 9 * Math.cos(angle + 0.4), ey - 9 * Math.sin(angle + 0.4));
                            ctx.closePath();
                            ctx.fill();
                        }

                        // Animation order matches topological order
                        const topoOrder = ['x', 'y', 'add', 'sin', 'mul'];

                        function draw(t) {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            state.animTime += 1;
                            if (state.animTime % 40 === 0 && state.animStep < topoOrder.length) {
                                state.animStep++;
                            }

                            const {vals, dots} = computeAll();

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Forward-Mode AD: computing \u2202f/\u2202' + state.wrt, W / 2, 8);

                            // Draw edges
                            for (const [from, to] of edgeList) {
                                const nf = getNode(from), nt = getNode(to);
                                const fIdx = topoOrder.indexOf(from);
                                const tIdx = topoOrder.indexOf(to);
                                const active = state.animStep > fIdx;
                                drawArrow(nf.px, nf.py, nt.px, nt.py, active ? viz.colors.teal + 'aa' : viz.colors.grid + '66', active ? 2.5 : 1.5);
                            }

                            // Draw nodes
                            for (let i = 0; i < nodeList.length; i++) {
                                const n = nodeList[i];
                                const active = i < state.animStep;
                                const current = i === state.animStep - 1;
                                const r = 27;

                                // Glow for current node
                                if (current) {
                                    ctx.fillStyle = viz.colors.yellow + '33';
                                    ctx.beginPath();
                                    ctx.arc(n.px, n.py, r + 12, 0, Math.PI * 2);
                                    ctx.fill();
                                }

                                // Circle
                                const isInput = n.op === 'input';
                                ctx.fillStyle = active ? (isInput ? '#1a2a3a' : '#1a2030') : '#111125';
                                ctx.strokeStyle = active ? (current ? viz.colors.yellow : viz.colors.teal) : viz.colors.grid;
                                ctx.lineWidth = current ? 3 : 2;
                                ctx.beginPath();
                                ctx.arc(n.px, n.py, r, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.stroke();

                                // Op label
                                ctx.fillStyle = active ? viz.colors.white : viz.colors.grid;
                                ctx.font = 'bold 15px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(n.label, n.px, n.py);

                                if (active) {
                                    // Primal value (above right)
                                    ctx.fillStyle = viz.colors.blue;
                                    ctx.font = '11px monospace';
                                    ctx.textAlign = 'left';
                                    ctx.textBaseline = 'bottom';
                                    ctx.fillText('v=' + vals[n.key].toFixed(3), n.px + r + 4, n.py - 2);

                                    // Tangent value (below right)
                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.font = 'bold 11px monospace';
                                    ctx.textAlign = 'left';
                                    ctx.textBaseline = 'top';
                                    const dotStr = '\u1e8f=' + dots[n.key].toFixed(3);
                                    ctx.fillText(dotStr, n.px + r + 4, n.py + 4);
                                }
                            }

                            // Derivative rules on the right
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            const rulesX = 520;
                            let rulesY = 50;
                            ctx.fillStyle = viz.colors.purple;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillText('Chain rules:', rulesX, rulesY);
                            rulesY += 22;

                            const rules = [
                                {text: '+:  \u1e8f\u2083 = \u1e8f\u2081 + \u1e8f\u2082', step: 2},
                                {text: 'sin: \u1e8f\u2084 = cos(v\u2081)\u00b7\u1e8f\u2081', step: 3},
                                {text: '\u00d7:  \u1e8f\u2085 = \u1e8f\u2083\u00b7v\u2084 + v\u2083\u00b7\u1e8f\u2084', step: 4},
                            ];

                            ctx.font = '11px monospace';
                            for (const rule of rules) {
                                ctx.fillStyle = state.animStep >= rule.step ? viz.colors.teal : viz.colors.grid;
                                ctx.fillText(rule.text, rulesX, rulesY);
                                rulesY += 20;
                            }

                            // Final result
                            if (state.animStep >= topoOrder.length) {
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = 'bold 13px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('\u2202f/\u2202' + state.wrt + ' = ' + dots.mul.toFixed(4), W / 2, H - 20);
                            }

                            // Legend
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            const lx = 15, ly = H - 55;
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(lx, ly, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Primal (value)', lx + 14, ly);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(lx, ly + 16, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Tangent (derivative)', lx + 14, ly + 16);
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Using dual numbers, evaluate \\(f(x) = x^2 + 3x\\) at \\(x = 2\\) and simultaneously compute \\(f\'(2)\\). Substitute the dual number \\(2 + \\varepsilon\\) into the expression.',
                    hint: '\\((2 + \\varepsilon)^2 = 4 + 4\\varepsilon + \\varepsilon^2 = 4 + 4\\varepsilon\\) since \\(\\varepsilon^2 = 0\\).',
                    solution: '\\(f(2 + \\varepsilon) = (2 + \\varepsilon)^2 + 3(2 + \\varepsilon) = (4 + 4\\varepsilon) + (6 + 3\\varepsilon) = 10 + 7\\varepsilon\\). The primal part gives \\(f(2) = 10\\) and the dual part gives \\(f\'(2) = 7\\). Verification: \\(f\'(x) = 2x + 3\\), so \\(f\'(2) = 7\\). Correct.'
                },
                {
                    question: 'For \\(f(x, y) = (x + y) \\cdot \\sin(x)\\), use forward-mode AD to compute \\(\\partial f / \\partial y\\) at \\((x, y) = (\\pi/4, 1)\\). Show all intermediate tangent values.',
                    hint: 'Seed with \\(\\dot{x} = 0, \\dot{y} = 1\\).',
                    solution: 'Seed: \\(\\dot{x} = 0, \\dot{y} = 1\\). (1) \\(v_1 = \\pi/4\\), \\(\\dot{v}_1 = 0\\). (2) \\(v_2 = 1\\), \\(\\dot{v}_2 = 1\\). (3) \\(v_3 = v_1 + v_2 = 1.785\\), \\(\\dot{v}_3 = 0 + 1 = 1\\). (4) \\(v_4 = \\sin(\\pi/4) = 0.707\\), \\(\\dot{v}_4 = \\cos(\\pi/4) \\cdot 0 = 0\\). (5) \\(v_5 = v_3 v_4 = 1.262\\), \\(\\dot{v}_5 = \\dot{v}_3 v_4 + v_3 \\dot{v}_4 = 1 \\cdot 0.707 + 1.785 \\cdot 0 = 0.707\\). So \\(\\partial f / \\partial y = 0.707 = \\sin(\\pi/4)\\), which makes sense since \\(f = (x+y)\\sin(x)\\) is linear in \\(y\\) with coefficient \\(\\sin(x)\\).'
                },
                {
                    question: 'A neural network has \\(n = 10^8\\) parameters and outputs a scalar loss \\(L\\). How many forward-mode AD passes are needed to compute the full gradient \\(\\nabla L\\)? Compare with the cost of one forward pass.',
                    hint: 'Forward-mode computes one partial derivative per pass.',
                    solution: 'Forward-mode computes \\(\\partial L / \\partial \\theta_i\\) for one \\(i\\) per pass (one column of the Jacobian). The full gradient has \\(10^8\\) components, so we need \\(10^8\\) forward passes. Each forward-mode pass costs roughly 2-3x the cost of a regular forward pass (tracking both primal and tangent values). Total cost: \\(\\sim 2 \\times 10^8\\) forward passes. This is wildly impractical. Reverse-mode AD (next section) computes the entire gradient in a single backward pass.'
                }
            ]
        },

        // ===================== Section 5: Reverse-Mode AD =====================
        {
            id: 'ch02-sec05',
            title: 'Reverse-Mode AD',
            content: `<h2>Reverse-Mode Automatic Differentiation</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Miracle of Backpropagation</div>
                    <div class="env-body"><p>Reverse-mode AD solves the key computational bottleneck: computing the gradient of a scalar-valued function with respect to <em>all</em> of its inputs in a single backward pass. Where forward-mode propagates tangents (\\(\\dot{v}_i = \\partial v_i / \\partial x_j\\)) from inputs to output, reverse-mode propagates <em>adjoints</em> (\\(\\bar{v}_i = \\partial f / \\partial v_i\\)) from the output back to the inputs. For a function \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\), forward-mode costs \\(O(n)\\) passes; reverse-mode costs \\(O(1)\\) passes. This is why backpropagation (the application of reverse-mode AD to neural networks) is the engine of deep learning.</p></div>
                </div>

                <h3>Adjoints and the Reverse Pass</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Adjoint)</div>
                    <div class="env-body"><p>For a computation graph computing \\(f\\), the <strong>adjoint</strong> of an intermediate variable \\(v_i\\) is
                    \\[\\bar{v}_i = \\frac{\\partial f}{\\partial v_i}\\]
                    It answers the question: "How much does the final output \\(f\\) change when \\(v_i\\) changes by a small amount?" The adjoint at the output node is always \\(\\bar{v}_{\\text{out}} = 1\\).</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Algorithm (Reverse-Mode AD)</div>
                    <div class="env-body">
                    <ol>
                        <li><strong>Forward pass</strong>: Evaluate all nodes in topological order, storing all intermediate values.</li>
                        <li><strong>Seed</strong>: Set \\(\\bar{v}_{\\text{out}} = 1\\).</li>
                        <li><strong>Backward pass</strong>: Process nodes in <em>reverse</em> topological order. For each node \\(v_i\\) with children \\(v_{c_1}, v_{c_2}, \\ldots\\):
                            <ul>
                                <li>For each child \\(v_{c_k}\\) that uses \\(v_i\\) as input, compute the <em>local partial derivative</em> \\(\\partial v_{c_k} / \\partial v_i\\).</li>
                                <li>Accumulate: \\(\\bar{v}_i = \\sum_k \\bar{v}_{c_k} \\cdot \\frac{\\partial v_{c_k}}{\\partial v_i}\\)</li>
                            </ul>
                        </li>
                        <li><strong>Read off</strong>: The adjoints at the input nodes give \\(\\partial f / \\partial x_j = \\bar{x}_j\\).</li>
                    </ol>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Reverse-Mode AD on \\(f(x,y) = (x+y) \\cdot \\sin(x)\\))</div>
                    <div class="env-body"><p>At \\(x = \\pi/4, y = 1\\).</p>
                    <p><strong>Forward pass</strong> (store values):</p>
                    <ul>
                        <li>\\(v_1 = x = 0.785, \\; v_2 = y = 1, \\; v_3 = v_1 + v_2 = 1.785\\)</li>
                        <li>\\(v_4 = \\sin(v_1) = 0.707, \\; v_5 = v_3 \\cdot v_4 = 1.262\\)</li>
                    </ul>
                    <p><strong>Backward pass</strong>:</p>
                    <ol>
                        <li>\\(\\bar{v}_5 = 1\\) (seed)</li>
                        <li>\\(v_5 = v_3 \\cdot v_4\\): \\(\\bar{v}_3 = \\bar{v}_5 \\cdot v_4 = 1 \\cdot 0.707 = 0.707\\) and \\(\\bar{v}_4 = \\bar{v}_5 \\cdot v_3 = 1 \\cdot 1.785 = 1.785\\)</li>
                        <li>\\(v_4 = \\sin(v_1)\\): \\(\\bar{v}_1 \\mathrel{+}= \\bar{v}_4 \\cdot \\cos(v_1) = 1.785 \\cdot 0.707 = 1.262\\)</li>
                        <li>\\(v_3 = v_1 + v_2\\): \\(\\bar{v}_1 \\mathrel{+}= \\bar{v}_3 \\cdot 1 = 0.707\\) and \\(\\bar{v}_2 = \\bar{v}_3 \\cdot 1 = 0.707\\)</li>
                    </ol>
                    <p>Final adjoints: \\(\\bar{x} = \\bar{v}_1 = 1.262 + 0.707 = 1.969\\), \\(\\bar{y} = \\bar{v}_2 = 0.707\\).</p>
                    <p>Both partial derivatives computed in a <strong>single backward pass</strong>.</p></div>
                </div>

                <h3>The Chain Rule in Matrix Form</h3>

                <p>Reverse-mode AD applies the chain rule from output to inputs. For a function \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\) decomposed as \\(f = f_L \\circ f_{L-1} \\circ \\cdots \\circ f_1\\), the gradient is:</p>
                <p>\\[\\nabla f = J_1^T J_2^T \\cdots J_L^T \\cdot 1\\]</p>
                <p>where \\(J_k\\) is the Jacobian of \\(f_k\\). Reverse-mode evaluates this product right-to-left (starting from the scalar 1), which is efficient because every intermediate result is a vector (not a matrix). Forward-mode evaluates left-to-right, starting from a seed vector.</p>

                <h3>Cost Analysis</h3>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Cheap Gradient Principle)</div>
                    <div class="env-body"><p>For a function \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\) composed of \\(T\\) elementary operations, the cost of computing \\(f\\) and its full gradient \\(\\nabla f \\in \\mathbb{R}^n\\) via reverse-mode AD is at most \\(5T\\) operations (approximately 2-3 times the cost of evaluating \\(f\\) alone), <strong>regardless of \\(n\\)</strong>.</p></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Memory Trade-off)</div>
                    <div class="env-body"><p>The cheap gradient comes with a cost: memory. The forward pass must store all intermediate values (needed for computing local partial derivatives in the backward pass). For a deep network with \\(L\\) layers and \\(d\\)-dimensional activations, this requires \\(O(Ld)\\) memory. Techniques like <strong>gradient checkpointing</strong> (recomputing some forward activations during the backward pass) trade computation for memory, reducing memory to \\(O(\\sqrt{L} \\cdot d)\\) at the cost of one extra forward pass.</p></div>
                </div>

                <h3>Connection to Backpropagation</h3>

                <p>Backpropagation in neural networks is simply reverse-mode AD applied to the computation graph of a neural network. The "backward pass" in PyTorch (<code>loss.backward()</code>) constructs the reversed graph and propagates adjoints. Every <code>torch.autograd.Function</code> defines both a <code>forward()</code> and a <code>backward()</code> method, corresponding to the primal evaluation and the adjoint propagation rule for that operation.</p>

                <div class="env-block warning">
                    <div class="env-title">Forward-Mode vs. Reverse-Mode: When to Use Which</div>
                    <div class="env-body">
                    <ul>
                        <li><strong>Reverse-mode</strong>: Optimal for \\(f: \\mathbb{R}^n \\to \\mathbb{R}\\) (many inputs, one output). This is the typical deep learning scenario (scalar loss, millions of parameters).</li>
                        <li><strong>Forward-mode</strong>: Optimal for \\(f: \\mathbb{R} \\to \\mathbb{R}^m\\) (one input, many outputs). Useful for computing Jacobian-vector products, directional derivatives, and in physics simulations.</li>
                        <li>For \\(f: \\mathbb{R}^n \\to \\mathbb{R}^m\\) in general, the full Jacobian costs \\(\\min(n, m)\\) passes.</li>
                    </ul>
                    </div>
                </div>

                <div class="viz-placeholder" data-viz="ch02-reverse-ad"></div>`,

            visualizations: [
                {
                    id: 'ch02-reverse-ad',
                    title: 'Reverse-Mode AD: Forward Pass then Backward Pass',
                    description: 'Watch the two-phase process: forward pass computes values (left to right), then the backward pass propagates adjoints (right to left). Adjoints appear in orange.',
                    setup: function(container, controls) {
                        const viz = new VizEngine(container, {width: 640, height: 480, scale: 1, originX: 0, originY: 0});
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;

                        const state = { x: 0.785, y: 1.0, phase: 'forward', animStep: 0, animTime: 0 };

                        VizEngine.createSlider(controls, 'x', -3.14, 3.14, 0.785, 0.01, v => { state.x = v; state.phase = 'forward'; state.animStep = 0; state.animTime = 0; });
                        VizEngine.createSlider(controls, 'y', -3, 3, 1.0, 0.1, v => { state.y = v; state.phase = 'forward'; state.animStep = 0; state.animTime = 0; });
                        VizEngine.createButton(controls, 'Replay', () => { state.phase = 'forward'; state.animStep = 0; state.animTime = 0; });

                        const nodeList = [
                            {key: 'x',   px: 70,  py: 130, label: 'x',    op: 'input'},
                            {key: 'y',   px: 70,  py: 310, label: 'y',    op: 'input'},
                            {key: 'add', px: 230, py: 230, label: '+',    op: 'add'},
                            {key: 'sin', px: 230, py: 110, label: 'sin',  op: 'sin'},
                            {key: 'mul', px: 420, py: 175, label: '\u00d7',    op: 'mul'},
                        ];

                        const edgeList = [
                            ['x', 'add'], ['y', 'add'], ['x', 'sin'],
                            ['add', 'mul'], ['sin', 'mul']
                        ];

                        const forwardOrder = ['x', 'y', 'add', 'sin', 'mul'];
                        const backwardOrder = ['mul', 'sin', 'add', 'y', 'x'];

                        function getNode(key) { return nodeList.find(n => n.key === key); }

                        function computeAll() {
                            const x = state.x, y = state.y;
                            const vals = {};
                            vals.x = x; vals.y = y;
                            vals.add = x + y;
                            vals.sin = Math.sin(x);
                            vals.mul = vals.add * vals.sin;

                            // Adjoints (backward)
                            const adj = {x: 0, y: 0, add: 0, sin: 0, mul: 0};
                            adj.mul = 1;
                            // mul = add * sin
                            adj.add = adj.mul * vals.sin;
                            adj.sin = adj.mul * vals.add;
                            // sin = sin(x) -> adj to x from sin
                            const adjXFromSin = adj.sin * Math.cos(x);
                            // add = x + y
                            const adjXFromAdd = adj.add * 1;
                            const adjYFromAdd = adj.add * 1;
                            adj.x = adjXFromSin + adjXFromAdd;
                            adj.y = adjYFromAdd;

                            return {vals, adj};
                        }

                        function drawArrow(x1, y1, x2, y2, color, lw, dashed) {
                            const dx = x2 - x1, dy = y2 - y1;
                            const len = Math.sqrt(dx * dx + dy * dy);
                            const ux = dx / len, uy = dy / len;
                            const r = 30;
                            const sx = x1 + ux * r, sy = y1 + uy * r;
                            const ex = x2 - ux * r, ey = y2 - uy * r;
                            const angle = Math.atan2(ey - sy, ex - sx);

                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw || 2;
                            if (dashed) ctx.setLineDash([5, 3]);
                            ctx.beginPath();
                            ctx.moveTo(sx, sy);
                            ctx.lineTo(ex, ey);
                            ctx.stroke();
                            if (dashed) ctx.setLineDash([]);

                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(ex, ey);
                            ctx.lineTo(ex - 9 * Math.cos(angle - 0.4), ey - 9 * Math.sin(angle - 0.4));
                            ctx.lineTo(ex - 9 * Math.cos(angle + 0.4), ey - 9 * Math.sin(angle + 0.4));
                            ctx.closePath();
                            ctx.fill();
                        }

                        function draw(t) {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            state.animTime += 1;
                            const stepInterval = 35;

                            if (state.phase === 'forward') {
                                if (state.animTime % stepInterval === 0 && state.animStep < forwardOrder.length) {
                                    state.animStep++;
                                }
                                if (state.animStep >= forwardOrder.length) {
                                    // Wait a bit, then switch to backward
                                    if (state.animTime % stepInterval === 0) {
                                        state.phase = 'backward';
                                        state.animStep = 0;
                                        state.animTime = 0;
                                    }
                                }
                            } else {
                                if (state.animTime % stepInterval === 0 && state.animStep < backwardOrder.length) {
                                    state.animStep++;
                                }
                            }

                            const {vals, adj} = computeAll();

                            // Phase indicator
                            ctx.fillStyle = state.phase === 'forward' ? viz.colors.blue : viz.colors.orange;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            const phaseLabel = state.phase === 'forward' ? 'FORWARD PASS (computing values)' : 'BACKWARD PASS (propagating adjoints)';
                            ctx.fillText(phaseLabel, W / 2, 8);

                            // Determine which nodes are "active" in current phase
                            const forwardActive = new Set();
                            const backwardActive = new Set();

                            if (state.phase === 'forward') {
                                for (let i = 0; i < state.animStep; i++) forwardActive.add(forwardOrder[i]);
                            } else {
                                forwardOrder.forEach(k => forwardActive.add(k));
                                for (let i = 0; i < state.animStep; i++) backwardActive.add(backwardOrder[i]);
                            }

                            // Draw forward edges
                            for (const [from, to] of edgeList) {
                                const nf = getNode(from), nt = getNode(to);
                                const active = forwardActive.has(from);
                                drawArrow(nf.px, nf.py, nt.px, nt.py, active ? viz.colors.blue + '77' : viz.colors.grid + '44', 1.5);
                            }

                            // Draw backward edges (reversed, dashed, offset slightly)
                            if (state.phase === 'backward') {
                                for (const [from, to] of edgeList) {
                                    const nf = getNode(from), nt = getNode(to);
                                    if (backwardActive.has(to)) {
                                        // Draw reversed arrow, offset
                                        drawArrow(nt.px, nt.py + 8, nf.px, nf.py + 8, viz.colors.orange + 'cc', 2, true);
                                    }
                                }
                            }

                            // Draw nodes
                            for (let i = 0; i < nodeList.length; i++) {
                                const n = nodeList[i];
                                const fActive = forwardActive.has(n.key);
                                const bActive = backwardActive.has(n.key);
                                const bCurrent = state.phase === 'backward' && state.animStep > 0 && backwardOrder[state.animStep - 1] === n.key;
                                const fCurrent = state.phase === 'forward' && state.animStep > 0 && forwardOrder[state.animStep - 1] === n.key;
                                const r = 27;

                                // Glow for current
                                if (fCurrent || bCurrent) {
                                    ctx.fillStyle = (fCurrent ? viz.colors.blue : viz.colors.orange) + '33';
                                    ctx.beginPath();
                                    ctx.arc(n.px, n.py, r + 12, 0, Math.PI * 2);
                                    ctx.fill();
                                }

                                // Circle
                                ctx.fillStyle = fActive ? '#141428' : '#0e0e1e';
                                ctx.strokeStyle = bCurrent ? viz.colors.orange : (fCurrent ? viz.colors.blue : (fActive ? viz.colors.teal : viz.colors.grid));
                                ctx.lineWidth = (fCurrent || bCurrent) ? 3 : 2;
                                ctx.beginPath();
                                ctx.arc(n.px, n.py, r, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.stroke();

                                // Op label
                                ctx.fillStyle = fActive ? viz.colors.white : viz.colors.grid;
                                ctx.font = 'bold 15px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(n.label, n.px, n.py);

                                // Value (shown once forward-activated)
                                if (fActive) {
                                    ctx.fillStyle = viz.colors.blue;
                                    ctx.font = '10px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'top';
                                    ctx.fillText('v=' + vals[n.key].toFixed(3), n.px, n.py + r + 4);
                                }

                                // Adjoint (shown once backward-activated)
                                if (bActive) {
                                    ctx.fillStyle = viz.colors.orange;
                                    ctx.font = 'bold 11px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'bottom';
                                    ctx.fillText('\u0305v=' + adj[n.key].toFixed(3), n.px, n.py - r - 4);
                                }
                            }

                            // Show rules on the right side
                            const rulesX = 510;
                            let rulesY = 55;
                            ctx.fillStyle = viz.colors.purple;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Adjoint rules:', rulesX, rulesY);
                            rulesY += 22;

                            const bRules = [
                                {text: '\u0305v\u2085 = 1 (seed)', step: 0, key: 'mul'},
                                {text: '\u0305v\u2083 = \u0305v\u2085 \u00b7 v\u2084', step: 0, key: 'mul'},
                                {text: '\u0305v\u2084 = \u0305v\u2085 \u00b7 v\u2083', step: 0, key: 'mul'},
                                {text: '\u0305v\u2081 += \u0305v\u2084\u00b7cos(v\u2081)', step: 1, key: 'sin'},
                                {text: '\u0305v\u2081 += \u0305v\u2083\u00b71', step: 2, key: 'add'},
                                {text: '\u0305v\u2082 = \u0305v\u2083\u00b71', step: 2, key: 'add'},
                            ];

                            ctx.font = '10px monospace';
                            for (const rule of bRules) {
                                const rActive = state.phase === 'backward' && state.animStep > rule.step;
                                ctx.fillStyle = rActive ? viz.colors.teal : viz.colors.grid;
                                ctx.fillText(rule.text, rulesX, rulesY);
                                rulesY += 17;
                            }

                            // Final result
                            if (state.phase === 'backward' && state.animStep >= backwardOrder.length) {
                                ctx.fillStyle = viz.colors.green;
                                ctx.font = 'bold 13px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText('\u2202f/\u2202x = ' + adj.x.toFixed(4) + '    \u2202f/\u2202y = ' + adj.y.toFixed(4) + '    (BOTH from one backward pass!)', W / 2, H - 18);
                            }

                            // Legend
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            const lx = 10, ly = H - 50;
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(lx, ly, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Forward (value)', lx + 14, ly);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(lx, ly + 15, 10, 10);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('Backward (adjoint)', lx + 14, ly + 15);
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Apply reverse-mode AD to compute the gradient of \\(f(x, y, z) = (x + y) \\cdot z\\) at \\((x, y, z) = (1, 2, 3)\\). Show both the forward pass and the backward pass with all adjoints.',
                    hint: 'Introduce \\(v_3 = x + y\\) and \\(v_4 = v_3 \\cdot z\\). The backward pass starts with \\(\\bar{v}_4 = 1\\).',
                    solution: '<strong>Forward pass</strong>: \\(v_1 = x = 1\\), \\(v_2 = y = 2\\), \\(v_3 = z = 3\\), \\(v_4 = v_1 + v_2 = 3\\), \\(v_5 = v_4 \\cdot v_3 = 9\\). <strong>Backward pass</strong>: \\(\\bar{v}_5 = 1\\). \\(v_5 = v_4 \\cdot v_3\\): \\(\\bar{v}_4 = \\bar{v}_5 \\cdot v_3 = 3\\), \\(\\bar{v}_3 = \\bar{v}_5 \\cdot v_4 = 3\\). \\(v_4 = v_1 + v_2\\): \\(\\bar{v}_1 = \\bar{v}_4 \\cdot 1 = 3\\), \\(\\bar{v}_2 = \\bar{v}_4 \\cdot 1 = 3\\). Result: \\(\\nabla f = (\\bar{x}, \\bar{y}, \\bar{z}) = (3, 3, 3)\\). Verification: \\(\\partial f/\\partial x = z = 3\\), \\(\\partial f/\\partial y = z = 3\\), \\(\\partial f/\\partial z = x + y = 3\\). Correct.'
                },
                {
                    question: 'Explain the "cheap gradient principle": why does reverse-mode AD compute the gradient in \\(O(1)\\) backward passes regardless of the number of parameters \\(n\\)? Where does the dependence on \\(n\\) go?',
                    hint: 'Think about what happens at each node in the backward pass. How many adjoints does each node compute?',
                    solution: 'In the backward pass, each node processes its adjoint once (regardless of \\(n\\)) and distributes it to its parents via local partial derivatives. Each edge in the graph is traversed exactly once backward. Since the graph has \\(O(T)\\) edges (where \\(T\\) is the number of operations), the backward pass costs \\(O(T)\\), same as the forward pass. The dependence on \\(n\\) is hidden in \\(T\\): a network with \\(n\\) parameters has \\(T \\geq n\\) operations (each parameter is used at least once). But the key insight is that the cost scales with the <em>computation graph size</em>, not with \\(n\\) separately. A single backward pass yields all \\(n\\) partial derivatives.'
                },
                {
                    question: 'A neural network with \\(L = 100\\) layers and hidden dimension \\(d = 512\\) uses \\(O(Ld^2)\\) parameters. Estimate the memory required to store all intermediate activations during the forward pass (needed for the backward pass). How does gradient checkpointing reduce this?',
                    hint: 'Each layer produces an activation of dimension \\(d\\). There are \\(L\\) such activations.',
                    solution: 'Each layer produces an activation vector of dimension \\(d = 512\\) (for a batch of size \\(B\\), the activation is \\(B \\times d\\)). Storing all \\(L = 100\\) activations requires \\(O(L \\cdot B \\cdot d)\\) memory. With \\(B = 64\\) and float32 (4 bytes): \\(100 \\times 64 \\times 512 \\times 4 = 13.1\\) MB for activations alone. For larger models (\\(d = 4096, L = 96\\)), this becomes multiple GB. Gradient checkpointing divides the network into \\(\\sqrt{L} \\approx 10\\) segments, storing only the activations at segment boundaries. During the backward pass, activations within each segment are recomputed. Memory: \\(O(\\sqrt{L} \\cdot B \\cdot d)\\), roughly \\(10\\times\\) reduction, at the cost of \\(\\sim 33\\%\\) extra compute (one additional partial forward pass per segment).'
                }
            ]
        }
    ]
});
