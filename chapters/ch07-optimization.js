window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch07',
    number: 7,
    title: 'Optimization Algorithms',
    subtitle: '从SGD到Adam：深度学习优化方法全景',
    sections: [

        // ===== Section 1: Stochastic Gradient Descent =====
        {
            id: 'ch07-sec01',
            title: 'Stochastic Gradient Descent',
            content: `
<div class="env-block intuition"><div class="env-title">Why Optimization Is the Engine of Deep Learning</div><div class="env-body"><p>Training a neural network means finding parameters \\(\\theta\\) that minimize a loss function \\(\\mathcal{L}(\\theta)\\). The loss landscape in deep learning is a high-dimensional, non-convex surface with saddle points, plateaus, and narrow valleys. Gradient descent and its variants are the workhorses that navigate this landscape. The choice of optimizer, and how it is configured, can mean the difference between a model that converges in hours and one that never converges at all.</p></div></div>

<h2>Full-Batch Gradient Descent</h2>

<p>The simplest optimization strategy computes the gradient using the <em>entire</em> training set at every step:</p>

<div class="env-block definition"><div class="env-title">Full-Batch Gradient Descent</div><div class="env-body"><p>Given a dataset \\(\\{(x_i, y_i)\\}_{i=1}^{N}\\) and loss function \\(\\ell\\), the full-batch update rule is:</p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t), \\qquad \\mathcal{L}(\\theta) = \\frac{1}{N}\\sum_{i=1}^{N} \\ell(f_\\theta(x_i),\\, y_i) \\]</p>
<p>where \\(\\eta &gt; 0\\) is the <strong>learning rate</strong> (step size).</p></div></div>

<p>Full-batch GD computes the exact gradient, so each step points in the true steepest-descent direction. However, for large datasets (millions of samples), computing the full gradient at every iteration is prohibitively expensive.</p>

<h2>Stochastic Gradient Descent (SGD)</h2>

<div class="env-block definition"><div class="env-title">Stochastic Gradient Descent</div><div class="env-body"><p>SGD approximates the full gradient using a <strong>single randomly sampled</strong> data point \\((x_i, y_i)\\):</p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\ell(f_\\theta(x_i),\\, y_i) \\]</p>
<p>The stochastic gradient \\(g_t = \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)\\) is an <strong>unbiased estimator</strong> of the true gradient: \\(\\mathbb{E}[g_t] = \\nabla_\\theta \\mathcal{L}(\\theta_t)\\).</p></div></div>

<h2>Mini-Batch SGD</h2>

<p>In practice, we use a compromise: sample a <strong>mini-batch</strong> \\(\\mathcal{B} \\subset \\{1, \\ldots, N\\}\\) of size \\(B\\):</p>

<div class="env-block definition"><div class="env-title">Mini-Batch SGD</div><div class="env-body"><p>\\[ \\theta_{t+1} = \\theta_t - \\eta \\cdot \\frac{1}{B}\\sum_{i \\in \\mathcal{B}_t} \\nabla_\\theta \\ell(f_\\theta(x_i),\\, y_i) \\]</p>
<p>Typical batch sizes range from 32 to 512. The variance of the gradient estimator scales as \\(\\sigma^2 / B\\), so larger batches give smoother updates at the cost of more computation per step.</p></div></div>

<div class="env-block remark"><div class="env-title">The Noise Is a Feature, Not a Bug</div><div class="env-body"><p>The stochasticity in SGD is often <em>beneficial</em>. Gradient noise helps the optimizer escape sharp local minima and saddle points, biasing convergence toward <strong>flat minima</strong> that tend to generalize better (Keskar et al., 2017). This is one reason why small-batch training often outperforms large-batch training in generalization, even when both reach similar training loss.</p></div></div>

<div class="env-block theorem"><div class="env-title">SGD Convergence (Convex Case)</div><div class="env-body"><p>For an \\(L\\)-smooth convex function with bounded gradient variance \\(\\sigma^2\\), SGD with learning rate \\(\\eta_t = \\eta_0 / \\sqrt{t}\\) achieves:</p>
<p>\\[ \\mathbb{E}[\\mathcal{L}(\\bar{\\theta}_T)] - \\mathcal{L}(\\theta^*) = O\\!\\left(\\frac{1}{\\sqrt{T}}\\right) \\]</p>
<p>where \\(\\bar{\\theta}_T\\) is the averaged iterate. For non-convex objectives (the typical deep learning setting), the convergence guarantee weakens to finding an \\(\\epsilon\\)-approximate stationary point (\\(\\|\\nabla \\mathcal{L}\\| \\leq \\epsilon\\)) in \\(O(1/\\epsilon^4)\\) steps.</p></div></div>

<div class="env-block warning"><div class="env-title">Learning Rate Sensitivity</div><div class="env-body"><p>SGD is notoriously sensitive to the learning rate. Too large: the iterates diverge. Too small: convergence is painfully slow. The "Goldilocks zone" depends on the loss landscape curvature and can vary by orders of magnitude across problems. This motivates adaptive methods (Sections 3 and 4).</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz01"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz01',
                    title: 'SGD on a 2D Loss Surface',
                    description: 'Adjust the learning rate and batch size to see how SGD navigates a quadratic loss surface. Larger learning rates take bigger steps but may overshoot. Smaller batch sizes introduce more noise.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 60, originX: 300, originY: 220, width: 600, height: 440 });
                        var ctx = viz.ctx;
                        var lr = 0.05;
                        var batchNoise = 0.5;
                        var running = false;
                        var path = [];
                        var pos = [3.5, 3.0];
                        var stepCount = 0;

                        // Loss: f(x,y) = 3x^2 + 0.5y^2 (elongated bowl)
                        function lossVal(x, y) { return 3 * x * x + 0.5 * y * y; }
                        function gradient(x, y) { return [6 * x, 1.0 * y]; }

                        function reset() {
                            pos = [3.5, 3.0];
                            path = [pos.slice()];
                            stepCount = 0;
                            running = false;
                        }
                        reset();

                        function drawContours() {
                            var levels = [0.5, 2, 5, 10, 20, 35, 55, 80];
                            for (var li = 0; li < levels.length; li++) {
                                var lev = levels[li];
                                ctx.strokeStyle = 'rgba(88,166,255,' + (0.15 + 0.08 * li) + ')';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                for (var a = 0; a <= 360; a += 2) {
                                    var rad = a * Math.PI / 180;
                                    var rx = Math.sqrt(lev / 3);
                                    var ry = Math.sqrt(lev / 0.5);
                                    var px = rx * Math.cos(rad);
                                    var py = ry * Math.sin(rad);
                                    var sp = viz.toScreen(px, py);
                                    a === 0 ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                                }
                                ctx.closePath();
                                ctx.stroke();
                            }
                        }

                        function step() {
                            var g = gradient(pos[0], pos[1]);
                            var noise0 = (Math.random() - 0.5) * 2 * batchNoise * Math.abs(g[0] + 0.1);
                            var noise1 = (Math.random() - 0.5) * 2 * batchNoise * Math.abs(g[1] + 0.1);
                            pos[0] -= lr * (g[0] + noise0);
                            pos[1] -= lr * (g[1] + noise1);
                            path.push(pos.slice());
                            stepCount++;
                        }

                        function draw() {
                            viz.clear();
                            drawContours();

                            // Draw minimum marker
                            viz.drawPoint(0, 0, viz.colors.green, '  min', 5);

                            // Draw path
                            if (path.length > 1) {
                                ctx.strokeStyle = viz.colors.orange;
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                for (var i = 0; i < path.length; i++) {
                                    var sp = viz.toScreen(path[i][0], path[i][1]);
                                    i === 0 ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                                }
                                ctx.stroke();
                                // Draw dots at each step
                                for (var j = 0; j < path.length; j++) {
                                    var col = j === path.length - 1 ? viz.colors.red : viz.colors.orange;
                                    viz.drawPoint(path[j][0], path[j][1], col, '', j === path.length - 1 ? 5 : 2.5);
                                }
                            }

                            // Info
                            viz.screenText('f(x,y) = 3x\u00B2 + 0.5y\u00B2', viz.width / 2, 18, viz.colors.blue, 13);
                            viz.screenText('Step: ' + stepCount + '  |  Loss: ' + lossVal(pos[0], pos[1]).toFixed(3), viz.width / 2, 36, viz.colors.white, 12);
                            viz.screenText('\u03B8 = (' + pos[0].toFixed(2) + ', ' + pos[1].toFixed(2) + ')', viz.width / 2, 52, viz.colors.text, 11);
                        }

                        var animId = null;
                        function animate() {
                            if (!running) return;
                            step();
                            draw();
                            if (stepCount > 500 || lossVal(pos[0], pos[1]) < 0.001) { running = false; }
                            animId = requestAnimationFrame(animate);
                        }

                        draw();

                        VizEngine.createSlider(controls, '\u03B7 (lr)', 0.001, 0.2, lr, 0.001, function(v) { lr = v; });
                        VizEngine.createSlider(controls, 'Noise', 0, 2, batchNoise, 0.1, function(v) { batchNoise = v; });
                        VizEngine.createButton(controls, 'Run SGD', function() { if (!running) { running = true; animate(); } });
                        VizEngine.createButton(controls, 'Step', function() { running = false; step(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; if (animId) cancelAnimationFrame(animId); reset(); draw(); });

                        return { stopAnimation: function() { running = false; if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex01',
                    type: 'conceptual',
                    question: 'Explain why the stochastic gradient \\(g_t = \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)\\) is an unbiased estimator of the full gradient. Under what condition does this hold?',
                    hint: 'Think about the expectation over the random choice of sample \\(i\\).',
                    solution: 'If \\(i\\) is drawn uniformly at random from \\(\\{1, \\ldots, N\\}\\), then \\(\\mathbb{E}_i[\\nabla_\\theta \\ell(f_\\theta(x_i), y_i)] = \\frac{1}{N}\\sum_{i=1}^N \\nabla_\\theta \\ell(f_\\theta(x_i), y_i) = \\nabla_\\theta \\mathcal{L}(\\theta)\\). This holds because the loss \\(\\mathcal{L}\\) is defined as the arithmetic mean over all samples. The key condition is uniform random sampling (or more generally, that the sampling distribution matches the averaging weights in \\(\\mathcal{L}\\)).'
                },
                {
                    id: 'ch07-ex02',
                    type: 'numeric',
                    question: 'Consider \\(f(x) = x^2\\) with \\(x_0 = 4\\). Perform 3 steps of gradient descent with \\(\\eta = 0.1\\). What is \\(x_3\\)?',
                    hint: '\\(f\'(x) = 2x\\). Update rule: \\(x_{t+1} = x_t - \\eta \\cdot 2x_t = x_t(1 - 2\\eta)\\).',
                    solution: 'With \\(\\eta = 0.1\\): \\(x_{t+1} = x_t(1 - 0.2) = 0.8 x_t\\). So \\(x_1 = 3.2\\), \\(x_2 = 2.56\\), \\(x_3 = 2.048\\). The loss drops from 16 to 4.194.'
                },
                {
                    id: 'ch07-ex03',
                    type: 'conceptual',
                    question: 'Why does increasing the mini-batch size \\(B\\) reduce gradient variance but give diminishing returns in wall-clock training speed?',
                    hint: 'Variance scales as \\(\\sigma^2/B\\). Consider the compute cost per step and the number of steps needed.',
                    solution: 'The variance of the mini-batch gradient estimate scales as \\(\\sigma^2/B\\), so doubling \\(B\\) halves the variance. However, each step now costs twice as much compute. The convergence rate (in number of steps) improves only as \\(O(1/\\sqrt{B})\\) in the stochastic regime, so there is a point of diminishing returns: doubling \\(B\\) halves variance but does not halve the number of required steps. Beyond a critical batch size \\(B_{\\text{crit}} \\approx \\sigma^2 / \\|\\nabla \\mathcal{L}\\|^2\\) (McCandlish et al., 2018), larger batches yield almost no speedup because the noise is already negligible compared to the gradient signal.'
                }
            ]
        },

        // ===== Section 2: Momentum Methods =====
        {
            id: 'ch07-sec02',
            title: 'Momentum Methods',
            content: `
<div class="env-block intuition"><div class="env-title">The Ball Rolling Down a Hill</div><div class="env-body"><p>Plain SGD treats each gradient as an independent instruction: "move this way." It has no memory. Imagine pushing a ball on a surface where one direction is steep and the other is a gentle slope. SGD oscillates wildly in the steep direction while creeping along the gentle one. <strong>Momentum</strong> gives the optimizer a "velocity": it accumulates past gradients into a running average, building up speed in consistent directions and dampening oscillations in noisy ones. The result is dramatically faster convergence on ill-conditioned problems.</p></div></div>

<h2>Classical Momentum (Polyak, 1964)</h2>

<div class="env-block definition"><div class="env-title">SGD with Momentum</div><div class="env-body"><p>Introduce a velocity variable \\(v_t\\) that accumulates an exponentially decaying moving average of past gradients:</p>
<p>\\[ v_{t+1} = \\beta\\, v_t + \\nabla_\\theta \\mathcal{L}(\\theta_t) \\]</p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\eta\\, v_{t+1} \\]</p>
<p>The hyperparameter \\(\\beta \\in [0, 1)\\) is the <strong>momentum coefficient</strong>. Typical values: \\(\\beta = 0.9\\) or \\(0.99\\). The velocity \\(v_t\\) is an exponential moving average of gradients with effective window \\(\\approx 1/(1-\\beta)\\).</p></div></div>

<div class="env-block remark"><div class="env-title">Why Momentum Helps</div><div class="env-body"><p>Consider a loss surface like \\(f(x,y) = 50x^2 + y^2\\). The condition number is \\(\\kappa = 50/1 = 50\\), meaning the curvature differs by a factor of 50 across directions. SGD with a safe learning rate oscillates in \\(x\\) and barely moves in \\(y\\). Momentum smooths these oscillations: the \\(x\\)-gradients alternate signs and cancel in \\(v_t\\), while the consistent \\(y\\)-gradients accumulate, effectively boosting the step size in the low-curvature direction.</p></div></div>

<h2>Nesterov Accelerated Gradient (NAG)</h2>

<div class="env-block definition"><div class="env-title">Nesterov Momentum</div><div class="env-body"><p>Nesterov's key idea: compute the gradient at the <strong>lookahead position</strong> \\(\\theta_t - \\eta \\beta v_t\\) rather than at the current position \\(\\theta_t\\):</p>
<p>\\[ v_{t+1} = \\beta\\, v_t + \\nabla_\\theta \\mathcal{L}(\\theta_t - \\eta \\beta\\, v_t) \\]</p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\eta\\, v_{t+1} \\]</p>
<p>By evaluating the gradient "ahead" of where we are, NAG gets a corrective signal sooner and can brake before overshooting. For smooth convex functions, NAG achieves the optimal convergence rate \\(O(1/T^2)\\) compared to \\(O(1/T)\\) for plain GD.</p></div></div>

<div class="env-block theorem"><div class="env-title">Nesterov's Optimal Rate</div><div class="env-body"><p>For an \\(L\\)-smooth, \\(\\mu\\)-strongly convex function, Nesterov accelerated gradient with \\(\\beta = \\frac{\\sqrt{L} - \\sqrt{\\mu}}{\\sqrt{L} + \\sqrt{\\mu}}\\) achieves:</p>
<p>\\[ f(\\theta_T) - f(\\theta^*) \\leq O\\!\\left(\\exp\\!\\left(-\\frac{T}{\\sqrt{\\kappa}}\\right)\\right), \\qquad \\kappa = L/\\mu \\]</p>
<p>This is provably optimal among first-order methods. Compared to GD's rate of \\(\\exp(-T/\\kappa)\\), NAG replaces \\(\\kappa\\) with \\(\\sqrt{\\kappa}\\), which is a massive improvement when \\(\\kappa\\) is large.</p></div></div>

<div class="env-block warning"><div class="env-title">Momentum in Practice</div><div class="env-body"><p>The theoretical advantage of Nesterov over classical momentum is clearest in convex optimization. In deep learning (non-convex, stochastic), the two often perform similarly. PyTorch's <code>SGD(momentum=0.9, nesterov=True)</code> implements Nesterov momentum and is a strong default for many tasks, especially with CNNs.</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz02"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz02',
                    title: 'SGD vs Momentum vs Nesterov',
                    description: 'Three optimizers race on an elongated valley (condition number \u224850). Watch how momentum dampens oscillations and Nesterov anticipates the correction. Adjust \u03B2 and learning rate.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 55, originX: 300, originY: 220, width: 600, height: 440 });
                        var ctx = viz.ctx;
                        var lr = 0.01;
                        var beta = 0.9;
                        var running = false;
                        var animId = null;

                        // f(x,y) = 50x^2 + y^2 — condition number 50
                        function grad(x, y) { return [100 * x, 2 * y]; }
                        function loss(x, y) { return 50 * x * x + y * y; }

                        var start = [2.0, 4.0];
                        var sgd, mom, nes;

                        function initPaths() {
                            sgd = { pos: start.slice(), path: [start.slice()] };
                            mom = { pos: start.slice(), vel: [0, 0], path: [start.slice()] };
                            nes = { pos: start.slice(), vel: [0, 0], path: [start.slice()] };
                        }
                        initPaths();
                        var stepCount = 0;

                        function stepAll() {
                            // SGD
                            var gS = grad(sgd.pos[0], sgd.pos[1]);
                            sgd.pos[0] -= lr * gS[0];
                            sgd.pos[1] -= lr * gS[1];
                            sgd.path.push(sgd.pos.slice());

                            // Momentum
                            var gM = grad(mom.pos[0], mom.pos[1]);
                            mom.vel[0] = beta * mom.vel[0] + gM[0];
                            mom.vel[1] = beta * mom.vel[1] + gM[1];
                            mom.pos[0] -= lr * mom.vel[0];
                            mom.pos[1] -= lr * mom.vel[1];
                            mom.path.push(mom.pos.slice());

                            // Nesterov
                            var lookX = nes.pos[0] - lr * beta * nes.vel[0];
                            var lookY = nes.pos[1] - lr * beta * nes.vel[1];
                            var gN = grad(lookX, lookY);
                            nes.vel[0] = beta * nes.vel[0] + gN[0];
                            nes.vel[1] = beta * nes.vel[1] + gN[1];
                            nes.pos[0] -= lr * nes.vel[0];
                            nes.pos[1] -= lr * nes.vel[1];
                            nes.path.push(nes.pos.slice());

                            stepCount++;
                        }

                        function drawContours() {
                            var levels = [5, 20, 50, 100, 200, 400, 700, 1000];
                            for (var li = 0; li < levels.length; li++) {
                                var lev = levels[li];
                                ctx.strokeStyle = 'rgba(88,166,255,' + (0.12 + 0.06 * li) + ')';
                                ctx.lineWidth = 0.8;
                                ctx.beginPath();
                                for (var a = 0; a <= 360; a += 2) {
                                    var rad = a * Math.PI / 180;
                                    var rx = Math.sqrt(lev / 50);
                                    var ry = Math.sqrt(lev);
                                    var px = rx * Math.cos(rad);
                                    var py = ry * Math.sin(rad);
                                    var sp = viz.toScreen(px, py);
                                    a === 0 ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                                }
                                ctx.closePath();
                                ctx.stroke();
                            }
                        }

                        function drawPath(p, color) {
                            if (p.length < 2) return;
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            var maxDraw = Math.min(p.length, 300);
                            for (var i = p.length - maxDraw; i < p.length; i++) {
                                var sp = viz.toScreen(p[i][0], p[i][1]);
                                i === p.length - maxDraw ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                            }
                            ctx.stroke();
                            var last = p[p.length - 1];
                            viz.drawPoint(last[0], last[1], color, '', 4);
                        }

                        function draw() {
                            viz.clear();
                            drawContours();
                            viz.drawPoint(0, 0, viz.colors.green, '', 4);

                            drawPath(sgd.path, viz.colors.red);
                            drawPath(mom.path, viz.colors.blue);
                            drawPath(nes.path, viz.colors.teal);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            var lx = 14, ly = 18;
                            ctx.fillStyle = viz.colors.red; ctx.fillRect(lx, ly - 4, 14, 3); ctx.fillText('SGD (loss: ' + loss(sgd.pos[0], sgd.pos[1]).toFixed(2) + ')', lx + 20, ly);
                            ctx.fillStyle = viz.colors.blue; ctx.fillRect(lx, ly + 14, 14, 3); ctx.fillText('Momentum (loss: ' + loss(mom.pos[0], mom.pos[1]).toFixed(2) + ')', lx + 20, ly + 18);
                            ctx.fillStyle = viz.colors.teal; ctx.fillRect(lx, ly + 32, 14, 3); ctx.fillText('Nesterov (loss: ' + loss(nes.pos[0], nes.pos[1]).toFixed(2) + ')', lx + 20, ly + 36);

                            viz.screenText('f(x,y) = 50x\u00B2 + y\u00B2   |   Step: ' + stepCount, viz.width / 2, viz.height - 12, viz.colors.text, 11);
                        }

                        function animate() {
                            if (!running) return;
                            stepAll();
                            draw();
                            if (stepCount > 800) running = false;
                            animId = requestAnimationFrame(animate);
                        }

                        draw();

                        VizEngine.createSlider(controls, '\u03B7', 0.001, 0.03, lr, 0.001, function(v) { lr = v; });
                        VizEngine.createSlider(controls, '\u03B2', 0.0, 0.99, beta, 0.01, function(v) { beta = v; });
                        VizEngine.createButton(controls, 'Run', function() { if (!running) { running = true; animate(); } });
                        VizEngine.createButton(controls, 'Step', function() { running = false; stepAll(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; if (animId) cancelAnimationFrame(animId); initPaths(); stepCount = 0; draw(); });

                        return { stopAnimation: function() { running = false; if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex04',
                    type: 'derivation',
                    question: 'Show that the momentum velocity \\(v_t = \\sum_{k=0}^{t} \\beta^{t-k} g_k\\) where \\(g_k = \\nabla \\mathcal{L}(\\theta_k)\\). What is the effective weight of a gradient from \\(\\tau\\) steps ago?',
                    hint: 'Unroll the recurrence \\(v_{t+1} = \\beta v_t + g_t\\) starting from \\(v_0 = 0\\).',
                    solution: 'Unrolling: \\(v_1 = g_0\\), \\(v_2 = \\beta g_0 + g_1\\), \\(v_3 = \\beta^2 g_0 + \\beta g_1 + g_2\\). In general, \\(v_t = \\sum_{k=0}^{t-1} \\beta^{t-1-k} g_k\\). A gradient from \\(\\tau\\) steps ago has weight \\(\\beta^\\tau\\). With \\(\\beta = 0.9\\), the weight halves every \\(\\log(0.5)/\\log(0.9) \\approx 6.6\\) steps. The effective window length is \\(1/(1-\\beta) = 10\\) steps.'
                },
                {
                    id: 'ch07-ex05',
                    type: 'conceptual',
                    question: 'In the visualization, set \\(\\beta = 0\\) for the momentum optimizer. What do you observe, and why?',
                    hint: 'When \\(\\beta = 0\\), the velocity formula becomes \\(v_{t+1} = g_t\\).',
                    solution: 'With \\(\\beta = 0\\), momentum reduces to plain SGD: \\(v_{t+1} = g_t\\) and \\(\\theta_{t+1} = \\theta_t - \\eta g_t\\). The momentum path should coincide exactly with the SGD path. This confirms that momentum is a strict generalization of SGD.'
                },
                {
                    id: 'ch07-ex06',
                    type: 'conceptual',
                    question: 'Why does Nesterov momentum "look ahead" before computing the gradient? What practical benefit does this provide over classical momentum?',
                    hint: 'Think about what happens when the velocity is about to overshoot the minimum.',
                    solution: 'Classical momentum computes the gradient at the current position and then adds the velocity. If the velocity is pointing past the minimum, the gradient correction comes one step too late. Nesterov first applies the velocity to get a lookahead position, then computes the gradient there. If the lookahead overshoots, the gradient at that position already points back, providing a corrective "braking" signal. This reduces oscillations and gives the theoretically optimal convergence rate \\(O(1/\\sqrt{\\kappa})\\) for convex problems.'
                }
            ]
        },

        // ===== Section 3: Adaptive Learning Rates =====
        {
            id: 'ch07-sec03',
            title: 'Adaptive Learning Rates',
            content: `
<div class="env-block intuition"><div class="env-title">One Size Does Not Fit All</div><div class="env-body"><p>In momentum methods, every parameter shares the same learning rate \\(\\eta\\). But different parameters may need very different step sizes. Consider word embeddings: frequent words get gradients at almost every step, while rare words get gradients infrequently. A learning rate good for frequent-word parameters may be far too small for rare-word parameters. <strong>Adaptive methods</strong> maintain a separate effective learning rate for each parameter, automatically scaling it based on the history of gradients for that parameter.</p></div></div>

<h2>AdaGrad (Duchi et al., 2011)</h2>

<div class="env-block definition"><div class="env-title">AdaGrad</div><div class="env-body"><p>AdaGrad accumulates the <strong>sum of squared gradients</strong> for each parameter:</p>
<p>\\[ G_{t+1,j} = G_{t,j} + g_{t,j}^2 \\]</p>
<p>\\[ \\theta_{t+1,j} = \\theta_{t,j} - \\frac{\\eta}{\\sqrt{G_{t+1,j}} + \\epsilon}\\, g_{t,j} \\]</p>
<p>where \\(g_{t,j} = \\frac{\\partial \\mathcal{L}}{\\partial \\theta_j}\\Big|_{\\theta_t}\\) and \\(\\epsilon \\approx 10^{-8}\\) prevents division by zero. Each parameter \\(j\\) gets an individually scaled learning rate \\(\\eta / (\\sqrt{G_j} + \\epsilon)\\).</p></div></div>

<div class="env-block remark"><div class="env-title">AdaGrad's Strengths</div><div class="env-body"><p><strong>Sparse features</strong>: Parameters associated with infrequent features accumulate small \\(G_j\\), so they retain a large effective learning rate. Frequent-feature parameters accumulate large \\(G_j\\) and get dampened. This is ideal for NLP tasks with large, sparse vocabularies.</p>
<p><strong>No manual tuning per parameter</strong>: A single global \\(\\eta\\) works well because the per-parameter scaling handles the rest.</p></div></div>

<div class="env-block warning"><div class="env-title">AdaGrad's Fatal Flaw</div><div class="env-body"><p>The accumulator \\(G_j\\) <em>only grows</em>. Over many iterations, the effective learning rate \\(\\eta/\\sqrt{G_j}\\) monotonically decreases toward zero. For non-convex problems requiring sustained learning, this premature decay can halt progress long before a good solution is found. This motivated RMSProp.</p></div></div>

<h2>RMSProp (Hinton, 2012)</h2>

<div class="env-block definition"><div class="env-title">RMSProp</div><div class="env-body"><p>RMSProp fixes AdaGrad's decay problem by using an <strong>exponential moving average</strong> of squared gradients instead of a cumulative sum:</p>
<p>\\[ E[g^2]_{t+1,j} = \\gamma\\, E[g^2]_{t,j} + (1 - \\gamma)\\, g_{t,j}^2 \\]</p>
<p>\\[ \\theta_{t+1,j} = \\theta_{t,j} - \\frac{\\eta}{\\sqrt{E[g^2]_{t+1,j}} + \\epsilon}\\, g_{t,j} \\]</p>
<p>The decay rate \\(\\gamma\\) (typically 0.9 or 0.99) controls the window. The denominator estimates the RMS (root mean square) of recent gradients, preventing the unbounded growth that plagues AdaGrad.</p></div></div>

<div class="env-block remark"><div class="env-title">RMSProp: The Unpublished Workhorse</div><div class="env-body"><p>RMSProp was proposed by Geoffrey Hinton in lecture slides for his Coursera course, never formally published. Despite this, it became one of the most widely used optimizers in deep learning and directly inspired Adam. The effective learning rate for parameter \\(j\\) is \\(\\eta / \\text{RMS}(g_j)\\), which adapts to the local curvature: parameters in steep directions (large RMS) get small steps, while parameters in flat directions (small RMS) get large steps.</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz03"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz03',
                    title: 'AdaGrad vs RMSProp',
                    description: 'Compare how AdaGrad and RMSProp handle an elongated loss surface. Notice how AdaGrad slows down over time while RMSProp maintains progress. The per-parameter learning rates are shown as bar heights.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 55, originX: 300, originY: 200, width: 600, height: 440 });
                        var ctx = viz.ctx;
                        var lr = 0.5;
                        var gamma = 0.9;
                        var running = false;
                        var animId = null;

                        // f(x,y) = 25x^2 + y^2
                        function grad(x, y) { return [50 * x, 2 * y]; }
                        function loss(x, y) { return 25 * x * x + y * y; }

                        var start = [2.0, 4.0];
                        var ada, rms;
                        var stepCount = 0;

                        function initPaths() {
                            ada = { pos: start.slice(), G: [0, 0], path: [start.slice()] };
                            rms = { pos: start.slice(), Eg2: [0, 0], path: [start.slice()] };
                            stepCount = 0;
                        }
                        initPaths();

                        function stepAll() {
                            var eps = 1e-8;
                            // AdaGrad
                            var gA = grad(ada.pos[0], ada.pos[1]);
                            ada.G[0] += gA[0] * gA[0];
                            ada.G[1] += gA[1] * gA[1];
                            ada.pos[0] -= lr / (Math.sqrt(ada.G[0]) + eps) * gA[0];
                            ada.pos[1] -= lr / (Math.sqrt(ada.G[1]) + eps) * gA[1];
                            ada.path.push(ada.pos.slice());

                            // RMSProp
                            var gR = grad(rms.pos[0], rms.pos[1]);
                            rms.Eg2[0] = gamma * rms.Eg2[0] + (1 - gamma) * gR[0] * gR[0];
                            rms.Eg2[1] = gamma * rms.Eg2[1] + (1 - gamma) * gR[1] * gR[1];
                            rms.pos[0] -= lr / (Math.sqrt(rms.Eg2[0]) + eps) * gR[0];
                            rms.pos[1] -= lr / (Math.sqrt(rms.Eg2[1]) + eps) * gR[1];
                            rms.path.push(rms.pos.slice());

                            stepCount++;
                        }

                        function drawContours() {
                            var levels = [2, 8, 20, 50, 100, 200, 400];
                            for (var li = 0; li < levels.length; li++) {
                                var lev = levels[li];
                                ctx.strokeStyle = 'rgba(88,166,255,' + (0.12 + 0.06 * li) + ')';
                                ctx.lineWidth = 0.8;
                                ctx.beginPath();
                                for (var a = 0; a <= 360; a += 2) {
                                    var rad = a * Math.PI / 180;
                                    var rx = Math.sqrt(lev / 25);
                                    var ry = Math.sqrt(lev);
                                    var sp = viz.toScreen(rx * Math.cos(rad), ry * Math.sin(rad));
                                    a === 0 ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                                }
                                ctx.closePath();
                                ctx.stroke();
                            }
                        }

                        function drawPath(p, color) {
                            if (p.length < 2) return;
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            for (var i = 0; i < p.length; i++) {
                                var sp = viz.toScreen(p[i][0], p[i][1]);
                                i === 0 ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                            }
                            ctx.stroke();
                            var last = p[p.length - 1];
                            viz.drawPoint(last[0], last[1], color, '', 4);
                        }

                        function draw() {
                            viz.clear();
                            drawContours();
                            viz.drawPoint(0, 0, viz.colors.green, '', 4);

                            drawPath(ada.path, viz.colors.orange);
                            drawPath(rms.path, viz.colors.teal);

                            // Effective LR bars
                            var eps = 1e-8;
                            var barX = viz.width - 130;
                            var barY = 60;
                            ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillText('Effective LR (x-param)', barX, barY - 8);
                            var adaLRx = ada.G[0] > 0 ? lr / (Math.sqrt(ada.G[0]) + eps) : lr;
                            var rmsLRx = rms.Eg2[0] > 0 ? lr / (Math.sqrt(rms.Eg2[0]) + eps) : lr;
                            var maxLR = Math.max(adaLRx, rmsLRx, 0.01);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(barX, barY, Math.min(100, (adaLRx / maxLR) * 100), 8);
                            ctx.fillText('Ada: ' + adaLRx.toExponential(1), barX, barY + 20);
                            ctx.fillStyle = viz.colors.teal;
                            ctx.fillRect(barX, barY + 28, Math.min(100, (rmsLRx / maxLR) * 100), 8);
                            ctx.fillText('RMS: ' + rmsLRx.toExponential(1), barX, barY + 48);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            var lx = 14, ly = 18;
                            ctx.fillStyle = viz.colors.orange; ctx.fillRect(lx, ly - 4, 14, 3); ctx.fillText('AdaGrad (loss: ' + loss(ada.pos[0], ada.pos[1]).toFixed(2) + ')', lx + 20, ly);
                            ctx.fillStyle = viz.colors.teal; ctx.fillRect(lx, ly + 16, 14, 3); ctx.fillText('RMSProp (loss: ' + loss(rms.pos[0], rms.pos[1]).toFixed(2) + ')', lx + 20, ly + 20);

                            viz.screenText('f(x,y) = 25x\u00B2 + y\u00B2   |   Step: ' + stepCount, viz.width / 2, viz.height - 12, viz.colors.text, 11);
                        }

                        function animate() {
                            if (!running) return;
                            stepAll();
                            draw();
                            if (stepCount > 500) running = false;
                            animId = requestAnimationFrame(animate);
                        }

                        draw();

                        VizEngine.createSlider(controls, '\u03B7', 0.05, 2.0, lr, 0.05, function(v) { lr = v; });
                        VizEngine.createSlider(controls, '\u03B3 (RMS)', 0.8, 0.999, gamma, 0.001, function(v) { gamma = v; });
                        VizEngine.createButton(controls, 'Run', function() { if (!running) { running = true; animate(); } });
                        VizEngine.createButton(controls, 'Step', function() { running = false; stepAll(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; if (animId) cancelAnimationFrame(animId); initPaths(); draw(); });

                        return { stopAnimation: function() { running = false; if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex07',
                    type: 'derivation',
                    question: 'Show that AdaGrad\'s effective learning rate for parameter \\(j\\) after \\(T\\) steps is approximately \\(\\eta \\sqrt{T} / \\|g_{1:T,j}\\|_2\\) where \\(\\|g_{1:T,j}\\|_2 = \\sqrt{\\sum_{t=1}^T g_{t,j}^2}\\). What happens if all gradients for parameter \\(j\\) have the same magnitude \\(c\\)?',
                    hint: 'Write out \\(G_{T,j}\\) and simplify.',
                    solution: 'After \\(T\\) steps, \\(G_{T,j} = \\sum_{t=1}^T g_{t,j}^2\\). The effective learning rate is \\(\\eta / \\sqrt{G_{T,j}} = \\eta / \\|g_{1:T,j}\\|_2\\). If all gradients have magnitude \\(c\\), then \\(G_{T,j} = Tc^2\\), giving effective rate \\(\\eta / (c\\sqrt{T})\\). This decays as \\(O(1/\\sqrt{T})\\) regardless of the gradient magnitude, which for convex problems matches the theoretically optimal learning rate schedule. For non-convex problems, however, this monotonic decay is too aggressive.'
                },
                {
                    id: 'ch07-ex08',
                    type: 'conceptual',
                    question: 'Explain why RMSProp can be seen as an "exponentially-weighted" version of AdaGrad. What is the effective window length?',
                    hint: 'Compare the accumulator formulas. The EMA has weights \\((1-\\gamma)\\gamma^k\\).',
                    solution: 'AdaGrad uses \\(G_{T,j} = \\sum_{t=1}^T g_{t,j}^2\\), weighting all past gradients equally. RMSProp uses \\(E[g^2]_{T,j} = (1-\\gamma)\\sum_{t=1}^T \\gamma^{T-t} g_{t,j}^2\\), an exponentially-weighted average that down-weights old gradients. The effective window length is \\(1/(1-\\gamma)\\): with \\(\\gamma=0.9\\), the optimizer "remembers" roughly the last 10 gradients. This prevents the accumulator from growing without bound, keeping the effective learning rate from vanishing.'
                },
                {
                    id: 'ch07-ex09',
                    type: 'conceptual',
                    question: 'Why are adaptive methods particularly useful for training models with embedding layers (e.g., word2vec, recommendation systems)?',
                    hint: 'Think about gradient sparsity: most embedding rows receive zero gradients at each step.',
                    solution: 'Embedding layers are extremely sparse: at each mini-batch, only the rows corresponding to the items in that batch receive non-zero gradients. With SGD, all rows share the same learning rate, so rare items learn very slowly. With AdaGrad/RMSProp, each embedding row has its own accumulator. Rare items have small \\(G_j\\), so their effective learning rate remains high, allowing them to learn from each of their infrequent gradient updates. Frequent items accumulate large \\(G_j\\), preventing their embeddings from changing too rapidly. This automatic frequency-based scaling is why adaptive methods dominate in NLP and recommendation systems.'
                }
            ]
        },

        // ===== Section 4: Adam & AdamW =====
        {
            id: 'ch07-sec04',
            title: 'Adam & AdamW',
            content: `
<div class="env-block intuition"><div class="env-title">The Best of Both Worlds</div><div class="env-body"><p>Momentum accumulates gradient <em>direction</em> (first moment). RMSProp accumulates gradient <em>magnitude</em> (second moment). <strong>Adam</strong> (Adaptive Moment Estimation; Kingma & Ba, 2015) combines both ideas into a single optimizer that is simultaneously momentum-accelerated and per-parameter adaptive. It is the most popular optimizer in deep learning for good reason: it works well out of the box on a wide range of architectures with minimal hyperparameter tuning.</p></div></div>

<h2>The Adam Algorithm</h2>

<div class="env-block definition"><div class="env-title">Adam (Kingma & Ba, 2015)</div><div class="env-body"><p>Adam maintains exponential moving averages of both the first moment (mean) and the second moment (uncentered variance) of the gradient:</p>
<p>\\[ m_t = \\beta_1 m_{t-1} + (1 - \\beta_1)\\, g_t \\qquad \\text{(first moment estimate)} \\]</p>
<p>\\[ v_t = \\beta_2 v_{t-1} + (1 - \\beta_2)\\, g_t^2 \\qquad \\text{(second moment estimate)} \\]</p>
<p><strong>Bias correction</strong> (critical in early steps when \\(m_t, v_t\\) are biased toward zero):</p>
<p>\\[ \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\]</p>
<p><strong>Parameter update:</strong></p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t \\]</p>
<p>Default hyperparameters: \\(\\beta_1 = 0.9\\), \\(\\beta_2 = 0.999\\), \\(\\epsilon = 10^{-8}\\), \\(\\eta = 0.001\\).</p></div></div>

<div class="env-block remark"><div class="env-title">Why Bias Correction Matters</div><div class="env-body"><p>Both \\(m_t\\) and \\(v_t\\) are initialized to zero. In the first few steps, the EMA is heavily biased toward zero. For example, with \\(\\beta_2 = 0.999\\), after step 1: \\(v_1 = 0.001 g_1^2\\), which is 1000x smaller than the true second moment. The correction \\(\\hat{v}_1 = v_1 / (1 - 0.999^1) = v_1 / 0.001 = g_1^2\\) recovers the correct scale. Without correction, Adam would take enormous steps early on.</p></div></div>

<h2>The Weight Decay Problem</h2>

<div class="env-block definition"><div class="env-title">L2 Regularization vs Weight Decay</div><div class="env-body"><p>In SGD, adding \\(\\frac{\\lambda}{2}\\|\\theta\\|^2\\) to the loss (L2 regularization) is equivalent to adding \\(-\\lambda\\theta\\) to the update (weight decay):</p>
<p>\\[ \\theta_{t+1} = \\theta_t - \\eta(g_t + \\lambda\\theta_t) = (1 - \\eta\\lambda)\\theta_t - \\eta g_t \\]</p>
<p>For Adam, these are <strong>not</strong> equivalent. L2 regularization adds \\(\\lambda\\theta\\) to the gradient, which then gets scaled by the adaptive denominator \\(1/\\sqrt{\\hat{v}_t}\\). This means the regularization strength varies per parameter, which is unintended.</p></div></div>

<h2>AdamW: Decoupled Weight Decay</h2>

<div class="env-block definition"><div class="env-title">AdamW (Loshchilov & Hutter, 2019)</div><div class="env-body"><p>AdamW applies weight decay <strong>directly</strong> to the parameters, decoupled from the adaptive gradient scaling:</p>
<p>\\[ \\theta_{t+1} = (1 - \\eta\\lambda)\\,\\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon}\\, \\hat{m}_t \\]</p>
<p>The weight decay term \\((1 - \\eta\\lambda)\\theta_t\\) is applied uniformly to all parameters, independent of the gradient statistics. This restores the proper regularization behavior and consistently outperforms Adam with L2 regularization in practice.</p></div></div>

<div class="env-block warning"><div class="env-title">Adam's Known Issues</div><div class="env-body"><p><strong>Generalization gap</strong>: On some tasks (especially image classification with CNNs), SGD with momentum still outperforms Adam in test accuracy, even when Adam achieves lower training loss. Hypotheses include: (1) Adam converges to sharper minima; (2) the adaptive learning rates allow some parameters to "overfit" independently.</p>
<p><strong>Non-convergence</strong>: Reddi et al. (2018) showed Adam can diverge on simple convex problems. AMSGrad fixes this by using \\(\\max(\\hat{v}_t, \\hat{v}_{t-1})\\), but the practical impact is minimal.</p>
<p><strong>Modern consensus</strong>: AdamW is the standard for transformers and LLMs. SGD with momentum remains competitive for CNNs. When in doubt, try AdamW first.</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz04"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz04',
                    title: 'Adam vs SGD vs Momentum on the Rosenbrock Function',
                    description: 'The Rosenbrock function f(x,y) = (1-x)\u00B2 + 100(y-x\u00B2)\u00B2 has a narrow curved valley. Watch Adam navigate it compared to SGD and Momentum.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 100, originX: 200, originY: 320, width: 600, height: 440 });
                        var ctx = viz.ctx;
                        var lr = 0.001;
                        var running = false;
                        var animId = null;

                        // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
                        function rosenbrock(x, y) { return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x); }
                        function rosenGrad(x, y) {
                            var dx = -2 * (1 - x) + 200 * (y - x * x) * (-2 * x);
                            var dy = 200 * (y - x * x);
                            return [dx, dy];
                        }

                        var start = [-1.0, 1.5];
                        var sgdOpt, momOpt, adamOpt;
                        var stepCount = 0;

                        function initAll() {
                            sgdOpt = { pos: start.slice(), path: [start.slice()] };
                            momOpt = { pos: start.slice(), vel: [0, 0], path: [start.slice()] };
                            adamOpt = { pos: start.slice(), m: [0, 0], v: [0, 0], t: 0, path: [start.slice()] };
                            stepCount = 0;
                        }
                        initAll();

                        function clampGrad(g, maxNorm) {
                            var norm = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                            if (norm > maxNorm) { g[0] *= maxNorm / norm; g[1] *= maxNorm / norm; }
                            return g;
                        }

                        function stepAll() {
                            var maxG = 50;
                            // SGD
                            var gS = clampGrad(rosenGrad(sgdOpt.pos[0], sgdOpt.pos[1]), maxG);
                            sgdOpt.pos[0] -= lr * gS[0];
                            sgdOpt.pos[1] -= lr * gS[1];
                            sgdOpt.path.push(sgdOpt.pos.slice());

                            // Momentum
                            var gM = clampGrad(rosenGrad(momOpt.pos[0], momOpt.pos[1]), maxG);
                            momOpt.vel[0] = 0.9 * momOpt.vel[0] + gM[0];
                            momOpt.vel[1] = 0.9 * momOpt.vel[1] + gM[1];
                            momOpt.pos[0] -= lr * momOpt.vel[0];
                            momOpt.pos[1] -= lr * momOpt.vel[1];
                            momOpt.path.push(momOpt.pos.slice());

                            // Adam
                            adamOpt.t++;
                            var gA = clampGrad(rosenGrad(adamOpt.pos[0], adamOpt.pos[1]), maxG);
                            var b1 = 0.9, b2 = 0.999, eps = 1e-8;
                            adamOpt.m[0] = b1 * adamOpt.m[0] + (1 - b1) * gA[0];
                            adamOpt.m[1] = b1 * adamOpt.m[1] + (1 - b1) * gA[1];
                            adamOpt.v[0] = b2 * adamOpt.v[0] + (1 - b2) * gA[0] * gA[0];
                            adamOpt.v[1] = b2 * adamOpt.v[1] + (1 - b2) * gA[1] * gA[1];
                            var mh0 = adamOpt.m[0] / (1 - Math.pow(b1, adamOpt.t));
                            var mh1 = adamOpt.m[1] / (1 - Math.pow(b1, adamOpt.t));
                            var vh0 = adamOpt.v[0] / (1 - Math.pow(b2, adamOpt.t));
                            var vh1 = adamOpt.v[1] / (1 - Math.pow(b2, adamOpt.t));
                            adamOpt.pos[0] -= lr * mh0 / (Math.sqrt(vh0) + eps);
                            adamOpt.pos[1] -= lr * mh1 / (Math.sqrt(vh1) + eps);
                            adamOpt.path.push(adamOpt.pos.slice());

                            stepCount++;
                        }

                        function drawContours() {
                            // Draw Rosenbrock contours using marching approach
                            var levels = [1, 5, 25, 100, 400, 1000, 3000];
                            var xMin = -2.0, xMax = 3.0, yMin = -1.0, yMax = 4.0;
                            var res = 2;
                            for (var li = 0; li < levels.length; li++) {
                                var lev = levels[li];
                                ctx.strokeStyle = 'rgba(88,166,255,' + (0.1 + 0.06 * li) + ')';
                                ctx.lineWidth = 0.7;
                                // Scan and draw contour segments
                                for (var px = xMin; px < xMax; px += 0.05) {
                                    for (var py = yMin; py < yMax; py += 0.05) {
                                        var step2 = 0.05;
                                        var v00 = rosenbrock(px, py) - lev;
                                        var v10 = rosenbrock(px + step2, py) - lev;
                                        var v01 = rosenbrock(px, py + step2) - lev;
                                        // Horizontal edge
                                        if (v00 * v10 < 0) {
                                            var frac = v00 / (v00 - v10);
                                            var cx = px + frac * step2;
                                            var s1 = viz.toScreen(cx, py);
                                            ctx.fillStyle = ctx.strokeStyle;
                                            ctx.fillRect(s1[0], s1[1], res, res);
                                        }
                                        // Vertical edge
                                        if (v00 * v01 < 0) {
                                            var frac2 = v00 / (v00 - v01);
                                            var cy = py + frac2 * step2;
                                            var s2 = viz.toScreen(px, cy);
                                            ctx.fillStyle = ctx.strokeStyle;
                                            ctx.fillRect(s2[0], s2[1], res, res);
                                        }
                                    }
                                }
                            }
                        }

                        function drawPath(p, color) {
                            if (p.length < 2) return;
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1.5;
                            ctx.globalAlpha = 0.8;
                            ctx.beginPath();
                            var maxDraw = Math.min(p.length, 500);
                            var startI = p.length - maxDraw;
                            for (var i = startI; i < p.length; i++) {
                                var sp = viz.toScreen(p[i][0], p[i][1]);
                                i === startI ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                            }
                            ctx.stroke();
                            ctx.globalAlpha = 1;
                            var last = p[p.length - 1];
                            viz.drawPoint(last[0], last[1], color, '', 4);
                        }

                        function draw() {
                            viz.clear();
                            drawContours();

                            // Mark global minimum at (1,1)
                            viz.drawPoint(1, 1, viz.colors.green, '  (1,1)', 5);

                            drawPath(sgdOpt.path, viz.colors.red);
                            drawPath(momOpt.path, viz.colors.blue);
                            drawPath(adamOpt.path, viz.colors.teal);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            var lx = 14, ly = 18;
                            ctx.fillStyle = viz.colors.red; ctx.fillRect(lx, ly - 4, 14, 3); ctx.fillText('SGD (' + rosenbrock(sgdOpt.pos[0], sgdOpt.pos[1]).toFixed(1) + ')', lx + 20, ly);
                            ctx.fillStyle = viz.colors.blue; ctx.fillRect(lx, ly + 14, 14, 3); ctx.fillText('Momentum (' + rosenbrock(momOpt.pos[0], momOpt.pos[1]).toFixed(1) + ')', lx + 20, ly + 18);
                            ctx.fillStyle = viz.colors.teal; ctx.fillRect(lx, ly + 32, 14, 3); ctx.fillText('Adam (' + rosenbrock(adamOpt.pos[0], adamOpt.pos[1]).toFixed(1) + ')', lx + 20, ly + 36);

                            viz.screenText('Rosenbrock: f = (1-x)\u00B2 + 100(y-x\u00B2)\u00B2   |   Step: ' + stepCount, viz.width / 2, viz.height - 12, viz.colors.text, 10);
                        }

                        function animate() {
                            if (!running) return;
                            for (var i = 0; i < 3; i++) stepAll(); // 3 sub-steps per frame
                            draw();
                            if (stepCount > 3000) running = false;
                            animId = requestAnimationFrame(animate);
                        }

                        draw();

                        VizEngine.createSlider(controls, '\u03B7', 0.0001, 0.01, lr, 0.0001, function(v) { lr = v; });
                        VizEngine.createButton(controls, 'Run', function() { if (!running) { running = true; animate(); } });
                        VizEngine.createButton(controls, 'Step x10', function() { running = false; for (var i = 0; i < 10; i++) stepAll(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; if (animId) cancelAnimationFrame(animId); initAll(); draw(); });

                        return { stopAnimation: function() { running = false; if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex10',
                    type: 'derivation',
                    question: 'Show that Adam\'s bias-corrected first moment \\(\\hat{m}_t\\) is an unbiased estimate of \\(\\mathbb{E}[g_t]\\), assuming the true gradient distribution is stationary (i.e., \\(\\mathbb{E}[g_t] = g\\) for all \\(t\\)).',
                    hint: 'Write \\(m_t\\) as a weighted sum of past gradients. Take the expectation and factor out.',
                    solution: 'Unrolling: \\(m_t = (1 - \\beta_1)\\sum_{k=1}^t \\beta_1^{t-k} g_k\\). Under stationarity, \\(\\mathbb{E}[m_t] = (1 - \\beta_1)g \\sum_{k=1}^t \\beta_1^{t-k} = (1 - \\beta_1) g \\cdot \\frac{1 - \\beta_1^t}{1 - \\beta_1} = g(1 - \\beta_1^t)\\). Thus \\(\\mathbb{E}[\\hat{m}_t] = \\mathbb{E}[m_t]/(1 - \\beta_1^t) = g\\), confirming unbiasedness.'
                },
                {
                    id: 'ch07-ex11',
                    type: 'conceptual',
                    question: 'Explain why L2 regularization and weight decay are equivalent for SGD but not for Adam. What goes wrong with L2 regularization in Adam?',
                    hint: 'In Adam, the gradient passes through the adaptive denominator \\(1/\\sqrt{\\hat{v}_t}\\).',
                    solution: 'With L2 regularization in Adam, the gradient becomes \\(g_t + \\lambda\\theta_t\\). Both terms are divided by \\(\\sqrt{\\hat{v}_t}\\). But \\(\\hat{v}_t\\) is estimated from \\(g_t + \\lambda\\theta_t\\), so the regularization penalty \\(\\lambda\\theta_t\\) is rescaled by gradient statistics it was never intended to interact with. Parameters with large gradient variance get their weight decay dampened (since \\(\\sqrt{\\hat{v}_t}\\) is large). AdamW fixes this by applying weight decay directly: \\(\\theta_{t+1} = (1 - \\eta\\lambda)\\theta_t - \\eta \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)\\), so the regularization strength is uniform across parameters.'
                },
                {
                    id: 'ch07-ex12',
                    type: 'numeric',
                    question: 'At step \\(t=1\\) with \\(\\beta_1=0.9\\), \\(\\beta_2=0.999\\), and gradient \\(g_1 = 5.0\\), compute \\(m_1\\), \\(v_1\\), \\(\\hat{m}_1\\), \\(\\hat{v}_1\\), and the update direction \\(\\hat{m}_1 / (\\sqrt{\\hat{v}_1} + \\epsilon)\\).',
                    hint: 'Recall \\(m_0 = 0\\), \\(v_0 = 0\\).',
                    solution: '\\(m_1 = 0.9 \\cdot 0 + 0.1 \\cdot 5 = 0.5\\). \\(v_1 = 0.999 \\cdot 0 + 0.001 \\cdot 25 = 0.025\\). Bias correction: \\(\\hat{m}_1 = 0.5 / (1 - 0.9) = 5.0\\). \\(\\hat{v}_1 = 0.025 / (1 - 0.999) = 25.0\\). Update direction: \\(5.0 / (\\sqrt{25} + 10^{-8}) = 5.0 / 5.0 \\approx 1.0\\). Notice: after bias correction, the effective step size is approximately \\(\\eta\\), regardless of gradient magnitude. This is the "trust region" property of Adam.'
                },
                {
                    id: 'ch07-ex13',
                    type: 'conceptual',
                    question: 'The "trust region" property of Adam means the effective step size is bounded by approximately \\(\\eta\\) regardless of gradient magnitude. Derive this bound from the update rule.',
                    hint: 'Consider the ratio \\(|\\hat{m}_t| / \\sqrt{\\hat{v}_t}\\) and apply the Cauchy-Schwarz or Jensen inequality.',
                    solution: 'By Jensen\'s inequality applied to the concave function \\(\\sqrt{\\cdot}\\): \\(|\\hat{m}_t|^2 \\leq \\hat{v}_t\\) (since the square of an EMA is at most the EMA of squares). Therefore \\(|\\hat{m}_t| / \\sqrt{\\hat{v}_t} \\leq 1\\), and the effective step size \\(\\eta |\\hat{m}_t| / (\\sqrt{\\hat{v}_t} + \\epsilon) \\leq \\eta\\). This means Adam\'s steps are naturally bounded, making it more robust to gradient explosions than SGD. The learning rate \\(\\eta\\) acts as a trust-region radius.'
                }
            ]
        },

        // ===== Section 5: Learning Rate Schedules =====
        {
            id: 'ch07-sec05',
            title: 'Learning Rate Schedules',
            content: `
<div class="env-block intuition"><div class="env-title">The Right Speed at the Right Time</div><div class="env-body"><p>A fixed learning rate forces a compromise: large enough to make progress early, but small enough to converge precisely at the end. <strong>Learning rate schedules</strong> resolve this tension by varying \\(\\eta\\) over the course of training. Start large to explore quickly, then decay to fine-tune. Modern schedules like cosine annealing and warmup have become essential components of state-of-the-art training recipes.</p></div></div>

<h2>Step Decay</h2>

<div class="env-block definition"><div class="env-title">Step Decay Schedule</div><div class="env-body"><p>Reduce the learning rate by a factor \\(\\gamma\\) every \\(k\\) epochs:</p>
<p>\\[ \\eta_t = \\eta_0 \\cdot \\gamma^{\\lfloor t / k \\rfloor} \\]</p>
<p>A common recipe: \\(\\gamma = 0.1\\), drop every 30 epochs (used in the original ResNet paper). Simple and effective, but requires choosing the drop schedule manually.</p></div></div>

<h2>Cosine Annealing</h2>

<div class="env-block definition"><div class="env-title">Cosine Annealing (Loshchilov & Hutter, 2017)</div><div class="env-body"><p>Smoothly decay the learning rate following a half-cosine curve:</p>
<p>\\[ \\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\frac{\\pi\\, t}{T}\\right)\\right) \\]</p>
<p>where \\(T\\) is the total number of training steps. The cosine schedule starts at \\(\\eta_{\\max}\\), stays high for a while (the cosine is flat near 1), then smoothly drops to \\(\\eta_{\\min}\\). No hyperparameters beyond \\(\\eta_{\\max}\\), \\(\\eta_{\\min}\\), and \\(T\\).</p></div></div>

<h2>Linear Warmup</h2>

<div class="env-block definition"><div class="env-title">Learning Rate Warmup</div><div class="env-body"><p>Linearly increase the learning rate from 0 to \\(\\eta_{\\max}\\) over \\(T_w\\) warmup steps:</p>
<p>\\[ \\eta_t = \\eta_{\\max} \\cdot \\frac{t}{T_w}, \\qquad t \\leq T_w \\]</p>
<p>After warmup, apply a decay schedule (e.g., cosine). Warmup prevents instability in the first few iterations when the model parameters are random and gradients can be very large. It is essential for training transformers with Adam.</p></div></div>

<div class="env-block remark"><div class="env-title">Why Warmup Helps Adam</div><div class="env-body"><p>Adam's second moment \\(v_t\\) needs several steps to calibrate. In the first few iterations, the bias-corrected \\(\\hat{v}_t\\) may be inaccurate, leading to oversized steps. Warmup keeps \\(\\eta\\) small during this calibration period. The combination of linear warmup followed by cosine decay (the "warmup + cosine" schedule) has become the standard for transformer training.</p></div></div>

<h2>One-Cycle Policy</h2>

<div class="env-block definition"><div class="env-title">One-Cycle Policy (Smith & Topin, 2019)</div><div class="env-body"><p>The learning rate follows a single cycle: linearly increase from \\(\\eta_{\\max}/\\text{div}\\) to \\(\\eta_{\\max}\\) during the first \\(p\\%\\) of training, then cosine-decay to \\(\\eta_{\\max}/(\\text{div} \\times 100)\\) for the remainder. Simultaneously, momentum is cycled inversely (high when LR is low, low when LR is high). This enables "super-convergence" with very high learning rates.</p></div></div>

<div class="env-block warning"><div class="env-title">Schedule Interactions with Optimizers</div><div class="env-body"><p>Learning rate schedules interact with the optimizer. For Adam/AdamW, the adaptive denominator already adjusts effective step sizes, so aggressive LR decay may be redundant. In practice, cosine decay to zero works well with AdamW. For SGD with momentum, step decay is the classic choice. The one-cycle policy was designed for SGD and may not always improve Adam-based training.</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz05"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz05',
                    title: 'Learning Rate Schedule Comparison',
                    description: 'Visualize different learning rate schedules over training. Adjust the total epochs, warmup fraction, and other parameters to see how each schedule behaves.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 1, originX: 0, originY: 0, width: 600, height: 380 });
                        var ctx = viz.ctx;

                        var totalEpochs = 100;
                        var warmupFrac = 0.1;
                        var etaMax = 0.1;
                        var stepDropEvery = 30;

                        function constantLR(t, T) { return etaMax; }
                        function stepDecay(t, T) { return etaMax * Math.pow(0.1, Math.floor(t / stepDropEvery)); }
                        function cosineAnneal(t, T) { return etaMax * 0.5 * (1 + Math.cos(Math.PI * t / T)); }
                        function warmupCosine(t, T) {
                            var Tw = Math.floor(warmupFrac * T);
                            if (t < Tw) return etaMax * t / Tw;
                            return etaMax * 0.5 * (1 + Math.cos(Math.PI * (t - Tw) / (T - Tw)));
                        }
                        function oneCycle(t, T) {
                            var phase1End = Math.floor(0.3 * T);
                            var lrMin = etaMax / 25;
                            var lrFinal = etaMax / 2500;
                            if (t < phase1End) {
                                return lrMin + (etaMax - lrMin) * t / phase1End;
                            } else {
                                var progress = (t - phase1End) / (T - phase1End);
                                return lrFinal + (etaMax - lrFinal) * 0.5 * (1 + Math.cos(Math.PI * progress));
                            }
                        }

                        var schedules = [
                            { name: 'Constant', fn: constantLR, color: '#6e7681' },
                            { name: 'Step Decay', fn: stepDecay, color: '#f85149' },
                            { name: 'Cosine', fn: cosineAnneal, color: '#58a6ff' },
                            { name: 'Warmup+Cosine', fn: warmupCosine, color: '#3fb950' },
                            { name: 'One-Cycle', fn: oneCycle, color: '#bc8cff' }
                        ];

                        function draw() {
                            viz.clear();
                            var pad = { left: 70, right: 30, top: 40, bottom: 50 };
                            var gw = viz.width - pad.left - pad.right;
                            var gh = viz.height - pad.top - pad.bottom;

                            // Axes
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(pad.left, pad.top);
                            ctx.lineTo(pad.left, pad.top + gh);
                            ctx.lineTo(pad.left + gw, pad.top + gh);
                            ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            ctx.fillText('Epoch', pad.left + gw / 2, pad.top + gh + 28);
                            ctx.save();
                            ctx.translate(16, pad.top + gh / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Learning Rate', 0, 0);
                            ctx.restore();

                            // Y-axis ticks
                            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                            for (var yi = 0; yi <= 5; yi++) {
                                var yv = etaMax * yi / 5;
                                var yy = pad.top + gh - (yi / 5) * gh;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(yv.toFixed(3), pad.left - 6, yy);
                                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + gw, yy); ctx.stroke();
                            }

                            // X-axis ticks
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            for (var xi = 0; xi <= 5; xi++) {
                                var xv = Math.round(totalEpochs * xi / 5);
                                var xx = pad.left + (xi / 5) * gw;
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText(xv, xx, pad.top + gh + 5);
                            }

                            // Draw schedules
                            var T = totalEpochs;
                            var nPoints = 200;
                            for (var si = 0; si < schedules.length; si++) {
                                var sch = schedules[si];
                                ctx.strokeStyle = sch.color;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                for (var pi = 0; pi <= nPoints; pi++) {
                                    var t = (pi / nPoints) * T;
                                    var lrVal = Math.max(0, sch.fn(t, T));
                                    var sx = pad.left + (pi / nPoints) * gw;
                                    var sy = pad.top + gh - (lrVal / etaMax) * gh;
                                    sy = Math.max(pad.top, Math.min(pad.top + gh, sy));
                                    pi === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();
                            }

                            // Legend
                            var legX = pad.left + 10, legY = pad.top + 6;
                            ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            for (var li = 0; li < schedules.length; li++) {
                                ctx.fillStyle = schedules[li].color;
                                ctx.fillRect(legX + li * 110, legY, 12, 3);
                                ctx.fillText(schedules[li].name, legX + li * 110 + 16, legY + 2);
                            }
                        }

                        draw();

                        VizEngine.createSlider(controls, 'Epochs', 20, 300, totalEpochs, 10, function(v) { totalEpochs = v; draw(); });
                        VizEngine.createSlider(controls, 'Warmup%', 0, 0.4, warmupFrac, 0.02, function(v) { warmupFrac = v; draw(); });
                        VizEngine.createSlider(controls, '\u03B7_max', 0.01, 0.5, etaMax, 0.01, function(v) { etaMax = v; draw(); });
                        VizEngine.createSlider(controls, 'Step drop', 10, 60, stepDropEvery, 5, function(v) { stepDropEvery = v; draw(); });

                        return { stopAnimation: function() {} };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex14',
                    type: 'derivation',
                    question: 'Show that cosine annealing with \\(\\eta_{\\min}=0\\) satisfies \\(\\eta_t = \\eta_0\\) at \\(t=0\\) and \\(\\eta_t = 0\\) at \\(t=T\\), and that the schedule is "slow-fast-slow" (decays slowly near the start and end, fast in the middle).',
                    hint: 'Compute \\(d\\eta_t/dt\\) and analyze where it is largest.',
                    solution: 'At \\(t=0\\): \\(\\eta_0 = \\frac{\\eta_0}{2}(1 + \\cos(0)) = \\frac{\\eta_0}{2} \\cdot 2 = \\eta_0\\). At \\(t=T\\): \\(\\eta_T = \\frac{\\eta_0}{2}(1 + \\cos(\\pi)) = \\frac{\\eta_0}{2}(1 - 1) = 0\\). The derivative \\(d\\eta_t/dt = -\\frac{\\pi \\eta_0}{2T}\\sin(\\pi t/T)\\). This is zero at \\(t=0\\) and \\(t=T\\), and maximal (in magnitude) at \\(t=T/2\\). So the LR decays slowly at first, accelerates in the middle, and decays slowly again at the end, matching the "slow-fast-slow" description.'
                },
                {
                    id: 'ch07-ex15',
                    type: 'conceptual',
                    question: 'Why is warmup particularly important when training transformers, but less critical for CNNs with SGD?',
                    hint: 'Consider the optimizer (Adam vs SGD) and the architecture (attention vs convolution).',
                    solution: 'Two factors converge. (1) <strong>Optimizer</strong>: Adam needs several steps for its second moment estimate \\(v_t\\) to calibrate. Before calibration, the bias-corrected \\(\\hat{v}_t\\) can be inaccurate, causing erratic step sizes. Warmup keeps \\(\\eta\\) small during this period. SGD has no such calibration issue. (2) <strong>Architecture</strong>: Transformers with random initialization can produce very large attention logits and gradient norms in early training. The self-attention mechanism has no built-in normalization of its output scale (unlike convolutions bounded by kernel size). Large initial gradients combined with a large LR cause training to diverge. Warmup mitigates this by starting with tiny steps.'
                },
                {
                    id: 'ch07-ex16',
                    type: 'conceptual',
                    question: 'In the one-cycle policy, momentum is cycled inversely to the learning rate (low momentum when LR is high, high momentum when LR is low). Explain the intuition for this coupling.',
                    hint: 'Think about what happens when the learning rate is very high: do you want to accumulate past gradients aggressively?',
                    solution: 'When the learning rate is high (exploration phase), the optimizer is already taking large steps. High momentum would amplify these steps further, risking divergence. Low momentum during the high-LR phase acts as a brake. Conversely, when the learning rate is low (fine-tuning phase), the steps are small and could get stuck in noise. High momentum during the low-LR phase accumulates signal across many steps, providing the inertia needed to keep making progress. The inverse coupling ensures the "aggressiveness" of the optimizer (LR times effective momentum) stays roughly balanced throughout training.'
                }
            ]
        },

        // ===== Section 6: Optimizer Comparison =====
        {
            id: 'ch07-sec06',
            title: 'Optimizer Comparison',
            content: `
<div class="env-block intuition"><div class="env-title">Choosing the Right Optimizer</div><div class="env-body"><p>There is no single best optimizer. The choice depends on the architecture, dataset, and computational budget. However, the deep learning community has converged on a few strong defaults that work well in practice. This section provides a comparative view and practical guidelines.</p></div></div>

<h2>Summary of Optimizers</h2>

<table style="width:100%;border-collapse:collapse;font-size:0.88rem;margin:16px 0;">
<tr style="border-bottom:2px solid #30363d;">
<th style="text-align:left;padding:8px;color:#58a6ff;">Optimizer</th>
<th style="text-align:left;padding:8px;color:#58a6ff;">Key Idea</th>
<th style="text-align:left;padding:8px;color:#58a6ff;">Pros</th>
<th style="text-align:left;padding:8px;color:#58a6ff;">Cons</th>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">SGD</td>
<td style="padding:8px;">Raw gradient</td>
<td style="padding:8px;">Simple, good generalization</td>
<td style="padding:8px;">Slow, sensitive to \\(\\eta\\)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">SGD+Momentum</td>
<td style="padding:8px;">EMA of gradients</td>
<td style="padding:8px;">Faster, dampens oscillations</td>
<td style="padding:8px;">Extra hyperparameter \\(\\beta\\)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">Nesterov</td>
<td style="padding:8px;">Lookahead gradient</td>
<td style="padding:8px;">Optimal rate for convex</td>
<td style="padding:8px;">Marginal gain in practice</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">AdaGrad</td>
<td style="padding:8px;">Per-param \\(\\sum g^2\\)</td>
<td style="padding:8px;">Great for sparse features</td>
<td style="padding:8px;">LR decays to zero</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">RMSProp</td>
<td style="padding:8px;">Per-param EMA of \\(g^2\\)</td>
<td style="padding:8px;">Fixes AdaGrad decay</td>
<td style="padding:8px;">No momentum</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">Adam</td>
<td style="padding:8px;">Momentum + adaptive</td>
<td style="padding:8px;">Works out of the box</td>
<td style="padding:8px;">L2 reg interaction</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px;color:#f0f6fc;">AdamW</td>
<td style="padding:8px;">Decoupled weight decay</td>
<td style="padding:8px;">Standard for transformers</td>
<td style="padding:8px;">Slightly more tuning</td>
</tr>
</table>

<h2>Practical Guidelines</h2>

<div class="env-block remark"><div class="env-title">When to Use What</div><div class="env-body"><p><strong>Transformers / LLMs / NLP</strong>: AdamW with linear warmup + cosine decay. This is essentially universal for modern language models and vision transformers.</p>
<p><strong>CNNs (image classification)</strong>: SGD with momentum (0.9) + step decay or cosine annealing. This remains the recipe behind most ImageNet SOTA results from ResNet through ConvNeXt.</p>
<p><strong>GANs</strong>: Adam with \\(\\beta_1 = 0.0\\) or \\(0.5\\) (reduced momentum helps stability). Separate learning rates for generator and discriminator.</p>
<p><strong>Sparse data / embeddings</strong>: Adam or AdaGrad. The per-parameter adaptivity handles the frequency imbalance automatically.</p>
<p><strong>Small dataset / fine-tuning</strong>: AdamW with small \\(\\eta\\) and warmup. Weight decay acts as regularization.</p></div></div>

<div class="env-block remark"><div class="env-title">Hyperparameter Tuning Priority</div><div class="env-body"><p>If you can tune only one hyperparameter, tune the <strong>learning rate</strong>. It has the largest impact on training dynamics. A reasonable search grid: \\(\\{3 \\times 10^{-4},\\, 10^{-3},\\, 3 \\times 10^{-3},\\, 10^{-2}\\}\\) for Adam; \\(\\{0.01,\\, 0.03,\\, 0.1,\\, 0.3\\}\\) for SGD. The second most important is the <strong>weight decay</strong> (\\(\\lambda\\)): try \\(\\{0,\\, 10^{-4},\\, 10^{-2},\\, 0.1\\}\\). Batch size and momentum rarely need tuning from defaults.</p></div></div>

<div class="env-block warning"><div class="env-title">Common Pitfalls</div><div class="env-body"><p><strong>Forgetting bias correction</strong>: Implementing Adam without bias correction produces incorrect step sizes in early training.</p>
<p><strong>Confusing L2 and weight decay with Adam</strong>: Use AdamW, not Adam + L2. The difference matters in practice.</p>
<p><strong>No warmup for transformers</strong>: Training diverges in the first few hundred steps.</p>
<p><strong>Too-large batch size without LR scaling</strong>: The "linear scaling rule" (Goyal et al., 2017) suggests scaling \\(\\eta\\) proportionally to batch size, but this breaks down for very large batches.</p></div></div>

<div class="viz-placeholder" data-viz="ch07-viz06"></div>
`,
            visualizations: [
                {
                    id: 'ch07-viz06',
                    title: 'Optimizer Race',
                    description: 'All optimizers racing simultaneously on the Beale function, a challenging non-convex surface with a narrow valley. Compare their trajectories and convergence speed.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, { scale: 80, originX: 260, originY: 280, width: 600, height: 440 });
                        var ctx = viz.ctx;
                        var lr = 0.002;
                        var running = false;
                        var animId = null;
                        var stepCount = 0;

                        // Beale function: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
                        // Minimum at (3, 0.5)
                        // Use a scaled/shifted version to keep things in view
                        function beale(x, y) {
                            var a = 1.5 - x + x * y;
                            var b = 2.25 - x + x * y * y;
                            var c = 2.625 - x + x * y * y * y;
                            return a * a + b * b + c * c;
                        }
                        function bealeGrad(x, y) {
                            var a = 1.5 - x + x * y;
                            var b = 2.25 - x + x * y * y;
                            var c = 2.625 - x + x * y * y * y;
                            var dadx = -1 + y;
                            var dady = x;
                            var dbdx = -1 + y * y;
                            var dbdy = 2 * x * y;
                            var dcdx = -1 + y * y * y;
                            var dcdy = 3 * x * y * y;
                            return [
                                2 * a * dadx + 2 * b * dbdx + 2 * c * dcdx,
                                2 * a * dady + 2 * b * dbdy + 2 * c * dcdy
                            ];
                        }

                        var start = [-1.5, -1.5];
                        var opts = {};

                        function initAll() {
                            stepCount = 0;
                            opts = {
                                sgd: { name: 'SGD', pos: start.slice(), path: [start.slice()], color: viz.colors.red },
                                mom: { name: 'Momentum', pos: start.slice(), vel: [0,0], path: [start.slice()], color: viz.colors.blue },
                                nes: { name: 'Nesterov', pos: start.slice(), vel: [0,0], path: [start.slice()], color: viz.colors.purple },
                                rms: { name: 'RMSProp', pos: start.slice(), Eg2: [0,0], path: [start.slice()], color: viz.colors.yellow },
                                adam: { name: 'Adam', pos: start.slice(), m: [0,0], v: [0,0], t: 0, path: [start.slice()], color: viz.colors.teal }
                            };
                        }
                        initAll();

                        function clamp(g, mx) {
                            var n = Math.sqrt(g[0]*g[0]+g[1]*g[1]);
                            if (n > mx) { g[0] *= mx/n; g[1] *= mx/n; }
                            return g;
                        }

                        function stepAll() {
                            var maxG = 30, eps = 1e-8, b1 = 0.9, b2 = 0.999;

                            // SGD
                            var gS = clamp(bealeGrad(opts.sgd.pos[0], opts.sgd.pos[1]), maxG);
                            opts.sgd.pos[0] -= lr * gS[0];
                            opts.sgd.pos[1] -= lr * gS[1];
                            opts.sgd.path.push(opts.sgd.pos.slice());

                            // Momentum
                            var gM = clamp(bealeGrad(opts.mom.pos[0], opts.mom.pos[1]), maxG);
                            opts.mom.vel[0] = 0.9 * opts.mom.vel[0] + gM[0];
                            opts.mom.vel[1] = 0.9 * opts.mom.vel[1] + gM[1];
                            opts.mom.pos[0] -= lr * opts.mom.vel[0];
                            opts.mom.pos[1] -= lr * opts.mom.vel[1];
                            opts.mom.path.push(opts.mom.pos.slice());

                            // Nesterov
                            var lookX = opts.nes.pos[0] - lr * 0.9 * opts.nes.vel[0];
                            var lookY = opts.nes.pos[1] - lr * 0.9 * opts.nes.vel[1];
                            var gN = clamp(bealeGrad(lookX, lookY), maxG);
                            opts.nes.vel[0] = 0.9 * opts.nes.vel[0] + gN[0];
                            opts.nes.vel[1] = 0.9 * opts.nes.vel[1] + gN[1];
                            opts.nes.pos[0] -= lr * opts.nes.vel[0];
                            opts.nes.pos[1] -= lr * opts.nes.vel[1];
                            opts.nes.path.push(opts.nes.pos.slice());

                            // RMSProp
                            var gR = clamp(bealeGrad(opts.rms.pos[0], opts.rms.pos[1]), maxG);
                            opts.rms.Eg2[0] = 0.9 * opts.rms.Eg2[0] + 0.1 * gR[0] * gR[0];
                            opts.rms.Eg2[1] = 0.9 * opts.rms.Eg2[1] + 0.1 * gR[1] * gR[1];
                            opts.rms.pos[0] -= lr * gR[0] / (Math.sqrt(opts.rms.Eg2[0]) + eps);
                            opts.rms.pos[1] -= lr * gR[1] / (Math.sqrt(opts.rms.Eg2[1]) + eps);
                            opts.rms.path.push(opts.rms.pos.slice());

                            // Adam
                            opts.adam.t++;
                            var gA = clamp(bealeGrad(opts.adam.pos[0], opts.adam.pos[1]), maxG);
                            opts.adam.m[0] = b1 * opts.adam.m[0] + (1 - b1) * gA[0];
                            opts.adam.m[1] = b1 * opts.adam.m[1] + (1 - b1) * gA[1];
                            opts.adam.v[0] = b2 * opts.adam.v[0] + (1 - b2) * gA[0] * gA[0];
                            opts.adam.v[1] = b2 * opts.adam.v[1] + (1 - b2) * gA[1] * gA[1];
                            var mh = [opts.adam.m[0]/(1-Math.pow(b1,opts.adam.t)), opts.adam.m[1]/(1-Math.pow(b1,opts.adam.t))];
                            var vh = [opts.adam.v[0]/(1-Math.pow(b2,opts.adam.t)), opts.adam.v[1]/(1-Math.pow(b2,opts.adam.t))];
                            opts.adam.pos[0] -= lr * mh[0] / (Math.sqrt(vh[0]) + eps);
                            opts.adam.pos[1] -= lr * mh[1] / (Math.sqrt(vh[1]) + eps);
                            opts.adam.path.push(opts.adam.pos.slice());

                            stepCount++;
                        }

                        function drawContours() {
                            var levels = [0.5, 2, 8, 25, 80, 200, 500, 1500];
                            var xMin = -3, xMax = 5, yMin = -3, yMax = 4;
                            var res = 2;
                            var cStep = 0.06;
                            for (var li = 0; li < levels.length; li++) {
                                var lev = levels[li];
                                ctx.fillStyle = 'rgba(88,166,255,' + (0.08 + 0.04 * li) + ')';
                                for (var px = xMin; px < xMax; px += cStep) {
                                    for (var py = yMin; py < yMax; py += cStep) {
                                        var v00 = beale(px, py) - lev;
                                        var v10 = beale(px + cStep, py) - lev;
                                        var v01 = beale(px, py + cStep) - lev;
                                        if (v00 * v10 < 0) {
                                            var frac = v00 / (v00 - v10);
                                            var sp = viz.toScreen(px + frac * cStep, py);
                                            ctx.fillRect(sp[0], sp[1], res, res);
                                        }
                                        if (v00 * v01 < 0) {
                                            var frac2 = v00 / (v00 - v01);
                                            var sp2 = viz.toScreen(px, py + frac2 * cStep);
                                            ctx.fillRect(sp2[0], sp2[1], res, res);
                                        }
                                    }
                                }
                            }
                        }

                        function drawPath(o) {
                            var p = o.path;
                            if (p.length < 2) return;
                            ctx.strokeStyle = o.color;
                            ctx.lineWidth = 1.5;
                            ctx.globalAlpha = 0.7;
                            ctx.beginPath();
                            var maxDraw = Math.min(p.length, 400);
                            var si = p.length - maxDraw;
                            for (var i = si; i < p.length; i++) {
                                var sp = viz.toScreen(p[i][0], p[i][1]);
                                i === si ? ctx.moveTo(sp[0], sp[1]) : ctx.lineTo(sp[0], sp[1]);
                            }
                            ctx.stroke();
                            ctx.globalAlpha = 1;
                            viz.drawPoint(p[p.length-1][0], p[p.length-1][1], o.color, '', 4);
                        }

                        function draw() {
                            viz.clear();
                            drawContours();

                            // Minimum at (3, 0.5)
                            viz.drawPoint(3, 0.5, viz.colors.green, '  min(3, 0.5)', 5);

                            // Draw all paths
                            var keys = ['sgd','mom','nes','rms','adam'];
                            for (var k = 0; k < keys.length; k++) drawPath(opts[keys[k]]);

                            // Legend + loss values
                            ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            var lx = 10, ly = 14;
                            for (var j = 0; j < keys.length; j++) {
                                var o = opts[keys[j]];
                                var lv = beale(o.pos[0], o.pos[1]);
                                ctx.fillStyle = o.color;
                                ctx.fillRect(lx, ly + j * 16 - 4, 12, 3);
                                ctx.fillText(o.name + ': ' + (lv < 1e4 ? lv.toFixed(2) : lv.toExponential(1)), lx + 16, ly + j * 16);
                            }

                            viz.screenText('Beale Function   |   Step: ' + stepCount, viz.width / 2, viz.height - 12, viz.colors.text, 11);
                        }

                        function animate() {
                            if (!running) return;
                            for (var i = 0; i < 3; i++) stepAll();
                            draw();
                            if (stepCount > 5000) running = false;
                            animId = requestAnimationFrame(animate);
                        }

                        draw();

                        VizEngine.createSlider(controls, '\u03B7', 0.0001, 0.01, lr, 0.0001, function(v) { lr = v; });
                        VizEngine.createButton(controls, 'Race!', function() { if (!running) { running = true; animate(); } });
                        VizEngine.createButton(controls, 'Step x10', function() { running = false; for (var i = 0; i < 10; i++) stepAll(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; if (animId) cancelAnimationFrame(animId); initAll(); draw(); });

                        return { stopAnimation: function() { running = false; if (animId) cancelAnimationFrame(animId); } };
                    }
                }
            ],
            exercises: [
                {
                    id: 'ch07-ex17',
                    type: 'conceptual',
                    question: 'In the optimizer race visualization, which optimizer typically reaches the minimum first? Why might this change with different learning rates or starting points?',
                    hint: 'Consider each optimizer\'s strengths: adaptive step sizes vs momentum vs both.',
                    solution: 'Adam typically reaches the minimum fastest due to its combination of momentum (to build up speed in consistent gradient directions) and adaptive per-parameter scaling (to handle the varying curvature of the Beale function). However, with a carefully tuned learning rate, SGD with momentum can match or beat Adam. The relative performance depends on the loss landscape geometry near the start point: in high-curvature regions, adaptive methods have a large advantage; in regions where the gradient is roughly constant, momentum methods are equally effective. This illustrates why there is no universally "best" optimizer.'
                },
                {
                    id: 'ch07-ex18',
                    type: 'conceptual',
                    question: 'A colleague says: "Adam is always better than SGD because it is more sophisticated." Provide a counterargument with a concrete scenario where SGD outperforms Adam.',
                    hint: 'Think about generalization, not just training loss.',
                    solution: 'Image classification with ResNets is the classic counterexample. Multiple studies (Wilson et al., 2017; Keskar & Socher, 2017) show that SGD with momentum achieves 1-2% higher test accuracy than Adam on CIFAR-10/ImageNet, despite Adam reaching lower training loss. The hypothesis: Adam\'s per-parameter adaptivity allows different parts of the network to converge at different rates, which can lead to "co-adaptation" and sharper minima that generalize worse. SGD\'s uniform learning rate acts as implicit regularization, keeping all parameters on a similar optimization trajectory. The lesson: training loss is not the goal; test performance is.'
                },
                {
                    id: 'ch07-ex19',
                    type: 'numeric',
                    question: 'You are training a transformer with AdamW (\\(\\eta = 3 \\times 10^{-4}\\), \\(\\lambda = 0.01\\), \\(\\beta_1 = 0.9\\), \\(\\beta_2 = 0.999\\)) for \\(T = 10{,}000\\) steps with 1000 warmup steps and cosine decay to 0. What is the learning rate at step 500? At step 5000?',
                    hint: 'Step 500 is in warmup (linear). Step 5000 is in cosine decay.',
                    solution: 'At step 500 (warmup phase): \\(\\eta_{500} = 3 \\times 10^{-4} \\times 500/1000 = 1.5 \\times 10^{-4}\\). At step 5000 (cosine phase): the cosine phase runs from step 1000 to 10000, so the progress is \\((5000 - 1000)/(10000 - 1000) = 4000/9000 \\approx 0.444\\). \\(\\eta_{5000} = \\frac{3 \\times 10^{-4}}{2}(1 + \\cos(0.444\\pi)) = 1.5 \\times 10^{-4} \\times (1 + \\cos(1.396)) = 1.5 \\times 10^{-4} \\times (1 + 0.174) \\approx 1.76 \\times 10^{-4}\\).'
                },
                {
                    id: 'ch07-ex20',
                    type: 'conceptual',
                    question: 'Explain the "linear scaling rule" for large-batch training: when you multiply the batch size by \\(k\\), you should multiply the learning rate by \\(k\\). Under what assumption does this hold, and when does it break down?',
                    hint: 'Think about the gradient variance and the effective step per sample.',
                    solution: 'The linear scaling rule (Goyal et al., 2017) comes from matching the expected parameter change per epoch. With batch size \\(B\\) and LR \\(\\eta\\), one epoch has \\(N/B\\) steps, each updating by \\(\\eta g_B\\). With batch size \\(kB\\) and LR \\(k\\eta\\), one epoch has \\(N/(kB)\\) steps, each updating by \\(k\\eta g_{kB}\\). Since \\(\\mathbb{E}[g_B] = \\mathbb{E}[g_{kB}]\\), the expected total update per epoch is the same. The rule assumes the loss surface is approximately linear over the region traversed by \\(k\\) consecutive small-batch steps (the "linear regime"). It breaks down when: (1) the batch size exceeds \\(B_{\\text{crit}}\\) (the noise is negligible, so further scaling just wastes compute), (2) the learning rate becomes so large that the linear approximation fails (the optimizer overshoots), or (3) batch normalization statistics change with batch size.'
                }
            ]
        }
    ]
});
