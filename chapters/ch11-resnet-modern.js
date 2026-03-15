// ============================================================
// Chapter 11 · ResNet & Modern Architectures
// Residual Learning, Skip Connections, and the Evolution of CNNs
// ============================================================
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch11',
    number: 11,
    title: 'ResNet & Modern Architectures',
    subtitle: 'Residual Learning, Skip Connections, and the Evolution of CNNs',
    sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: The Degradation Problem
    // ─────────────────────────────────────────────────────────────────────────
    {
        id: 'ch11-sec01',
        title: 'The Degradation Problem',
        content: `
<h2>The Degradation Problem</h2>

<div class="env-block intuition">
<div class="env-title">Deeper Should Mean Better... Right?</div>
<div class="env-body">
<p>By the end of Chapter 10, a clear pattern had emerged across LeNet, AlexNet, VGGNet, and GoogLeNet: deeper networks learn richer feature hierarchies. Low layers capture edges and textures; middle layers capture parts and motifs; deep layers capture high-level semantics. The natural impulse was to keep stacking layers. But a surprising empirical finding halted that intuition: beyond a certain depth, <strong>training accuracy itself begins to degrade</strong>. Not just test accuracy (which could be explained by overfitting), but training accuracy. Something more fundamental was going wrong.</p>
</div>
</div>

<p>In 2015, Kaiming He and colleagues at Microsoft Research documented this phenomenon carefully in their landmark paper "Deep Residual Learning for Image Recognition." They trained plain (non-residual) networks on CIFAR-10 with 20 and 56 layers. The 56-layer network had <em>higher training error</em> than the 20-layer network. This is the <strong>degradation problem</strong>.</p>

<div class="env-block definition">
<div class="env-title">The Degradation Problem</div>
<div class="env-body">
<p>When deeper plain networks are trained, both training and test error increase compared to shallower counterparts. This is <em>not</em> caused by overfitting (which would show low training error but high test error). Instead, the deeper network fails to learn even what the shallower network can learn.</p>
</div>
</div>

<h3>Why Deeper Should Be At Least As Good</h3>

<p>Consider a thought experiment. Suppose a 20-layer network achieves some solution. A 56-layer network could, in principle, copy those 20 layers exactly and set the remaining 36 layers to identity mappings (each layer outputs its input unchanged). This "construction by identity" would achieve at least the same training error. Yet in practice, gradient-based optimizers cannot discover this solution.</p>

<div class="env-block theorem">
<div class="env-title">The Identity Mapping Argument</div>
<div class="env-body">
<p>Let \\(f^{*}_{20}\\) be the optimal 20-layer mapping. A 56-layer network contains the hypothesis class of the 20-layer network as a subset (by setting extra layers to identity). Therefore:</p>
\\[\\min_{\\theta_{56}} \\mathcal{L}(\\theta_{56}) \\le \\min_{\\theta_{20}} \\mathcal{L}(\\theta_{20})\\]
<p>The 56-layer network's optimal error is at most the 20-layer's optimal error. But optimizers fail to reach this optimum, revealing a fundamental <strong>optimization difficulty</strong>, not a representation limitation.</p>
</div>
</div>

<h3>Root Causes</h3>

<p>The degradation problem stems from multiple interacting factors:</p>

<ul>
<li><strong>Vanishing/exploding gradients</strong>: Even with BatchNorm and careful initialization (Chapter 6), gradients across many layers can become extremely small or large, making optimization unreliable.</li>
<li><strong>Optimization landscape</strong>: The loss surface of very deep plain networks contains pathological regions, saddle points, and flat plateaus that trap SGD-based optimizers.</li>
<li><strong>Difficulty of learning identity</strong>: For a layer computing \\(\\mathcal{H}(\\mathbf{x}) = \\mathbf{x}\\), the weights must be driven to a specific configuration (identity matrix with zero bias). Nonlinear activation functions like ReLU make exact identity impossible; the network can only approximate it.</li>
</ul>

<div class="env-block warning">
<div class="env-title">Degradation vs. Overfitting</div>
<div class="env-body">
<p>Be careful not to confuse these two phenomena. Overfitting manifests as low training error but high test error (the model memorizes noise). Degradation manifests as high training error in deeper networks. They are fundamentally different problems requiring different solutions. Regularization (Chapter 8) addresses overfitting; residual connections (this chapter) address degradation.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="ch11-viz-degradation"></div>

<div class="env-block remark">
<div class="env-title">Historical Context</div>
<div class="env-body">
<p>The degradation problem was not entirely unknown before ResNet. Highway Networks (Srivastava et al., 2015) proposed gating mechanisms to ease training of very deep networks. But ResNet's elegant simplicity, its strong empirical results (winning ILSVRC 2015 with a 152-layer network at 3.57% top-5 error), and its theoretical clarity made residual learning the dominant paradigm. ResNet has been cited over 200,000 times, making it one of the most influential papers in all of computer science.</p>
</div>
</div>`,
        visualizations: [
            {
                id: 'ch11-viz-degradation',
                title: 'The Degradation Problem: Depth vs. Accuracy',
                description: 'Plain networks degrade with depth. The shallower network (20 layers) outperforms the deeper one (56 layers) on both training and test error. Adjust depth to see the non-monotonic accuracy curve.',
                setup(container, controls) {
                    const viz = new VizEngine(container, { width: 700, height: 420, scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx;
                    const W = viz.width, H = viz.height;
                    const pad = { left: 75, right: 30, top: 45, bottom: 55 };
                    const plotW = W - pad.left - pad.right;
                    const plotH = H - pad.top - pad.bottom;

                    // Simulated accuracy curves
                    // Plain network: accuracy peaks around 20 layers, then degrades
                    function plainTrainAcc(depth) {
                        if (depth <= 1) return 0.55;
                        const peak = 20;
                        const rise = 0.95 - 0.55 * Math.exp(-0.15 * depth);
                        const degrade = 1 - 0.003 * Math.max(0, depth - peak) * Math.max(0, depth - peak) * 0.02;
                        return Math.max(0.6, Math.min(0.97, rise * degrade));
                    }
                    function plainTestAcc(depth) {
                        return Math.max(0.55, plainTrainAcc(depth) - 0.015 - 0.0004 * depth * depth * 0.01);
                    }
                    // ResNet: accuracy keeps improving (or plateaus) with depth
                    function resnetTrainAcc(depth) {
                        if (depth <= 1) return 0.55;
                        return Math.min(0.995, 0.55 + 0.44 * (1 - Math.exp(-0.08 * depth)));
                    }
                    function resnetTestAcc(depth) {
                        return Math.min(0.97, 0.55 + 0.41 * (1 - Math.exp(-0.065 * depth)));
                    }

                    let showResnet = false;

                    function depthToX(d) { return pad.left + (d / 80) * plotW; }
                    function accToY(a) { return pad.top + plotH - ((a - 0.5) / 0.55) * plotH; }

                    function draw() {
                        viz.clear();

                        // Title
                        viz.screenText('Depth vs. Classification Accuracy', W / 2, 22, viz.colors.white, 15);

                        // Axes
                        ctx.strokeStyle = viz.colors.axis;
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        ctx.moveTo(pad.left, pad.top);
                        ctx.lineTo(pad.left, pad.top + plotH);
                        ctx.lineTo(pad.left + plotW, pad.top + plotH);
                        ctx.stroke();

                        // X-axis labels
                        ctx.fillStyle = viz.colors.text;
                        ctx.font = '11px -apple-system,sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'top';
                        for (let d = 0; d <= 80; d += 10) {
                            const x = depthToX(d);
                            ctx.fillText(d, x, pad.top + plotH + 8);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
                        }
                        viz.screenText('Number of Layers', W / 2, H - 10, viz.colors.text, 12);

                        // Y-axis labels
                        ctx.textAlign = 'right';
                        ctx.textBaseline = 'middle';
                        for (let a = 0.5; a <= 1.0; a += 0.05) {
                            const y = accToY(a);
                            if (y < pad.top - 5 || y > pad.top + plotH + 5) continue;
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText((a * 100).toFixed(0) + '%', pad.left - 8, y);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 0.5;
                            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
                        }
                        ctx.save();
                        ctx.translate(15, pad.top + plotH / 2);
                        ctx.rotate(-Math.PI / 2);
                        ctx.fillStyle = viz.colors.text;
                        ctx.font = '12px -apple-system,sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText('Accuracy', 0, 0);
                        ctx.restore();

                        // Draw plain network curves
                        function drawCurve(fn, color, lw, dashed) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = lw;
                            if (dashed) ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            for (let d = 1; d <= 80; d++) {
                                const x = depthToX(d), y = accToY(fn(d));
                                if (d === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            }
                            ctx.stroke();
                            if (dashed) ctx.setLineDash([]);
                        }

                        drawCurve(plainTrainAcc, viz.colors.blue, 2.5, false);
                        drawCurve(plainTestAcc, viz.colors.blue, 2, true);

                        if (showResnet) {
                            drawCurve(resnetTrainAcc, viz.colors.green, 2.5, false);
                            drawCurve(resnetTestAcc, viz.colors.green, 2, true);
                        }

                        // Mark degradation region
                        ctx.fillStyle = 'rgba(248, 81, 73, 0.08)';
                        const degStart = depthToX(25);
                        ctx.fillRect(degStart, pad.top, pad.left + plotW - degStart, plotH);
                        viz.screenText('Degradation Zone', (degStart + pad.left + plotW) / 2, pad.top + 18, viz.colors.red, 11);

                        // Annotations for 20 vs 56 layer
                        const x20 = depthToX(20), y20train = accToY(plainTrainAcc(20));
                        const x56 = depthToX(56), y56train = accToY(plainTrainAcc(56));
                        ctx.fillStyle = viz.colors.orange;
                        ctx.beginPath(); ctx.arc(x20, y20train, 5, 0, Math.PI * 2); ctx.fill();
                        ctx.beginPath(); ctx.arc(x56, y56train, 5, 0, Math.PI * 2); ctx.fill();
                        viz.screenText('20-layer', x20, y20train - 14, viz.colors.orange, 11);
                        viz.screenText('56-layer', x56, y56train - 14, viz.colors.orange, 11);

                        // Legend
                        const lx = pad.left + plotW - 180, ly = pad.top + 40;
                        ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5;
                        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 24, ly); ctx.stroke();
                        viz.screenText('Plain train', lx + 30, ly, viz.colors.blue, 10, 'left');

                        ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2; ctx.setLineDash([6,4]);
                        ctx.beginPath(); ctx.moveTo(lx, ly + 16); ctx.lineTo(lx + 24, ly + 16); ctx.stroke();
                        ctx.setLineDash([]);
                        viz.screenText('Plain test', lx + 30, ly + 16, viz.colors.blue, 10, 'left');

                        if (showResnet) {
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2.5;
                            ctx.beginPath(); ctx.moveTo(lx, ly + 34); ctx.lineTo(lx + 24, ly + 34); ctx.stroke();
                            viz.screenText('ResNet train', lx + 30, ly + 34, viz.colors.green, 10, 'left');

                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2; ctx.setLineDash([6,4]);
                            ctx.beginPath(); ctx.moveTo(lx, ly + 50); ctx.lineTo(lx + 24, ly + 50); ctx.stroke();
                            ctx.setLineDash([]);
                            viz.screenText('ResNet test', lx + 30, ly + 50, viz.colors.green, 10, 'left');
                        }
                    }

                    draw();

                    VizEngine.createButton(controls, showResnet ? 'Hide ResNet' : 'Show ResNet', function() {
                        showResnet = !showResnet;
                        this.textContent = showResnet ? 'Hide ResNet' : 'Show ResNet';
                        draw();
                    });

                    return viz;
                }
            }
        ],
        exercises: [
            {
                question: 'A 56-layer plain CNN achieves 8.2% training error on CIFAR-10, while a 20-layer plain CNN achieves 6.5% training error. Is this overfitting? Explain why or why not, and identify the correct term for this phenomenon.',
                hint: 'Overfitting is characterized by low training error but high test error. What characterizes the degradation problem?',
                solution: 'This is <em>not</em> overfitting. Overfitting would manifest as very low training error but high test error (the model memorizes training data but fails to generalize). Here, the deeper network has <em>higher training error</em>, meaning it fails to fit even the training data as well as the shallower network. This is the <strong>degradation problem</strong>: an optimization difficulty where deeper plain networks cannot learn what shallower networks can learn, despite having strictly greater representational capacity.'
            },
            {
                question: 'Explain the "construction by identity" argument. If a 20-layer network achieves error \\(\\epsilon\\), why must the optimal 56-layer network achieve error \\(\\le \\epsilon\\)? What does the failure of this bound in practice tell us?',
                hint: 'Think about what the extra 36 layers could do if they simply passed their input through unchanged.',
                solution: 'The argument proceeds by construction: take the learned 20-layer solution and append 36 identity layers \\(\\mathbf{y} = \\mathbf{x}\\). This 56-layer network reproduces the 20-layer solution exactly, so its error equals \\(\\epsilon\\). Since this is just one point in the 56-layer hypothesis space, the optimal must be \\(\\le \\epsilon\\). The fact that SGD-trained 56-layer networks achieve error \\(&gt; \\epsilon\\) proves the problem is not representational but <strong>optimizational</strong>: gradient-based methods cannot navigate the loss landscape of very deep plain networks to find solutions that are provably reachable.'
            },
            {
                question: 'List three distinct mechanisms that contribute to the degradation problem. For each, briefly explain how it impedes training of very deep networks.',
                hint: 'Consider gradient flow, the shape of the loss landscape, and the difficulty of learning particular functions.',
                solution: '<ol><li><strong>Vanishing/exploding gradients</strong>: Through the chain rule, gradients are products of many Jacobians. Even with BatchNorm, accumulated numerical errors across 50+ layers make gradient signals unreliable, causing weights in early layers to barely update.</li><li><strong>Pathological loss landscape</strong>: Very deep networks have loss surfaces with numerous saddle points, plateaus, and narrow ravines. SGD can get trapped in these regions, unable to descend toward good solutions that provably exist.</li><li><strong>Difficulty of learning identity</strong>: For a nonlinear layer with ReLU to approximate \\(\\mathcal{H}(\\mathbf{x}) = \\mathbf{x}\\), it needs specific weight configurations. ReLU zeroes out negative inputs, making exact identity impossible. The optimizer must drive weights toward an unusual configuration that general-purpose initialization and learning dynamics do not favor.</li></ol>'
            }
        ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Residual Learning & Skip Connections
    // ─────────────────────────────────────────────────────────────────────────
    {
        id: 'ch11-sec02',
        title: 'Residual Learning & Skip Connections',
        content: `
<h2>Residual Learning & Skip Connections</h2>

<div class="env-block intuition">
<div class="env-title">The Key Insight</div>
<div class="env-body">
<p>If learning the identity mapping is hard, why not make it the <em>default</em>? Instead of asking a stack of layers to learn the desired mapping \\(\\mathcal{H}(\\mathbf{x})\\) directly, we ask them to learn only the <strong>residual</strong> \\(\\mathcal{F}(\\mathbf{x}) = \\mathcal{H}(\\mathbf{x}) - \\mathbf{x}\\). The output then becomes \\(\\mathcal{F}(\\mathbf{x}) + \\mathbf{x}\\). If the optimal transformation is close to identity, the network only needs to push \\(\\mathcal{F}\\) toward zero, which is far easier than pushing a full mapping toward identity.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Residual Learning Formulation</div>
<div class="env-body">
<p>A <strong>residual block</strong> computes:</p>
\\[\\mathbf{y} = \\mathcal{F}(\\mathbf{x}, \\{W_i\\}) + \\mathbf{x}\\]
<p>where \\(\\mathcal{F}(\\mathbf{x}, \\{W_i\\})\\) is the residual function learned by two or more stacked layers, and the \\(+ \\mathbf{x}\\) is the <strong>skip connection</strong> (also called a shortcut connection or identity shortcut). The dimensions of \\(\\mathcal{F}\\) and \\(\\mathbf{x}\\) must match; when they differ, a linear projection \\(W_s\\) is applied:</p>
\\[\\mathbf{y} = \\mathcal{F}(\\mathbf{x}, \\{W_i\\}) + W_s \\mathbf{x}\\]
</div>
</div>

<h3>Why Residual Learning Works</h3>

<p>The advantages of residual learning operate through several complementary mechanisms:</p>

<h4>1. Easy Identity Mapping</h4>
<p>If the optimal mapping at some layer is identity (\\(\\mathcal{H}(\\mathbf{x}) = \\mathbf{x}\\)), then the residual \\(\\mathcal{F}(\\mathbf{x}) = 0\\). Driving weights to zero is trivial with weight decay. In a plain network, learning identity requires the weights to form an identity matrix, which is a much harder optimization target.</p>

<h4>2. Gradient Highway</h4>
<p>Consider the gradient flow during backpropagation. For a residual block \\(\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}\\):</p>
\\[\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} = \\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}} + \\mathbf{I}\\]
<p>The identity matrix \\(\\mathbf{I}\\) ensures that gradients always have a direct path back through the skip connection, regardless of what \\(\\mathcal{F}\\) does. Even if \\(\\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}}\\) vanishes, the gradient through \\(\\mathbf{I}\\) remains intact.</p>

<div class="env-block theorem">
<div class="env-title">Gradient Flow in Deep Residual Networks</div>
<div class="env-body">
<p>For a network with \\(L\\) residual blocks, the gradient of the loss \\(\\mathcal{L}\\) with respect to a feature \\(\\mathbf{x}_l\\) at layer \\(l\\) is:</p>
\\[\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{x}_l} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{x}_L} \\left(1 + \\frac{\\partial}{\\partial \\mathbf{x}_l} \\sum_{i=l}^{L-1} \\mathcal{F}_i(\\mathbf{x}_i)\\right)\\]
<p>The key term is the \\(1\\) inside the parentheses: it means the gradient signal is <strong>never multiplicatively attenuated</strong> by the intermediate layers. This is a sum rather than a product, which prevents the exponential decay (vanishing) or explosion that plagues plain networks.</p>
</div>
</div>

<h4>3. Ensemble Interpretation</h4>
<p>Veit et al. (2016) showed that residual networks can be viewed as an exponential ensemble of many shorter paths. An \\(n\\)-block ResNet has \\(2^n\\) paths of varying lengths (each block's skip connection creates a binary choice: go through \\(\\mathcal{F}\\) or skip). The network's robustness comes from this implicit ensemble: deleting individual blocks has limited impact because many alternative paths remain.</p>

<div class="env-block example">
<div class="env-title">Concrete Gradient Comparison</div>
<div class="env-body">
<p>Consider a 50-layer network where each layer's Jacobian has spectral norm 0.9 (slight contraction).</p>
<p><strong>Plain network</strong>: gradient magnitude \\(\\propto 0.9^{50} \\approx 0.005\\). The signal reaching layer 1 is 200x weaker than at layer 50.</p>
<p><strong>Residual network</strong>: each block's Jacobian is \\(\\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}} + \\mathbf{I}\\), so the spectral norm is at least 1 (from the identity). Gradient magnitude stays \\(\\mathcal{O}(1)\\) throughout the network.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="ch11-viz-gradient-flow"></div>

<div class="env-block remark">
<div class="env-title">Pre-Activation vs. Post-Activation ResNet</div>
<div class="env-body">
<p>He et al. (2016) later showed that placing BatchNorm and ReLU <em>before</em> the weight layers (pre-activation) further improves gradient flow. The original design applies BN and ReLU after addition, which means the skip connection passes through a nonlinearity, partially breaking the clean identity path. The pre-activation design keeps the skip connection as a pure identity, achieving cleaner gradient propagation and better results on very deep networks (1001 layers on CIFAR-10).</p>
</div>
</div>`,
        visualizations: [
            {
                id: 'ch11-viz-gradient-flow',
                title: 'Gradient Flow: Plain vs. Residual Networks',
                description: 'Compare gradient magnitude across layers. In plain networks, gradients vanish exponentially. In residual networks, the skip connection maintains gradient flow. Adjust the per-layer Jacobian norm to see the effect.',
                setup(container, controls) {
                    const viz = new VizEngine(container, { width: 720, height: 440, scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx;
                    const W = viz.width, H = viz.height;
                    const pad = { left: 60, right: 30, top: 50, bottom: 50 };
                    const plotW = W - pad.left - pad.right;
                    const plotH = H - pad.top - pad.bottom;
                    const numLayers = 50;

                    let jacobianNorm = 0.9;

                    function draw() {
                        viz.clear();

                        viz.screenText('Gradient Magnitude per Layer', W / 2, 22, viz.colors.white, 15);

                        // Compute gradient magnitudes
                        const plainGrad = [];
                        const resGrad = [];
                        for (let i = 0; i < numLayers; i++) {
                            // Plain: gradient decays as jacobianNorm^(numLayers - i)
                            const depth = numLayers - i;
                            plainGrad.push(Math.pow(jacobianNorm, depth));
                            // Residual: gradient through skip connection stays ~1,
                            // combined with residual path. Simplified model:
                            // each block contributes (jacobianNorm + 1) effectively,
                            // but the direct skip gives a floor of 1.
                            // More realistic: gradient = 1 + accumulated residual contribution
                            resGrad.push(1.0 + 0.1 * Math.pow(jacobianNorm, depth / 3));
                        }

                        // Normalize for display: use log scale for plain, linear for comparison
                        const maxGrad = Math.max(...resGrad, 1.2);

                        // Draw as heatmap bars
                        const barW = plotW / numLayers;

                        // Plain network (top half)
                        const halfH = (plotH - 30) / 2;
                        const topY = pad.top;
                        const botY = pad.top + halfH + 30;

                        viz.screenText('Plain Network', pad.left + plotW / 2, topY - 5, viz.colors.blue, 12);
                        viz.screenText('Residual Network', pad.left + plotW / 2, botY - 5, viz.colors.green, 12);

                        for (let i = 0; i < numLayers; i++) {
                            const x = pad.left + i * barW;

                            // Plain network bar (log-scale coloring)
                            const plainVal = plainGrad[i];
                            const plainNorm = Math.max(0, Math.min(1, (Math.log10(plainVal) + 3) / 3));
                            const pR = Math.round(88 + plainNorm * 167);
                            const pG = Math.round(73 + plainNorm * (166 - 73));
                            const pB = Math.round(73 + plainNorm * (255 - 73));
                            ctx.fillStyle = 'rgb(' + pR + ',' + pG + ',' + pB + ')';
                            const plainBarH = Math.max(2, plainNorm * halfH);
                            ctx.fillRect(x, topY + halfH - plainBarH, barW - 1, plainBarH);

                            // Residual network bar
                            const resVal = resGrad[i];
                            const resNorm = Math.max(0, Math.min(1, resVal / maxGrad));
                            const rR = Math.round(63 + resNorm * (63));
                            const rG = Math.round(100 + resNorm * 85);
                            const rB = Math.round(80 + resNorm * 80);
                            ctx.fillStyle = 'rgb(' + rR + ',' + rG + ',' + rB + ')';
                            const resBarH = Math.max(2, resNorm * halfH);
                            ctx.fillRect(x, botY + halfH - resBarH, barW - 1, resBarH);
                        }

                        // Axes and labels
                        ctx.strokeStyle = viz.colors.axis;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(pad.left, topY + halfH);
                        ctx.lineTo(pad.left + plotW, topY + halfH);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(pad.left, botY + halfH);
                        ctx.lineTo(pad.left + plotW, botY + halfH);
                        ctx.stroke();

                        // Layer labels
                        ctx.fillStyle = viz.colors.text;
                        ctx.font = '10px -apple-system,sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'top';
                        for (let i = 0; i < numLayers; i += 5) {
                            const x = pad.left + (i + 0.5) * barW;
                            ctx.fillText(i + 1, x, topY + halfH + 3);
                            ctx.fillText(i + 1, x, botY + halfH + 3);
                        }
                        viz.screenText('Layer', pad.left + plotW / 2, H - 10, viz.colors.text, 11);

                        // Gradient magnitude annotations
                        const layer1Plain = plainGrad[0].toExponential(1);
                        const layer50Plain = plainGrad[numLayers - 1].toFixed(2);
                        viz.screenText('Layer 1: ' + layer1Plain, pad.left + 5, topY + 5, viz.colors.red, 10, 'left');
                        viz.screenText('Layer 50: ' + layer50Plain, pad.left + plotW - 5, topY + 5, viz.colors.teal, 10, 'right');

                        viz.screenText('Jacobian norm: ' + jacobianNorm.toFixed(2), W - pad.right - 5, pad.top + halfH + 18, viz.colors.yellow, 11, 'right');

                        // Color scale
                        viz.screenText('\u2190 Weak gradient', pad.left + 5, topY + halfH - 5, viz.colors.red, 9, 'left', 'bottom');
                        viz.screenText('Strong gradient \u2192', pad.left + plotW - 5, topY + halfH - 5, viz.colors.teal, 9, 'right', 'bottom');
                    }

                    draw();

                    VizEngine.createSlider(controls, 'Jacobian norm: ', 0.5, 1.1, jacobianNorm, 0.02, function(val) {
                        jacobianNorm = val;
                        draw();
                    });

                    return viz;
                }
            }
        ],
        exercises: [
            {
                question: 'Derive the Jacobian of a residual block \\(\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}\\) with respect to \\(\\mathbf{x}\\). Why does this prevent vanishing gradients, even when \\(\\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}} \\approx 0\\)?',
                hint: 'Apply the chain rule to the sum. What is the Jacobian of the identity term?',
                solution: 'The Jacobian is: \\[\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} = \\frac{\\partial \\mathcal{F}(\\mathbf{x})}{\\partial \\mathbf{x}} + \\mathbf{I}\\] Even when \\(\\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}} \\approx 0\\) (the residual branch contributes nothing to the gradient), the identity matrix \\(\\mathbf{I}\\) ensures that the Jacobian has eigenvalues close to 1. The gradient passes through unattenuated via the skip connection. In a chain of \\(L\\) such blocks, instead of a product \\(\\prod_{i=1}^L J_i\\) that can vanish, we get a sum of terms that always includes a path multiplied by 1.'
            },
            {
                question: 'A 3-block residual network has blocks \\(\\mathcal{F}_1, \\mathcal{F}_2, \\mathcal{F}_3\\). List all possible paths from input to output, where each block is either "used" (signal goes through \\(\\mathcal{F}_i\\)) or "skipped" (signal goes through identity). How many paths are there? What is the shortest path?',
                hint: 'Each block independently offers two choices. Think binary.',
                solution: 'Each of the 3 blocks contributes a binary choice (use \\(\\mathcal{F}_i\\) or skip), giving \\(2^3 = 8\\) paths:<ol><li>Skip all: \\(\\mathbf{x} \\to \\mathbf{x}\\) (length 0 blocks)</li><li>\\(\\mathcal{F}_1\\) only (length 1)</li><li>\\(\\mathcal{F}_2\\) only (length 1)</li><li>\\(\\mathcal{F}_3\\) only (length 1)</li><li>\\(\\mathcal{F}_1, \\mathcal{F}_2\\) (length 2)</li><li>\\(\\mathcal{F}_1, \\mathcal{F}_3\\) (length 2)</li><li>\\(\\mathcal{F}_2, \\mathcal{F}_3\\) (length 2)</li><li>\\(\\mathcal{F}_1, \\mathcal{F}_2, \\mathcal{F}_3\\) (length 3)</li></ol>The shortest path skips all blocks (length 0), passing input directly to output. This ensemble of \\(2^n\\) paths of varying lengths is why ResNets are robust to block deletion and can be interpreted as implicit ensembles.'
            },
            {
                question: 'In a residual block with a dimension mismatch (input has 64 channels, output needs 128 channels), the shortcut uses a linear projection \\(W_s\\). Write the full equation and explain why \\(W_s\\) is typically implemented as a \\(1 \\times 1\\) convolution with stride 2.',
                hint: 'Consider what happens when spatial dimensions are halved (stride-2 convolution in the main branch) and channels are doubled.',
                solution: 'The block computes: \\[\\mathbf{y} = \\mathcal{F}(\\mathbf{x}, \\{W_i\\}) + W_s \\mathbf{x}\\] When the main branch uses a stride-2 convolution (halving spatial dimensions from \\(H \\times W\\) to \\(H/2 \\times W/2\\)) and doubles channels from 64 to 128, the shortcut \\(\\mathbf{x}\\) must be transformed to match. A \\(1 \\times 1\\) convolution with 128 output filters and stride 2 achieves both: it maps 64 channels to 128 (channel projection) and halves spatial dimensions (stride 2), using only \\(64 \\times 128 = 8192\\) parameters with no spatial receptive field. This is computationally cheap and preserves the benefit of the skip connection.'
            }
        ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: ResNet Architecture
    // ─────────────────────────────────────────────────────────────────────────
    {
        id: 'ch11-sec03',
        title: 'ResNet Architecture',
        content: `
<h2>ResNet Architecture</h2>

<div class="env-block intuition">
<div class="env-title">From Concept to Architecture</div>
<div class="env-body">
<p>Residual learning is a principle; ResNet is its embodiment. The architecture systematically applies skip connections throughout a deep convolutional network, organized into stages of increasing channel depth and decreasing spatial resolution. Two building blocks serve different depth regimes: the <strong>basic block</strong> for shallower variants and the <strong>bottleneck block</strong> for deeper ones.</p>
</div>
</div>

<h3>The Basic Block</h3>

<div class="env-block definition">
<div class="env-title">Basic Residual Block</div>
<div class="env-body">
<p>Used in ResNet-18 and ResNet-34. The block consists of two \\(3 \\times 3\\) convolutions with BatchNorm and ReLU:</p>
\\[\\mathbf{y} = \\text{ReLU}\\Big(\\text{BN}\\big(W_2 * \\text{ReLU}(\\text{BN}(W_1 * \\mathbf{x}))\\big) + \\mathbf{x}\\Big)\\]
<p>Each convolution preserves the spatial dimensions (padding=1, stride=1 within a stage). Channels: \\(C \\to C \\to C\\). Total parameters per block: \\(2 \\times (3^2 \\times C^2) = 18C^2\\).</p>
</div>
</div>

<h3>The Bottleneck Block</h3>

<div class="env-block definition">
<div class="env-title">Bottleneck Residual Block</div>
<div class="env-body">
<p>Used in ResNet-50, ResNet-101, and ResNet-152. The block uses a three-layer "bottleneck" design:</p>
<ol>
<li>\\(1 \\times 1\\) conv: reduces channels from \\(4C\\) to \\(C\\) (compression)</li>
<li>\\(3 \\times 3\\) conv: operates on \\(C\\) channels (spatial processing)</li>
<li>\\(1 \\times 1\\) conv: expands channels from \\(C\\) back to \\(4C\\) (expansion)</li>
</ol>
\\[\\mathbf{y} = \\text{ReLU}\\Big(\\text{BN}(W_3^{1\\times1} * \\text{ReLU}(\\text{BN}(W_2^{3\\times3} * \\text{ReLU}(\\text{BN}(W_1^{1\\times1} * \\mathbf{x})))))+ \\mathbf{x}\\Big)\\]
<p>Parameters per block: \\(1 \\cdot 4C \\cdot C + 9 \\cdot C \\cdot C + 1 \\cdot C \\cdot 4C = 17C^2\\). Despite having 3 layers (vs. 2 in the basic block), the bottleneck uses <em>fewer</em> parameters when \\(C\\) is large, because the expensive \\(3 \\times 3\\) convolution operates on the reduced \\(C\\) channels instead of \\(4C\\).</p>
</div>
</div>

<h3>Full Architecture</h3>

<p>All ResNet variants follow the same high-level structure:</p>

<ol>
<li><strong>Stem</strong>: \\(7 \\times 7\\) conv with stride 2, BatchNorm, ReLU, followed by \\(3 \\times 3\\) max pooling with stride 2. Input \\(224 \\times 224 \\times 3\\) becomes \\(56 \\times 56 \\times 64\\).</li>
<li><strong>Stage 1</strong> (conv2_x): residual blocks with 64 channels, spatial size \\(56 \\times 56\\)</li>
<li><strong>Stage 2</strong> (conv3_x): residual blocks with 128 channels, spatial size \\(28 \\times 28\\)</li>
<li><strong>Stage 3</strong> (conv4_x): residual blocks with 256 channels, spatial size \\(14 \\times 14\\)</li>
<li><strong>Stage 4</strong> (conv5_x): residual blocks with 512 channels, spatial size \\(7 \\times 7\\)</li>
<li><strong>Head</strong>: global average pooling \\(\\to\\) fully-connected layer \\(\\to\\) 1000-class softmax</li>
</ol>

<p>The transition between stages uses stride-2 convolution to halve spatial dimensions, with a projection shortcut to match dimensions.</p>

<div class="env-block example">
<div class="env-title">ResNet Variants</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.9em;">
<tr style="border-bottom:1px solid #30363d;"><th style="padding:6px;">Variant</th><th style="padding:6px;">Block Type</th><th style="padding:6px;">Blocks per Stage</th><th style="padding:6px;">Total Layers</th><th style="padding:6px;">Parameters</th></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">ResNet-18</td><td style="padding:6px;">Basic</td><td style="padding:6px;">[2, 2, 2, 2]</td><td style="padding:6px;">18</td><td style="padding:6px;">11.7M</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">ResNet-34</td><td style="padding:6px;">Basic</td><td style="padding:6px;">[3, 4, 6, 3]</td><td style="padding:6px;">34</td><td style="padding:6px;">21.8M</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">ResNet-50</td><td style="padding:6px;">Bottleneck</td><td style="padding:6px;">[3, 4, 6, 3]</td><td style="padding:6px;">50</td><td style="padding:6px;">25.6M</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">ResNet-101</td><td style="padding:6px;">Bottleneck</td><td style="padding:6px;">[3, 4, 23, 3]</td><td style="padding:6px;">101</td><td style="padding:6px;">44.5M</td></tr>
<tr><td style="padding:6px;">ResNet-152</td><td style="padding:6px;">Bottleneck</td><td style="padding:6px;">[3, 8, 36, 3]</td><td style="padding:6px;">152</td><td style="padding:6px;">60.2M</td></tr>
</table>
<p>Layer count formula: For basic blocks, total layers = \\(2 \\times \\sum n_i + 2\\) (the +2 is the stem conv and the FC). For bottleneck blocks, total layers = \\(3 \\times \\sum n_i + 2\\).</p>
</div>
</div>

<div class="viz-placeholder" data-viz="ch11-viz-resnet-blocks"></div>

<div class="env-block remark">
<div class="env-title">Why Bottleneck?</div>
<div class="env-body">
<p>The bottleneck design is motivated by computational efficiency. A \\(3 \\times 3\\) convolution on 256 channels costs \\(9 \\times 256^2 \\approx 590K\\) multiply-adds per spatial position. The bottleneck squeezes to 64 channels first (\\(1 \\times 256 \\times 64 = 16K\\)), does the \\(3 \\times 3\\) there (\\(9 \\times 64^2 \\approx 37K\\)), then expands back (\\(1 \\times 64 \\times 256 = 16K\\)), totaling about 69K. That is roughly 8.5x cheaper. This savings allows ResNet-50 to have roughly the same parameter count as ResNet-34 while being significantly deeper and more accurate.</p>
</div>
</div>`,
        visualizations: [
            {
                id: 'ch11-viz-resnet-blocks',
                title: 'ResNet Building Blocks: Basic vs. Bottleneck',
                description: 'Toggle between the basic block (2 layers, used in ResNet-18/34) and the bottleneck block (3 layers, used in ResNet-50/101/152). Observe how dimensions change through each layer.',
                setup(container, controls) {
                    const viz = new VizEngine(container, { width: 720, height: 460, scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx;
                    const W = viz.width, H = viz.height;

                    let blockType = 'basic'; // 'basic' or 'bottleneck'
                    let channels = 64;

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
                        const angle = Math.atan2(y2 - y1, x2 - x1);
                        ctx.fillStyle = color;
                        ctx.beginPath();
                        ctx.moveTo(x2, y2);
                        ctx.lineTo(x2 - 8 * Math.cos(angle - Math.PI / 6), y2 - 8 * Math.sin(angle - Math.PI / 6));
                        ctx.lineTo(x2 - 8 * Math.cos(angle + Math.PI / 6), y2 - 8 * Math.sin(angle + Math.PI / 6));
                        ctx.closePath();
                        ctx.fill();
                    }

                    function draw() {
                        viz.clear();

                        const title = blockType === 'basic' ? 'Basic Residual Block' : 'Bottleneck Residual Block';
                        viz.screenText(title, W / 2, 22, viz.colors.white, 16);

                        const cx = W / 2;
                        const boxW = 180, boxH = 40;
                        const gap = 18;
                        const C = channels;
                        const C4 = C * 4;

                        if (blockType === 'basic') {
                            // Input
                            const inputY = 55;
                            viz.screenText('Input: ' + C + ' channels', cx, inputY, viz.colors.teal, 12);

                            // Layer 1: 3x3 conv
                            const l1Y = inputY + 30;
                            drawRoundedRect(cx - boxW / 2, l1Y, boxW, boxH, 6, 'rgba(88,166,255,0.15)', viz.colors.blue);
                            viz.screenText('3\u00D73 Conv, ' + C, cx, l1Y + boxH / 2, viz.colors.blue, 12);
                            drawArrow(cx, inputY + 10, cx, l1Y, viz.colors.text);

                            // BN + ReLU
                            const bn1Y = l1Y + boxH + gap;
                            drawRoundedRect(cx - boxW / 2, bn1Y, boxW, boxH * 0.7, 6, 'rgba(188,140,255,0.12)', viz.colors.purple);
                            viz.screenText('BatchNorm + ReLU', cx, bn1Y + boxH * 0.35, viz.colors.purple, 11);
                            drawArrow(cx, l1Y + boxH, cx, bn1Y, viz.colors.text);

                            // Layer 2: 3x3 conv
                            const l2Y = bn1Y + boxH * 0.7 + gap;
                            drawRoundedRect(cx - boxW / 2, l2Y, boxW, boxH, 6, 'rgba(88,166,255,0.15)', viz.colors.blue);
                            viz.screenText('3\u00D73 Conv, ' + C, cx, l2Y + boxH / 2, viz.colors.blue, 12);
                            drawArrow(cx, bn1Y + boxH * 0.7, cx, l2Y, viz.colors.text);

                            // BN
                            const bn2Y = l2Y + boxH + gap;
                            drawRoundedRect(cx - boxW / 2, bn2Y, boxW, boxH * 0.7, 6, 'rgba(188,140,255,0.12)', viz.colors.purple);
                            viz.screenText('BatchNorm', cx, bn2Y + boxH * 0.35, viz.colors.purple, 11);
                            drawArrow(cx, l2Y + boxH, cx, bn2Y, viz.colors.text);

                            // Addition
                            const addY = bn2Y + boxH * 0.7 + gap + 10;
                            ctx.fillStyle = 'rgba(63,185,160,0.2)';
                            ctx.beginPath();
                            ctx.arc(cx, addY + 15, 18, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.teal;
                            ctx.lineWidth = 2;
                            ctx.stroke();
                            viz.screenText('+', cx, addY + 15, viz.colors.teal, 20);
                            drawArrow(cx, bn2Y + boxH * 0.7, cx, addY - 3, viz.colors.text);

                            // Skip connection
                            const skipX = cx + boxW / 2 + 40;
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            ctx.moveTo(cx + boxW / 2 + 5, inputY + 10);
                            ctx.lineTo(skipX, inputY + 10);
                            ctx.lineTo(skipX, addY + 15);
                            ctx.lineTo(cx + 18, addY + 15);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            // Arrowhead
                            ctx.fillStyle = viz.colors.orange;
                            ctx.beginPath();
                            ctx.moveTo(cx + 18, addY + 15);
                            ctx.lineTo(cx + 26, addY + 10);
                            ctx.lineTo(cx + 26, addY + 20);
                            ctx.closePath();
                            ctx.fill();
                            viz.screenText('Identity', skipX + 5, (inputY + 10 + addY + 15) / 2, viz.colors.orange, 11, 'left');
                            viz.screenText('shortcut', skipX + 5, (inputY + 10 + addY + 15) / 2 + 14, viz.colors.orange, 11, 'left');

                            // ReLU
                            const reluY = addY + 40;
                            drawRoundedRect(cx - boxW / 2, reluY, boxW, boxH * 0.7, 6, 'rgba(63,185,160,0.12)', viz.colors.teal);
                            viz.screenText('ReLU', cx, reluY + boxH * 0.35, viz.colors.teal, 11);
                            drawArrow(cx, addY + 33, cx, reluY, viz.colors.text);

                            // Output
                            const outY = reluY + boxH * 0.7 + 15;
                            viz.screenText('Output: ' + C + ' channels', cx, outY, viz.colors.teal, 12);
                            drawArrow(cx, reluY + boxH * 0.7, cx, outY - 8, viz.colors.text);

                            // Params
                            const params = 2 * 9 * C * C;
                            viz.screenText('Parameters: 2 \u00D7 (3\u00B2 \u00D7 ' + C + '\u00B2) = ' + params.toLocaleString(), cx, H - 20, viz.colors.text, 11);

                        } else {
                            // Bottleneck block
                            const inputY = 50;
                            viz.screenText('Input: ' + C4 + ' channels', cx, inputY, viz.colors.teal, 12);

                            // Layer 1: 1x1 conv (reduce)
                            const l1Y = inputY + 28;
                            drawRoundedRect(cx - boxW / 2, l1Y, boxW, boxH, 6, 'rgba(88,166,255,0.15)', viz.colors.blue);
                            viz.screenText('1\u00D71 Conv, ' + C4 + '\u2192' + C, cx, l1Y + boxH / 2, viz.colors.blue, 11);
                            drawArrow(cx, inputY + 10, cx, l1Y, viz.colors.text);

                            const bn1Y = l1Y + boxH + gap * 0.7;
                            drawRoundedRect(cx - boxW / 2, bn1Y, boxW, boxH * 0.6, 6, 'rgba(188,140,255,0.12)', viz.colors.purple);
                            viz.screenText('BN + ReLU', cx, bn1Y + boxH * 0.3, viz.colors.purple, 10);
                            drawArrow(cx, l1Y + boxH, cx, bn1Y, viz.colors.text);

                            // Layer 2: 3x3 conv
                            const l2Y = bn1Y + boxH * 0.6 + gap * 0.7;
                            drawRoundedRect(cx - boxW / 2, l2Y, boxW, boxH, 6, 'rgba(240,136,62,0.15)', viz.colors.orange);
                            viz.screenText('3\u00D73 Conv, ' + C, cx, l2Y + boxH / 2, viz.colors.orange, 11);
                            drawArrow(cx, bn1Y + boxH * 0.6, cx, l2Y, viz.colors.text);

                            const bn2Y = l2Y + boxH + gap * 0.7;
                            drawRoundedRect(cx - boxW / 2, bn2Y, boxW, boxH * 0.6, 6, 'rgba(188,140,255,0.12)', viz.colors.purple);
                            viz.screenText('BN + ReLU', cx, bn2Y + boxH * 0.3, viz.colors.purple, 10);
                            drawArrow(cx, l2Y + boxH, cx, bn2Y, viz.colors.text);

                            // Layer 3: 1x1 conv (expand)
                            const l3Y = bn2Y + boxH * 0.6 + gap * 0.7;
                            drawRoundedRect(cx - boxW / 2, l3Y, boxW, boxH, 6, 'rgba(88,166,255,0.15)', viz.colors.blue);
                            viz.screenText('1\u00D71 Conv, ' + C + '\u2192' + C4, cx, l3Y + boxH / 2, viz.colors.blue, 11);
                            drawArrow(cx, bn2Y + boxH * 0.6, cx, l3Y, viz.colors.text);

                            const bn3Y = l3Y + boxH + gap * 0.7;
                            drawRoundedRect(cx - boxW / 2, bn3Y, boxW, boxH * 0.6, 6, 'rgba(188,140,255,0.12)', viz.colors.purple);
                            viz.screenText('BatchNorm', cx, bn3Y + boxH * 0.3, viz.colors.purple, 10);
                            drawArrow(cx, l3Y + boxH, cx, bn3Y, viz.colors.text);

                            // Addition
                            const addY = bn3Y + boxH * 0.6 + gap * 0.5 + 5;
                            ctx.fillStyle = 'rgba(63,185,160,0.2)';
                            ctx.beginPath();
                            ctx.arc(cx, addY + 12, 16, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.teal;
                            ctx.lineWidth = 2;
                            ctx.stroke();
                            viz.screenText('+', cx, addY + 12, viz.colors.teal, 18);
                            drawArrow(cx, bn3Y + boxH * 0.6, cx, addY - 4, viz.colors.text);

                            // Skip connection
                            const skipX = cx + boxW / 2 + 40;
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            ctx.moveTo(cx + boxW / 2 + 5, inputY + 10);
                            ctx.lineTo(skipX, inputY + 10);
                            ctx.lineTo(skipX, addY + 12);
                            ctx.lineTo(cx + 16, addY + 12);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.beginPath();
                            ctx.moveTo(cx + 16, addY + 12);
                            ctx.lineTo(cx + 24, addY + 7);
                            ctx.lineTo(cx + 24, addY + 17);
                            ctx.closePath();
                            ctx.fill();
                            viz.screenText('Identity', skipX + 5, (inputY + 10 + addY + 12) / 2, viz.colors.orange, 10, 'left');

                            // ReLU + output
                            const reluY = addY + 30;
                            drawRoundedRect(cx - boxW / 2, reluY, boxW, boxH * 0.6, 6, 'rgba(63,185,160,0.12)', viz.colors.teal);
                            viz.screenText('ReLU', cx, reluY + boxH * 0.3, viz.colors.teal, 10);
                            drawArrow(cx, addY + 28, cx, reluY, viz.colors.text);

                            const outY = reluY + boxH * 0.6 + 12;
                            viz.screenText('Output: ' + C4 + ' channels', cx, outY, viz.colors.teal, 12);
                            drawArrow(cx, reluY + boxH * 0.6, cx, outY - 8, viz.colors.text);

                            // Params
                            const params = C4 * C + 9 * C * C + C * C4;
                            viz.screenText('Parameters: ' + C4 + '\u00D7' + C + ' + 9\u00D7' + C + '\u00B2 + ' + C + '\u00D7' + C4 + ' = ' + params.toLocaleString(), cx, H - 20, viz.colors.text, 11);

                            // Channel width annotation
                            const annoX = cx - boxW / 2 - 55;
                            viz.screenText(C4 + 'ch', annoX, l1Y + boxH / 2, viz.colors.text, 10);
                            viz.screenText(C + 'ch', annoX, l2Y + boxH / 2, viz.colors.text, 10);
                            viz.screenText(C4 + 'ch', annoX, l3Y + boxH / 2, viz.colors.text, 10);
                        }
                    }

                    draw();

                    VizEngine.createButton(controls, 'Basic Block', function() {
                        blockType = 'basic';
                        draw();
                    });
                    VizEngine.createButton(controls, 'Bottleneck Block', function() {
                        blockType = 'bottleneck';
                        draw();
                    });
                    VizEngine.createSlider(controls, 'C = ', 16, 256, channels, 16, function(val) {
                        channels = Math.round(val / 16) * 16;
                        draw();
                    });

                    return viz;
                }
            }
        ],
        exercises: [
            {
                question: 'Compute the total number of learnable parameters in ResNet-18 (excluding batch normalization parameters). The architecture uses basic blocks with configuration [2, 2, 2, 2] and channel widths [64, 128, 256, 512].',
                hint: 'Count the stem (7x7 conv with 3 input channels, 64 output), each basic block (two 3x3 convs), the projection shortcuts at stage transitions (1x1 convs), and the final FC layer (512 to 1000).',
                solution: '<strong>Stem</strong>: \\(7^2 \\times 3 \\times 64 + 64 = 9472\\) (weights + biases).<br><strong>Stage 1</strong> (64ch, 2 blocks): \\(2 \\times 2 \\times 3^2 \\times 64^2 = 147{,}456\\).<br><strong>Stage 2</strong> (128ch, 2 blocks): first block has a projection shortcut \\(1 \\times 1 \\times 64 \\times 128 = 8{,}192\\). Convs: \\(3^2 \\times 64 \\times 128 + 3^2 \\times 128^2 = 73{,}728 + 147{,}456 = 221{,}184\\). Second block: \\(2 \\times 3^2 \\times 128^2 = 294{,}912\\). Subtotal: \\(524{,}288\\).<br><strong>Stage 3</strong> (256ch, 2 blocks): similarly, projection \\(= 32{,}768\\), conv subtotal \\(\\approx 2{,}097{,}152\\).<br><strong>Stage 4</strong> (512ch, 2 blocks): projection \\(= 131{,}072\\), conv subtotal \\(\\approx 8{,}388{,}608\\).<br><strong>FC</strong>: \\(512 \\times 1000 + 1000 = 513{,}000\\).<br><strong>Total</strong>: approximately <strong>11.69M</strong> parameters.'
            },
            {
                question: 'Why does ResNet-50 use bottleneck blocks instead of basic blocks, even though ResNet-34 (basic blocks) has a similar number of blocks per stage [3, 4, 6, 3]? Compare the parameter counts and explain the computational advantage.',
                hint: 'Compare the FLOPs of two basic blocks vs. one bottleneck block for 256 channels.',
                solution: 'Two basic blocks on 256 channels cost \\(2 \\times 2 \\times 9 \\times 256^2 \\approx 2.36M\\) parameters. One bottleneck block on 256 channels (with 64 internal channels) costs \\(256 \\times 64 + 9 \\times 64^2 + 64 \\times 256 = 16{,}384 + 36{,}864 + 16{,}384 = 69{,}632 \\approx 0.07M\\) parameters. The bottleneck is about 34x cheaper per unit. This allows ResNet-50 to be significantly deeper (50 vs. 34 layers) while using comparable total parameters (25.6M vs. 21.8M). The \\(1 \\times 1\\) convolutions act as dimensionality reducers, confining the expensive \\(3 \\times 3\\) spatial convolution to a narrow channel bottleneck.'
            },
            {
                question: 'In the original ResNet paper, three options were explored for downsampling shortcuts: (A) zero-padding for extra channels, (B) projection shortcuts only at dimension-changing blocks, (C) projection shortcuts at all blocks. Which option was used in practice and why?',
                hint: 'Consider the trade-off between accuracy gain and parameter/computational cost.',
                solution: 'Option (B) was used in practice. Option (A) (zero-padding) is parameter-free but performs slightly worse because the zero-padded channels carry no learned information. Option (C) (projections everywhere) gives marginal accuracy improvement over (B) but adds many parameters and FLOPs. Option (B) is the Goldilocks choice: it uses \\(1 \\times 1\\) projections only at stage boundaries where channel dimensions change (64\u219264, 64\u2192128, 128\u2192256, 256\u2192512), keeping the majority of shortcuts as parameter-free identity mappings. The accuracy difference between (B) and (C) was small (~0.1%), making the extra parameters of (C) unjustified.'
            }
        ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: DenseNet & EfficientNet
    // ─────────────────────────────────────────────────────────────────────────
    {
        id: 'ch11-sec04',
        title: 'DenseNet & EfficientNet',
        content: `
<h2>DenseNet & EfficientNet</h2>

<div class="env-block intuition">
<div class="env-title">Beyond Skip Connections</div>
<div class="env-body">
<p>ResNet showed that shortcut connections are essential for training deep networks. But the ResNet skip connection only goes one block back. What if every layer was directly connected to <em>every other layer</em>? And once we can train arbitrarily deep networks, how should we balance depth, width, and resolution to maximize accuracy for a given computational budget? DenseNet answers the first question; EfficientNet answers the second.</p>
</div>
</div>

<h3>DenseNet: Dense Connectivity</h3>

<div class="env-block definition">
<div class="env-title">Dense Block</div>
<div class="env-body">
<p>In a <strong>Dense Block</strong> (Huang et al., 2017), every layer receives the feature maps of <em>all preceding layers</em> as input and passes its own feature maps to <em>all subsequent layers</em>. For a block with \\(L\\) layers:</p>
\\[\\mathbf{x}_l = H_l\\big([\\mathbf{x}_0, \\mathbf{x}_1, \\ldots, \\mathbf{x}_{l-1}]\\big)\\]
<p>where \\([\\cdot]\\) denotes channel-wise concatenation and \\(H_l\\) is BN-ReLU-Conv. The block has \\(\\frac{L(L+1)}{2}\\) connections (compared to \\(L\\) in a plain network).</p>
</div>
</div>

<h4>Growth Rate</h4>
<p>Each layer \\(H_l\\) produces \\(k\\) feature maps (called the <strong>growth rate</strong>). After \\(l\\) layers, the input to layer \\(l+1\\) has \\(k_0 + l \\cdot k\\) channels, where \\(k_0\\) is the initial channel count. Typical growth rates are \\(k = 12\\) or \\(k = 32\\), much smaller than traditional channel widths. The concatenation mechanism ensures information from all layers is preserved, so each layer can be narrow.</p>

<h4>Transition Layers</h4>
<p>Between dense blocks, <strong>transition layers</strong> reduce spatial dimensions and channel counts:</p>
<ul>
<li>\\(1 \\times 1\\) convolution (typically halving channels, called <strong>compression</strong> with ratio \\(\\theta = 0.5\\))</li>
<li>\\(2 \\times 2\\) average pooling with stride 2</li>
</ul>

<div class="env-block example">
<div class="env-title">DenseNet Advantages</div>
<div class="env-body">
<ul>
<li><strong>Feature reuse</strong>: Each layer has direct access to all earlier features, enabling maximum feature reuse and reducing redundancy.</li>
<li><strong>Parameter efficiency</strong>: DenseNet-BC-100 with \\(k=12\\) achieves comparable accuracy to ResNet-1001 on CIFAR-10 with 90% fewer parameters.</li>
<li><strong>Implicit deep supervision</strong>: Because every layer connects to the loss through short paths, all layers receive strong gradient signals.</li>
<li><strong>Gradient flow</strong>: Dense connections provide \\(\\frac{L(L+1)}{2}\\) gradient paths, far more than ResNet's \\(L\\) paths.</li>
</ul>
</div>
</div>

<div class="viz-placeholder" data-viz="ch11-viz-densenet"></div>

<h3>EfficientNet: Compound Scaling</h3>

<div class="env-block definition">
<div class="env-title">Compound Scaling</div>
<div class="env-body">
<p>Tan & Le (2019) observed that network depth \\(d\\), width \\(w\\), and input resolution \\(r\\) should be scaled together, not independently. They proposed <strong>compound scaling</strong>:</p>
\\[d = \\alpha^\\phi, \\quad w = \\beta^\\phi, \\quad r = \\gamma^\\phi\\]
<p>subject to \\(\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2\\) (approximately doubling FLOPs), where \\(\\phi\\) is a compound coefficient that controls total resources. The constraint reflects the fact that FLOPs scale linearly with depth but quadratically with width and resolution.</p>
</div>
</div>

<p>The key insight is that scaling any single dimension alone yields diminishing returns. Making a network deeper helps, but only if the network is also wide enough to learn rich features and the input resolution is high enough to contain fine-grained information. Compound scaling finds the optimal balance.</p>

<div class="env-block example">
<div class="env-title">EfficientNet Family</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.9em;">
<tr style="border-bottom:1px solid #30363d;"><th style="padding:6px;">Model</th><th style="padding:6px;">Resolution</th><th style="padding:6px;">Params</th><th style="padding:6px;">Top-1 Acc</th><th style="padding:6px;">FLOPs</th></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">EfficientNet-B0</td><td style="padding:6px;">224</td><td style="padding:6px;">5.3M</td><td style="padding:6px;">77.1%</td><td style="padding:6px;">0.39B</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">EfficientNet-B3</td><td style="padding:6px;">300</td><td style="padding:6px;">12M</td><td style="padding:6px;">81.6%</td><td style="padding:6px;">1.8B</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">EfficientNet-B7</td><td style="padding:6px;">600</td><td style="padding:6px;">66M</td><td style="padding:6px;">84.3%</td><td style="padding:6px;">37B</td></tr>
<tr><td style="padding:6px;">ResNet-50</td><td style="padding:6px;">224</td><td style="padding:6px;">26M</td><td style="padding:6px;">76.0%</td><td style="padding:6px;">4.1B</td></tr>
</table>
<p>EfficientNet-B0 achieves higher accuracy than ResNet-50 with 5x fewer parameters and 10x fewer FLOPs.</p>
</div>
</div>

<h3>ConvNeXt: Modernizing ConvNets</h3>

<p>Liu et al. (2022) asked: can a pure ConvNet match the performance of Vision Transformers (ViT)? By systematically applying design ideas borrowed from Transformers to a ResNet-50 starting point, they created <strong>ConvNeXt</strong>:</p>

<ul>
<li><strong>Macro design</strong>: Stage ratio changed from [3,4,6,3] to [3,3,9,3] (following Swin Transformer's 1:1:3:1)</li>
<li><strong>Patchify stem</strong>: Replace 7x7 conv + pool with non-overlapping 4x4 conv, stride 4</li>
<li><strong>Depthwise separable convolutions</strong>: Replace standard 3x3 convs with 7x7 depthwise convolutions</li>
<li><strong>Inverted bottleneck</strong>: Expand channels in the middle (opposite of ResNet bottleneck)</li>
<li><strong>Fewer activation functions</strong>: Use GELU, only one activation per block</li>
<li><strong>LayerNorm</strong> instead of BatchNorm</li>
</ul>

<p>The result: ConvNeXt-T (29M params) achieves 82.1% top-1 on ImageNet, competitive with Swin-T (28M, 81.3%). This demonstrates that the architectural innovations of Transformers, not self-attention per se, drive their strong performance.</p>
`,
        visualizations: [
            {
                id: 'ch11-viz-densenet',
                title: 'DenseNet: Dense Connectivity Pattern',
                description: 'Each layer in a dense block connects to all subsequent layers via concatenation. Watch the connections animate to see how features flow through the network.',
                setup(container, controls) {
                    const viz = new VizEngine(container, { width: 720, height: 420, scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx;
                    const W = viz.width, H = viz.height;

                    let numLayers = 5;
                    let animPhase = 0;
                    let activeLayer = -1;

                    function draw(time) {
                        viz.clear();
                        animPhase = (time || 0) * 0.001;

                        viz.screenText('Dense Block: Every Layer Connected to All Previous', W / 2, 22, viz.colors.white, 14);

                        const padX = 80, padY = 80;
                        const layerW = 50, layerH = 200;
                        const totalW = W - 2 * padX;
                        const spacing = totalW / (numLayers + 1);
                        const topY = padY;

                        // Compute channel heights
                        const growthRate = 12;
                        const initChannels = 24;

                        // Draw connections first (behind layers)
                        const cycleTime = 3; // seconds per layer highlight
                        const totalCycle = numLayers * cycleTime;
                        activeLayer = Math.floor((animPhase % totalCycle) / cycleTime);

                        for (let to = 1; to <= numLayers; to++) {
                            for (let from = 0; from < to; from++) {
                                const fromX = padX + (from + 0.5) * spacing;
                                const toX = padX + (to + 0.5) * spacing;
                                const fromYpos = topY + layerH * 0.5;
                                const toYpos = topY + layerH * 0.5;

                                // Curve the connection
                                const midY = topY + layerH + 15 + (to - from) * 18;

                                const isActive = (to === activeLayer) || (from === activeLayer && to > from);
                                const alpha = isActive ? 0.8 : 0.15;
                                const lw = isActive ? 2.5 : 1;

                                // Color based on source layer
                                const colors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.green, viz.colors.purple, viz.colors.pink, viz.colors.yellow, viz.colors.red];
                                const color = colors[from % colors.length];

                                ctx.strokeStyle = color;
                                ctx.globalAlpha = alpha;
                                ctx.lineWidth = lw;
                                ctx.beginPath();
                                ctx.moveTo(fromX, fromYpos + layerH * 0.4);
                                ctx.quadraticCurveTo((fromX + toX) / 2, midY, toX, toYpos + layerH * 0.4);
                                ctx.stroke();

                                // Arrow head
                                if (isActive) {
                                    const t = 0.95;
                                    const ax = (1-t)*(1-t)*fromX + 2*(1-t)*t*((fromX+toX)/2) + t*t*toX;
                                    const ay = (1-t)*(1-t)*(fromYpos + layerH*0.4) + 2*(1-t)*t*midY + t*t*(toYpos + layerH*0.4);
                                    ctx.fillStyle = color;
                                    ctx.beginPath();
                                    ctx.arc(ax, ay, 3, 0, Math.PI * 2);
                                    ctx.fill();
                                }
                            }
                        }
                        ctx.globalAlpha = 1;

                        // Draw layers
                        for (let i = 0; i <= numLayers; i++) {
                            const x = padX + (i + 0.5) * spacing - layerW / 2;
                            const channels = initChannels + i * growthRate;
                            const barH = Math.min(layerH, 30 + channels * 1.2);

                            const isHL = (i === activeLayer);
                            const fillColor = isHL ? 'rgba(88,166,255,0.3)' : 'rgba(88,166,255,0.1)';
                            const strokeColor = isHL ? viz.colors.blue : viz.colors.axis;
                            const borderW = isHL ? 2.5 : 1;

                            ctx.fillStyle = fillColor;
                            ctx.strokeStyle = strokeColor;
                            ctx.lineWidth = borderW;
                            ctx.fillRect(x, topY + (layerH - barH) / 2, layerW, barH);
                            ctx.strokeRect(x, topY + (layerH - barH) / 2, layerW, barH);

                            // Label
                            const label = i === 0 ? 'x\u2080' : 'H' + i;
                            viz.screenText(label, x + layerW / 2, topY - 12, isHL ? viz.colors.white : viz.colors.text, 12);
                            viz.screenText(channels + 'ch', x + layerW / 2, topY + layerH + 18 + 10, viz.colors.text, 10);
                        }

                        // Info
                        const totalConnections = numLayers * (numLayers + 1) / 2;
                        viz.screenText('Layers: ' + (numLayers + 1) + '    Connections: ' + totalConnections + '    Growth rate k=' + growthRate, W / 2, H - 18, viz.colors.yellow, 11);

                        // Legend
                        viz.screenText('Hover concept: each layer receives ALL previous feature maps via concatenation', W / 2, H - 40, viz.colors.text, 10);
                    }

                    draw(0);
                    viz.animate(draw);

                    VizEngine.createSlider(controls, 'Layers: ', 3, 8, numLayers, 1, function(val) {
                        numLayers = Math.round(val);
                    });

                    return { stopAnimation: function() { viz.stopAnimation(); } };
                }
            }
        ],
        exercises: [
            {
                question: 'In a DenseNet dense block with 6 layers and growth rate \\(k = 32\\), starting from \\(k_0 = 64\\) input channels, compute (a) the number of input channels to the 6th layer, (b) the total number of connections, and (c) the total output channels of the block.',
                hint: 'Layer \\(l\\) receives all features from layers 0 through \\(l-1\\) via concatenation. Count connections as ordered pairs \\((i, j)\\) with \\(i &lt; j\\).',
                solution: '<p>(a) The 6th layer (\\(l=6\\)) receives features from layers 0, 1, 2, 3, 4, 5. Layer 0 contributes \\(k_0 = 64\\) channels; layers 1 through 5 each contribute \\(k = 32\\) channels. Total input: \\(64 + 5 \\times 32 = 224\\) channels.</p><p>(b) Total connections: \\(\\frac{L(L+1)}{2} = \\frac{6 \\times 7}{2} = 21\\) connections (each of the 6 layers connects to all preceding layers including the input).</p><p>(c) The output of the dense block concatenates all features: \\(k_0 + L \\times k = 64 + 6 \\times 32 = 256\\) channels.</p>'
            },
            {
                question: 'EfficientNet uses compound scaling with \\(\\alpha = 1.2\\), \\(\\beta = 1.1\\), \\(\\gamma = 1.15\\). Verify that the constraint \\(\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2\\) holds. Then compute the depth, width, and resolution multipliers for \\(\\phi = 3\\) (EfficientNet-B3).',
                hint: 'The constraint ensures FLOPs roughly double with each unit increase in \\(\\phi\\). Compute \\(\\alpha \\cdot \\beta^2 \\cdot \\gamma^2\\) first.',
                solution: '<p>Verify constraint: \\(1.2 \\times 1.1^2 \\times 1.15^2 = 1.2 \\times 1.21 \\times 1.3225 = 1.919 \\approx 2\\). The constraint holds.</p><p>For \\(\\phi = 3\\):</p><ul><li>Depth: \\(d = \\alpha^3 = 1.2^3 = 1.728\\) (73% deeper)</li><li>Width: \\(w = \\beta^3 = 1.1^3 = 1.331\\) (33% wider)</li><li>Resolution: \\(r = \\gamma^3 = 1.15^3 = 1.521\\) (52% higher resolution, so \\(224 \\times 1.521 \\approx 341\\), rounded to 300 in practice)</li></ul><p>FLOPs scale by \\(2^3 = 8\\times\\) relative to B0.</p>'
            },
            {
                question: 'Compare the design philosophies of ResNet, DenseNet, and ConvNeXt. For each architecture, state (a) the core innovation, (b) how it handles feature reuse, and (c) one key limitation.',
                hint: 'Think about what each architecture does with information from earlier layers, and what trade-offs each design makes.',
                solution: '<p><strong>ResNet</strong>: (a) Skip connections via element-wise addition. (b) Features from the input are added to the output of each block; the network learns residuals. (c) Addition can lose information: once features are summed, individual contributions cannot be disentangled. Also, only connects to the immediately preceding block.</p><p><strong>DenseNet</strong>: (a) Dense connectivity via channel concatenation. (b) All previous features are preserved and concatenated, enabling maximum reuse. (c) Memory-intensive: the concatenation of all features grows linearly with depth, requiring significant GPU memory. The implementation needs careful memory management (shared memory allocations).</p><p><strong>ConvNeXt</strong>: (a) Modernizing ConvNets with Transformer-era design principles (large kernels, inverted bottleneck, fewer activations, LayerNorm). (b) Uses ResNet-style skip connections (element-wise addition). (c) Still uses fixed-size kernels rather than adaptive attention, so it cannot dynamically route information based on input content (though the large 7x7 depthwise kernels partially compensate).</p>'
            }
        ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 5: Transfer Learning
    // ─────────────────────────────────────────────────────────────────────────
    {
        id: 'ch11-sec05',
        title: 'Transfer Learning',
        content: `
<h2>Transfer Learning</h2>

<div class="env-block intuition">
<div class="env-title">Standing on the Shoulders of ImageNet</div>
<div class="env-body">
<p>Training a deep CNN from scratch requires millions of labeled images, specialized hardware, and days of compute. But the features learned on one large dataset are often useful for many other tasks. <strong>Transfer learning</strong> leverages a model pretrained on a large-scale dataset (typically ImageNet with 1.2M images, 1000 classes) and adapts it to a new, often much smaller, target dataset. This is arguably the single most impactful practical technique in modern deep learning.</p>
</div>
</div>

<h3>Why Transfer Learning Works</h3>

<p>The features learned by deep CNNs are hierarchical and increasingly task-specific:</p>

<ul>
<li><strong>Layer 1-2</strong>: Gabor-like edge detectors, color blobs. These are universal across nearly all visual tasks.</li>
<li><strong>Layer 3-5</strong>: Texture patterns, corners, simple part detectors. Still broadly useful.</li>
<li><strong>Layer 6-8</strong>: Object parts, grid patterns, class-specific textures. More task-dependent.</li>
<li><strong>Final layers</strong>: High-level semantic features, class-specific. Most task-dependent.</li>
</ul>

<div class="env-block definition">
<div class="env-title">Feature Extraction vs. Fine-Tuning</div>
<div class="env-body">
<p>There are two main transfer learning strategies:</p>
<p><strong>Feature Extraction</strong>: Freeze all pretrained layers. Remove the final classification head and replace it with a new one for the target task. Only train the new head. The pretrained network acts as a fixed feature extractor.</p>
<p><strong>Fine-Tuning</strong>: Start from the pretrained weights, replace the classification head, and continue training (some or all of) the pretrained layers on the target data with a small learning rate. This allows the network to adapt its features to the target domain.</p>
</div>
</div>

<h3>Fine-Tuning Strategies</h3>

<p>The depth of fine-tuning depends on the size of the target dataset and its similarity to the source domain:</p>

<div class="env-block example">
<div class="env-title">Four Transfer Learning Regimes</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.9em;">
<tr style="border-bottom:1px solid #30363d;"><th style="padding:6px;"></th><th style="padding:6px;">Small Target Dataset</th><th style="padding:6px;">Large Target Dataset</th></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;font-weight:bold;">Similar to Source</td><td style="padding:6px;">Feature extraction (freeze all) or fine-tune last 1-2 layers only. Low risk of overfitting.</td><td style="padding:6px;">Fine-tune all layers with small learning rate. Plenty of data to adapt features.</td></tr>
<tr><td style="padding:6px;font-weight:bold;">Different from Source</td><td style="padding:6px;">Hardest case. Feature extraction with a linear SVM. Fine-tuning risks overfitting. Consider data augmentation.</td><td style="padding:6px;">Fine-tune from middle layers or even train from scratch. Early features may still transfer; later features likely need relearning.</td></tr>
</table>
</div>
</div>

<h3>Practical Guidelines</h3>

<div class="env-block remark">
<div class="env-title">Best Practices</div>
<div class="env-body">
<ol>
<li><strong>Use a smaller learning rate</strong> for pretrained layers (often 10x smaller than for the new head). The pretrained weights are already in a good region; large updates could destroy them.</li>
<li><strong>Discriminative learning rates</strong>: Use progressively smaller learning rates for earlier layers (e.g., layer group \\(i\\) gets \\(\\eta / 2.6^i\\)). This technique was popularized by ULMFiT (Howard & Ruder, 2018).</li>
<li><strong>Gradual unfreezing</strong>: Start by training only the new head. Then unfreeze the last stage and train. Then unfreeze the second-to-last stage, and so on. This prevents catastrophic forgetting of pretrained features.</li>
<li><strong>Data augmentation</strong>: Critical when the target dataset is small. Random crops, flips, color jitter, and mixup all help.</li>
<li><strong>Batch normalization</strong>: During fine-tuning, keep BN layers in eval mode (use running statistics from pretraining) when the target batch size is small, to avoid noisy batch statistics.</li>
</ol>
</div>
</div>

<div class="viz-placeholder" data-viz="ch11-viz-transfer"></div>

<div class="env-block theorem">
<div class="env-title">When Does Transfer Hurt?</div>
<div class="env-body">
<p>Transfer learning is not always beneficial. Negative transfer can occur when:</p>
<ul>
<li>The source and target domains are very different (e.g., natural images to medical X-rays with inverted contrast).</li>
<li>The pretrained model has learned features that are actively misleading for the target task.</li>
<li>The target dataset is large enough that training from scratch with proper regularization achieves comparable results (He et al., 2019, "Rethinking ImageNet Pre-training").</li>
</ul>
<p>However, even in these cases, pretrained initialization usually converges faster, even if the final accuracy is similar to training from scratch.</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Beyond Classification</div>
<div class="env-body">
<p>Transfer learning extends far beyond image classification. Pretrained backbones (ResNet, EfficientNet, ConvNeXt) serve as feature extractors for object detection (Faster R-CNN, YOLO), semantic segmentation (U-Net, DeepLab), instance segmentation (Mask R-CNN), and many other tasks. The pretrained backbone provides rich features; task-specific heads are added for the target task. This pretrain-then-adapt paradigm is now the dominant approach across both computer vision and NLP (Chapter 19).</p>
</div>
</div>`,
        visualizations: [
            {
                id: 'ch11-viz-transfer',
                title: 'Transfer Learning: Freeze/Unfreeze Strategy',
                description: 'Drag the slider to control how many layers to fine-tune. Frozen layers (blue) keep pretrained weights; unfrozen layers (orange) are updated during training. The new classification head (green) is always trained.',
                setup(container, controls) {
                    const viz = new VizEngine(container, { width: 720, height: 420, scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx;
                    const W = viz.width, H = viz.height;

                    // Represent a ResNet-like architecture as stages
                    const stages = [
                        { name: 'Stem', sublayers: 1, desc: '7x7 conv' },
                        { name: 'Stage 1', sublayers: 3, desc: '64 ch' },
                        { name: 'Stage 2', sublayers: 4, desc: '128 ch' },
                        { name: 'Stage 3', sublayers: 6, desc: '256 ch' },
                        { name: 'Stage 4', sublayers: 3, desc: '512 ch' },
                        { name: 'GAP', sublayers: 1, desc: 'pool' },
                        { name: 'New Head', sublayers: 1, desc: 'FC→K' }
                    ];

                    const totalSublayers = stages.reduce((s, st) => s + st.sublayers, 0);
                    let unfreezeFrom = 5; // Index into stages array; stages >= unfreezeFrom are unfrozen

                    function draw() {
                        viz.clear();
                        viz.screenText('Transfer Learning Pipeline', W / 2, 22, viz.colors.white, 15);

                        const padX = 40, padY = 60;
                        const layerAreaW = W - 2 * padX;
                        const layerH = 200;
                        const topY = padY + 30;

                        // Draw layer bars proportional to sublayer count
                        let xOffset = padX;
                        const totalUnits = totalSublayers;
                        const unitW = layerAreaW / totalUnits;

                        for (let i = 0; i < stages.length; i++) {
                            const stage = stages[i];
                            const w = stage.sublayers * unitW;
                            const isNewHead = (i === stages.length - 1);
                            const isFrozen = (i < unfreezeFrom) && !isNewHead;
                            const isUnfrozen = !isFrozen && !isNewHead;

                            // Layer height varies by stage
                            const barH = isNewHead ? layerH * 0.4 : layerH * (0.3 + 0.12 * i);
                            const barY = topY + (layerH - barH) / 2;

                            let fillColor, strokeColor, labelColor;
                            if (isNewHead) {
                                fillColor = 'rgba(63,185,160,0.25)';
                                strokeColor = viz.colors.green;
                                labelColor = viz.colors.green;
                            } else if (isFrozen) {
                                fillColor = 'rgba(88,166,255,0.12)';
                                strokeColor = viz.colors.blue;
                                labelColor = viz.colors.blue;
                            } else {
                                fillColor = 'rgba(240,136,62,0.2)';
                                strokeColor = viz.colors.orange;
                                labelColor = viz.colors.orange;
                            }

                            ctx.fillStyle = fillColor;
                            ctx.fillRect(xOffset + 2, barY, w - 4, barH);
                            ctx.strokeStyle = strokeColor;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(xOffset + 2, barY, w - 4, barH);

                            // Frozen indicator (lock icon or snowflake)
                            if (isFrozen) {
                                viz.screenText('\u2744', xOffset + w / 2, barY + barH / 2 - 10, viz.colors.blue, 16);
                            } else if (!isNewHead) {
                                viz.screenText('\u{1F525}', xOffset + w / 2, barY + barH / 2 - 10, viz.colors.orange, 14);
                            }

                            // Stage name
                            viz.screenText(stage.name, xOffset + w / 2, topY + layerH + 20, labelColor, 10);
                            viz.screenText(stage.desc, xOffset + w / 2, topY + layerH + 34, viz.colors.text, 9);

                            // Sublayer lines
                            if (stage.sublayers > 1) {
                                for (let j = 1; j < stage.sublayers; j++) {
                                    const lx = xOffset + j * unitW;
                                    ctx.strokeStyle = strokeColor;
                                    ctx.lineWidth = 0.5;
                                    ctx.globalAlpha = 0.3;
                                    ctx.beginPath();
                                    ctx.moveTo(lx, barY);
                                    ctx.lineTo(lx, barY + barH);
                                    ctx.stroke();
                                    ctx.globalAlpha = 1;
                                }
                            }

                            // Arrows between stages
                            if (i < stages.length - 1) {
                                const arrowX = xOffset + w;
                                ctx.strokeStyle = viz.colors.axis;
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                ctx.moveTo(arrowX, topY + layerH / 2);
                                ctx.lineTo(arrowX + 2, topY + layerH / 2);
                                ctx.stroke();
                            }

                            xOffset += w;
                        }

                        // Legend
                        const legY = topY + layerH + 60;
                        ctx.fillStyle = 'rgba(88,166,255,0.2)';
                        ctx.fillRect(padX + 30, legY, 14, 14);
                        ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 1.5; ctx.strokeRect(padX + 30, legY, 14, 14);
                        viz.screenText('Frozen (pretrained weights kept)', padX + 50, legY + 7, viz.colors.blue, 11, 'left');

                        ctx.fillStyle = 'rgba(240,136,62,0.3)';
                        ctx.fillRect(padX + 300, legY, 14, 14);
                        ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.5; ctx.strokeRect(padX + 300, legY, 14, 14);
                        viz.screenText('Unfrozen (fine-tuned)', padX + 320, legY + 7, viz.colors.orange, 11, 'left');

                        ctx.fillStyle = 'rgba(63,185,160,0.3)';
                        ctx.fillRect(padX + 510, legY, 14, 14);
                        ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1.5; ctx.strokeRect(padX + 510, legY, 14, 14);
                        viz.screenText('New head (trained from scratch)', padX + 530, legY + 7, viz.colors.green, 11, 'left');

                        // Status text
                        const frozenCount = unfreezeFrom;
                        const unfrozenCount = stages.length - 1 - frozenCount; // exclude head
                        let strategy = '';
                        if (unfreezeFrom >= stages.length - 1) {
                            strategy = 'Feature Extraction: only the new head is trained';
                        } else if (unfreezeFrom <= 0) {
                            strategy = 'Full Fine-Tuning: all layers updated (use small LR!)';
                        } else {
                            strategy = 'Partial Fine-Tuning: ' + frozenCount + ' stages frozen, ' + unfrozenCount + ' stages fine-tuned';
                        }
                        viz.screenText(strategy, W / 2, legY + 35, viz.colors.yellow, 12);

                        // Learning rate suggestion
                        let lrAdvice = '';
                        if (unfreezeFrom >= stages.length - 1) {
                            lrAdvice = 'Recommended LR: 1e-3 for head';
                        } else if (unfreezeFrom <= 0) {
                            lrAdvice = 'Recommended: backbone LR=1e-5, head LR=1e-3';
                        } else {
                            lrAdvice = 'Recommended: frozen=0, unfrozen LR=1e-4, head LR=1e-3';
                        }
                        viz.screenText(lrAdvice, W / 2, legY + 52, viz.colors.text, 10);
                    }

                    draw();

                    VizEngine.createSlider(controls, 'Freeze up to stage: ', 0, 6, unfreezeFrom, 1, function(val) {
                        unfreezeFrom = Math.round(val);
                        draw();
                    });

                    return viz;
                }
            }
        ],
        exercises: [
            {
                question: 'You have a dataset of 500 labeled chest X-ray images and want to classify them into 3 disease categories. You have access to a ResNet-50 pretrained on ImageNet. Describe your transfer learning strategy and justify each decision.',
                hint: 'Consider: (a) Is the target domain similar to ImageNet? (b) Is the dataset small or large? (c) What is the risk of overfitting?',
                solution: '<p>This falls into the "different domain, small dataset" quadrant, the hardest case. Strategy:</p><ol><li><strong>Replace the classification head</strong>: Remove the 1000-class FC layer, add a new FC(512, 3) with random initialization.</li><li><strong>Feature extraction first</strong>: Freeze all pretrained layers. Train only the new head for ~10 epochs with LR=1e-3. This establishes a reasonable classifier on pretrained features.</li><li><strong>Gradual unfreezing</strong>: Unfreeze Stage 4, train with LR=1e-4 for the backbone and LR=1e-3 for the head. Then unfreeze Stage 3, reducing backbone LR to 1e-5. Stop here; earlier layers capture low-level features (edges, textures) that are reasonably universal.</li><li><strong>Heavy data augmentation</strong>: Random rotations (\u00b115\u00b0), horizontal flips, random crops, brightness/contrast jitter. Medical images have less variation than natural images, so augmentation is essential.</li><li><strong>Keep BN in eval mode</strong>: With only 500 images and small batch sizes, batch statistics would be too noisy. Use the running mean/variance from ImageNet pretraining.</li></ol><p>Justification: X-rays differ significantly from natural images (grayscale, inverted contrast, specific anatomy), but low-level edge/texture features still transfer. The small dataset makes overfitting the primary risk, so we freeze most layers and rely heavily on augmentation.</p>'
            },
            {
                question: 'Explain discriminative learning rates. If the base learning rate is \\(\\eta = 10^{-3}\\) and the decay factor per layer group is \\(2.6\\), what are the learning rates for a 5-group model (groups numbered 0 to 4, with group 4 being the head)?',
                hint: 'Group \\(i\\) gets learning rate \\(\\eta / 2.6^{(4-i)}\\) where group 4 (the head) gets the full rate.',
                solution: '<p>Discriminative learning rates assign different learning rates to different parts of the network, with later (higher) layers getting larger rates because their features are more task-specific and need more adaptation.</p><p>With \\(\\eta = 10^{-3}\\) and decay factor 2.6, group \\(i\\) gets \\(\\eta / 2.6^{(4-i)}\\):</p><ul><li>Group 0 (earliest): \\(10^{-3} / 2.6^4 = 10^{-3} / 45.7 \\approx 2.19 \\times 10^{-5}\\)</li><li>Group 1: \\(10^{-3} / 2.6^3 = 10^{-3} / 17.6 \\approx 5.69 \\times 10^{-5}\\)</li><li>Group 2: \\(10^{-3} / 2.6^2 = 10^{-3} / 6.76 \\approx 1.48 \\times 10^{-4}\\)</li><li>Group 3: \\(10^{-3} / 2.6^1 \\approx 3.85 \\times 10^{-4}\\)</li><li>Group 4 (head): \\(10^{-3} / 2.6^0 = 10^{-3}\\)</li></ul><p>This creates a 46x spread between the earliest and latest layers, reflecting the intuition that early features (edges, textures) need minimal adaptation while the classification head needs to be learned from scratch.</p>'
            },
            {
                question: 'He et al. (2019) showed that, given sufficient data and training time, training from scratch can match transfer learning performance on object detection. Does this invalidate transfer learning? Discuss both the accuracy perspective and the practical perspective.',
                hint: 'Consider computational cost, convergence speed, and the common scenario where the target dataset is not "sufficient."',
                solution: '<p><strong>Accuracy perspective</strong>: The result shows that ImageNet pretraining is not strictly necessary for achieving high accuracy, <em>given enough target-domain data and training time</em>. The pretrained features do not provide an unreachable advantage; they primarily provide a good initialization. With sufficient data, random initialization can reach the same quality, though possibly a different local optimum.</p><p><strong>Practical perspective</strong>: Transfer learning remains invaluable for several reasons:</p><ol><li><strong>Convergence speed</strong>: Pretrained models converge 3-10x faster, saving significant compute cost.</li><li><strong>Small datasets</strong>: Most real-world applications have far less data than ImageNet. Transfer learning is essential when labeled data is scarce (medical imaging, satellite imagery, rare species classification).</li><li><strong>Compute budget</strong>: Training from scratch requires expensive hyperparameter search; pretrained models are more robust to hyperparameter choices.</li><li><strong>Reproducibility</strong>: Starting from a standard pretrained checkpoint makes results more reproducible.</li></ol><p>The result does not invalidate transfer learning; it clarifies that the benefit is primarily <em>computational and practical</em> rather than a fundamental representational advantage. In the common scenario of limited data and compute, transfer learning remains the dominant strategy.</p>'
            }
        ]
    }

    ]
});
