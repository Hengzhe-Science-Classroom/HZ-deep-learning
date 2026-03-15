// === Chapter 4: Multilayer Perceptron ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch04',
    number: 4,
    title: 'Multilayer Perceptron',
    subtitle: 'Hidden layers, activation functions, and the universal approximation theorem',
    sections: [
        // ===== SECTION 1: MLP Architecture =====
        {
            id: 'sec-mlp-architecture',
            title: 'MLP Architecture',
            content: `
<h2>4.1 MLP Architecture</h2>

<div class="env-block intuition">
<div class="env-title">From Linear to Nonlinear</div>
<div class="env-body">
<p>In Chapter 3 we saw that a single perceptron computes a linear decision boundary. This means it cannot learn the XOR function, cannot separate concentric rings, and more generally cannot model any relationship that is not linearly separable. The multilayer perceptron (MLP) solves this limitation by composing multiple layers of linear transformations interleaved with nonlinear <em>activation functions</em>. Each additional layer grants the network the ability to carve out increasingly complex decision regions.</p>
</div>
</div>

<h3>Layers of an MLP</h3>

<p>An MLP is organized into three types of layers:</p>
<ul>
<li><strong>Input layer:</strong> receives the raw feature vector \\(\\mathbf{x} \\in \\mathbb{R}^d\\). It performs no computation; it simply passes data forward.</li>
<li><strong>Hidden layers:</strong> one or more layers of neurons that transform representations. Each hidden layer applies an affine transformation followed by an elementwise nonlinearity.</li>
<li><strong>Output layer:</strong> produces the final prediction. Its form depends on the task (regression, binary classification, multiclass classification).</li>
</ul>

<div class="env-block definition">
<div class="env-title">Definition 4.1 — Fully Connected (Dense) Layer</div>
<div class="env-body">
<p>A <strong>fully connected layer</strong> maps an input \\(\\mathbf{h} \\in \\mathbb{R}^{n_{\\mathrm{in}}}\\) to an output \\(\\mathbf{h}' \\in \\mathbb{R}^{n_{\\mathrm{out}}}\\) via</p>
\\[\\mathbf{h}' = \\sigma\\bigl(W\\mathbf{h} + \\mathbf{b}\\bigr),\\]
<p>where \\(W \\in \\mathbb{R}^{n_{\\mathrm{out}} \\times n_{\\mathrm{in}}}\\) is the weight matrix, \\(\\mathbf{b} \\in \\mathbb{R}^{n_{\\mathrm{out}}}\\) is the bias vector, and \\(\\sigma\\) is a nonlinear activation function applied elementwise. The term "fully connected" means every input unit is connected to every output unit.</p>
</div>
</div>

<h3>The Forward Pass</h3>

<p>Consider an MLP with \\(L\\) hidden layers. Denote the input as \\(\\mathbf{h}^{(0)} = \\mathbf{x}\\). The forward pass computes layer by layer:</p>

\\[\\mathbf{z}^{(l)} = W^{(l)} \\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad \\mathbf{h}^{(l)} = \\sigma\\bigl(\\mathbf{z}^{(l)}\\bigr), \\qquad l = 1, 2, \\ldots, L.\\]

<p>The vector \\(\\mathbf{z}^{(l)}\\) is called the <strong>pre-activation</strong> and \\(\\mathbf{h}^{(l)}\\) is the <strong>activation</strong> (or hidden representation) at layer \\(l\\). The final output layer then applies a task-specific transformation to \\(\\mathbf{h}^{(L)}\\).</p>

<div class="env-block remark">
<div class="env-title">Notation Convention</div>
<div class="env-body">
<p>We index layers by superscript: \\(W^{(l)}\\) is the weight matrix of layer \\(l\\), which maps \\(\\mathbf{h}^{(l-1)}\\) (the activation of the previous layer, with \\(n_{l-1}\\) units) to \\(\\mathbf{z}^{(l)}\\) (the pre-activation of the current layer, with \\(n_l\\) units). So \\(W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}\\).</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 4.1 — A Two-Hidden-Layer MLP</div>
<div class="env-body">
<p>Suppose \\(\\mathbf{x} \\in \\mathbb{R}^3\\), the first hidden layer has 4 neurons, the second hidden layer has 2 neurons, and the output is a single scalar (regression). The parameter dimensions are:</p>
<ul>
<li>\\(W^{(1)} \\in \\mathbb{R}^{4 \\times 3}\\), \\(\\mathbf{b}^{(1)} \\in \\mathbb{R}^4\\) (12 + 4 = 16 parameters)</li>
<li>\\(W^{(2)} \\in \\mathbb{R}^{2 \\times 4}\\), \\(\\mathbf{b}^{(2)} \\in \\mathbb{R}^2\\) (8 + 2 = 10 parameters)</li>
<li>\\(W^{(3)} \\in \\mathbb{R}^{1 \\times 2}\\), \\(b^{(3)} \\in \\mathbb{R}\\) (2 + 1 = 3 parameters)</li>
</ul>
<p>Total: 29 learnable parameters. The forward pass is:</p>
\\[\\mathbf{h}^{(1)} = \\sigma(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}), \\quad \\mathbf{h}^{(2)} = \\sigma(W^{(2)}\\mathbf{h}^{(1)} + \\mathbf{b}^{(2)}), \\quad \\hat{y} = W^{(3)}\\mathbf{h}^{(2)} + b^{(3)}.\\]
</div>
</div>

<h3>Why the Nonlinearity is Essential</h3>

<div class="env-block proposition">
<div class="env-title">Proposition 4.1 — Collapsing Without Activation</div>
<div class="env-body">
<p>An MLP with \\(L\\) layers but <em>no</em> activation functions (i.e., \\(\\sigma\\) is the identity) collapses to a single affine map. Specifically:</p>
\\[W^{(L)} \\cdots W^{(2)} W^{(1)} \\mathbf{x} + \\text{(bias terms)} = \\tilde{W} \\mathbf{x} + \\tilde{\\mathbf{b}},\\]
<p>where \\(\\tilde{W} = W^{(L)} \\cdots W^{(1)}\\). Adding layers without nonlinearities does not increase the representational power of the model.</p>
</div>
</div>

<div class="env-block proof">
<div class="env-title">Proof</div>
<div class="env-body">
<p>With identity activations, layer \\(l\\) computes \\(\\mathbf{h}^{(l)} = W^{(l)}\\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)}\\). Substituting recursively:</p>
\\[\\mathbf{h}^{(2)} = W^{(2)}(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}) + \\mathbf{b}^{(2)} = W^{(2)}W^{(1)}\\mathbf{x} + W^{(2)}\\mathbf{b}^{(1)} + \\mathbf{b}^{(2)}.\\]
<p>By induction, the entire network computes \\(\\tilde{W}\\mathbf{x} + \\tilde{\\mathbf{b}}\\) for matrices \\(\\tilde{W}\\) and \\(\\tilde{\\mathbf{b}}\\) that are products and sums of the original parameters. The composition of linear maps is linear.</p>
<div class="qed">∎</div>
</div>
</div>

<h3>Parameter Count</h3>

<p>For an MLP with layer widths \\(n_0, n_1, \\ldots, n_L, n_{L+1}\\) (where \\(n_0\\) is the input dimension and \\(n_{L+1}\\) is the output dimension), the total number of parameters is:</p>
\\[\\sum_{l=1}^{L+1} (n_{l-1} \\cdot n_l + n_l) = \\sum_{l=1}^{L+1} n_l(n_{l-1} + 1).\\]
<p>Each term \\(n_l(n_{l-1} + 1)\\) counts the weights plus biases for layer \\(l\\). For wide hidden layers, the parameter count grows quadratically with width, which is both a blessing (expressiveness) and a curse (overfitting risk, computation cost).</p>

<div class="viz-placeholder" data-viz="viz-mlp-architecture"></div>
`,
            visualizations: [
                {
                    id: 'viz-mlp-architecture',
                    title: 'Interactive MLP Architecture',
                    description: 'Adjust the number of hidden neurons to see how the network structure changes. Each edge represents a learnable weight; each node represents a neuron.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        let nHidden = 4;
                        let nInputs = 3;
                        let nOutputs = 2;

                        const sliderH = VizEngine.createSlider(controls, 'Hidden neurons', 1, 12, nHidden, 1, v => { nHidden = Math.round(v); draw(); });
                        const sliderI = VizEngine.createSlider(controls, 'Input dim', 1, 6, nInputs, 1, v => { nInputs = Math.round(v); draw(); });
                        const sliderO = VizEngine.createSlider(controls, 'Output dim', 1, 4, nOutputs, 1, v => { nOutputs = Math.round(v); draw(); });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            const layers = [nInputs, nHidden, nOutputs];
                            const layerNames = ['Input', 'Hidden', 'Output'];
                            const layerColors = [viz.colors.blue, viz.colors.teal, viz.colors.orange];
                            const nLayers = layers.length;
                            const xPad = 100;
                            const xSpacing = (W - 2 * xPad) / (nLayers - 1);
                            const maxN = Math.max(...layers);
                            const nodeRadius = Math.max(8, Math.min(16, H / (maxN * 3)));
                            const yPad = 50;

                            // Compute positions for each neuron
                            const positions = [];
                            for (let l = 0; l < nLayers; l++) {
                                const n = layers[l];
                                const x = xPad + l * xSpacing;
                                const totalH = H - 2 * yPad;
                                const spacing = n > 1 ? totalH / (n - 1) : 0;
                                const startY = n > 1 ? yPad : H / 2;
                                const layerPos = [];
                                for (let i = 0; i < n; i++) {
                                    layerPos.push({ x: x, y: startY + i * spacing });
                                }
                                positions.push(layerPos);
                            }

                            // Draw edges
                            for (let l = 0; l < nLayers - 1; l++) {
                                for (let i = 0; i < positions[l].length; i++) {
                                    for (let j = 0; j < positions[l + 1].length; j++) {
                                        const p1 = positions[l][i];
                                        const p2 = positions[l + 1][j];
                                        ctx.strokeStyle = '#2a2a5a';
                                        ctx.lineWidth = 0.8;
                                        ctx.beginPath();
                                        ctx.moveTo(p1.x, p1.y);
                                        ctx.lineTo(p2.x, p2.y);
                                        ctx.stroke();
                                    }
                                }
                            }

                            // Draw nodes
                            for (let l = 0; l < nLayers; l++) {
                                for (let i = 0; i < positions[l].length; i++) {
                                    const p = positions[l][i];
                                    // Glow
                                    ctx.fillStyle = layerColors[l] + '22';
                                    ctx.beginPath();
                                    ctx.arc(p.x, p.y, nodeRadius + 5, 0, Math.PI * 2);
                                    ctx.fill();
                                    // Node
                                    ctx.fillStyle = layerColors[l];
                                    ctx.beginPath();
                                    ctx.arc(p.x, p.y, nodeRadius, 0, Math.PI * 2);
                                    ctx.fill();
                                    // Highlight
                                    ctx.fillStyle = '#ffffff33';
                                    ctx.beginPath();
                                    ctx.arc(p.x - nodeRadius * 0.2, p.y - nodeRadius * 0.2, nodeRadius * 0.35, 0, Math.PI * 2);
                                    ctx.fill();
                                }
                            }

                            // Layer labels
                            ctx.font = 'bold 13px -apple-system, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (let l = 0; l < nLayers; l++) {
                                const x = xPad + l * xSpacing;
                                ctx.fillStyle = layerColors[l];
                                ctx.fillText(layerNames[l], x, H - 30);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '11px -apple-system, sans-serif';
                                ctx.fillText(layers[l] + ' neuron' + (layers[l] > 1 ? 's' : ''), x, H - 14);
                                ctx.font = 'bold 13px -apple-system, sans-serif';
                            }

                            // Parameter count
                            let totalParams = 0;
                            for (let l = 0; l < nLayers - 1; l++) {
                                totalParams += layers[l] * layers[l + 1] + layers[l + 1];
                            }
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '13px -apple-system, sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Total parameters: ' + totalParams, 14, 14);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system, sans-serif';
                            let paramDetail = '';
                            for (let l = 0; l < nLayers - 1; l++) {
                                const w = layers[l] * layers[l + 1];
                                const b = layers[l + 1];
                                if (l > 0) paramDetail += ' + ';
                                paramDetail += '(' + w + 'W + ' + b + 'b)';
                            }
                            ctx.fillText(paramDetail, 14, 32);
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'An MLP has input dimension 10, two hidden layers of width 64 and 32, and output dimension 5. How many learnable parameters does it have?',
                    hint: 'Count weights and biases for each layer: \\(n_{\\text{in}} \\times n_{\\text{out}} + n_{\\text{out}}\\) per layer.',
                    solution: 'Layer 1: \\(10 \\times 64 + 64 = 704\\). Layer 2: \\(64 \\times 32 + 32 = 2080\\). Layer 3: \\(32 \\times 5 + 5 = 165\\). Total: \\(704 + 2080 + 165 = 2949\\) parameters.'
                },
                {
                    question: 'Prove that an MLP with two hidden layers and no activation functions has the same representational power as a single linear layer (i.e., it can only represent affine functions).',
                    hint: 'Write out the composition and use the fact that the product of matrices is a matrix.',
                    solution: 'Let \\(f(\\mathbf{x}) = W^{(3)}(W^{(2)}(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}) + \\mathbf{b}^{(2)}) + \\mathbf{b}^{(3)}\\). Expanding: \\(f(\\mathbf{x}) = W^{(3)}W^{(2)}W^{(1)}\\mathbf{x} + W^{(3)}W^{(2)}\\mathbf{b}^{(1)} + W^{(3)}\\mathbf{b}^{(2)} + \\mathbf{b}^{(3)} = \\tilde{W}\\mathbf{x} + \\tilde{\\mathbf{b}}\\), which is a single affine map. The three layers of weights collapse into one.'
                },
                {
                    question: 'If we have an MLP with hidden layer widths \\(n_1 &lt; n_0\\) where \\(n_0\\) is the input dimension, what does this mean geometrically for the hidden representation?',
                    hint: 'Think about what happens to the dimensionality of the data as it passes through the layer.',
                    solution: 'A layer mapping from \\(\\mathbb{R}^{n_0}\\) to \\(\\mathbb{R}^{n_1}\\) with \\(n_1 &lt; n_0\\) acts as a <em>dimensionality reduction</em> (or "bottleneck"). The affine map \\(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}\\) projects the input into a lower-dimensional subspace. Information that lies in the null space of \\(W^{(1)}\\) is lost. After the activation function, the representation lives on a nonlinear manifold embedded in \\(\\mathbb{R}^{n_1}\\). This forces the network to learn a compressed representation of the input.'
                }
            ]
        },

        // ===== SECTION 2: Activation Functions =====
        {
            id: 'sec-activation-functions',
            title: 'Activation Functions',
            content: `
<h2>4.2 Activation Functions</h2>

<div class="env-block intuition">
<div class="env-title">The Gatekeepers of Nonlinearity</div>
<div class="env-body">
<p>The activation function \\(\\sigma\\) is the ingredient that makes an MLP more powerful than a single linear model. Without it, as Proposition 4.1 showed, stacking layers is pointless. The choice of activation function affects training dynamics (gradient flow), representational capacity, and computational cost. Over the decades, the field has progressed from sigmoid and tanh to ReLU and its modern variants.</p>
</div>
</div>

<h3>Classical Activations</h3>

<div class="env-block definition">
<div class="env-title">Definition 4.2 — Sigmoid (Logistic) Function</div>
<div class="env-body">
<p>The <strong>sigmoid</strong> function is</p>
\\[\\sigma(z) = \\frac{1}{1 + e^{-z}}.\\]
<p>It maps \\(\\mathbb{R} \\to (0,1)\\). Its derivative is \\(\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))\\), which has a maximum of \\(1/4\\) at \\(z = 0\\).</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Historical Significance</div>
<div class="env-body">
<p>Sigmoid was historically motivated by biological plausibility (modeling neuron firing rates) and by its role as the canonical link function in logistic regression. However, it has two major drawbacks for deep networks: (1) <strong>vanishing gradients</strong>, since \\(|\\sigma'(z)| \\leq 1/4\\), meaning gradients shrink by at least a factor of 4 per layer during backpropagation; and (2) <strong>non-zero-centered outputs</strong>, since \\(\\sigma(z) &gt; 0\\) for all \\(z\\), which can slow optimization by introducing systematic bias in the gradient updates.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 4.3 — Hyperbolic Tangent (tanh)</div>
<div class="env-body">
<p>The <strong>tanh</strong> function is</p>
\\[\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\\sigma(2z) - 1.\\]
<p>It maps \\(\\mathbb{R} \\to (-1,1)\\). Its derivative is \\(\\tanh'(z) = 1 - \\tanh^2(z)\\), which has a maximum of 1 at \\(z = 0\\).</p>
</div>
</div>

<p>Tanh addresses the zero-centering issue (its outputs are symmetric around zero), and its gradient at the origin is larger than sigmoid's. Before ReLU, tanh was the default choice for hidden layers. It still suffers from gradient saturation in the tails: for \\(|z| \\gg 0\\), \\(\\tanh'(z) \\approx 0\\).</p>

<h3>Modern Activations</h3>

<div class="env-block definition">
<div class="env-title">Definition 4.4 — Rectified Linear Unit (ReLU)</div>
<div class="env-body">
<p>The <strong>ReLU</strong> function is</p>
\\[\\mathrm{ReLU}(z) = \\max(0, z) = \\begin{cases} z & \\text{if } z \\geq 0, \\\\ 0 & \\text{if } z &lt; 0. \\end{cases}\\]
<p>Its derivative is \\(\\mathrm{ReLU}'(z) = \\mathbf{1}_{z &gt; 0}\\) (1 for positive inputs, 0 otherwise; the derivative at 0 is conventionally set to 0 or 1).</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Why ReLU Dominates</div>
<div class="env-body">
<p>ReLU (Nair &amp; Hinton, 2010; Glorot, Bordes &amp; Bengio, 2011) has become the default activation for most deep networks because:</p>
<ul>
<li><strong>No saturation for positive inputs:</strong> the gradient is exactly 1 for \\(z &gt; 0\\), preventing the vanishing gradient problem in positive regions.</li>
<li><strong>Computational efficiency:</strong> both the forward and backward passes involve only comparisons and multiplications; no exponentials needed.</li>
<li><strong>Sparsity:</strong> for typical inputs, roughly half the neurons output zero, yielding sparse representations that can improve efficiency and generalization.</li>
</ul>
<p>The main drawback is the <strong>dying ReLU problem</strong>: if a neuron's pre-activation becomes permanently negative (e.g., due to a large negative bias learned during training), it outputs zero forever and its gradient is zero, so it can never recover.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 4.5 — Leaky ReLU</div>
<div class="env-body">
<p>The <strong>Leaky ReLU</strong> with slope \\(\\alpha &gt; 0\\) (typically \\(\\alpha = 0.01\\)) is</p>
\\[\\mathrm{LeakyReLU}(z) = \\begin{cases} z & \\text{if } z \\geq 0, \\\\ \\alpha z & \\text{if } z &lt; 0. \\end{cases}\\]
<p>This fixes the dying ReLU problem by allowing a small gradient for negative inputs.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 4.6 — GELU (Gaussian Error Linear Unit)</div>
<div class="env-body">
<p>The <strong>GELU</strong> (Hendrycks &amp; Gimpel, 2016) is</p>
\\[\\mathrm{GELU}(z) = z \\cdot \\Phi(z),\\]
<p>where \\(\\Phi(z)\\) is the CDF of the standard normal distribution. A practical approximation is</p>
\\[\\mathrm{GELU}(z) \\approx 0.5\\, z \\left(1 + \\tanh\\!\\left[\\sqrt{\\frac{2}{\\pi}}\\left(z + 0.044715\\, z^3\\right)\\right]\\right).\\]
<p>GELU smoothly gates the input: values likely to be positive (under a Gaussian prior) pass through, while those likely negative are suppressed. It is the default activation in GPT, BERT, and most modern transformer architectures.</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 4.7 — Swish (SiLU)</div>
<div class="env-body">
<p>The <strong>Swish</strong> function (Ramachandran, Zoph &amp; Le, 2017), also called SiLU, is</p>
\\[\\mathrm{Swish}(z) = z \\cdot \\sigma(z) = \\frac{z}{1 + e^{-z}}.\\]
<p>Its derivative is \\(\\mathrm{Swish}'(z) = \\sigma(z) + z\\,\\sigma(z)(1 - \\sigma(z)) = \\sigma(z)(1 + z(1-\\sigma(z)))\\). Like GELU, it is smooth, non-monotone (with a slight negative dip for \\(z &lt; 0\\)), and self-gated. Swish has been shown to outperform ReLU in many deep network experiments.</p>
</div>
</div>

<h3>Comparison at a Glance</h3>

<table style="width:100%;border-collapse:collapse;margin:16px 0;">
<tr style="border-bottom:2px solid #30363d;">
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Activation</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Range</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Smooth?</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Zero-centered?</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Saturates?</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">Sigmoid</td>
<td style="padding:6px;">\\((0,1)\\)</td>
<td style="padding:6px;">Yes</td>
<td style="padding:6px;">No</td>
<td style="padding:6px;">Both tails</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">Tanh</td>
<td style="padding:6px;">\\((-1,1)\\)</td>
<td style="padding:6px;">Yes</td>
<td style="padding:6px;">Yes</td>
<td style="padding:6px;">Both tails</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">ReLU</td>
<td style="padding:6px;">\\([0, \\infty)\\)</td>
<td style="padding:6px;">No</td>
<td style="padding:6px;">No</td>
<td style="padding:6px;">Left only</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">Leaky ReLU</td>
<td style="padding:6px;">\\((-\\infty, \\infty)\\)</td>
<td style="padding:6px;">No</td>
<td style="padding:6px;">Approx.</td>
<td style="padding:6px;">No</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">GELU</td>
<td style="padding:6px;">\\(\\approx [-0.17, \\infty)\\)</td>
<td style="padding:6px;">Yes</td>
<td style="padding:6px;">Approx.</td>
<td style="padding:6px;">Left (soft)</td>
</tr>
<tr>
<td style="padding:6px;">Swish</td>
<td style="padding:6px;">\\(\\approx [-0.28, \\infty)\\)</td>
<td style="padding:6px;">Yes</td>
<td style="padding:6px;">Approx.</td>
<td style="padding:6px;">Left (soft)</td>
</tr>
</table>

<div class="viz-placeholder" data-viz="viz-activation-gallery"></div>
`,
            visualizations: [
                {
                    id: 'viz-activation-gallery',
                    title: 'Activation Function Gallery',
                    description: 'Toggle between activation functions and their derivatives. Each function is drawn with its derivative side by side.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50, originX: 0, originY: 0 });
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        const activations = {
                            'Sigmoid': {
                                fn: z => 1 / (1 + Math.exp(-z)),
                                dfn: z => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); },
                                color: '#58a6ff', range: [-6, 6], yRange: [-0.5, 1.5]
                            },
                            'Tanh': {
                                fn: z => Math.tanh(z),
                                dfn: z => 1 - Math.tanh(z) ** 2,
                                color: '#3fb9a0', range: [-5, 5], yRange: [-1.5, 1.5]
                            },
                            'ReLU': {
                                fn: z => Math.max(0, z),
                                dfn: z => z > 0 ? 1 : 0,
                                color: '#f0883e', range: [-4, 4], yRange: [-1, 4]
                            },
                            'Leaky ReLU': {
                                fn: z => z >= 0 ? z : 0.1 * z,
                                dfn: z => z >= 0 ? 1 : 0.1,
                                color: '#d29922', range: [-4, 4], yRange: [-1, 4]
                            },
                            'GELU': {
                                fn: z => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z * z * z))),
                                dfn: z => {
                                    const a = Math.sqrt(2 / Math.PI);
                                    const t = a * (z + 0.044715 * z * z * z);
                                    const th = Math.tanh(t);
                                    const dtdz = a * (1 + 3 * 0.044715 * z * z);
                                    return 0.5 * (1 + th) + 0.5 * z * (1 - th * th) * dtdz;
                                },
                                color: '#bc8cff', range: [-5, 5], yRange: [-1, 4]
                            },
                            'Swish': {
                                fn: z => z / (1 + Math.exp(-z)),
                                dfn: z => {
                                    const s = 1 / (1 + Math.exp(-z));
                                    return s + z * s * (1 - s);
                                },
                                color: '#f778ba', range: [-5, 5], yRange: [-1, 4]
                            }
                        };

                        let current = 'Sigmoid';
                        let showDeriv = true;

                        // Create buttons
                        const names = Object.keys(activations);
                        const btnContainer = document.createElement('div');
                        btnContainer.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;';
                        const buttons = {};
                        names.forEach(name => {
                            const b = document.createElement('button');
                            b.style.cssText = 'padding:3px 10px;border:1px solid #30363d;border-radius:4px;background:#1a1a40;color:#c9d1d9;font-size:0.75rem;cursor:pointer;';
                            b.textContent = name;
                            b.addEventListener('click', () => { current = name; draw(); });
                            btnContainer.appendChild(b);
                            buttons[name] = b;
                        });
                        controls.appendChild(btnContainer);

                        const derivBtn = VizEngine.createButton(controls, 'Toggle derivative', () => {
                            showDeriv = !showDeriv;
                            derivBtn.textContent = showDeriv ? 'Hide derivative' : 'Show derivative';
                            draw();
                        });

                        function mapX(val, xMin, xMax, pxMin, pxMax) {
                            return pxMin + (val - xMin) / (xMax - xMin) * (pxMax - pxMin);
                        }
                        function mapY(val, yMin, yMax, pxMin, pxMax) {
                            return pxMax - (val - yMin) / (yMax - yMin) * (pxMax - pxMin);
                        }

                        function drawPlot(fn, color, xRange, yRange, px0, py0, pw, ph, label) {
                            const xMin = xRange[0], xMax = xRange[1];
                            const yMin = yRange[0], yMax = yRange[1];

                            // Background
                            ctx.fillStyle = '#0e0e28';
                            ctx.fillRect(px0, py0, pw, ph);

                            // Grid lines
                            ctx.strokeStyle = '#1a1a40';
                            ctx.lineWidth = 0.5;
                            for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
                                const py = mapY(y, yMin, yMax, py0, py0 + ph);
                                ctx.beginPath(); ctx.moveTo(px0, py); ctx.lineTo(px0 + pw, py); ctx.stroke();
                            }
                            for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
                                const px = mapX(x, xMin, xMax, px0, px0 + pw);
                                ctx.beginPath(); ctx.moveTo(px, py0); ctx.lineTo(px, py0 + ph); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = '#4a4a7a';
                            ctx.lineWidth = 1;
                            const axisY = mapY(0, yMin, yMax, py0, py0 + ph);
                            if (axisY >= py0 && axisY <= py0 + ph) {
                                ctx.beginPath(); ctx.moveTo(px0, axisY); ctx.lineTo(px0 + pw, axisY); ctx.stroke();
                            }
                            const axisX = mapX(0, xMin, xMax, px0, px0 + pw);
                            if (axisX >= px0 && axisX <= px0 + pw) {
                                ctx.beginPath(); ctx.moveTo(axisX, py0); ctx.lineTo(axisX, py0 + ph); ctx.stroke();
                            }

                            // Tick labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
                                if (x === 0) continue;
                                const px = mapX(x, xMin, xMax, px0, px0 + pw);
                                if (axisY >= py0 && axisY <= py0 + ph) {
                                    ctx.fillText(x, px, Math.min(axisY + 3, py0 + ph - 12));
                                }
                            }
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
                                if (y === 0) continue;
                                const py = mapY(y, yMin, yMax, py0, py0 + ph);
                                if (axisX >= px0 && axisX <= px0 + pw) {
                                    ctx.fillText(y, Math.max(axisX - 4, px0 + 18), py);
                                }
                            }

                            // Draw function
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            const steps = pw;
                            for (let i = 0; i <= steps; i++) {
                                const t = i / steps;
                                const xv = xMin + t * (xMax - xMin);
                                const yv = fn(xv);
                                const px = px0 + i;
                                const py = mapY(yv, yMin, yMax, py0, py0 + ph);
                                const pyc = Math.max(py0, Math.min(py0 + ph, py));
                                if (i === 0) ctx.moveTo(px, pyc);
                                else ctx.lineTo(px, pyc);
                            }
                            ctx.stroke();

                            // Label
                            ctx.fillStyle = color;
                            ctx.font = 'bold 13px -apple-system, sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText(label, px0 + 8, py0 + 8);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Highlight selected button
                            names.forEach(name => {
                                const b = buttons[name];
                                if (name === current) {
                                    b.style.background = activations[name].color + '33';
                                    b.style.borderColor = activations[name].color;
                                    b.style.color = activations[name].color;
                                } else {
                                    b.style.background = '#1a1a40';
                                    b.style.borderColor = '#30363d';
                                    b.style.color = '#c9d1d9';
                                }
                            });

                            const act = activations[current];
                            const margin = 20;
                            const gap = 30;

                            if (showDeriv) {
                                const pw = (W - 2 * margin - gap) / 2;
                                const ph = H - 2 * margin;
                                drawPlot(act.fn, act.color, act.range, act.yRange, margin, margin, pw, ph, current + '(z)');
                                drawPlot(act.dfn, act.color + 'bb', act.range, [-0.5, 1.5], margin + pw + gap, margin, pw, ph, current + "'(z)");
                            } else {
                                const pw = W - 2 * margin;
                                const ph = H - 2 * margin;
                                drawPlot(act.fn, act.color, act.range, act.yRange, margin, margin, pw, ph, current + '(z)');
                            }
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that \\(\\tanh(z) = 2\\sigma(2z) - 1\\), where \\(\\sigma\\) is the sigmoid function.',
                    hint: 'Write out both sides explicitly and simplify.',
                    solution: '\\(2\\sigma(2z) - 1 = \\frac{2}{1+e^{-2z}} - 1 = \\frac{2 - (1+e^{-2z})}{1+e^{-2z}} = \\frac{1 - e^{-2z}}{1 + e^{-2z}}\\). Multiply numerator and denominator by \\(e^z\\): \\(= \\frac{e^z - e^{-z}}{e^z + e^{-z}} = \\tanh(z)\\).'
                },
                {
                    question: 'Compute the derivative of the Swish function \\(f(z) = z\\sigma(z)\\) and show that it can be written as \\(f\'(z) = f(z) + \\sigma(z)(1 - f(z))\\).',
                    hint: 'Use the product rule: \\((z\\sigma(z))\' = \\sigma(z) + z\\sigma\'(z) = \\sigma(z) + z\\sigma(z)(1-\\sigma(z))\\).',
                    solution: 'By the product rule: \\(f\'(z) = \\sigma(z) + z\\sigma\'(z) = \\sigma(z) + z\\sigma(z)(1-\\sigma(z))\\). Now \\(f(z) = z\\sigma(z)\\), so \\(z\\sigma(z)(1-\\sigma(z)) = f(z)(1-\\sigma(z))\\). Thus \\(f\'(z) = \\sigma(z) + f(z)(1-\\sigma(z)) = \\sigma(z) + f(z) - f(z)\\sigma(z) = f(z) + \\sigma(z)(1 - f(z))\\).'
                },
                {
                    question: 'The "dying ReLU" problem occurs when a neuron always outputs 0. Explain why Leaky ReLU and GELU do not suffer from this issue. What is the gradient of Leaky ReLU at \\(z = -10\\)?',
                    hint: 'Think about what happens to the gradient when the pre-activation is negative.',
                    solution: 'For ReLU, if \\(z &lt; 0\\) then \\(\\mathrm{ReLU}\'(z) = 0\\), so the neuron receives zero gradient and cannot update its weights; it is "dead." For Leaky ReLU, \\(\\mathrm{LeakyReLU}\'(z) = \\alpha &gt; 0\\) when \\(z &lt; 0\\), so there is always a nonzero gradient. At \\(z = -10\\): \\(\\mathrm{LeakyReLU}\'(-10) = 0.01\\) (or whatever \\(\\alpha\\) is). For GELU, since it is smooth and \\(\\mathrm{GELU}\'(z) = \\Phi(z) + z\\phi(z) &gt; 0\\) for a range of negative \\(z\\), there is always a small gradient for moderately negative inputs, allowing the neuron to recover.'
                }
            ]
        },

        // ===== SECTION 3: Hidden Layer Representations =====
        {
            id: 'sec-hidden-representations',
            title: 'Hidden Layer Representations',
            content: `
<h2>4.3 Hidden Layer Representations</h2>

<div class="env-block intuition">
<div class="env-title">Learning to See</div>
<div class="env-body">
<p>One of the most powerful ideas in deep learning is that each hidden layer transforms the raw input into a progressively more useful <em>representation</em>. A first hidden layer might learn to detect simple features (edges, thresholds); a second layer combines these into more complex patterns; and so on. This hierarchical feature extraction is what allows deep networks to solve problems that are intractable for linear models.</p>
</div>
</div>

<h3>What a Single Hidden Layer Does</h3>

<p>Consider a two-class classification problem in \\(\\mathbb{R}^2\\) where the data is not linearly separable (e.g., spiral data). A single-layer perceptron fails because it can only draw a single linear boundary. An MLP with one hidden layer, however, proceeds in two stages:</p>
<ol>
<li><strong>Feature extraction:</strong> the hidden layer computes \\(\\mathbf{h} = \\sigma(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)})\\). Each hidden neuron \\(h_j\\) fires based on which side of a hyperplane \\(\\mathbf{w}_j^T \\mathbf{x} + b_j = 0\\) the input lies. With ReLU, this amounts to computing a "distance from boundary" for positive inputs.</li>
<li><strong>Linear classification in feature space:</strong> the output layer applies a linear classifier to \\(\\mathbf{h}\\). The hope is that in the new coordinate system defined by \\(\\mathbf{h}\\), the classes become linearly separable.</li>
</ol>

<div class="env-block proposition">
<div class="env-title">Proposition 4.2 — Representational Geometry</div>
<div class="env-body">
<p>A single hidden layer with \\(n_1\\) ReLU neurons partitions the input space into at most \\(\\sum_{k=0}^{d} \\binom{n_1}{k}\\) linear regions, where \\(d\\) is the input dimension. Within each region, the network output is an affine function of the input. More hidden neurons create a finer partition, enabling the approximation of more complex decision boundaries.</p>
</div>
</div>

<h3>More Neurons, Richer Boundaries</h3>

<p>With just 2 hidden neurons, the network can form a simple "wedge" or "strip" decision region. With 4, it can approximate convex regions. With many more, it can trace out arbitrarily complex boundaries, including the interleaved arms of a spiral. The visualization below demonstrates this progression: as you increase the number of hidden neurons, the decision boundary becomes increasingly flexible.</p>

<div class="env-block remark">
<div class="env-title">The Manifold Hypothesis</div>
<div class="env-body">
<p>A deeper theoretical perspective is the <strong>manifold hypothesis</strong>: real-world data (images, text, audio) lies on or near low-dimensional manifolds embedded in high-dimensional space. A well-trained MLP learns a sequence of transformations that "unfold" these manifolds, mapping tangled, nonlinear structures in input space to linearly separable configurations in hidden-layer space. This is why adding more hidden neurons (and more layers) generally helps, up to the point where overfitting becomes a concern.</p>
</div>
</div>

<h3>Multi-Layer Feature Hierarchies</h3>

<p>Adding a second hidden layer allows the network to compose the features learned by the first layer. Consider classifying images of handwritten digits:</p>
<ul>
<li><strong>Layer 1</strong> might learn edge detectors and stroke fragments.</li>
<li><strong>Layer 2</strong> combines edges into loops, corners, and curves.</li>
<li><strong>Output layer</strong> combines these higher-level features into digit classifications.</li>
</ul>

<div class="env-block theorem">
<div class="env-title">Theorem 4.1 — Depth Separation (Informal)</div>
<div class="env-body">
<p>There exist functions that can be represented by a network with \\(L\\) layers and \\(O(n)\\) neurons per layer, but require \\(\\Omega(2^n)\\) neurons if only a single hidden layer is used (Telgarsky, 2016; Eldan &amp; Shamir, 2016). In other words, depth provides an exponential gain in representational efficiency for certain function classes. This is a key theoretical justification for deep (as opposed to merely wide) architectures.</p>
</div>
</div>

<div class="env-block example">
<div class="env-title">Example 4.2 — The "Sawtooth" Function</div>
<div class="env-body">
<p>Consider the function that maps \\([0,1] \\to [0,1]\\) by applying \\(g(x) = 2\\min(x, 1-x)\\) (a triangle wave) repeatedly \\(k\\) times: \\(f = g \\circ g \\circ \\cdots \\circ g\\). The resulting function oscillates \\(2^k\\) times. A single-hidden-layer ReLU network needs \\(O(2^k)\\) neurons to represent this exactly, but a network with \\(k\\) layers needs only \\(O(k)\\) neurons total, because each layer can implement one application of \\(g\\) with just 2 neurons.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-spiral-classification"></div>
`,
            visualizations: [
                {
                    id: 'viz-spiral-classification',
                    title: 'MLP Decision Boundary on Spiral Data',
                    description: 'Adjust the number of hidden neurons and observe how the decision boundary changes. More neurons allow the MLP to capture the spiral structure. Click "Retrain" to re-initialize and train a fresh network.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        let nHidden = 10;
                        let training = false;
                        let weights = null;
                        let epoch = 0;

                        const slider = VizEngine.createSlider(controls, 'Hidden neurons', 2, 30, nHidden, 1, v => {
                            nHidden = Math.round(v);
                        });
                        const trainBtn = VizEngine.createButton(controls, 'Train network', () => {
                            initAndTrain();
                        });

                        // Generate spiral data
                        function generateSpiral(n, noise) {
                            const points = [];
                            for (let c = 0; c < 2; c++) {
                                for (let i = 0; i < n; i++) {
                                    const r = i / n * 1.0;
                                    const t = c * Math.PI + i / n * Math.PI * 2.5 + (Math.random() - 0.5) * noise;
                                    points.push({
                                        x: r * Math.cos(t),
                                        y: r * Math.sin(t),
                                        label: c
                                    });
                                }
                            }
                            return points;
                        }

                        const data = generateSpiral(100, 0.3);

                        // Simple MLP: 2 -> nHidden (ReLU) -> 1 (sigmoid)
                        function initWeights(nH) {
                            const w1 = [], b1 = [], w2 = [], b2 = [0];
                            for (let j = 0; j < nH; j++) {
                                w1.push([(Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2]);
                                b1.push((Math.random() - 0.5) * 0.5);
                                w2.push((Math.random() - 0.5) * 2);
                            }
                            return { w1, b1, w2, b2, nH };
                        }

                        function forward(x0, x1, wt) {
                            const h = [];
                            for (let j = 0; j < wt.nH; j++) {
                                const z = wt.w1[j][0] * x0 + wt.w1[j][1] * x1 + wt.b1[j];
                                h.push(Math.max(0, z)); // ReLU
                            }
                            let out = wt.b2[0];
                            for (let j = 0; j < wt.nH; j++) {
                                out += wt.w2[j] * h[j];
                            }
                            const sig = 1 / (1 + Math.exp(-out));
                            return { h, out, sig };
                        }

                        function trainStep(wt, lr) {
                            // Mini-batch SGD
                            for (let idx = 0; idx < data.length; idx++) {
                                const p = data[idx];
                                const { h, out, sig } = forward(p.x, p.y, wt);
                                const err = sig - p.label;

                                // dL/dout = err (cross-entropy with sigmoid)
                                const dOut = err;

                                // Output layer gradients
                                for (let j = 0; j < wt.nH; j++) {
                                    wt.w2[j] -= lr * dOut * h[j];
                                }
                                wt.b2[0] -= lr * dOut;

                                // Hidden layer gradients
                                for (let j = 0; j < wt.nH; j++) {
                                    const z = wt.w1[j][0] * p.x + wt.w1[j][1] * p.y + wt.b1[j];
                                    const dRelu = z > 0 ? 1 : 0;
                                    const dH = dOut * wt.w2[j] * dRelu;
                                    wt.w1[j][0] -= lr * dH * p.x;
                                    wt.w1[j][1] -= lr * dH * p.y;
                                    wt.b1[j] -= lr * dH;
                                }
                            }
                        }

                        function drawState() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            if (!weights) {
                                // Draw just the data
                                const margin = 40;
                                const plotW = W - 2 * margin;
                                const plotH = H - 2 * margin;
                                const scale = Math.min(plotW, plotH) / 2.4;
                                const cx = W / 2, cy = H / 2;

                                data.forEach(p => {
                                    const sx = cx + p.x * scale;
                                    const sy = cy - p.y * scale;
                                    ctx.fillStyle = p.label === 0 ? viz.colors.blue : viz.colors.orange;
                                    ctx.beginPath();
                                    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                                    ctx.fill();
                                });

                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '13px -apple-system, sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('Click "Train network" to begin', W / 2, 20);
                                return;
                            }

                            // Draw decision boundary as colored grid
                            const margin = 40;
                            const plotW = W - 2 * margin;
                            const plotH = H - 2 * margin;
                            const scale = Math.min(plotW, plotH) / 2.4;
                            const cx = W / 2, cy = H / 2;
                            const res = 3;

                            for (let px = margin; px < W - margin; px += res) {
                                for (let py = margin; py < H - margin; py += res) {
                                    const xv = (px - cx) / scale;
                                    const yv = -(py - cy) / scale;
                                    const { sig } = forward(xv, yv, weights);
                                    const r = Math.round(sig * 255);
                                    const b = Math.round((1 - sig) * 255);
                                    ctx.fillStyle = 'rgba(' + r + ',' + Math.round(40 + sig * 40) + ',' + b + ',0.3)';
                                    ctx.fillRect(px, py, res, res);
                                }
                            }

                            // Draw data points
                            data.forEach(p => {
                                const sx = cx + p.x * scale;
                                const sy = cy - p.y * scale;
                                ctx.fillStyle = p.label === 0 ? viz.colors.blue : viz.colors.orange;
                                ctx.strokeStyle = '#000';
                                ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.stroke();
                            });

                            // Accuracy
                            let correct = 0;
                            data.forEach(p => {
                                const { sig } = forward(p.x, p.y, weights);
                                if ((sig >= 0.5 ? 1 : 0) === p.label) correct++;
                            });
                            const acc = (correct / data.length * 100).toFixed(1);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '13px -apple-system, sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Epoch: ' + epoch + '  |  Accuracy: ' + acc + '%  |  Hidden: ' + weights.nH, 14, 14);
                        }

                        function initAndTrain() {
                            if (training) return;
                            weights = initWeights(nHidden);
                            epoch = 0;
                            training = true;
                            trainBtn.textContent = 'Training...';

                            function step() {
                                if (!training) return;
                                for (let i = 0; i < 5; i++) {
                                    trainStep(weights, 0.02);
                                    epoch++;
                                }
                                drawState();
                                if (epoch < 500) {
                                    requestAnimationFrame(step);
                                } else {
                                    training = false;
                                    trainBtn.textContent = 'Retrain';
                                    drawState();
                                }
                            }
                            step();
                        }

                        drawState();

                        return {
                            stopAnimation() { training = false; }
                        };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A single ReLU neuron in 2D computes \\(h = \\max(0, w_1 x_1 + w_2 x_2 + b)\\). Describe the geometric meaning of this computation.',
                    hint: 'What does the hyperplane \\(w_1 x_1 + w_2 x_2 + b = 0\\) represent? What happens on each side?',
                    solution: 'The hyperplane \\(w_1 x_1 + w_2 x_2 + b = 0\\) divides \\(\\mathbb{R}^2\\) into two half-planes. On the positive side (where \\(w_1 x_1 + w_2 x_2 + b &gt; 0\\)), the neuron outputs the signed distance to the boundary (scaled by \\(\\|\\mathbf{w}\\|\\)). On the negative side, it outputs zero. So a single ReLU neuron creates a "ramp" that rises linearly from zero on one side of a line, acting as a half-space indicator with a soft (linear) transition.'
                },
                {
                    question: 'Why might a very wide single-hidden-layer network (say 10,000 neurons) still perform worse in practice than a narrower but deeper network (say 3 layers of 100 neurons each), even though the wide network has more parameters?',
                    hint: 'Think about the depth-separation result (Theorem 4.1) and about how optimization works in practice.',
                    solution: 'Three key reasons: (1) <em>Depth separation</em>: certain functions require exponentially many neurons in a shallow network but only polynomially many in a deep one. A 3-layer network can compose features hierarchically, representing the same function much more compactly. (2) <em>Optimization</em>: a very wide single layer creates a huge parameter space that may be harder to navigate with gradient descent, while deeper networks often have smoother loss landscapes (locally) due to hierarchical structure. (3) <em>Generalization</em>: the wide network may overfit more easily because it has more free parameters without the structural inductive bias that depth provides. Deep networks implicitly regularize through compositional structure.'
                },
                {
                    question: 'Suppose we have 2D input data from two classes arranged in concentric circles (inner circle = class 0, outer ring = class 1). What is the minimum number of hidden ReLU neurons in a single hidden layer needed to separate these classes?',
                    hint: 'Think about how many linear boundaries are needed to approximate a circle. The answer is related to polygon approximation.',
                    solution: 'A circle cannot be exactly represented by finitely many ReLU neurons, but it can be approximated. Each ReLU neuron contributes one linear boundary. To form a convex polygon that separates the inner circle from the outer ring, we need at least 3 neurons (forming a triangle), but for good approximation we need more. In theory, \\(k\\) hidden neurons can form a convex polygon with at most \\(k\\) sides. A regular \\(k\\)-gon inscribed in the circle boundary becomes a better approximation as \\(k\\) grows. In practice, about 6-10 neurons suffice for a reasonable circular boundary, and the output layer combines these half-spaces to form the enclosed region.'
                }
            ]
        },

        // ===== SECTION 4: Universal Approximation Theorem =====
        {
            id: 'sec-universal-approximation',
            title: 'Universal Approximation Theorem',
            content: `
<h2>4.4 Universal Approximation Theorem</h2>

<div class="env-block intuition">
<div class="env-title">MLPs Can Learn (Almost) Anything</div>
<div class="env-body">
<p>One of the most celebrated results in neural network theory states that a single-hidden-layer MLP can approximate any continuous function to arbitrary accuracy, provided it has enough hidden neurons. This is the <strong>Universal Approximation Theorem</strong> (UAT). It provides a strong theoretical foundation for using MLPs as general-purpose function approximators, though it says nothing about how many neurons are needed or how to find the right weights.</p>
</div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 4.2 — Universal Approximation (Cybenko, 1989; Hornik, 1991)</div>
<div class="env-body">
<p>Let \\(\\sigma: \\mathbb{R} \\to \\mathbb{R}\\) be a continuous, non-constant activation function. Let \\(f: [0,1]^d \\to \\mathbb{R}\\) be any continuous function, and let \\(\\varepsilon &gt; 0\\). Then there exist an integer \\(n\\), weights \\(\\{w_{ij}\\}\\), biases \\(\\{b_j\\}\\), and output weights \\(\\{\\alpha_j\\}\\) such that the function</p>
\\[F(\\mathbf{x}) = \\sum_{j=1}^{n} \\alpha_j \\, \\sigma\\!\\left(\\sum_{i=1}^{d} w_{ij} x_i + b_j\\right)\\]
<p>satisfies \\(\\sup_{\\mathbf{x} \\in [0,1]^d} |F(\\mathbf{x}) - f(\\mathbf{x})| &lt; \\varepsilon\\).</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Versions of the Theorem</div>
<div class="env-body">
<p>The theorem has been proved under various conditions on \\(\\sigma\\):</p>
<ul>
<li><strong>Cybenko (1989):</strong> \\(\\sigma\\) is sigmoid-like (sigmoidal: \\(\\sigma(z) \\to 1\\) as \\(z \\to +\\infty\\) and \\(\\sigma(z) \\to 0\\) as \\(z \\to -\\infty\\)).</li>
<li><strong>Hornik (1991):</strong> any continuous, non-constant \\(\\sigma\\) suffices.</li>
<li><strong>Leshno et al. (1993):</strong> any non-polynomial \\(\\sigma\\) suffices (in the \\(L^p\\) sense). This notably includes ReLU, even though it is not continuous everywhere differentiable or bounded.</li>
</ul>
</div>
</div>

<h3>Intuition: How the Approximation Works</h3>

<p>For sigmoid activations, the key idea is that a sigmoid neuron \\(\\sigma(w x + b)\\) can be made into an approximate step function by taking \\(w\\) very large. Two such step functions with opposite signs can form a "bump." By combining many bumps of different heights and positions, we can approximate any continuous function, much like a Riemann sum approximates an integral.</p>

<div class="env-block example">
<div class="env-title">Example 4.3 — Approximating a Bump</div>
<div class="env-body">
<p>Consider two sigmoid neurons: \\(\\sigma(100(x - 0.3))\\) and \\(-\\sigma(100(x - 0.7))\\). Their sum creates an approximate "rectangular bump" that is roughly 1 on \\([0.3, 0.7]\\) and 0 elsewhere. By adjusting the positions, widths, and heights of such bumps, we can approximate any target shape.</p>
</div>
</div>

<h3>For ReLU Networks</h3>

<p>With ReLU activations, the approximation works differently. Each ReLU neuron contributes a "hinge" (a piecewise-linear function that bends at one point). By summing many hinges with carefully chosen slopes and positions, we can construct a piecewise-linear approximation to any continuous function. Since continuous functions on compact sets can be uniformly approximated by piecewise-linear functions, the result follows.</p>

<div class="env-block proposition">
<div class="env-title">Proposition 4.3 — ReLU Basis Functions</div>
<div class="env-body">
<p>Any piecewise-linear function on \\([a,b]\\) with \\(k\\) breakpoints can be written as a linear combination of at most \\(k+1\\) ReLU functions \\(\\max(0, w_j x + b_j)\\) plus a bias term. Thus a single-hidden-layer ReLU network with \\(k+1\\) neurons can represent any piecewise-linear function with \\(k\\) breakpoints exactly.</p>
</div>
</div>

<h3>Important Caveats</h3>

<div class="env-block warning">
<div class="env-title">What the Theorem Does NOT Say</div>
<div class="env-body">
<p>The Universal Approximation Theorem is an <em>existence</em> result. It has several critical limitations:</p>
<ol>
<li><strong>No bound on width:</strong> the required number of neurons \\(n\\) may be astronomically large (potentially exponential in \\(d\\)).</li>
<li><strong>No algorithm:</strong> the theorem says nothing about how to <em>find</em> the right weights. Gradient descent might not converge to the global optimum.</li>
<li><strong>No generalization guarantee:</strong> a network that approximates \\(f\\) on the training data may not generalize to unseen data. The theorem is about <em>representational capacity</em>, not statistical learning.</li>
<li><strong>Compact domain:</strong> the approximation holds on compact sets. For unbounded domains, additional conditions are needed.</li>
</ol>
<p>In practice, this is why we use deep networks (for efficiency), regularization (for generalization), and sophisticated optimization algorithms (to find good weights).</p>
</div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 4.3 — Width-Bounded Universal Approximation (Lu et al., 2017)</div>
<div class="env-body">
<p>ReLU networks of width \\(d + 1\\) (where \\(d\\) is the input dimension) can approximate any continuous function on a compact set, provided the depth is allowed to grow. This is a "narrow but deep" counterpart to the classical "wide but shallow" result. It shows that depth can compensate for limited width.</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-universal-approx"></div>
`,
            visualizations: [
                {
                    id: 'viz-universal-approx',
                    title: 'Universal Approximation Demo',
                    description: 'The blue curve is the target function. Increase the number of hidden neurons and click "Fit" to watch the MLP approximation (orange) converge to the target. More neurons produce a better fit.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        let nHidden = 5;
                        let weights = null;
                        let fitting = false;
                        let epoch = 0;
                        let targetFnIdx = 0;

                        const targetFns = [
                            { name: 'sin(2\u03C0x)', fn: x => Math.sin(2 * Math.PI * x) },
                            { name: 'x\u00B2 - 0.5', fn: x => x * x - 0.5 },
                            { name: 'step + ramp', fn: x => x < 0.3 ? -0.5 : (x < 0.7 ? 2 * (x - 0.3) - 0.5 : 0.3) },
                            { name: '|sin(5x)|', fn: x => Math.abs(Math.sin(5 * x)) }
                        ];

                        const sliderN = VizEngine.createSlider(controls, 'Hidden neurons', 1, 40, nHidden, 1, v => {
                            nHidden = Math.round(v);
                        });

                        const fnBtnContainer = document.createElement('div');
                        fnBtnContainer.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;';
                        targetFns.forEach((tf, i) => {
                            const b = document.createElement('button');
                            b.style.cssText = 'padding:3px 8px;border:1px solid #30363d;border-radius:4px;background:#1a1a40;color:#c9d1d9;font-size:0.72rem;cursor:pointer;';
                            b.textContent = tf.name;
                            b.addEventListener('click', () => { targetFnIdx = i; weights = null; epoch = 0; drawState(); });
                            fnBtnContainer.appendChild(b);
                        });
                        controls.appendChild(fnBtnContainer);

                        const fitBtn = VizEngine.createButton(controls, 'Fit', () => {
                            startFit();
                        });

                        // Generate training data from target function
                        function getTrainData() {
                            const pts = [];
                            for (let i = 0; i <= 60; i++) {
                                const x = i / 60;
                                pts.push({ x: x, y: targetFns[targetFnIdx].fn(x) });
                            }
                            return pts;
                        }

                        function initW(nH) {
                            const w1 = [], b1 = [], w2 = [], b2 = [0];
                            for (let j = 0; j < nH; j++) {
                                w1.push((Math.random() - 0.5) * 6);
                                b1.push((Math.random() - 0.5) * 4);
                                w2.push((Math.random() - 0.5) * 2);
                            }
                            return { w1, b1, w2, b2, nH };
                        }

                        function fwd(x, wt) {
                            let out = wt.b2[0];
                            for (let j = 0; j < wt.nH; j++) {
                                const z = wt.w1[j] * x + wt.b1[j];
                                const h = Math.max(0, z); // ReLU
                                out += wt.w2[j] * h;
                            }
                            return out;
                        }

                        function trainBatch(wt, data, lr) {
                            for (let idx = 0; idx < data.length; idx++) {
                                const p = data[idx];
                                const pred = fwd(p.x, wt);
                                const err = pred - p.y;

                                wt.b2[0] -= lr * err;
                                for (let j = 0; j < wt.nH; j++) {
                                    const z = wt.w1[j] * p.x + wt.b1[j];
                                    const h = Math.max(0, z);
                                    const dRelu = z > 0 ? 1 : 0;

                                    wt.w2[j] -= lr * err * h;
                                    wt.w1[j] -= lr * err * wt.w2[j] * dRelu * p.x;
                                    wt.b1[j] -= lr * err * wt.w2[j] * dRelu;
                                }
                            }
                        }

                        function drawState() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            const margin = 40;
                            const plotW = W - 2 * margin;
                            const plotH = H - 2 * margin;
                            const xMin = -0.05, xMax = 1.05;
                            const yMin = -1.5, yMax = 1.5;

                            function mx(v) { return margin + (v - xMin) / (xMax - xMin) * plotW; }
                            function my(v) { return margin + plotH - (v - yMin) / (yMax - yMin) * plotH; }

                            // Background
                            ctx.fillStyle = '#0e0e28';
                            ctx.fillRect(margin, margin, plotW, plotH);

                            // Grid
                            ctx.strokeStyle = '#1a1a40';
                            ctx.lineWidth = 0.5;
                            for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
                                const py = my(y);
                                ctx.beginPath(); ctx.moveTo(margin, py); ctx.lineTo(margin + plotW, py); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = '#4a4a7a';
                            ctx.lineWidth = 1;
                            const zeroY = my(0);
                            ctx.beginPath(); ctx.moveTo(margin, zeroY); ctx.lineTo(margin + plotW, zeroY); ctx.stroke();

                            // Tick labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            for (let x = 0; x <= 1; x += 0.2) {
                                ctx.fillText(x.toFixed(1), mx(x), margin + plotH + 4);
                            }
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
                                ctx.fillText(y, margin - 4, my(y));
                            }

                            // Target function (blue)
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (let i = 0; i <= 200; i++) {
                                const x = i / 200;
                                const y = targetFns[targetFnIdx].fn(x);
                                const px = mx(x), py = my(y);
                                if (i === 0) ctx.moveTo(px, Math.max(margin, Math.min(margin + plotH, py)));
                                else ctx.lineTo(px, Math.max(margin, Math.min(margin + plotH, py)));
                            }
                            ctx.stroke();

                            // MLP approximation (orange)
                            if (weights) {
                                ctx.strokeStyle = viz.colors.orange;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                for (let i = 0; i <= 200; i++) {
                                    const x = i / 200;
                                    const y = fwd(x, weights);
                                    const px = mx(x), py = my(y);
                                    if (i === 0) ctx.moveTo(px, Math.max(margin, Math.min(margin + plotH, py)));
                                    else ctx.lineTo(px, Math.max(margin, Math.min(margin + plotH, py)));
                                }
                                ctx.stroke();
                            }

                            // Legend
                            ctx.fillStyle = viz.colors.blue;
                            ctx.font = '12px -apple-system, sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('\u2500 Target: ' + targetFns[targetFnIdx].name, margin + 10, margin + 8);
                            if (weights) {
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('\u2500 MLP (' + weights.nH + ' neurons, epoch ' + epoch + ')', margin + 10, margin + 24);

                                // Compute MSE
                                let mse = 0;
                                const nPts = 100;
                                for (let i = 0; i <= nPts; i++) {
                                    const x = i / nPts;
                                    const diff = fwd(x, weights) - targetFns[targetFnIdx].fn(x);
                                    mse += diff * diff;
                                }
                                mse /= (nPts + 1);
                                ctx.fillStyle = viz.colors.text;
                                ctx.fillText('MSE: ' + mse.toFixed(5), margin + 10, margin + 40);
                            }
                        }

                        function startFit() {
                            if (fitting) return;
                            weights = initW(nHidden);
                            epoch = 0;
                            fitting = true;
                            fitBtn.textContent = 'Fitting...';
                            const data = getTrainData();

                            function step() {
                                if (!fitting) return;
                                for (let i = 0; i < 10; i++) {
                                    trainBatch(weights, data, 0.005);
                                    epoch++;
                                }
                                drawState();
                                if (epoch < 2000) {
                                    requestAnimationFrame(step);
                                } else {
                                    fitting = false;
                                    fitBtn.textContent = 'Refit';
                                    drawState();
                                }
                            }
                            step();
                        }

                        drawState();

                        return {
                            stopAnimation() { fitting = false; }
                        };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Explain why the Universal Approximation Theorem does not imply that neural networks can learn any function from finite training data.',
                    hint: 'Distinguish between <em>approximation</em> (representation) and <em>learning</em> (generalization from data). What role does the number of training samples play?',
                    solution: 'The UAT is purely about <em>representational capacity</em>: given <em>any</em> continuous function \\(f\\) and \\(\\varepsilon &gt; 0\\), there <em>exists</em> a network that approximates \\(f\\) within \\(\\varepsilon\\). But: (1) We only have finite training data, so many different functions could fit the data equally well. The network might learn a different function from \\(f\\) that agrees on training points but diverges elsewhere. (2) Finding the correct weights is a non-convex optimization problem; gradient descent may not converge to the optimal solution. (3) The required width \\(n\\) may be impractically large. (4) No statistical guarantee about test error is provided. Generalization requires additional inductive biases, regularization, and sufficient data.'
                },
                {
                    question: 'Use the ReLU basis function idea to show that \\(f(x) = |x|\\) on \\([-1, 1]\\) can be represented exactly by a single-hidden-layer ReLU network with 2 hidden neurons.',
                    hint: 'Express \\(|x| = \\max(0, x) + \\max(0, -x)\\).',
                    solution: 'We have \\(|x| = \\max(0, x) + \\max(0, -x) = \\mathrm{ReLU}(x) + \\mathrm{ReLU}(-x)\\). This is a single-hidden-layer ReLU network with 2 neurons: hidden unit 1 computes \\(h_1 = \\mathrm{ReLU}(1 \\cdot x + 0)\\), hidden unit 2 computes \\(h_2 = \\mathrm{ReLU}(-1 \\cdot x + 0)\\), and the output is \\(F(x) = 1 \\cdot h_1 + 1 \\cdot h_2 + 0 = |x|\\). So \\(w_1 = 1, w_2 = -1, b_1 = b_2 = 0, \\alpha_1 = \\alpha_2 = 1, \\beta = 0\\).'
                },
                {
                    question: 'Construct a single-hidden-layer ReLU network that represents the piecewise-linear function \\(f(x) = \\begin{cases} 0 & x \\leq 0, \\\\ 2x & 0 &lt; x \\leq 1, \\\\ 2 & x &gt; 1. \\end{cases}\\)',
                    hint: 'Write \\(f\\) as a sum of ReLU functions. Note that \\(f(x) = 2\\mathrm{ReLU}(x) - 2\\mathrm{ReLU}(x-1)\\).',
                    solution: 'We need \\(f(x)\\) that ramps up from 0 to 2 between \\(x=0\\) and \\(x=1\\), then stays at 2. Write: \\(f(x) = 2\\max(0, x) - 2\\max(0, x-1)\\). Check: For \\(x \\leq 0\\): \\(2 \\cdot 0 - 2 \\cdot 0 = 0\\). For \\(0 &lt; x \\leq 1\\): \\(2x - 0 = 2x\\). For \\(x &gt; 1\\): \\(2x - 2(x-1) = 2\\). This is a 2-neuron network: \\(h_1 = \\mathrm{ReLU}(x)\\), \\(h_2 = \\mathrm{ReLU}(x - 1)\\), output \\(= 2h_1 - 2h_2\\). The parameters are \\(w_1 = 1, b_1 = 0, w_2 = 1, b_2 = -1, \\alpha_1 = 2, \\alpha_2 = -2, \\beta = 0\\).'
                }
            ]
        },

        // ===== SECTION 5: MLP for Classification & Regression =====
        {
            id: 'sec-mlp-tasks',
            title: 'MLP for Classification & Regression',
            content: `
<h2>4.5 MLP for Classification &amp; Regression</h2>

<div class="env-block intuition">
<div class="env-title">Matching the Output to the Task</div>
<div class="env-body">
<p>The hidden layers of an MLP learn feature representations. The <em>output layer</em> must translate these features into the format required by the task: a real number for regression, a probability for binary classification, or a probability distribution over classes for multiclass classification. The choice of output activation and loss function must be matched to the task, forming a coherent probabilistic model.</p>
</div>
</div>

<h3>Regression: Linear Output + MSE Loss</h3>

<div class="env-block definition">
<div class="env-title">Definition 4.8 — Regression Output</div>
<div class="env-body">
<p>For regression, the output layer applies no activation function (or equivalently, the identity activation):</p>
\\[\\hat{y} = \\mathbf{w}^T \\mathbf{h}^{(L)} + b,\\]
<p>where \\(\\mathbf{h}^{(L)}\\) is the last hidden layer's activation. The most common loss is the <strong>mean squared error</strong> (MSE):</p>
\\[\\mathcal{L}_{\\mathrm{MSE}} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2.\\]
</div>
</div>

<div class="env-block remark">
<div class="env-title">Probabilistic Interpretation</div>
<div class="env-body">
<p>MSE corresponds to maximizing the likelihood under the model \\(y = f(\\mathbf{x}; \\theta) + \\varepsilon\\) where \\(\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)\\). Minimizing MSE is equivalent to maximizing the log-likelihood of a Gaussian with mean \\(f(\\mathbf{x}; \\theta)\\) and fixed variance.</p>
</div>
</div>

<h3>Binary Classification: Sigmoid Output + Binary Cross-Entropy</h3>

<div class="env-block definition">
<div class="env-title">Definition 4.9 — Binary Classification Output</div>
<div class="env-body">
<p>For binary classification (\\(y \\in \\{0, 1\\}\\)), the output layer applies the sigmoid function:</p>
\\[\\hat{p} = \\sigma(\\mathbf{w}^T \\mathbf{h}^{(L)} + b) = \\frac{1}{1 + \\exp(-(\\mathbf{w}^T \\mathbf{h}^{(L)} + b))}.\\]
<p>The output \\(\\hat{p} \\in (0,1)\\) is interpreted as \\(P(y=1 \\mid \\mathbf{x})\\). The loss is the <strong>binary cross-entropy</strong>:</p>
\\[\\mathcal{L}_{\\mathrm{BCE}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\log \\hat{p}_i + (1 - y_i) \\log(1 - \\hat{p}_i) \\right].\\]
</div>
</div>

<h3>Multiclass Classification: Softmax + Cross-Entropy</h3>

<div class="env-block definition">
<div class="env-title">Definition 4.10 — Softmax Function</div>
<div class="env-body">
<p>For \\(K\\)-class classification, the output layer computes <strong>logits</strong> \\(\\mathbf{z} = W^{(L+1)} \\mathbf{h}^{(L)} + \\mathbf{b}^{(L+1)} \\in \\mathbb{R}^K\\), and the <strong>softmax</strong> function converts them to a probability distribution:</p>
\\[\\mathrm{softmax}(\\mathbf{z})_k = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}, \\qquad k = 1, \\ldots, K.\\]
<p>Properties: (1) \\(\\mathrm{softmax}(\\mathbf{z})_k \\in (0,1)\\) for all \\(k\\); (2) \\(\\sum_{k=1}^K \\mathrm{softmax}(\\mathbf{z})_k = 1\\); (3) \\(\\mathrm{softmax}(\\mathbf{z} + c\\mathbf{1}) = \\mathrm{softmax}(\\mathbf{z})\\) for any constant \\(c\\) (shift invariance).</p>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 4.11 — Categorical Cross-Entropy Loss</div>
<div class="env-body">
<p>Given one-hot encoded targets \\(\\mathbf{y} \\in \\{0,1\\}^K\\) (with exactly one entry equal to 1), the <strong>cross-entropy loss</strong> is:</p>
\\[\\mathcal{L}_{\\mathrm{CE}} = -\\sum_{k=1}^{K} y_k \\log \\hat{p}_k = -\\log \\hat{p}_{c},\\]
<p>where \\(c\\) is the true class index. This is the negative log-likelihood under the categorical distribution parameterized by \\(\\hat{\\mathbf{p}} = \\mathrm{softmax}(\\mathbf{z})\\).</p>
</div>
</div>

<h3>The Softmax Temperature</h3>

<p>A generalization of softmax introduces a <strong>temperature</strong> parameter \\(T &gt; 0\\):</p>
\\[\\mathrm{softmax}(\\mathbf{z}; T)_k = \\frac{e^{z_k / T}}{\\sum_{j=1}^{K} e^{z_j / T}}.\\]

<p>The temperature controls the "sharpness" of the distribution:</p>
<ul>
<li><strong>\\(T \\to 0^+\\):</strong> the distribution concentrates all mass on the largest logit (approaches argmax, or a "hard" one-hot vector).</li>
<li><strong>\\(T = 1\\):</strong> standard softmax.</li>
<li><strong>\\(T \\to \\infty\\):</strong> the distribution approaches uniform \\((1/K, \\ldots, 1/K)\\).</li>
</ul>

<div class="env-block remark">
<div class="env-title">Temperature in Practice</div>
<div class="env-body">
<p>Temperature scaling is used in several important contexts: (1) <strong>knowledge distillation</strong> (Hinton et al., 2015), where a "teacher" network's softmax outputs are softened with high temperature to produce more informative targets for a "student" network; (2) <strong>calibration</strong>, where \\(T\\) is tuned on a validation set to make predicted probabilities better reflect true frequencies; (3) <strong>language model sampling</strong>, where temperature controls the diversity of generated text (low \\(T\\) for focused output, high \\(T\\) for creative/diverse output).</p>
</div>
</div>

<h3>Numerical Stability: The Log-Sum-Exp Trick</h3>

<div class="env-block proposition">
<div class="env-title">Proposition 4.4 — Log-Sum-Exp Trick</div>
<div class="env-body">
<p>Computing \\(\\log \\sum_{j} e^{z_j}\\) directly can overflow (if some \\(z_j\\) is very large) or underflow (if all \\(z_j\\) are very negative). The numerically stable version uses:</p>
\\[\\log \\sum_{j=1}^K e^{z_j} = m + \\log \\sum_{j=1}^K e^{z_j - m}, \\qquad m = \\max_j z_j.\\]
<p>After subtracting \\(m\\), the largest exponent is \\(e^0 = 1\\), preventing overflow. This trick is universally used in deep learning frameworks.</p>
</div>
</div>

<h3>Summary of Output Layer Design</h3>

<table style="width:100%;border-collapse:collapse;margin:16px 0;">
<tr style="border-bottom:2px solid #30363d;">
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Task</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Output activation</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Loss function</td>
<td style="padding:8px;font-weight:bold;color:#f0f6fc;">Output range</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">Regression</td>
<td style="padding:6px;">None (identity)</td>
<td style="padding:6px;">MSE</td>
<td style="padding:6px;">\\((-\\infty, \\infty)\\)</td>
</tr>
<tr style="border-bottom:1px solid #30363d;">
<td style="padding:6px;">Binary classification</td>
<td style="padding:6px;">Sigmoid</td>
<td style="padding:6px;">Binary cross-entropy</td>
<td style="padding:6px;">\\((0, 1)\\)</td>
</tr>
<tr>
<td style="padding:6px;">Multiclass (\\(K\\) classes)</td>
<td style="padding:6px;">Softmax</td>
<td style="padding:6px;">Cross-entropy</td>
<td style="padding:6px;">Simplex \\(\\Delta^{K-1}\\)</td>
</tr>
</table>

<div class="viz-placeholder" data-viz="viz-softmax-temperature"></div>
`,
            visualizations: [
                {
                    id: 'viz-softmax-temperature',
                    title: 'Softmax Temperature Visualizer',
                    description: 'The bar chart shows softmax probabilities for 5 classes with fixed logits. Adjust the temperature to see how the distribution changes from peaked (low T) to uniform (high T).',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const W = viz.width, H = viz.height;
                        const ctx = viz.ctx;

                        let T = 1.0;
                        const logits = [2.0, 1.0, 0.5, -0.5, -1.0];
                        const K = logits.length;
                        const classColors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.purple, viz.colors.pink];
                        const classNames = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'];

                        const sliderT = VizEngine.createSlider(controls, 'Temperature T', 0.1, 5.0, T, 0.1, v => {
                            T = v;
                            draw();
                        });

                        function softmax(logits, temp) {
                            const scaled = logits.map(z => z / temp);
                            const maxZ = Math.max(...scaled);
                            const exps = scaled.map(z => Math.exp(z - maxZ));
                            const sum = exps.reduce((a, b) => a + b, 0);
                            return exps.map(e => e / sum);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            const probs = softmax(logits, T);
                            const margin = { top: 50, bottom: 60, left: 60, right: 30 };
                            const plotW = W - margin.left - margin.right;
                            const plotH = H - margin.top - margin.bottom;

                            // Background
                            ctx.fillStyle = '#0e0e28';
                            ctx.fillRect(margin.left, margin.top, plotW, plotH);

                            // Y-axis gridlines
                            ctx.strokeStyle = '#1a1a40';
                            ctx.lineWidth = 0.5;
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system, sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (let p = 0; p <= 1.0; p += 0.2) {
                                const y = margin.top + plotH - p * plotH;
                                ctx.beginPath();
                                ctx.moveTo(margin.left, y);
                                ctx.lineTo(margin.left + plotW, y);
                                ctx.stroke();
                                ctx.fillText(p.toFixed(1), margin.left - 6, y);
                            }

                            // Bars
                            const barGap = 12;
                            const totalGap = (K + 1) * barGap;
                            const barW = (plotW - totalGap) / K;

                            for (let k = 0; k < K; k++) {
                                const x = margin.left + barGap + k * (barW + barGap);
                                const barH = probs[k] * plotH;
                                const y = margin.top + plotH - barH;

                                // Bar gradient effect
                                const grad = ctx.createLinearGradient(x, y, x, margin.top + plotH);
                                grad.addColorStop(0, classColors[k]);
                                grad.addColorStop(1, classColors[k] + '44');
                                ctx.fillStyle = grad;
                                ctx.fillRect(x, y, barW, barH);

                                // Bar outline
                                ctx.strokeStyle = classColors[k];
                                ctx.lineWidth = 1.5;
                                ctx.strokeRect(x, y, barW, barH);

                                // Probability text on top of bar
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 12px -apple-system, sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';
                                ctx.fillText(probs[k].toFixed(3), x + barW / 2, y - 4);

                                // Class label + logit
                                ctx.fillStyle = classColors[k];
                                ctx.font = '11px -apple-system, sans-serif';
                                ctx.textBaseline = 'top';
                                ctx.fillText(classNames[k], x + barW / 2, margin.top + plotH + 6);
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system, sans-serif';
                                ctx.fillText('z=' + logits[k].toFixed(1), x + barW / 2, margin.top + plotH + 22);
                            }

                            // Title / info
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system, sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('T = ' + T.toFixed(1), margin.left, 12);

                            // Entropy
                            let entropy = 0;
                            for (let k = 0; k < K; k++) {
                                if (probs[k] > 1e-10) entropy -= probs[k] * Math.log2(probs[k]);
                            }
                            const maxEntropy = Math.log2(K);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system, sans-serif';
                            ctx.fillText('Entropy: ' + entropy.toFixed(3) + ' bits (max: ' + maxEntropy.toFixed(3) + ')', margin.left + 80, 14);

                            // Description of regime
                            let regime = '';
                            if (T < 0.3) regime = 'Near-argmax: almost all mass on largest logit';
                            else if (T < 0.8) regime = 'Sharp: peaked distribution';
                            else if (T < 1.5) regime = 'Standard softmax range';
                            else if (T < 3.0) regime = 'Soft: approaching uniform';
                            else regime = 'Very soft: nearly uniform distribution';
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system, sans-serif';
                            ctx.textAlign = 'right';
                            ctx.fillText(regime, W - margin.right, 14);

                            // Y-axis label
                            ctx.save();
                            ctx.translate(16, margin.top + plotH / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Probability', 0, 0);
                            ctx.restore();
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that softmax is invariant to adding a constant to all logits: \\(\\mathrm{softmax}(\\mathbf{z} + c\\mathbf{1}) = \\mathrm{softmax}(\\mathbf{z})\\) for any \\(c \\in \\mathbb{R}\\).',
                    hint: 'Write out the definition and factor \\(e^c\\) from numerator and denominator.',
                    solution: '\\(\\mathrm{softmax}(\\mathbf{z} + c\\mathbf{1})_k = \\frac{e^{z_k + c}}{\\sum_j e^{z_j + c}} = \\frac{e^c \\cdot e^{z_k}}{e^c \\cdot \\sum_j e^{z_j}} = \\frac{e^{z_k}}{\\sum_j e^{z_j}} = \\mathrm{softmax}(\\mathbf{z})_k\\). The factor \\(e^c\\) cancels between numerator and denominator.'
                },
                {
                    question: 'Show that as temperature \\(T \\to 0^+\\), the softmax output approaches a one-hot vector with 1 at the position of the largest logit (assuming a unique maximum).',
                    hint: 'Consider what happens to \\(e^{z_k/T}\\) relative to \\(e^{z_j/T}\\) when \\(z_k &gt; z_j\\) and \\(T \\to 0\\).',
                    solution: 'Let \\(c = \\arg\\max_k z_k\\) (unique by assumption), so \\(z_c &gt; z_j\\) for all \\(j \\neq c\\). Write \\(\\mathrm{softmax}(\\mathbf{z}/T)_c = \\frac{1}{1 + \\sum_{j \\neq c} e^{(z_j - z_c)/T}}\\). Since \\(z_j - z_c &lt; 0\\), as \\(T \\to 0^+\\), \\((z_j - z_c)/T \\to -\\infty\\), so \\(e^{(z_j - z_c)/T} \\to 0\\). Therefore \\(\\mathrm{softmax}(\\mathbf{z}/T)_c \\to 1/(1+0) = 1\\). For \\(k \\neq c\\): \\(\\mathrm{softmax}(\\mathbf{z}/T)_k = \\frac{e^{(z_k - z_c)/T}}{1 + \\sum_{j \\neq c} e^{(z_j - z_c)/T}} \\to 0\\). So the output approaches \\(\\mathbf{e}_c\\).'
                },
                {
                    question: 'Derive the gradient of the cross-entropy loss with respect to the logits: show that \\(\\frac{\\partial \\mathcal{L}_{\\mathrm{CE}}}{\\partial z_k} = \\hat{p}_k - y_k\\), where \\(\\hat{p}_k = \\mathrm{softmax}(\\mathbf{z})_k\\) and \\(y_k\\) is the one-hot target.',
                    hint: 'First compute \\(\\frac{\\partial \\hat{p}_k}{\\partial z_j}\\) (the Jacobian of softmax), then apply the chain rule to \\(\\mathcal{L} = -\\sum_k y_k \\log \\hat{p}_k\\).',
                    solution: 'The softmax Jacobian is \\(\\frac{\\partial \\hat{p}_k}{\\partial z_j} = \\hat{p}_k(\\delta_{kj} - \\hat{p}_j)\\), where \\(\\delta_{kj}\\) is the Kronecker delta. Now \\(\\frac{\\partial \\mathcal{L}}{\\partial z_j} = -\\sum_k y_k \\frac{1}{\\hat{p}_k} \\cdot \\hat{p}_k(\\delta_{kj} - \\hat{p}_j) = -\\sum_k y_k (\\delta_{kj} - \\hat{p}_j)\\). Since \\(\\sum_k y_k = 1\\) (one-hot), this simplifies to \\(-y_j + \\hat{p}_j \\sum_k y_k = \\hat{p}_j - y_j\\). This elegant result is one reason cross-entropy with softmax is computationally convenient: the gradient has the simple form "prediction minus target."'
                }
            ]
        }
    ]
});
