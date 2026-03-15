window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch06',
  number: 6,
  title: 'Initialization & Normalization',
  subtitle: 'Xavier/He initialization, Batch Normalization, Layer Normalization, and their roles in enabling deep training',
  sections: [

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 1: Weight Initialization
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch06-sec01',
      title: '1. Weight Initialization',
      content: `
<div class="env-block intuition">
<strong>From Backpropagation to Practical Training.</strong>
Chapter 5 showed us <em>how</em> gradients flow through a network via backpropagation. But knowing the algorithm is not enough: if we initialize the weights carelessly, gradients can explode or vanish before the first epoch finishes. This chapter addresses the two key techniques that make deep training reliable: proper weight initialization and normalization layers.
</div>

<h2>Weight Initialization</h2>

<div class="env-block intuition">
<strong>Section Roadmap.</strong>
We begin with the question: what numerical values should we assign to the weight matrices before training starts? We will see that naive choices (too large or too small) cause activations to either explode or collapse to zero as signals propagate through layers. Xavier/Glorot initialization (2010) and He/Kaiming initialization (2015) solve this problem by calibrating variance to the layer width.
</div>

<h3>Why Initialization Matters</h3>

<p>
Consider a feedforward network with \\(L\\) layers. The pre-activation at layer \\(l\\) is
\\[ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \\]
where \\(a^{(l-1)}\\) is the activation from the previous layer. If we initialize every entry of \\(W^{(l)}\\) from the same distribution, the variance of \\(z^{(l)}\\) depends on the variance of the weights and the dimension of the input.
</p>

<div class="env-block definition">
<div class="env-title">Variance Propagation</div>
<div class="env-body">
Suppose the weights \\(W^{(l)}_{ij}\\) are i.i.d. with mean zero and variance \\(\\sigma^2_w\\), and the inputs \\(a^{(l-1)}_j\\) are i.i.d. with mean zero and variance \\(\\text{Var}(a)\\). Then by the linearity of variance for independent terms,
\\[ \\text{Var}(z^{(l)}_i) = n_{l-1} \\cdot \\sigma^2_w \\cdot \\text{Var}(a^{(l-1)}), \\]
where \\(n_{l-1}\\) is the fan-in (number of input units to layer \\(l\\)).
</div>
</div>

<p>
This formula reveals the core problem. If \\(n_{l-1} \\cdot \\sigma^2_w &gt; 1\\), the variance of activations <em>grows</em> exponentially through layers, leading to numerical overflow. If \\(n_{l-1} \\cdot \\sigma^2_w &lt; 1\\), it <em>shrinks</em> exponentially, and deep layers receive signals indistinguishable from noise.
</p>

<h3>The Naive Approaches</h3>

<div class="env-block warning">
<div class="env-title">Too Large: \\(W \\sim \\mathcal{N}(0, 1)\\)</div>
<div class="env-body">
With standard normal initialization for a layer of width \\(n = 256\\), we get \\(\\text{Var}(z) = 256 \\cdot 1 \\cdot \\text{Var}(a) = 256 \\cdot \\text{Var}(a)\\). After just 10 layers, the variance is amplified by \\(256^{10} \\approx 10^{24}\\), causing immediate saturation (for sigmoid/tanh) or overflow (for ReLU).
</div>
</div>

<div class="env-block warning">
<div class="env-title">Too Small: \\(W \\sim \\mathcal{N}(0, 0.001^2)\\)</div>
<div class="env-body">
With tiny weights, the variance multiplier per layer is \\(n \\cdot 10^{-6}\\). For \\(n = 256\\), that is \\(2.56 \\times 10^{-4}\\) per layer. After 10 layers: \\((2.56 \\times 10^{-4})^{10} \\approx 10^{-36}\\). All activations collapse to near-zero, and gradients vanish.
</div>
</div>

<h3>Xavier/Glorot Initialization (2010)</h3>

<p>
Glorot and Bengio (2010) proposed choosing \\(\\sigma^2_w\\) so that the variance is preserved through each layer during both the forward pass and the backward pass. Specifically, the forward pass requires \\(n_{\\text{in}} \\cdot \\sigma^2_w = 1\\), while the backward pass requires \\(n_{\\text{out}} \\cdot \\sigma^2_w = 1\\). Averaging these two constraints:
</p>

<div class="env-block theorem">
<div class="env-title">Xavier/Glorot Initialization</div>
<div class="env-body">
Initialize each weight as
\\[ W_{ij} \\sim \\mathcal{N}\\!\\left(0,\\; \\frac{2}{n_{\\text{in}} + n_{\\text{out}}}\\right) \\]
or equivalently from a uniform distribution
\\[ W_{ij} \\sim \\text{Uniform}\\!\\left(-\\sqrt{\\frac{6}{n_{\\text{in}} + n_{\\text{out}}}},\\; \\sqrt{\\frac{6}{n_{\\text{in}} + n_{\\text{out}}}}\\right). \\]
This keeps the variance of activations and gradients approximately constant across layers, assuming linear or tanh activations near zero.
</div>
</div>

<div class="env-block remark">
<div class="env-title">Why the Uniform Variant?</div>
<div class="env-body">
A uniform distribution on \\([-a, a]\\) has variance \\(a^2/3\\). Setting \\(a^2/3 = 2/(n_{\\text{in}} + n_{\\text{out}})\\) gives \\(a = \\sqrt{6/(n_{\\text{in}} + n_{\\text{out}})}\\).
</div>
</div>

<h3>He/Kaiming Initialization (2015)</h3>

<p>
Xavier initialization assumes the activation function is approximately linear around zero (valid for tanh and sigmoid in their central region). But ReLU sets half of its inputs to zero, effectively halving the variance at each layer. He et al. (2015) accounted for this by doubling the variance:
</p>

<div class="env-block theorem">
<div class="env-title">He/Kaiming Initialization</div>
<div class="env-body">
For networks using ReLU activations, initialize as
\\[ W_{ij} \\sim \\mathcal{N}\\!\\left(0,\\; \\frac{2}{n_{\\text{in}}}\\right). \\]
The factor of 2 compensates for ReLU zeroing out roughly half the activations (so the effective fan-in is \\(n_{\\text{in}}/2\\)).
</div>
</div>

<div class="env-block intuition">
<div class="env-title">The ReLU Halving Argument</div>
<div class="env-body">
If \\(a = \\text{ReLU}(z)\\) and \\(z\\) is symmetric around zero, then \\(\\text{Var}(a) = \\frac{1}{2}\\text{Var}(z)\\) because exactly half the values are zeroed. To maintain \\(\\text{Var}(z^{(l)}) = \\text{Var}(z^{(l-1)})\\), we need \\(n_{\\text{in}} \\cdot \\sigma^2_w \\cdot \\frac{1}{2} = 1\\), hence \\(\\sigma^2_w = 2/n_{\\text{in}}\\).
</div>
</div>

<h3>Biases and Practical Considerations</h3>

<p>
Biases are typically initialized to zero. For ReLU networks, some practitioners initialize biases to a small positive value (e.g., 0.01) to ensure all neurons are active at the start, though modern practice has largely moved back to zero initialization.
</p>

<div class="env-block remark">
<div class="env-title">Summary of Initialization Recipes</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:0.5rem 0;">
<tr style="background:#1a1a40;">
  <th style="padding:8px;border:1px solid #30363d;">Activation</th>
  <th style="padding:8px;border:1px solid #30363d;">Method</th>
  <th style="padding:8px;border:1px solid #30363d;">Variance \\(\\sigma^2_w\\)</th>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">tanh / sigmoid</td>
  <td style="padding:8px;border:1px solid #30363d;">Xavier/Glorot</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(\\frac{2}{n_{\\text{in}} + n_{\\text{out}}}\\)</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">ReLU / Leaky ReLU</td>
  <td style="padding:8px;border:1px solid #30363d;">He/Kaiming</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(\\frac{2}{n_{\\text{in}}}\\)</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">SELU</td>
  <td style="padding:8px;border:1px solid #30363d;">LeCun</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(\\frac{1}{n_{\\text{in}}}\\)</td>
</tr>
</table>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-init-histograms"></div>
`,
      visualizations: [
        {
          id: 'viz-init-histograms',
          title: 'Activation Distributions Across Layers',
          description: 'Compare how different initialization strategies affect activation distributions as signals propagate through a 10-layer tanh network (width 256). Use the buttons to switch strategies. Observe how Xavier preserves variance while others explode or collapse.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            // Strategies
            const strategies = {
              'Too Large (N(0,1))': (nIn) => 1.0,
              'Too Small (N(0,0.01\u00B2))': (nIn) => 0.0001,
              'Xavier': (nIn) => 2.0 / (nIn + nIn),
              'He/Kaiming': (nIn) => 2.0 / nIn
            };
            let currentStrategy = 'Xavier';

            // Box-Muller for normal random numbers
            function randn() {
              let u = 0, v = 0;
              while (u === 0) u = Math.random();
              while (v === 0) v = Math.random();
              return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            }

            function simulate(strategyName) {
              const nLayers = 10;
              const width = 256;
              const batchSize = 512;
              const varianceFn = strategies[strategyName];

              // Generate input batch
              let activations = [];
              for (let i = 0; i < batchSize; i++) {
                activations.push(randn());
              }

              const layerActivations = [activations.slice()];

              for (let l = 0; l < nLayers; l++) {
                const sigma2 = varianceFn(width);
                const sigma = Math.sqrt(sigma2);
                // Simulate one layer: z = sum of width inputs * weights, then tanh
                const newAct = [];
                for (let i = 0; i < batchSize; i++) {
                  let sum = 0;
                  for (let j = 0; j < width; j++) {
                    sum += randn() * sigma * activations[Math.floor(Math.random() * batchSize)];
                  }
                  sum /= 1; // no additional scaling
                  newAct.push(Math.tanh(sum));
                }
                activations = newAct;
                layerActivations.push(activations.slice());
              }
              return layerActivations;
            }

            function computeHistogram(values, nBins, lo, hi) {
              const bins = new Array(nBins).fill(0);
              const binWidth = (hi - lo) / nBins;
              for (const v of values) {
                let idx = Math.floor((v - lo) / binWidth);
                if (idx < 0) idx = 0;
                if (idx >= nBins) idx = nBins - 1;
                bins[idx]++;
              }
              // Normalize
              const maxCount = Math.max(...bins, 1);
              return { bins, maxCount, binWidth };
            }

            function draw() {
              const data = simulate(currentStrategy);
              // Show layers 0, 2, 4, 6, 9 (5 histograms)
              const layerIndices = [0, 2, 4, 7, 10];
              const nHist = layerIndices.length;

              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              const margin = { top: 40, bottom: 30, left: 55, right: 15 };
              const plotW = (viz.width - margin.left - margin.right) / nHist;
              const plotH = viz.height - margin.top - margin.bottom;
              const nBins = 40;

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('Strategy: ' + currentStrategy, viz.width / 2, 8);

              for (let h = 0; h < nHist; h++) {
                const li = layerIndices[h];
                const vals = data[li];
                const lo = -1.2, hi = 1.2;
                const { bins, maxCount } = computeHistogram(vals, nBins, lo, hi);
                const barW = plotW / nBins * 0.85;
                const x0 = margin.left + h * plotW;
                const y0 = margin.top;

                // Subplot background
                ctx.fillStyle = '#0f0f28';
                ctx.fillRect(x0 + 2, y0, plotW - 4, plotH);

                // Bars
                const barColor = h === 0 ? viz.colors.text :
                  (currentStrategy === 'Xavier' ? viz.colors.teal :
                   currentStrategy === 'He/Kaiming' ? viz.colors.green :
                   currentStrategy.startsWith('Too Large') ? viz.colors.red : viz.colors.orange);

                for (let b = 0; b < nBins; b++) {
                  const barH = (bins[b] / maxCount) * (plotH - 20);
                  const bx = x0 + 4 + b * (plotW - 8) / nBins;
                  const by = y0 + plotH - barH;
                  ctx.fillStyle = barColor + 'cc';
                  ctx.fillRect(bx, by, (plotW - 8) / nBins * 0.85, barH);
                }

                // X-axis line
                ctx.strokeStyle = viz.colors.axis;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x0 + 2, y0 + plotH);
                ctx.lineTo(x0 + plotW - 2, y0 + plotH);
                ctx.stroke();

                // Label
                ctx.fillStyle = viz.colors.text;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(li === 0 ? 'Input' : 'Layer ' + li, x0 + plotW / 2, y0 + plotH + 6);

                // Variance
                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
                ctx.fillStyle = viz.colors.yellow;
                ctx.font = '10px monospace';
                ctx.fillText('\u03C3\u00B2=' + variance.toFixed(4), x0 + plotW / 2, y0 + plotH + 18);

                // X-axis ticks
                ctx.fillStyle = viz.colors.text;
                ctx.font = '9px -apple-system,sans-serif';
                ctx.fillText('-1', x0 + 8, y0 + plotH + 1);
                ctx.fillText('1', x0 + plotW - 12, y0 + plotH + 1);
              }

              // Y-axis label
              ctx.save();
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.translate(15, margin.top + plotH / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillText('Frequency', 0, 0);
              ctx.restore();
            }

            draw();

            // Strategy buttons
            for (const name of Object.keys(strategies)) {
              VizEngine.createButton(controls, name, () => {
                currentStrategy = name;
                draw();
              });
            }

            VizEngine.createButton(controls, '\u21BB Resample', () => draw());

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Consider a 5-layer MLP with hidden dimension 512 and tanh activations. If we initialize weights as \\(W_{ij} \\sim \\mathcal{N}(0, 1)\\), by what factor does the variance of activations change per layer (approximately)?',
          hint: 'Use the variance propagation formula: \\(\\text{Var}(z^{(l)}) = n_{l-1} \\cdot \\sigma^2_w \\cdot \\text{Var}(a^{(l-1)})\\). For tanh near zero, the activation is approximately linear.',
          solution: 'The variance multiplier per layer is \\(n_{l-1} \\cdot \\sigma^2_w = 512 \\times 1 = 512\\). After 5 layers, the variance is amplified by \\(512^5 \\approx 3.5 \\times 10^{13}\\). In practice, tanh saturates at \\(\\pm 1\\), so all activations will be pinned to \\(\\pm 1\\) almost immediately.'
        },
        {
          question: 'Derive the Xavier initialization variance for a layer with \\(n_{\\text{in}} = 784\\) and \\(n_{\\text{out}} = 256\\). What is the standard deviation of the weights?',
          hint: 'Xavier variance is \\(\\sigma^2_w = \\frac{2}{n_{\\text{in}} + n_{\\text{out}}}\\).',
          solution: '\\(\\sigma^2_w = \\frac{2}{784 + 256} = \\frac{2}{1040} \\approx 0.00192\\). The standard deviation is \\(\\sigma_w = \\sqrt{0.00192} \\approx 0.0439\\).'
        },
        {
          question: 'Why does He initialization use \\(\\frac{2}{n_{\\text{in}}}\\) instead of \\(\\frac{1}{n_{\\text{in}}}\\)? Explain the factor of 2.',
          hint: 'Think about what ReLU does to a symmetric distribution centered at zero.',
          solution: 'ReLU sets all negative inputs to zero. For a symmetric distribution centered at zero, roughly half the values are negative. This means \\(\\text{Var}(\\text{ReLU}(z)) = \\frac{1}{2} \\text{Var}(z)\\). To compensate for this halving of variance, we double the weight variance from \\(1/n_{\\text{in}}\\) to \\(2/n_{\\text{in}}\\), ensuring \\(n_{\\text{in}} \\cdot \\frac{2}{n_{\\text{in}}} \\cdot \\frac{1}{2} = 1\\).'
        },
        {
          question: 'If you are building a network with Leaky ReLU (negative slope \\(\\alpha = 0.01\\)), how would you modify He initialization?',
          hint: 'Leaky ReLU does not zero out negative values; instead it scales them by \\(\\alpha\\). Compute \\(\\text{Var}(\\text{LeakyReLU}(z))\\) assuming \\(z \\sim \\mathcal{N}(0, \\sigma^2)\\).',
          solution: 'For Leaky ReLU with slope \\(\\alpha\\), \\(\\text{Var}(\\text{LeakyReLU}(z)) = \\frac{1 + \\alpha^2}{2} \\text{Var}(z)\\). The initialization variance becomes \\(\\sigma^2_w = \\frac{2}{n_{\\text{in}}(1 + \\alpha^2)}\\). For \\(\\alpha = 0.01\\), this is nearly identical to standard He initialization since \\(1 + 0.01^2 \\approx 1\\).'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 2: Batch Normalization
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch06-sec02',
      title: '2. Batch Normalization',
      content: `
<h2>Batch Normalization</h2>

<div class="env-block intuition">
<strong>Section Roadmap.</strong>
Even with proper initialization, the distribution of activations drifts as training progresses: each layer's input distribution changes because the parameters of all preceding layers are being updated simultaneously. Ioffe and Szegedy (2015) called this <em>internal covariate shift</em> and proposed <strong>Batch Normalization</strong> (BN) as a solution. BN normalizes each feature using statistics computed from the mini-batch, then applies a learnable affine transform. This section covers the algorithm, its behavior during training and inference, and its practical effects.
</div>

<h3>The Internal Covariate Shift Problem</h3>

<p>
During training, layer \\(l\\) tries to learn a mapping from \\(a^{(l-1)}\\) to the target. But \\(a^{(l-1)}\\) is itself a function of the parameters of layers \\(1, \\ldots, l{-}1\\), all of which are changing. From layer \\(l\\)'s perspective, the input distribution is non-stationary. This makes optimization harder because the layer must continuously adapt to a moving input distribution.
</p>

<div class="env-block definition">
<div class="env-title">Internal Covariate Shift</div>
<div class="env-body">
The change in the distribution of a layer's inputs caused by updates to the parameters of preceding layers. Batch Normalization aims to reduce this shift by re-centering and re-scaling activations at each layer.
</div>
</div>

<h3>The Batch Normalization Algorithm</h3>

<p>
Given a mini-batch \\(\\mathcal{B} = \\{x_1, \\ldots, x_m\\}\\) of pre-activations at some layer, BN performs the following steps:
</p>

<div class="env-block theorem">
<div class="env-title">Batch Normalization Transform</div>
<div class="env-body">
<strong>Step 1: Compute batch statistics.</strong>
\\[ \\mu_{\\mathcal{B}} = \\frac{1}{m} \\sum_{i=1}^{m} x_i, \\qquad \\sigma^2_{\\mathcal{B}} = \\frac{1}{m} \\sum_{i=1}^{m} (x_i - \\mu_{\\mathcal{B}})^2 \\]

<strong>Step 2: Normalize.</strong>
\\[ \\hat{x}_i = \\frac{x_i - \\mu_{\\mathcal{B}}}{\\sqrt{\\sigma^2_{\\mathcal{B}} + \\epsilon}} \\]

<strong>Step 3: Scale and shift.</strong>
\\[ y_i = \\gamma \\hat{x}_i + \\beta \\]

The parameters \\(\\gamma\\) (scale) and \\(\\beta\\) (shift) are <em>learnable</em>. The constant \\(\\epsilon\\) (typically \\(10^{-5}\\)) prevents division by zero.
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Why Learnable \\(\\gamma\\) and \\(\\beta\\)?</div>
<div class="env-body">
If we only normalized to zero mean and unit variance, we would constrain the network's representational power. For instance, sigmoid activations would be forced into their linear regime (near zero), losing their nonlinearity. The learnable parameters \\(\\gamma\\) and \\(\\beta\\) allow the network to undo the normalization if that is optimal. In particular, setting \\(\\gamma = \\sqrt{\\sigma^2_{\\mathcal{B}} + \\epsilon}\\) and \\(\\beta = \\mu_{\\mathcal{B}}\\) recovers the original activations.
</div>
</div>

<h3>BN Placement: Before or After Activation?</h3>

<p>
The original paper applied BN to pre-activations: \\(a = \\sigma(\\text{BN}(Wx + b))\\). However, some practitioners place BN after the activation. In practice, both orderings work, but the "before activation" placement is more common and better motivated theoretically (it normalizes the input to the nonlinearity).
</p>

<div class="env-block remark">
<div class="env-title">Absorbing the Bias</div>
<div class="env-body">
When using BN, the bias \\(b\\) in \\(z = Wx + b\\) is redundant: BN subtracts the batch mean, which absorbs any additive constant. In practice, layers followed by BN omit the bias term (e.g., <code>nn.Linear(n_in, n_out, bias=False)</code> in PyTorch).
</div>
</div>

<h3>Training vs. Inference</h3>

<p>
During <strong>training</strong>, BN uses the mini-batch mean and variance. During <strong>inference</strong>, there may be only a single input (batch size 1), so batch statistics are meaningless. Instead, BN maintains <em>running averages</em> of the mean and variance, computed via exponential moving average during training:
</p>

\\[ \\mu_{\\text{run}} \\leftarrow (1 - \\alpha)\\,\\mu_{\\text{run}} + \\alpha\\,\\mu_{\\mathcal{B}}, \\qquad \\sigma^2_{\\text{run}} \\leftarrow (1 - \\alpha)\\,\\sigma^2_{\\text{run}} + \\alpha\\,\\sigma^2_{\\mathcal{B}} \\]

<p>
where \\(\\alpha\\) is the momentum (typically 0.1). At inference time, BN uses \\(\\mu_{\\text{run}}\\) and \\(\\sigma^2_{\\text{run}}\\) as fixed constants, making the BN layer a simple affine transform.
</p>

<div class="env-block warning">
<div class="env-title">Train/Eval Mode Mismatch</div>
<div class="env-body">
Forgetting to switch between training mode (<code>model.train()</code>) and evaluation mode (<code>model.eval()</code>) is one of the most common bugs when using BN. In training mode at test time, the model uses batch statistics from the test batch, which can be unreliable (especially for small batches). In evaluation mode, it uses the stable running averages.
</div>
</div>

<h3>Benefits of Batch Normalization</h3>

<p>
BN provides several practical advantages:
</p>
<ul>
  <li><strong>Faster convergence.</strong> BN allows higher learning rates because the normalization prevents activations from drifting into saturation regions.</li>
  <li><strong>Reduced sensitivity to initialization.</strong> Since BN re-normalizes activations at every layer, the specific choice of initialization matters less (though good initialization still helps).</li>
  <li><strong>Implicit regularization.</strong> The noise introduced by batch statistics (each mini-batch produces slightly different means and variances) acts as a mild regularizer, similar to dropout.</li>
</ul>

<div class="env-block remark">
<div class="env-title">Limitations of BN</div>
<div class="env-body">
<ul>
  <li>Batch dependence: BN's statistics are noisy for small batch sizes, degrading performance.</li>
  <li>Sequence models: For variable-length sequences (RNNs, Transformers), BN is awkward because different time steps may have different batch compositions.</li>
  <li>Distributed training: BN requires synchronizing batch statistics across GPUs, adding communication overhead.</li>
</ul>
These limitations motivate the normalization variants in the next section.
</div>
</div>

<div class="viz-placeholder" data-viz="viz-batch-norm"></div>
`,
      visualizations: [
        {
          id: 'viz-batch-norm',
          title: 'Before vs. After Batch Normalization',
          description: 'Toggle BN on/off to see how it reshapes the activation distribution at a hidden layer. The left panel shows the raw pre-activations; the right panel shows the distribution after BN is applied. Adjust the layer depth and weight scale to see how BN stabilizes distributions.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 760, height: 400, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            let enableBN = false;
            let weightScale = 1.0;
            let numLayers = 5;
            let gamma = 1.0;
            let beta = 0.0;

            function randn() {
              let u = 0, v = 0;
              while (u === 0) u = Math.random();
              while (v === 0) v = Math.random();
              return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            }

            function simulateLayer(inputs, width, sigma) {
              const outputs = [];
              for (let i = 0; i < inputs.length; i++) {
                let sum = 0;
                for (let j = 0; j < width; j++) {
                  sum += randn() * sigma * inputs[Math.floor(Math.random() * inputs.length)];
                }
                outputs.push(sum);
              }
              return outputs;
            }

            function applyBN(values, g, b) {
              const m = values.length;
              const mean = values.reduce((a, v) => a + v, 0) / m;
              const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / m;
              const eps = 1e-5;
              return values.map(v => g * (v - mean) / Math.sqrt(variance + eps) + b);
            }

            function relu(values) {
              return values.map(v => Math.max(0, v));
            }

            function simulate() {
              const width = 128;
              const batchSize = 1000;
              const sigma = Math.sqrt(weightScale * 2.0 / width);

              let activations = [];
              for (let i = 0; i < batchSize; i++) activations.push(randn());

              let rawLast = activations.slice();
              let bnLast = activations.slice();

              for (let l = 0; l < numLayers; l++) {
                // Without BN path
                rawLast = simulateLayer(rawLast, width, sigma * weightScale);
                rawLast = relu(rawLast);

                // With BN path
                bnLast = simulateLayer(bnLast, width, sigma * weightScale);
                bnLast = applyBN(bnLast, gamma, beta);
                bnLast = relu(bnLast);
              }

              return { raw: rawLast, bn: bnLast };
            }

            function computeHistogram(values, nBins, lo, hi) {
              const bins = new Array(nBins).fill(0);
              const binWidth = (hi - lo) / nBins;
              for (const v of values) {
                let idx = Math.floor((v - lo) / binWidth);
                if (idx < 0) idx = 0;
                if (idx >= nBins) idx = nBins - 1;
                bins[idx]++;
              }
              return bins;
            }

            function drawHistPanel(vals, x0, y0, pw, ph, title, color) {
              // Auto-range
              const sorted = vals.slice().sort((a, b) => a - b);
              let lo = sorted[Math.floor(sorted.length * 0.01)];
              let hi = sorted[Math.floor(sorted.length * 0.99)];
              if (Math.abs(hi - lo) < 0.01) { lo = -1; hi = 1; }
              const range = hi - lo;
              lo -= range * 0.1;
              hi += range * 0.1;

              const nBins = 50;
              const bins = computeHistogram(vals, nBins, lo, hi);
              const maxCount = Math.max(...bins, 1);

              // Background
              ctx.fillStyle = '#0f0f28';
              ctx.fillRect(x0, y0, pw, ph);

              // Bars
              for (let b = 0; b < nBins; b++) {
                const barH = (bins[b] / maxCount) * (ph - 40);
                const bx = x0 + 5 + b * (pw - 10) / nBins;
                const by = y0 + ph - 20 - barH;
                ctx.fillStyle = color + 'bb';
                ctx.fillRect(bx, by, (pw - 10) / nBins * 0.88, barH);
              }

              // X-axis
              ctx.strokeStyle = viz.colors.axis;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(x0, y0 + ph - 20);
              ctx.lineTo(x0 + pw, y0 + ph - 20);
              ctx.stroke();

              // Title
              ctx.fillStyle = color;
              ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText(title, x0 + pw / 2, y0 + 6);

              // Stats
              const mean = vals.reduce((a, v) => a + v, 0) / vals.length;
              const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length;
              ctx.fillStyle = viz.colors.text;
              ctx.font = '10px monospace';
              ctx.fillText('\u03BC=' + mean.toFixed(3) + '  \u03C3\u00B2=' + variance.toFixed(3), x0 + pw / 2, y0 + ph - 12);

              // Range labels
              ctx.font = '9px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText(lo.toFixed(1), x0 + 5, y0 + ph - 8);
              ctx.textAlign = 'right';
              ctx.fillText(hi.toFixed(1), x0 + pw - 5, y0 + ph - 8);
            }

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              const data = simulate();
              const margin = 20;
              const pw = (viz.width - 3 * margin) / 2;
              const ph = viz.height - 2 * margin;

              drawHistPanel(data.raw, margin, margin, pw, ph, 'Without Batch Normalization', viz.colors.red);
              drawHistPanel(data.bn, 2 * margin + pw, margin, pw, ph, 'With Batch Normalization (\u03B3=' + gamma.toFixed(1) + ', \u03B2=' + beta.toFixed(1) + ')', viz.colors.teal);

              // Info
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Depth: ' + numLayers + ' layers  |  Weight scale: ' + weightScale.toFixed(1) + '\u00D7', viz.width / 2, viz.height - 5);
            }

            draw();

            VizEngine.createSlider(controls, 'Depth', 1, 15, numLayers, 1, v => { numLayers = Math.round(v); draw(); });
            VizEngine.createSlider(controls, 'W scale', 0.2, 3.0, weightScale, 0.1, v => { weightScale = v; draw(); });
            VizEngine.createSlider(controls, '\u03B3', 0.1, 3.0, gamma, 0.1, v => { gamma = v; draw(); });
            VizEngine.createSlider(controls, '\u03B2', -2.0, 2.0, beta, 0.1, v => { beta = v; draw(); });
            VizEngine.createButton(controls, '\u21BB Resample', () => draw());

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'In the Batch Normalization transform, why is the small constant \\(\\epsilon\\) added inside the square root?',
          hint: 'Consider what happens if all values in the mini-batch are identical.',
          solution: 'If all values in the mini-batch are identical, \\(\\sigma^2_{\\mathcal{B}} = 0\\), and we would divide by zero. The constant \\(\\epsilon\\) (typically \\(10^{-5}\\)) prevents this numerical instability. It also ensures the gradient \\(\\partial \\hat{x}_i / \\partial \\sigma^2_{\\mathcal{B}}\\) remains bounded.'
        },
        {
          question: 'Explain why the bias term \\(b\\) in \\(z = Wx + b\\) is redundant when BN follows the linear layer.',
          hint: 'Write out the full BN computation starting from \\(z = Wx + b\\).',
          solution: 'BN computes \\(\\hat{z}_i = (z_i - \\mu_{\\mathcal{B}}) / \\sqrt{\\sigma^2_{\\mathcal{B}} + \\epsilon}\\). Since \\(\\mu_{\\mathcal{B}} = \\frac{1}{m}\\sum_i (Wx_i + b) = \\frac{1}{m}\\sum_i Wx_i + b\\), the bias \\(b\\) is subtracted out in the centering step: \\(z_i - \\mu_{\\mathcal{B}} = Wx_i + b - (\\overline{Wx} + b) = Wx_i - \\overline{Wx}\\). The bias cancels completely, and the learnable shift \\(\\beta\\) in BN serves the same role. Therefore keeping \\(b\\) wastes parameters.'
        },
        {
          question: 'During inference, BN uses running statistics instead of batch statistics. Describe a scenario where this distinction leads to a significant performance difference.',
          hint: 'Think about what happens if the model is accidentally left in training mode during evaluation with a small test batch.',
          solution: 'Suppose the model is left in <code>model.train()</code> mode during evaluation, and test samples are processed one at a time (batch size 1). Then \\(\\mu_{\\mathcal{B}} = x\\) and \\(\\sigma^2_{\\mathcal{B}} = 0\\), so BN produces \\(\\hat{x} = 0\\) for every input, yielding constant (and useless) outputs. With <code>model.eval()</code>, BN uses the stable running statistics accumulated over many training batches, producing meaningful normalized values. This train/eval mismatch is one of the most common bugs in deployed models using BN.'
        },
        {
          question: 'Batch Normalization has been described as providing implicit regularization. Explain the mechanism and how it differs from dropout.',
          hint: 'Consider the stochasticity introduced by mini-batch sampling.',
          solution: 'BN introduces stochasticity because the mean \\(\\mu_{\\mathcal{B}}\\) and variance \\(\\sigma^2_{\\mathcal{B}}\\) change from mini-batch to mini-batch. Each sample\'s normalized value depends on which other samples happen to be in the same mini-batch, adding noise to the forward pass. This noise is a function of the batch composition rather than an explicit random mask (as in dropout). The regularization effect strengthens with smaller batch sizes (more noise) and weakens with larger batches. Unlike dropout, BN\'s noise is correlated across features and is not independently applied to each unit.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 3: Layer & Group Normalization
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch06-sec03',
      title: '3. Layer & Group Normalization',
      content: `
<h2>Layer & Group Normalization</h2>

<div class="env-block intuition">
<strong>Section Roadmap.</strong>
Batch Normalization's reliance on batch statistics creates problems for small batches and sequential models. This section introduces three alternatives that normalize over different dimensions of the activation tensor: <strong>Layer Normalization</strong> (Ba et al., 2016), <strong>Instance Normalization</strong> (Ulyanov et al., 2016), and <strong>Group Normalization</strong> (Wu and He, 2018). Understanding <em>which</em> dimensions are normalized is the key insight.
</div>

<h3>The Geometry of Normalization</h3>

<p>
Consider a 4D activation tensor with shape \\((N, C, H, W)\\): batch size \\(N\\), channels \\(C\\), spatial height \\(H\\), and width \\(W\\). Each normalization method computes its mean and variance over a different subset of these dimensions:
</p>

<div class="env-block definition">
<div class="env-title">Normalization Dimensions</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:0.5rem 0;">
<tr style="background:#1a1a40;">
  <th style="padding:8px;border:1px solid #30363d;">Method</th>
  <th style="padding:8px;border:1px solid #30363d;">Normalizes over</th>
  <th style="padding:8px;border:1px solid #30363d;">Statistics per</th>
  <th style="padding:8px;border:1px solid #30363d;">Use case</th>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">Batch Norm (BN)</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(N, H, W\\)</td>
  <td style="padding:8px;border:1px solid #30363d;">channel</td>
  <td style="padding:8px;border:1px solid #30363d;">CNNs (large batch)</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">Layer Norm (LN)</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(C, H, W\\)</td>
  <td style="padding:8px;border:1px solid #30363d;">sample</td>
  <td style="padding:8px;border:1px solid #30363d;">Transformers, RNNs</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">Instance Norm (IN)</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(H, W\\)</td>
  <td style="padding:8px;border:1px solid #30363d;">sample + channel</td>
  <td style="padding:8px;border:1px solid #30363d;">Style transfer</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">Group Norm (GN)</td>
  <td style="padding:8px;border:1px solid #30363d;">\\(C/G, H, W\\)</td>
  <td style="padding:8px;border:1px solid #30363d;">sample + group</td>
  <td style="padding:8px;border:1px solid #30363d;">CNNs (small batch)</td>
</tr>
</table>
</div>
</div>

<h3>Layer Normalization</h3>

<p>
Layer Normalization (LN) computes statistics across all features <em>within a single sample</em>, making it completely independent of other samples in the batch.
</p>

<div class="env-block theorem">
<div class="env-title">Layer Normalization</div>
<div class="env-body">
For the activations \\(\\mathbf{x} \\in \\mathbb{R}^D\\) of a single sample at a given layer:
\\[ \\mu = \\frac{1}{D} \\sum_{i=1}^{D} x_i, \\qquad \\sigma^2 = \\frac{1}{D} \\sum_{i=1}^{D} (x_i - \\mu)^2 \\]
\\[ \\text{LN}(x_i) = \\gamma_i \\cdot \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta_i \\]
where \\(\\gamma, \\beta \\in \\mathbb{R}^D\\) are learnable parameters (one per feature).
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Why LN for Transformers?</div>
<div class="env-body">
In a Transformer, the sequence length varies across samples, and the "batch" dimension is entangled with the sequence dimension. BN would compute statistics across different positions of different sequences, which is semantically meaningless. LN normalizes each token's representation vector independently, making it natural for self-attention architectures. This is why every major Transformer (BERT, GPT, T5) uses Layer Normalization.
</div>
</div>

<div class="env-block remark">
<div class="env-title">Pre-LN vs. Post-LN Transformers</div>
<div class="env-body">
The original Transformer (Vaswani et al., 2017) places LN <em>after</em> the residual addition: \\(\\text{LN}(x + \\text{Sublayer}(x))\\). This is called <strong>Post-LN</strong>. GPT-2 and many modern architectures instead place LN <em>before</em> the sublayer: \\(x + \\text{Sublayer}(\\text{LN}(x))\\). This <strong>Pre-LN</strong> variant is more stable during training and allows for larger learning rates without warmup.
</div>
</div>

<h3>Instance Normalization</h3>

<p>
Instance Normalization normalizes each channel of each sample independently, computing statistics over the spatial dimensions \\((H, W)\\) only. It was introduced for style transfer, where per-instance feature normalization removes style information (encoded in feature statistics) while preserving content.
</p>

<h3>Group Normalization</h3>

<p>
Group Normalization divides the \\(C\\) channels into \\(G\\) groups and normalizes within each group. It interpolates between LN (\\(G = 1\\), all channels in one group) and IN (\\(G = C\\), each channel is its own group).
</p>

<div class="env-block theorem">
<div class="env-title">Group Normalization</div>
<div class="env-body">
Divide the \\(C\\) channels into \\(G\\) groups, each of size \\(C/G\\). For sample \\(n\\) and group \\(g\\):
\\[ \\mu_{ng} = \\frac{1}{(C/G) \\cdot H \\cdot W} \\sum_{c \\in \\text{group } g} \\sum_{h,w} x_{nchw} \\]
\\[ \\sigma^2_{ng} = \\frac{1}{(C/G) \\cdot H \\cdot W} \\sum_{c \\in \\text{group } g} \\sum_{h,w} (x_{nchw} - \\mu_{ng})^2 \\]
Then normalize and apply per-channel learnable parameters \\(\\gamma_c, \\beta_c\\).
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Why Does Grouping Work?</div>
<div class="env-body">
Many visual features are learned in groups: early CNN layers often learn oriented edge detectors at multiple orientations, color channels, or frequency bands. Channels within a group tend to have correlated statistics. Normalizing within groups respects this structure while providing enough elements for stable mean/variance estimates, even with batch size 1.
</div>
</div>

<div class="env-block remark">
<div class="env-title">Practical Guidelines</div>
<div class="env-body">
<ul>
  <li><strong>CNNs with large batches (batch size &ge; 32):</strong> BN remains the default.</li>
  <li><strong>CNNs with small batches (detection, segmentation):</strong> GN with \\(G = 32\\) is preferred.</li>
  <li><strong>Transformers / RNNs:</strong> LN is the universal choice.</li>
  <li><strong>Style transfer:</strong> IN normalizes away style while preserving content.</li>
</ul>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-norm-dims"></div>
`,
      visualizations: [
        {
          id: 'viz-norm-dims',
          title: 'Normalization Methods: Which Dimensions?',
          description: 'This diagram shows a 3D activation tensor \\((N, C, H{\\times}W)\\) and highlights which elements are used to compute the mean and variance for each normalization method. Blue cells are included in the normalization group for a single output element (shown by the white dot).',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 760, height: 460, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            let currentMethod = 'BN';
            const methods = ['BN', 'LN', 'IN', 'GN'];
            const methodLabels = {
              'BN': 'Batch Normalization',
              'LN': 'Layer Normalization',
              'IN': 'Instance Normalization',
              'GN': 'Group Normalization (G=2)'
            };

            // Tensor dimensions for illustration
            const N = 4; // batch
            const C = 6; // channels
            const S = 4; // spatial (flattened H*W)

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 15px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText(methodLabels[currentMethod], viz.width / 2, 10);

              // Draw 2D grid: rows = (N samples), cols = (C channels)
              // Each cell represents a (H*W) spatial block
              const margin = { top: 50, left: 100, right: 30, bottom: 60 };
              const cellW = 85;
              const cellH = 55;
              const gap = 4;

              // Target element: sample 1, channel 2
              const targetN = 1;
              const targetC = 2;
              const G = 2; // groups for GN
              const groupSize = C / G;
              const targetGroup = Math.floor(targetC / groupSize);

              // Determine which cells are highlighted
              function isHighlighted(n, c) {
                switch (currentMethod) {
                  case 'BN': return c === targetC; // all samples, same channel
                  case 'LN': return n === targetN; // same sample, all channels
                  case 'IN': return n === targetN && c === targetC; // same sample, same channel
                  case 'GN': {
                    const g = Math.floor(c / groupSize);
                    return n === targetN && g === targetGroup;
                  }
                }
                return false;
              }

              function isTarget(n, c) {
                return n === targetN && c === targetC;
              }

              // Draw axis labels
              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'bottom';
              ctx.fillText('Channels (C=' + C + ')', margin.left + C * (cellW + gap) / 2, margin.top - 5);

              ctx.save();
              ctx.translate(margin.left - 30, margin.top + N * (cellH + gap) / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillText('Batch (N=' + N + ')', 0, 0);
              ctx.restore();

              // Draw group boundaries for GN
              if (currentMethod === 'GN') {
                for (let g = 0; g < G; g++) {
                  const x0 = margin.left + g * groupSize * (cellW + gap) - 2;
                  const gWidth = groupSize * (cellW + gap) - gap + 4;
                  ctx.strokeStyle = viz.colors.purple + '66';
                  ctx.lineWidth = 2;
                  ctx.setLineDash([5, 3]);
                  ctx.strokeRect(x0, margin.top - 2, gWidth, N * (cellH + gap) - gap + 4);
                  ctx.setLineDash([]);

                  ctx.fillStyle = viz.colors.purple;
                  ctx.font = '10px -apple-system,sans-serif';
                  ctx.textAlign = 'center';
                  ctx.fillText('Group ' + (g + 1), x0 + gWidth / 2, margin.top + N * (cellH + gap) + 10);
                }
              }

              // Draw cells
              for (let n = 0; n < N; n++) {
                for (let c = 0; c < C; c++) {
                  const x = margin.left + c * (cellW + gap);
                  const y = margin.top + n * (cellH + gap);
                  const highlighted = isHighlighted(n, c);
                  const target = isTarget(n, c);

                  // Cell background
                  if (target) {
                    ctx.fillStyle = viz.colors.blue;
                  } else if (highlighted) {
                    ctx.fillStyle = viz.colors.blue + '55';
                  } else {
                    ctx.fillStyle = '#1a1a40';
                  }
                  ctx.fillRect(x, y, cellW, cellH);

                  // Cell border
                  ctx.strokeStyle = highlighted ? viz.colors.blue : '#30363d';
                  ctx.lineWidth = highlighted ? 2 : 1;
                  ctx.strokeRect(x, y, cellW, cellH);

                  // Draw small grid inside to represent spatial (H*W)
                  const innerGap = 2;
                  const innerCellW = (cellW - 8) / S;
                  const innerCellH = (cellH - 8) / S;
                  for (let sh = 0; sh < S; sh++) {
                    for (let sw = 0; sw < S; sw++) {
                      const ix = x + 4 + sw * innerCellW;
                      const iy = y + 4 + sh * innerCellH;
                      if (highlighted) {
                        ctx.fillStyle = target ? '#ffffff44' : viz.colors.blue + '33';
                      } else {
                        ctx.fillStyle = '#0f0f28';
                      }
                      ctx.fillRect(ix + 0.5, iy + 0.5, innerCellW - 1, innerCellH - 1);
                    }
                  }

                  // Target marker
                  if (target) {
                    ctx.fillStyle = viz.colors.white;
                    ctx.beginPath();
                    ctx.arc(x + cellW / 2, y + cellH / 2, 5, 0, Math.PI * 2);
                    ctx.fill();
                  }
                }

                // Row label
                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText('n=' + (n + 1), margin.left - 8, margin.top + n * (cellH + gap) + cellH / 2);
              }

              // Column labels
              for (let c = 0; c < C; c++) {
                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText('c=' + (c + 1), margin.left + c * (cellW + gap) + cellW / 2, margin.top + N * (cellH + gap) + (currentMethod === 'GN' ? 20 : 5));
              }

              // Legend
              const ly = viz.height - 45;
              ctx.fillStyle = viz.colors.blue;
              ctx.fillRect(viz.width / 2 - 160, ly, 14, 14);
              ctx.fillStyle = viz.colors.white;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText('Target element', viz.width / 2 - 140, ly + 7);

              ctx.fillStyle = viz.colors.blue + '55';
              ctx.fillRect(viz.width / 2 + 10, ly, 14, 14);
              ctx.fillStyle = viz.colors.white;
              ctx.fillText('Normalized together', viz.width / 2 + 30, ly + 7);

              // Description of which dimensions
              ctx.fillStyle = viz.colors.yellow;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              let desc = '';
              switch (currentMethod) {
                case 'BN': desc = 'Statistics computed over (N, H, W) for each channel'; break;
                case 'LN': desc = 'Statistics computed over (C, H, W) for each sample'; break;
                case 'IN': desc = 'Statistics computed over (H, W) for each sample \u00D7 channel'; break;
                case 'GN': desc = 'Statistics computed over (C/G, H, W) for each sample \u00D7 group'; break;
              }
              ctx.fillText(desc, viz.width / 2, viz.height - 10);
            }

            draw();

            // Method selector buttons
            for (const m of methods) {
              VizEngine.createButton(controls, m, () => {
                currentMethod = m;
                draw();
              });
            }

            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'For a Transformer with input shape \\((N, T, D)\\) where \\(T\\) is sequence length and \\(D\\) is embedding dimension, describe exactly which elements Layer Normalization computes its mean and variance over.',
          hint: 'Layer Normalization normalizes each sample independently. In a Transformer, each token at each position has a \\(D\\)-dimensional vector.',
          solution: 'Layer Normalization in a Transformer computes mean and variance over the \\(D\\) embedding dimensions for each token independently. That is, for sample \\(n\\) at position \\(t\\), it computes \\(\\mu_{nt} = \\frac{1}{D}\\sum_{d=1}^{D} x_{ntd}\\) and \\(\\sigma^2_{nt} = \\frac{1}{D}\\sum_{d=1}^{D}(x_{ntd} - \\mu_{nt})^2\\). Each of the \\(N \\times T\\) token vectors is normalized independently using its own statistics. This produces \\(N \\times T\\) independent normalizations, each over \\(D\\) values.'
        },
        {
          question: 'Group Normalization with \\(G = 1\\) is equivalent to which method? What about \\(G = C\\)?',
          hint: 'Think about what happens when all channels form a single group versus each channel being its own group.',
          solution: '\\(G = 1\\): All channels are in one group. Statistics are computed over all \\(C \\times H \\times W\\) values for each sample. This is exactly <strong>Layer Normalization</strong>. \\(G = C\\): Each channel is its own group. Statistics are computed over \\(H \\times W\\) values for each sample-channel pair. This is exactly <strong>Instance Normalization</strong>. GN thus provides a smooth interpolation between LN and IN.'
        },
        {
          question: 'Why is Batch Normalization problematic for object detection models like Faster R-CNN, and what normalization method is typically used instead?',
          hint: 'Object detection models often use small batch sizes due to high-resolution images consuming GPU memory.',
          solution: 'Object detection models typically process high-resolution images, so each image consumes substantial GPU memory. This limits the batch size to 1 or 2 images per GPU. With such small batches, BN\'s batch statistics (mean and variance) are extremely noisy and unreliable, degrading performance. Group Normalization (GN) is the preferred alternative because its statistics are computed per sample and are independent of batch size. Wu and He (2018) showed that GN with \\(G = 32\\) matches BN performance at large batch sizes and significantly outperforms BN when the batch size drops below 8.'
        }
      ]
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 4: Putting It Together
    // ═══════════════════════════════════════════════════════════════════════════
    {
      id: 'ch06-sec04',
      title: '4. Putting It Together',
      content: `
<h2>Putting It Together</h2>

<div class="env-block intuition">
<strong>Section Roadmap.</strong>
We have covered two separate mechanisms: initialization sets the starting point, and normalization stabilizes the trajectory. This section shows how they work <em>together</em> to enable training of very deep networks. We will compare training dynamics with and without these techniques, and establish practical recipes for configuring modern architectures.
</div>

<h3>The Synergy of Initialization and Normalization</h3>

<p>
Good initialization and normalization layers address the same fundamental problem, variance control, but at different stages:
</p>
<ul>
  <li><strong>Initialization</strong> ensures signals are well-scaled <em>at the start</em> of training. Without it, the first few gradient steps may be completely uninformative.</li>
  <li><strong>Normalization</strong> ensures signals remain well-scaled <em>throughout</em> training, even as the parameters evolve.</li>
</ul>

<p>
With both in place, a network can be trained reliably even at substantial depth. Without either, deep training often fails entirely.
</p>

<div class="env-block theorem">
<div class="env-title">The Gradient Flow Perspective</div>
<div class="env-body">
Recall that the gradient of the loss with respect to a weight in layer \\(l\\) involves a product of Jacobians:
\\[ \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\frac{\\partial \\mathcal{L}}{\\partial z^{(L)}} \\cdot \\prod_{k=l+1}^{L} \\frac{\\partial z^{(k)}}{\\partial z^{(k-1)}} \\cdot \\frac{\\partial z^{(l)}}{\\partial W^{(l)}} \\]
For this product to neither explode nor vanish, each Jacobian \\(\\partial z^{(k)} / \\partial z^{(k-1)}\\) must have spectral norm close to 1. Proper initialization achieves this at step 0. Normalization layers act as a continuous correction, constraining the Jacobian norms throughout training.
</div>
</div>

<h3>Common Architecture Patterns</h3>

<div class="env-block definition">
<div class="env-title">Modern Recipes</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;margin:0.5rem 0;">
<tr style="background:#1a1a40;">
  <th style="padding:8px;border:1px solid #30363d;">Architecture</th>
  <th style="padding:8px;border:1px solid #30363d;">Initialization</th>
  <th style="padding:8px;border:1px solid #30363d;">Normalization</th>
  <th style="padding:8px;border:1px solid #30363d;">Notes</th>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">ResNet</td>
  <td style="padding:8px;border:1px solid #30363d;">He</td>
  <td style="padding:8px;border:1px solid #30363d;">BN</td>
  <td style="padding:8px;border:1px solid #30363d;">BN before ReLU; final BN in residual branch init to 0</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">GPT / LLaMA</td>
  <td style="padding:8px;border:1px solid #30363d;">Scaled Xavier</td>
  <td style="padding:8px;border:1px solid #30363d;">Pre-LN (RMSNorm)</td>
  <td style="padding:8px;border:1px solid #30363d;">Output projections scaled by \\(1/\\sqrt{2L}\\)</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">ViT</td>
  <td style="padding:8px;border:1px solid #30363d;">Xavier (truncated)</td>
  <td style="padding:8px;border:1px solid #30363d;">Pre-LN</td>
  <td style="padding:8px;border:1px solid #30363d;">Patch embedding + class token</td>
</tr>
<tr>
  <td style="padding:8px;border:1px solid #30363d;">U-Net (detection)</td>
  <td style="padding:8px;border:1px solid #30363d;">He</td>
  <td style="padding:8px;border:1px solid #30363d;">GN (G=32)</td>
  <td style="padding:8px;border:1px solid #30363d;">Small batch sizes in dense prediction</td>
</tr>
</table>
</div>
</div>

<h3>What Happens Without These Techniques?</h3>

<p>
To appreciate the impact, consider training a 20-layer ReLU MLP under three configurations:
</p>
<ol>
  <li><strong>No tricks:</strong> Random \\(\\mathcal{N}(0, 0.01^2)\\) initialization, no normalization. Gradients vanish within the first epoch; the loss plateaus immediately.</li>
  <li><strong>Xavier + BN:</strong> Xavier initialization with Batch Normalization after each linear layer. The loss decreases steadily, converging in reasonable time.</li>
  <li><strong>He + LN:</strong> He initialization with Layer Normalization. Similar convergence, slightly different trajectory, but equally stable.</li>
</ol>

<div class="env-block remark">
<div class="env-title">Residual Connections Change the Game</div>
<div class="env-body">
<p>
Residual connections (Chapter 11) add another mechanism for gradient flow: the skip connection provides a direct path for gradients to bypass layers. With residual connections, very deep networks (100+ layers) become trainable even with moderate initialization and normalization. In a sense, residual connections, proper initialization, and normalization form a triad that enables modern deep learning at scale.
</p>
</div>
</div>

<h3>RMSNorm: A Simplified Alternative</h3>

<p>
RMSNorm (Zhang and Sennrich, 2019) simplifies Layer Normalization by removing the mean-centering step:
</p>

\\[ \\text{RMSNorm}(x_i) = \\gamma_i \\cdot \\frac{x_i}{\\sqrt{\\frac{1}{D}\\sum_{j=1}^{D} x_j^2 + \\epsilon}} \\]

<p>
This is computationally cheaper (no need to compute and subtract the mean) and has been adopted by many modern LLMs (LLaMA, Gemma, Mistral). Empirically, the re-centering step of LN contributes little, while the re-scaling step is the essential component.
</p>

<h3>Looking Ahead</h3>

<p>
With initialization and normalization in hand, we have the tools to train deep networks reliably. Chapter 7 will address <em>how</em> to traverse the loss landscape efficiently using advanced optimization algorithms (SGD with momentum, Adam, learning rate schedules), and Chapter 8 will cover techniques to prevent overfitting (dropout, weight decay, data augmentation).
</p>

<div class="viz-placeholder" data-viz="viz-training-curves"></div>
`,
      visualizations: [
        {
          id: 'viz-training-curves',
          title: 'Training Curves: Initialization + Normalization',
          description: 'Simulated training loss for a 20-layer MLP under three configurations. Observe how the combination of proper initialization and normalization enables convergence, while the unassisted network stalls. Click "Run" to restart the animation.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 760, height: 420, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;

            // Simulate training curves with realistic characteristics
            function generateCurve(style) {
              const nSteps = 200;
              const curve = [];
              let loss;
              switch (style) {
                case 'none': {
                  // No tricks: loss barely decreases, plateaus high
                  loss = 2.3; // ~ ln(10) for 10-class
                  for (let i = 0; i < nSteps; i++) {
                    loss -= 0.0002 * Math.exp(-i / 20) + (Math.random() - 0.5) * 0.015;
                    loss = Math.max(loss, 2.25);
                    curve.push(loss);
                  }
                  break;
                }
                case 'xavier_bn': {
                  // Xavier + BN: smooth convergence
                  loss = 2.3;
                  for (let i = 0; i < nSteps; i++) {
                    const lr = 0.015 * Math.exp(-i / 80);
                    loss -= lr + (Math.random() - 0.5) * 0.008;
                    loss = Math.max(loss, 0.15);
                    curve.push(loss);
                  }
                  break;
                }
                case 'he_ln': {
                  // He + LN: slightly different trajectory
                  loss = 2.3;
                  for (let i = 0; i < nSteps; i++) {
                    const lr = 0.012 * Math.exp(-i / 90);
                    loss -= lr + (Math.random() - 0.5) * 0.01;
                    loss = Math.max(loss, 0.18);
                    curve.push(loss);
                  }
                  break;
                }
              }
              return curve;
            }

            let curves = null;
            let animFrame = 0;
            let isRunning = false;

            function startSimulation() {
              curves = {
                none: generateCurve('none'),
                xavier_bn: generateCurve('xavier_bn'),
                he_ln: generateCurve('he_ln')
              };
              animFrame = 0;
              isRunning = true;
            }

            function drawFrame() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              if (!curves) return;

              const margin = { top: 40, bottom: 50, left: 65, right: 30 };
              const plotW = viz.width - margin.left - margin.right;
              const plotH = viz.height - margin.top - margin.bottom;

              // Plot area background
              ctx.fillStyle = '#0a0a1e';
              ctx.fillRect(margin.left, margin.top, plotW, plotH);

              // Grid
              ctx.strokeStyle = '#1a1a3a';
              ctx.lineWidth = 0.5;
              for (let i = 0; i <= 5; i++) {
                const y = margin.top + i * plotH / 5;
                ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
              }
              for (let i = 0; i <= 10; i++) {
                const x = margin.left + i * plotW / 10;
                ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
              }

              // Axes
              ctx.strokeStyle = viz.colors.axis;
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(margin.left, margin.top);
              ctx.lineTo(margin.left, margin.top + plotH);
              ctx.lineTo(margin.left + plotW, margin.top + plotH);
              ctx.stroke();

              // Y-axis labels (loss)
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'right';
              ctx.textBaseline = 'middle';
              const yMin = 0.0, yMax = 2.5;
              for (let i = 0; i <= 5; i++) {
                const val = yMax - i * (yMax - yMin) / 5;
                const y = margin.top + i * plotH / 5;
                ctx.fillText(val.toFixed(1), margin.left - 8, y);
              }

              // X-axis labels (epoch)
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              const nSteps = 200;
              for (let i = 0; i <= 10; i++) {
                const val = Math.round(i * nSteps / 10);
                const x = margin.left + i * plotW / 10;
                ctx.fillText(val, x, margin.top + plotH + 5);
              }

              // Axis titles
              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Training Step', margin.left + plotW / 2, viz.height - 10);

              ctx.save();
              ctx.translate(18, margin.top + plotH / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillText('Loss', 0, 0);
              ctx.restore();

              // Draw curves
              const drawN = Math.min(animFrame, nSteps);
              const configs = [
                { key: 'none', color: viz.colors.red, label: 'No tricks (N(0,0.01\u00B2))' },
                { key: 'xavier_bn', color: viz.colors.teal, label: 'Xavier + BN' },
                { key: 'he_ln', color: viz.colors.green, label: 'He + LN' }
              ];

              for (const cfg of configs) {
                const data = curves[cfg.key];
                ctx.strokeStyle = cfg.color;
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                for (let i = 0; i < drawN; i++) {
                  const x = margin.left + (i / (nSteps - 1)) * plotW;
                  const y = margin.top + ((yMax - data[i]) / (yMax - yMin)) * plotH;
                  if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
              }

              // Legend
              const lx = margin.left + plotW - 200;
              const ly = margin.top + 15;
              ctx.fillStyle = '#0a0a1ecc';
              ctx.fillRect(lx - 10, ly - 8, 210, 70);
              ctx.strokeStyle = '#30363d';
              ctx.lineWidth = 1;
              ctx.strokeRect(lx - 10, ly - 8, 210, 70);

              configs.forEach((cfg, i) => {
                const yy = ly + i * 20;
                ctx.strokeStyle = cfg.color;
                ctx.lineWidth = 2.5;
                ctx.beginPath(); ctx.moveTo(lx, yy + 4); ctx.lineTo(lx + 25, yy + 4); ctx.stroke();
                ctx.fillStyle = cfg.color;
                ctx.font = '11px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(cfg.label, lx + 30, yy + 4);
              });

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillText('20-Layer MLP Training Loss Comparison', viz.width / 2, 10);

              if (isRunning && animFrame < nSteps) {
                animFrame += 2;
              }
            }

            startSimulation();

            viz.animate((t) => {
              drawFrame();
            });

            VizEngine.createButton(controls, '\u25B6 Run', () => {
              startSimulation();
            });

            VizEngine.createButton(controls, '\u23F8 Pause/Resume', () => {
              isRunning = !isRunning;
            });

            return { stopAnimation: () => viz.stopAnimation() };
          }
        }
      ],
      exercises: [
        {
          question: 'A colleague initializes a 50-layer ReLU network with Xavier initialization (instead of He) and uses no normalization. Predict what will happen to the activation variances and explain why.',
          hint: 'Xavier assumes the activation is approximately linear (no variance halving). ReLU halves the variance at each layer.',
          solution: 'Xavier initialization sets \\(\\sigma^2_w = 2/(n_{\\text{in}} + n_{\\text{out}})\\). For a ReLU network, the variance is halved at each activation, so the effective variance multiplier per layer is approximately \\(n_{\\text{in}} \\cdot \\frac{2}{n_{\\text{in}} + n_{\\text{out}}} \\cdot \\frac{1}{2} = \\frac{n_{\\text{in}}}{n_{\\text{in}} + n_{\\text{out}}}\\). For equal widths, this is \\(0.5\\). After 50 layers, the variance shrinks by \\(0.5^{50} \\approx 10^{-15}\\). Activations in the final layers will be essentially zero, and gradients will vanish. He initialization fixes this by using \\(2/n_{\\text{in}}\\), giving a per-layer multiplier of 1 after accounting for ReLU\'s halving.'
        },
        {
          question: 'In ResNet, the final BN layer in each residual branch is sometimes initialized with \\(\\gamma = 0\\). Why would this be beneficial?',
          hint: 'Consider the residual connection: the output is \\(x + F(x)\\). What happens if \\(F(x) = 0\\) at initialization?',
          solution: 'If the final BN in the residual branch has \\(\\gamma = 0\\), then the BN output is zero (since \\(y = \\gamma \\hat{x} + \\beta\\) with \\(\\gamma = 0, \\beta = 0\\) gives \\(y = 0\\)). This means the residual function \\(F(x) = 0\\) at initialization, so the block computes the identity: \\(\\text{output} = x + 0 = x\\). This effectively makes the network behave like a shallower network at the start of training, which is easier to optimize. As training progresses, \\(\\gamma\\) moves away from zero and the residual branches gradually begin contributing. This is known as the "zero-init" trick and was used in Goyal et al. (2017) for training very deep ResNets.'
        },
        {
          question: 'Compare RMSNorm and Layer Normalization. When would the mean-centering step of LN matter, and when is it safe to drop it (use RMSNorm)?',
          hint: 'Consider what mean-centering does geometrically to the feature vector, and think about architectures where feature means carry important information.',
          solution: 'Layer Normalization performs two operations: (1) centering (subtract mean) and (2) scaling (divide by standard deviation). RMSNorm only does scaling (divide by RMS). The mean-centering step removes the DC component of the feature vector, projecting it onto a hyperplane. This is safe when the mean carries no useful information, which is the case in most Transformer architectures (the important information is in the relative magnitudes and directions of features, not the offset). RMSNorm is therefore a drop-in replacement in Transformers with negligible performance loss but lower computational cost. Mean-centering <em>would</em> matter if the absolute activation level encodes information, for example in some CNN architectures where the mean activation of a channel carries semantic meaning (e.g., average brightness). In practice, LLaMA, Gemma, and Mistral all use RMSNorm without meaningful quality loss.'
        },
        {
          question: 'Design an initialization scheme for a network that alternates between linear layers and GELU activations. Justify your choice.',
          hint: 'GELU is smoother than ReLU and does not zero out exactly half of its inputs. Consider the expected variance reduction of GELU.',
          solution: 'GELU is defined as \\(\\text{GELU}(x) = x \\cdot \\Phi(x)\\), where \\(\\Phi\\) is the standard normal CDF. Unlike ReLU, GELU does not hard-zero negative inputs; it smoothly attenuates them. The variance reduction factor for GELU applied to a standard normal input is approximately 0.425 (between ReLU\'s 0.5 and the identity\'s 1.0). So the appropriate initialization is \\(\\sigma^2_w = 1 / (0.425 \\cdot n_{\\text{in}}) \\approx 2.35 / n_{\\text{in}}\\). In practice, He initialization (\\(2/n_{\\text{in}}\\)) is commonly used for GELU networks and works well, since the difference between 2.0 and 2.35 is small. Some frameworks (e.g., GPT-2) use Xavier initialization with an additional scaling factor for residual branches rather than adjusting for the exact activation function.'
        }
      ]
    }
  ]
});
