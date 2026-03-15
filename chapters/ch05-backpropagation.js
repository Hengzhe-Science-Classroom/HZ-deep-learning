window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch05',
  number: 5,
  title: 'Backpropagation',
  subtitle: 'The chain rule at scale: how neural networks learn by propagating error signals backward through computation graphs',
  sections: [

    // ─── SECTION 1: The Chain Rule ──────────────────────────────────────────
    {
      id: 'ch05-sec01',
      title: '1. The Chain Rule',
      content: `
<div class="env-block intuition">
<strong>Where we are.</strong>
In Chapter 4 we built multilayer perceptrons and saw that deeper networks can represent richer functions.
But representation is only half the story: we need to <em>train</em> these networks, and training requires
computing the gradient of the loss with respect to every weight.
This chapter develops the backpropagation algorithm, which makes that computation tractable.
We begin with the mathematical foundation that makes it all possible: the chain rule.
</div>

<h2>The Chain Rule</h2>

<p><strong>Section roadmap.</strong>
We review the single-variable chain rule, generalize to the multivariate case via Jacobians,
and see why function composition is the natural setting for neural network gradient computation.</p>

<p>
A neural network is a composition of simple functions: affine transformations followed by element-wise
nonlinearities, stacked layer after layer. To differentiate such a composition, we need the
<em>chain rule</em>, the single most important result in differential calculus for deep learning.
</p>

<h3>Single-Variable Chain Rule</h3>

<div class="env-block definition">
<div class="env-title">Definition 5.1 (Chain Rule, Single Variable)</div>
<div class="env-body">
If \\(y = f(u)\\) and \\(u = g(x)\\) are both differentiable, then the composite function
\\(y = f(g(x))\\) is differentiable, and
\\[
  \\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx} = f'(g(x)) \\cdot g'(x).
\\]
</div>
</div>

<p>
The key insight is <em>locality</em>: each function only needs to know its own derivative,
and the derivatives multiply together. This is exactly the principle behind backpropagation.
</p>

<div class="env-block example">
<div class="env-title">Example 5.1</div>
<div class="env-body">
Let \\(y = \\sin(x^2)\\). We identify \\(u = g(x) = x^2\\) and \\(y = f(u) = \\sin(u)\\). Then:
\\[
  \\frac{dy}{dx} = \\cos(u) \\cdot 2x = 2x\\cos(x^2).
\\]
</div>
</div>

<h3>Multivariate Chain Rule</h3>

<p>
Neural networks are vector-valued functions of vector inputs.
We need the chain rule for the multivariate case.
</p>

<div class="env-block definition">
<div class="env-title">Definition 5.2 (Jacobian Matrix)</div>
<div class="env-body">
<p>
Let \\(\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m\\) be differentiable at \\(\\mathbf{x}\\).
The <em>Jacobian matrix</em> \\(\\mathbf{J}_{\\mathbf{f}}(\\mathbf{x}) \\in \\mathbb{R}^{m \\times n}\\) is defined as:
</p>
\\[
  \\mathbf{J}_{\\mathbf{f}}(\\mathbf{x}) = \\begin{pmatrix}
    \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\
    \\vdots & \\ddots & \\vdots \\\\
    \\frac{\\partial f_m}{\\partial x_1} & \\cdots & \\frac{\\partial f_m}{\\partial x_n}
  \\end{pmatrix}.
\\]
<p>Row \\(i\\) of the Jacobian is the gradient \\(\\nabla f_i\\) of the \\(i\\)-th output component.</p>
</div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 5.1 (Multivariate Chain Rule)</div>
<div class="env-body">
<p>
If \\(\\mathbf{y} = \\mathbf{f}(\\mathbf{u})\\) where \\(\\mathbf{f}: \\mathbb{R}^k \\to \\mathbb{R}^m\\),
and \\(\\mathbf{u} = \\mathbf{g}(\\mathbf{x})\\) where \\(\\mathbf{g}: \\mathbb{R}^n \\to \\mathbb{R}^k\\),
then the Jacobian of the composition \\(\\mathbf{f} \\circ \\mathbf{g}\\) is:
</p>
\\[
  \\mathbf{J}_{\\mathbf{f} \\circ \\mathbf{g}}(\\mathbf{x}) = \\mathbf{J}_{\\mathbf{f}}(\\mathbf{g}(\\mathbf{x})) \\cdot \\mathbf{J}_{\\mathbf{g}}(\\mathbf{x}).
\\]
<p>
In other words, the Jacobian of a composition is the <em>product</em> of the individual Jacobians.
This is the matrix generalization of "multiply the derivatives."
</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Intuition: Why Matrix Multiplication?</div>
<div class="env-body">
<p>
Consider a small perturbation \\(\\Delta \\mathbf{x}\\). The function \\(\\mathbf{g}\\) maps it to
\\(\\Delta \\mathbf{u} \\approx \\mathbf{J}_{\\mathbf{g}} \\Delta \\mathbf{x}\\), and then \\(\\mathbf{f}\\) maps that to
\\(\\Delta \\mathbf{y} \\approx \\mathbf{J}_{\\mathbf{f}} \\Delta \\mathbf{u} = \\mathbf{J}_{\\mathbf{f}} \\mathbf{J}_{\\mathbf{g}} \\Delta \\mathbf{x}\\).
Composing linear approximations is just matrix multiplication.
</p>
</div>
</div>

<h3>Scalar Loss and the Gradient</h3>

<p>
In deep learning, the final output is typically a scalar loss \\(L \\in \\mathbb{R}\\).
When \\(m = 1\\), the Jacobian \\(\\mathbf{J}_{L}\\) is a row vector, the transpose of the gradient \\(\\nabla L\\).
The chain rule for the gradient of a scalar loss through a chain of functions
\\(L = f_K \\circ f_{K-1} \\circ \\cdots \\circ f_1(\\mathbf{x})\\) becomes:
</p>
\\[
  \\nabla_{\\mathbf{x}} L = \\mathbf{J}_{f_1}^\\top \\mathbf{J}_{f_2}^\\top \\cdots \\mathbf{J}_{f_K}^\\top.
\\]
<p>
The order of multiplication matters: we can evaluate this product from right to left
(starting at the loss and working backward through the network).
This is exactly what backpropagation does.
</p>

<div class="viz-placeholder" data-viz="viz-chain-rule"></div>
`,
      visualizations: [
        {
          id: 'viz-chain-rule',
          title: 'Chain Rule on a Composition \\(f(g(x))\\)',
          description: 'Drag the input point \\(x\\) to see how the chain rule decomposes \\(df/dx\\) into \\(df/dg \\cdot dg/dx\\). Animated arrows show the flow of derivatives.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 700, height: 420, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;

            let xVal = 1.5;
            VizEngine.createSlider(controls, 'x', -3, 3, xVal, 0.1, v => { xVal = v; });

            let animT = 0;

            function g(x) { return x * x; }
            function gp(x) { return 2 * x; }
            function f(u) { return Math.sin(u); }
            function fp(u) { return Math.cos(u); }

            function draw(t) {
              animT = t * 0.001;
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              const cx1 = 130, cx2 = 350, cx3 = 570;
              const cy = 140;
              const boxW = 100, boxH = 50;

              // Node boxes
              const boxes = [
                { x: cx1, y: cy, label: 'x', color: viz.colors.blue },
                { x: cx2, y: cy, label: 'u = g(x) = x\u00B2', color: viz.colors.teal },
                { x: cx3, y: cy, label: 'y = f(u) = sin(u)', color: viz.colors.orange }
              ];

              boxes.forEach(b => {
                ctx.strokeStyle = b.color;
                ctx.lineWidth = 2;
                ctx.strokeRect(b.x - boxW / 2, b.y - boxH / 2, boxW, boxH);
                ctx.fillStyle = b.color + '15';
                ctx.fillRect(b.x - boxW / 2, b.y - boxH / 2, boxW, boxH);
                ctx.fillStyle = b.color;
                ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(b.label, b.x, b.y);
              });

              // Forward arrows
              const arrowY = cy;
              drawArrowH(ctx, cx1 + boxW / 2, arrowY, cx2 - boxW / 2, arrowY, viz.colors.blue, 'g');
              drawArrowH(ctx, cx2 + boxW / 2, arrowY, cx3 - boxW / 2, arrowY, viz.colors.teal, 'f');

              // Compute values
              const uVal = g(xVal);
              const yVal = f(uVal);
              const dgdx = gp(xVal);
              const dfdu = fp(uVal);
              const dydx = dfdu * dgdx;

              // Display values
              ctx.font = '13px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillStyle = viz.colors.blue;
              ctx.fillText('x = ' + xVal.toFixed(2), cx1, cy + 45);
              ctx.fillStyle = viz.colors.teal;
              ctx.fillText('u = ' + uVal.toFixed(3), cx2, cy + 45);
              ctx.fillStyle = viz.colors.orange;
              ctx.fillText('y = ' + yVal.toFixed(3), cx3, cy + 45);

              // Backward gradient flow (animated)
              const gradY = cy + 110;
              const pulse = 0.5 + 0.5 * Math.sin(animT * 3);

              // dy/dy = 1
              ctx.fillStyle = viz.colors.red;
              ctx.font = 'bold 13px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('\u2202y/\u2202y = 1', cx3, gradY - 20);

              // Backward arrow: dy/du
              const alpha1 = Math.min(1, Math.max(0, (pulse * 2)));
              ctx.globalAlpha = 0.3 + 0.7 * alpha1;
              drawArrowH(ctx, cx3 - boxW / 2, gradY, cx2 + boxW / 2, gradY, viz.colors.red, '');
              ctx.globalAlpha = 1;
              ctx.fillStyle = viz.colors.red;
              ctx.fillText('\u2202y/\u2202u = cos(u) = ' + dfdu.toFixed(3), (cx2 + cx3) / 2, gradY + 20);

              // Backward arrow: du/dx
              const alpha2 = Math.min(1, Math.max(0, (pulse * 2 - 1)));
              ctx.globalAlpha = 0.3 + 0.7 * alpha2;
              drawArrowH(ctx, cx2 - boxW / 2, gradY, cx1 + boxW / 2, gradY, viz.colors.purple, '');
              ctx.globalAlpha = 1;
              ctx.fillStyle = viz.colors.purple;
              ctx.fillText('\u2202u/\u2202x = 2x = ' + dgdx.toFixed(3), (cx1 + cx2) / 2, gradY + 20);

              // Result box
              const resY = gradY + 70;
              ctx.strokeStyle = viz.colors.yellow;
              ctx.lineWidth = 2;
              ctx.strokeRect(W / 2 - 200, resY - 20, 400, 44);
              ctx.fillStyle = viz.colors.yellow + '10';
              ctx.fillRect(W / 2 - 200, resY - 20, 400, 44);
              ctx.fillStyle = viz.colors.yellow;
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.fillText('dy/dx = (dy/du) \u00D7 (du/dx) = ' + dfdu.toFixed(3) + ' \u00D7 ' + dgdx.toFixed(3) + ' = ' + dydx.toFixed(3), W / 2, resY + 2);

              // Labels
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Forward pass (top): compute values left \u2192 right', 20, H - 40);
              ctx.fillText('Backward pass (bottom): propagate gradients right \u2192 left', 20, H - 22);
            }

            function drawArrowH(ctx, x1, y1, x2, y2, color, label) {
              const dx = x2 - x1;
              const dy = y2 - y1;
              const len = Math.sqrt(dx * dx + dy * dy);
              const angle = Math.atan2(dy, dx);

              ctx.strokeStyle = color;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2 - 10 * Math.cos(angle), y2 - 10 * Math.sin(angle));
              ctx.stroke();

              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.moveTo(x2, y2);
              ctx.lineTo(x2 - 12 * Math.cos(angle - Math.PI / 6), y2 - 12 * Math.sin(angle - Math.PI / 6));
              ctx.lineTo(x2 - 12 * Math.cos(angle + Math.PI / 6), y2 - 12 * Math.sin(angle + Math.PI / 6));
              ctx.closePath();
              ctx.fill();

              if (label) {
                ctx.fillStyle = color;
                ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(label, (x1 + x2) / 2, Math.min(y1, y2) - 6);
                ctx.textBaseline = 'middle';
              }
            }

            viz.animate(draw);
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Let \\(y = (3x^2 + 1)^5\\). Use the chain rule to find \\(\\frac{dy}{dx}\\).',
          hint: 'Identify the outer function \\(f(u) = u^5\\) and the inner function \\(u = g(x) = 3x^2 + 1\\). Apply \\(\\frac{dy}{dx} = f\'(u) \\cdot g\'(x)\\).',
          solution: 'Let \\(u = 3x^2 + 1\\). Then \\(\\frac{dy}{du} = 5u^4\\) and \\(\\frac{du}{dx} = 6x\\). By the chain rule: \\(\\frac{dy}{dx} = 5(3x^2 + 1)^4 \\cdot 6x = 30x(3x^2 + 1)^4\\).'
        },
        {
          question: 'Let \\(\\mathbf{f}: \\mathbb{R}^2 \\to \\mathbb{R}^2\\) with \\(\\mathbf{f}(u_1, u_2) = (u_1 u_2,\\; u_1 + u_2^2)\\), and \\(\\mathbf{g}: \\mathbb{R} \\to \\mathbb{R}^2\\) with \\(\\mathbf{g}(x) = (x^2,\\; 3x)\\). Compute the Jacobian \\(\\mathbf{J}_{\\mathbf{f} \\circ \\mathbf{g}}(x)\\) at \\(x = 1\\).',
          hint: 'First compute \\(\\mathbf{J}_{\\mathbf{f}}\\) (a \\(2 \\times 2\\) matrix) and \\(\\mathbf{J}_{\\mathbf{g}}\\) (a \\(2 \\times 1\\) matrix). Evaluate at \\(\\mathbf{g}(1) = (1, 3)\\), then multiply.',
          solution: '\\(\\mathbf{J}_{\\mathbf{f}} = \\begin{pmatrix} u_2 & u_1 \\\\ 1 & 2u_2 \\end{pmatrix}\\). At \\(\\mathbf{g}(1) = (1, 3)\\): \\(\\mathbf{J}_{\\mathbf{f}} = \\begin{pmatrix} 3 & 1 \\\\ 1 & 6 \\end{pmatrix}\\). And \\(\\mathbf{J}_{\\mathbf{g}} = \\begin{pmatrix} 2x \\\\ 3 \\end{pmatrix}\\), at \\(x=1\\): \\(\\begin{pmatrix} 2 \\\\ 3 \\end{pmatrix}\\). Product: \\(\\begin{pmatrix} 3 & 1 \\\\ 1 & 6 \\end{pmatrix}\\begin{pmatrix} 2 \\\\ 3 \\end{pmatrix} = \\begin{pmatrix} 9 \\\\ 20 \\end{pmatrix}\\).'
        },
        {
          question: 'A neural network computes \\(L = \\ell(\\sigma(\\mathbf{W}\\mathbf{x} + \\mathbf{b}))\\) where \\(\\sigma\\) is applied element-wise. Write the expression for \\(\\nabla_{\\mathbf{W}} L\\) using the chain rule and Jacobians.',
          hint: 'Define intermediate variables: \\(\\mathbf{z} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}\\), \\(\\mathbf{a} = \\sigma(\\mathbf{z})\\), \\(L = \\ell(\\mathbf{a})\\). Apply the chain rule through each step.',
          solution: 'Let \\(\\mathbf{z} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}\\), \\(\\mathbf{a} = \\sigma(\\mathbf{z})\\). Then \\(\\frac{\\partial L}{\\partial \\mathbf{a}} = \\nabla_{\\mathbf{a}} \\ell\\), \\(\\frac{\\partial \\mathbf{a}}{\\partial \\mathbf{z}} = \\mathrm{diag}(\\sigma\'(\\mathbf{z}))\\), and \\(\\frac{\\partial z_i}{\\partial W_{ij}} = x_j\\). Combining: \\(\\frac{\\partial L}{\\partial W_{ij}} = \\delta_i x_j\\) where \\(\\boldsymbol{\\delta} = \\nabla_{\\mathbf{a}} \\ell \\odot \\sigma\'(\\mathbf{z})\\). In matrix form: \\(\\nabla_{\\mathbf{W}} L = \\boldsymbol{\\delta} \\mathbf{x}^\\top\\).'
        }
      ]
    },

    // ─── SECTION 2: Forward Pass ────────────────────────────────────────────
    {
      id: 'ch05-sec02',
      title: '2. Forward Pass',
      content: `
<h2>Forward Pass</h2>

<p><strong>Section roadmap.</strong>
Before we can compute gradients, we must first compute the network's output.
The <em>forward pass</em> evaluates the network layer by layer, from input to loss.
We formalize the computation and see why we must cache intermediate values for the backward pass.</p>

<h3>Layer-by-Layer Computation</h3>

<p>
Consider an \\(L\\)-layer feedforward network. The forward pass computes a sequence of intermediate values:
</p>
\\[
  \\mathbf{a}^{(0)} = \\mathbf{x}, \\quad
  \\mathbf{z}^{(l)} = \\mathbf{W}^{(l)} \\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\quad
  \\mathbf{a}^{(l)} = \\sigma^{(l)}(\\mathbf{z}^{(l)}), \\quad
  l = 1, \\ldots, L.
\\]
<p>
Here \\(\\mathbf{z}^{(l)}\\) is the <em>pre-activation</em> (linear part) and \\(\\mathbf{a}^{(l)}\\) is the
<em>post-activation</em> at layer \\(l\\). The final output \\(\\hat{\\mathbf{y}} = \\mathbf{a}^{(L)}\\)
is compared with the target \\(\\mathbf{y}\\) via a loss function:
</p>
\\[
  \\mathcal{L} = \\ell(\\hat{\\mathbf{y}}, \\mathbf{y}).
\\]

<div class="env-block definition">
<div class="env-title">Algorithm 5.1 (Forward Pass)</div>
<div class="env-body">
<p><strong>Input:</strong> Input vector \\(\\mathbf{x}\\), weights \\(\\{\\mathbf{W}^{(l)}, \\mathbf{b}^{(l)}\\}_{l=1}^L\\), target \\(\\mathbf{y}\\).</p>
<ol>
  <li>Set \\(\\mathbf{a}^{(0)} = \\mathbf{x}\\).</li>
  <li>For \\(l = 1, 2, \\ldots, L\\):
    <ul>
      <li>Compute \\(\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)} \\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}\\).</li>
      <li>Compute \\(\\mathbf{a}^{(l)} = \\sigma^{(l)}(\\mathbf{z}^{(l)})\\).</li>
      <li>Cache \\(\\mathbf{z}^{(l)}\\) and \\(\\mathbf{a}^{(l-1)}\\) (needed for the backward pass).</li>
    </ul>
  </li>
  <li>Compute loss \\(\\mathcal{L} = \\ell(\\mathbf{a}^{(L)}, \\mathbf{y})\\).</li>
</ol>
<p><strong>Output:</strong> Loss \\(\\mathcal{L}\\) and cached intermediates \\(\\{\\mathbf{z}^{(l)}, \\mathbf{a}^{(l)}\\}\\).</p>
</div>
</div>

<div class="env-block remark">
<div class="env-title">Remark: Why Cache?</div>
<div class="env-body">
<p>
The backward pass needs \\(\\mathbf{a}^{(l-1)}\\) to compute \\(\\nabla_{\\mathbf{W}^{(l)}} \\mathcal{L}\\),
and \\(\\mathbf{z}^{(l)}\\) to compute \\(\\sigma'(\\mathbf{z}^{(l)})\\).
Without caching, we would need to recompute the entire forward pass for every layer's gradient,
which would be prohibitively expensive.
This is the fundamental <em>time-memory tradeoff</em> in backpropagation:
we spend \\(O(\\sum_l n_l)\\) extra memory to save a factor of \\(L\\) in computation time.
</p>
</div>
</div>

<h3>Concrete Example: A 3-Layer Network</h3>

<p>
Consider a network with input dimension 2, hidden layers of size 3 and 2, and a single output (for regression).
The forward pass proceeds:
</p>
<ol>
  <li>\\(\\mathbf{a}^{(0)} = \\mathbf{x} \\in \\mathbb{R}^2\\)</li>
  <li>\\(\\mathbf{z}^{(1)} = \\mathbf{W}^{(1)} \\mathbf{a}^{(0)} + \\mathbf{b}^{(1)} \\in \\mathbb{R}^3\\),
      \\(\\mathbf{a}^{(1)} = \\sigma(\\mathbf{z}^{(1)}) \\in \\mathbb{R}^3\\)</li>
  <li>\\(\\mathbf{z}^{(2)} = \\mathbf{W}^{(2)} \\mathbf{a}^{(1)} + \\mathbf{b}^{(2)} \\in \\mathbb{R}^2\\),
      \\(\\mathbf{a}^{(2)} = \\sigma(\\mathbf{z}^{(2)}) \\in \\mathbb{R}^2\\)</li>
  <li>\\(\\mathbf{z}^{(3)} = \\mathbf{W}^{(3)} \\mathbf{a}^{(2)} + \\mathbf{b}^{(3)} \\in \\mathbb{R}^1\\),
      \\(\\hat{y} = a^{(3)} = z^{(3)}\\) (linear output for regression)</li>
  <li>\\(\\mathcal{L} = \\frac{1}{2}(\\hat{y} - y)^2\\)</li>
</ol>

<div class="env-block warning">
<div class="env-title">Warning: Matrix Dimensions</div>
<div class="env-body">
<p>
Always check dimensions. If layer \\(l\\) maps from \\(n_{l-1}\\) to \\(n_l\\) neurons, then
\\(\\mathbf{W}^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}\\) and
\\(\\mathbf{b}^{(l)} \\in \\mathbb{R}^{n_l}\\).
A mismatch in matrix dimensions is the most common implementation bug in neural networks.
</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-forward-pass"></div>
`,
      visualizations: [
        {
          id: 'viz-forward-pass',
          title: 'Animated Forward Pass Through a 3-Layer Network',
          description: 'Watch data flow left to right through the network. Each node shows its computed value. Adjust the input values and observe how activations propagate layer by layer.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 750, height: 420, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;

            let x1 = 0.8, x2 = -0.5;
            VizEngine.createSlider(controls, 'x\u2081', -2, 2, x1, 0.1, v => { x1 = v; });
            VizEngine.createSlider(controls, 'x\u2082', -2, 2, x2, 0.1, v => { x2 = v; });

            // Fixed small weights for visualization
            const W1 = [[0.5, -0.3], [0.8, 0.4], [-0.2, 0.7]];
            const b1 = [0.1, -0.1, 0.2];
            const W2 = [[0.6, -0.4, 0.3], [-0.5, 0.7, 0.2]];
            const b2 = [0.1, -0.2];
            const W3 = [[0.8, -0.6]];
            const b3 = [0.1];

            function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

            function forwardPass(input) {
              const a0 = input;
              const z1 = W1.map((row, i) => row.reduce((s, w, j) => s + w * a0[j], 0) + b1[i]);
              const a1 = z1.map(sigmoid);
              const z2 = W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + b2[i]);
              const a2 = z2.map(sigmoid);
              const z3 = W3.map((row, i) => row.reduce((s, w, j) => s + w * a2[j], 0) + b3[i]);
              return { a0, z1, a1, z2, a2, z3 };
            }

            // Layout
            const layers = [2, 3, 2, 1];
            const layerX = [100, 260, 420, 580];
            const layerLabels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];

            function nodeY(layerIdx, nodeIdx) {
              const n = layers[layerIdx];
              const totalH = (n - 1) * 70;
              return H / 2 - totalH / 2 + nodeIdx * 70;
            }

            let animPhase = 0;

            function draw(t) {
              animPhase = (t * 0.0008) % 4;
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              const fp = forwardPass([x1, x2]);
              const allVals = [fp.a0, fp.a1, fp.a2, fp.z3];
              const preVals = [null, fp.z1, fp.z2, fp.z3];

              // Draw connections
              for (let l = 0; l < layers.length - 1; l++) {
                for (let i = 0; i < layers[l]; i++) {
                  for (let j = 0; j < layers[l + 1]; j++) {
                    const x1c = layerX[l], y1c = nodeY(l, i);
                    const x2c = layerX[l + 1], y2c = nodeY(l + 1, j);
                    const progress = Math.max(0, Math.min(1, animPhase - l));
                    const alpha = l < Math.floor(animPhase) ? 0.4 : (progress > 0 ? 0.15 + 0.25 * progress : 0.1);
                    ctx.strokeStyle = viz.colors.text;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x1c + 18, y1c);
                    ctx.lineTo(x2c - 18, y2c);
                    ctx.stroke();
                  }
                }
              }
              ctx.globalAlpha = 1;

              // Draw animated data pulse
              for (let l = 0; l < layers.length - 1; l++) {
                const progress = Math.max(0, Math.min(1, animPhase - l));
                if (progress > 0 && progress < 1) {
                  for (let i = 0; i < layers[l]; i++) {
                    for (let j = 0; j < layers[l + 1]; j++) {
                      const x1c = layerX[l] + 18, y1c = nodeY(l, i);
                      const x2c = layerX[l + 1] - 18, y2c = nodeY(l + 1, j);
                      const px = x1c + (x2c - x1c) * progress;
                      const py = y1c + (y2c - y1c) * progress;
                      ctx.fillStyle = viz.colors.blue + '88';
                      ctx.beginPath();
                      ctx.arc(px, py, 3, 0, Math.PI * 2);
                      ctx.fill();
                    }
                  }
                }
              }

              // Draw nodes
              const nodeColors = [viz.colors.blue, viz.colors.teal, viz.colors.purple, viz.colors.orange];
              for (let l = 0; l < layers.length; l++) {
                const lit = l <= Math.floor(animPhase);
                for (let i = 0; i < layers[l]; i++) {
                  const nx = layerX[l], ny = nodeY(l, i);
                  const val = allVals[l][i];

                  // Node circle
                  ctx.beginPath();
                  ctx.arc(nx, ny, 18, 0, Math.PI * 2);
                  ctx.fillStyle = lit ? nodeColors[l] + '30' : viz.colors.bg;
                  ctx.fill();
                  ctx.strokeStyle = lit ? nodeColors[l] : viz.colors.text + '44';
                  ctx.lineWidth = 2;
                  ctx.stroke();

                  // Value
                  ctx.fillStyle = lit ? viz.colors.white : viz.colors.text + '66';
                  ctx.font = 'bold 11px monospace';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillText(lit ? val.toFixed(2) : '?', nx, ny);

                  // Pre-activation annotation
                  if (l > 0 && lit && preVals[l]) {
                    ctx.fillStyle = viz.colors.text;
                    ctx.font = '9px -apple-system,sans-serif';
                    ctx.fillText('z=' + preVals[l][i].toFixed(2), nx, ny + 28);
                  }
                }
              }

              // Layer labels
              ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              for (let l = 0; l < layers.length; l++) {
                ctx.fillStyle = nodeColors[l];
                ctx.fillText(layerLabels[l], layerX[l], 30);
                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px -apple-system,sans-serif';
                const dimLabel = l === 0 ? 'n=' + layers[l] : 'n=' + layers[l];
                ctx.fillText(dimLabel, layerX[l], 46);
                ctx.font = 'bold 12px -apple-system,sans-serif';
              }

              // Title
              ctx.fillStyle = viz.colors.white;
              ctx.font = 'bold 11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Forward pass: data flows left \u2192 right', 20, H - 16);

              // Loss
              const yHat = fp.z3[0];
              const loss = 0.5 * (yHat - 1.0) * (yHat - 1.0);
              if (animPhase >= 3) {
                ctx.fillStyle = viz.colors.red;
                ctx.font = '13px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText('y\u0302 = ' + yHat.toFixed(3) + ',  L = \u00BD(y\u0302 - y)\u00B2 = ' + loss.toFixed(4) + '  (target y = 1.0)', layerX[3] + 30, H / 2);
              }
            }

            viz.animate(draw);
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'A network has input dimension 4, hidden layers of size 8 and 6, and output dimension 3. What are the dimensions of \\(\\mathbf{W}^{(1)}, \\mathbf{W}^{(2)}, \\mathbf{W}^{(3)}\\)?',
          hint: 'Layer \\(l\\) maps from \\(n_{l-1}\\) to \\(n_l\\) dimensions, so \\(\\mathbf{W}^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}\\).',
          solution: '\\(\\mathbf{W}^{(1)} \\in \\mathbb{R}^{8 \\times 4}\\), \\(\\mathbf{W}^{(2)} \\in \\mathbb{R}^{6 \\times 8}\\), \\(\\mathbf{W}^{(3)} \\in \\mathbb{R}^{3 \\times 6}\\). Total parameters in weights: \\(32 + 48 + 18 = 98\\), plus biases: \\(8 + 6 + 3 = 17\\), giving 115 total parameters.'
        },
        {
          question: 'During the forward pass of a network with \\(L\\) layers and hidden dimension \\(n\\), how much memory is required to cache the intermediate values needed for backpropagation?',
          hint: 'At each layer, we cache \\(\\mathbf{z}^{(l)}\\) (for \\(\\sigma\'\\)) and \\(\\mathbf{a}^{(l-1)}\\) (for the weight gradient). Count the total number of cached scalars.',
          solution: 'We cache \\(\\mathbf{z}^{(l)} \\in \\mathbb{R}^{n_l}\\) and \\(\\mathbf{a}^{(l-1)} \\in \\mathbb{R}^{n_{l-1}}\\) for each layer \\(l = 1, \\ldots, L\\). If all hidden layers have size \\(n\\), this is \\(O(Ln)\\) scalars. For a minibatch of size \\(B\\), total memory is \\(O(BLn)\\), which grows linearly in both batch size and depth. This is the dominant memory cost during training.'
        },
        {
          question: 'Suppose we use \\(\\sigma(z) = \\tanh(z)\\) as the activation. Show that \\(\\sigma\'(z)\\) can be computed cheaply from \\(\\sigma(z)\\) itself, without re-evaluating the forward pass.',
          hint: 'Use the identity \\(\\tanh\'(z) = 1 - \\tanh^2(z)\\). What does that mean for caching?',
          solution: 'Since \\(\\tanh\'(z) = 1 - \\tanh^2(z) = 1 - a^2\\) where \\(a = \\tanh(z)\\), we can compute the derivative directly from the cached activation \\(a^{(l)}\\), without needing \\(z^{(l)}\\) at all. This reduces caching requirements: we only need \\(\\{\\mathbf{a}^{(l)}\\}\\), not \\(\\{\\mathbf{z}^{(l)}\\}\\). The same trick works for sigmoid: \\(\\sigma\'(z) = \\sigma(z)(1 - \\sigma(z)) = a(1 - a)\\).'
        }
      ]
    },

    // ─── SECTION 3: Backward Pass ───────────────────────────────────────────
    {
      id: 'ch05-sec03',
      title: '3. Backward Pass',
      content: `
<h2>Backward Pass</h2>

<p><strong>Section roadmap.</strong>
The backward pass is the heart of backpropagation.
We derive the recursive gradient computation from the output layer back to the input,
showing how each layer computes its gradient using only local information and the gradient
signal from the layer above.
</p>

<h3>The Core Idea</h3>

<p>
After the forward pass computes \\(\\mathcal{L}\\), we want to find \\(\\nabla_{\\mathbf{W}^{(l)}} \\mathcal{L}\\) and
\\(\\nabla_{\\mathbf{b}^{(l)}} \\mathcal{L}\\) for every layer \\(l\\). The chain rule gives us a recursive formula.
Define the <em>error signal</em> (or "delta") at layer \\(l\\):
</p>
\\[
  \\boldsymbol{\\delta}^{(l)} \\triangleq \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{(l)}} \\in \\mathbb{R}^{n_l}.
\\]

<div class="env-block theorem">
<div class="env-title">Theorem 5.2 (Backpropagation Equations)</div>
<div class="env-body">
<p>For an \\(L\\)-layer network with loss \\(\\mathcal{L}\\):</p>
<p><strong>(1) Output layer delta:</strong></p>
\\[
  \\boldsymbol{\\delta}^{(L)} = \\nabla_{\\mathbf{a}^{(L)}} \\mathcal{L} \\odot \\sigma'^{(L)}(\\mathbf{z}^{(L)})
\\]
<p><strong>(2) Recursive delta (backward recurrence):</strong></p>
\\[
  \\boldsymbol{\\delta}^{(l)} = \\bigl(\\mathbf{W}^{(l+1)\\top} \\boldsymbol{\\delta}^{(l+1)}\\bigr) \\odot \\sigma'^{(l)}(\\mathbf{z}^{(l)}), \\quad l = L-1, \\ldots, 1
\\]
<p><strong>(3) Parameter gradients:</strong></p>
\\[
  \\nabla_{\\mathbf{W}^{(l)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(l)} \\mathbf{a}^{(l-1)\\top}, \\quad
  \\nabla_{\\mathbf{b}^{(l)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(l)}
\\]
</div>
</div>

<div class="env-block proof">
<div class="env-title">Derivation</div>
<div class="env-body">
<p>
For the parameter gradient, apply the chain rule to \\(\\mathcal{L}\\) through \\(\\mathbf{z}^{(l)}\\):
\\[
  \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}_{ij}} = \\frac{\\partial \\mathcal{L}}{\\partial z^{(l)}_i} \\cdot \\frac{\\partial z^{(l)}_i}{\\partial W^{(l)}_{ij}} = \\delta^{(l)}_i \\cdot a^{(l-1)}_j,
\\]
since \\(z^{(l)}_i = \\sum_j W^{(l)}_{ij} a^{(l-1)}_j + b^{(l)}_i\\). In matrix form: \\(\\nabla_{\\mathbf{W}} \\mathcal{L} = \\boldsymbol{\\delta} \\mathbf{a}^\\top\\).
</p>
<p>
For the backward recurrence, apply the chain rule through \\(\\mathbf{z}^{(l+1)}\\):
\\[
  \\delta^{(l)}_j = \\frac{\\partial \\mathcal{L}}{\\partial z^{(l)}_j} = \\sum_i \\frac{\\partial \\mathcal{L}}{\\partial z^{(l+1)}_i} \\cdot \\frac{\\partial z^{(l+1)}_i}{\\partial a^{(l)}_j} \\cdot \\sigma'^{(l)}(z^{(l)}_j) = \\left(\\sum_i \\delta^{(l+1)}_i W^{(l+1)}_{ij}\\right) \\sigma'^{(l)}(z^{(l)}_j).
\\]
In matrix form: \\(\\boldsymbol{\\delta}^{(l)} = (\\mathbf{W}^{(l+1)\\top} \\boldsymbol{\\delta}^{(l+1)}) \\odot \\sigma'(\\mathbf{z}^{(l)})\\).
</p>
<div class="qed">&#8718;</div>
</div>
</div>

<div class="env-block definition">
<div class="env-title">Algorithm 5.2 (Backward Pass)</div>
<div class="env-body">
<p><strong>Input:</strong> Cached intermediates \\(\\{\\mathbf{z}^{(l)}, \\mathbf{a}^{(l)}\\}\\) from the forward pass.</p>
<ol>
  <li>Compute output error: \\(\\boldsymbol{\\delta}^{(L)} = \\nabla_{\\mathbf{a}^{(L)}} \\mathcal{L} \\odot \\sigma'^{(L)}(\\mathbf{z}^{(L)})\\).</li>
  <li>For \\(l = L, L-1, \\ldots, 1\\):
    <ul>
      <li>Compute \\(\\nabla_{\\mathbf{W}^{(l)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(l)} \\mathbf{a}^{(l-1)\\top}\\).</li>
      <li>Compute \\(\\nabla_{\\mathbf{b}^{(l)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(l)}\\).</li>
      <li>If \\(l &gt; 1\\): propagate \\(\\boldsymbol{\\delta}^{(l-1)} = (\\mathbf{W}^{(l)\\top} \\boldsymbol{\\delta}^{(l)}) \\odot \\sigma'^{(l-1)}(\\mathbf{z}^{(l-1)})\\).</li>
    </ul>
  </li>
</ol>
<p><strong>Output:</strong> Gradients \\(\\{\\nabla_{\\mathbf{W}^{(l)}}, \\nabla_{\\mathbf{b}^{(l)}}\\}_{l=1}^L\\).</p>
</div>
</div>

<div class="env-block intuition">
<div class="env-title">Intuition: Error Signal as "Blame"</div>
<div class="env-body">
<p>
The delta \\(\\boldsymbol{\\delta}^{(l)}\\) tells each neuron in layer \\(l\\) how much "blame" it bears for the final loss.
The backward recurrence distributes blame proportionally through the weights:
a neuron connected to the loss through large weights receives more blame.
The activation derivative \\(\\sigma'(\\mathbf{z}^{(l)})\\) acts as a gate: if the neuron is saturated
(\\(\\sigma' \\approx 0\\)), it receives almost no blame, regardless of the upstream signal.
</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-backward-pass"></div>
`,
      visualizations: [
        {
          id: 'viz-backward-pass',
          title: 'Animated Backward Pass: Gradient Flow Right to Left',
          description: 'Gradients propagate from the loss backward through the network. Each node displays its delta value \\(\\delta^{(l)}_i\\). Observe how the error signal is distributed by the weights and gated by the activation derivative.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 750, height: 440, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;

            let targetY = 1.0;
            VizEngine.createSlider(controls, 'target y', -2, 2, targetY, 0.1, v => { targetY = v; });

            const W1 = [[0.5, -0.3], [0.8, 0.4], [-0.2, 0.7]];
            const b1 = [0.1, -0.1, 0.2];
            const W2 = [[0.6, -0.4, 0.3], [-0.5, 0.7, 0.2]];
            const b2 = [0.1, -0.2];
            const W3 = [[0.8, -0.6]];
            const b3 = [0.1];

            function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
            function sigmoidDeriv(z) { const s = sigmoid(z); return s * (1 - s); }

            function fullPass(input) {
              const a0 = input;
              const z1 = W1.map((row, i) => row.reduce((s, w, j) => s + w * a0[j], 0) + b1[i]);
              const a1 = z1.map(sigmoid);
              const z2 = W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + b2[i]);
              const a2 = z2.map(sigmoid);
              const z3 = W3.map((row, i) => row.reduce((s, w, j) => s + w * a2[j], 0) + b3[i]);
              const yHat = z3[0]; // Linear output
              const loss = 0.5 * (yHat - targetY) * (yHat - targetY);

              // Backward
              const dLdyHat = yHat - targetY;
              const delta3 = [dLdyHat]; // linear output, sigma' = 1

              // delta2 = (W3^T delta3) .* sigma'(z2)
              const delta2 = a2.map((_, j) => {
                let s = 0;
                for (let i = 0; i < delta3.length; i++) s += W3[i][j] * delta3[i];
                return s * sigmoidDeriv(z2[j]);
              });

              // delta1 = (W2^T delta2) .* sigma'(z1)
              const delta1 = a1.map((_, j) => {
                let s = 0;
                for (let i = 0; i < delta2.length; i++) s += W2[i][j] * delta2[i];
                return s * sigmoidDeriv(z1[j]);
              });

              return { a0, z1, a1, z2, a2, z3, yHat, loss, delta3, delta2, delta1 };
            }

            const layers = [2, 3, 2, 1];
            const layerX = [100, 250, 410, 570];
            function nodeY(layerIdx, nodeIdx) {
              const n = layers[layerIdx];
              const totalH = (n - 1) * 70;
              return H / 2 - totalH / 2 + nodeIdx * 70;
            }

            let animPhase = 0;

            function draw(t) {
              animPhase = (t * 0.0006) % 4;
              const backPhase = animPhase; // 0 to 4
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              const fp = fullPass([0.8, -0.5]);
              const allVals = [fp.a0, fp.a1, fp.a2, [fp.yHat]];
              const deltas = [fp.delta1, fp.delta2, fp.delta3];

              // Draw connections with gradient flow
              for (let l = 0; l < layers.length - 1; l++) {
                const backL = 2 - l; // backward index: layer 3->2 is backL=0, 2->1 is backL=1, 1->0 is backL=2
                const litBack = backL < Math.floor(backPhase);
                const pulsingBack = backL >= Math.floor(backPhase) && backL < Math.floor(backPhase) + 1;
                const progress = backPhase - Math.floor(backPhase);

                for (let i = 0; i < layers[l]; i++) {
                  for (let j = 0; j < layers[l + 1]; j++) {
                    const x1c = layerX[l] + 18, y1c = nodeY(l, i);
                    const x2c = layerX[l + 1] - 18, y2c = nodeY(l + 1, j);

                    // Forward connections (dim)
                    ctx.strokeStyle = viz.colors.text;
                    ctx.globalAlpha = 0.12;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x1c, y1c);
                    ctx.lineTo(x2c, y2c);
                    ctx.stroke();

                    // Backward gradient pulse
                    if (pulsingBack && progress > 0) {
                      const px = x2c + (x1c - x2c) * progress;
                      const py = y2c + (y1c - y2c) * progress;
                      ctx.globalAlpha = 0.7;
                      ctx.fillStyle = viz.colors.red;
                      ctx.beginPath();
                      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
                      ctx.fill();
                    }
                    if (litBack) {
                      ctx.strokeStyle = viz.colors.red;
                      ctx.globalAlpha = 0.25;
                      ctx.lineWidth = 1.5;
                      ctx.beginPath();
                      ctx.moveTo(x1c, y1c);
                      ctx.lineTo(x2c, y2c);
                      ctx.stroke();
                    }
                  }
                }
              }
              ctx.globalAlpha = 1;

              // Draw nodes
              const fwdColors = [viz.colors.blue, viz.colors.teal, viz.colors.purple, viz.colors.orange];
              for (let l = 0; l < layers.length; l++) {
                for (let i = 0; i < layers[l]; i++) {
                  const nx = layerX[l], ny = nodeY(l, i);
                  const val = allVals[l][i];

                  ctx.beginPath();
                  ctx.arc(nx, ny, 18, 0, Math.PI * 2);
                  ctx.fillStyle = fwdColors[l] + '20';
                  ctx.fill();
                  ctx.strokeStyle = fwdColors[l];
                  ctx.lineWidth = 2;
                  ctx.stroke();

                  // Forward value
                  ctx.fillStyle = viz.colors.white;
                  ctx.font = '10px monospace';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillText(val.toFixed(2), nx, ny - 4);

                  // Delta value (for layers 1,2,3 -- indices 0,1,2 in deltas)
                  if (l > 0) {
                    const backL = 3 - l; // which backward step reveals this
                    const show = backL < Math.floor(backPhase) || (backL < backPhase);
                    if (show) {
                      const deltaArr = l === 3 ? fp.delta3 : (l === 2 ? fp.delta2 : fp.delta1);
                      const d = deltaArr[i];
                      ctx.fillStyle = viz.colors.red;
                      ctx.font = 'bold 9px monospace';
                      ctx.fillText('\u03B4=' + d.toFixed(3), nx, ny + 10);
                    }
                  }
                }
              }

              // Layer labels
              const layerLabels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
              ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              for (let l = 0; l < layers.length; l++) {
                ctx.fillStyle = fwdColors[l];
                ctx.fillText(layerLabels[l], layerX[l], 30);
              }

              // Loss display
              ctx.fillStyle = viz.colors.red;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('L = \u00BD(y\u0302 - y)\u00B2 = ' + fp.loss.toFixed(4), layerX[3] + 30, H / 2 - 15);
              ctx.fillText('\u2202L/\u2202y\u0302 = ' + (fp.yHat - targetY).toFixed(3), layerX[3] + 30, H / 2 + 5);

              // Legend
              ctx.fillStyle = viz.colors.text;
              ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Backward pass: gradients flow right \u2192 left', 20, H - 16);
              ctx.fillStyle = viz.colors.red;
              ctx.fillText('\u03B4 = error signal at each neuron', 20, H - 34);
            }

            viz.animate(draw);
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Consider a single-hidden-layer network with sigmoid activations and MSE loss \\(\\mathcal{L} = \\frac{1}{2}(\\hat{y} - y)^2\\). Write out the complete backpropagation update for \\(\\mathbf{W}^{(1)}\\) in terms of \\(\\mathbf{x}\\), \\(\\mathbf{W}^{(2)}\\), \\(\\mathbf{a}^{(1)}\\), \\(\\mathbf{z}^{(1)}\\), \\(\\hat{y}\\), and \\(y\\).',
          hint: 'Start with \\(\\delta^{(2)} = \\hat{y} - y\\) (since the output is linear). Then backpropagate: \\(\\boldsymbol{\\delta}^{(1)} = (\\mathbf{W}^{(2)\\top} \\delta^{(2)}) \\odot \\sigma\'(\\mathbf{z}^{(1)})\\).',
          solution: '\\(\\delta^{(2)} = \\hat{y} - y\\). Then \\(\\boldsymbol{\\delta}^{(1)} = (\\mathbf{W}^{(2)\\top}(\\hat{y} - y)) \\odot \\mathbf{a}^{(1)} \\odot (\\mathbf{1} - \\mathbf{a}^{(1)})\\), using \\(\\sigma\'(z) = \\sigma(z)(1-\\sigma(z)) = a(1-a)\\). The weight gradient is \\(\\nabla_{\\mathbf{W}^{(1)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(1)} \\mathbf{x}^\\top\\).'
        },
        {
          question: 'Explain why computing \\(\\nabla_{\\mathbf{b}^{(l)}} \\mathcal{L} = \\boldsymbol{\\delta}^{(l)}\\) follows immediately from the definition, with no additional matrix multiplication.',
          hint: 'Look at how \\(\\mathbf{b}^{(l)}\\) enters the computation: \\(\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}\\). What is \\(\\frac{\\partial z^{(l)}_i}{\\partial b^{(l)}_j}\\)?',
          solution: 'Since \\(z^{(l)}_i = \\sum_j W^{(l)}_{ij} a^{(l-1)}_j + b^{(l)}_i\\), we have \\(\\frac{\\partial z^{(l)}_i}{\\partial b^{(l)}_j} = \\delta_{ij}\\) (the Kronecker delta, i.e., the identity matrix). Therefore \\(\\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}_i} = \\sum_j \\delta^{(l)}_j \\cdot \\delta_{ji} = \\delta^{(l)}_i\\). The Jacobian of \\(\\mathbf{z}\\) with respect to \\(\\mathbf{b}\\) is simply the identity, so the gradient passes through unchanged.'
        },
        {
          question: 'Suppose we add a skip connection so that \\(\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)} + \\mathbf{a}^{(l-2)}\\) (assuming dimensions match). How does the backward pass change?',
          hint: 'The skip connection means \\(\\mathbf{a}^{(l-2)}\\) now has two "children" in the computation graph. Use the multivariate chain rule: when a variable influences the loss through multiple paths, the gradients add.',
          solution: 'With the skip connection, \\(\\mathbf{a}^{(l-2)}\\) affects the loss through both layer \\(l-1\\) (the normal path) and layer \\(l\\) (the skip path). By the chain rule, the gradient \\(\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{a}^{(l-2)}}\\) now has an additional term: \\(\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{a}^{(l-2)}} = \\mathbf{W}^{(l-1)\\top} \\boldsymbol{\\delta}^{(l-1)} + \\boldsymbol{\\delta}^{(l)}\\). The gradient from the skip connection \\(\\boldsymbol{\\delta}^{(l)}\\) flows directly back without passing through \\(\\mathbf{W}^{(l-1)}\\) or \\(\\sigma\'\\). This is precisely why ResNets mitigate vanishing gradients: the identity shortcut provides an unimpeded gradient highway.'
        }
      ]
    },

    // ─── SECTION 4: Vanishing & Exploding Gradients ─────────────────────────
    {
      id: 'ch05-sec04',
      title: '4. Vanishing & Exploding Gradients',
      content: `
<h2>Vanishing & Exploding Gradients</h2>

<p><strong>Section roadmap.</strong>
The backward recurrence \\(\\boldsymbol{\\delta}^{(l)} = (\\mathbf{W}^{(l+1)\\top} \\boldsymbol{\\delta}^{(l+1)}) \\odot \\sigma'(\\mathbf{z}^{(l)})\\)
reveals a potential pathology: the gradient at layer \\(l\\) depends on the product of many Jacobians.
If these products shrink or grow exponentially with depth, training becomes impossible.
This section analyzes the problem and previews the solutions.
</p>

<h3>The Product of Jacobians</h3>

<p>
The gradient of the loss with respect to the pre-activations at layer \\(l\\) involves a product
of \\(L - l\\) Jacobians:
</p>
\\[
  \\boldsymbol{\\delta}^{(l)} = \\left(\\prod_{k=l+1}^{L} \\mathrm{diag}(\\sigma'(\\mathbf{z}^{(k)})) \\, \\mathbf{W}^{(k)\\top}\\right) \\boldsymbol{\\delta}^{(L)}.
\\]
<p>
Each factor \\(\\mathrm{diag}(\\sigma'(\\mathbf{z}^{(k)})) \\, \\mathbf{W}^{(k)\\top}\\) has a spectral norm.
If most factors have norm \\(&lt; 1\\), the product vanishes exponentially.
If most have norm \\(&gt; 1\\), the product explodes.
</p>

<h3>Vanishing Gradients with Sigmoid/Tanh</h3>

<div class="env-block theorem">
<div class="env-title">Proposition 5.1 (Sigmoid Gradient Bound)</div>
<div class="env-body">
<p>
For the sigmoid function \\(\\sigma(z) = \\frac{1}{1+e^{-z}}\\), we have \\(\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))\\).
The maximum of \\(\\sigma'\\) is \\(\\frac{1}{4}\\), attained at \\(z = 0\\).
Therefore \\(0 &lt; \\sigma'(z) \\le \\frac{1}{4}\\) for all \\(z\\).
</p>
</div>
</div>

<p>
The factor \\(\\mathrm{diag}(\\sigma'(\\mathbf{z}^{(k)}))\\) has all diagonal entries at most \\(\\frac{1}{4}\\).
Even if \\(\\|\\mathbf{W}^{(k)}\\| = 1\\), the product of \\(L - l\\) such factors is bounded by \\((\\frac{1}{4})^{L-l}\\).
For a 20-layer network, this gives a factor of \\(4^{-20} \\approx 10^{-12}\\).
Gradients in the early layers become astronomically small, making learning effectively impossible.
</p>

<div class="env-block example">
<div class="env-title">Example 5.2 (Gradient Decay)</div>
<div class="env-body">
<p>
With sigmoid activations and unit-norm weights, the gradient magnitude at layer \\(l\\) relative to the output is bounded by:
</p>
\\[
  \\frac{\\|\\boldsymbol{\\delta}^{(l)}\\|}{\\|\\boldsymbol{\\delta}^{(L)}\\|} \\le \\left(\\frac{1}{4}\\right)^{L-l}.
\\]
<table style="width:100%;border-collapse:collapse;margin:1rem 0;">
<thead><tr style="background:#1a1a40;">
  <th style="padding:6px;border:1px solid #30363d;color:#58a6ff;">Depth \\(L - l\\)</th>
  <th style="padding:6px;border:1px solid #30363d;color:#3fb9a0;">Gradient ratio (sigmoid)</th>
  <th style="padding:6px;border:1px solid #30363d;color:#f0883e;">Gradient ratio (ReLU, ideal)</th>
</tr></thead>
<tbody>
  <tr><td style="padding:6px;border:1px solid #30363d;">5</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 10^{-3}\\)</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 1\\)</td></tr>
  <tr><td style="padding:6px;border:1px solid #30363d;">10</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 10^{-6}\\)</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 1\\)</td></tr>
  <tr><td style="padding:6px;border:1px solid #30363d;">20</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 10^{-12}\\)</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 1\\)</td></tr>
  <tr><td style="padding:6px;border:1px solid #30363d;">50</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 10^{-30}\\)</td><td style="padding:6px;border:1px solid #30363d;">\\(\\approx 1\\)</td></tr>
</tbody>
</table>
</div>
</div>

<h3>ReLU: A Partial Fix</h3>

<p>
The ReLU activation \\(\\sigma(z) = \\max(0, z)\\) has derivative:
</p>
\\[
  \\sigma'(z) = \\begin{cases} 1 & \\text{if } z &gt; 0, \\\\ 0 & \\text{if } z &lt; 0. \\end{cases}
\\]
<p>
When a ReLU neuron is active (\\(z &gt; 0\\)), it passes the gradient through with no attenuation.
This eliminates the \\(\\frac{1}{4}\\) shrinkage factor of sigmoid.
However, dead neurons (\\(z &lt; 0\\)) pass zero gradient, creating a different kind of vanishing problem.
</p>

<h3>Exploding Gradients</h3>

<p>
If \\(\\|\\mathbf{W}^{(l)}\\| &gt; 1/\\max \\sigma'\\), the Jacobian product can grow exponentially.
Exploding gradients cause numerical overflow and erratic parameter updates.
The standard remedy is <em>gradient clipping</em>: rescale the gradient vector whenever
\\(\\|\\nabla \\mathcal{L}\\| &gt; \\tau\\) for some threshold \\(\\tau\\).
</p>

<div class="env-block remark">
<div class="env-title">Remark: The Landscape of Solutions</div>
<div class="env-body">
<p>
The vanishing/exploding gradient problem motivated several major innovations in deep learning:
</p>
<ul>
  <li><strong>Better activations</strong>: ReLU (Chapter 4), LeakyReLU, GELU, Swish</li>
  <li><strong>Better initialization</strong>: Xavier/He initialization (Chapter 6)</li>
  <li><strong>Normalization</strong>: BatchNorm, LayerNorm (Chapter 6)</li>
  <li><strong>Architectural solutions</strong>: Skip connections / ResNet (Chapter 11), LSTM gates (Chapter 13)</li>
  <li><strong>Gradient clipping</strong>: for exploding gradients, especially in RNNs (Chapter 12)</li>
</ul>
<p>
Each of these attacks the Jacobian product from a different angle: controlling \\(\\sigma'\\),
controlling \\(\\|\\mathbf{W}\\|\\), or bypassing the product altogether.
</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-vanishing-gradients"></div>
`,
      visualizations: [
        {
          id: 'viz-vanishing-gradients',
          title: 'Gradient Magnitude Across Layers: Sigmoid vs. ReLU',
          description: 'Compare gradient magnitudes at each layer for different activation functions. Adjust the network depth with the slider to see how deep networks exacerbate the vanishing gradient problem.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 720, height: 400, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;

            let depth = 10;
            let weightScale = 1.0;
            VizEngine.createSlider(controls, 'Depth', 3, 50, depth, 1, v => { depth = Math.round(v); });
            VizEngine.createSlider(controls, '||W||', 0.5, 2.0, weightScale, 0.1, v => { weightScale = v; });

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              const pad = { left: 80, right: 40, top: 50, bottom: 60 };
              const plotW = W - pad.left - pad.right;
              const plotH = H - pad.top - pad.bottom;

              // Compute gradient magnitudes (log scale)
              const sigmoidGrads = [];
              const reluGrads = [];
              const tanhGrads = [];

              // Sigmoid: max sigma' = 0.25
              // tanh: max sigma' = 1.0
              // ReLU: sigma' = 1 for active neurons (assume ~50% active)
              for (let l = 0; l < depth; l++) {
                const layersBack = depth - 1 - l;
                const sigmoidFactor = weightScale * 0.25; // |W| * max(sigma')
                const tanhFactor = weightScale * 1.0;     // |W| * max(sigma')
                const reluFactor = weightScale * 0.5;     // |W| * (prob active)

                sigmoidGrads.push(Math.pow(sigmoidFactor, layersBack));
                tanhGrads.push(Math.pow(tanhFactor, layersBack));
                reluGrads.push(Math.pow(reluFactor, layersBack));
              }

              // Determine y-axis range (log scale)
              const allVals = [...sigmoidGrads, ...reluGrads, ...tanhGrads].filter(v => v > 0);
              let logMin = Math.floor(Math.log10(Math.min(...allVals))) - 1;
              let logMax = Math.ceil(Math.log10(Math.max(...allVals))) + 1;
              logMin = Math.max(logMin, -35);
              logMax = Math.min(logMax, 10);
              if (logMax - logMin < 4) logMax = logMin + 4;

              // Draw grid
              ctx.strokeStyle = viz.colors.grid;
              ctx.lineWidth = 0.5;
              for (let v = logMin; v <= logMax; v++) {
                const y = pad.top + plotH - (v - logMin) / (logMax - logMin) * plotH;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(pad.left + plotW, y);
                ctx.stroke();

                ctx.fillStyle = viz.colors.text;
                ctx.font = '10px monospace';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText('10^' + v, pad.left - 8, y);
              }

              // Reference line at 1.0
              const yOne = pad.top + plotH - (0 - logMin) / (logMax - logMin) * plotH;
              if (yOne >= pad.top && yOne <= pad.top + plotH) {
                ctx.strokeStyle = viz.colors.yellow + '44';
                ctx.lineWidth = 1;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(pad.left, yOne);
                ctx.lineTo(pad.left + plotW, yOne);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = viz.colors.yellow;
                ctx.textAlign = 'left';
                ctx.font = '10px -apple-system,sans-serif';
                ctx.fillText('gradient = 1 (ideal)', pad.left + plotW + 4, yOne);
              }

              // Draw curves
              function drawCurve(grads, color, label, labelIdx) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                let started = false;
                for (let l = 0; l < depth; l++) {
                  const x = pad.left + (l / (depth - 1)) * plotW;
                  const logV = grads[l] > 0 ? Math.log10(grads[l]) : logMin;
                  const clampedLog = Math.max(logMin, Math.min(logMax, logV));
                  const y = pad.top + plotH - (clampedLog - logMin) / (logMax - logMin) * plotH;
                  if (!started) { ctx.moveTo(x, y); started = true; }
                  else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // Label
                ctx.fillStyle = color;
                ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(label, pad.left + 10, pad.top + 16 + labelIdx * 18);
              }

              drawCurve(sigmoidGrads, viz.colors.orange, 'Sigmoid (max \u03C3\' = 0.25)', 0);
              drawCurve(tanhGrads, viz.colors.purple, 'Tanh (max \u03C3\' = 1.0)', 1);
              drawCurve(reluGrads, viz.colors.green, 'ReLU (~50% active)', 2);

              // X-axis label
              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Layer index (left = early, right = output)', pad.left + plotW / 2, H - 12);

              // Y-axis label
              ctx.save();
              ctx.translate(18, pad.top + plotH / 2);
              ctx.rotate(-Math.PI / 2);
              ctx.fillStyle = viz.colors.text;
              ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Gradient magnitude (log scale)', 0, 0);
              ctx.restore();

              // Layer ticks
              ctx.font = '9px monospace';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'top';
              ctx.fillStyle = viz.colors.text;
              const tickStep = depth <= 20 ? 1 : Math.ceil(depth / 20);
              for (let l = 0; l < depth; l += tickStep) {
                const x = pad.left + (l / (depth - 1)) * plotW;
                ctx.fillText(l + 1, x, pad.top + plotH + 6);
              }

              // Annotations
              if (depth >= 10) {
                const sigGrad1 = sigmoidGrads[0];
                if (sigGrad1 > 0 && sigGrad1 < 1e-3) {
                  ctx.fillStyle = viz.colors.red;
                  ctx.font = '11px -apple-system,sans-serif';
                  ctx.textAlign = 'right';
                  ctx.fillText('Layer 1 sigmoid gradient: ' + sigGrad1.toExponential(1), W - pad.right, H - 12);
                }
              }
            }

            viz.animate(draw);
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'For a network of depth \\(L\\) with sigmoid activations and weights initialized so that \\(\\|\\mathbf{W}^{(l)}\\| = c\\) for all \\(l\\), find the value of \\(c\\) such that the expected gradient magnitude neither grows nor shrinks.',
          hint: 'We need the spectral norm of each Jacobian factor to equal 1. Each factor is \\(\\mathrm{diag}(\\sigma\'(\\mathbf{z})) \\mathbf{W}^\\top\\). At \\(z = 0\\), \\(\\sigma\'(0) = 1/4\\).',
          solution: 'For the Jacobian product to be stable, we need \\(\\|\\mathrm{diag}(\\sigma\') \\mathbf{W}^\\top\\| \\approx 1\\). Since \\(\\max \\sigma\' = 1/4\\), this requires \\(\\frac{1}{4} \\cdot c \\approx 1\\), giving \\(c = 4\\). However, this is only marginally stable at the peak of \\(\\sigma\'\\); for most inputs, \\(\\sigma\' &lt; 1/4\\), so even \\(c = 4\\) will eventually lead to vanishing gradients. This is why sigmoid is fundamentally problematic for deep networks.'
        },
        {
          question: 'Explain the "dying ReLU" problem. If a neuron has \\(z^{(l)}_i &lt; 0\\) for all training examples, what happens to its gradient, and can it ever recover?',
          hint: 'If \\(z &lt; 0\\), then \\(\\sigma\'(z) = 0\\) for ReLU. What does this mean for \\(\\delta_i^{(l)}\\) and for \\(\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}_{ij}}\\)?',
          solution: 'If \\(z^{(l)}_i &lt; 0\\) for all inputs, then \\(\\sigma\'(z^{(l)}_i) = 0\\), so \\(\\delta^{(l)}_i = 0\\) regardless of the upstream gradient. This means \\(\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}_{ij}} = \\delta^{(l)}_i \\cdot a^{(l-1)}_j = 0\\) for all \\(j\\). The weights feeding into this neuron receive zero gradient, so they never update. The neuron is permanently "dead," it can never recover because the gradient is exactly zero. Leaky ReLU (\\(\\sigma(z) = \\max(\\alpha z, z)\\) with \\(\\alpha &gt; 0\\)) fixes this by ensuring a small nonzero gradient even for \\(z &lt; 0\\).'
        },
        {
          question: 'Gradient clipping rescales the gradient whenever \\(\\|\\nabla \\mathcal{L}\\| &gt; \\tau\\). Write the clipped gradient \\(\\tilde{\\mathbf{g}}\\) as a formula and explain why it preserves gradient direction.',
          hint: 'The clipped gradient should point in the same direction as the original gradient but have its norm capped at \\(\\tau\\).',
          solution: 'The clipped gradient is \\(\\tilde{\\mathbf{g}} = \\begin{cases} \\mathbf{g} & \\text{if } \\|\\mathbf{g}\\| \\le \\tau, \\\\ \\frac{\\tau}{\\|\\mathbf{g}\\|} \\mathbf{g} & \\text{if } \\|\\mathbf{g}\\| &gt; \\tau. \\end{cases}\\) This is equivalent to \\(\\tilde{\\mathbf{g}} = \\frac{\\mathbf{g}}{\\max(1, \\|\\mathbf{g}\\|/\\tau)}\\). Direction is preserved because we scale by a positive scalar \\(\\tau/\\|\\mathbf{g}\\|\\), which does not change the sign of any component. The norm is capped at \\(\\tau\\), preventing any single update from being catastrophically large.'
        }
      ]
    },

    // ─── SECTION 5: Computational Efficiency ────────────────────────────────
    {
      id: 'ch05-sec05',
      title: '5. Computational Efficiency',
      content: `
<h2>Computational Efficiency</h2>

<p><strong>Section roadmap.</strong>
Backpropagation is not just mathematically elegant; it is computationally essential.
This section analyzes the cost of forward and backward passes, compares forward-mode and
reverse-mode automatic differentiation, and discusses the memory-compute tradeoff.
</p>

<h3>Counting Operations</h3>

<p>
Consider a single fully connected layer mapping \\(\\mathbb{R}^n \\to \\mathbb{R}^m\\).
</p>

<div class="env-block definition">
<div class="env-title">Operation Counts Per Layer</div>
<div class="env-body">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#1a1a40;">
  <th style="padding:8px;border:1px solid #30363d;color:#58a6ff;">Operation</th>
  <th style="padding:8px;border:1px solid #30363d;color:#3fb9a0;">Cost (multiply-adds)</th>
</tr></thead>
<tbody>
  <tr><td style="padding:8px;border:1px solid #30363d;">Forward: \\(\\mathbf{z} = \\mathbf{W}\\mathbf{a} + \\mathbf{b}\\)</td><td style="padding:8px;border:1px solid #30363d;">\\(mn\\)</td></tr>
  <tr><td style="padding:8px;border:1px solid #30363d;">Forward: \\(\\mathbf{a} = \\sigma(\\mathbf{z})\\)</td><td style="padding:8px;border:1px solid #30363d;">\\(O(m)\\)</td></tr>
  <tr><td style="padding:8px;border:1px solid #30363d;">Backward: \\(\\boldsymbol{\\delta} = (\\mathbf{W}^\\top \\boldsymbol{\\delta}_{\\text{next}}) \\odot \\sigma'(\\mathbf{z})\\)</td><td style="padding:8px;border:1px solid #30363d;">\\(mn + m\\)</td></tr>
  <tr><td style="padding:8px;border:1px solid #30363d;">Backward: \\(\\nabla_\\mathbf{W} = \\boldsymbol{\\delta}\\mathbf{a}^\\top\\)</td><td style="padding:8px;border:1px solid #30363d;">\\(mn\\)</td></tr>
</tbody>
</table>
</div>
</div>

<p>
The forward pass costs \\(\\sim mn\\) per layer. The backward pass costs \\(\\sim 2mn\\) per layer
(one matrix-vector product for propagating \\(\\boldsymbol{\\delta}\\), one outer product for the weight gradient).
Therefore, <strong>the backward pass costs roughly twice the forward pass</strong>.
This is a fundamental constant, not an implementation artifact.
</p>

<div class="env-block theorem">
<div class="env-title">Theorem 5.3 (Backpropagation Complexity)</div>
<div class="env-body">
<p>
For a network with \\(L\\) layers and \\(P\\) total parameters, the cost of computing
<em>all</em> gradients via backpropagation is \\(O(P)\\), the same order as a single forward pass.
</p>
<p>
In contrast, computing each partial derivative independently by finite differences
(perturbing one weight at a time) costs \\(O(P^2)\\), since each of the \\(P\\) perturbations
requires a full forward pass costing \\(O(P)\\).
</p>
</div>
</div>

<h3>Forward-Mode vs. Reverse-Mode AD</h3>

<p>
Backpropagation is an instance of <em>reverse-mode automatic differentiation</em> (AD).
The alternative, <em>forward-mode AD</em>, computes directional derivatives in the forward direction.
The two modes differ dramatically in efficiency depending on the shape of the function.
</p>

<div class="env-block definition">
<div class="env-title">Definition 5.3 (Forward-Mode vs. Reverse-Mode AD)</div>
<div class="env-body">
<p>Consider a function \\(f: \\mathbb{R}^n \\to \\mathbb{R}^m\\) computed as a composition of \\(K\\) operations.</p>
<ul>
  <li><strong>Forward-mode AD</strong> computes one column of the Jacobian per pass. To get the full Jacobian, we need \\(n\\) passes. Cost: \\(O(nK)\\).</li>
  <li><strong>Reverse-mode AD</strong> computes one row of the Jacobian per pass. To get the full Jacobian, we need \\(m\\) passes. Cost: \\(O(mK)\\).</li>
</ul>
</div>
</div>

<p>
In deep learning, the loss is a scalar (\\(m = 1\\)), so reverse-mode needs only <em>one</em> backward pass
to compute the gradient with respect to all \\(n\\) inputs (or parameters).
Forward-mode would need \\(n\\) passes, one for each parameter.
Since \\(n\\) can be in the millions, reverse-mode (backpropagation) is the clear winner.
</p>

<div class="env-block intuition">
<div class="env-title">Intuition: "Many-to-one" vs. "One-to-many"</div>
<div class="env-body">
<p>
Think of the computation graph as a tree. Reverse-mode starts at the root (scalar loss) and
propagates down to all leaves (parameters) in one sweep, like a broadcast. Forward-mode starts
at one leaf and traces its influence up to the root, which must be repeated for each leaf.
When there are millions of leaves and one root, the broadcast approach wins overwhelmingly.
</p>
</div>
</div>

<h3>Memory-Compute Tradeoff</h3>

<p>
The efficiency of backpropagation comes at a memory cost: we must store all intermediate
activations from the forward pass. For a network with \\(L\\) layers of width \\(n\\) processing
a minibatch of size \\(B\\):
</p>
\\[
  \\text{Activation memory} = O(BLn).
\\]
<p>
This can be substantial for deep networks with large batch sizes. Several techniques reduce this cost:
</p>
<ul>
  <li><strong>Gradient checkpointing</strong> (Chen et al., 2016): store activations only at every \\(\\sqrt{L}\\) layers; recompute the rest during the backward pass. Reduces memory from \\(O(L)\\) to \\(O(\\sqrt{L})\\) at the cost of one extra forward pass.</li>
  <li><strong>Mixed-precision training</strong>: store activations in FP16 instead of FP32, halving memory.</li>
  <li><strong>Activation recomputation</strong>: discard and recompute specific expensive-to-store but cheap-to-compute activations (e.g., layer normalization).</li>
</ul>

<div class="env-block remark">
<div class="env-title">Remark: Why Not Symbolic Differentiation?</div>
<div class="env-body">
<p>
Symbolic differentiation (like in a CAS) computes exact derivative expressions.
But for neural networks, the expression for the derivative can be exponentially
larger than the function itself (expression swell). Automatic differentiation
avoids this by computing numerical values of derivatives alongside the function
evaluation, never forming the symbolic expression.
</p>
</div>
</div>

<div class="viz-placeholder" data-viz="viz-ad-comparison"></div>
`,
      visualizations: [
        {
          id: 'viz-ad-comparison',
          title: 'Forward-Mode vs. Reverse-Mode AD: Operation Count Comparison',
          description: 'Compare the number of operations needed by forward-mode and reverse-mode AD as you vary the number of input parameters and output dimensions. Neural networks have millions of inputs but a single scalar output, making reverse-mode overwhelmingly more efficient.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 720, height: 420, scale: 1, originX: 0, originY: 0 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;

            let numInputs = 1000;
            let numOutputs = 1;
            let opCost = 100; // K: operations in computational graph

            VizEngine.createSlider(controls, 'Inputs (n)', 1, 6, 3, 0.1, v => {
              numInputs = Math.round(Math.pow(10, v));
            });
            VizEngine.createSlider(controls, 'Outputs (m)', 1, 4, 0, 0.1, v => {
              numOutputs = Math.max(1, Math.round(Math.pow(10, v)));
            });

            function draw() {
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, W, H);

              const pad = { left: 60, right: 40, top: 50, bottom: 55 };
              const plotW = W - pad.left - pad.right;
              const plotH = H - pad.top - pad.bottom;

              const n = numInputs;
              const m = numOutputs;
              const K = opCost;

              // Forward-mode: n passes, each costs O(K) -> total n*K
              // Reverse-mode: m passes, each costs O(K) -> total m*K
              // Finite differences: n perturbations, each costs O(K) -> total n*K (similar to forward but with approximation error)
              const fwdCost = n * K;
              const revCost = m * K;
              const fdCost = n * K;

              // Bar chart
              const barData = [
                { label: 'Reverse-Mode AD\n(Backprop)', cost: revCost, color: viz.colors.green },
                { label: 'Forward-Mode AD', cost: fwdCost, color: viz.colors.orange },
                { label: 'Finite Differences', cost: fdCost, color: viz.colors.red }
              ];

              const maxCost = Math.max(...barData.map(d => d.cost));
              const barW = plotW / barData.length - 40;
              const barGap = 40;

              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillStyle = viz.colors.white;
              ctx.fillText('Cost of Computing Full Gradient (\u2207L)', W / 2, 24);

              ctx.font = '11px -apple-system,sans-serif';
              ctx.fillStyle = viz.colors.text;
              ctx.fillText('n = ' + n.toLocaleString() + ' parameters,  m = ' + m + ' output(s),  K = ' + K + ' ops per pass', W / 2, 42);

              barData.forEach((d, i) => {
                const bx = pad.left + i * (barW + barGap) + barGap / 2;
                const logMax = Math.log10(maxCost);
                const logVal = d.cost > 0 ? Math.log10(d.cost) : 0;
                const bh = (logVal / Math.max(logMax, 1)) * plotH;
                const by = pad.top + plotH - bh;

                // Bar
                ctx.fillStyle = d.color + '33';
                ctx.fillRect(bx, by, barW, bh);
                ctx.strokeStyle = d.color;
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, barW, bh);

                // Cost value
                ctx.fillStyle = d.color;
                ctx.font = 'bold 13px monospace';
                ctx.textAlign = 'center';
                if (d.cost >= 1e6) {
                  ctx.fillText(d.cost.toExponential(1), bx + barW / 2, by - 22);
                } else {
                  ctx.fillText(d.cost.toLocaleString(), bx + barW / 2, by - 22);
                }
                ctx.font = '11px monospace';
                ctx.fillStyle = viz.colors.text;
                const formula = i === 0 ? 'm\u00B7K = ' + m + '\u00B7' + K : 'n\u00B7K = ' + n.toLocaleString() + '\u00B7' + K;
                ctx.fillText(formula, bx + barW / 2, by - 8);

                // Label
                ctx.fillStyle = d.color;
                ctx.font = 'bold 11px -apple-system,sans-serif';
                const lines = d.label.split('\n');
                lines.forEach((line, li) => {
                  ctx.fillText(line, bx + barW / 2, pad.top + plotH + 16 + li * 14);
                });
              });

              // Speedup annotation
              if (fwdCost > 0 && revCost > 0) {
                const speedup = fwdCost / revCost;
                ctx.fillStyle = viz.colors.green;
                ctx.font = 'bold 14px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Reverse-mode speedup: ' + speedup.toLocaleString() + '\u00D7', W / 2, H - 8);
              }

              // Draw baseline
              ctx.strokeStyle = viz.colors.axis;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(pad.left, pad.top + plotH);
              ctx.lineTo(pad.left + plotW, pad.top + plotH);
              ctx.stroke();
            }

            viz.animate(draw);
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'A ResNet-50 has approximately 25 million parameters. Compare the cost of computing the full gradient using (a) backpropagation and (b) finite differences.',
          hint: 'Backpropagation needs one forward pass plus one backward pass (roughly 3 forward passes total). Finite differences need one perturbation per parameter, each requiring a full forward pass.',
          solution: 'Let \\(P = 25 \\times 10^6\\) and let \\(C\\) be the cost of one forward pass. (a) Backpropagation: \\(\\sim 3C\\) (one forward + one backward \\(\\approx 2C\\)). (b) Finite differences: \\(\\sim P \\cdot C = 25 \\times 10^6 C\\). The ratio is \\(25 \\times 10^6 / 3 \\approx 8.3 \\times 10^6\\). Backpropagation is about 8 million times faster. If one forward pass takes 10ms, backprop computes the gradient in 30ms, while finite differences would take about 70 hours.'
        },
        {
          question: 'Gradient checkpointing stores activations at every \\(\\sqrt{L}\\) layers and recomputes the rest. Explain why the memory cost is \\(O(\\sqrt{L})\\) and the compute overhead is at most one additional forward pass.',
          hint: 'Divide the \\(L\\) layers into \\(\\sqrt{L}\\) segments of \\(\\sqrt{L}\\) layers each. Store the boundary activations (\\(\\sqrt{L}\\) of them). During the backward pass for segment \\(k\\), recompute the \\(\\sqrt{L}\\) intermediate activations from the stored boundary.',
          solution: 'We store \\(\\sqrt{L}\\) boundary activations (memory \\(O(\\sqrt{L} \\cdot n)\\)). During the backward pass, when we reach segment \\(k\\), we recompute its \\(\\sqrt{L}\\) internal activations from boundary \\(k\\) in \\(O(\\sqrt{L})\\) operations, use them for backprop through that segment, then discard them. Each of the \\(\\sqrt{L}\\) segments requires recomputing \\(\\sqrt{L}\\) layers, totaling \\(\\sqrt{L} \\times \\sqrt{L} = L\\) recomputations, equivalent to one extra forward pass. So we pay \\(\\sim 4C\\) compute (1 forward + 1 recompute + 2 backward) instead of \\(3C\\), a 33% overhead, but reduce activation memory from \\(O(L)\\) to \\(O(\\sqrt{L})\\).'
        },
        {
          question: 'A function \\(f: \\mathbb{R}^{1000} \\to \\mathbb{R}^{1000}\\) is computed as a sequence of 50 operations. How many AD passes does forward-mode need vs. reverse-mode to compute the full \\(1000 \\times 1000\\) Jacobian?',
          hint: 'Forward-mode computes one column of the Jacobian per pass. Reverse-mode computes one row per pass.',
          solution: 'The full Jacobian is \\(1000 \\times 1000\\). Forward-mode needs \\(n = 1000\\) passes (one per input dimension, computing one column each). Reverse-mode needs \\(m = 1000\\) passes (one per output dimension, computing one row each). In this case, \\(n = m = 1000\\), so both modes require the same number of passes: 1000. Neither mode has an advantage when \\(n = m\\). In general, use forward-mode when \\(n \\ll m\\) and reverse-mode when \\(m \\ll n\\). Neural network training (\\(m = 1\\), \\(n = P \\gg 1\\)) is the extreme case favoring reverse-mode.'
        }
      ]
    }
  ]
});
