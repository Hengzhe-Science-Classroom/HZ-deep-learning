window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch03',
  number: 3,
  title: 'Perceptron & Linear Models',
  subtitle: 'From Biological Neurons to the Foundations of Neural Networks',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: The Perceptron Model
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch03-sec01',
      title: 'The Perceptron Model',
      content: `
<div class="env-block intuition"><div class="env-title">From Biology to Computation</div><div class="env-body"><p>The story of deep learning begins with a question: can we build a mathematical model that captures how a biological neuron processes information? A biological neuron collects electrical signals through its <strong>dendrites</strong>, integrates them in the <strong>cell body</strong>, and fires an output signal along its <strong>axon</strong> if the total stimulation exceeds a threshold. In 1943, Warren McCulloch and Walter Pitts proposed a radical simplification of this process into a purely mathematical object, the <em>McCulloch-Pitts neuron</em>, laying the groundwork for everything that follows in this course.</p></div></div>

<h2>The McCulloch-Pitts Neuron</h2>

<p>The McCulloch-Pitts model reduces a biological neuron to three essential operations:</p>
<ol>
  <li><strong>Weighted aggregation.</strong> Each input \\(x_i\\) is multiplied by a <em>weight</em> \\(w_i\\) that controls how strongly that input influences the neuron.</li>
  <li><strong>Bias shift.</strong> A <em>bias</em> term \\(b\\) shifts the activation threshold up or down.</li>
  <li><strong>Nonlinear activation.</strong> The weighted sum plus bias is passed through an <em>activation function</em> to produce the output.</li>
</ol>

<div class="env-block definition"><div class="env-title">Definition 3.1 (Perceptron)</div><div class="env-body">
<p>A <strong>perceptron</strong> is a function \\(f : \\mathbb{R}^d \\to \\{0, 1\\}\\) defined by</p>
\\[f(\\mathbf{x}) = \\sigma\\!\\left(\\sum_{i=1}^{d} w_i x_i + b\\right) = \\sigma(\\mathbf{w}^\\top \\mathbf{x} + b),\\]
<p>where \\(\\mathbf{w} = (w_1, \\dots, w_d)^\\top \\in \\mathbb{R}^d\\) is the <strong>weight vector</strong>, \\(b \\in \\mathbb{R}\\) is the <strong>bias</strong>, and \\(\\sigma\\) is the <strong>Heaviside step function</strong>:</p>
\\[\\sigma(z) = \\begin{cases} 1 & \\text{if } z \\geq 0, \\\\ 0 & \\text{if } z &lt; 0. \\end{cases}\\]
</div></div>

<h3>Geometric Interpretation</h3>

<p>The equation \\(\\mathbf{w}^\\top \\mathbf{x} + b = 0\\) defines a <strong>hyperplane</strong> in \\(\\mathbb{R}^d\\). In two dimensions, this is simply a line. The perceptron classifies points on one side of this hyperplane as class 1 and points on the other side as class 0. The weight vector \\(\\mathbf{w}\\) is <em>normal</em> (perpendicular) to the decision boundary, and the bias \\(b\\) controls the offset of the boundary from the origin.</p>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>Changing the weights rotates the decision boundary; changing the bias translates it. The interactive visualization below lets you explore this directly.</p></div></div>

<div class="viz-placeholder" data-viz="viz-perceptron-interactive"></div>

<h3>The Bias as an Extra Weight</h3>

<p>A common notational convenience is to absorb the bias into the weight vector. Define \\(\\tilde{\\mathbf{x}} = (1, x_1, \\dots, x_d)^\\top\\) and \\(\\tilde{\\mathbf{w}} = (b, w_1, \\dots, w_d)^\\top\\). Then</p>
\\[\\mathbf{w}^\\top \\mathbf{x} + b = \\tilde{\\mathbf{w}}^\\top \\tilde{\\mathbf{x}},\\]
<p>so the bias is equivalent to a weight on a constant input of 1. This "augmented" representation simplifies many derivations and implementations.</p>

<div class="env-block example"><div class="env-title">Example 3.1 (A 2D Perceptron)</div><div class="env-body">
<p>Consider \\(\\mathbf{w} = (1, 1)^\\top\\) and \\(b = -1.5\\). The decision boundary is \\(x_1 + x_2 - 1.5 = 0\\), or equivalently \\(x_2 = -x_1 + 1.5\\). A point \\((1, 1)\\) gives \\(1 + 1 - 1.5 = 0.5 \\geq 0\\), so it is classified as 1. A point \\((0, 0)\\) gives \\(0 + 0 - 1.5 = -1.5 &lt; 0\\), so it is classified as 0.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Historical Note</div><div class="env-body"><p>Frank Rosenblatt introduced the perceptron in 1958 at the Cornell Aeronautical Laboratory and demonstrated it on the Mark I Perceptron machine, a hardware implementation that could learn to classify simple visual patterns. It generated enormous excitement, but also set the stage for a dramatic backlash when its limitations were exposed.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-perceptron-interactive',
          title: 'Interactive Perceptron: Decision Boundary in 2D',
          description: 'Drag the sliders to change weights and bias. Watch how the decision boundary (green line) moves. The weight vector (white arrow) is always perpendicular to the boundary. Blue dots are class 1, red dots are class 0.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 260 });

            let w1 = 1.0, w2 = 1.0, bias = -1.5;

            // Data points for demonstration
            const dataPoints = [
              { x: 0, y: 0, label: 0 },
              { x: 1, y: 0, label: 0 },
              { x: 0, y: 1, label: 0 },
              { x: 1, y: 1, label: 1 },
              { x: 2, y: 1, label: 1 },
              { x: 1, y: 2, label: 1 },
              { x: 2, y: 2, label: 1 },
              { x: 0.5, y: 0.3, label: 0 },
              { x: 1.5, y: 1.8, label: 1 },
              { x: 0.2, y: 1.5, label: 0 },
              { x: 2.5, y: 0.5, label: 1 },
              { x: 1.8, y: 0.2, label: 0 },
              { x: 0.8, y: 2.2, label: 1 },
              { x: 2.2, y: 1.5, label: 1 },
            ];

            VizEngine.createSlider(controls, 'w\u2081', -3, 3, w1, 0.1, v => { w1 = v; draw(); });
            VizEngine.createSlider(controls, 'w\u2082', -3, 3, w2, 0.1, v => { w2 = v; draw(); });
            VizEngine.createSlider(controls, 'b', -5, 5, bias, 0.1, v => { bias = v; draw(); });

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();

              const ctx = viz.ctx;

              // Shade the positive region
              const norm = Math.sqrt(w1 * w1 + w2 * w2);
              if (norm > 0.01) {
                // Draw decision boundary as a line: w1*x + w2*y + b = 0
                // Two points on the line
                let lx1, ly1, lx2, ly2;
                if (Math.abs(w2) > Math.abs(w1)) {
                  lx1 = -6; ly1 = -(w1 * lx1 + bias) / w2;
                  lx2 = 6; ly2 = -(w1 * lx2 + bias) / w2;
                } else {
                  ly1 = -6; lx1 = -(w2 * ly1 + bias) / w1;
                  ly2 = 6; lx2 = -(w2 * ly2 + bias) / w1;
                }

                // Shade positive half-plane
                const [sx1, sy1] = viz.toScreen(lx1, ly1);
                const [sx2, sy2] = viz.toScreen(lx2, ly2);
                // Direction of the normal (positive side)
                const nx = w1 / norm, ny = w2 / norm;
                // Offset the shading region by pushing two corners in the normal direction
                const push = 800; // pixels
                const [snx, sny] = [nx * push, -ny * push]; // screen coords: y is inverted
                ctx.fillStyle = 'rgba(88, 166, 255, 0.06)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1);
                ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 + snx, sy2 + sny);
                ctx.lineTo(sx1 + snx, sy1 + sny);
                ctx.closePath();
                ctx.fill();

                // Shade negative half-plane
                ctx.fillStyle = 'rgba(248, 81, 73, 0.06)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1);
                ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 - snx, sy2 - sny);
                ctx.lineTo(sx1 - snx, sy1 - sny);
                ctx.closePath();
                ctx.fill();

                // Decision boundary
                viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green, 2.5);

                // Weight vector (normal to boundary) drawn from a point on the line
                const px = -bias * w1 / (w1 * w1 + w2 * w2);
                const py = -bias * w2 / (w1 * w1 + w2 * w2);
                const arrowScale = 1.2 / norm;
                viz.drawVector(px, py, px + w1 * arrowScale, py + w2 * arrowScale, viz.colors.white, 'w', 2);
              }

              // Draw data points and classify them
              for (const p of dataPoints) {
                const z = w1 * p.x + w2 * p.y + bias;
                const predicted = z >= 0 ? 1 : 0;
                const correct = predicted === p.label;
                const color = p.label === 1 ? viz.colors.blue : viz.colors.red;
                // Draw point
                const [sx, sy] = viz.toScreen(p.x, p.y);
                ctx.beginPath();
                ctx.arc(sx, sy, 6, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                // Draw X mark if misclassified
                if (!correct) {
                  ctx.strokeStyle = viz.colors.yellow;
                  ctx.lineWidth = 2;
                  ctx.beginPath();
                  ctx.moveTo(sx - 9, sy - 9); ctx.lineTo(sx + 9, sy + 9);
                  ctx.moveTo(sx + 9, sy - 9); ctx.lineTo(sx - 9, sy + 9);
                  ctx.stroke();
                }
              }

              // Labels
              viz.screenText('x\u2081', viz.width - 15, viz.originY - 10, viz.colors.text, 13, 'right');
              viz.screenText('x\u2082', viz.originX + 10, 12, viz.colors.text, 13, 'left');

              // Info text
              const accuracy = dataPoints.filter(p => (w1 * p.x + w2 * p.y + bias >= 0 ? 1 : 0) === p.label).length;
              viz.screenText('Boundary: ' + w1.toFixed(1) + 'x\u2081 + ' + w2.toFixed(1) + 'x\u2082 + (' + bias.toFixed(1) + ') = 0', viz.width / 2, viz.height - 30, viz.colors.text, 12);
              viz.screenText('Correctly classified: ' + accuracy + '/' + dataPoints.length + (accuracy < dataPoints.length ? '  (\u2716 = misclassified)' : '  \u2714 All correct!'), viz.width / 2, viz.height - 12, accuracy === dataPoints.length ? viz.colors.green : viz.colors.yellow, 12);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'A perceptron has weights \\(w_1 = 2\\), \\(w_2 = -1\\) and bias \\(b = 1\\). Compute the output for \\(\\mathbf{x} = (3, 5)^\\top\\).',
          hint: 'Compute the pre-activation \\(z = w_1 x_1 + w_2 x_2 + b\\), then apply the step function.',
          solution: 'We compute \\(z = 2 \\cdot 3 + (-1) \\cdot 5 + 1 = 6 - 5 + 1 = 2\\). Since \\(z = 2 \\geq 0\\), the step function outputs \\(\\sigma(z) = 1\\).'
        },
        {
          question: 'Write the equation of the decision boundary for the perceptron in the previous exercise. What is the slope and intercept of this line?',
          hint: 'The decision boundary is where \\(w_1 x_1 + w_2 x_2 + b = 0\\). Solve for \\(x_2\\) in terms of \\(x_1\\).',
          solution: 'Setting \\(2x_1 - x_2 + 1 = 0\\) and solving for \\(x_2\\): \\(x_2 = 2x_1 + 1\\). The slope is 2 and the \\(x_2\\)-intercept is 1.'
        },
        {
          question: 'Explain geometrically why the weight vector \\(\\mathbf{w}\\) is perpendicular to the decision boundary \\(\\mathbf{w}^\\top \\mathbf{x} + b = 0\\).',
          hint: 'Take two points \\(\\mathbf{x}_1, \\mathbf{x}_2\\) on the boundary and consider \\(\\mathbf{w}^\\top(\\mathbf{x}_1 - \\mathbf{x}_2)\\).',
          solution: 'If both \\(\\mathbf{x}_1\\) and \\(\\mathbf{x}_2\\) lie on the boundary, then \\(\\mathbf{w}^\\top \\mathbf{x}_1 + b = 0\\) and \\(\\mathbf{w}^\\top \\mathbf{x}_2 + b = 0\\). Subtracting: \\(\\mathbf{w}^\\top (\\mathbf{x}_1 - \\mathbf{x}_2) = 0\\). Since \\(\\mathbf{x}_1 - \\mathbf{x}_2\\) is an arbitrary vector along the boundary, \\(\\mathbf{w}\\) is perpendicular to every such direction, hence normal to the boundary.'
        },
        {
          question: 'Show that the perpendicular distance from the origin to the decision boundary \\(\\mathbf{w}^\\top \\mathbf{x} + b = 0\\) is \\(|b| / \\|\\mathbf{w}\\|\\).',
          hint: 'The closest point on the hyperplane to the origin is \\(\\mathbf{x}^* = -b\\mathbf{w}/\\|\\mathbf{w}\\|^2\\). Verify it lies on the boundary and compute its norm.',
          solution: 'The point on the hyperplane closest to the origin is the projection: \\(\\mathbf{x}^* = -\\frac{b}{\\|\\mathbf{w}\\|^2}\\mathbf{w}\\). Check: \\(\\mathbf{w}^\\top \\mathbf{x}^* + b = -\\frac{b}{\\|\\mathbf{w}\\|^2}\\|\\mathbf{w}\\|^2 + b = -b + b = 0\\). The distance is \\(\\|\\mathbf{x}^*\\| = \\frac{|b|}{\\|\\mathbf{w}\\|^2}\\|\\mathbf{w}\\| = \\frac{|b|}{\\|\\mathbf{w}\\|}\\).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Linear Separability
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch03-sec02',
      title: 'Linear Separability',
      content: `
<h2>Linear Separability</h2>

<p>A single perceptron partitions input space with a hyperplane. This immediately raises a fundamental question: which classification problems can a single perceptron solve?</p>

<div class="env-block definition"><div class="env-title">Definition 3.2 (Linear Separability)</div><div class="env-body">
<p>A dataset \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^{n}\\) with \\(y_i \\in \\{0, 1\\}\\) is <strong>linearly separable</strong> if there exist \\(\\mathbf{w} \\in \\mathbb{R}^d\\) and \\(b \\in \\mathbb{R}\\) such that</p>
\\[y_i = 1 \\implies \\mathbf{w}^\\top \\mathbf{x}_i + b &gt; 0 \\quad \\text{and} \\quad y_i = 0 \\implies \\mathbf{w}^\\top \\mathbf{x}_i + b &lt; 0\\]
<p>for all \\(i = 1, \\dots, n\\).</p>
</div></div>

<h3>Boolean Functions as Perceptrons</h3>

<p>The simplest test cases for a perceptron are <strong>Boolean functions</strong> on two binary inputs \\(x_1, x_2 \\in \\{0, 1\\}\\). There are exactly four input patterns: \\((0,0), (0,1), (1,0), (1,1)\\). Can a single perceptron compute AND, OR, and XOR?</p>

<div class="env-block example"><div class="env-title">Example 3.2 (AND Gate)</div><div class="env-body">
<p>The AND function outputs 1 only when both inputs are 1. We need \\(\\mathbf{w}^\\top \\mathbf{x} + b \\geq 0\\) only at \\((1,1)\\). Setting \\(w_1 = w_2 = 1\\) and \\(b = -1.5\\):</p>
<ul>
  <li>\\((0,0) \\to 0 + 0 - 1.5 = -1.5 &lt; 0\\) \\(\\Rightarrow 0\\) \\(\\checkmark\\)</li>
  <li>\\((0,1) \\to 0 + 1 - 1.5 = -0.5 &lt; 0\\) \\(\\Rightarrow 0\\) \\(\\checkmark\\)</li>
  <li>\\((1,0) \\to 1 + 0 - 1.5 = -0.5 &lt; 0\\) \\(\\Rightarrow 0\\) \\(\\checkmark\\)</li>
  <li>\\((1,1) \\to 1 + 1 - 1.5 = 0.5 \\geq 0\\) \\(\\Rightarrow 1\\) \\(\\checkmark\\)</li>
</ul>
</div></div>

<div class="env-block example"><div class="env-title">Example 3.3 (OR Gate)</div><div class="env-body">
<p>The OR function outputs 1 when at least one input is 1. Setting \\(w_1 = w_2 = 1\\) and \\(b = -0.5\\):</p>
<ul>
  <li>\\((0,0) \\to -0.5 &lt; 0\\) \\(\\Rightarrow 0\\) \\(\\checkmark\\)</li>
  <li>\\((0,1) \\to 0.5 \\geq 0\\) \\(\\Rightarrow 1\\) \\(\\checkmark\\)</li>
  <li>\\((1,0) \\to 0.5 \\geq 0\\) \\(\\Rightarrow 1\\) \\(\\checkmark\\)</li>
  <li>\\((1,1) \\to 1.5 \\geq 0\\) \\(\\Rightarrow 1\\) \\(\\checkmark\\)</li>
</ul>
</div></div>

<h3>The XOR Problem</h3>

<div class="env-block theorem"><div class="env-title">Theorem 3.1 (XOR Is Not Linearly Separable)</div><div class="env-body">
<p>The XOR function \\(f(x_1, x_2) = x_1 \\oplus x_2\\) cannot be computed by any single perceptron. That is, there exist no \\(w_1, w_2, b \\in \\mathbb{R}\\) such that \\(\\text{sign}(w_1 x_1 + w_2 x_2 + b)\\) equals the XOR truth table.</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Suppose for contradiction that such \\(w_1, w_2, b\\) exist. Then:</p>
<ul>
  <li>From \\((0,0) \\to 0\\): \\(b &lt; 0\\)</li>
  <li>From \\((0,1) \\to 1\\): \\(w_2 + b \\geq 0\\), so \\(w_2 \\geq -b &gt; 0\\)</li>
  <li>From \\((1,0) \\to 1\\): \\(w_1 + b \\geq 0\\), so \\(w_1 \\geq -b &gt; 0\\)</li>
  <li>From \\((1,1) \\to 0\\): \\(w_1 + w_2 + b &lt; 0\\)</li>
</ul>
<p>But \\(w_1 + w_2 + b \\geq (-b) + (-b) + b = -b &gt; 0\\), contradicting the fourth condition.</p>
<div class="qed">&#8718;</div>
</div></div>

<p>This result, popularized by Minsky and Papert in their 1969 book <em>Perceptrons</em>, showed that a single-layer perceptron has fundamental limitations. The interactive visualization below lets you try to find a line that separates the XOR points (spoiler: you cannot).</p>

<div class="viz-placeholder" data-viz="viz-linear-separability"></div>

<div class="env-block intuition"><div class="env-title">Why XOR Fails Geometrically</div><div class="env-body"><p>In the XOR problem, the class-1 points \\((0,1)\\) and \\((1,0)\\) are on <em>opposite corners</em> of the unit square, while the class-0 points \\((0,0)\\) and \\((1,1)\\) occupy the other two corners. No single straight line can separate opposite corners of a square. You would need at least two lines (and a way to combine them), which is exactly what a two-layer network provides.</p></div></div>

<div class="env-block remark"><div class="env-title">Counting Boolean Functions</div><div class="env-body"><p>Of the \\(2^{2^2} = 16\\) Boolean functions on two inputs, exactly 14 are linearly separable. Only XOR and XNOR are not. However, as the number of inputs grows, the fraction of linearly separable Boolean functions shrinks exponentially.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-linear-separability',
          title: 'Linear Separability: AND, OR, and XOR',
          description: 'Select a Boolean gate. For AND and OR, the green decision boundary correctly separates the classes. For XOR, try dragging the sliders to find a separating line. You will find it is impossible.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 140, originX: 100, originY: 340 });

            let currentGate = 'AND';
            let w1 = 1, w2 = 1, bias = -1.5;

            const gates = {
              AND:  { targets: [0, 0, 0, 1], w1: 1, w2: 1, b: -1.5 },
              OR:   { targets: [0, 1, 1, 1], w1: 1, w2: 1, b: -0.5 },
              XOR:  { targets: [0, 1, 1, 0], w1: 1, w2: 1, b: -0.5 }
            };
            const corners = [[0, 0], [0, 1], [1, 0], [1, 1]];

            const btnRow = document.createElement('div');
            btnRow.style.cssText = 'display:flex;gap:6px;';
            ['AND', 'OR', 'XOR'].forEach(g => {
              const b = VizEngine.createButton(btnRow, g, () => {
                currentGate = g;
                const gate = gates[g];
                w1 = gate.w1; w2 = gate.w2; bias = gate.b;
                slW1.value = w1; slW2.value = w2; slB.value = bias;
                slW1.previousElementSibling && (slW1.nextElementSibling.textContent = w1.toFixed(1));
                slW2.nextElementSibling && (slW2.nextElementSibling.textContent = w2.toFixed(1));
                slB.nextElementSibling && (slB.nextElementSibling.textContent = bias.toFixed(1));
                draw();
              });
              if (g === 'AND') b.style.borderColor = '#58a6ff';
            });
            controls.appendChild(btnRow);

            const slW1 = VizEngine.createSlider(controls, 'w\u2081', -3, 3, w1, 0.1, v => { w1 = v; draw(); });
            const slW2 = VizEngine.createSlider(controls, 'w\u2082', -3, 3, w2, 0.1, v => { w2 = v; draw(); });
            const slB = VizEngine.createSlider(controls, 'b', -3, 3, bias, 0.1, v => { bias = v; draw(); });

            function draw() {
              viz.clear();
              const ctx = viz.ctx;

              // Draw faint grid lines
              ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
              for (let i = 0; i <= 1; i++) {
                const [sx] = viz.toScreen(i, 0);
                ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, viz.height); ctx.stroke();
                const [, sy] = viz.toScreen(0, i);
                ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(viz.width, sy); ctx.stroke();
              }

              // Axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
              ctx.beginPath(); ctx.moveTo(viz.toScreen(-0.2, 0)[0], viz.toScreen(0, 0)[1]); ctx.lineTo(viz.toScreen(1.4, 0)[0], viz.toScreen(0, 0)[1]); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(viz.toScreen(0, -0.2)[0], viz.toScreen(0, -0.2)[1]); ctx.lineTo(viz.toScreen(0, 1.4)[0], viz.toScreen(0, 1.4)[1]); ctx.stroke();

              // Axis labels
              viz.screenText('x\u2081', viz.toScreen(1.35, 0)[0], viz.toScreen(1.35, 0)[1] - 15, viz.colors.text, 13);
              viz.screenText('x\u2082', viz.toScreen(0, 1.35)[0] + 15, viz.toScreen(0, 1.35)[1], viz.colors.text, 13);

              // Tick labels
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText('0', viz.toScreen(0, 0)[0], viz.toScreen(0, 0)[1] + 5);
              ctx.fillText('1', viz.toScreen(1, 0)[0], viz.toScreen(1, 0)[1] + 5);
              ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText('1', viz.toScreen(0, 1)[0] - 8, viz.toScreen(0, 1)[1]);

              const targets = gates[currentGate].targets;
              const norm = Math.sqrt(w1 * w1 + w2 * w2);

              // Decision boundary
              if (norm > 0.01) {
                let lx1, ly1, lx2, ly2;
                if (Math.abs(w2) > Math.abs(w1)) {
                  lx1 = -0.5; ly1 = -(w1 * lx1 + bias) / w2;
                  lx2 = 1.5; ly2 = -(w1 * lx2 + bias) / w2;
                } else {
                  ly1 = -0.5; lx1 = -(w2 * ly1 + bias) / w1;
                  ly2 = 1.5; lx2 = -(w2 * ly2 + bias) / w1;
                }

                // Shade regions
                const [sx1, sy1] = viz.toScreen(lx1, ly1);
                const [sx2, sy2] = viz.toScreen(lx2, ly2);
                const nx = w1 / norm, ny = w2 / norm;
                const push = 600;
                const snx = nx * push, sny = -ny * push;

                ctx.fillStyle = 'rgba(88, 166, 255, 0.08)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 + snx, sy2 + sny); ctx.lineTo(sx1 + snx, sy1 + sny);
                ctx.closePath(); ctx.fill();

                ctx.fillStyle = 'rgba(248, 81, 73, 0.08)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 - snx, sy2 - sny); ctx.lineTo(sx1 - snx, sy1 - sny);
                ctx.closePath(); ctx.fill();

                viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green, 2.5);
              }

              // Data points
              let correct = 0;
              corners.forEach(([x, y], i) => {
                const t = targets[i];
                const z = w1 * x + w2 * y + bias;
                const pred = z >= 0 ? 1 : 0;
                if (pred === t) correct++;

                const color = t === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(x, y);

                ctx.beginPath();
                ctx.arc(sx, sy, 10, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = pred === t ? viz.colors.green : viz.colors.yellow;
                ctx.lineWidth = pred === t ? 1.5 : 2.5;
                ctx.stroke();

                // Label inside
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(t.toString(), sx, sy);
              });

              // Gate title and status
              viz.screenText(currentGate + ' Gate', viz.width / 2, 20, viz.colors.white, 18);

              const allCorrect = correct === 4;
              const statusColor = allCorrect ? viz.colors.green : viz.colors.red;
              const statusText = allCorrect ? 'Linearly separable! All 4 points correct.' : 'Correct: ' + correct + '/4' + (currentGate === 'XOR' ? '  (impossible to get 4/4!)' : '');
              viz.screenText(statusText, viz.width / 2, viz.height - 15, statusColor, 12);

              // Truth table
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              const tx = viz.width - 130, ty = 15;
              ctx.fillStyle = viz.colors.white; ctx.font = 'bold 11px -apple-system,sans-serif';
              ctx.fillText('Truth Table:', tx, ty);
              ctx.font = '11px monospace'; ctx.fillStyle = viz.colors.text;
              corners.forEach(([x, y], i) => {
                ctx.fillText(x + ' ' + currentGate + ' ' + y + ' = ' + targets[i], tx, ty + 16 + i * 15);
              });
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Find weights \\(w_1, w_2\\) and bias \\(b\\) that implement the NAND gate (NOT AND) as a perceptron. Verify your answer on all four input patterns.',
          hint: 'NAND outputs 0 only when both inputs are 1, and 1 otherwise. Try \\(w_1 = w_2 = -1\\) and adjust \\(b\\).',
          solution: 'Set \\(w_1 = w_2 = -1\\) and \\(b = 1.5\\). Verify: \\((0,0) \\to 1.5 \\geq 0 \\Rightarrow 1\\) \\(\\checkmark\\); \\((0,1) \\to 0.5 \\geq 0 \\Rightarrow 1\\) \\(\\checkmark\\); \\((1,0) \\to 0.5 \\geq 0 \\Rightarrow 1\\) \\(\\checkmark\\); \\((1,1) \\to -0.5 &lt; 0 \\Rightarrow 0\\) \\(\\checkmark\\). Note: the NAND gate is significant because it is <em>universal</em>; any Boolean function can be built from NAND gates alone.'
        },
        {
          question: 'Prove that linear separability of a dataset \\(\\{(\\mathbf{x}_i, y_i)\\}\\) is equivalent to the existence of a solution to a system of strict linear inequalities.',
          hint: 'Rewrite the separability conditions as \\((2y_i - 1)(\\mathbf{w}^\\top \\mathbf{x}_i + b) &gt; 0\\) for all \\(i\\).',
          solution: 'Define \\(s_i = 2y_i - 1 \\in \\{-1, +1\\}\\). The dataset is linearly separable if and only if there exist \\(\\mathbf{w}, b\\) satisfying \\(s_i(\\mathbf{w}^\\top \\mathbf{x}_i + b) &gt; 0\\) for all \\(i\\). This is a system of \\(n\\) strict linear inequalities in \\(d+1\\) unknowns. By rescaling \\((\\mathbf{w}, b)\\) by a positive constant, we can replace strict inequalities with \\(s_i(\\mathbf{w}^\\top \\mathbf{x}_i + b) \\geq 1\\), a standard linear feasibility problem solvable by linear programming.'
        },
        {
          question: 'Consider three points in \\(\\mathbb{R}^2\\): \\((0,0) \\to 0\\), \\((1,0) \\to 1\\), \\((0,1) \\to 1\\). Is this dataset linearly separable? If so, find a separating line. How many degrees of freedom remain in the choice of line?',
          hint: 'You need \\(b &lt; 0\\), \\(w_1 + b \\geq 0\\), and \\(w_2 + b \\geq 0\\). This leaves a family of solutions.',
          solution: 'Yes, it is linearly separable. We need \\(b &lt; 0\\), \\(w_1 &gt; -b\\), and \\(w_2 &gt; -b\\). One solution: \\(w_1 = w_2 = 1, b = -0.5\\), giving boundary \\(x_1 + x_2 = 0.5\\). Since we have 3 constraints and 3 unknowns \\((w_1, w_2, b)\\) with one degree of freedom from scaling invariance (\\(\\alpha\\mathbf{w}, \\alpha b\\) for \\(\\alpha &gt; 0\\) gives the same boundary), the separating hyperplane has an infinite one-parameter family of solutions (we can tilt the line anywhere between the extremes).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Perceptron Learning Algorithm
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch03-sec03',
      title: 'Perceptron Learning Algorithm',
      content: `
<h2>The Perceptron Learning Algorithm</h2>

<p>So far we have discussed how a perceptron classifies data given fixed weights. But how do we <em>find</em> the right weights? Frank Rosenblatt proposed a remarkably simple iterative procedure: if a point is misclassified, nudge the weights to fix it.</p>

<div class="env-block definition"><div class="env-title">Algorithm 3.1 (Perceptron Learning Algorithm)</div><div class="env-body">
<p><strong>Input:</strong> Training set \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^{n}\\) with \\(y_i \\in \\{0,1\\}\\), learning rate \\(\\eta &gt; 0\\).</p>
<p><strong>Initialize:</strong> \\(\\mathbf{w} \\leftarrow \\mathbf{0}\\), \\(b \\leftarrow 0\\).</p>
<p><strong>Repeat</strong> until no misclassifications (or max iterations):</p>
<ol>
  <li>For each \\((\\mathbf{x}_i, y_i)\\) in the training set:</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;Compute \\(\\hat{y}_i = \\sigma(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\)</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;If \\(\\hat{y}_i \\neq y_i\\):</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\\(\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta(y_i - \\hat{y}_i)\\mathbf{x}_i\\)</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\\(b \\leftarrow b + \\eta(y_i - \\hat{y}_i)\\)</li>
</ol>
</div></div>

<h3>Understanding the Update Rule</h3>

<p>The update rule has a simple geometric interpretation. There are two types of errors:</p>

<div class="env-block example"><div class="env-title">Case 1: False Negative (\\(y_i = 1\\), \\(\\hat{y}_i = 0\\))</div><div class="env-body">
<p>The point should be positive but was classified negative, meaning \\(\\mathbf{w}^\\top \\mathbf{x}_i + b &lt; 0\\). The update is:</p>
\\[\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\mathbf{x}_i, \\qquad b \\leftarrow b + \\eta.\\]
<p>This rotates \\(\\mathbf{w}\\) toward \\(\\mathbf{x}_i\\) and increases the bias, pushing the decision boundary so that \\(\\mathbf{x}_i\\) falls on the positive side.</p>
</div></div>

<div class="env-block example"><div class="env-title">Case 2: False Positive (\\(y_i = 0\\), \\(\\hat{y}_i = 1\\))</div><div class="env-body">
<p>The point should be negative but was classified positive, meaning \\(\\mathbf{w}^\\top \\mathbf{x}_i + b \\geq 0\\). The update is:</p>
\\[\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta\\mathbf{x}_i, \\qquad b \\leftarrow b - \\eta.\\]
<p>This rotates \\(\\mathbf{w}\\) away from \\(\\mathbf{x}_i\\) and decreases the bias, pushing the boundary so that \\(\\mathbf{x}_i\\) falls on the negative side.</p>
</div></div>

<h3>The Perceptron Convergence Theorem</h3>

<div class="env-block theorem"><div class="env-title">Theorem 3.2 (Perceptron Convergence Theorem, Novikoff 1962)</div><div class="env-body">
<p>If the training data is linearly separable with margin \\(\\gamma &gt; 0\\) (i.e., there exists a unit vector \\(\\mathbf{w}^*\\) and bias \\(b^*\\) such that \\(y_i(\\mathbf{w}^{*\\top}\\mathbf{x}_i + b^*) \\geq \\gamma\\) for all \\(i\\), where \\(y_i \\in \\{-1,+1\\}\\)), and \\(\\|\\mathbf{x}_i\\| \\leq R\\) for all \\(i\\), then the perceptron algorithm makes at most \\(R^2/\\gamma^2\\) mistakes before converging.</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof Sketch</div><div class="env-body">
<p>We track two quantities after each update. Let \\(\\tilde{\\mathbf{w}}^{(t)}\\) denote the weight vector (with bias absorbed) after \\(t\\) mistakes.</p>
<p><strong>Progress:</strong> Each mistake on a positive point adds \\(\\mathbf{x}_i\\) to \\(\\mathbf{w}\\), so \\(\\tilde{\\mathbf{w}}^{(t)} \\cdot \\tilde{\\mathbf{w}}^* \\geq t\\gamma\\) (the weights get increasingly aligned with the optimal separator).</p>
<p><strong>Bounded growth:</strong> Each update adds at most \\(R^2\\) to \\(\\|\\tilde{\\mathbf{w}}^{(t)}\\|^2\\), so \\(\\|\\tilde{\\mathbf{w}}^{(t)}\\|^2 \\leq tR^2\\).</p>
<p>Combining via Cauchy-Schwarz: \\(t\\gamma \\leq \\tilde{\\mathbf{w}}^{(t)} \\cdot \\tilde{\\mathbf{w}}^* \\leq \\|\\tilde{\\mathbf{w}}^{(t)}\\| \\leq \\sqrt{t}R\\), giving \\(t \\leq R^2/\\gamma^2\\).</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="env-block warning"><div class="env-title">Critical Limitation</div><div class="env-body"><p>If the data is <strong>not</strong> linearly separable, the perceptron algorithm will <strong>never converge</strong>. It will cycle through the data indefinitely, constantly adjusting the boundary without finding a solution. This is not a bug; it is a fundamental limitation of the perceptron model for non-separable data.</p></div></div>

<p>Watch the algorithm in action below. Click "Step" to advance one update at a time or "Run" to watch the boundary converge automatically.</p>

<div class="viz-placeholder" data-viz="viz-perceptron-learning"></div>

<div class="env-block remark"><div class="env-title">Connection to Gradient Descent</div><div class="env-body"><p>The perceptron update rule can be viewed as (sub)gradient descent on the <em>perceptron loss</em> \\(L(\\mathbf{w}, b) = -\\sum_{i \\in \\mathcal{M}} (2y_i - 1)(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\), where \\(\\mathcal{M}\\) is the set of misclassified points. This connection to optimization foreshadows the gradient-based training of modern neural networks (Chapter 5).</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-perceptron-learning',
          title: 'Perceptron Learning Algorithm',
          description: 'Watch the decision boundary adjust step by step as the perceptron corrects each misclassified point. The highlighted point (yellow ring) is the current point being processed.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 65, originX: 220, originY: 300 });

            // Linearly separable dataset
            const data = [
              { x: [0.5, 2.5], y: 1 }, { x: [1.0, 3.0], y: 1 },
              { x: [1.5, 2.0], y: 1 }, { x: [2.0, 3.5], y: 1 },
              { x: [2.5, 2.8], y: 1 }, { x: [1.2, 3.8], y: 1 },
              { x: [3.0, 0.5], y: 0 }, { x: [3.5, 1.0], y: 0 },
              { x: [4.0, 0.8], y: 0 }, { x: [4.5, 1.5], y: 0 },
              { x: [3.8, 0.2], y: 0 }, { x: [5.0, 1.0], y: 0 },
            ];

            let w1 = 0, w2 = 0, bias = 0;
            const eta = 1.0;
            let step = 0;
            let currentIdx = 0;
            let epoch = 0;
            let converged = false;
            let animRunning = false;
            let history = []; // track weight history

            function classify(x) { return (w1 * x[0] + w2 * x[1] + bias) >= 0 ? 1 : 0; }

            function countErrors() {
              return data.filter(p => classify(p.x) !== p.y).length;
            }

            function doStep() {
              if (converged) return;

              // Find next misclassified point starting from currentIdx
              let checked = 0;
              while (checked < data.length) {
                const idx = currentIdx % data.length;
                const p = data[idx];
                const pred = classify(p.x);
                currentIdx = (currentIdx + 1) % data.length;
                if (idx === 0 && checked > 0) epoch++;

                if (pred !== p.y) {
                  const error = p.y - pred; // +1 or -1
                  w1 += eta * error * p.x[0];
                  w2 += eta * error * p.x[1];
                  bias += eta * error;
                  step++;
                  history.push({ w1, w2, bias, idx, error });
                  break;
                }
                checked++;
              }

              if (checked >= data.length) {
                converged = true;
              }
            }

            function reset() {
              w1 = 0; w2 = 0; bias = 0;
              step = 0; currentIdx = 0; epoch = 0;
              converged = false; animRunning = false;
              history = [];
              viz.stopAnimation();
              draw();
            }

            const btnRow = document.createElement('div');
            btnRow.style.cssText = 'display:flex;gap:6px;';

            VizEngine.createButton(btnRow, 'Step', () => {
              if (!converged) { doStep(); draw(); }
            });

            let runBtn;
            runBtn = VizEngine.createButton(btnRow, 'Run', () => {
              if (converged) return;
              if (animRunning) {
                animRunning = false;
                viz.stopAnimation();
                runBtn.textContent = 'Run';
                return;
              }
              animRunning = true;
              runBtn.textContent = 'Pause';
              let lastStep = 0;
              viz.animate((t) => {
                if (!animRunning || converged) {
                  animRunning = false;
                  viz.stopAnimation();
                  runBtn.textContent = 'Run';
                  draw();
                  return;
                }
                if (t - lastStep > 400) {
                  doStep();
                  draw();
                  lastStep = t;
                }
              });
            });

            VizEngine.createButton(btnRow, 'Reset', reset);
            controls.appendChild(btnRow);

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();

              const ctx = viz.ctx;
              const norm = Math.sqrt(w1 * w1 + w2 * w2);

              // Decision boundary
              if (norm > 0.01) {
                let lx1, ly1, lx2, ly2;
                if (Math.abs(w2) > Math.abs(w1)) {
                  lx1 = -2; ly1 = -(w1 * lx1 + bias) / w2;
                  lx2 = 8; ly2 = -(w1 * lx2 + bias) / w2;
                } else {
                  ly1 = -2; lx1 = -(w2 * ly1 + bias) / w1;
                  ly2 = 8; lx2 = -(w2 * ly2 + bias) / w1;
                }

                // Light shading
                const [sx1, sy1] = viz.toScreen(lx1, ly1);
                const [sx2, sy2] = viz.toScreen(lx2, ly2);
                const nx = w1 / norm, ny = w2 / norm;
                const push = 600;
                const snx = nx * push, sny = -ny * push;

                ctx.fillStyle = 'rgba(88, 166, 255, 0.05)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 + snx, sy2 + sny); ctx.lineTo(sx1 + snx, sy1 + sny);
                ctx.closePath(); ctx.fill();

                ctx.fillStyle = 'rgba(248, 81, 73, 0.05)';
                ctx.beginPath();
                ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
                ctx.lineTo(sx2 - snx, sy2 - sny); ctx.lineTo(sx1 - snx, sy1 - sny);
                ctx.closePath(); ctx.fill();

                viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green, 2.5);

                // Weight vector
                const arrowScale = 0.8 / norm;
                viz.drawVector(0, 0, w1 * arrowScale, w2 * arrowScale, viz.colors.white, 'w', 2);
              }

              // Data points
              data.forEach((p, i) => {
                const pred = classify(p.x);
                const color = p.y === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(p.x[0], p.x[1]);
                ctx.beginPath();
                ctx.arc(sx, sy, 7, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Misclassification marker
                if (pred !== p.y) {
                  ctx.strokeStyle = viz.colors.yellow;
                  ctx.lineWidth = 2;
                  ctx.beginPath();
                  ctx.arc(sx, sy, 11, 0, Math.PI * 2);
                  ctx.stroke();
                }
              });

              // Highlight last corrected point
              if (history.length > 0) {
                const last = history[history.length - 1];
                const pt = data[last.idx];
                const [sx, sy] = viz.toScreen(pt.x[0], pt.x[1]);
                ctx.strokeStyle = viz.colors.yellow;
                ctx.lineWidth = 3;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.arc(sx, sy, 14, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
              }

              // Info
              const errors = countErrors();
              viz.screenText('Step: ' + step + '  |  Errors: ' + errors + '/' + data.length, viz.width / 2, 18, viz.colors.text, 12);
              viz.screenText('w = (' + w1.toFixed(2) + ', ' + w2.toFixed(2) + ')   b = ' + bias.toFixed(2), viz.width / 2, 36, viz.colors.text, 12);

              if (converged) {
                viz.screenText('Converged! No more misclassified points.', viz.width / 2, viz.height - 15, viz.colors.green, 13);
              } else if (step === 0) {
                viz.screenText('Click "Step" or "Run" to begin learning.', viz.width / 2, viz.height - 15, viz.colors.text, 12);
              }

              // Axis labels
              viz.screenText('x\u2081', viz.width - 15, viz.originY - 10, viz.colors.text, 13, 'right');
              viz.screenText('x\u2082', viz.originX + 10, 50, viz.colors.text, 13, 'left');
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Starting from \\(\\mathbf{w} = (0,0)^\\top\\), \\(b = 0\\), \\(\\eta = 1\\), apply one step of the perceptron algorithm to the misclassified point \\((2, 1)\\) with label \\(y = 1\\). What are the new weights and bias?',
          hint: 'The prediction is \\(\\sigma(0 \\cdot 2 + 0 \\cdot 1 + 0) = \\sigma(0) = 1\\). Wait, is this actually misclassified? Re-check the step function definition.',
          solution: 'With \\(\\mathbf{w} = (0,0)^\\top\\) and \\(b = 0\\), we compute \\(z = 0\\). By our convention \\(\\sigma(0) = 1\\), so \\(\\hat{y} = 1 = y\\), meaning this point is correctly classified and no update occurs. If instead we had the point \\((2,1)\\) with \\(b = -1\\), then \\(z = -1 &lt; 0\\), \\(\\hat{y} = 0 \\neq 1\\), and the update would be: \\(\\mathbf{w} \\leftarrow (0,0) + 1 \\cdot (2,1) = (2,1)\\), \\(b \\leftarrow -1 + 1 = 0\\).'
        },
        {
          question: 'The convergence theorem guarantees at most \\(R^2/\\gamma^2\\) mistakes. For a dataset where all points lie within a circle of radius \\(R = 5\\) and the margin is \\(\\gamma = 0.5\\), what is the maximum number of mistakes?',
          hint: 'Direct substitution into the bound.',
          solution: 'The maximum number of mistakes is \\(R^2/\\gamma^2 = 25/0.25 = 100\\). Note that this is a <em>worst-case</em> bound; in practice, the algorithm typically converges much faster.'
        },
        {
          question: 'Explain why the perceptron convergence theorem does <strong>not</strong> apply to XOR data. What happens if you run the perceptron algorithm on XOR?',
          hint: 'What is the assumption of the theorem?',
          solution: 'The theorem requires the data to be linearly separable (\\(\\gamma &gt; 0\\)). XOR data is not linearly separable (Theorem 3.1), so no margin \\(\\gamma &gt; 0\\) exists. If you run the perceptron algorithm on XOR data, it will cycle indefinitely: each correction of one point will cause another point to become misclassified. The weights oscillate without settling.'
        },
        {
          question: 'Show that the perceptron update rule \\(\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta(y - \\hat{y})\\mathbf{x}\\) is equivalent to gradient descent on the loss \\(L = \\max(0, -(2y-1)(\\mathbf{w}^\\top\\mathbf{x} + b))\\) for a single misclassified point.',
          hint: 'Compute \\(-\\partial L / \\partial \\mathbf{w}\\) for a misclassified point where \\((2y-1)(\\mathbf{w}^\\top\\mathbf{x}+b) &lt; 0\\).',
          solution: 'Let \\(s = 2y - 1 \\in \\{-1,+1\\}\\). For a misclassified point, \\(s(\\mathbf{w}^\\top\\mathbf{x}+b) &lt; 0\\), so \\(L = -s(\\mathbf{w}^\\top\\mathbf{x}+b)\\). The gradient is \\(\\partial L/\\partial \\mathbf{w} = -s\\mathbf{x}\\), so the gradient descent update is \\(\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta s \\mathbf{x}\\). Now \\(s = 2y - 1\\), and for a misclassified point, \\(y - \\hat{y} = y - (1-y) = 2y - 1 = s\\) (if \\(y=1,\\hat{y}=0\\)) or \\(y - \\hat{y} = 0 - 1 = -1 = s\\) (if \\(y=0,\\hat{y}=1\\)). In both cases, \\(y - \\hat{y} = s\\), confirming the equivalence.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: From Perceptron to Neural Networks
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch03-sec04',
      title: 'From Perceptron to Neural Networks',
      content: `
<h2>From Perceptron to Neural Networks</h2>

<p>The XOR impossibility result sent shockwaves through the AI community in the late 1960s and contributed to the first "AI winter," a period of reduced funding and interest. But the solution was already implicit in Minsky and Papert's analysis: while a <em>single</em> perceptron cannot solve XOR, <em>multiple</em> perceptrons working together can.</p>

<div class="env-block intuition"><div class="env-title">The Key Insight</div><div class="env-body"><p>A single perceptron draws one line. Two perceptrons draw two lines. If we feed the outputs of these two perceptrons into a third perceptron, the third perceptron can implement AND, OR, or any linearly separable function on the two regions. This composition creates non-convex decision boundaries and can solve XOR.</p></div></div>

<h3>Solving XOR with Two Layers</h3>

<div class="env-block theorem"><div class="env-title">Proposition 3.1 (XOR via Composition)</div><div class="env-body">
<p>XOR can be decomposed as:</p>
\\[x_1 \\oplus x_2 = (x_1 \\lor x_2) \\land \\lnot(x_1 \\land x_2).\\]
<p>Each of these sub-functions (OR, NAND) is linearly separable and can be computed by a single perceptron. Feeding their outputs into an AND perceptron yields XOR.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 3.4 (Explicit XOR Network)</div><div class="env-body">
<p>Define two hidden neurons and one output neuron:</p>
<ul>
  <li><strong>Hidden neuron 1</strong> (OR): \\(h_1 = \\sigma(x_1 + x_2 - 0.5)\\)</li>
  <li><strong>Hidden neuron 2</strong> (NAND): \\(h_2 = \\sigma(-x_1 - x_2 + 1.5)\\)</li>
  <li><strong>Output neuron</strong> (AND): \\(y = \\sigma(h_1 + h_2 - 1.5)\\)</li>
</ul>
<p>Verification:</p>
<ul>
  <li>\\((0,0)\\): \\(h_1 = \\sigma(-0.5)=0\\), \\(h_2=\\sigma(1.5)=1\\), \\(y=\\sigma(0+1-1.5)=\\sigma(-0.5)=0\\) \\(\\checkmark\\)</li>
  <li>\\((0,1)\\): \\(h_1 = \\sigma(0.5)=1\\), \\(h_2=\\sigma(0.5)=1\\), \\(y=\\sigma(1+1-1.5)=\\sigma(0.5)=1\\) \\(\\checkmark\\)</li>
  <li>\\((1,0)\\): \\(h_1 = \\sigma(0.5)=1\\), \\(h_2=\\sigma(0.5)=1\\), \\(y=\\sigma(1+1-1.5)=\\sigma(0.5)=1\\) \\(\\checkmark\\)</li>
  <li>\\((1,1)\\): \\(h_1 = \\sigma(1.5)=1\\), \\(h_2=\\sigma(-0.5)=0\\), \\(y=\\sigma(1+0-1.5)=\\sigma(-0.5)=0\\) \\(\\checkmark\\)</li>
</ul>
</div></div>

<h3>Geometric View: Combining Decision Boundaries</h3>

<p>The two hidden perceptrons create two lines in the input space. Hidden neuron 1 (OR) places all inputs except \\((0,0)\\) on the positive side. Hidden neuron 2 (NAND) places all inputs except \\((1,1)\\) on the positive side. The AND output neuron requires <em>both</em> conditions to be true, which selects exactly the "strip" between the two lines where \\((0,1)\\) and \\((1,0)\\) reside.</p>

<div class="viz-placeholder" data-viz="viz-xor-network"></div>

<h3>The Multilayer Perceptron</h3>

<div class="env-block definition"><div class="env-title">Definition 3.3 (Multilayer Perceptron)</div><div class="env-body">
<p>A <strong>multilayer perceptron</strong> (MLP) is a feedforward neural network consisting of:</p>
<ol>
  <li>An <strong>input layer</strong> that receives \\(\\mathbf{x} \\in \\mathbb{R}^d\\)</li>
  <li>One or more <strong>hidden layers</strong>, each computing \\(\\mathbf{h}^{(l)} = \\sigma(\\mathbf{W}^{(l)}\\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)})\\)</li>
  <li>An <strong>output layer</strong> producing the final prediction</li>
</ol>
<p>The key difference from a single perceptron: the nonlinear activations in hidden layers enable the network to learn <strong>non-linear</strong> decision boundaries.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 3.3 (Universal Approximation, Cybenko 1989; Hornik 1991)</div><div class="env-body">
<p>A feedforward neural network with a single hidden layer containing a finite but sufficiently large number of neurons, and a non-polynomial activation function, can approximate any continuous function on a compact subset of \\(\\mathbb{R}^d\\) to arbitrary accuracy.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>The universal approximation theorem is an <strong>existence</strong> result. It guarantees that a sufficiently wide single-hidden-layer network <em>can</em> represent any function, but it says nothing about whether gradient-based training will <em>find</em> such a representation, nor does it say anything about the width required. In practice, deep (multi-layer) networks often need far fewer parameters than wide shallow networks. This depth-efficiency advantage is the primary motivation for <em>deep</em> learning, which we explore starting in Chapter 4.</p></div></div>

<div class="env-block intuition"><div class="env-title">The Road Ahead</div><div class="env-body"><p>The single perceptron taught us the fundamentals: weighted sums, activation functions, and decision boundaries. Its inability to solve XOR motivated multi-layer architectures. But stacking perceptrons introduces a new challenge: how do you train the hidden layers when you only have error information at the output? The answer, <strong>backpropagation</strong>, was the breakthrough that unlocked deep learning. In the next chapter we will build multi-layer networks, and in Chapter 5 we will derive the backpropagation algorithm that makes training them possible.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-xor-network',
          title: 'Solving XOR with Two Perceptrons',
          description: 'Two hidden perceptrons (OR and NAND) create two decision boundaries. Their intersection (the AND region) correctly classifies XOR. Adjust the hidden neurons\' parameters to see how the combined region changes.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 130, originX: 100, originY: 340 });

            // Hidden neuron 1 (OR): h1 = step(w11*x1 + w12*x2 + b1)
            let w11 = 1, w12 = 1, b1 = -0.5;
            // Hidden neuron 2 (NAND): h2 = step(w21*x1 + w22*x2 + b2)
            let w21 = -1, w22 = -1, b2 = 1.5;

            const corners = [[0, 0], [0, 1], [1, 0], [1, 1]];
            const xorTargets = [0, 1, 1, 0];

            const sliderBox = document.createElement('div');
            sliderBox.style.cssText = 'display:flex;flex-direction:column;gap:2px;';

            const lbl1 = document.createElement('span');
            lbl1.style.cssText = 'font-size:0.75rem;color:#58a6ff;font-weight:600;';
            lbl1.textContent = 'Hidden 1 (blue line):';
            sliderBox.appendChild(lbl1);
            VizEngine.createSlider(sliderBox, 'w\u2081\u2081', -3, 3, w11, 0.1, v => { w11 = v; draw(); });
            VizEngine.createSlider(sliderBox, 'w\u2081\u2082', -3, 3, w12, 0.1, v => { w12 = v; draw(); });
            VizEngine.createSlider(sliderBox, 'b\u2081', -3, 3, b1, 0.1, v => { b1 = v; draw(); });

            const lbl2 = document.createElement('span');
            lbl2.style.cssText = 'font-size:0.75rem;color:#f0883e;font-weight:600;margin-top:4px;';
            lbl2.textContent = 'Hidden 2 (orange line):';
            sliderBox.appendChild(lbl2);
            VizEngine.createSlider(sliderBox, 'w\u2082\u2081', -3, 3, w21, 0.1, v => { w21 = v; draw(); });
            VizEngine.createSlider(sliderBox, 'w\u2082\u2082', -3, 3, w22, 0.1, v => { w22 = v; draw(); });
            VizEngine.createSlider(sliderBox, 'b\u2082', -3, 3, b2, 0.1, v => { b2 = v; draw(); });

            controls.appendChild(sliderBox);

            const resetBtn = VizEngine.createButton(controls, 'Reset XOR solution', () => {
              w11 = 1; w12 = 1; b1 = -0.5;
              w21 = -1; w22 = -1; b2 = 1.5;
              // Update slider values visually
              const sliders = sliderBox.querySelectorAll('input[type=range]');
              const vals = [w11, w12, b1, w21, w22, b2];
              sliders.forEach((s, i) => {
                s.value = vals[i];
                const vSpan = s.nextElementSibling;
                if (vSpan) vSpan.textContent = vals[i].toFixed(1);
              });
              draw();
            });

            function step(z) { return z >= 0 ? 1 : 0; }

            function computeXOR(x1, x2) {
              const h1 = step(w11 * x1 + w12 * x2 + b1);
              const h2 = step(w21 * x1 + w22 * x2 + b2);
              // AND output: step(h1 + h2 - 1.5)
              return step(h1 + h2 - 1.5);
            }

            function draw() {
              viz.clear();
              const ctx = viz.ctx;

              // Pixel-by-pixel classification for the combined region
              const resolution = 3;
              for (let px = 0; px < viz.width; px += resolution) {
                for (let py = 0; py < viz.height; py += resolution) {
                  const [mx, my] = viz.toMath(px, py);
                  const h1 = step(w11 * mx + w12 * my + b1);
                  const h2 = step(w21 * mx + w22 * my + b2);
                  const out = step(h1 + h2 - 1.5);
                  if (out === 1) {
                    ctx.fillStyle = 'rgba(63, 185, 160, 0.12)';
                    ctx.fillRect(px, py, resolution, resolution);
                  }
                }
              }

              // Grid and axes
              ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
              for (let i = 0; i <= 1; i++) {
                const [sx] = viz.toScreen(i, 0);
                ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, viz.height); ctx.stroke();
                const [, sy] = viz.toScreen(0, i);
                ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(viz.width, sy); ctx.stroke();
              }

              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
              ctx.beginPath(); ctx.moveTo(viz.toScreen(-0.3, 0)[0], viz.toScreen(0, 0)[1]); ctx.lineTo(viz.toScreen(1.5, 0)[0], viz.toScreen(0, 0)[1]); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(viz.toScreen(0, -0.3)[0], viz.toScreen(0, -0.3)[1]); ctx.lineTo(viz.toScreen(0, 1.5)[0], viz.toScreen(0, 1.5)[1]); ctx.stroke();

              // Tick labels
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText('0', viz.toScreen(0, 0)[0], viz.toScreen(0, 0)[1] + 5);
              ctx.fillText('1', viz.toScreen(1, 0)[0], viz.toScreen(1, 0)[1] + 5);
              ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText('1', viz.toScreen(0, 1)[0] - 8, viz.toScreen(0, 1)[1]);

              // Decision boundaries
              const norm1 = Math.sqrt(w11 * w11 + w12 * w12);
              if (norm1 > 0.01) {
                let lx1, ly1, lx2, ly2;
                if (Math.abs(w12) > Math.abs(w11)) {
                  lx1 = -0.5; ly1 = -(w11 * lx1 + b1) / w12;
                  lx2 = 1.5; ly2 = -(w11 * lx2 + b1) / w12;
                } else {
                  ly1 = -0.5; lx1 = -(w12 * ly1 + b1) / w11;
                  ly2 = 1.5; lx2 = -(w12 * ly2 + b1) / w11;
                }
                viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.blue, 2, false);
                // Label
                const [slx, sly] = viz.toScreen(lx2, ly2);
                ctx.fillStyle = viz.colors.blue; ctx.font = 'bold 11px -apple-system,sans-serif';
                ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
                ctx.fillText('h\u2081', slx + 4, sly - 4);
              }

              const norm2 = Math.sqrt(w21 * w21 + w22 * w22);
              if (norm2 > 0.01) {
                let lx1, ly1, lx2, ly2;
                if (Math.abs(w22) > Math.abs(w21)) {
                  lx1 = -0.5; ly1 = -(w21 * lx1 + b2) / w22;
                  lx2 = 1.5; ly2 = -(w21 * lx2 + b2) / w22;
                } else {
                  ly1 = -0.5; lx1 = -(w22 * ly1 + b2) / w21;
                  ly2 = 1.5; lx2 = -(w22 * ly2 + b2) / w21;
                }
                viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.orange, 2, false);
                const [slx, sly] = viz.toScreen(lx2, ly2);
                ctx.fillStyle = viz.colors.orange; ctx.font = 'bold 11px -apple-system,sans-serif';
                ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
                ctx.fillText('h\u2082', slx + 4, sly - 4);
              }

              // Data points
              let correct = 0;
              corners.forEach(([x, y], i) => {
                const target = xorTargets[i];
                const pred = computeXOR(x, y);
                if (pred === target) correct++;

                const color = target === 1 ? viz.colors.teal : viz.colors.red;
                const [sx, sy] = viz.toScreen(x, y);
                ctx.beginPath();
                ctx.arc(sx, sy, 11, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Border indicates correct/incorrect
                ctx.strokeStyle = pred === target ? viz.colors.green : viz.colors.yellow;
                ctx.lineWidth = pred === target ? 1.5 : 3;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(target.toString(), sx, sy);
              });

              // Title
              viz.screenText('XOR via Two Perceptrons', viz.width / 2, 18, viz.colors.white, 16);

              // Status
              const allCorrect = correct === 4;
              viz.screenText(
                allCorrect ? 'XOR solved! The teal region contains exactly the class-1 points.' : 'Correct: ' + correct + '/4. Adjust sliders to solve XOR.',
                viz.width / 2, viz.height - 15,
                allCorrect ? viz.colors.green : viz.colors.yellow, 12
              );

              // Network diagram (right side)
              const nx = viz.width - 100;
              const ny = 80;
              // Input nodes
              ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
              ctx.textAlign = 'center';

              // x1
              ctx.beginPath(); ctx.arc(nx - 60, ny, 12, 0, Math.PI * 2);
              ctx.fillStyle = viz.colors.bg; ctx.fill();
              ctx.strokeStyle = viz.colors.text; ctx.lineWidth = 1; ctx.stroke();
              ctx.fillStyle = viz.colors.text; ctx.fillText('x\u2081', nx - 60, ny + 1);

              // x2
              ctx.beginPath(); ctx.arc(nx - 60, ny + 50, 12, 0, Math.PI * 2);
              ctx.fillStyle = viz.colors.bg; ctx.fill();
              ctx.strokeStyle = viz.colors.text; ctx.lineWidth = 1; ctx.stroke();
              ctx.fillStyle = viz.colors.text; ctx.fillText('x\u2082', nx - 60, ny + 51);

              // h1
              ctx.beginPath(); ctx.arc(nx, ny - 10, 12, 0, Math.PI * 2);
              ctx.fillStyle = viz.colors.bg; ctx.fill();
              ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 1.5; ctx.stroke();
              ctx.fillStyle = viz.colors.blue; ctx.fillText('h\u2081', nx, ny - 9);

              // h2
              ctx.beginPath(); ctx.arc(nx, ny + 60, 12, 0, Math.PI * 2);
              ctx.fillStyle = viz.colors.bg; ctx.fill();
              ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.5; ctx.stroke();
              ctx.fillStyle = viz.colors.orange; ctx.fillText('h\u2082', nx, ny + 61);

              // output
              ctx.beginPath(); ctx.arc(nx + 60, ny + 25, 12, 0, Math.PI * 2);
              ctx.fillStyle = viz.colors.bg; ctx.fill();
              ctx.strokeStyle = viz.colors.teal; ctx.lineWidth = 1.5; ctx.stroke();
              ctx.fillStyle = viz.colors.teal; ctx.fillText('y', nx + 60, ny + 26);

              // Connections
              ctx.strokeStyle = viz.colors.text + '66'; ctx.lineWidth = 1;
              // x1 -> h1, h2
              ctx.beginPath(); ctx.moveTo(nx - 48, ny); ctx.lineTo(nx - 12, ny - 10); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(nx - 48, ny); ctx.lineTo(nx - 12, ny + 60); ctx.stroke();
              // x2 -> h1, h2
              ctx.beginPath(); ctx.moveTo(nx - 48, ny + 50); ctx.lineTo(nx - 12, ny - 10); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(nx - 48, ny + 50); ctx.lineTo(nx - 12, ny + 60); ctx.stroke();
              // h1, h2 -> output
              ctx.beginPath(); ctx.moveTo(nx + 12, ny - 10); ctx.lineTo(nx + 48, ny + 25); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(nx + 12, ny + 60); ctx.lineTo(nx + 48, ny + 25); ctx.stroke();

              // Axis labels
              viz.screenText('x\u2081', viz.toScreen(1.45, 0)[0], viz.toScreen(1.45, 0)[1] - 15, viz.colors.text, 13);
              viz.screenText('x\u2082', viz.toScreen(0, 1.45)[0] + 15, viz.toScreen(0, 1.45)[1], viz.colors.text, 13);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Verify the XOR network from Example 3.4 by computing the output for all four input combinations. Show each intermediate value \\(h_1, h_2\\).',
          hint: 'Apply the step function at each neuron. Remember \\(\\sigma(z) = 1\\) if \\(z \\geq 0\\), and 0 otherwise.',
          solution: '<strong>(0,0):</strong> \\(h_1 = \\sigma(0+0-0.5) = \\sigma(-0.5) = 0\\), \\(h_2 = \\sigma(0+0+1.5) = \\sigma(1.5) = 1\\), \\(y = \\sigma(0+1-1.5) = \\sigma(-0.5) = 0\\). <strong>(0,1):</strong> \\(h_1 = \\sigma(0+1-0.5) = 1\\), \\(h_2 = \\sigma(0-1+1.5) = 1\\), \\(y = \\sigma(1+1-1.5) = 1\\). <strong>(1,0):</strong> \\(h_1 = \\sigma(1+0-0.5) = 1\\), \\(h_2 = \\sigma(-1+0+1.5) = 1\\), \\(y = \\sigma(1+1-1.5) = 1\\). <strong>(1,1):</strong> \\(h_1 = \\sigma(1+1-0.5) = 1\\), \\(h_2 = \\sigma(-1-1+1.5) = 0\\), \\(y = \\sigma(1+0-1.5) = 0\\). All outputs match XOR.'
        },
        {
          question: 'Find an alternative XOR network using the decomposition \\(x_1 \\oplus x_2 = (x_1 \\lor x_2) \\land \\lnot(x_1 \\land x_2)\\) but with different weight values than Example 3.4.',
          hint: 'Any weights that implement OR, NAND, and AND will work. Try scaling the weights.',
          solution: 'Use \\(h_1 = \\sigma(2x_1 + 2x_2 - 1)\\) for OR and \\(h_2 = \\sigma(-2x_1 - 2x_2 + 3)\\) for NAND. Then \\(y = \\sigma(h_1 + h_2 - 1.5)\\) for AND. Verification: (0,0): \\(h_1=0, h_2=1, y=0\\); (0,1): \\(h_1=1, h_2=1, y=1\\); (1,0): \\(h_1=1, h_2=1, y=1\\); (1,1): \\(h_1=1, h_2=0, y=0\\). Many other solutions exist; the key is that any weights implementing the correct Boolean sub-functions work.'
        },
        {
          question: 'How many hidden neurons are needed to implement an arbitrary Boolean function on \\(d\\) binary inputs? Give an upper bound and justify it.',
          hint: 'Think about the number of input patterns that map to 1, and how each hidden neuron can isolate one pattern.',
          solution: 'Any Boolean function can be expressed as a disjunction (OR) of conjunctions (AND) of literals, by the disjunctive normal form (DNF). Each AND-clause can be computed by one hidden perceptron, and the OR of all clauses is a single output perceptron. Since there are at most \\(2^d\\) input patterns, we need at most \\(2^d\\) hidden neurons (one per positive-output pattern). In practice, many functions need far fewer, but this exponential upper bound shows that width can substitute for depth. Achieving this with polynomial hidden neurons for specific function classes is a central question in computational complexity.'
        },
        {
          question: 'The universal approximation theorem says a single hidden layer suffices to approximate any continuous function. Why, then, do we use <em>deep</em> networks with many hidden layers?',
          hint: 'Think about efficiency (number of parameters needed) and practical trainability.',
          solution: 'Three reasons: (1) <strong>Efficiency.</strong> Deep networks can represent certain functions with exponentially fewer parameters than shallow ones. For example, a depth-\\(k\\) network with polynomial width can represent functions requiring exponential width with depth 2. (2) <strong>Hierarchical features.</strong> Deep networks naturally learn hierarchical representations (edges, then textures, then object parts, then objects), which mirrors the compositional structure of real-world data. (3) <strong>Trainability.</strong> With modern techniques (batch normalization, residual connections, Adam optimizer), deep networks are practically trainable, while extremely wide shallow networks face optimization difficulties. The universal approximation theorem is an existence result; it guarantees representability but not that gradient descent will find the right representation.'
        }
      ]
    }
  ]
});
